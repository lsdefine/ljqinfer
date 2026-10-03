// A100-compatible V2-style exact selection: coarse histogram then boundary refinement.
// Original implementation; not a copy of SGLang's SM90/SM100 cluster kernels.
#include <torch/extension.h>
#include <algorithm>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cub/block/block_scan.cuh>
#include <cub/block/block_reduce.cuh>
#include <cuda_fp16.h>
constexpr int TILE=4096, THREADS=256, BINS=1024;
__device__ __forceinline__ unsigned ordered(float v) {
 unsigned u=__float_as_uint(v); return (u&0x80000000u)?~u:(u|0x80000000u);
}
__device__ __forceinline__ int coarse(float v) {
 unsigned u=__half_as_ushort(__float2half_rn(v));
 return ((u&0x8000u)?((~u)&0xffffu):(u|0x8000u))>>6;
}
__device__ __forceinline__ void add_hist(int* h,int b,bool valid) {
 unsigned mask=__ballot_sync(0xffffffffu,valid);
 if(valid){unsigned group=__match_any_sync(mask,b);if((threadIdx.x&31)==(__ffs(group)-1))atomicAdd(h+b,__popc(group));}
}
__global__ void coarse_hist(const float* s,const int64_t* pos,int* hist,int N,int K,int C){
 int r=blockIdx.y,c=blockIdx.x,t=threadIdx.x;
 int lim=(int)max((int64_t)0,min(pos[r]+1,(int64_t)N));
 if(lim<=K)return;
 int live=(lim+TILE-1)/TILE;
 for(c=blockIdx.x;c<live;c+=gridDim.x){
 __shared__ int h[BINS];for(int b=t;b<BINS;b+=THREADS)h[b]=0;__syncthreads();
 if(lim>K)for(int i=c*TILE+t;i<(c+1)*TILE;i+=THREADS){bool valid=i<lim;int b=valid?coarse(s[(int64_t)r*N+i]):0;if(valid)atomicAdd(h+b,1);}
 __syncthreads();for(int b=t;b<BINS;b+=THREADS)hist[((int64_t)r*C+c)*BINS+b]=h[b];
 __syncthreads();
 }

}
// state = coarse threshold, greater count, boundary count, remaining K.
// offsets[chunk] = exclusive greater prefix, exclusive boundary prefix.
__global__ void plan(const int* hist,const int64_t* pos,int* st,int* offsets,int N,int K,int C){
 int r=blockIdx.x,t=threadIdx.x;
 int lim=(int)max((int64_t)0,min(pos[r]+1,(int64_t)N));
 if(lim<=K){if(t<4)st[r*4+t]=0;return;}
 int live=(lim+TILE-1)/TILE;
 using Scan=cub::BlockScan<int,THREADS>;__shared__ Scan::TempStorage temp;
 __shared__ int threshold,gcarry,ecarry,total[BINS]; int a[4],sum[4];
 for(int j=0;j<4;j++){int b=t+j*THREADS,v=0;for(int c=0;c<live;c++)v+=hist[((int64_t)r*C+c)*BINS+b];total[b]=v;}
 __syncthreads();for(int j=0;j<4;j++)a[j]=total[BINS-1-(t*4+j)];
 Scan(temp).InclusiveSum(a,sum);__syncthreads();
 for(int j=0;j<4;j++)if(sum[j]>=K&&sum[j]-a[j]<K)threshold=BINS-1-(t*4+j);
 __syncthreads();int b=threshold;
 // Warp-cooperative per-chunk counts. offsets temporarily holds counts.
 for(int c=t/32;c<live;c+=THREADS/32){
  int g=0;for(int j=b+1+(t&31);j<BINS;j+=32)g+=hist[((int64_t)r*C+c)*BINS+j];
  g=__reduce_add_sync(0xffffffffu,g);
  if((t&31)==0){offsets[((int64_t)r*C+c)*2]=g;offsets[((int64_t)r*C+c)*2+1]=hist[((int64_t)r*C+c)*BINS+b];}
 }
 __syncthreads();if(t==0){gcarry=0;ecarry=0;}__syncthreads();
 for(int base=0;base<live;base+=THREADS){
  int c=base+t,g=c<live?offsets[((int64_t)r*C+c)*2]:0,e=c<live?offsets[((int64_t)r*C+c)*2+1]:0,pg,pe,gt,et;
  Scan(temp).ExclusiveSum(g,pg,gt);__syncthreads();Scan(temp).ExclusiveSum(e,pe,et);__syncthreads();
  if(c<live){offsets[((int64_t)r*C+c)*2]=gcarry+pg;offsets[((int64_t)r*C+c)*2+1]=ecarry+pe;}
  __syncthreads();if(t==0){gcarry+=gt;ecarry+=et;}__syncthreads();
 }
 if(t==0){st[r*4]=b;st[r*4+1]=gcarry;st[r*4+2]=ecarry;st[r*4+3]=K-gcarry;}
}
__global__ void collect(const float* s,const int64_t* pos,const int* st,const int* offsets,int* ids,int* keys,int* out,int N,int K,int C){
 int r=blockIdx.y,c=blockIdx.x,t=threadIdx.x,lim=(int)max((int64_t)0,min(pos[r]+1,(int64_t)N));
 if(lim<=K){for(int i=blockIdx.x*THREADS+t;i<K;i+=gridDim.x*THREADS)out[(int64_t)r*K+i]=i<lim?i:-1;return;}
 int live=(lim+TILE-1)/TILE;
 for(c=blockIdx.x;c<live;c+=gridDim.x){
 int threshold=st[r*4],bg=offsets[((int64_t)r*C+c)*2],be=offsets[((int64_t)r*C+c)*2+1];
 constexpr int ITEMS=TILE/THREADS,WARPS=TILE/32;
 __shared__ int gs[WARPS],es[WARPS];
 unsigned gm[ITEMS],em[ITEMS],keyreg[ITEMS];int lane=t&31,warp=t>>5;
 #pragma unroll
 for(int j=0;j<ITEMS;j++){
  int i=c*TILE+j*THREADS+t;bool valid=i<lim;float v=valid?s[(int64_t)r*N+i]:0;
  int b=valid?coarse(v):-1;keyreg[j]=ordered(v);
  gm[j]=__ballot_sync(0xffffffffu,valid&&b>threshold);em[j]=__ballot_sync(0xffffffffu,valid&&b==threshold);
  if(lane==0){gs[j*8+warp]=__popc(gm[j]);es[j*8+warp]=__popc(em[j]);}
 }
 __syncthreads();
 using Scan=cub::BlockScan<int,THREADS>;__shared__ Scan::TempStorage tmp;
 int g=t<WARPS?gs[t]:0,e=t<WARPS?es[t]:0,pg,pe;
 Scan(tmp).ExclusiveSum(g,pg);__syncthreads();Scan(tmp).ExclusiveSum(e,pe);__syncthreads();
 if(t<WARPS){gs[t]=pg;es[t]=pe;}__syncthreads();unsigned mask=lane==0?0u:((1u<<lane)-1u);
 #pragma unroll
 for(int j=0;j<ITEMS;j++){
  int i=c*TILE+j*THREADS+t;
  if(gm[j]&(1u<<lane))out[(int64_t)r*K+bg+gs[j*8+warp]+__popc(gm[j]&mask)]=i;
  if(em[j]&(1u<<lane)){int dst=be+es[j*8+warp]+__popc(em[j]&mask);ids[(int64_t)r*N+dst]=i;keys[(int64_t)r*N+dst]=keyreg[j];}
 }

 __syncthreads();
 }
}
__global__ void refine(const int64_t* pos,const int* st,const int* ids,const int* keys,int* out,int N,int K){
 int r=blockIdx.x,t=threadIdx.x,lim=(int)max((int64_t)0,min(pos[r]+1,(int64_t)N));if(lim<=K)return;
 int M=st[r*4+2],G=st[r*4+1],wanted=st[r*4+3];
 const unsigned* kk=(const unsigned*)keys+(int64_t)r*N;
 using Reduce=cub::BlockReduce<unsigned,THREADS>;using Scan=cub::BlockScan<int,THREADS>;
 __shared__ union {typename Reduce::TempStorage red;typename Scan::TempStorage scan;} temp;
 __shared__ unsigned lo,hi,prefix;__shared__ int remain,h[256],greater,tiecount;
 unsigned mn=0xffffffffu,mx=0;
 for(int i=t;i<M;i+=THREADS){unsigned u=kk[i];mn=min(mn,u);mx=max(mx,u);}
 unsigned v=Reduce(temp.red).Reduce(mn,cub::Min());if(t==0)lo=v;__syncthreads();
 v=Reduce(temp.red).Reduce(mx,cub::Max());if(t==0){hi=v;prefix=0;remain=wanted;}__syncthreads();
 // Exact FP32-key refinement inside just the coarse boundary bin.
 if(lo!=hi){
  for(int pass=3;pass>=0;--pass){
   if((lo>>(pass*8))==(hi>>(pass*8))){if(t==0)prefix=lo&(~0u<<(pass*8));__syncthreads();continue;}
   h[t]=0;__syncthreads();unsigned mask=pass==3?0u:(~0u<<((pass+1)*8)),pref=prefix;
   for(int base=0;base<M;base+=THREADS){int i=base+t;unsigned u=i<M?kk[i]:0;add_hist(h,(u>>(pass*8))&255,i<M&&(u&mask)==pref);}
   __syncthreads();int b=255-t,cnt=h[b],suffix;
   Scan(temp.scan).InclusiveSum(cnt,suffix);__syncthreads();int need=remain;__syncthreads();
   if(suffix>=need&&suffix-cnt<need){prefix=pref|((unsigned)b<<(pass*8));remain=need-(suffix-cnt);}
   __syncthreads();
  }
 }else{if(t==0)prefix=lo;__syncthreads();}
 unsigned threshold=prefix;int ng=0,ne=0;
 for(int i=t;i<M;i+=THREADS){unsigned u=kk[i];ng+=u>threshold;ne+=u==threshold;}
 v=Reduce(temp.red).Sum((unsigned)ng);if(t==0)greater=v;__syncthreads();
 if(t==0)tiecount=wanted-greater;__syncthreads();int bg=0,be=0;
 for(int base=0;base<M;base+=THREADS){
  int i=base+t;unsigned u=i<M?kk[i]:0;int g=i<M&&u>threshold,e=i<M&&u==threshold,pg,pe,ag,ae;
  Scan(temp.scan).ExclusiveSum(g,pg,ag);__syncthreads();Scan(temp.scan).ExclusiveSum(e,pe,ae);__syncthreads();
  if(g)out[(int64_t)r*K+G+bg+pg]=ids[(int64_t)r*N+i];
  if(e&&be+pe<tiecount)out[(int64_t)r*K+G+greater+be+pe]=ids[(int64_t)r*N+i];
  bg+=ag;be+=ae;
 }
}
void select_out(at::Tensor s,at::Tensor pos,at::Tensor out,at::Tensor hist,at::Tensor state,at::Tensor offsets,at::Tensor ids,at::Tensor keys){
 c10::cuda::CUDAGuard guard(s.device());
 TORCH_CHECK(s.dim()==2&&pos.dim()==1&&out.dim()==2,"rank");
 for(const auto& x:{s,pos,out,hist,state,offsets,ids,keys})TORCH_CHECK(x.is_cuda()&&x.device()==s.device()&&x.is_contiguous(),"device/contiguous");
 TORCH_CHECK(s.scalar_type()==at::kFloat&&pos.scalar_type()==at::kLong,"input dtype");
 for(const auto& x:{out,hist,state,offsets,ids,keys})TORCH_CHECK(x.scalar_type()==at::kInt,"workspace/output dtype");
 int64_t R=s.size(0),N=s.size(1),K=out.size(1),C=(N+TILE-1)/TILE;
 TORCH_CHECK(R<=65535&&N>0&&N<=2147483647&&K>0&&K<=N&&pos.numel()==R&&out.size(0)==R,"shape");
 TORCH_CHECK(hist.numel()>=R*C*BINS&&state.numel()>=R*4&&offsets.numel()>=R*C*2&&ids.numel()>=R*N&&keys.numel()>=R*N,"workspace capacity");
 if(!R)return;auto stream=at::cuda::getCurrentCUDAStream();dim3 grid(std::min<int64_t>(C,32),R);
 coarse_hist<<<grid,THREADS,0,stream>>>(s.data_ptr<float>(),pos.data_ptr<int64_t>(),hist.data_ptr<int>(),N,K,C);
 plan<<<R,THREADS,0,stream>>>(hist.data_ptr<int>(),pos.data_ptr<int64_t>(),state.data_ptr<int>(),offsets.data_ptr<int>(),N,K,C);
 collect<<<grid,THREADS,0,stream>>>(s.data_ptr<float>(),pos.data_ptr<int64_t>(),state.data_ptr<int>(),offsets.data_ptr<int>(),ids.data_ptr<int>(),keys.data_ptr<int>(),out.data_ptr<int>(),N,K,C);
 refine<<<R,THREADS,0,stream>>>(pos.data_ptr<int64_t>(),state.data_ptr<int>(),ids.data_ptr<int>(),keys.data_ptr<int>(),out.data_ptr<int>(),N,K);
 C10_CUDA_KERNEL_LAUNCH_CHECK();
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("select_out",&select_out);}
