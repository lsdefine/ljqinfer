// Decode-only exact multi-CTA top-k; no query-count dispatch.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cub/block/block_scan.cuh>
__device__ __forceinline__ unsigned key(float v){unsigned u=__float_as_uint(v);return (u&0x80000000u)?~u:(u|0x80000000u);}
constexpr int CHUNK=1024;
__global__ void histogram(const float* scores,const int64_t* pos,int* hist,const int* state,int N,int K,int C,int pass){
 int row=blockIdx.y,c=blockIdx.x,t=threadIdx.x;
 int lim=max(0,min((int)(pos[row]+1),N));
 unsigned prefix=pass==3?0u:(unsigned)state[row*2];
 unsigned mask=pass==3?0u:(0xffffffffu<<((pass+1)*8));
 __shared__ int h[256];h[t]=0;__syncthreads();
 if(lim>K){
  for(int j=0;j<4;j++){int i=c*CHUNK+j*256+t;if(i<lim){unsigned u=key(scores[(long)row*N+i]);if((u&mask)==prefix)atomicAdd(h+((u>>(pass*8))&255),1);}}
 }
 __syncthreads();hist[((long)row*C+c)*256+t]=h[t];
}
__global__ void threshold(const int* hist,int* state,int C,int K,int pass){
 int row=blockIdx.x,t=threadIdx.x,b=255-t,count=0;
 for(int c=0;c<C;c++)count+=hist[((long)row*C+c)*256+b];
 using Scan=cub::BlockScan<int,256>;__shared__ Scan::TempStorage temp;
 int suffix;Scan(temp).InclusiveSum(count,suffix);
 int k=pass==3?K:state[row*2+1];unsigned prefix=pass==3?0u:(unsigned)state[row*2];
 __syncthreads();
 if(suffix>=k&&suffix-count<k){state[row*2]=(int)(prefix|((unsigned)b<<(pass*8)));state[row*2+1]=k-(suffix-count);}
}
__global__ void counts(const float* scores,const int64_t* pos,const int* state,int* counts,int N,int K,int C){
 int row=blockIdx.y,c=blockIdx.x,t=threadIdx.x,lim=max(0,min((int)(pos[row]+1),N));unsigned thr=(unsigned)state[row*2];
 int g=0,e=0;
 if(lim>K)for(int j=0;j<4;j++){int i=c*CHUNK+j*256+t;if(i<lim){unsigned u=key(scores[(long)row*N+i]);g+=u>thr;e+=u==thr;}}
 g=__reduce_add_sync(0xffffffffu,g);e=__reduce_add_sync(0xffffffffu,e);
 __shared__ int gs[8],es[8];if((t&31)==0){gs[t>>5]=g;es[t>>5]=e;}__syncthreads();
 if(t==0){g=0;e=0;for(int w=0;w<8;w++){g+=gs[w];e+=es[w];}counts[((long)row*C+c)*2]=g;counts[((long)row*C+c)*2+1]=e;}
}
__global__ void emit(const float* scores,const int64_t* pos,const int* state,const int* counts,int* out,int N,int K,int C){
 int row=blockIdx.y,c=blockIdx.x,t=threadIdx.x,lim=max(0,min((int)(pos[row]+1),N));
 if(lim<=K){for(int j=0;j<4;j++){int i=c*CHUNK+j*256+t;if(i<K)out[(long)row*K+i]=i<lim?i:-1;}return;}
 unsigned thr=(unsigned)state[row*2];int ties=state[row*2+1],ng=K-ties;
 __shared__ int gs[32],es[32],bg,be;
 unsigned gm[4],em[4];int lane=t&31,warp=t>>5;
 for(int j=0;j<4;j++){
  int i=c*CHUNK+j*256+t;unsigned u=i<lim?key(scores[(long)row*N+i]):0u;
  gm[j]=__ballot_sync(0xffffffffu,i<lim&&u>thr);em[j]=__ballot_sync(0xffffffffu,i<lim&&u==thr);
  if(lane==0){gs[j*8+warp]=__popc(gm[j]);es[j*8+warp]=__popc(em[j]);}
 }
 __syncthreads();
 if(t<32){
  int g=gs[t],e=es[t],cg=g,ce=e;
  for(int d=1;d<32;d*=2){int a=__shfl_up_sync(0xffffffffu,g,d),b=__shfl_up_sync(0xffffffffu,e,d);if(t>=d){g+=a;e+=b;}}
  gs[t]=g-cg;es[t]=e-ce;
  g=0;e=0;for(int i=t;i<c;i+=32){g+=counts[((long)row*C+i)*2];e+=counts[((long)row*C+i)*2+1];}
  g=__reduce_add_sync(0xffffffffu,g);e=__reduce_add_sync(0xffffffffu,e);if(t==0){bg=g;be=e;}
 }
 __syncthreads();unsigned mask=lane==0?0u:((1u<<lane)-1u);
 for(int j=0;j<4;j++){
  int i=c*CHUNK+j*256+t;
  if(gm[j]&(1u<<lane)){int dst=bg+gs[j*8+warp]+__popc(gm[j]&mask);if(dst<ng)out[(long)row*K+dst]=i;}
  if(em[j]&(1u<<lane)){int dst=be+es[j*8+warp]+__popc(em[j]&mask);if(dst<ties)out[(long)row*K+ng+dst]=i;}
 }
}
void select_out(at::Tensor scores,at::Tensor positions,at::Tensor out){
 c10::cuda::CUDAGuard guard(scores.device());
 TORCH_CHECK(scores.dim()==2&&positions.dim()==1&&out.dim()==2,"rank");
 for(const auto& x:{scores,positions,out})TORCH_CHECK(x.is_cuda()&&x.device()==scores.device()&&x.is_contiguous(),"device/contiguous");
 TORCH_CHECK(scores.scalar_type()==at::kFloat&&positions.scalar_type()==at::kLong&&out.scalar_type()==at::kInt,"dtype");
 int R=scores.size(0),N=scores.size(1),K=out.size(1),C=(N+CHUNK-1)/CHUNK;
 TORCH_CHECK(positions.numel()==R&&out.size(0)==R&&K>0&&K<=N,"shape");if(!R)return;
 auto opt=out.options();auto h=at::empty({R,C,256},opt),state=at::zeros({R,2},opt),cnt=at::empty({R,C,2},opt);
 auto stream=at::cuda::getCurrentCUDAStream();dim3 grid(C,R);
 for(int pass=3;pass>=0;pass--){histogram<<<grid,256,0,stream>>>(scores.data_ptr<float>(),positions.data_ptr<int64_t>(),h.data_ptr<int>(),state.data_ptr<int>(),N,K,C,pass);threshold<<<R,256,0,stream>>>(h.data_ptr<int>(),state.data_ptr<int>(),C,K,pass);}
 counts<<<grid,256,0,stream>>>(scores.data_ptr<float>(),positions.data_ptr<int64_t>(),state.data_ptr<int>(),cnt.data_ptr<int>(),N,K,C);
 emit<<<grid,256,0,stream>>>(scores.data_ptr<float>(),positions.data_ptr<int64_t>(),state.data_ptr<int>(),cnt.data_ptr<int>(),out.data_ptr<int>(),N,K,C);
 C10_CUDA_KERNEL_LAUNCH_CHECK();
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("select_out",&select_out);}
