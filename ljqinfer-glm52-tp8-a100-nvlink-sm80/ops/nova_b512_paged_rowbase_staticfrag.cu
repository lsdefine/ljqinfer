#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <float.h>
#include <mutex>
#include <array>
#include <atomic>
using namespace nvcuda;
#ifndef NATIVE_H8_SPLITS
#define NATIVE_H8_SPLITS 108
#endif
#ifndef NOVA_ROWBASE
#define NOVA_ROWBASE 1
#endif
namespace {constexpr int DL=512,DR=64,D=576,TK=16,OH=32;
__device__ __forceinline__ void cp16(void* d,const void* s,int n){unsigned a=__cvta_generic_to_shared(d);asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;"::"r"(a),"l"(s),"r"(n));}
__device__ __forceinline__ void prepare_row_bases(unsigned long long* row_base,const half* pool,const int64_t* pt,int page_size,int kb,int beg,int end){
 int r=threadIdx.x;
 if(r<TK){int k=kb+r;const half* src=pool;if(k>=beg&&k<end){int logical_page=k/page_size;int offset=k-logical_page*page_size;long long pp=pt[logical_page];src=pool+(pp*page_size+(long long)offset)*(long long)D;}row_base[r]=reinterpret_cast<unsigned long long>(src);}
 __syncthreads();
}
__global__ void fused_tc(const half* ql,const half* qr,const half* pool,const int64_t* pt,const int* k0,float* po,float* pl,float* pm,int M,int Kcap,int page_size,int S,int H,int Q){
 int sp=blockIdx.x, mg=blockIdx.y, tid=threadIdx.x, warp=tid>>5, lane=tid&31, m0=mg*16; int qid_max=min((m0+15)/H,Q-1); int K=max(0,min(*k0+qid_max+1,Kcap));int beg=(long long)K*sp/S,end=(long long)K*(sp+1)/S;
 __shared__ __align__(16) half qs[16*D],ks2[2][TK][D],ps[256];__shared__ __align__(16) float scores[256],qpart[4][256],rmax[16],alpha[16];
 for(int i=tid;i<16*D;i+=blockDim.x){int r=i/D,c=i%D,mi=m0+r;qs[i]=mi<M?(c<DL?ql[mi*DL+c]:qr[mi*DR+c-DL]):__float2half(0.f);}if(tid<16)rmax[tid]=-FLT_MAX;__syncthreads();float lacc=0.f;
 wmma::fragment<wmma::accumulator,16,16,16,float> outfrag0,outfrag1;wmma::fill_fragment(outfrag0,0.f);wmma::fill_fragment(outfrag1,0.f);
 int kb=beg-(beg%16);
 for(int it=0;kb<end;kb+=16,it++){int cur=it&1,nxt=cur^1;
  if(it==0){
#if NOVA_ROWBASE
   prepare_row_bases(reinterpret_cast<unsigned long long*>(ps),pool,pt,page_size,kb,beg,end);
   for(int x=tid*8;x<TK*D;x+=blockDim.x*8){int r=x/D,c=x%D,k=kb+r;bool ok=k>=beg&&k<end;const half* src=reinterpret_cast<const half*>(reinterpret_cast<unsigned long long*>(ps)[r]);cp16(&ks2[cur][r][c],src+c,ok?16:0);}
#else
   for(int x=tid*8;x<TK*D;x+=blockDim.x*8){int r=x/D,c=x%D,k=kb+r;bool ok=k>=beg&&k<end;const half* src=pool;if(ok){long long pp=pt[k/page_size];src=pool+(pp*page_size+(k%page_size))*(long long)D+c;}cp16(&ks2[cur][r][c],src,ok?16:0);}
#endif
   asm volatile("cp.async.commit_group;");asm volatile("cp.async.wait_group 0;");__syncthreads();
 }
 if(kb+16<end){
#if NOVA_ROWBASE
   prepare_row_bases(reinterpret_cast<unsigned long long*>(ps),pool,pt,page_size,kb+16,beg,end);
   for(int x=tid*8;x<TK*D;x+=blockDim.x*8){int r=x/D,c=x%D,k=kb+16+r;bool ok=k>=beg&&k<end;const half* src=reinterpret_cast<const half*>(reinterpret_cast<unsigned long long*>(ps)[r]);cp16(&ks2[nxt][r][c],src+c,ok?16:0);}
#else
   for(int x=tid*8;x<TK*D;x+=blockDim.x*8){int r=x/D,c=x%D,k=kb+16+r;bool ok=k>=beg&&k<end;const half* src=pool;if(ok){long long pp=pt[k/page_size];src=pool+(pp*page_size+(k%page_size))*(long long)D+c;}cp16(&ks2[nxt][r][c],src,ok?16:0);}
#endif
   asm volatile("cp.async.commit_group;");
 }
  half* ks=&ks2[cur][0][0];
  if(warp<4){wmma::fragment<wmma::accumulator,16,16,16,float> sc;wmma::fill_fragment(sc,0.f);for(int d=warp*16;d<D;d+=64){wmma::fragment<wmma::matrix_a,16,16,16,half,wmma::row_major>a;wmma::fragment<wmma::matrix_b,16,16,16,half,wmma::col_major>b;wmma::load_matrix_sync(a,qs+d,D);wmma::load_matrix_sync(b,ks+d,D);wmma::mma_sync(sc,a,b,sc);}wmma::store_matrix_sync(qpart[warp],sc,16,wmma::mem_row_major);}__syncthreads();if(tid<256)scores[tid]=qpart[0][tid]+qpart[1][tid]+qpart[2][tid]+qpart[3][tid];
  __syncthreads();if(tid<16){int r=tid;int qid=min((m0+r)/H,Q-1);int Kr=max(0,min(*k0+qid+1,Kcap));float tm=-FLT_MAX;for(int c=0;c<16;c++){int k=kb+c;if(m0+r<M&&k>=beg&&k<end&&k<Kr)tm=fmaxf(tm,scores[r*16+c]*0.0625f);}float nm=fmaxf(rmax[r],tm);alpha[r]=(rmax[r]==-FLT_MAX)?0.f:__expf(rmax[r]-nm);rmax[r]=nm;}__syncthreads();
  {int rr=lane>>2;float a0=alpha[rr],a1=alpha[rr+8];
#define SCALE_OUTFRAG(f) do{(f).x[0]*=a0;(f).x[1]*=a0;(f).x[4]*=a0;(f).x[5]*=a0;(f).x[2]*=a1;(f).x[3]*=a1;(f).x[6]*=a1;(f).x[7]*=a1;}while(0)
   SCALE_OUTFRAG(outfrag0);SCALE_OUTFRAG(outfrag1);
#undef SCALE_OUTFRAG
  }
  if(tid<256){int r=tid>>4,col=tid&15,k=kb+col;int qid=min((m0+r)/H,Q-1);int Kr=max(0,min(*k0+qid+1,Kcap));float pv=(m0+r<M&&k>=beg&&k<end&&k<Kr)?__expf(scores[tid]*0.0625f-rmax[r]):0.f;ps[tid]=__float2half_rn(pv);}__syncthreads();
  if(tid<16){float z=0.f;for(int c=0;c<16;c++)z+=__half2float(ps[tid*16+c]);lacc=lacc*alpha[tid]+z;}__syncthreads();
  {wmma::fragment<wmma::matrix_a,16,16,16,half,wmma::row_major>a;wmma::fragment<wmma::matrix_b,16,16,16,half,wmma::row_major>b;wmma::load_matrix_sync(a,ps,16);wmma::load_matrix_sync(b,ks+warp*16,D);wmma::mma_sync(outfrag0,a,b,outfrag0);wmma::load_matrix_sync(b,ks+(warp+16)*16,D);wmma::mma_sync(outfrag1,a,b,outfrag1);}
  if(kb+16<end)asm volatile("cp.async.wait_group 0;");__syncthreads();}
 wmma::store_matrix_sync(po+(long long)m0*S*DL+(long long)sp*DL+warp*16,outfrag0,S*DL,wmma::mem_row_major);wmma::store_matrix_sync(po+(long long)m0*S*DL+(long long)sp*DL+(warp+16)*16,outfrag1,S*DL,wmma::mem_row_major);
 if(tid<16&&m0+tid<M){pl[(m0+tid)*S+sp]=lacc;pm[(m0+tid)*S+sp]=rmax[tid];}
}
template<int SS> __global__ void reduce_fixed(const float* __restrict__ po,const float* __restrict__ pl,const float* __restrict__ pm,half* __restrict__ out,int M){int m=blockIdx.x;__shared__ float den,gm,warp_mx[16];float mx=-FLT_MAX;for(int s=threadIdx.x;s<SS;s+=blockDim.x)mx=fmaxf(mx,pm[m*SS+s]);for(int o=16;o;o>>=1)mx=fmaxf(mx,__shfl_down_sync(0xffffffff,mx,o));if((threadIdx.x&31)==0)warp_mx[threadIdx.x>>5]=mx;__syncthreads();if(threadIdx.x==0){gm=-FLT_MAX;for(int w=0;w<16;++w)gm=fmaxf(gm,warp_mx[w]);den=0.f;}__syncthreads();float z=0.f;for(int s=threadIdx.x;s<SS;s+=blockDim.x)z+=pl[m*SS+s]*__expf(pm[m*SS+s]-gm);for(int o=16;o;o>>=1)z+=__shfl_down_sync(0xffffffff,z,o);if((threadIdx.x&31)==0)atomicAdd(&den,z);__syncthreads();float id=1.f/den;for(int j=threadIdx.x;j<DL;j+=blockDim.x){float v=0.f;for(int s=0;s<SS;s++)v+=po[((long long)m*SS+s)*DL+j]*__expf(pm[m*SS+s]-gm);out[m*DL+j]=__float2half_rn(v*id);}}
__global__ void reduce_tc(const float* po,const float* pl,half* out,int M,int S){int m=blockIdx.x;__shared__ float den;float z=0.f;for(int s=threadIdx.x;s<S;s+=blockDim.x)z+=pl[m*S+s];if(threadIdx.x==0)den=0.f;__syncthreads();for(int o=16;o;o>>=1)z+=__shfl_down_sync(0xffffffff,z,o);if((threadIdx.x&31)==0)atomicAdd(&den,z);__syncthreads();for(int j=threadIdx.x;j<DL;j+=blockDim.x){float v=0.f;for(int s=0;s<S;s++)v+=po[(long long)m*S*DL+(long long)s*DL+j];out[m*DL+j]=__float2half_rn(v*(1.f/den));}}

// Static workspace sized for production H<=16 pad + Q<=4 (CUDA-graph safe after warmup alloc).
struct B512Ws { torch::Tensor po, pl, pm, ql_pad, qr_pad, out_pad; };
static B512Ws& b512_ws(const torch::Tensor& ref) {
  static std::array<B512Ws,8> ws; static std::array<std::once_flag,8> once;
  int d=ref.device().index(); TORCH_CHECK(d>=0&&d<8,"device index");
  std::call_once(once[d],[&,d](){
    c10::cuda::CUDAGuard g(ref.device());
    auto fopt=torch::TensorOptions().device(ref.device()).dtype(torch::kFloat32);
    auto hopt=torch::TensorOptions().device(ref.device()).dtype(torch::kFloat16);
    constexpr int Mmax=64, S=NATIVE_H8_SPLITS, Qmax=4, Hp=16;
    ws[d].po=torch::empty({Mmax,S,DL},fopt);
    ws[d].pl=torch::empty({Mmax,S},fopt);
    ws[d].pm=torch::empty({Mmax,S},fopt);
    ws[d].ql_pad=torch::empty({Qmax,Hp,DL},hopt);
    ws[d].qr_pad=torch::empty({Qmax,Hp,DR},hopt);
    ws[d].out_pad=torch::empty({Qmax,Hp,DL},hopt);
  });
  return ws[d];
}
} // namespace

// Host entry: pad H to multiple of 16 when needed (production H=8), then run B512 kernel.
static std::atomic<int64_t> native_h8_b512_calls{0};
int64_t native_h8_b512_call_count(){ return native_h8_b512_calls.load(std::memory_order_relaxed); }

torch::Tensor native_h8_b512_paged_out(torch::Tensor ql,torch::Tensor qr,torch::Tensor pool,torch::Tensor page_table,torch::Tensor k0,torch::Tensor out){
 native_h8_b512_calls.fetch_add(1, std::memory_order_relaxed);
 TORCH_CHECK(ql.is_cuda()&&qr.is_cuda()&&pool.is_cuda()&&page_table.is_cuda()&&k0.is_cuda());
 TORCH_CHECK(ql.scalar_type()==at::kHalf&&qr.scalar_type()==at::kHalf&&pool.scalar_type()==at::kHalf&&page_table.scalar_type()==at::kLong&&k0.scalar_type()==at::kInt);
 TORCH_CHECK(ql.dim()==3&&ql.size(2)==DL&&qr.sizes().slice(0,2)==ql.sizes().slice(0,2)&&qr.size(2)==DR&&pool.dim()==3&&pool.size(2)==D&&page_table.dim()==1);
 TORCH_CHECK(out.sizes()==ql.sizes(),"out shape");
 int Q=(int)ql.size(0), H=(int)ql.size(1);
 TORCH_CHECK(Q>=1&&Q<=4,"b512 Q in 1..4");
 TORCH_CHECK(H>=1&&H<=16,"b512 supports H<=16 (pad to 16)");
 int Hp=((H+15)/16)*16; // 16 for H=1..16
 int S=NATIVE_H8_SPLITS;
 c10::cuda::CUDAGuard g(ql.device());
 auto st=at::cuda::getCurrentCUDAStream();
 auto& w=b512_ws(ql);

 half* ql_ptr; half* qr_ptr; half* out_ptr;
 int H_run, M;
 if(Q==2&&H==8){
   // Native production layout: q0[H8],q1[H8] exactly fill the 16 WMMA rows.
   H_run=H; M=Q*H;
   fused_tc<<<dim3(S,1),512,0,st>>>(reinterpret_cast<half*>(ql.data_ptr<at::Half>()),reinterpret_cast<half*>(qr.data_ptr<at::Half>()),reinterpret_cast<half*>(pool.data_ptr<at::Half>()),page_table.data_ptr<int64_t>(),k0.data_ptr<int>(),w.po.data_ptr<float>(),w.pl.data_ptr<float>(),w.pm.data_ptr<float>(),M,page_table.size(0)*pool.size(1),pool.size(1),S,H_run,Q);
   C10_CUDA_KERNEL_LAUNCH_CHECK();
   reduce_fixed<NATIVE_H8_SPLITS><<<M,512,0,st>>>(w.po.data_ptr<float>(),w.pl.data_ptr<float>(),w.pm.data_ptr<float>(),reinterpret_cast<half*>(out.data_ptr<at::Half>()),M);
   C10_CUDA_KERNEL_LAUNCH_CHECK();
 }else if(Hp!=H){
   // zero-fill pad heads in static buffers (graph-capturable ops)
   auto ql_pad=w.ql_pad.narrow(0,0,Q);
   auto qr_pad=w.qr_pad.narrow(0,0,Q);
   auto out_pad=w.out_pad.narrow(0,0,Q);
   ql_pad.zero_(); qr_pad.zero_();
   ql_pad.narrow(1,0,H).copy_(ql);
   qr_pad.narrow(1,0,H).copy_(qr);
   ql_ptr=reinterpret_cast<half*>(ql_pad.data_ptr<at::Half>());
   qr_ptr=reinterpret_cast<half*>(qr_pad.data_ptr<at::Half>());
   out_ptr=reinterpret_cast<half*>(out_pad.data_ptr<at::Half>());
   H_run=Hp; M=Q*Hp;
   fused_tc<<<dim3(S,(M+15)/16),512,0,st>>>(ql_ptr,qr_ptr,reinterpret_cast<half*>(pool.data_ptr<at::Half>()),page_table.data_ptr<int64_t>(),k0.data_ptr<int>(),w.po.data_ptr<float>(),w.pl.data_ptr<float>(),w.pm.data_ptr<float>(),M,page_table.size(0)*pool.size(1),pool.size(1),S,H_run,Q);
   C10_CUDA_KERNEL_LAUNCH_CHECK();
   reduce_fixed<NATIVE_H8_SPLITS><<<M,512,0,st>>>(w.po.data_ptr<float>(),w.pl.data_ptr<float>(),w.pm.data_ptr<float>(),out_ptr,M);
   C10_CUDA_KERNEL_LAUNCH_CHECK();
   out.copy_(out_pad.narrow(1,0,H));
 }else{
   H_run=H; M=Q*H;
   TORCH_CHECK(w.po.size(0)>=M&&w.po.size(1)>=S,"b512 workspace too small");
   fused_tc<<<dim3(S,(M+15)/16),512,0,st>>>(reinterpret_cast<half*>(ql.data_ptr<at::Half>()),reinterpret_cast<half*>(qr.data_ptr<at::Half>()),reinterpret_cast<half*>(pool.data_ptr<at::Half>()),page_table.data_ptr<int64_t>(),k0.data_ptr<int>(),w.po.data_ptr<float>(),w.pl.data_ptr<float>(),w.pm.data_ptr<float>(),M,page_table.size(0)*pool.size(1),pool.size(1),S,H_run,Q);
   C10_CUDA_KERNEL_LAUNCH_CHECK();
   reduce_fixed<NATIVE_H8_SPLITS><<<M,512,0,st>>>(w.po.data_ptr<float>(),w.pl.data_ptr<float>(),w.pm.data_ptr<float>(),reinterpret_cast<half*>(out.data_ptr<at::Half>()),M);
   C10_CUDA_KERNEL_LAUNCH_CHECK();
 }
 return out;
}
