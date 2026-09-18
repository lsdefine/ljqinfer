// decode_rms_rope.cu - fused RMSNorm + RoPE for T=1..4 decode
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math.h>

__device__ __forceinline__ float warp_sum_f(float v){
#pragma unroll
  for(int d=16;d;d>>=1) v+=__shfl_down_sync(0xffffffff,v,d);
  return v;
}

__global__ void rms_norm_half_kernel(const half* __restrict__ x, const half* __restrict__ w,
                                     half* __restrict__ y, int T, int D, float eps){
  int t=blockIdx.x; if(t>=T) return;
  const half* xi=x+(size_t)t*D;
  half* yi=y+(size_t)t*D;
  float sum=0.f;
  for(int i=threadIdx.x;i<D;i+=blockDim.x){
    float v=__half2float(xi[i]); sum+=v*v;
  }
  // block reduce: warp sums into smem, then first warp reduces
  __shared__ float sm[32];
  float wsum=warp_sum_f(sum);
  int lane=threadIdx.x & 31;
  int wid=threadIdx.x >> 5;
  if(lane==0) sm[wid]=wsum;
  __syncthreads();
  float tot=0.f;
  int nwarps=(blockDim.x+31)>>5;
  if(threadIdx.x < nwarps) tot=sm[threadIdx.x];
  if(wid==0){
    tot=warp_sum_f(tot);
    if(lane==0) sm[0]=tot;
  }
  __syncthreads();
  tot=sm[0]; // all threads read reduced sum
  float inv=rsqrtf(tot/(float)D + eps);
  for(int i=threadIdx.x;i<D;i+=blockDim.x){
    float v=__half2float(xi[i])*inv*__half2float(w[i]);
    yi[i]=__float2half_rn(v);
  }
}

__global__ void rope_half_kernel(const half* __restrict__ x, const int64_t* __restrict__ pos,
                                 half* __restrict__ y, int T, int H){
  int th=blockIdx.x; if(th>=T*H) return;
  int t=th/H; int h=th%H;
  int i=threadIdx.x;
  if(i>=32) return;
  const half* xi=x+((size_t)t*H+h)*64;
  half* yi=y+((size_t)t*H+h)*64;
  float p=(float)pos[t];
  float inv_freq=powf(8000000.f, -2.f*(float)i/64.f);
  float tht=p*inv_freq;
  float c=cosf(tht), s=sinf(tht);
  float e=__half2float(xi[2*i]);
  float o=__half2float(xi[2*i+1]);
  yi[2*i]=__float2half_rn(e*c - o*s);
  yi[2*i+1]=__float2half_rn(e*s + o*c);
}

// Stride-aware RoPE for qb[...,192:256].  The last dimension remains contiguous,
// while token/head strides may be larger than 64 elements.
__global__ void rope_half_strided_kernel(const half* __restrict__ x,
                                         const int64_t* __restrict__ pos,
                                         half* __restrict__ y,
                                         int T, int H,
                                         int64_t xs0, int64_t xs1){
  int th=blockIdx.x; if(th>=T*H) return;
  int t=th/H; int h=th%H;
  int i=threadIdx.x;
  if(i>=32) return;
  const half* xi=x+(size_t)t*xs0+(size_t)h*xs1;
  half* yi=y+(size_t)(t*H+h)*64;
  float p=(float)pos[t];
  float inv_freq=powf(8000000.f, -2.f*(float)i/64.f);
  float tht=p*inv_freq;
  float c=cosf(tht), s=sinf(tht);
  float e=__half2float(xi[2*i]);
  float o=__half2float(xi[2*i+1]);
  yi[2*i]=__float2half_rn(e*c - o*s);
  yi[2*i+1]=__float2half_rn(e*s + o*c);
}


// Decode KV postprocess: preserve production RMS/RoPE arithmetic while writing
// the final 576-wide row directly into cache[position]. One 256-thread block
// per token matches rms_norm_half_out(D=512)'s reduction order exactly.
__global__ void kv_post_cache_fused_kernel(const half* __restrict__ kv,
                                            const half* __restrict__ w,
                                            const int64_t* __restrict__ pos,
                                            half* __restrict__ pool,
                                            const int64_t* __restrict__ page_table,
                                            int T, int page_size, int n_page){
  int t=blockIdx.x; if(t>=T) return;
  int64_t pidx=pos[t];
  const int64_t cache_rows=(int64_t)page_size*n_page;
  if(pidx<0 || pidx>=cache_rows) return;
  const int logical_page=(int)(pidx/page_size);
  const int offset=(int)(pidx-(int64_t)logical_page*page_size);
  const int64_t physical_page=page_table[logical_page];
  if(physical_page<0 || physical_page>=n_page) return;
  const half* xi=kv+(size_t)t*576;
  half* yi=pool+((size_t)physical_page*page_size+offset)*576;
  float sum=0.f;
  for(int i=threadIdx.x;i<512;i+=blockDim.x){
    float v=__half2float(xi[i]); sum+=v*v;
  }
  __shared__ float sm[32];
  float wsum=warp_sum_f(sum);
  int lane=threadIdx.x & 31;
  int wid=threadIdx.x >> 5;
  if(lane==0) sm[wid]=wsum;
  __syncthreads();
  float tot=0.f;
  int nwarps=(blockDim.x+31)>>5;
  if(threadIdx.x<nwarps) tot=sm[threadIdx.x];
  if(wid==0){
    tot=warp_sum_f(tot);
    if(lane==0) sm[0]=tot;
  }
  __syncthreads();
  tot=sm[0];
  float inv=rsqrtf(tot/512.f + 1e-5f);
  for(int i=threadIdx.x;i<512;i+=blockDim.x){
    float v=__half2float(xi[i])*inv*__half2float(w[i]);
    yi[i]=__float2half_rn(v);
  }
  int i=threadIdx.x;
  if(i<32){
    float pf=(float)pidx;
    float inv_freq=powf(8000000.f, -2.f*(float)i/64.f);
    float tht=pf*inv_freq;
    float c=cosf(tht), ss=sinf(tht);
    float e=__half2float(xi[512+2*i]);
    float o=__half2float(xi[512+2*i+1]);
    yi[512+2*i]=__float2half_rn(e*c-o*ss);
    yi[512+2*i+1]=__float2half_rn(e*ss+o*c);
  }
}

torch::Tensor kv_post_cache_fused_out(torch::Tensor kv, torch::Tensor w,
                                        torch::Tensor positions, torch::Tensor pool,
                                        torch::Tensor page_table){
  TORCH_CHECK(kv.is_cuda()&&w.is_cuda()&&positions.is_cuda()&&pool.is_cuda()&&page_table.is_cuda());
  TORCH_CHECK(kv.dtype()==torch::kFloat16&&w.dtype()==torch::kFloat16&&pool.dtype()==torch::kFloat16);
  TORCH_CHECK(positions.dtype()==torch::kInt64);
  TORCH_CHECK(kv.dim()==2&&kv.size(1)==576&&kv.size(0)>=1&&kv.size(0)<=4);
  TORCH_CHECK(w.dim()==1&&w.numel()==512);
  TORCH_CHECK(pool.dim()==3&&pool.size(2)==576);
  TORCH_CHECK(page_table.dim()==1&&page_table.dtype()==torch::kInt64&&page_table.numel()==pool.size(0));
  TORCH_CHECK(positions.dim()==1&&positions.numel()>=kv.size(0));
  TORCH_CHECK(kv.is_contiguous()&&w.is_contiguous()&&positions.is_contiguous()&&pool.is_contiguous()&&page_table.is_contiguous());
  c10::cuda::CUDAGuard g(kv.device());
  auto st=at::cuda::getCurrentCUDAStream();
  kv_post_cache_fused_kernel<<<(int)kv.size(0),256,0,st>>>(
    reinterpret_cast<const half*>(kv.data_ptr<at::Half>()),
    reinterpret_cast<const half*>(w.data_ptr<at::Half>()),
    positions.data_ptr<int64_t>(),
    reinterpret_cast<half*>(pool.data_ptr<at::Half>()),
    page_table.data_ptr<int64_t>(),
    (int)kv.size(0),(int)pool.size(1),(int)pool.size(0));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return pool;
}

torch::Tensor rope_half_strided_out(torch::Tensor x, torch::Tensor positions, torch::Tensor y){
  TORCH_CHECK(x.is_cuda()&&positions.is_cuda()&&y.is_cuda());
  TORCH_CHECK(x.dtype()==torch::kFloat16 && y.dtype()==torch::kFloat16);
  TORCH_CHECK(x.dim()==3 && x.size(-1)==64 && x.stride(-1)==1, "strided rope x");
  TORCH_CHECK(y.sizes()==x.sizes() && y.is_contiguous(), "strided rope y");
  TORCH_CHECK(positions.is_contiguous() && positions.numel()>=x.size(0), "strided rope positions");
  c10::cuda::CUDAGuard g(x.device());
  auto st=at::cuda::getCurrentCUDAStream();
  const half* xp=reinterpret_cast<const half*>(x.data_ptr<at::Half>());
  half* yp=reinterpret_cast<half*>(y.data_ptr<at::Half>());
  rope_half_strided_kernel<<<(int)(x.size(0)*x.size(1)),32,0,st>>>(
      xp, positions.data_ptr<int64_t>(), yp, (int)x.size(0),(int)x.size(1),
      x.stride(0),x.stride(1));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}

torch::Tensor rms_norm_half(torch::Tensor x, torch::Tensor w){
  TORCH_CHECK(x.is_cuda()&&w.is_cuda());
  TORCH_CHECK(x.dtype()==torch::kFloat16 && w.dtype()==torch::kFloat16);
  TORCH_CHECK(x.dim()==2 && w.dim()==1 && w.size(0)==x.size(1));
  x=x.contiguous(); w=w.contiguous();
  auto y=torch::empty_like(x);
  c10::cuda::CUDAGuard g(x.device());
  auto st=at::cuda::getCurrentCUDAStream();
  int T=x.size(0), D=x.size(1);
  int threads=256; if(D<256) threads=128;
  rms_norm_half_kernel<<<T,threads,0,st>>>(
    reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
    reinterpret_cast<const half*>(w.data_ptr<at::Half>()),
    reinterpret_cast<half*>(y.data_ptr<at::Half>()), T,D,1e-5f);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}

torch::Tensor rope_half(torch::Tensor x, torch::Tensor positions){
  TORCH_CHECK(x.is_cuda()&&positions.is_cuda());
  TORCH_CHECK(x.dtype()==torch::kFloat16);
  TORCH_CHECK(positions.dtype()==torch::kLong);
  x=x.contiguous(); positions=positions.contiguous();
  int64_t T,H;
  torch::Tensor xin;
  if(x.dim()==2){
    TORCH_CHECK(x.size(1)==64);
    T=x.size(0); H=1; xin=x.view({T,1,64});
  }else{
    TORCH_CHECK(x.dim()==3 && x.size(-1)==64);
    T=x.size(0); H=x.size(1); xin=x;
  }
  auto y=torch::empty_like(xin);
  c10::cuda::CUDAGuard g(x.device());
  auto st=at::cuda::getCurrentCUDAStream();
  rope_half_kernel<<<(int)(T*H),32,0,st>>>(
    reinterpret_cast<const half*>(xin.data_ptr<at::Half>()),
    positions.data_ptr<int64_t>(),
    reinterpret_cast<half*>(y.data_ptr<at::Half>()), (int)T,(int)H);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  if(x.dim()==2) return y.view({T,64});
  return y;
}
torch::Tensor rms_norm_half_out(torch::Tensor x, torch::Tensor w, torch::Tensor y){
  TORCH_CHECK(x.is_cuda()&&w.is_cuda()&&y.is_cuda());
  TORCH_CHECK(x.dtype()==torch::kFloat16 && w.dtype()==torch::kFloat16 && y.dtype()==torch::kFloat16);
  TORCH_CHECK(x.dim()==2 && y.sizes()==x.sizes());
  TORCH_CHECK(x.is_contiguous() && w.is_contiguous() && y.is_contiguous());
  c10::cuda::CUDAGuard g(x.device());
  auto st=at::cuda::getCurrentCUDAStream();
  int T=x.size(0), D=x.size(1);
  int threads=256; if(D<256) threads=128;
  rms_norm_half_kernel<<<T,threads,0,st>>>(
    reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
    reinterpret_cast<const half*>(w.data_ptr<at::Half>()),
    reinterpret_cast<half*>(y.data_ptr<at::Half>()), T,D,1e-5f);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}
torch::Tensor rope_half_out(torch::Tensor x, torch::Tensor positions, torch::Tensor y){
  TORCH_CHECK(x.is_cuda()&&positions.is_cuda()&&y.is_cuda());
  TORCH_CHECK(x.dtype()==torch::kFloat16 && y.dtype()==torch::kFloat16);
  TORCH_CHECK(x.is_contiguous() && y.is_contiguous() && positions.is_contiguous());
  int64_t T,H;
  if(x.dim()==2){ TORCH_CHECK(x.size(1)==64); T=x.size(0); H=1; }
  else { TORCH_CHECK(x.dim()==3 && x.size(-1)==64); T=x.size(0); H=x.size(1); }
  c10::cuda::CUDAGuard g(x.device());
  auto st=at::cuda::getCurrentCUDAStream();
  const half* xp=reinterpret_cast<const half*>(x.data_ptr<at::Half>());
  half* yp=reinterpret_cast<half*>(y.data_ptr<at::Half>());
  rope_half_kernel<<<(int)(T*H),32,0,st>>>(xp, positions.data_ptr<int64_t>(), yp, (int)T,(int)H);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}
