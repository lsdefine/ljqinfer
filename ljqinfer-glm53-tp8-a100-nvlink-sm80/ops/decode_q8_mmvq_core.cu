// q8_mmvq.cu - decode-specialized GGML Q8_0 matvec (T=1..4)
// Layout: packed row = nb blocks * 34 bytes; block = half d + 32 int8 qs
// Goal: beat naive q8_linear and cublas dequant path on T=1.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

#ifndef Q8_DUAL_WARPS
#define Q8_DUAL_WARPS 8
#endif
#ifndef Q8_DUAL_SMEMX
#define Q8_DUAL_SMEMX 0
#endif

static __device__ __forceinline__ float warp_sum32(float v){
#pragma unroll
  for(int d=16;d>0;d>>=1) v+=__shfl_down_sync(0xffffffff,v,d);
  return v;
}

// One warp owns one output row; lane iterates quant blocks in steps of 32.
// For each block: lane reads qs[lane], multiplies x[k0+lane]*d*q
// Then advances by 32 blocks using all lanes - actually 1 block per step with 32 lanes.
// Better: each lane handles one of 32 qs in a block; warps stride over blocks.
__device__ __forceinline__ float q8_dot_warp_t1(const half* __restrict__ x,
                                               const uint8_t* __restrict__ row,
                                               int K){
  const int lane=threadIdx.x & 31;
  const int nb=K>>5; // K/32
  float sum=0.f;
  // process 1 quant block per iteration; all 32 lanes active
  for(int b=0;b<nb;++b){
    const uint8_t* p=row+(size_t)b*34;
    // scale broadcast from lane0
    float d=__half2float(*reinterpret_cast<const half*>(p));
    // qs[lane]
    float q=(float)reinterpret_cast<const int8_t*>(p+2)[lane];
    float xv=__half2float(x[(b<<5)+lane]);
    float partial=xv*(d*q);
    sum+=warp_sum32(partial); // every lane gets full block sum? wait warp_sum only lane0 has full if we use down
    // fix: use full warp reduce properly - only accumulate on all lanes then reduce once at end
  }
  // The above adds reduced sum every block on ALL lanes incorrectly after shfl_down.
  // Rewritten below in kernel without this helper.
  return sum;
}

// Correct T=1 kernel: each warp = one output row
// Accumulator per-lane; reduce once at end.
__global__ void q8_mmvq_t1_kernel(const half* __restrict__ x,
                                  const uint8_t* __restrict__ w,
                                  half* __restrict__ y,
                                  int K,int N,int rb){
  const int row=blockIdx.x*blockDim.y+threadIdx.y;
  if(row>=N) return;
  const int lane=threadIdx.x; // 0..31
  const uint8_t* rowp=w+(size_t)row*rb;
  const int nb=K>>5;
  float sum=0.f;
  // grid-stride over blocks by warp: each lane always handles qs index = lane
  for(int b=0;b<nb;++b){
    const uint8_t* p=rowp+(size_t)b*34;
    float d=__half2float(*reinterpret_cast<const half*>(p));
    float q=(float)reinterpret_cast<const int8_t*>(p+2)[lane];
    float xv=__half2float(x[(b<<5)+lane]);
    sum=fmaf(xv,d*q,sum);
  }
  sum=warp_sum32(sum);
  if(lane==0) y[row]=__float2half_rn(sum);
}

// T=1 dual-matrix kernel: one launch, same row arithmetic/order as q8_mmvq_t1_kernel.
__global__ void q8_mmvq_dual_t1_kernel(const half* __restrict__ x,
    const uint8_t* __restrict__ w0, const uint8_t* __restrict__ w1,
    half* __restrict__ y0, half* __restrict__ y1,
    int K, int N0, int N1, int rb){
#if Q8_DUAL_SMEMX
  extern __shared__ half xs[];
  {
    const int tid=threadIdx.y*32+threadIdx.x;
    const int nthreads=blockDim.y*32;
    for(int k=tid;k<K;k+=nthreads) xs[k]=x[k];
    __syncthreads();
  }
#endif
  const int idx=blockIdx.x*blockDim.y+threadIdx.y;
  if(idx>=N0+N1) return;
  const int lane=threadIdx.x;
  const bool second=idx>=N0;
  const int row=second ? idx-N0 : idx;
  const uint8_t* rowp=(second?w1:w0)+(size_t)row*rb;
  half* y=second?y1:y0;
  float sum=0.f;
  for(int b=0;b<(K>>5);++b){
    const uint8_t* p=rowp+(size_t)b*34;
    float d=__half2float(*reinterpret_cast<const half*>(p));
    float q=(float)reinterpret_cast<const int8_t*>(p+2)[lane];
#if Q8_DUAL_SMEMX
    float xv=__half2float(xs[(b<<5)+lane]);
#else
    float xv=__half2float(x[(b<<5)+lane]);
#endif
    sum=fmaf(xv,d*q,sum);
  }
  sum=warp_sum32(sum);
  if(lane==0) y[row]=__float2half_rn(sum);
}


// T<=4 dual-matrix: one launch for q_a + kv_a (same row arithmetic as dual_t1, T sums)
template<int TMAX>
__global__ void q8_mmvq_dual_t_kernel(const half* __restrict__ x,
    const uint8_t* __restrict__ w0, const uint8_t* __restrict__ w1,
    half* __restrict__ y0, half* __restrict__ y1,
    int T, int K, int N0, int N1, int rb){
  const int idx=blockIdx.x*blockDim.y+threadIdx.y;
  if(idx>=N0+N1) return;
  const int lane=threadIdx.x;
  const bool second=idx>=N0;
  const int row=second ? idx-N0 : idx;
  const uint8_t* rowp=(second?w1:w0)+(size_t)row*rb;
  half* y=second?y1:y0;
  const int nb=K>>5;
  float sum[4]={0,0,0,0};
  for(int b=0;b<nb;++b){
    const uint8_t* p=rowp+(size_t)b*34;
    float d=__half2float(*reinterpret_cast<const half*>(p));
    float q=(float)reinterpret_cast<const int8_t*>(p+2)[lane];
    float wq=d*q;
    const int k0=b<<5;
#pragma unroll
    for(int t=0;t<TMAX;++t){
      if(t<T){
        float xv=__half2float(x[(size_t)t*K+k0+lane]);
        sum[t]=fmaf(xv,wq,sum[t]);
      }
    }
  }
#pragma unroll
  for(int t=0;t<TMAX;++t){
    if(t<T){
      float v=warp_sum32(sum[t]);
      if(lane==0) y[(size_t)t*(second?N1:N0)+row]=__float2half_rn(v);
    }
  }
}

// T=1 multi-row: each block has multiple warps (blockDim.y)
// Launch: block=(32, warps), grid=((N+warps-1)/warps)

// T<=4: same but accumulate T independent sums
template<int TMAX>
__global__ void q8_mmvq_t_kernel(const half* __restrict__ x,
                                 const uint8_t* __restrict__ w,
                                 half* __restrict__ y,
                                 int T,int K,int N,int rb){
  const int row=blockIdx.x*blockDim.y+threadIdx.y;
  if(row>=N) return;
  const int lane=threadIdx.x;
  const uint8_t* rowp=w+(size_t)row*rb;
  const int nb=K>>5;
  float sum[4]={0,0,0,0};
  for(int b=0;b<nb;++b){
    const uint8_t* p=rowp+(size_t)b*34;
    float d=__half2float(*reinterpret_cast<const half*>(p));
    float q=(float)reinterpret_cast<const int8_t*>(p+2)[lane];
    float wq=d*q;
    const int k0=b<<5;
#pragma unroll
    for(int t=0;t<TMAX;++t){
      if(t<T){
        float xv=__half2float(x[(size_t)t*K+k0+lane]);
        sum[t]=fmaf(xv,wq,sum[t]);
      }
    }
  }
#pragma unroll
  for(int t=0;t<TMAX;++t){
    if(t<T){
      float v=warp_sum32(sum[t]);
      if(lane==0) y[(size_t)t*N+row]=__float2half_rn(v);
    }
  }
}

// Selected-expert / id path: for each output channel c, use weight row = ids[c] * N_local? 
// Simpler ABI for MoE shared/select: y[i] = dot(x, W[ids[i]]) for i in 0..n_ids-1
// Here N = number of selected rows; packed is full [N_full, rb]; ids int32 [N]
__global__ void q8_mmvq_id_t1_kernel(const half* __restrict__ x,
                                     const uint8_t* __restrict__ w,
                                     const int32_t* __restrict__ ids,
                                     half* __restrict__ y,
                                     int K,int n_ids,int rb){
  const int i=blockIdx.x*blockDim.y+threadIdx.y;
  if(i>=n_ids) return;
  const int lane=threadIdx.x;
  const int row=ids[i];
  const uint8_t* rowp=w+(size_t)row*rb;
  const int nb=K>>5;
  float sum=0.f;
  for(int b=0;b<nb;++b){
    const uint8_t* p=rowp+(size_t)b*34;
    float d=__half2float(*reinterpret_cast<const half*>(p));
    float q=(float)reinterpret_cast<const int8_t*>(p+2)[lane];
    float xv=__half2float(x[(b<<5)+lane]);
    sum=fmaf(xv,d*q,sum);
  }
  sum=warp_sum32(sum);
  if(lane==0) y[i]=__float2half_rn(sum);
}


// T=1 fused RMS(weighted) + Q8_0 MMVQ. One 128-thread block owns 4 output rows.
// RMS reduction is shared by its 4 warps; avoids materializing normalized x and one launch.
__global__ void q8_mmvq_rms_t1_kernel(const half* __restrict__ x,
                                      const half* __restrict__ nw,
                                      const uint8_t* __restrict__ w,
                                      half* __restrict__ y,
                                      int K,int N,int rb,float eps){
  __shared__ float red[4];
  __shared__ float scale;
  const int tid=threadIdx.y*32+threadIdx.x;
  float ss=0.f;
  for(int k=tid;k<K;k+=128){ float v=__half2float(x[k]); ss=fmaf(v,v,ss); }
  ss=warp_sum32(ss);
  if(threadIdx.x==0) red[threadIdx.y]=ss;
  __syncthreads();
  if(tid==0) scale=rsqrtf((red[0]+red[1]+red[2]+red[3])/(float)K+eps);
  __syncthreads();
  const int row=blockIdx.x*4+threadIdx.y;
  if(row>=N) return;
  const int lane=threadIdx.x;
  const uint8_t* rowp=w+(size_t)row*rb;
  float sum=0.f;
  for(int b=0;b<(K>>5);++b){
    const uint8_t* p=rowp+(size_t)b*34;
    float d=__half2float(*reinterpret_cast<const half*>(p));
    float q=(float)reinterpret_cast<const int8_t*>(p+2)[lane];
    int k=(b<<5)+lane;
    float xv=__half2float(x[k])*__half2float(nw[k]);
    sum=fmaf(xv,d*q,sum);
  }
  sum=warp_sum32(sum);
  if(lane==0) y[row]=__float2half_rn(sum*scale);
}

// ==== T=2 specialized kernels (merged from exp_t2_attn, numerics = baseline) ====

// ---- T=2 plain MMVQ: dual-token acc + optional smem for both x rows ----
// smem layout: xs[0..K) = token0, xs[K..2K) = token1 (when dynamic smem >= 2*K*sizeof(half))
__global__ void q8_mmvq_t2_kernel(const half* __restrict__ x,
                                  const uint8_t* __restrict__ w,
                                  half* __restrict__ y,
                                  int K, int N, int rb, int use_smem) {
  extern __shared__ half xs[];
  const int tid = threadIdx.y * 32 + threadIdx.x;
  const int nthreads = blockDim.y * 32;
  if (use_smem) {
    for (int k = tid; k < K; k += nthreads) {
      xs[k] = x[k];
      xs[K + k] = x[(size_t)K + k];
    }
    __syncthreads();
  }
  const int row = blockIdx.x * blockDim.y + threadIdx.y;
  if (row >= N) return;
  const int lane = threadIdx.x;
  const uint8_t* rowp = w + (size_t)row * rb;
  const int nb = K >> 5;
  float sum0 = 0.f, sum1 = 0.f;
  for (int b = 0; b < nb; ++b) {
    const uint8_t* p = rowp + (size_t)b * 34;
    float d = __half2float(*reinterpret_cast<const half*>(p));
    float q = (float)reinterpret_cast<const int8_t*>(p + 2)[lane];
    float wq = d * q;
    const int k0 = b << 5;
    float xv0, xv1;
    if (use_smem) {
      xv0 = __half2float(xs[k0 + lane]);
      xv1 = __half2float(xs[K + k0 + lane]);
    } else {
      xv0 = __half2float(x[k0 + lane]);
      xv1 = __half2float(x[(size_t)K + k0 + lane]);
    }
    sum0 = fmaf(xv0, wq, sum0);
    sum1 = fmaf(xv1, wq, sum1);
  }
  sum0 = warp_sum32(sum0);
  sum1 = warp_sum32(sum1);
  if (lane == 0) {
    y[row] = __float2half_rn(sum0);
    y[(size_t)N + row] = __float2half_rn(sum1);
  }
}

// ---- T=2 dual MMVQ (q_a + kv_a): one launch, two weight matrices ----
__global__ void q8_mmvq_dual_t2_kernel(const half* __restrict__ x,
    const uint8_t* __restrict__ w0, const uint8_t* __restrict__ w1,
    half* __restrict__ y0, half* __restrict__ y1,
    int K, int N0, int N1, int rb) {
  extern __shared__ half xs[];
  const int lane = threadIdx.x;
  const int tid = threadIdx.y * 32 + lane;
  const int nthreads = blockDim.y * 32;
  for (int i = tid; i < 2 * K; i += nthreads) xs[i] = x[i];
  __syncthreads();
  const int idx = blockIdx.x * blockDim.y + threadIdx.y;
  if (idx >= N0 + N1) return;
  const bool second = idx >= N0;
  const int row = second ? idx - N0 : idx;
  const uint8_t* rowp = (second ? w1 : w0) + (size_t)row * rb;
  half* y = second ? y1 : y0;
  const int N = second ? N1 : N0;
  const int nb = K >> 5;
  float sum0 = 0.f, sum1 = 0.f;
  for (int b = 0; b < nb; ++b) {
    const uint8_t* p = rowp + (size_t)b * 34;
    float d = __half2float(*reinterpret_cast<const half*>(p));
    float q = (float)reinterpret_cast<const int8_t*>(p + 2)[lane];
    float wq = d * q;
    const int k0 = b << 5;
    float xv0 = __half2float(xs[k0 + lane]);
    float xv1 = __half2float(xs[K + k0 + lane]);
    sum0 = fmaf(xv0, wq, sum0);
    sum1 = fmaf(xv1, wq, sum1);
  }
  sum0 = warp_sum32(sum0);
  sum1 = warp_sum32(sum1);
  if (lane == 0) {
    y[row] = __float2half_rn(sum0);
    y[(size_t)N + row] = __float2half_rn(sum1);
  }
}

// ---- T=2 fused RMS + MMVQ: two scales, shared weight-row pass ----
// Block: 128 threads (4 warps). Warps share RMS reductions for both tokens,
// then each warp owns one output row and accumulates 2 tokens.
__global__ void q8_mmvq_rms_t2_kernel(const half* __restrict__ x,
                                      const half* __restrict__ nw,
                                      const uint8_t* __restrict__ w,
                                      half* __restrict__ y,
                                      int K, int N, int rb, float eps) {
  __shared__ float red0[4], red1[4];
  __shared__ float scale0, scale1;
  const int tid = threadIdx.y * 32 + threadIdx.x;
  float ss0 = 0.f, ss1 = 0.f;
  for (int k = tid; k < K; k += 128) {
    float v0 = __half2float(x[k]);
    float v1 = __half2float(x[(size_t)K + k]);
    ss0 = fmaf(v0, v0, ss0);
    ss1 = fmaf(v1, v1, ss1);
  }
  ss0 = warp_sum32(ss0);
  ss1 = warp_sum32(ss1);
  if (threadIdx.x == 0) {
    red0[threadIdx.y] = ss0;
    red1[threadIdx.y] = ss1;
  }
  __syncthreads();
  if (tid == 0) {
    scale0 = rsqrtf((red0[0] + red0[1] + red0[2] + red0[3]) / (float)K + eps);
    scale1 = rsqrtf((red1[0] + red1[1] + red1[2] + red1[3]) / (float)K + eps);
  }
  __syncthreads();
  const int row = blockIdx.x * 4 + threadIdx.y;
  if (row >= N) return;
  const int lane = threadIdx.x;
  const uint8_t* rowp = w + (size_t)row * rb;
  float sum0 = 0.f, sum1 = 0.f;
  for (int b = 0; b < (K >> 5); ++b) {
    const uint8_t* p = rowp + (size_t)b * 34;
    float d = __half2float(*reinterpret_cast<const half*>(p));
    float q = (float)reinterpret_cast<const int8_t*>(p + 2)[lane];
    float wq = d * q;
    int k = (b << 5) + lane;
    float nw_k = __half2float(nw[k]);
    float xv0 = __half2float(x[k]) * nw_k;
    float xv1 = __half2float(x[(size_t)K + k]) * nw_k;
    sum0 = fmaf(xv0, wq, sum0);
    sum1 = fmaf(xv1, wq, sum1);
  }
  sum0 = warp_sum32(sum0);
  sum1 = warp_sum32(sum1);
  if (lane == 0) {
    y[row] = __float2half_rn(sum0 * scale0);
    y[(size_t)N + row] = __float2half_rn(sum1 * scale1);
  }
}

// ---- T=2 grouped MMVQ: x[2,H,K] @ packed[H,N,rb] -> y[2,H,N] one launch ----
__global__ void q8_mmvq_grouped_t2_kernel(
    const half* __restrict__ x,
    const uint8_t* __restrict__ packed,
    half* __restrict__ y,
    int H, int N, int K, int rb,
    int64_t x_t_stride, int64_t x_h_stride,
    int64_t y_t_stride, int64_t y_h_stride) {
  const int warps = blockDim.y;
  const int lane = threadIdx.x;
  const int local_row = threadIdx.y;
  const int head = blockIdx.y;
  if (head >= H) return;
  const int row = blockIdx.x * warps + local_row;
  if (row >= N) return;
  const half* xh0 = x + (int64_t)head * x_h_stride;
  const half* xh1 = x + x_t_stride + (int64_t)head * x_h_stride;
  const uint8_t* rowp = packed + ((int64_t)head * N + row) * (int64_t)rb;
  float acc0 = 0.f, acc1 = 0.f;
  const int nb = K >> 5;
  for (int b = 0; b < nb; ++b) {
    const uint8_t* p = rowp + (size_t)b * 34;
    float d = __half2float(*reinterpret_cast<const half*>(p));
    float q = (float)reinterpret_cast<const int8_t*>(p + 2)[lane];
    float wq = d * q;
    int k = (b << 5) + lane;
    acc0 = fmaf(wq, __half2float(xh0[k]), acc0);
    acc1 = fmaf(wq, __half2float(xh1[k]), acc1);
  }
#pragma unroll
  for (int d = 16; d; d >>= 1) {
    acc0 += __shfl_down_sync(0xffffffff, acc0, d);
    acc1 += __shfl_down_sync(0xffffffff, acc1, d);
  }
  if (lane == 0) {
    y[(int64_t)head * y_h_stride + row] = __float2half(acc0);
    y[y_t_stride + (int64_t)head * y_h_stride + row] = __float2half(acc1);
  }
}

static void check_inputs(torch::Tensor x, torch::Tensor packed, int64_t K){
  TORCH_CHECK(x.is_cuda() && packed.is_cuda(), "cuda");
  TORCH_CHECK(x.dtype()==torch::kFloat16 && packed.dtype()==torch::kUInt8, "dtype");
  TORCH_CHECK(x.is_contiguous() && packed.is_contiguous(), "contig");
  TORCH_CHECK(K%32==0, "K%32");
  TORCH_CHECK(x.dim()==2 && x.size(1)==K, "x shape");
  TORCH_CHECK(packed.dim()==2, "packed 2d");
  int64_t rb=K/32*34;
  TORCH_CHECK(packed.size(1)==rb, "row bytes");
}

torch::Tensor q8_mmvq_forward(torch::Tensor x, torch::Tensor packed, int64_t K){
  check_inputs(x,packed,K);
  const int T=x.size(0), N=packed.size(0), rb=packed.size(1);
  auto y=torch::empty({T,N}, x.options());
  c10::cuda::CUDAGuard guard(x.device());
  auto stream=at::cuda::getCurrentCUDAStream();
  const int warps=4; // 4 rows per block
  dim3 block(32, warps);
  dim3 grid((N+warps-1)/warps);
  if(T==1){
    q8_mmvq_t1_kernel<<<grid,block,0,stream>>>(
      reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
      packed.data_ptr<uint8_t>(),
      reinterpret_cast<half*>(y.data_ptr<at::Half>()),
      (int)K,N,rb);
  }else if(T==2){
    const int use_smem=(K<=12288)?1:0;
    size_t smem=use_smem?(size_t)K*2*sizeof(half):0;
    q8_mmvq_t2_kernel<<<grid,block,smem,stream>>>(
      reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
      packed.data_ptr<uint8_t>(),
      reinterpret_cast<half*>(y.data_ptr<at::Half>()),
      (int)K,N,rb,use_smem);
  }else if(T<=4){
    q8_mmvq_t_kernel<4><<<grid,block,0,stream>>>(
      reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
      packed.data_ptr<uint8_t>(),
      reinterpret_cast<half*>(y.data_ptr<at::Half>()),
      T,(int)K,N,rb);
  }else{
    TORCH_CHECK(false, "q8_mmvq supports T<=4; use prefill path for larger T");
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}

torch::Tensor q8_mmvq_forward_out(torch::Tensor x, torch::Tensor packed, int64_t K, torch::Tensor y){

  check_inputs(x,packed,K);

  const int T=x.size(0), N=packed.size(0), rb=packed.size(1);

  TORCH_CHECK(y.is_cuda() && y.dtype()==x.dtype() && y.is_contiguous(), "y");

  TORCH_CHECK(y.size(0)==T && y.size(1)==N, "y shape");

  c10::cuda::CUDAGuard guard(x.device());

  auto stream=at::cuda::getCurrentCUDAStream();

  const int warps=4;

  dim3 block(32, warps);

  dim3 grid((N+warps-1)/warps);

  if(T==1){

    q8_mmvq_t1_kernel<<<grid,block,0,stream>>>(

      reinterpret_cast<const half*>(x.data_ptr<at::Half>()),

      packed.data_ptr<uint8_t>(),

      reinterpret_cast<half*>(y.data_ptr<at::Half>()),

      (int)K,N,rb);

  }else if(T==2){
    const int use_smem=(K<=12288)?1:0;
    size_t smem=use_smem?(size_t)K*2*sizeof(half):0;
    q8_mmvq_t2_kernel<<<grid,block,smem,stream>>>(
      reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
      packed.data_ptr<uint8_t>(),
      reinterpret_cast<half*>(y.data_ptr<at::Half>()),
      (int)K,N,rb,use_smem);

  }else if(T<=4){

    q8_mmvq_t_kernel<4><<<grid,block,0,stream>>>(

      reinterpret_cast<const half*>(x.data_ptr<at::Half>()),

      packed.data_ptr<uint8_t>(),

      reinterpret_cast<half*>(y.data_ptr<at::Half>()),

      T,(int)K,N,rb);

  }else{ TORCH_CHECK(false, "T"); }

  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return y;

}

void q8_mmvq_dual_forward_out(torch::Tensor x, torch::Tensor p0, torch::Tensor p1, int64_t K,
                                  torch::Tensor y0, torch::Tensor y1){
  check_inputs(x,p0,K); check_inputs(x,p1,K);
  const int T=(int)x.size(0);
  TORCH_CHECK(T>=1 && T<=4, "dual MMVQ decode T=1..4");
  const int N0=p0.size(0), N1=p1.size(0);
  TORCH_CHECK(y0.is_cuda() && y1.is_cuda() && y0.scalar_type()==at::kHalf && y1.scalar_type()==at::kHalf, "dual y cuda half");
  TORCH_CHECK(y0.is_contiguous() && y1.is_contiguous(), "dual y contiguous");
  TORCH_CHECK(y0.dim()==2 && y1.dim()==2 && y0.size(0)==T && y1.size(0)==T && y0.size(1)==N0 && y1.size(1)==N1, "dual y [T,N]");
  c10::cuda::CUDAGuard guard(x.device());
  auto stream=at::cuda::getCurrentCUDAStream();
  const int warps=Q8_DUAL_WARPS, rb=(int)p0.size(1);
  auto xh=reinterpret_cast<const half*>(x.data_ptr<at::Half>());
  auto y0h=reinterpret_cast<half*>(y0.data_ptr<at::Half>());
  auto y1h=reinterpret_cast<half*>(y1.data_ptr<at::Half>());
  auto w0=p0.data_ptr<uint8_t>(); auto w1=p1.data_ptr<uint8_t>();
  dim3 block(32,warps);
  int grid=((N0+N1)+warps-1)/warps;
  if(T==1){
    size_t smem=0;
#if Q8_DUAL_SMEMX
    smem=(size_t)K*sizeof(half);
#endif
    q8_mmvq_dual_t1_kernel<<<grid,block,smem,stream>>>(xh,w0,w1,y0h,y1h,(int)K,N0,N1,rb);
  }else if(T==2){
    const int w2=32; dim3 block2(32,w2); int grid2=((N0+N1)+w2-1)/w2;
    const size_t smem2=(size_t)2*K*sizeof(half);
    q8_mmvq_dual_t2_kernel<<<grid2,block2,smem2,stream>>>(xh,w0,w1,y0h,y1h,(int)K,N0,N1,rb);
  }else if(T==3){
    q8_mmvq_dual_t_kernel<3><<<grid,block,0,stream>>>(xh,w0,w1,y0h,y1h,T,(int)K,N0,N1,rb);
  }else{
    q8_mmvq_dual_t_kernel<4><<<grid,block,0,stream>>>(xh,w0,w1,y0h,y1h,T,(int)K,N0,N1,rb);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

torch::Tensor q8_mmvq_id_forward(torch::Tensor x, torch::Tensor packed, torch::Tensor ids, int64_t K){
  // x:[1,K] or [T,K] T=1 only for now; ids:[n] int32; y:[n] or [1,n]
  check_inputs(x,packed,K);
  TORCH_CHECK(x.size(0)==1, "mmvq_id T=1 only");
  TORCH_CHECK(ids.is_cuda() && ids.dtype()==torch::kInt32 && ids.is_contiguous(), "ids");
  const int n=ids.numel(), rb=packed.size(1);
  auto y=torch::empty({1,n}, x.options());
  c10::cuda::CUDAGuard guard(x.device());
  auto stream=at::cuda::getCurrentCUDAStream();
  const int warps=4;
  dim3 block(32,warps);
  dim3 grid((n+warps-1)/warps);
  q8_mmvq_id_t1_kernel<<<grid,block,0,stream>>>(
    reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
    packed.data_ptr<uint8_t>(),
    ids.data_ptr<int32_t>(),
    reinterpret_cast<half*>(y.data_ptr<at::Half>()),
    (int)K,n,rb);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}



torch::Tensor q8_mmvq_rms_forward_out(torch::Tensor x, torch::Tensor norm_w,
    torch::Tensor packed, int64_t K, torch::Tensor y, double eps){
  check_inputs(x,packed,K);
  const int T=x.size(0); TORCH_CHECK(T>=1&&T<=4,"decode fused rms mmvq requires T=1..4");
  TORCH_CHECK(norm_w.is_cuda()&&norm_w.dtype()==torch::kFloat16&&norm_w.numel()==K&&norm_w.is_contiguous(),"norm_w");
  const int N=packed.size(0), rb=packed.size(1);
  TORCH_CHECK(y.is_cuda()&&y.dtype()==x.dtype()&&y.is_contiguous()&&y.size(0)==T&&y.size(1)==N,"y");
  c10::cuda::CUDAGuard guard(x.device());
  auto stream=at::cuda::getCurrentCUDAStream();
  dim3 block(32,4); dim3 grid((N+3)/4);
  const half* xp=reinterpret_cast<const half*>(x.data_ptr<at::Half>());
  half* yp=reinterpret_cast<half*>(y.data_ptr<at::Half>());
  if(T==2){
    q8_mmvq_rms_t2_kernel<<<grid,block,0,stream>>>(
      xp,reinterpret_cast<const half*>(norm_w.data_ptr<at::Half>()),
      packed.data_ptr<uint8_t>(),yp,(int)K,N,rb,(float)eps);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return y;
  }
  for(int t=0;t<T;++t){
    q8_mmvq_rms_t1_kernel<<<grid,block,0,stream>>>(
      xp+(size_t)t*K,reinterpret_cast<const half*>(norm_w.data_ptr<at::Half>()),
      packed.data_ptr<uint8_t>(),yp+(size_t)t*N,(int)K,N,rb,(float)eps);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return y;
}

// Multi-head Q8_0 matvec: x[1,H,K] @ packed[H,N,rb] -> y[1,H,N] (one launch)
__global__ void q8_mmvq_grouped_t1_kernel(
    const half* __restrict__ x,
    const uint8_t* __restrict__ packed,
    half* __restrict__ y,
    int H, int N, int K, int rb,
    int64_t x_h_stride, int64_t y_h_stride){
  const int warps=blockDim.y;
  const int lane=threadIdx.x;
  const int local_row=threadIdx.y;
  const int head=blockIdx.y;
  if(head>=H) return;
  const int row=blockIdx.x*warps+local_row;
  if(row>=N) return;
  const half* xh=x+(int64_t)head*x_h_stride;
  const uint8_t* rowp=packed+((int64_t)head*N+row)*(int64_t)rb;
  float acc=0.f;
  const int nb=K>>5;
  for(int b=0;b<nb;++b){
    const uint8_t* p=rowp+(size_t)b*34;
    float d=__half2float(*reinterpret_cast<const half*>(p));
    float q=(float)reinterpret_cast<const int8_t*>(p+2)[lane];
    int k=(b<<5)+lane;
    acc+=d*q*__half2float(xh[k]);
  }
  #pragma unroll
  for(int d=16;d;d>>=1) acc+=__shfl_down_sync(0xffffffff,acc,d);
  if(lane==0) y[(int64_t)head*y_h_stride+row]=__float2half(acc);
}


// T=4 grouped MMVQ in one launch. grid.z selects the token; arithmetic stays
// identical to grouped_t1 to preserve bit-exact decode results.
__global__ void q8_mmvq_grouped_t4_kernel(
    const half* __restrict__ x,
    const uint8_t* __restrict__ packed,
    half* __restrict__ y,
    int H, int N, int K, int rb,
    int64_t x_t_stride, int64_t x_h_stride,
    int64_t y_t_stride, int64_t y_h_stride) {
  const int warps=blockDim.y;
  const int lane=threadIdx.x;
  const int local_row=threadIdx.y;
  const int head=blockIdx.y;
  const int t=blockIdx.z;
  if(head>=H) return;
  const int row=blockIdx.x*warps+local_row;
  if(row>=N) return;
  const half* xh=x+(int64_t)t*x_t_stride+(int64_t)head*x_h_stride;
  const uint8_t* rowp=packed+((int64_t)head*N+row)*(int64_t)rb;
  float acc=0.f;
  const int nb=K>>5;
  for(int b=0;b<nb;++b){
    const uint8_t* qblock=rowp+(size_t)b*34;
    float d=__half2float(*reinterpret_cast<const half*>(qblock));
    float q=(float)reinterpret_cast<const int8_t*>(qblock+2)[lane];
    int k=(b<<5)+lane;
    acc+=d*q*__half2float(xh[k]);
  }
  #pragma unroll
  for(int d=16;d;d>>=1) acc+=__shfl_down_sync(0xffffffff,acc,d);
  if(lane==0) y[(int64_t)t*y_t_stride+(int64_t)head*y_h_stride+row]=__float2half(acc);
}

// Decode grouped Q8: only small T=1..4. Large T belongs to prefill.
void q8_mmvq_grouped_forward_out(torch::Tensor x, torch::Tensor packed, int64_t K, torch::Tensor y){
  TORCH_CHECK(x.is_cuda()&&packed.is_cuda()&&y.is_cuda());
  TORCH_CHECK(x.scalar_type()==torch::kFloat16&&y.scalar_type()==torch::kFloat16);
  TORCH_CHECK(packed.scalar_type()==torch::kUInt8 && x.dim()==3);
  const int64_t T=x.size(0);
  TORCH_CHECK(T>=1&&T<=4,"decode grouped MMVQ requires T=1..4");
  int64_t H=packed.size(0), N=packed.size(1), rb=packed.size(2);
  TORCH_CHECK(x.size(1)==H&&x.size(2)==K);
  TORCH_CHECK(y.dim()==3&&y.size(0)==T&&y.size(1)==H&&y.size(2)==N);
  TORCH_CHECK((K%32)==0 && rb==(K/32)*34);
  c10::cuda::CUDAGuard guard(x.device());
  auto stream=at::cuda::getCurrentCUDAStream();
  const int warps=4;
  dim3 block(32,warps), grid((int)((N+warps-1)/warps),(int)H);
  auto xst=x.strides(); TORCH_CHECK(xst[2]==1,"x last dim contiguous");
  auto yst=y.strides(); TORCH_CHECK(yst[2]==1&&y.is_contiguous());
  if(T==4){
    dim3 grid4((int)((N+warps-1)/warps),(int)H,4);
    q8_mmvq_grouped_t4_kernel<<<grid4,block,0,stream>>>(
      reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
      packed.data_ptr<uint8_t>(),
      reinterpret_cast<half*>(y.data_ptr<at::Half>()),
      (int)H,(int)N,(int)K,(int)rb,xst[0],xst[1],yst[0],yst[1]);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return;
  }
  if(T==2){
    q8_mmvq_grouped_t2_kernel<<<grid,block,0,stream>>>(
      reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
      packed.data_ptr<uint8_t>(),
      reinterpret_cast<half*>(y.data_ptr<at::Half>()),
      (int)H,(int)N,(int)K,(int)rb,xst[0],xst[1],yst[0],yst[1]);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return;
  }
  for(int64_t t=0;t<T;++t){
    const half* xp=reinterpret_cast<const half*>(x.data_ptr<at::Half>())+t*xst[0];
    half* yp=reinterpret_cast<half*>(y.data_ptr<at::Half>())+t*yst[0];
    q8_mmvq_grouped_t1_kernel<<<grid,block,0,stream>>>(
      xp,packed.data_ptr<uint8_t>(),yp,
      (int)H,(int)N,(int)K,(int)rb,xst[1],yst[1]);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
}

