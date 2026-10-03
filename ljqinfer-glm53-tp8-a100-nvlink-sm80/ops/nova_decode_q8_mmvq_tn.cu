// q8_mmvq.cu - decode-specialized GGML Q8_0 matvec (T=1..6)
// Layout: packed row = nb blocks * 34 bytes; block = half d + 32 int8 qs
// Goal: beat naive q8_linear and cublas dequant path on T=1.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

#ifndef Q8_DUAL_WARPS
#define Q8_DUAL_WARPS 32
#endif
#ifndef Q8_DUAL_SMEMX
#define Q8_DUAL_SMEMX 1
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
  x += (size_t)blockIdx.z * K;
  y += (size_t)blockIdx.z * N;
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
  x += (size_t)blockIdx.z * K;
  y0 += (size_t)blockIdx.z * N0;
  y1 += (size_t)blockIdx.z * N1;
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
    float d=__half2float(__ldg(reinterpret_cast<const half*>(p)));
    float q=(float)__ldg(reinterpret_cast<const int8_t*>(p+2)+lane);
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
  extern __shared__ half xs[];
  {
    const int tid=threadIdx.y*32+threadIdx.x;
    const int nthreads=blockDim.y*32;
    const int n=T*K;
    for(int i=tid;i<n;i+=nthreads) xs[i]=x[i];
    __syncthreads();
  }
  const int idx=blockIdx.x*blockDim.y+threadIdx.y;
  if(idx>=N0+N1) return;
  const int lane=threadIdx.x;
  const bool second=idx>=N0;
  const int row=second ? idx-N0 : idx;
  const uint8_t* rowp=(second?w1:w0)+(size_t)row*rb;
  half* y=second?y1:y0;
  const int nb=K>>5;
  float sum[TMAX];
#pragma unroll
  for(int t=0;t<TMAX;++t) sum[t]=0.f;
  for(int b=0;b<nb;++b){
    const uint8_t* p=rowp+(size_t)b*34;
    float d=__half2float(__ldg(reinterpret_cast<const half*>(p)));
    float q=(float)__ldg(reinterpret_cast<const int8_t*>(p+2)+lane);
    float wq=d*q;
    const int k0=b<<5;
#pragma unroll
    for(int t=0;t<TMAX;++t){
      if(t<T){
        float xv=__half2float(xs[(size_t)t*K+k0+lane]);
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
  // Stage all T x-rows once; each warp reuses xs for its output row.
  extern __shared__ half xs_plain[];
  {
    const int tid=threadIdx.y*32+threadIdx.x;
    const int nthreads=blockDim.y*32;
    const int n=T*K;
    for(int i=tid;i<n;i+=nthreads) xs_plain[i]=x[i];
    __syncthreads();
  }
  const int row=blockIdx.x*blockDim.y+threadIdx.y;
  if(row>=N) return;
  const int lane=threadIdx.x;
  const uint8_t* rowp=w+(size_t)row*rb;
  const int nb=K>>5;
  float sum[TMAX];
#pragma unroll
  for(int t=0;t<TMAX;++t) sum[t]=0.f;
  for(int b=0;b<nb;++b){
    const uint8_t* p=rowp+(size_t)b*34;
    float d=__half2float(__ldg(reinterpret_cast<const half*>(p)));
    float q=(float)__ldg(reinterpret_cast<const int8_t*>(p+2)+lane);
    float wq=d*q;
    const int k0=b<<5;
#pragma unroll
    for(int t=0;t<TMAX;++t){
      if(t<T){
        float xv=__half2float(xs_plain[(size_t)t*K+k0+lane]);
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

template<int TMAX>
__global__ void q8_mmvq_t_strided_batch_kernel(
    const half* __restrict__ x, const uint8_t* __restrict__ w,
    half* __restrict__ y, int T, int K, int N, int rb,
    int64_t x_request_stride, int64_t y_request_stride) {
  x += (int64_t)blockIdx.z * x_request_stride;
  y += (int64_t)blockIdx.z * y_request_stride;
  extern __shared__ half xs_plain[];
  {
    const int tid=threadIdx.y*32+threadIdx.x;
    const int nthreads=blockDim.y*32;
    const int n=T*K;
    for(int i=tid;i<n;i+=nthreads) xs_plain[i]=x[i];
    __syncthreads();
  }
  const int row=blockIdx.x*blockDim.y+threadIdx.y;
  if(row>=N) return;
  const int lane=threadIdx.x;
  const uint8_t* rowp=w+(size_t)row*rb;
  const int nb=K>>5;
  float sum[TMAX];
#pragma unroll
  for(int t=0;t<TMAX;++t) sum[t]=0.f;
  for(int b=0;b<nb;++b){
    const uint8_t* p=rowp+(size_t)b*34;
    float d=__half2float(__ldg(reinterpret_cast<const half*>(p)));
    float q=(float)__ldg(reinterpret_cast<const int8_t*>(p+2)+lane);
    float wq=d*q;
    const int k0=b<<5;
#pragma unroll
    for(int t=0;t<TMAX;++t){
      if(t<T){
        float xv=__half2float(xs_plain[(size_t)t*K+k0+lane]);
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
  x += (size_t)blockIdx.z * K;
  y += (size_t)blockIdx.z * N;
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
  x += (size_t)blockIdx.z * 2 * K;
  y += (size_t)blockIdx.z * 2 * N;
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

__global__ void q8_mmvq_t2_strided_batch_kernel(
    const half* __restrict__ x, const uint8_t* __restrict__ w,
    half* __restrict__ y, int K, int N, int rb, int use_smem,
    int64_t x_request_stride, int64_t y_request_stride) {
  x += (int64_t)blockIdx.z * x_request_stride;
  y += (int64_t)blockIdx.z * y_request_stride;
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
  sum0 = warp_sum32(sum0); sum1 = warp_sum32(sum1);
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
  x += (size_t)blockIdx.z * 2 * K;
  y0 += (size_t)blockIdx.z * 2 * N0;
  y1 += (size_t)blockIdx.z * 2 * N1;
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


// Runtime-T RMS + Q8_0 small-M matrix path. T is the matrix M dimension;
// TOKEN_TILE is only an internal row tile, not a T-specific operator ABI.
template<int TOKEN_TILE, int OUT_TILE>
__global__ void q8_mmvq_rms_tn_matrix_kernel(
    const half* __restrict__ x,
    const half* __restrict__ nw,
    const uint8_t* __restrict__ w,
    half* __restrict__ y,
    int K, int N, int rb, int T, float eps) {
  // Each token warp owns one M row and the same OUT_TILE output rows. Weight
  // addresses match across token warps, so caches provide reuse without CTA
  // barriers. Four local RMS streams reproduce the old 128-thread reduction.
  const int lane = threadIdx.x;
  const int token_lane = threadIdx.y;
  const int t = blockIdx.y * TOKEN_TILE + token_lane;
  const bool valid_t = t < T;
  const half* xt = x + (size_t)(valid_t ? t : 0) * K;

  float ss0 = 0.f, ss1 = 0.f, ss2 = 0.f, ss3 = 0.f;
  if (valid_t) {
    for (int k = lane; k < K; k += 128) {
      float v = __half2float(xt[k]);
      ss0 = fmaf(v, v, ss0);
    }
    for (int k = 32 + lane; k < K; k += 128) {
      float v = __half2float(xt[k]);
      ss1 = fmaf(v, v, ss1);
    }
    for (int k = 64 + lane; k < K; k += 128) {
      float v = __half2float(xt[k]);
      ss2 = fmaf(v, v, ss2);
    }
    for (int k = 96 + lane; k < K; k += 128) {
      float v = __half2float(xt[k]);
      ss3 = fmaf(v, v, ss3);
    }
  }
  ss0 = warp_sum32(ss0);
  ss1 = warp_sum32(ss1);
  ss2 = warp_sum32(ss2);
  ss3 = warp_sum32(ss3);
  float scale = 0.f;
  if (lane == 0)
    scale = rsqrtf((ss0 + ss1 + ss2 + ss3) / (float)K + eps);
  scale = __shfl_sync(0xffffffff, scale, 0);

  const int row0 = blockIdx.x * OUT_TILE;
  float acc[OUT_TILE];
#pragma unroll
  for (int r = 0; r < OUT_TILE; ++r) acc[r] = 0.f;
  for (int b = 0; b < (K >> 5); ++b) {
    const int k = (b << 5) + lane;
    const float xv = valid_t
        ? __half2float(xt[k]) * __half2float(nw[k])
        : 0.f;
#pragma unroll
    for (int r = 0; r < OUT_TILE; ++r) {
      const int row = row0 + r;
      if (row < N) {
        const uint8_t* qblock = w + (size_t)row * rb + (size_t)b * 34;
        const float d = __half2float(*reinterpret_cast<const half*>(qblock));
        const float q = (float)reinterpret_cast<const int8_t*>(qblock + 2)[lane];
        acc[r] = fmaf(xv, d * q, acc[r]);
      }
    }
  }
#pragma unroll
  for (int r = 0; r < OUT_TILE; ++r) acc[r] = warp_sum32(acc[r]);
  if (valid_t && lane == 0) {
#pragma unroll
    for (int r = 0; r < OUT_TILE; ++r) {
      const int row = row0 + r;
      if (row < N)
        y[(size_t)t * N + row] = __float2half_rn(acc[r] * scale);
    }
  }
}

template<int TMAX>
__global__ void q8_mmvq_rms_tn_weightreuse_kernel(
    const half* __restrict__ x,
    const half* __restrict__ nw,
    const uint8_t* __restrict__ w,
    half* __restrict__ y,
    int K, int N, int rb, int T, float eps) {
  extern __shared__ half sx[];  // T*K half
  __shared__ float scales[TMAX];
  const int lane = threadIdx.x;
  const int warp = threadIdx.y;
  const int tid = warp * 32 + lane;
  const int nthreads = blockDim.y * 32;

  for (int i = tid; i < T * K; i += nthreads) sx[i] = x[i];
  __syncthreads();

  if (warp < T) {
    const half* xt = sx + (size_t)warp * K;
    float ss0 = 0.f, ss1 = 0.f, ss2 = 0.f, ss3 = 0.f;
    for (int k = lane; k < K; k += 128) {
      float v = __half2float(xt[k]);
      ss0 = fmaf(v, v, ss0);
    }
    for (int k = 32 + lane; k < K; k += 128) {
      float v = __half2float(xt[k]);
      ss1 = fmaf(v, v, ss1);
    }
    for (int k = 64 + lane; k < K; k += 128) {
      float v = __half2float(xt[k]);
      ss2 = fmaf(v, v, ss2);
    }
    for (int k = 96 + lane; k < K; k += 128) {
      float v = __half2float(xt[k]);
      ss3 = fmaf(v, v, ss3);
    }
    ss0 = warp_sum32(ss0);
    ss1 = warp_sum32(ss1);
    ss2 = warp_sum32(ss2);
    ss3 = warp_sum32(ss3);
    if (lane == 0)
      scales[warp] = rsqrtf((ss0 + ss1 + ss2 + ss3) / (float)K + eps);
  }
  __syncthreads();

  const int row = blockIdx.x * blockDim.y + warp;
  if (row >= N) return;
  const uint8_t* rowp = w + (size_t)row * rb;
  float acc[TMAX];
#pragma unroll
  for (int t = 0; t < TMAX; ++t) acc[t] = 0.f;

  for (int b = 0; b < (K >> 5); ++b) {
    const uint8_t* p = rowp + (size_t)b * 34;
    const float d = __half2float(__ldg(reinterpret_cast<const half*>(p)));
    const float q = (float)__ldg(reinterpret_cast<const int8_t*>(p + 2) + lane);
    const float wq = d * q;
    const int k = (b << 5) + lane;
    const float nwv = __half2float(nw[k]);
#pragma unroll
    for (int t = 0; t < TMAX; ++t) {
      if (t < T) {
        const float xv = __half2float(sx[(size_t)t * K + k]) * nwv;
        acc[t] = fmaf(xv, wq, acc[t]);
      }
    }
  }
#pragma unroll
  for (int t = 0; t < TMAX; ++t) {
    if (t < T) {
      float v = warp_sum32(acc[t]);
      if (lane == 0)
        y[(size_t)t * N + row] = __float2half_rn(v * scales[t]);
    }
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
  x += (int64_t)blockIdx.z * 2 * x_t_stride;
  y += (int64_t)blockIdx.z * 2 * y_t_stride;
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

torch::Tensor q8_mmvq_forward_out(torch::Tensor x, torch::Tensor packed,
    int64_t K, torch::Tensor y);
void q8_mmvq_grouped_forward_out(torch::Tensor x, torch::Tensor packed,
    int64_t K, torch::Tensor y);

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
  }else if(T<=6){
    // Native T=5/6 single launch — avoid Q4+Q2 double weight scan.
    q8_mmvq_t_kernel<6><<<grid,block,0,stream>>>(
      reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
      packed.data_ptr<uint8_t>(),
      reinterpret_cast<half*>(y.data_ptr<at::Half>()),
      T,(int)K,N,rb);
  }else{
    TORCH_CHECK(false, "q8_mmvq supports T<=6");
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}

void q8_mmvq_batch_forward_out(torch::Tensor x, torch::Tensor packed, int64_t K,
                                  torch::Tensor y, int64_t B, int64_t Q) {
  check_inputs(x, packed, K);
  TORCH_CHECK(B >= 1 && Q >= 1 && Q <= 6 && x.size(0) == B * Q,
              "batch MMVQ supports BnQ1..6");
  // For B3/B4 Q6, keep each request's legacy Q4+Q2 arithmetic while
  // collapsing 2*B launches into two request-strided grid.z launches.
  if (B >= 3 && Q == 6) {
    const int N = packed.size(0), rb = packed.size(1);
    TORCH_CHECK(y.is_cuda() && y.dtype() == x.dtype() && y.is_contiguous() &&
                y.dim() == 2 && y.size(0) == B * Q && y.size(1) == N,
                "batch MMVQ y shape");
    c10::cuda::CUDAGuard guard(x.device());
    auto stream = at::cuda::getCurrentCUDAStream();
    constexpr int warps = 16;
    dim3 block(32, warps), grid((N + warps - 1) / warps, 1, (unsigned)B);
    auto xp = reinterpret_cast<const half*>(x.data_ptr<at::Half>());
    auto yp = reinterpret_cast<half*>(y.data_ptr<at::Half>());
    q8_mmvq_t_strided_batch_kernel<4><<<grid, block,
        (size_t)4*K*sizeof(half), stream>>>(
        xp, packed.data_ptr<uint8_t>(), yp, 4, (int)K, N, rb, Q*K, Q*N);
    const int use_smem = (K <= 12288) ? 1 : 0;
    const size_t smem = use_smem ? (size_t)K * 2 * sizeof(half) : 0;
    q8_mmvq_t2_strided_batch_kernel<<<grid, block, smem, stream>>>(
        xp + 4*K, packed.data_ptr<uint8_t>(), yp + 4*N,
        (int)K, N, rb, use_smem, Q*K, Q*N);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return;
  }
  if (B > 1 && Q > 2) {
    const int64_t Tflat = B * Q;
    // Flat launch disabled: E2E profile shows t_kernel<12> slower than the
    // per-request 4+2 loop (85.6 vs 72.0 ms/round aggregate) despite winning
    // in isolated micro-benchmarks. Keep the loop below.
    if (false && Tflat <= 12) {
      // Flat path: one launch over all B*Q rows so each weight block is read
      // once per CTA instead of once per request. Per-token arithmetic matches
      // the legacy leaves (same per-block mul/fma order + warp reduction).
      const int N = packed.size(0), rb = packed.size(1);
      TORCH_CHECK(y.is_cuda() && y.dtype() == x.dtype() && y.is_contiguous() &&
                  y.dim() == 2 && y.size(0) == Tflat && y.size(1) == N,
                  "batch MMVQ y shape");
      c10::cuda::CUDAGuard guard(x.device());
      auto stream = at::cuda::getCurrentCUDAStream();
      constexpr int warps = 4;
      dim3 block(32, warps), grid((N + warps - 1) / warps);
      q8_mmvq_t_kernel<12><<<grid, block, 0, stream>>>(
          reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
          packed.data_ptr<uint8_t>(),
          reinterpret_cast<half*>(y.data_ptr<at::Half>()),
          (int)Tflat, (int)K, N, rb);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
      return;
    }
    // Keep each request independent; the legacy leaves accept at most four rows.
    for (int64_t b = 0; b < B; ++b) {
      auto xb = x.narrow(0, b * Q, Q);
      auto yb = y.narrow(0, b * Q, Q);
      const int64_t first = Q >= 4 ? 4 : Q;
      q8_mmvq_forward_out(xb.narrow(0, 0, first), packed, K,
                          yb.narrow(0, 0, first));
      if (Q > first)
        q8_mmvq_forward_out(xb.narrow(0, first, Q - first), packed, K,
                            yb.narrow(0, first, Q - first));
    }
    return;
  }
  if (B == 1 && Q > 2) {
    // Native T=Q single launch (T<=6).
    q8_mmvq_forward_out(x, packed, K, y);
    return;
  }
  const int N = packed.size(0), rb = packed.size(1);
  TORCH_CHECK(y.is_cuda() && y.dtype() == x.dtype() && y.is_contiguous() &&
              y.dim() == 2 && y.size(0) == B * Q && y.size(1) == N,
              "batch MMVQ y shape");
  c10::cuda::CUDAGuard guard(x.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  constexpr int warps = 4;
  dim3 block(32, warps), grid((N + warps - 1) / warps, 1, (unsigned)B);
  auto xp = reinterpret_cast<const half*>(x.data_ptr<at::Half>());
  auto yp = reinterpret_cast<half*>(y.data_ptr<at::Half>());
  if (Q == 1) {
    q8_mmvq_t1_kernel<<<grid, block, 0, stream>>>(xp, packed.data_ptr<uint8_t>(),
                                                  yp, (int)K, N, rb);
  } else {
    const int use_smem = (K <= 12288) ? 1 : 0;
    const size_t smem = use_smem ? (size_t)K * 2 * sizeof(half) : 0;
    q8_mmvq_t2_kernel<<<grid, block, smem, stream>>>(xp, packed.data_ptr<uint8_t>(),
                                                     yp, (int)K, N, rb, use_smem);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

torch::Tensor q8_mmvq_forward_out(torch::Tensor x, torch::Tensor packed, int64_t K, torch::Tensor y){

  check_inputs(x,packed,K);

  const int T=x.size(0), N=packed.size(0), rb=packed.size(1);

  TORCH_CHECK(y.is_cuda() && y.dtype()==x.dtype() && y.is_contiguous(), "y");

  TORCH_CHECK(y.size(0)==T && y.size(1)==N, "y shape");

  c10::cuda::CUDAGuard guard(x.device());

  auto stream=at::cuda::getCurrentCUDAStream();

  const int warps=(N>=1024)?16:((N>=256)?8:4);

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
    const size_t smem4=(size_t)T*K*sizeof(half);
    q8_mmvq_t_kernel<4><<<grid,block,smem4,stream>>>(
      reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
      packed.data_ptr<uint8_t>(),
      reinterpret_cast<half*>(y.data_ptr<at::Half>()),
      T,(int)K,N,rb);
  }else if(T<=6){
    const size_t smem6=(size_t)T*K*sizeof(half);
    q8_mmvq_t_kernel<6><<<grid,block,smem6,stream>>>(
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
  TORCH_CHECK(T>=1 && T<=6, "dual MMVQ decode T=1..6");
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
    const size_t smem=(size_t)T*K*sizeof(half);
    q8_mmvq_dual_t_kernel<3><<<grid,block,smem,stream>>>(xh,w0,w1,y0h,y1h,T,(int)K,N0,N1,rb);
  }else if(T==4){
    const size_t smem=(size_t)T*K*sizeof(half);
    q8_mmvq_dual_t_kernel<4><<<grid,block,smem,stream>>>(xh,w0,w1,y0h,y1h,T,(int)K,N0,N1,rb);
  }else{
    const size_t smem=(size_t)T*K*sizeof(half);
    q8_mmvq_dual_t_kernel<6><<<grid,block,smem,stream>>>(xh,w0,w1,y0h,y1h,T,(int)K,N0,N1,rb);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void q8_mmvq_batch_dual_forward_out(torch::Tensor x, torch::Tensor p0,
    torch::Tensor p1, int64_t K, torch::Tensor y0, torch::Tensor y1,
    int64_t B, int64_t Q) {
  check_inputs(x, p0, K); check_inputs(x, p1, K);
  TORCH_CHECK(B >= 1 && Q >= 1 && Q <= 6 && x.size(0) == B * Q,
              "batch dual MMVQ supports BnQ1..6");
  if (B > 1 && Q > 2) {
    const int64_t Tflat = B * Q;
    // Flat launch disabled: production dual calls run at Q<=1 (never
    // triggers) and micro-benchmark shows parity only (187us vs 188us).
    if (false && Tflat <= 12) {
      // Flat path: one launch over all B*Q rows across both weight matrices.
      const int N0 = p0.size(0), N1 = p1.size(0), rb = p0.size(1);
      TORCH_CHECK(y0.is_cuda() && y1.is_cuda() && y0.scalar_type() == at::kHalf &&
                  y1.scalar_type() == at::kHalf && y0.is_contiguous() &&
                  y1.is_contiguous() && y0.dim() == 2 && y1.dim() == 2 &&
                  y0.size(0) == Tflat && y1.size(0) == Tflat &&
                  y0.size(1) == N0 && y1.size(1) == N1, "flat dual y shape");
      c10::cuda::CUDAGuard guard(x.device());
      auto stream = at::cuda::getCurrentCUDAStream();
      constexpr int warps = 4;
      dim3 block(32, warps), grid(((N0 + N1) + warps - 1) / warps);
      q8_mmvq_dual_t_kernel<12><<<grid, block, 0, stream>>>(
          reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
          p0.data_ptr<uint8_t>(), p1.data_ptr<uint8_t>(),
          reinterpret_cast<half*>(y0.data_ptr<at::Half>()),
          reinterpret_cast<half*>(y1.data_ptr<at::Half>()),
          (int)Tflat, (int)K, N0, N1, rb);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
      return;
    }
    for (int64_t b = 0; b < B; ++b) {
      auto xb = x.narrow(0, b * Q, Q);
      auto y0b = y0.narrow(0, b * Q, Q);
      auto y1b = y1.narrow(0, b * Q, Q);
      const int64_t first = Q >= 4 ? 4 : Q;
      q8_mmvq_dual_forward_out(xb.narrow(0, 0, first), p0, p1, K,
                               y0b.narrow(0, 0, first),
                               y1b.narrow(0, 0, first));
      if (Q > first)
        q8_mmvq_dual_forward_out(xb.narrow(0, first, Q - first), p0, p1, K,
                                 y0b.narrow(0, first, Q - first),
                                 y1b.narrow(0, first, Q - first));
    }
    return;
  }
  if (B == 1 && Q > 2) {
    // Native T=6 dual was ~1.25x slower; keep Q4+Q2.
    const int64_t first = Q >= 4 ? 4 : Q;
    q8_mmvq_dual_forward_out(x.narrow(0, 0, first), p0, p1, K,
                              y0.narrow(0, 0, first),
                              y1.narrow(0, 0, first));
    if (Q > first)
      q8_mmvq_dual_forward_out(x.narrow(0, first, Q - first), p0, p1, K,
                                y0.narrow(0, first, Q - first),
                                y1.narrow(0, first, Q - first));
    return;
  }
  const int N0 = p0.size(0), N1 = p1.size(0), rb = p0.size(1);
  TORCH_CHECK(y0.is_cuda() && y1.is_cuda() && y0.scalar_type() == at::kHalf &&
              y1.scalar_type() == at::kHalf && y0.is_contiguous() && y1.is_contiguous(),
              "batch dual y cuda half contiguous");
  TORCH_CHECK(y0.dim() == 2 && y1.dim() == 2 && y0.size(0) == B * Q &&
              y1.size(0) == B * Q && y0.size(1) == N0 && y1.size(1) == N1,
              "batch dual y shape");
  c10::cuda::CUDAGuard guard(x.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  auto xp = reinterpret_cast<const half*>(x.data_ptr<at::Half>());
  auto y0p = reinterpret_cast<half*>(y0.data_ptr<at::Half>());
  auto y1p = reinterpret_cast<half*>(y1.data_ptr<at::Half>());
  auto w0 = p0.data_ptr<uint8_t>(); auto w1 = p1.data_ptr<uint8_t>();
  if (Q == 1) {
    const int warps = Q8_DUAL_WARPS;
    dim3 block(32, warps), grid(((N0 + N1) + warps - 1) / warps, 1, (unsigned)B);
    size_t smem = 0;
#if Q8_DUAL_SMEMX
    smem = (size_t)K * sizeof(half);
#endif
    q8_mmvq_dual_t1_kernel<<<grid, block, smem, stream>>>(
        xp, w0, w1, y0p, y1p, (int)K, N0, N1, rb);
  } else {
    constexpr int warps = 32;
    dim3 block(32, warps), grid(((N0 + N1) + warps - 1) / warps, 1, (unsigned)B);
    const size_t smem = (size_t)2 * K * sizeof(half);
    q8_mmvq_dual_t2_kernel<<<grid, block, smem, stream>>>(
        xp, w0, w1, y0p, y1p, (int)K, N0, N1, rb);
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
  const int T=x.size(0);
  TORCH_CHECK(T>=1&&T<=32,"decode rms+q8 TN requires T=1..32");
  TORCH_CHECK(norm_w.is_cuda()&&norm_w.dtype()==torch::kFloat16&&norm_w.numel()==K&&norm_w.is_contiguous(),"norm_w");
  const int N=packed.size(0), rb=packed.size(1);
  TORCH_CHECK(y.is_cuda()&&y.dtype()==x.dtype()&&y.is_contiguous()&&y.size(0)==T&&y.size(1)==N,"y");
  c10::cuda::CUDAGuard guard(x.device());
  auto stream=at::cuda::getCurrentCUDAStream();
  const half* xp=reinterpret_cast<const half*>(x.data_ptr<at::Half>());
  half* yp=reinterpret_cast<half*>(y.data_ptr<at::Half>());
  if(T==1){
    dim3 block(32,4), grid((N+3)/4);
    q8_mmvq_rms_t1_kernel<<<grid,block,0,stream>>>(
        xp,reinterpret_cast<const half*>(norm_w.data_ptr<at::Half>()),
        packed.data_ptr<uint8_t>(),yp,(int)K,N,rb,(float)eps);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return y;
  }
  constexpr int TOKEN_TILE=8, OUT_TILE=4;
  const int token_warps = T < TOKEN_TILE ? T : TOKEN_TILE;
  dim3 block(32,token_warps);
  dim3 grid((N+OUT_TILE-1)/OUT_TILE,(T+token_warps-1)/token_warps);
  q8_mmvq_rms_tn_matrix_kernel<TOKEN_TILE,OUT_TILE><<<grid,block,0,stream>>>(
      xp,reinterpret_cast<const half*>(norm_w.data_ptr<at::Half>()),
      packed.data_ptr<uint8_t>(),yp,(int)K,N,rb,T,(float)eps);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}

void q8_mmvq_batch_rms_forward_out(torch::Tensor x, torch::Tensor norm_w,
    torch::Tensor packed, int64_t K, torch::Tensor y, double eps,
    int64_t B, int64_t Q) {
  check_inputs(x, packed, K);
  TORCH_CHECK(B >= 1 && Q >= 1 && Q <= 6 && x.size(0) == B * Q,
              "batch RMS MMVQ supports BnQ1..6");
  if (B > 1 && Q > 2) {
    const int64_t Tflat = B * Q;
    if (Tflat <= 12) {
      // Flat path: all B*Q token warps live in one CTA slice so weight blocks
      // are fetched once and served to every request via L1/L2 reuse. Each
      // token warp keeps the exact leaf arithmetic (own RMS scale, same
      // fma/warp_sum32 order), so per-row results stay bit-identical.
      TORCH_CHECK(norm_w.is_cuda() && norm_w.dtype() == torch::kFloat16 &&
                  norm_w.numel() == K && norm_w.is_contiguous(), "flat RMS norm_w");
      const int N = packed.size(0), rb = packed.size(1);
      TORCH_CHECK(y.is_cuda() && y.dtype() == x.dtype() && y.is_contiguous() &&
                  y.size(0) == Tflat && y.size(1) == N, "flat RMS y shape");
      c10::cuda::CUDAGuard guard(x.device());
      auto stream = at::cuda::getCurrentCUDAStream();
      constexpr int TOKEN_TILE = 12, OUT_TILE = 4;
      dim3 block(32, (int)Tflat);
      dim3 grid((N + OUT_TILE - 1) / OUT_TILE, 1);
      q8_mmvq_rms_tn_matrix_kernel<TOKEN_TILE, OUT_TILE><<<grid, block, 0, stream>>>(
          reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
          reinterpret_cast<const half*>(norm_w.data_ptr<at::Half>()),
          packed.data_ptr<uint8_t>(),
          reinterpret_cast<half*>(y.data_ptr<at::Half>()),
          (int)K, N, rb, (int)Tflat, (float)eps);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
      return;
    }
    for (int64_t b = 0; b < B; ++b) {
      auto xb = x.narrow(0, b * Q, Q);
      auto yb = y.narrow(0, b * Q, Q);
      q8_mmvq_rms_forward_out(xb, norm_w, packed, K, yb, eps);
    }
    return;
  }
  if (B == 1 && Q > 2) {
    TORCH_CHECK(norm_w.is_cuda() && norm_w.dtype() == torch::kFloat16 &&
                norm_w.numel() == K && norm_w.is_contiguous(), "batch RMS norm_w");
    const int N = packed.size(0), rb = packed.size(1), T = x.size(0);
    c10::cuda::CUDAGuard guard(x.device());
    auto stream = at::cuda::getCurrentCUDAStream();
    constexpr int WARPS = 8;
    dim3 block(32, WARPS), grid((N + WARPS - 1) / WARPS);
    const size_t smem = (size_t)T * K * sizeof(half);
    q8_mmvq_rms_tn_weightreuse_kernel<6><<<grid, block, smem, stream>>>(
        reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(norm_w.data_ptr<at::Half>()),
        packed.data_ptr<uint8_t>(),
        reinterpret_cast<half*>(y.data_ptr<at::Half>()),
        (int)K, N, rb, T, (float)eps);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return;
  }
  TORCH_CHECK(norm_w.is_cuda() && norm_w.dtype() == torch::kFloat16 &&
              norm_w.numel() == K && norm_w.is_contiguous(), "batch RMS norm_w");
  const int N = packed.size(0), rb = packed.size(1), T = x.size(0);
  TORCH_CHECK(y.is_cuda() && y.dtype() == x.dtype() && y.is_contiguous() &&
              y.size(0) == T && y.size(1) == N, "batch RMS y shape");
  c10::cuda::CUDAGuard guard(x.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  auto xp = reinterpret_cast<const half*>(x.data_ptr<at::Half>());
  auto np = reinterpret_cast<const half*>(norm_w.data_ptr<at::Half>());
  auto yp = reinterpret_cast<half*>(y.data_ptr<at::Half>());
  if (Q == 1) {
    dim3 block(32, 4), grid((N + 3) / 4, 1, (unsigned)B);
    q8_mmvq_rms_t1_kernel<<<grid, block, 0, stream>>>(
        xp, np, packed.data_ptr<uint8_t>(), yp, (int)K, N, rb, (float)eps);
  } else {
    constexpr int TOKEN_TILE = 8, OUT_TILE = 4;
    const int token_warps = T < TOKEN_TILE ? T : TOKEN_TILE;
    dim3 block(32, token_warps);
    dim3 grid((N + OUT_TILE - 1) / OUT_TILE,
              (T + token_warps - 1) / token_warps);
    q8_mmvq_rms_tn_matrix_kernel<TOKEN_TILE, OUT_TILE><<<grid, block, 0, stream>>>(
        xp, np, packed.data_ptr<uint8_t>(), yp, (int)K, N, rb, T, (float)eps);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Multi-head Q8_0 matvec: x[1,H,K] @ packed[H,N,rb] -> y[1,H,N] (one launch)
__global__ void q8_mmvq_grouped_t1_kernel(
    const half* __restrict__ x,
    const uint8_t* __restrict__ packed,
    half* __restrict__ y,
    int H, int N, int K, int rb,
    int64_t x_h_stride, int64_t y_h_stride){
  x += (int64_t)blockIdx.z * H * x_h_stride;
  y += (int64_t)blockIdx.z * H * y_h_stride;
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

// Flat T<=TMAX grouped MMVQ: one launch reads each weight block once and
// accumulates all tokens. Per-token arithmetic order matches grouped_t1/t4
// exactly (mul d*q, then fma into a per-token accumulator, then the same
// shfl_down reduction), preserving bit-exact decode results.
template<int TMAX>
__global__ void q8_mmvq_grouped_t_kernel(
    const half* __restrict__ x,
    const uint8_t* __restrict__ packed,
    half* __restrict__ y,
    int H, int N, int K, int rb, int T,
    int64_t x_t_stride, int64_t x_h_stride,
    int64_t y_t_stride, int64_t y_h_stride) {
  // grid.z selects the request; arithmetic inside each request is unchanged.
  x += (int64_t)blockIdx.z * T * x_t_stride;
  y += (int64_t)blockIdx.z * T * y_t_stride;
  const int warps=blockDim.y;
  const int lane=threadIdx.x;
  const int local_row=threadIdx.y;
  const int head=blockIdx.y;
  if(head>=H) return;
  const int row=blockIdx.x*warps+local_row;
  if(row>=N) return;
  const half* xh=x+(int64_t)head*x_h_stride;
  const uint8_t* rowp=packed+((int64_t)head*N+row)*(int64_t)rb;
  float acc[TMAX];
#pragma unroll
  for(int t=0;t<TMAX;++t) acc[t]=0.f;
  const int nb=K>>5;
  for(int b=0;b<nb;++b){
    const uint8_t* qblock=rowp+(size_t)b*34;
    float d=__half2float(*reinterpret_cast<const half*>(qblock));
    float q=(float)reinterpret_cast<const int8_t*>(qblock+2)[lane];
    int k=(b<<5)+lane;
    float wq=d*q;
#pragma unroll
    for(int t=0;t<TMAX;++t){
      if(t<T) acc[t]+=wq*__half2float(xh[(int64_t)t*x_t_stride+k]);
    }
  }
#pragma unroll
  for(int t=0;t<TMAX;++t){
    if(t<T){
      float a=acc[t];
      #pragma unroll
      for(int d=16;d;d>>=1) a+=__shfl_down_sync(0xffffffff,a,d);
      if(lane==0) y[(int64_t)t*y_t_stride+(int64_t)head*y_h_stride+row]=__float2half(a);
    }
  }
}

void q8_mmvq_grouped_forward_out(torch::Tensor x, torch::Tensor packed,
                                    int64_t K, torch::Tensor y);

// Decode grouped Q8: only small T=1..6. Large T belongs to prefill.
// B1Q5/Q6 reuses the stable Q4 kernel plus a Q1/Q2 tail launch; this is the
// correctness path for the fixed-width Q6 verify graph, not a tuned Q6 leaf.
void q8_mmvq_batch_grouped_forward_out(torch::Tensor x, torch::Tensor packed,
    int64_t K, torch::Tensor y, int64_t B, int64_t Q) {
  TORCH_CHECK(x.is_cuda() && packed.is_cuda() && y.is_cuda());
  TORCH_CHECK(x.scalar_type() == torch::kFloat16 &&
              y.scalar_type() == torch::kFloat16 &&
              packed.scalar_type() == torch::kUInt8 && x.dim() == 3);
  TORCH_CHECK(B >= 1 && Q >= 1 && Q <= 6 && x.size(0) == B * Q,
              "batch grouped MMVQ supports BnQ1..6");
  if (B > 1 && Q > 2) {
    const int64_t H = packed.size(0), N = packed.size(1), rb = packed.size(2);
    TORCH_CHECK(x.size(1) == H && x.size(2) == K && y.dim() == 3 &&
                y.size(0) == B * Q && y.size(1) == H && y.size(2) == N,
                "batch grouped shapes");
    TORCH_CHECK((K % 32) == 0 && rb == (K / 32) * 34);
    c10::cuda::CUDAGuard guard(x.device());
    auto stream = at::cuda::getCurrentCUDAStream();
    constexpr int warps = 4;
    dim3 block(32, warps), grid((int)((N + warps - 1) / warps),
                                (int)H, (unsigned)B);
    auto xst = x.strides(); auto yst = y.strides();
    TORCH_CHECK(xst[2] == 1 && yst[2] == 1 && y.is_contiguous());
    auto xp = reinterpret_cast<const half*>(x.data_ptr<at::Half>());
    auto yp = reinterpret_cast<half*>(y.data_ptr<at::Half>());
    if (Q == 3) {
      q8_mmvq_grouped_t_kernel<3><<<grid, block, 0, stream>>>(
          xp, packed.data_ptr<uint8_t>(), yp, (int)H, (int)N, (int)K,
          (int)rb, (int)Q, xst[0], xst[1], yst[0], yst[1]);
    } else if (Q == 4) {
      q8_mmvq_grouped_t_kernel<4><<<grid, block, 0, stream>>>(
          xp, packed.data_ptr<uint8_t>(), yp, (int)H, (int)N, (int)K,
          (int)rb, (int)Q, xst[0], xst[1], yst[0], yst[1]);
    } else {
      q8_mmvq_grouped_t_kernel<6><<<grid, block, 0, stream>>>(
          xp, packed.data_ptr<uint8_t>(), yp, (int)H, (int)N, (int)K,
          (int)rb, (int)Q, xst[0], xst[1], yst[0], yst[1]);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return;
  }
  if (B == 1 && Q > 2) {
    q8_mmvq_grouped_forward_out(x, packed, K, y);
    return;
  }
  const int64_t H = packed.size(0), N = packed.size(1), rb = packed.size(2);
  TORCH_CHECK(x.size(1) == H && x.size(2) == K && y.dim() == 3 &&
              y.size(0) == B * Q && y.size(1) == H && y.size(2) == N,
              "batch grouped shapes");
  TORCH_CHECK((K % 32) == 0 && rb == (K / 32) * 34);
  c10::cuda::CUDAGuard guard(x.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  constexpr int warps = 4;
  dim3 block(32, warps), grid((int)((N + warps - 1) / warps),
                              (int)H, (unsigned)B);
  auto xst = x.strides(); auto yst = y.strides();
  TORCH_CHECK(xst[2] == 1 && yst[2] == 1 && y.is_contiguous());
  auto xp = reinterpret_cast<const half*>(x.data_ptr<at::Half>());
  auto yp = reinterpret_cast<half*>(y.data_ptr<at::Half>());
  if (Q == 1) {
    q8_mmvq_grouped_t1_kernel<<<grid, block, 0, stream>>>(
        xp, packed.data_ptr<uint8_t>(), yp, (int)H, (int)N, (int)K, (int)rb,
        xst[1], yst[1]);
  } else {
    q8_mmvq_grouped_t2_kernel<<<grid, block, 0, stream>>>(
        xp, packed.data_ptr<uint8_t>(), yp, (int)H, (int)N, (int)K, (int)rb,
        xst[0], xst[1], yst[0], yst[1]);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void q8_mmvq_grouped_forward_out(torch::Tensor x, torch::Tensor packed, int64_t K, torch::Tensor y){
  TORCH_CHECK(x.is_cuda()&&packed.is_cuda()&&y.is_cuda());
  TORCH_CHECK(x.scalar_type()==torch::kFloat16&&y.scalar_type()==torch::kFloat16);
  TORCH_CHECK(packed.scalar_type()==torch::kUInt8 && x.dim()==3);
  const int64_t T=x.size(0);
  TORCH_CHECK(T>=1&&T<=6,"decode grouped MMVQ requires T=1..6");
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
  if(T>=4){
    auto xp=reinterpret_cast<const half*>(x.data_ptr<at::Half>());
    auto yp=reinterpret_cast<half*>(y.data_ptr<at::Half>());
    if(T==4){
      dim3 grid4((int)((N+warps-1)/warps),(int)H,4);
      q8_mmvq_grouped_t4_kernel<<<grid4,block,0,stream>>>(
        xp, packed.data_ptr<uint8_t>(), yp,
        (int)H,(int)N,(int)K,(int)rb,xst[0],xst[1],yst[0],yst[1]);
    }else{
      // T=5/6: one launch, each weight block read once for all tokens.
      q8_mmvq_grouped_t_kernel<6><<<grid,block,0,stream>>>(
        xp, packed.data_ptr<uint8_t>(), yp,
        (int)H,(int)N,(int)K,(int)rb,(int)T,
        xst[0],xst[1],yst[0],yst[1]);
    }
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
