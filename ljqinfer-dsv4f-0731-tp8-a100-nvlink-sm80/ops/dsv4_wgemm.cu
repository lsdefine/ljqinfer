// DSV4 prefill weight GEMM (SM80): dequant fp8/fp4 -> bf16 cache + torch matmul.
// No process-global mutable state: every entry point is a pure function of its inputs.
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
#include <ATen/ops/linalg_vector_norm.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cublas_v2.h>
#include <cfloat>
#include <stdint.h>
#include <vector>

__device__ __forceinline__ float e4m3_to_float(uint8_t v) {
  const uint32_t sign = v >> 7;
  const uint32_t exp = (v >> 3) & 15;
  const uint32_t man = v & 7;
  float a;
  if (exp == 0) a = ldexpf((float)man, -9);
  else if (exp == 15 && man == 7) return __int_as_float(0x7fffffff);
  else a = ldexpf(1.0f + (float)man * 0.125f, (int)exp - 7);
  return sign ? -a : a;
}

__device__ __forceinline__ float e8m0_scale(uint8_t v) {
  // E8M0 is a bare fp32 exponent field: 2^(v-127) == bits (v << 23).
  return v == 255 ? __int_as_float(0x7fffffff) : __int_as_float((uint32_t)v << 23);
}

// e2m1 -> fp32 by bit assembly (no __constant__ load).  Magnitudes are
// {0,.5,1,1.5,2,3,4,6}: for exp>0 the value is 2^(exp-1)*(1+man/2), giving
// fp32 exponent exp+126 with the mantissa msb = man.  exp==0 yields 0.0/0.5.
__device__ __forceinline__ float e2m1_to_float(uint8_t v) {
  const uint32_t e = (v >> 1) & 3, m = v & 1;
  uint32_t bits = e ? (((e + 126u) << 23) | (m << 22)) : (m ? (126u << 23) : 0u);
  bits |= (uint32_t)(v & 8) << 28;   // sign
  return __int_as_float(bits);
}

// One thread per output element w[n,k] (bf16 row-major [N,K]).
__global__ void dequant_fp8_kernel(const uint8_t* __restrict__ w,
                                   const uint8_t* __restrict__ s,
                                   __nv_bfloat16* __restrict__ o,
                                   long long total, int K, int sblocks_k) {
  long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= total) return;
  int n = (int)(i / K), k = (int)(i % K);
  float sf = e8m0_scale(s[(size_t)(n >> 7) * sblocks_k + (k >> 7)]);
  o[i] = __float2bfloat16(e4m3_to_float(w[i]) * sf);
}

__global__ void dequant_fp4_kernel(const uint8_t* __restrict__ w,
                                   const uint8_t* __restrict__ s,
                                   __nv_bfloat16* __restrict__ o,
                                   long long total, int K, int stride) {
  long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= total) return;
  int n = (int)(i / K), k = (int)(i % K);
  uint8_t byte = w[(size_t)n * stride + (k >> 1)];
  float wf = e2m1_to_float((k & 1) ? (byte >> 4) : (byte & 15));
  float sf = e8m0_scale(s[(size_t)n * (K >> 5) + (k >> 5)]);
  o[i] = __float2bfloat16(wf * sf);
}

// 8 elems per thread (4 packed bytes, single scale fetch per 32-block boundary-safe: 8|32)
__global__ void dequant_fp4_bf16_kernel(const uint8_t* __restrict__ w,
                                        const uint8_t* __restrict__ s,
                                        __nv_bfloat16* __restrict__ o,
                                        long long total8, int K, int stride) {
  long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= total8) return;
  long long e8 = i * 8;
  long long n = e8 / K; int k = (int)(e8 % K);
  uint32_t four = *reinterpret_cast<const uint32_t*>(w + (size_t)n * stride + (k >> 1));
  float sf = e8m0_scale(s[(size_t)n * (K >> 5) + (k >> 5)]);
  __nv_bfloat162 r[4];
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    uint32_t byte = (four >> (8 * j)) & 0xFF;
    r[j] = __floats2bfloat162_rn(e2m1_to_float(byte & 15) * sf,
                             e2m1_to_float(byte >> 4) * sf);
  }
  *reinterpret_cast<uint4*>(o + e8) = *reinterpret_cast<uint4*>(r);
}

// Batch dequant fp4 -> bf16 [E, N, K] for grouped GEMM (cutlass so from ref).

// ---- Sinkhorn 4x4 fused (HC=4, 20 iters) ----
// One thread owns one token's 4x4 matrix in registers; 80 ATen launches -> 1.
template <int HC>
__global__ void sinkhorn_hc_kernel(const float* __restrict__ inp,
                                   float* __restrict__ out,
                                   int64_t T, float eps, int iters) {
  int64_t t = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
  if (t >= T) return;
  const int N = HC * HC;
  float m[HC * HC];
  const float* src = inp + t * N;
#pragma unroll
  for (int i = 0; i < N; ++i) m[i] = src[i];
  // softmax over last dim (-1), then +eps
#pragma unroll
  for (int r = 0; r < HC; ++r) {
    float mx = m[r * HC];
#pragma unroll
    for (int c = 1; c < HC; ++c) mx = fmaxf(mx, m[r * HC + c]);
    float s = 0.f;
#pragma unroll
    for (int c = 0; c < HC; ++c) { m[r * HC + c] = __expf(m[r * HC + c] - mx); s += m[r * HC + c]; }
#pragma unroll
    for (int c = 0; c < HC; ++c) m[r * HC + c] = m[r * HC + c] / s + eps;
  }
  // col-normalize once (sum over dim -2)
#pragma unroll
  for (int c = 0; c < HC; ++c) {
    float s = 0.f;
#pragma unroll
    for (int r = 0; r < HC; ++r) s += m[r * HC + c];
    s += eps;
#pragma unroll
    for (int r = 0; r < HC; ++r) m[r * HC + c] /= s;
  }
  for (int it = 0; it < iters - 1; ++it) {
#pragma unroll
    for (int r = 0; r < HC; ++r) {
      float s = 0.f;
#pragma unroll
      for (int c = 0; c < HC; ++c) s += m[r * HC + c];
      s += eps;
#pragma unroll
      for (int c = 0; c < HC; ++c) m[r * HC + c] /= s;
    }
#pragma unroll
    for (int c = 0; c < HC; ++c) {
      float s = 0.f;
#pragma unroll
      for (int r = 0; r < HC; ++r) s += m[r * HC + c];
      s += eps;
#pragma unroll
      for (int r = 0; r < HC; ++r) m[r * HC + c] /= s;
    }
  }
  float* dst = out + t * N;
#pragma unroll
  for (int i = 0; i < N; ++i) dst[i] = m[i];
}


static torch::Tensor sinkhorn_hc(torch::Tensor comb, double eps, int64_t iters) {
  TORCH_CHECK(comb.dim() == 3 && comb.size(1) == comb.size(2), "comb must be [T,HC,HC]");
  TORCH_CHECK(comb.size(1) == 4, "only HC=4 supported");
  c10::cuda::CUDAGuard device_guard(comb.device());
  auto c = comb.to(torch::kFloat32).contiguous();
  auto out = at::empty_like(c);
  const int64_t T = c.size(0);
  const int threads = 128;
  const int blocks = (int)((T + threads - 1) / threads);
  auto stream = at::cuda::getCurrentCUDAStream();
  sinkhorn_hc_kernel<4><<<blocks, threads, 0, stream>>>(
      c.data_ptr<float>(), out.data_ptr<float>(), T, (float)eps, (int)iters);
  return out;
}

// fused hc_pre host wrapper (defined after hc_fused_pre_kernel below).
// Returns {post, comb, x_normed[, x_raw]}; x_raw only when want_xraw.
static std::vector<torch::Tensor> hc_fused_pre(
    torch::Tensor x4, torch::Tensor fn, torch::Tensor hs, torch::Tensor hb,
    torch::Tensor norm_w, double eps, bool want_xraw);

// selected fp4->bf16 dequant: only active experts, fused index_select (port of
// GLM dequant_iq3_selected_cuda). out [na*N, K]; src row = act[a]*N + r.
// One block per output row (expert a, row r): all indexing from blockIdx, no
// 64-bit integer division in the inner loop (K/N are runtime values, so the
// compiler cannot strength-reduce the old i/K, n/N divides).
// 256 threads per block, each thread emits one 8-element (uint4) chunk.
// A block covers `rpb` consecutive rows so that occupancy stays high even when
// K is small (w2 has K=256 -> only 32 chunks per row).  kc_shift = log2(K/8),
// so the row/chunk split is a shift, never a division.
__global__ void dequant_fp4_selected_kernel(const uint8_t* __restrict__ w,
                                            const uint8_t* __restrict__ s,
                                            const int64_t* __restrict__ act,
                                            __nv_bfloat16* __restrict__ o,
                                            int K, int N, int stride,
                                            int rpb, int kc_shift) {
  const int kc = 1 << kc_shift;              // chunks per row = K/8
  const int lr = threadIdx.x >> kc_shift;    // row within this block
  const int c = threadIdx.x & (kc - 1);      // chunk within the row
  const int r0 = blockIdx.x * rpb;
  const int a = blockIdx.y;
  const int r = r0 + lr;
  if (r >= N) return;
  const long long n = (long long)a * N + r;
  const long long src = act[a] * (long long)N + r;
  const uint8_t* __restrict__ wrow = w + (size_t)src * stride;
  const uint8_t* __restrict__ srow = s + (size_t)src * (K >> 5);
  __nv_bfloat16* __restrict__ orow = o + n * K;
  for (int k = c * 8; k < K; k += kc * 8) {
    uint32_t four = *reinterpret_cast<const uint32_t*>(wrow + (k >> 1));
    float sf = e8m0_scale(srow[k >> 5]);
    __nv_bfloat162 rr[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      uint32_t byte = (four >> (8 * j)) & 0xFF;
      rr[j] = __floats2bfloat162_rn(e2m1_to_float(byte & 15) * sf,
                                    e2m1_to_float(byte >> 4) * sf);
    }
    *reinterpret_cast<uint4*>(orow + k) = *reinterpret_cast<uint4*>(rr);
  }
}

static torch::Tensor dequant_fp4_selected_bf16(torch::Tensor weight, torch::Tensor scale,
                                               torch::Tensor act, int64_t K) {
  const int64_t N = weight.size(1), na = act.numel();
  auto o = torch::empty({na * N, K},
      torch::TensorOptions().dtype(torch::kBFloat16).device(weight.device()));
  int kc = (int)(K / 8);                       // chunks per row (K is a power of 2)
  if (kc > 256) kc = 256;
  int kc_shift = 0;
  while ((1 << kc_shift) < kc) ++kc_shift;
  const int th = 256;
  const int rpb = th >> kc_shift;              // rows handled per block
  dim3 grid((unsigned)((N + rpb - 1) / rpb), (unsigned)na);
  auto stream = at::cuda::getCurrentCUDAStream();
  dequant_fp4_selected_kernel<<<grid, th, 0, stream>>>(
      reinterpret_cast<const uint8_t*>(weight.data_ptr()),
      reinterpret_cast<const uint8_t*>(scale.data_ptr()),
      act.data_ptr<int64_t>(),
      reinterpret_cast<__nv_bfloat16*>(o.data_ptr()), (int)K, (int)N, (int)(K >> 1),
      rpb, kc_shift);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return o;
}



static torch::Tensor dequant_w(torch::Tensor weight, torch::Tensor scale, int64_t K, bool fp4) {
  auto opts = torch::TensorOptions().dtype(torch::kBFloat16).device(weight.device());
  const int64_t N = weight.size(0);
  auto w = torch::empty({N, K}, opts);
  long long total = (long long)N * K;
  int th = 256;
  long long bl = (total + th - 1) / th;
  auto stream = at::cuda::getCurrentCUDAStream();
  if (fp4) {
    dequant_fp4_kernel<<<(int)bl, th, 0, stream>>>(
        reinterpret_cast<const uint8_t*>(weight.data_ptr()),
        reinterpret_cast<const uint8_t*>(scale.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(w.data_ptr()),
        total, (int)K, (int)weight.size(1));
  } else {
    int sblocks_k = (int)((K + 127) >> 7);
    dequant_fp8_kernel<<<(int)bl, th, 0, stream>>>(
        reinterpret_cast<const uint8_t*>(weight.data_ptr()),
        reinterpret_cast<const uint8_t*>(scale.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(w.data_ptr()),
        total, (int)K, sblocks_k);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return w;
}


static torch::Tensor get_or_cast(torch::Tensor t, torch::ScalarType dt) {
  // No pointer-keyed cache: callers may pass temporaries (e.g. `.to(dev)`),
  // whose freed addresses get reused -> stale hits. Casting small params is cheap.
  if (t.scalar_type() == dt && t.is_contiguous()) return t;
  return t.to(dt).contiguous();
}


// Shape-stable HC projection. One block owns one token, so total T cannot
// change either the norm reduction or any of the 24 dot-product trees.
__global__ void hc_project_fixed_kernel(
    const __nv_bfloat16* __restrict__ x, const float* __restrict__ w,
    float* __restrict__ out, int K, int N, float eps) {
  const int t = blockIdx.x, tid = threadIdx.x;
  const int lane = tid & 31, warp = tid >> 5;
  const __nv_bfloat16* xr = x + (int64_t)t * K;
  __shared__ float warp_sum[8];
  __shared__ float inv_rms;
  float ss = 0.f;
  for (int k = tid; k < K; k += blockDim.x) {
    const float v = __bfloat162float(xr[k]);
    ss = fmaf(v, v, ss);
  }
  for (int d = 16; d; d >>= 1) ss += __shfl_down_sync(0xffffffff, ss, d);
  if (lane == 0) warp_sum[warp] = ss;
  __syncthreads();
  if (warp == 0) {
    float s = lane < 8 ? warp_sum[lane] : 0.f;
    for (int d = 16; d; d >>= 1) s += __shfl_down_sync(0xffffffff, s, d);
    if (lane == 0) inv_rms = rsqrtf(s / (float)K + eps);
  }
  __syncthreads();
  for (int n = warp; n < N; n += 8) {
    const float* wr = w + (int64_t)n * K;
    float acc = 0.f;
    for (int k = lane; k < K; k += 32)
      acc = fmaf(__bfloat162float(xr[k]), wr[k], acc);
    for (int d = 16; d; d >>= 1) acc += __shfl_down_sync(0xffffffff, acc, d);
    if (lane == 0) out[(int64_t)t * N + n] = acc * inv_rms;
  }
}

static torch::Tensor hc_project_fixed(torch::Tensor x4, torch::Tensor fn,
                                      double eps) {
  at::cuda::CUDAGuard guard(x4.device());
  auto x = (x4.scalar_type() == torch::kBFloat16 ? x4 : x4.to(torch::kBFloat16)).contiguous();
  auto w = get_or_cast(fn, torch::kFloat32).contiguous();
  const int64_t T = x.size(0), K = x.numel() / T, N = w.size(0);
  TORCH_CHECK(w.dim() == 2 && w.size(1) == K && N == 24,
              "HC projection shape mismatch");
  auto out = at::empty({T, N}, x.options().dtype(torch::kFloat32));
  hc_project_fixed_kernel<<<(int)T, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), w.data_ptr<float>(),
      out.data_ptr<float>(), (int)K, (int)N, (float)eps);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}


// Pure BF16 GEMM y = x @ w^T.  Weights are dequantized ONCE at load time
// (see dequant_fp8_bf16 + model/bind.py); no runtime cache, no hidden state.
torch::Tensor bf16_gemm(torch::Tensor x, torch::Tensor w) {
  at::cuda::CUDAGuard guard(x.device());
  TORCH_CHECK(w.scalar_type() == torch::kBFloat16 && w.dim() == 2,
              "bf16_gemm: weight must be a pre-dequantized BF16 [N,K] tensor (got ",
              w.scalar_type(), ", dim=", w.dim(), ")");
  TORCH_CHECK(x.dim() == 2 && x.size(1) == w.size(1),
              "bf16_gemm: shape mismatch x", x.sizes(), " w", w.sizes());
  const int64_t M = x.size(0), K = x.size(1), N = w.size(0);
  // NOTE(2026-09-05): the hand-written M==8 kernel was removed after measurement.
  // It was 33~51% SLOWER than cuBLAS at the only shapes it ever ran on
  //   (N=2048,K=7168): 31.2us vs 20.9us   (N=7168,K=2048): 40.0us vs 19.5us
  // and its accuracy was IDENTICAL to cuBLAS (same rel-err vs fp32 reference).
  // Real engine step time B1Q8: 31.20ms -> 30.86ms (3 runs each, no overlap).
  // Semantic + tool-call gate passes on cuBLAS path. Backup: dsv4_wgemm.cu.bak_smallm
  (void)M; (void)K; (void)N;
  return at::matmul(x, w.t());
}

// Load-time dequant (pure): fp8 e4m3 [N,K] + block scale -> BF16 [N,K].
torch::Tensor dequant_fp8_bf16(torch::Tensor weight, torch::Tensor scale) {
  at::cuda::CUDAGuard guard(weight.device());
  TORCH_CHECK(weight.dim() == 2 && weight.scalar_type() == torch::kFloat8_e4m3fn,
              "dequant_fp8_bf16 expects [N,K] fp8 e4m3 weight");
  TORCH_CHECK(scale.dim() == 2 && scale.size(0) == weight.size(0) / 128 &&
              scale.size(1) == (weight.size(1) + 127) / 128,
              "dequant_fp8_bf16 scale shape mismatch");
  return dequant_w(weight.contiguous(), scale.contiguous(), weight.size(1), false);
}

// BF16 input/output GEMM with an explicit FP32 accumulation contract. Used
// only for the prefill WKV projection; all other GEMMs keep their hot path.
static torch::Tensor bf16_gemm_fp32_accum(torch::Tensor x, torch::Tensor w) {
  at::cuda::CUDAGuard guard(x.device());
  x = x.contiguous();
  w = w.contiguous();
  TORCH_CHECK(x.scalar_type() == torch::kBFloat16 &&
              w.scalar_type() == torch::kBFloat16,
              "stable WKV GEMM expects BF16 tensors");
  TORCH_CHECK(x.dim() == 2 && w.dim() == 2 && x.size(1) == w.size(1),
              "stable WKV GEMM shape mismatch");
  const int M = (int)x.size(0), N = (int)w.size(0), K = (int)x.size(1);
  auto y = at::empty({M, N}, x.options());
  const float alpha = 1.f, beta = 0.f;
  const cublasStatus_t st = cublasGemmEx(
      at::cuda::getCurrentCUDABlasHandle(), CUBLAS_OP_T, CUBLAS_OP_N,
      N, M, K, &alpha,
      w.data_ptr(), CUDA_R_16BF, K,
      x.data_ptr(), CUDA_R_16BF, K,
      &beta, y.data_ptr(), CUDA_R_16BF, N,
      CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
  TORCH_CHECK(st == CUBLAS_STATUS_SUCCESS,
              "stable WKV cublasGemmEx failed with status ", (int)st);
  return y;
}





// wo_a grouped projection: o [T,G,D] bf16 x w [G,R,D] bf16 -> y [T,G,R] bf16 (fp32 accum). Pure.
static torch::Tensor wo_a_grouped(torch::Tensor o, torch::Tensor w) {
  at::cuda::CUDAGuard guard(o.device());
  TORCH_CHECK(o.scalar_type() == torch::kBFloat16 && w.scalar_type() == torch::kBFloat16,
              "wo_a_grouped: bf16 only");
  TORCH_CHECK(o.dim() == 3 && w.dim() == 3 && o.is_contiguous() && w.is_contiguous(),
              "wo_a_grouped: o [T,G,D], w [G,R,D] must be contiguous 3D");
  const int T = (int)o.size(0), G = (int)o.size(1), D = (int)o.size(2), Rr = (int)w.size(1);
  TORCH_CHECK(w.size(0) == G && w.size(2) == D, "wo_a_grouped: shape mismatch");
  auto y = at::empty({T, G, Rr}, o.options());
  if (T == 0) return y;
  const float alpha = 1.f, beta = 0.f;
  const cublasStatus_t st = cublasGemmStridedBatchedEx(
      at::cuda::getCurrentCUDABlasHandle(), CUBLAS_OP_T, CUBLAS_OP_N,
      Rr, T, D, &alpha,
      w.data_ptr(), CUDA_R_16BF, D, (long long)Rr * D,
      o.data_ptr(), CUDA_R_16BF, G * D, (long long)D,
      &beta, y.data_ptr(), CUDA_R_16BF, G * Rr, (long long)Rr,
      G, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
  TORCH_CHECK(st == CUBLAS_STATUS_SUCCESS, "wo_a_grouped cublasGemmStridedBatchedEx failed ", (int)st);
  return y;
}

// mxfp8 qdq (block=128), copied from dsv4_mxfp8_qdq.cu for the fused MoE path.
__device__ __forceinline__ float wgemm_warp_max(float v) {
#pragma unroll
  for (int d = 16; d; d >>= 1) v = fmaxf(v, __shfl_down_sync(0xffffffff, v, d));
  return v;
}
__global__ void qdq128_kernel(const __nv_bfloat16* __restrict__ x,
                              __nv_bfloat16* __restrict__ y, int64_t nblocks) {
  int64_t b = blockIdx.x;
  if (b >= nblocks) return;
  int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  int64_t i = b * 128 + threadIdx.x;
  float v = __bfloat162float(x[i]);
  float m = wgemm_warp_max(fabsf(v));
  __shared__ float wm[4];
  if (lane == 0) wm[warp] = m;
  __syncthreads();
  if (warp == 0) {
    float z = lane < 4 ? wm[lane] : 0.f;
    z = wgemm_warp_max(z);
    if (lane == 0) wm[0] = z;
  }
  __syncthreads();
  float amax = fmaxf(wm[0], 1.0e-4f);
  float scale = exp2f(ceilf(log2f(amax / 448.0f)));
  float z = fminf(fmaxf(v / scale, -448.0f), 448.0f);
  __nv_fp8_storage_t q = __nv_cvt_float_to_fp8(z, __NV_SATFINITE, __NV_E4M3);
  __nv_fp8_e4m3 qv; qv.__x = q;
  y[i] = __float2bfloat16_rn((float)qv * scale);
}

__device__ __forceinline__ float ma_warp_max(float v) {
  for (int o = 16; o > 0; o >>= 1)
    v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, o));
  return v;
}

__global__ void moe_act_qdq_kernel(const __nv_bfloat16* __restrict__ g,
                                   const __nv_bfloat16* __restrict__ u,
                                   const float* __restrict__ rw,
                                   __nv_bfloat16* __restrict__ out,
                                   int64_t nblocks, int M) {
  int64_t b = blockIdx.x;
  if (b >= nblocks) return;
  int64_t i = b * 128 + threadIdx.x;
  int64_t row = i / M;

  float gv = fminf(__bfloat162float(g[i]), 10.0f);
  float uv = fminf(fmaxf(__bfloat162float(u[i]), -10.0f), 10.0f);
  float v = (gv / (1.0f + __expf(-gv))) * uv * rw[row];

  int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  float m = ma_warp_max(fabsf(v));
  __shared__ float wm[4];
  if (lane == 0) wm[warp] = m;
  __syncthreads();
  if (warp == 0) {
    float z = lane < 4 ? wm[lane] : 0.f;
    z = ma_warp_max(z);
    if (lane == 0) wm[0] = z;
  }
  __syncthreads();
  float amax = fmaxf(wm[0], 1.0e-4f);
  float scale = exp2f(ceilf(log2f(amax / 448.0f)));
  float z = fminf(fmaxf(v / scale, -448.0f), 448.0f);
  __nv_fp8_storage_t q = __nv_cvt_float_to_fp8(z, __NV_SATFINITE, __NV_E4M3);
  float dq = __half2float(__nv_cvt_fp8_to_halfraw(q, __NV_E4M3)) * scale;
  out[i] = __float2bfloat16(dq);
}

static torch::Tensor moe_act_qdq(torch::Tensor g, torch::Tensor u, torch::Tensor rw) {
  at::cuda::CUDAGuard guard(g.device());
  auto gc = g.contiguous(), uc = u.contiguous();
  auto rwf = rw.to(torch::kFloat32).contiguous();
  const int64_t n = gc.size(0), M = gc.size(1);
  auto out = torch::empty_like(gc);
  auto stream = at::cuda::getCurrentCUDAStream();
  int64_t nb = n * M / 128;
  moe_act_qdq_kernel<<<(int)nb, 128, 0, stream>>>(
      (const __nv_bfloat16*)gc.data_ptr(), (const __nv_bfloat16*)uc.data_ptr(),
      rwf.data_ptr<float>(), (__nv_bfloat16*)out.data_ptr(), nb, (int)M);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

// ---- FP4 dequant-on-the-fly grouped GEMM (sm80 WMMA bf16) ----
// x:[n,K] bf16 (rows grouped by expert) ; w:[E,N,K] fp4 packed ; s:[E,N,K/32] e8m0
// tile_e[]/tile_r0[]: per-tile expert id and first row (16 rows per tile)
// out:[n,N] bf16
#include <mma.h>


// ---- GPU MoE dispatch (ported from ref/ops/prefill_moe_dispatch.cu, TOPK=6) ----
__global__ void dsv4_moe_count256(const int64_t* e, int n, int* cnt) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) atomicAdd(&cnt[e[i]], 1);
}
__global__ void dsv4_moe_scan256(const int* cnt, int* off, int* pos) {
  __shared__ int a[256];
  int t = threadIdx.x;
  a[t] = cnt[t];
  __syncthreads();
  for (int d = 1; d < 256; d <<= 1) {
    int v = t >= d ? a[t - d] : 0;
    __syncthreads();
    if (t >= d) a[t] += v;
    __syncthreads();
  }
  off[t + 1] = a[t];
  if (t == 0) off[0] = 0;
  pos[t] = t ? a[t - 1] : 0;
}
// scatter: also emit slot index per (token,k) for the later slot-sum kernel
__global__ void dsv4_moe_scatter256(const int64_t* e, int n, int* pos,
                                    int64_t* tok, int64_t* tk, int32_t* slot) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    int x = (int)e[i];
    int j = atomicAdd(&pos[x], 1);
    tok[j] = i / 6;
    tk[j] = i % 6;
    slot[i] = j;
  }
}

__global__ void fp4_grouped_gemm_kernel(
    const __nv_bfloat16* __restrict__ x,
    const uint8_t* __restrict__ w,
    const uint8_t* __restrict__ s,
    const int* __restrict__ tile_e,
    const int* __restrict__ tile_r0,
    const int* __restrict__ tile_rn,
    __nv_bfloat16* __restrict__ out,
    int n, int N, int K) {
  const int t = blockIdx.x;
  const int e = tile_e[t], r0 = tile_r0[t], rn = tile_rn[t];
  const int col0 = blockIdx.y * 64;
  const int tid = threadIdx.x, warp = tid >> 5;

  __shared__ __nv_bfloat16 Bs[64 * 64];  // [k=64][ncol=64]
  __shared__ __nv_bfloat16 As[64 * 64];  // [m=64][k=64]

  const size_t wrow = (size_t)K >> 1;          // bytes per weight row
  const size_t srow = (size_t)K >> 5;          // scale entries per row
  const uint8_t* wE = w + (size_t)e * N * wrow;
  const uint8_t* sE = s + (size_t)e * N * srow;

  using namespace nvcuda::wmma;
  fragment<accumulator, 16, 16, 16, float> c_frag[2];
  fill_fragment(c_frag[0], 0.f);
  fill_fragment(c_frag[1], 0.f);
  const int mb = warp & 3;          // m-block 0..3  (rows mb*16)
  const int nb0 = (warp >> 2) * 2;  // n-block pair

  const int brow = tid >> 3;          // 0..31 -> two passes for 64 rows
  const int kseg = (tid & 7) * 8;     // 8 elems per thread

  for (int k0 = 0; k0 < K; k0 += 64) {
    // decode B: 64 weight rows x 64 k
    #pragma unroll
    for (int p = 0; p < 2; ++p) {
      const int nc = brow + p * 32;
      const int kk = k0 + kseg;
      uint32_t four = *reinterpret_cast<const uint32_t*>(wE + (size_t)(col0 + nc) * wrow + (kk >> 1));
      float sf = e8m0_scale(sE[(size_t)(col0 + nc) * srow + (kk >> 5)]);
      #pragma unroll
      for (int j = 0; j < 4; ++j) {
        uint32_t byte = (four >> (8 * j)) & 0xFF;
        Bs[(kseg + 2 * j) * 64 + nc] = __float2bfloat16(e2m1_to_float(byte & 15) * sf);
        Bs[(kseg + 2 * j + 1) * 64 + nc] = __float2bfloat16(e2m1_to_float(byte >> 4) * sf);
      }
    }
    // load A tile [64,64]
    for (int i = tid; i < 64 * 64; i += 256) {
      int m = i >> 6, k = i & 63;
      As[i] = (m < rn) ? x[(size_t)(r0 + m) * K + k0 + k] : __float2bfloat16(0.f);
    }
    __syncthreads();
    {
      fragment<matrix_a, 16, 16, 16, __nv_bfloat16, row_major> a_frag;
      fragment<matrix_b, 16, 16, 16, __nv_bfloat16, row_major> b_frag;
      #pragma unroll
      for (int kk = 0; kk < 4; ++kk) {
        load_matrix_sync(a_frag, As + mb * 16 * 64 + kk * 16, 64);
        #pragma unroll
        for (int j = 0; j < 2; ++j) {
          load_matrix_sync(b_frag, Bs + (nb0 + j) * 16 + kk * 16 * 64, 64);
          mma_sync(c_frag[j], a_frag, b_frag, c_frag[j]);
        }
      }
    }
    __syncthreads();
  }
  // store 64x64 tile -> bf16 out (8 warps x 2 frags)
  __shared__ float Cs[64 * 64];
  #pragma unroll
  for (int j = 0; j < 2; ++j)
    store_matrix_sync(Cs + mb * 16 * 64 + (nb0 + j) * 16, c_frag[j], 64,
                      nvcuda::wmma::mem_row_major);
  __syncthreads();
  for (int i = tid; i < 64 * 64; i += 256) {
    int m = i >> 6, c = i & 63;
    if (m < rn) out[(size_t)(r0 + m) * N + col0 + c] = __float2bfloat16(Cs[i]);
  }
}

static torch::Tensor fp4_grouped_gemm(torch::Tensor x, torch::Tensor w,
                                      torch::Tensor s, torch::Tensor tiles) {
  at::cuda::CUDAGuard g(x.device());
  const int64_t n = x.size(0), K = x.size(1), N = w.size(1);
  TORCH_CHECK(N % 64 == 0 && K % 64 == 0, "fp4 gemm: N,K must be mult of 64");
  auto out = torch::empty({n, N}, x.options());
  const int64_t nt = tiles.size(1);
  if (nt == 0) return out;
  auto tc = tiles.to(x.device(), /*non_blocking=*/true).contiguous();
  auto stream = at::cuda::getCurrentCUDAStream();
  dim3 grid((unsigned)nt, (unsigned)(N / 64));
  fp4_grouped_gemm_kernel<<<grid, 256, 0, stream>>>(
      (const __nv_bfloat16*)x.data_ptr(),
      reinterpret_cast<const uint8_t*>(w.data_ptr()),
      reinterpret_cast<const uint8_t*>(s.data_ptr()),
      tc[0].data_ptr<int>(), tc[1].data_ptr<int>(), tc[2].data_ptr<int>(),
      (__nv_bfloat16*)out.data_ptr(), (int)n, (int)N, (int)K);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

// in-place fp8 e4m3 qdq, block=64 along last dim, pow2 scale (== act_quant(..., 64, inplace=True)).
// x may be a last-dim slice (row stride != cols); one warp per 64-block, 4 warps per CTA.
__global__ void fp8_qdq64_inplace_kernel(__nv_bfloat16* __restrict__ x, int64_t rows, int64_t cols, int64_t rs) {
  const int64_t nblk_row = cols / 64;
  const int64_t b = (int64_t)blockIdx.x * 4 + (threadIdx.x >> 5);
  if (b >= rows * nblk_row) return;
  const int lane = threadIdx.x & 31;
  const int64_t r = b / nblk_row, c0 = (b % nblk_row) * 64;
  __nv_bfloat16* p = x + r * rs + c0;
  float v0 = __bfloat162float(p[lane]), v1 = __bfloat162float(p[lane + 32]);
  float a = fmaxf(fabsf(v0), fabsf(v1));
  #pragma unroll
  for (int o = 16; o > 0; o >>= 1) a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, o));
  const float amax = fmaxf(a, 1.0e-4f);
  const float scale = exp2f(ceilf(log2f(amax / 448.0f)));
  float z0 = fminf(fmaxf(v0 / scale, -448.0f), 448.0f), z1 = fminf(fmaxf(v1 / scale, -448.0f), 448.0f);
  __nv_fp8_e4m3 q0, q1;
  q0.__x = __nv_cvt_float_to_fp8(z0, __NV_SATFINITE, __NV_E4M3);
  q1.__x = __nv_cvt_float_to_fp8(z1, __NV_SATFINITE, __NV_E4M3);
  p[lane] = __float2bfloat16_rn((float)q0 * scale);
  p[lane + 32] = __float2bfloat16_rn((float)q1 * scale);
}

torch::Tensor fp8_qdq64_(torch::Tensor x) {
  TORCH_CHECK(x.scalar_type() == torch::kBFloat16 && x.is_cuda(), "fp8_qdq64_: bf16 cuda");
  const int64_t cols = x.size(-1);
  TORCH_CHECK(cols % 64 == 0 && x.stride(-1) == 1, "fp8_qdq64_: last dim %64, contiguous");
  auto x2 = x.reshape({-1, cols});  // view if outer dims are contiguous
  TORCH_CHECK(x2.data_ptr() == x.data_ptr(), "fp8_qdq64_: reshape must be a view");
  const int64_t rows = x2.size(0), rs = x2.stride(0);
  const int64_t nblk = rows * (cols / 64);
  if (nblk == 0) return x;
  auto st = at::cuda::getCurrentCUDAStream();
  fp8_qdq64_inplace_kernel<<<(int)((nblk + 3) / 4), 128, 0, st>>>(
      reinterpret_cast<__nv_bfloat16*>(x2.data_ptr()), rows, cols, rs);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return x;
}

// in-place per-row rms scale: x *= rsqrt(mean(x^2) + eps), emulating torch bf16 step rounding
// (square->bf16, mean->bf16, +eps->bf16, rsqrt->bf16, mul->bf16). x [rows, D] contiguous bf16.
__global__ void rms_scale_inplace_kernel(__nv_bfloat16* __restrict__ x, int D, float eps) {
  __nv_bfloat16* xi = x + (size_t)blockIdx.x * D;
  float sum = 0.f;
  for (int i = threadIdx.x; i < D; i += blockDim.x) {
    const float v = __bfloat162float(xi[i]);
    sum += __bfloat162float(__float2bfloat16(v * v));
  }
  __shared__ float sm[32];
#pragma unroll
  for (int d = 16; d; d >>= 1) sum += __shfl_down_sync(0xffffffffu, sum, d);
  const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
  if (lane == 0) sm[wid] = sum;
  __syncthreads();
  const int nw = (blockDim.x + 31) >> 5;
  if (wid == 0) {
    float tot = (lane < nw) ? sm[lane] : 0.f;
#pragma unroll
    for (int d = 16; d; d >>= 1) tot += __shfl_down_sync(0xffffffffu, tot, d);
    if (lane == 0) sm[0] = tot;
  }
  __syncthreads();
  const float mean = __bfloat162float(__float2bfloat16(sm[0] / (float)D));
  const float t = __bfloat162float(__float2bfloat16(mean + eps));
  const float inv = __bfloat162float(__float2bfloat16(rsqrtf(t)));
  for (int i = threadIdx.x; i < D; i += blockDim.x)
    xi[i] = __float2bfloat16(__bfloat162float(xi[i]) * inv);
}

// ---- fp32 skinny GEMM y[M,N]=x[M,K]@w[N,K]^T, ANY M.  (A7: batched rewrite, v2)
// 256 threads split K exactly like the pre-A7 kernel (thread t owns float4 indices t, t+256, ...; warp
// xor-shuffle 16..1; then the 8 warps are summed in order) -> every output is BITWISE IDENTICAL to the
// pre-A7 kernel and batch-invariant (row m never depends on M).
// Why the rewrite: the old grid(N, ceil(M/8)) launched the ceil(M/8) M-tiles of one column far apart in
// time, so w (29MB for 7168x1024, > what L2 keeps hot) was re-read from HBM once per M-tile: cost grew
// ~linearly in M (7168x1024: M=8 33.8us -> M=32 140.6us).  Now (1) grid.x = M-tiles (fastest) so the
// tiles of the same column group are co-resident and share w through L2, and (2) each thread applies one
// x float4 to NT columns held in registers (x is tiny and L2-resident; NO smem staging / __syncthreads:
// an smem-staged variant measured 2x slower because 32KB smem per block killed occupancy).
// NO cuBLAS fallback: no fallback, no silent fork.
#ifndef SKINNY_MT
#define SKINNY_MT 8
#endif
#ifndef SKINNY_NT
#define SKINNY_NT 2
#endif
template <int MT, int NT>
__global__ void __launch_bounds__(256) sgemm_skinny2_f32_kernel(const float* __restrict__ x, const float* __restrict__ w,
                                         float* __restrict__ y, int N, int K, int M) {
  const int t = threadIdx.x;
  const int mb = blockIdx.x * MT;               // first row of this tile (M-tiles fastest -> w shared via L2)
  const int nb = blockIdx.y * NT;               // first column of this tile
  const int mcnt = min(MT, M - mb);             // valid rows (block-uniform)
  const int ncnt = min(NT, N - nb);             // valid cols (block-uniform)
  const int K4 = K >> 2;
  const float4* x4 = reinterpret_cast<const float4*>(x + (size_t)mb * K);
  const float4* w4 = reinterpret_cast<const float4*>(w + (size_t)nb * K);
  float acc[NT][MT];
#pragma unroll
  for (int j = 0; j < NT; ++j)
#pragma unroll
    for (int m = 0; m < MT; ++m) acc[j][m] = 0.f;
  for (int i = t; i < K4; i += 256) {
    float4 a[NT];
#pragma unroll
    for (int j = 0; j < NT; ++j) a[j] = (j < ncnt) ? w4[(size_t)j * K4 + i] : make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
    for (int m = 0; m < MT; ++m) {
      if (m >= mcnt) break;                     // block-uniform
      const float4 b = x4[(size_t)m * K4 + i];
#pragma unroll
      for (int j = 0; j < NT; ++j) {
        acc[j][m] = fmaf(a[j].x, b.x, acc[j][m]); acc[j][m] = fmaf(a[j].y, b.y, acc[j][m]);
        acc[j][m] = fmaf(a[j].z, b.z, acc[j][m]); acc[j][m] = fmaf(a[j].w, b.w, acc[j][m]);
      }
    }
  }
  __shared__ float red[8][NT][MT];
  const int lane = t & 31, wid = t >> 5;
#pragma unroll
  for (int j = 0; j < NT; ++j)
#pragma unroll
    for (int m = 0; m < MT; ++m) {
      float v = acc[j][m];
      for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
      if (lane == 0) red[wid][j][m] = v;
    }
  __syncthreads();
  for (int e = t; e < NT * MT; e += 256) {
    const int j = e / MT, m = e - j * MT;
    if (j < ncnt && m < mcnt) {
      float s = 0.f;
#pragma unroll
      for (int w2 = 0; w2 < 8; ++w2) s += red[w2][j][m];
      y[(size_t)(mb + m) * N + nb + j] = s;
    }
  }
}

torch::Tensor sgemm_skinny2_f32(torch::Tensor x, torch::Tensor w) {
  TORCH_CHECK(x.is_cuda() && w.is_cuda() && x.dtype() == torch::kFloat32 && w.dtype() == torch::kFloat32);
  TORCH_CHECK(x.dim() == 2 && w.dim() == 2 && x.size(1) == w.size(1));
  TORCH_CHECK(x.is_contiguous() && w.is_contiguous());
  int M = x.size(0), K = x.size(1), N = w.size(0);
  TORCH_CHECK(K % 4 == 0, "K must be a multiple of 4");
  auto y = torch::empty({M, N}, x.options());
  auto st = at::cuda::getCurrentCUDAStream();
  constexpr int MT = SKINNY_MT, NT = SKINNY_NT;
  dim3 grid((M + MT - 1) / MT, (N + NT - 1) / NT);
  sgemm_skinny2_f32_kernel<MT, NT><<<grid, 256, 0, st>>>(x.data_ptr<float>(), w.data_ptr<float>(), y.data_ptr<float>(), N, K, M);
  return y;
}

// ---- shared-expert act: bf16 sg,su -> bf16 silu(clamp_max(sg,10)) * clamp(su,-10,10), torch f32 cadence
__global__ void silu_mul_clamp_bf16_kernel(const __nv_bfloat16* __restrict__ sg, const __nv_bfloat16* __restrict__ su,
                                           __nv_bfloat16* __restrict__ out, int64_t n) {
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float g = __bfloat162float(sg[i]); float u = __bfloat162float(su[i]);
  g = fminf(g, 10.f); u = fminf(fmaxf(u, -10.f), 10.f);
  float s = g / (1.f + expf(-g));
  out[i] = __float2bfloat16(s * u);
}
torch::Tensor silu_mul_clamp_bf16(torch::Tensor sg, torch::Tensor su) {
  TORCH_CHECK(sg.scalar_type() == torch::kBFloat16 && su.scalar_type() == torch::kBFloat16 && sg.is_contiguous() && su.is_contiguous() && sg.numel() == su.numel(), "silu_mul_clamp_bf16: contiguous bf16 same numel");
  auto out = torch::empty_like(sg);
  const int64_t n = sg.numel();
  auto st = at::cuda::getCurrentCUDAStream();
  silu_mul_clamp_bf16_kernel<<<(int)((n + 255) / 256), 256, 0, st>>>(
      (const __nv_bfloat16*)sg.data_ptr(), (const __nv_bfloat16*)su.data_ptr(), (__nv_bfloat16*)out.data_ptr(), n);
  return out;
}
torch::Tensor rms_scale_(torch::Tensor x, double eps) {
  TORCH_CHECK(x.scalar_type() == torch::kBFloat16 && x.is_cuda() && x.is_contiguous(), "rms_scale_: contiguous bf16 cuda");
  const int64_t D = x.size(-1), rows = x.numel() / D;
  if (rows == 0) return x;
  auto st = at::cuda::getCurrentCUDAStream();
  rms_scale_inplace_kernel<<<(int)rows, 128, 0, st>>>(
      reinterpret_cast<__nv_bfloat16*>(x.data_ptr()), (int)D, (float)eps);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return x;
}

// Vectorized qdq (block=BLK): LPB = BLK/8 lanes x 8 bf16 (one uint4) per lane; the per-block
// amax is a shuffle-xor reduction within the LPB-lane group. Reduction is max (order-invariant),
// so results are bitwise identical to the scalar 1-elem/thread kernel above; ~1.9x on A100.
template <int BLK>
__global__ void qdq_vec_kernel(const __nv_bfloat16* __restrict__ x,
                               __nv_bfloat16* __restrict__ y, int64_t nblocks) {
  constexpr int LPB = BLK / 8;
  int64_t gt = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  int64_t b = gt / LPB;
  if (b >= nblocks) return;
  int64_t base = gt * 8;
  uint4 raw = *reinterpret_cast<const uint4*>(x + base);
  const __nv_bfloat16* xv = reinterpret_cast<const __nv_bfloat16*>(&raw);
  float v[8]; float m = 0.f;
#pragma unroll
  for (int k = 0; k < 8; k++) { v[k] = __bfloat162float(xv[k]); m = fmaxf(m, fabsf(v[k])); }
#pragma unroll
  for (int d = LPB / 2; d; d >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, d));
  float amax = fmaxf(m, 1.0e-4f);
  float scale = exp2f(ceilf(log2f(amax / 448.0f)));
  uint4 out; __nv_bfloat16* ov = reinterpret_cast<__nv_bfloat16*>(&out);
#pragma unroll
  for (int k = 0; k < 8; k++) {
    float z = fminf(fmaxf(v[k] / scale, -448.0f), 448.0f);
    __nv_fp8_storage_t q = __nv_cvt_float_to_fp8(z, __NV_SATFINITE, __NV_E4M3);
    __nv_fp8_e4m3 qv; qv.__x = q;
    ov[k] = __float2bfloat16_rn((float)qv * scale);
  }
  *reinterpret_cast<uint4*>(y + base) = out;
}
template <int BLK>
static void qdq_vec_launch(const torch::Tensor& x, torch::Tensor& y, cudaStream_t st) {
  int64_t nb = x.numel() / BLK;
  int64_t threads = nb * (BLK / 8);
  qdq_vec_kernel<BLK><<<(int)((threads + 255) / 256), 256, 0, st>>>(
      reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(y.data_ptr()), nb);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

static torch::Tensor qdq128(torch::Tensor x) {
  auto y = torch::empty_like(x);
  int64_t n = x.numel();
  auto st = at::cuda::getCurrentCUDAStream();
  (void)n; qdq_vec_launch<128>(x, y, st);
  return y;
}

// fp8 Linear leaf (pure): x bf16 [M,K] -> act qdq (block 128, pow2 scale, e4m3 sat)
// -> bf16 GEMM vs pre-dequantized weight w bf16 [N,K] (fp8 * e8m0 scale is exact in bf16).
// Matches kernels_torch.act_quant(block=128, ue8m0) + fp8_gemm (fp32 accumulate).
torch::Tensor fp8_linear(torch::Tensor x, torch::Tensor w) {
  at::cuda::CUDAGuard guard(x.device());
  TORCH_CHECK(x.dim() == 2 && x.scalar_type() == torch::kBFloat16 && x.is_contiguous(),
              "fp8_linear: x must be contiguous bf16 [M,K], got ", x.sizes(), " ", x.scalar_type());
  TORCH_CHECK((x.size(1) & 127) == 0, "fp8_linear: K must be a multiple of 128, got ", x.size(1));
  TORCH_CHECK(w.dim() == 2 && w.scalar_type() == torch::kBFloat16 && w.is_contiguous() &&
              w.size(1) == x.size(1),
              "fp8_linear: w must be contiguous pre-dequantized bf16 [N,K] with K==x.K, got ",
              w.sizes(), " ", w.scalar_type());
  return bf16_gemm(qdq128(x), w);
}

// qdq block=64 variant (kv path uses mxfp8_qdq(x, 64)).
__global__ void qdq64_kernel(const __nv_bfloat16* __restrict__ x,
                             __nv_bfloat16* __restrict__ y, int64_t nblocks) {
  int64_t b = blockIdx.x;
  if (b >= nblocks) return;
  int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  int64_t i = b * 64 + threadIdx.x;
  float v = __bfloat162float(x[i]);
  float m = wgemm_warp_max(fabsf(v));
  __shared__ float wm[2];
  if (lane == 0) wm[warp] = m;
  __syncthreads();
  float amax = fmaxf(fmaxf(wm[0], wm[1]), 1.0e-4f);
  float scale = exp2f(ceilf(log2f(amax / 448.0f)));
  float z = fminf(fmaxf(v / scale, -448.0f), 448.0f);
  __nv_fp8_storage_t q = __nv_cvt_float_to_fp8(z, __NV_SATFINITE, __NV_E4M3);
  __nv_fp8_e4m3 qv; qv.__x = q;
  y[i] = __float2bfloat16_rn((float)qv * scale);
}

static torch::Tensor qdq64(torch::Tensor x) {
  auto y = torch::empty_like(x);
  int64_t n = x.numel();
  auto st = at::cuda::getCurrentCUDAStream();
  (void)n; qdq_vec_launch<64>(x, y, st);
  return y;
}

__global__ void rms_norm_bf16_kernel(const __nv_bfloat16* __restrict__ x,
                                     const __nv_bfloat16* __restrict__ w,
                                     __nv_bfloat16* __restrict__ y,
                                     int D, float eps) {
  const int row = blockIdx.x;
  const __nv_bfloat16* xi = x + (size_t)row * D;
  __nv_bfloat16* yi = y + (size_t)row * D;
  float sum = 0.f;
  for (int i = threadIdx.x; i < D; i += blockDim.x) {
    const float v = __bfloat162float(xi[i]);
    sum += v * v;
  }
  __shared__ float sm[32];
#pragma unroll
  for (int d = 16; d; d >>= 1) sum += __shfl_down_sync(0xffffffffu, sum, d);
  const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
  if (lane == 0) sm[wid] = sum;
  __syncthreads();
  const int nw = (blockDim.x + 31) >> 5;
  if (wid == 0) {
    float tot = (lane < nw) ? sm[lane] : 0.f;
#pragma unroll
    for (int d = 16; d; d >>= 1) tot += __shfl_down_sync(0xffffffffu, tot, d);
    if (lane == 0) sm[0] = tot;
  }
  __syncthreads();
  const float inv = rsqrtf(sm[0] / (float)D + eps);
  for (int i = threadIdx.x; i < D; i += blockDim.x)
    yi[i] = __float2bfloat16(__bfloat162float(xi[i]) * inv * __bfloat162float(w[i]));
}

__global__ void rms_norm_f32w_kernel(const __nv_bfloat16* __restrict__ x,
                                     const float* __restrict__ w,
                                     __nv_bfloat16* __restrict__ y,
                                     int D, float eps) {
  const int row = blockIdx.x;
  const __nv_bfloat16* xi = x + (size_t)row * D;
  __nv_bfloat16* yi = y + (size_t)row * D;
  float sum = 0.f;
  for (int i = threadIdx.x; i < D; i += blockDim.x) {
    const float v = __bfloat162float(xi[i]);
    sum += v * v;
  }
  __shared__ float sm[32];
#pragma unroll
  for (int d = 16; d; d >>= 1) sum += __shfl_down_sync(0xffffffffu, sum, d);
  const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
  if (lane == 0) sm[wid] = sum;
  __syncthreads();
  const int nw = (blockDim.x + 31) >> 5;
  if (wid == 0) {
    float tot = (lane < nw) ? sm[lane] : 0.f;
#pragma unroll
    for (int d = 16; d; d >>= 1) tot += __shfl_down_sync(0xffffffffu, tot, d);
    if (lane == 0) sm[0] = tot;
  }
  __syncthreads();
  const float inv = rsqrtf(sm[0] / (float)D + eps);
  for (int i = threadIdx.x; i < D; i += blockDim.x)
    yi[i] = __float2bfloat16(w[i] * (__bfloat162float(xi[i]) * inv));
}

// Pure leaf: y = (w_f32 * (x_f32 * rsqrt(mean(x^2)+eps))).bf16 -- same as arch.RMSNorm.
torch::Tensor rms_norm(torch::Tensor x, torch::Tensor w, double eps) {
  at::cuda::CUDAGuard guard(x.device());
  TORCH_CHECK(x.scalar_type() == torch::kBFloat16 && x.dim() == 2 && x.is_contiguous(),
              "rms_norm: x must be contiguous bf16 [T,D] (got ", x.scalar_type(), ", dim=", x.dim(), ")");
  TORCH_CHECK(w.scalar_type() == torch::kFloat32 && w.dim() == 1 && w.is_contiguous() && w.size(0) == x.size(1),
              "rms_norm: w must be contiguous fp32 [D] matching x (got ", w.scalar_type(), " ", w.sizes(), ")");
  auto y = torch::empty_like(x);
  const int T = (int)x.size(0), D = (int)x.size(1);
  if (T == 0) return y;
  auto st = at::cuda::getCurrentCUDAStream();
  rms_norm_f32w_kernel<<<T, 256, 0, st>>>(
      reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), w.data_ptr<float>(),
      reinterpret_cast<__nv_bfloat16*>(y.data_ptr()), D, (float)eps);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}

static torch::Tensor rms_norm_at(torch::Tensor x, torch::Tensor w, double eps) {
  if (x.scalar_type() == torch::kBFloat16 && w.scalar_type() == torch::kBFloat16 &&
      x.dim() == 2 && x.is_contiguous() && w.is_contiguous()) {
    auto y = torch::empty_like(x);
    const int T = (int)x.size(0), D = (int)x.size(1);
    auto st = at::cuda::getCurrentCUDAStream();
    rms_norm_bf16_kernel<<<T, 256, 0, st>>>(
        reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(w.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(y.data_ptr()), D, (float)eps);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return y;
  }
  auto x32 = x.to(torch::kFloat32);
  auto var = x32.square().mean(-1, true);
  return (x32 * at::rsqrt(var + eps) * w.to(torch::kFloat32)).to(x.scalar_type());
}

__global__ void rotary_bf16_kernel(__nv_bfloat16* __restrict__ x, long row_stride,
                                   int rows_per_t, const float2* __restrict__ freq,
                                   int inverse) {
  const int row = blockIdx.x;
  const int i = threadIdx.x;  // 0..31
  __nv_bfloat16* xi = x + (size_t)row * row_stride;
  const float2 f = freq[(size_t)(row / rows_per_t) * 32 + i];
  const float fi = inverse ? -f.y : f.y;
  const float a = __bfloat162float(xi[2 * i]);
  const float b = __bfloat162float(xi[2 * i + 1]);
  xi[2 * i]     = __float2bfloat16(a * f.x - b * fi);
  xi[2 * i + 1] = __float2bfloat16(a * fi + b * f.x);
}

__global__ void rope_inplace_kernel(__nv_bfloat16* __restrict__ x,
                                    long sb, long ss, long sh, int S, int H, int half,
                                    const float2* __restrict__ freq, long fs, int inverse) {
  // grid: (B*S*H) rows; block: half threads (one complex pair each)
  const long row = blockIdx.x;
  const int h = (int)(row % H);
  const long bs = row / H;
  const int s = (int)(bs % S);
  const long b = bs / S;
  const int i = threadIdx.x;
  if (i >= half) return;
  __nv_bfloat16* xi = x + b * sb + (long)s * ss + (long)h * sh;
  const float2 f = freq[(size_t)s * fs + i];
  const float fi = inverse ? -f.y : f.y;
  const float a = __bfloat162float(xi[2 * i]);
  const float c = __bfloat162float(xi[2 * i + 1]);
  xi[2 * i]     = __float2bfloat16(a * f.x - c * fi);
  xi[2 * i + 1] = __float2bfloat16(a * fi + c * f.x);
}

// rope_inplace(x, freqs_cis, inverse): rotate x's last dim in place (bf16, fp32 math), returns x.
// x: [B,S,rd] or [B,S,H,rd] view with stride(-1)==1; freqs_cis: complex64 [S, rd/2], any row stride (strided slices ok).
// Pure w.r.t. everything except x's own storage (documented in-place op, mirrors arch.apply_rotary_emb).
torch::Tensor rope_inplace(torch::Tensor x, torch::Tensor freqs, bool inverse) {
  at::cuda::CUDAGuard guard(x.device());
  TORCH_CHECK(x.scalar_type() == torch::kBFloat16, "rope_inplace: x must be bf16, got ", x.scalar_type());
  TORCH_CHECK(x.dim() == 3 || x.dim() == 4, "rope_inplace: x must be [B,S,rd] or [B,S,H,rd], dim=", x.dim());
  TORCH_CHECK(x.stride(-1) == 1, "rope_inplace: x last-dim stride must be 1");
  const int64_t rd = x.size(-1);
  TORCH_CHECK(rd % 2 == 0 && rd <= 2048, "rope_inplace: bad rd=", rd);
  TORCH_CHECK(freqs.scalar_type() == torch::kComplexFloat && freqs.dim() == 2 && freqs.stride(1) == 1,
              "rope_inplace: freqs must be complex64 [S, rd/2] with unit inner stride (row stride free)");
  const int64_t B = x.size(0), S = x.size(1), H = (x.dim() == 4) ? x.size(2) : 1;
  TORCH_CHECK(freqs.size(0) == S && freqs.size(1) == rd / 2,
              "rope_inplace: freqs [", freqs.size(0), ",", freqs.size(1), "] vs x S=", S, " rd/2=", rd / 2);
  const long sb = x.stride(0), ss = x.stride(1), sh = (x.dim() == 4) ? x.stride(2) : 0;
  const int64_t rows = B * S * H;
  if (rows == 0) return x;
  int th = (int)(rd / 2); th = ((th + 31) / 32) * 32;
  auto st = at::cuda::getCurrentCUDAStream();
  rope_inplace_kernel<<<(unsigned)rows, th, 0, st>>>(
      reinterpret_cast<__nv_bfloat16*>(x.data_ptr()), sb, ss, sh, (int)S, (int)H, (int)(rd / 2),
      reinterpret_cast<const float2*>(freqs.data_ptr()), (long)freqs.stride(0), inverse ? 1 : 0);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return x;
}

static void rotary_at(torch::Tensor x, torch::Tensor freq, bool inverse) {
  // x: (..., 64) bf16 slice; freq: (T, 32) complex64 contiguous
  const int64_t rd = x.size(-1);
  bool uniform = (rd == 64) && x.scalar_type() == torch::kBFloat16 &&
                 x.stride(-1) == 1 && freq.scalar_type() == torch::kComplexFloat &&
                 freq.is_contiguous() && freq.size(1) == 32;
  int64_t rs = 0;
  if (uniform && x.dim() == 2) rs = x.stride(0);
  else if (uniform && x.dim() == 3 &&
           x.stride(0) == x.size(1) * x.stride(1)) rs = x.stride(1);
  else uniform = false;
  if (uniform) {
    const int64_t rows = x.numel() / 64;
    const int rows_per_t = (int)(rows / freq.size(0));
    if (rows_per_t > 0 && rows == (int64_t)rows_per_t * freq.size(0)) {
      auto st = at::cuda::getCurrentCUDAStream();
      rotary_bf16_kernel<<<(int)rows, 32, 0, st>>>(
          reinterpret_cast<__nv_bfloat16*>(x.data_ptr()), rs, rows_per_t,
          reinterpret_cast<const float2*>(freq.data_ptr()), inverse ? 1 : 0);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
      return;
    }
  }
  auto shp = x.sizes().vec();
  shp.back() = rd / 2; shp.push_back(2);
  auto z = at::view_as_complex(x.to(torch::kFloat32).reshape(shp).contiguous());
  std::vector<int64_t> fshp; fshp.push_back(freq.size(0));
  for (int64_t d = 1; d + 1 < (int64_t)x.dim(); ++d) fshp.push_back(1);
  fshp.push_back(rd / 2);
  auto f = freq.reshape(fshp);
  if (inverse) f = f.conj();
  z = z * f;
  x.copy_(at::view_as_real(z).flatten(-2).to(x.scalar_type()));
}


// ATen reference path; also builds the per-flag freq table (32 floats).



// ---- declared in sparse_attn_paged.cu / paged_io.cu / compressor_tail.cu ----
at::Tensor sparse_attn_paged(at::Tensor q, at::Tensor pool, at::Tensor page_table,
                             at::Tensor cpool, at::Tensor ctable, at::Tensor sink,
                             at::Tensor idxs, int64_t total, double scale,
                             c10::optional<at::Tensor> keff = c10::nullopt);


void paged_scatter_positions_masked(
    torch::Tensor pool, torch::Tensor page_table, torch::Tensor positions,
    torch::Tensor valid, torch::Tensor src);
torch::Tensor compressor_rows(
    torch::Tensor kvin, torch::Tensor scin, torch::Tensor ape,
    torch::Tensor norm_w, torch::Tensor freqs, torch::Tensor valid,
    int64_t ratio, int64_t d, int64_t rd, double eps,
    bool overlap, bool rotate, bool noq);
torch::Tensor compressor_rows_ring(
    torch::Tensor kvr, torch::Tensor scr, torch::Tensor rowmap, torch::Tensor hasprev,
    torch::Tensor ape, torch::Tensor norm_w, torch::Tensor freqs, torch::Tensor valid,
    int64_t ratio, int64_t d, int64_t rd, double eps,
    bool overlap, bool rotate, bool noq);
torch::Tensor index_score_fused(torch::Tensor q, torch::Tensor pool,
                                torch::Tensor prow, torch::Tensor weights,
                                torch::Tensor positions, int64_t ratio);
torch::Tensor index_score_reduce_positions(
    torch::Tensor score, torch::Tensor weights, int64_t ratio,
    torch::Tensor positions);



// ---- indexer-layer split: p1 runs up to index_score (caller all-reduces it),
// p2 finishes topk + sparse attention + o-proj. State is carried in the
// returned tensors so no host round-trip of the activations is needed.
torch::Tensor index_score_reduce(torch::Tensor score, torch::Tensor weights,
                                 int64_t ratio, int64_t pos0);
torch::Tensor topk_select_post(torch::Tensor score, int64_t K, int64_t ratio,
                               int64_t offset, int64_t pos0);
torch::Tensor topk_select_post_positions(torch::Tensor score, int64_t K,
                                         int64_t ratio, torch::Tensor offsets,
                                         torch::Tensor positions);
torch::Tensor had_fp4_qdq_(torch::Tensor x);

static torch::Tensor hadamard_at(torch::Tensor x, torch::Tensor H, double scale) {
  return (at::matmul(x.to(torch::kFloat32), H) * scale).to(x.scalar_type());
}

__global__ void fp4_qdq_kernel(const __nv_bfloat16* __restrict__ x,
                               const float* __restrict__ lut,
                               const float* __restrict__ bnd,
                               __nv_bfloat16* __restrict__ y,
                               int64_t nblk) {
  // one 32-lane warp per 32-elem quant block; 4 warps per CTA
  const int64_t b = (int64_t)blockIdx.x * 4 + (threadIdx.x >> 5);
  if (b >= nblk) return;
  const int lane = threadIdx.x & 31;
  const int64_t i = b * 32 + lane;
  const float v = __bfloat162float(x[i]);
  float a = fabsf(v);
  #pragma unroll
  for (int o = 16; o > 0; o >>= 1)
    a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, o));
  a = fmaxf(a, 1e-38f);
  // s = ldexp(m==0.5 ? 0.5 : 1.0, e) with frexp(a) = m*2^e, m in [0.5,1)
  int e;
  const float m = frexpf(a, &e);
  const float s = ldexpf(m == 0.5f ? 0.5f : 1.0f, e);
  const float zn = v / s;
  const float az = fabsf(zn);
  // bucketize(right=False): first idx with bnd[idx] >= az
  int idx = 0;
  #pragma unroll
  for (int k = 0; k < 7; ++k) idx += (bnd[k] < az) ? 1 : 0;
  const float q = copysignf(lut[idx], zn);
  y[i] = __float2bfloat16_rn(q * s);
}

static torch::Tensor fp4_qdq(torch::Tensor x, torch::Tensor lut, torch::Tensor bnd,
                             int64_t bs) {
  if (bs == 32 && x.scalar_type() == torch::kBFloat16 && x.is_contiguous() &&
      lut.scalar_type() == torch::kFloat32 && bnd.scalar_type() == torch::kFloat32) {
    auto y = torch::empty_like(x);
    const int64_t nblk = x.numel() / 32;
    auto st = at::cuda::getCurrentCUDAStream();
    fp4_qdq_kernel<<<(int)((nblk + 3) / 4), 128, 0, st>>>(
        reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
        lut.data_ptr<float>(), bnd.data_ptr<float>(),
        reinterpret_cast<__nv_bfloat16*>(y.data_ptr()), nblk);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return y;
  }
  auto shp = x.sizes().vec();
  const int64_t N = shp.back();
  auto z = x.contiguous().reshape({-1, N / bs, bs}).to(torch::kFloat32);
  auto amax = std::get<0>(z.abs().max(-1, true)).clamp_min(1e-38);
  auto fr = at::frexp(amax);
  auto m = std::get<0>(fr), e = std::get<1>(fr);
  auto s = at::ldexp(at::where(m == 0.5, 0.5, 1.0), e);
  auto zn = z / s;
  auto idx = at::bucketize(zn.abs(), bnd);
  auto q = at::copysign(lut.index({idx}), zn);
  return (q * s).to(x.scalar_type()).reshape(shp);
}








// Z1: fused paged gather of the whole indexer pool into a dense [1,NSLOT,D]
// buffer. Replaces the arange/div/remainder/index_select chain (bit-exact copy).
template <typename IdxT>
__global__ void ipool_gather_all_kernel(const IdxT* __restrict__ ctable,
                                        const uint4* __restrict__ pool,
                                        uint4* __restrict__ out, long nslot,
                                        int ipage, int vpr) {
  long v = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (v >= (long)gridDim.y * nslot * (long)vpr) return;
  const int b = blockIdx.y;
  long local = v;
  long slot = local / vpr;
  long page = (long)ctable[(long)b * (nslot / ipage) + slot / ipage];
  out[((long)b * nslot + slot) * vpr + local % vpr] =
      pool[(page * ipage + slot % ipage) * (long)vpr + local % vpr];
}

static torch::Tensor ipool_gather_all(torch::Tensor ctable, torch::Tensor ipool) {
  auto stream = at::cuda::getCurrentCUDAStream();
  const int64_t IPAGE = ipool.size(1), D = ipool.size(-1);
  const int64_t B = ctable.dim() == 2 ? ctable.size(0) : 1;
  const int64_t NP = ctable.numel() / B;
  const int64_t NSLOT = NP * IPAGE;
  const int64_t rowb = D * ipool.element_size();
  TORCH_CHECK(rowb % 16 == 0 && ipool.is_contiguous() && ctable.is_contiguous(),
              "ipool_gather_all: layout");
  auto out = at::empty({B, NSLOT, D}, ipool.options());
  const int vpr = (int)(rowb / 16);
  const long total = NSLOT * vpr;
  const long blocks = (total + 255) / 256;
  dim3 grid(blocks, B);
  if (ctable.scalar_type() == torch::kLong)
    ipool_gather_all_kernel<int64_t><<<grid, 256, 0, stream>>>(
        ctable.data_ptr<int64_t>(), (const uint4*)ipool.data_ptr(),
        (uint4*)out.data_ptr(), NSLOT, (int)IPAGE, vpr);
  else
    ipool_gather_all_kernel<int><<<grid, 256, 0, stream>>>(
        ctable.data_ptr<int>(), (const uint4*)ipool.data_ptr(),
        (uint4*)out.data_ptr(), NSLOT, (int)IPAGE, vpr);
  return out;
}



// fused slot-sum: y[t] = sum_k dn[pos[t*6+k]]  (replaces zeros(T,6,K)+index_put_+6x add)
__global__ void moe_slot_sum_kernel(const __nv_bfloat16* __restrict__ dn,
                                    const int32_t* __restrict__ pos,
                                    __nv_bfloat16* __restrict__ y,
                                    int K) {
  const int t = blockIdx.x;
  const int32_t* pt = pos + (long)t * 6;
  for (int c = threadIdx.x; c < K; c += blockDim.x) {
    float a = 0.f;
    #pragma unroll
    for (int k = 0; k < 6; ++k)
      a += __bfloat162float(dn[(long)pt[k] * K + c]);
    y[(long)t * K + c] = __float2bfloat16(a);
  }
}

// Same as above with the routing weight folded in: y[t] = sum_k bf16(dn[p]*rw[p]).
// The product is rounded to bf16 before the f32 accumulate, so this is
// bit-identical to the former `dn.mul_(rw_bf16)` + slot_sum pair, minus one
// full read+write pass over dn (T*6*K bf16).
__global__ void moe_slot_sum_w_kernel(const __nv_bfloat16* __restrict__ dn,
                                      const __nv_bfloat16* __restrict__ rw,
                                      const int32_t* __restrict__ pos,
                                      __nv_bfloat16* __restrict__ y,
                                      int K) {
  const int t = blockIdx.x;
  const int32_t* pt = pos + (long)t * 6;
  int32_t p[6]; float w[6];
  #pragma unroll
  for (int k = 0; k < 6; ++k) { p[k] = pt[k]; w[k] = __bfloat162float(rw[p[k]]); }
  for (int c = threadIdx.x; c < K; c += blockDim.x) {
    float a = 0.f;
    #pragma unroll
    for (int k = 0; k < 6; ++k) {
      float v = __bfloat162float(dn[(long)p[k] * K + c]) * w[k];
      a += __bfloat162float(__float2bfloat16(v));
    }
    y[(long)t * K + c] = __float2bfloat16(a);
  }
}

// Decode-specialized routed FP4 leaf, following ref's small-T GEMV strategy while
// keeping the DSV4 prefill math and packed FP4 weights.  All routing stays on
// device; grid geometry depends only on captured tensor shapes (T<=8, TOPK=6).
// N: vectorised E2M1 -> bf16x8. The 8 magnitudes of e2m1 are all exactly
// representable in bf16, so PRMT can table-lookup the low/high bytes of the
// bf16 pattern (index = the 3-bit magnitude) and the sign nibble is OR-ed back
// in afterwards. Bit-identical to a per-element e2m1_to_float. Sign bits are
// merged via two extra PRMTs (13 ALU ops / 8 elems vs 20 before; the gu decode
// kernel is ALU-pipe bound at T=32, ncu 83%).
__device__ __forceinline__ void fp4x8_to_bf16x8(uint32_t p, uint32_t o[4]) {
  const uint32_t LO_A = 0xC0800000u, LO_B = 0xC0804000u;
  const uint32_t HI_A = 0x3F3F3F00u, HI_B = 0x40404040u;
  // Sign: nibble e has its sign at bit 4e+3.  sgh keeps the odd-nibble signs
  // at byte msb (bits 7,15,23,31), sgl moves the even-nibble signs there.  One
  // PRMT then interleaves them into the same byte order as the hi LUT word,
  // so a single OR sets bit 15 of each bf16 before the lo/hi interleave.
  const uint32_t mag = p & 0x77777777u;
  const uint32_t sgh = p & 0x80808080u;
  const uint32_t sgl = (p << 4) & 0x80808080u;
  {
    const uint32_t lo = __byte_perm(LO_A, LO_B, mag);          // PRMT uses c[15:0]
    const uint32_t hi = __byte_perm(HI_A, HI_B, mag) | __byte_perm(sgl, sgh, 0x5140u);
    o[0] = __byte_perm(lo, hi, 0x5140u);
    o[1] = __byte_perm(lo, hi, 0x7362u);
  }
  {
    const uint32_t sel = mag >> 16;
    const uint32_t lo = __byte_perm(LO_A, LO_B, sel);
    const uint32_t hi = __byte_perm(HI_A, HI_B, sel) | __byte_perm(sgl, sgh, 0x7362u);
    o[2] = __byte_perm(lo, hi, 0x5140u);
    o[3] = __byte_perm(lo, hi, 0x7362u);
  }
}

__global__ void moe_decode_gu_fp4_kernel(
    const __nv_bfloat16* __restrict__ x,
    const int64_t* __restrict__ ids,
    const uint8_t* __restrict__ w1, const uint8_t* __restrict__ s1,
    const uint8_t* __restrict__ w3, const uint8_t* __restrict__ s3,
    __nv_bfloat16* __restrict__ mid,
    int N, int K, int packed_stride) {
  const int slot = blockIdx.y;
  const int t = slot / 6;
  const int64_t e = ids[slot];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int n = blockIdx.x * (blockDim.x >> 5) + warp;
  if (n >= N) return;
  const __nv_bfloat16* __restrict__ xrow = x + (size_t)t * K;
  const size_t row = (size_t)e * N + n;
  const uint4* w1v = (const uint4*)(w1 + row * packed_stride);
  const uint4* w3v = (const uint4*)(w3 + row * packed_stride);
  const uint8_t* s1r = s1 + row * (K >> 5);
  const uint8_t* s3r = s3 + row * (K >> 5);
  float ag = 0.f, au = 0.f;
  const int nvec = K >> 5;  // uint4 == 32 nibbles == one e8m0 scale block
  for (int i = lane; i < nvec; i += 32) {
    const uint4 q1 = w1v[i], q3 = w3v[i];
    const float f1 = e8m0_scale(s1r[i]);
    const float f3 = e8m0_scale(s3r[i]);
    const uint32_t v1[4] = {q1.x, q1.y, q1.z, q1.w};
    const uint32_t v3[4] = {q3.x, q3.y, q3.z, q3.w};
    const int kb = i << 5;
    // Keep decode's BF16 local arithmetic, but shorten each 16-FMA chain to
    // two independent 8-FMA chains before merging them into the FP32 outer
    // accumulator.  This is the validated GU-only precision/speed tradeoff.
    __nv_bfloat162 bg2a = __float2bfloat162_rn(0.f);
    __nv_bfloat162 bg2b = bg2a;
    __nv_bfloat162 bu2a = bg2a;
    __nv_bfloat162 bu2b = bg2a;
#pragma unroll
    for (int c = 0; c < 4; ++c) {
      const uint4 xw = *(const uint4*)(xrow + kb + (c << 3));
      const __nv_bfloat162* xp2 = (const __nv_bfloat162*)&xw;
      uint32_t og[4], ou[4];
      fp4x8_to_bf16x8(v1[c], og);
      fp4x8_to_bf16x8(v3[c], ou);
#pragma unroll
      for (int j2 = 0; j2 < 4; ++j2) {
        if (c < 2) {
          bg2a = __hfma2(xp2[j2], *(const __nv_bfloat162*)&og[j2], bg2a);
          bu2a = __hfma2(xp2[j2], *(const __nv_bfloat162*)&ou[j2], bu2a);
        } else {
          bg2b = __hfma2(xp2[j2], *(const __nv_bfloat162*)&og[j2], bg2b);
          bu2b = __hfma2(xp2[j2], *(const __nv_bfloat162*)&ou[j2], bu2b);
        }
      }
    }
    const float2 bgfa = __bfloat1622float2(bg2a);
    const float2 bgfb = __bfloat1622float2(bg2b);
    const float2 bufa = __bfloat1622float2(bu2a);
    const float2 bufb = __bfloat1622float2(bu2b);
    ag = fmaf(bgfa.x + bgfa.y + bgfb.x + bgfb.y, f1, ag);
    au = fmaf(bufa.x + bufa.y + bufb.x + bufb.y, f3, au);
  }
#pragma unroll
  for (int d = 16; d; d >>= 1) {
    ag += __shfl_down_sync(0xffffffffu, ag, d);
    au += __shfl_down_sync(0xffffffffu, au, d);
  }
  if (lane == 0) {
    const float gv = fminf(__bfloat162float(__float2bfloat16_rn(ag)), 10.0f);
    const float uv =
        fminf(fmaxf(__bfloat162float(__float2bfloat16_rn(au)), -10.0f), 10.0f);
    mid[(size_t)slot * N + n] =
        __float2bfloat16_rn((gv / (1.0f + __expf(-gv))) * uv);
  }
}

__global__ void moe_decode_down_fp4_kernel(
    const __nv_bfloat16* __restrict__ mid,
    const int64_t* __restrict__ ids,
    const float* __restrict__ wts,
    const uint8_t* __restrict__ w2, const uint8_t* __restrict__ s2,
    __nv_bfloat16* __restrict__ y,
    int Nout, int M, int packed_stride) {
  const int t = blockIdx.y;
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  // V2 (decode 刀15): memory-granularity rewrite. The old kernel gave every lane
  // one 4-byte word per row (a 256-wide fp4 row is exactly 128 B = 32 words), so a
  // warp had at most 6 x 128 B transactions in flight and reached ~31% of HBM BW.
  // Now 8 lanes cover one row with 16 B (uint4) loads each and a warp processes
  // kRows=4 rows *concurrently* (lane>>3 selects the row), so 4 rows x 6 experts =
  // 24 independent 128 B transactions are in flight per warp.
  //
  // Numerics are bit-identical to V1: per-word math is unchanged and the 32-lane
  // shuffle tree is replayed over "virtual lanes" v = 4*L + j (L = lane&7, j =
  // word within the uint4). Tree steps d=16/8/4 pair v with v+d = same j, lane
  // L+4/L+2/L+1 (shfl_down by 4/2/1); steps d=2/1 stay inside the register
  // quadruple (j += j+2, j0 += j1). Same pairs, same order => same bits.
  constexpr int kRows = 4;
  const int rsel = lane >> 3;   // row within the warp's 4-row group
  const int L = lane & 7;       // lane within the 8-lane row group
  const int n_base = (blockIdx.x * (blockDim.x >> 5) + warp) * kRows;
  extern __shared__ __nv_bfloat16 ms[];
  for (int i = threadIdx.x; i < 6 * M; i += blockDim.x)
    ms[i] = mid[(size_t)t * 6 * M + i];
  __syncthreads();
  if (n_base >= Nout) return;
  const int n = n_base + rsel;
  const bool nvalid = n < Nout;
  const int nvec4 = M >> 5;     // uint4 (8 words = 64 nibbles... 32 fp4 values) per row
  float acc[6][4];
#pragma unroll
  for (int ktop = 0; ktop < 6; ++ktop) {
    const size_t row = (size_t)ids[t * 6 + ktop] * Nout + (nvalid ? n : (Nout - 1));
    const uint4* wv4 = (const uint4*)(w2 + row * packed_stride);
    const uint8_t* sr = s2 + row * (M >> 5);
    float a[4] = {0.f, 0.f, 0.f, 0.f};
    for (int q = L; q < nvec4; q += 8) {
      const uint4 p4 = wv4[q];
      // word index i = 4q + j  ->  scale byte index i>>2 == q for all j
      const float sf = e8m0_scale(sr[q]);
      const uint32_t pw[4] = {p4.x, p4.y, p4.z, p4.w};
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const int i = (q << 2) + j;
        const int mb = i << 3;
        uint32_t ow[4];
        fp4x8_to_bf16x8(pw[j], ow);
        __nv_bfloat162 hb = __float2bfloat162_rn(0.f);
        const __nv_bfloat162* ms2 = (const __nv_bfloat162*)(ms + ktop * M + mb);
#pragma unroll
        for (int jj = 0; jj < 4; ++jj)
          hb = __hfma2(ms2[jj], *(const __nv_bfloat162*)&ow[jj], hb);
        const float2 bf = __bfloat1622float2(hb);
        a[j] = fmaf(bf.x + bf.y, sf, a[j]);
      }
    }
#pragma unroll
    for (int j = 0; j < 4; ++j) acc[ktop][j] = a[j];
  }
  // virtual-lane tree: d=16,8,4 -> shfl_down 4,2,1 (same j)
#pragma unroll
  for (int d = 4; d; d >>= 1)
#pragma unroll
    for (int ktop = 0; ktop < 6; ++ktop)
#pragma unroll
      for (int j = 0; j < 4; ++j)
        acc[ktop][j] += __shfl_down_sync(0xffffffffu, acc[ktop][j], d);
  if (L == 0 && nvalid) {
    float total = 0.f;
#pragma unroll
    for (int ktop = 0; ktop < 6; ++ktop) {
      // d=2: j += j+2 ; d=1: j0 += j1
      const float v0 = acc[ktop][0] + acc[ktop][2];
      const float v1 = acc[ktop][1] + acc[ktop][3];
      const float dn = __bfloat162float(__float2bfloat16_rn(v0 + v1));
      const float rw =
          __bfloat162float(__float2bfloat16_rn(wts[t * 6 + ktop]));
      total += __bfloat162float(__float2bfloat16_rn(dn * rw));
    }
    y[(size_t)t * Nout + n] = __float2bfloat16_rn(total);
  }
}

static torch::Tensor moe_rank_decode_fp4(
    torch::Tensor xq, torch::Tensor ids, torch::Tensor wts,
    torch::Tensor w1w, torch::Tensor w1s,
    torch::Tensor w3w, torch::Tensor w3s,
    torch::Tensor w2w, torch::Tensor w2s) {
  const int64_t T = xq.size(0), K = xq.size(1), Nff = w1w.size(1);
  const int64_t Nout = w2w.size(1), nsel = T * 6;
  auto mid = torch::empty({nsel, Nff}, xq.options().dtype(torch::kBFloat16));
  auto y = torch::empty({T, Nout}, xq.options().dtype(torch::kBFloat16));
  auto stream = at::cuda::getCurrentCUDAStream();
  constexpr int kWarps = 8;
  moe_decode_gu_fp4_kernel<<<dim3((unsigned)((Nff + kWarps - 1) / kWarps), (unsigned)nsel),
                             kWarps * 32, 0, stream>>>(
      (const __nv_bfloat16*)xq.data_ptr(), ids.data_ptr<int64_t>(),
      (const uint8_t*)w1w.data_ptr(), (const uint8_t*)w1s.data_ptr(),
      (const uint8_t*)w3w.data_ptr(), (const uint8_t*)w3s.data_ptr(),
      (__nv_bfloat16*)mid.data_ptr(), (int)Nff, (int)K, (int)w1w.size(2));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  constexpr int kDownRows = 4;  // must match kRows in the down kernel
  moe_decode_down_fp4_kernel<<<dim3((unsigned)((Nout + kWarps * kDownRows - 1) / (kWarps * kDownRows)), (unsigned)T),
                               kWarps * 32, (size_t)(6 * Nff) * sizeof(__nv_bfloat16), stream>>>(
      (const __nv_bfloat16*)mid.data_ptr(), ids.data_ptr<int64_t>(),
      wts.data_ptr<float>(), (const uint8_t*)w2w.data_ptr(),
      (const uint8_t*)w2s.data_ptr(), (__nv_bfloat16*)y.data_ptr(),
      (int)Nout, (int)Nff, (int)w2w.size(2));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}

// Fused routed-expert loop: single C++ call replaying model/prefill.py moe_rank
// numerics exactly (grouped per expert, bf16 top-order accumulation outside).
// xq: (T,K) bf16 qdq'd activations; ids/wts: (T,6); w*: [E,...] fp4 banks.
// grouped GEMM (compiled from prefill_moe_cutlass_gemm.cu in same extension)
torch::Tensor grouped_gemm_sm80(torch::Tensor x, torch::Tensor w, torch::Tensor sizes, int64_t threadblock_count);

// Single-call grouped MoE pipeline: CSR + gather + batch fp4 dequant + 3 grouped
// GEMMs + act/qdq cadence + scatter. Numerics identical to the python pipeline.
static torch::Tensor moe_rank_grouped_prefill_fp4(
    torch::Tensor xq, torch::Tensor ids, torch::Tensor wts,
    torch::Tensor w1w, torch::Tensor w1s,
    torch::Tensor w3w, torch::Tensor w3s,
    torch::Tensor w2w, torch::Tensor w2s) {
  at::cuda::CUDAGuard guard(xq.device());
  const int64_t T = xq.size(0), K = xq.size(1);
  const int64_t E = w1w.size(0);
  auto stream0 = at::cuda::getCurrentCUDAStream();
  auto iopt = torch::TensorOptions().dtype(torch::kInt32).device(xq.device());
  auto lopt2 = torch::TensorOptions().dtype(torch::kInt64).device(xq.device());
  const int64_t nsel = T * 6;
  auto ids_i = ids.contiguous();
  auto cnt_d = torch::zeros({E}, iopt);
  auto off_d = torch::empty({E + 1}, iopt);
  auto posc_d = torch::empty({E}, iopt);
  auto toks = torch::empty({nsel}, lopt2);
  auto tks = torch::empty({nsel}, lopt2);
  auto slot_d = torch::empty({nsel}, iopt);
  dsv4_moe_count256<<<(int)((nsel + 255) / 256), 256, 0, stream0>>>(
      ids_i.data_ptr<int64_t>(), (int)nsel, cnt_d.data_ptr<int>());
  dsv4_moe_scan256<<<1, 256, 0, stream0>>>(
      cnt_d.data_ptr<int>(), off_d.data_ptr<int>(), posc_d.data_ptr<int>());
  dsv4_moe_scatter256<<<(int)((nsel + 255) / 256), 256, 0, stream0>>>(
      ids_i.data_ptr<int64_t>(), (int)nsel, posc_d.data_ptr<int>(),
      toks.data_ptr<int64_t>(), tks.data_ptr<int64_t>(), slot_d.data_ptr<int>());
  auto xin = xq.index_select(0, toks).to(torch::kBFloat16);
  auto rw = wts.index_select(0, toks).gather(1, tks.unsqueeze(1)).to(torch::kFloat32);

  // single D2H per layer (port of GLM dispatch_meta's off.to(kCPU)); act/sizes
  // are then built host-side so no further GPU sync happens downstream.
  auto off_h = off_d.to(torch::kCPU);
  const int* offp = off_h.data_ptr<int>();
  std::vector<int64_t> active, sizes_host;
  active.reserve(E); sizes_host.reserve(E);
  for (int64_t e = 0; e < E; ++e) {
    int64_t c = offp[e + 1] - offp[e];
    if (c <= 0) continue;
    active.push_back(e); sizes_host.push_back(c);
  }
  const int64_t na = (int64_t)active.size();
  if (na == 0) return torch::zeros({T, K}, xq.options().dtype(torch::kBFloat16));
  auto lo = torch::TensorOptions().dtype(torch::kInt64);
  auto act = torch::from_blob(active.data(), {na}, lo).clone().to(xq.device());
  auto sizes = torch::from_blob(sizes_host.data(), {na}, lo).clone().to(xq.device());

  const int64_t Nff = w1w.size(1);
  auto w1d = dequant_fp4_selected_bf16(w1w, w1s, act, K).reshape({na, Nff, K});
  auto g = grouped_gemm_sm80(xin, w1d, sizes, 0);
  w1d.reset();
  auto w3d = dequant_fp4_selected_bf16(w3w, w3s, act, K).reshape({na, Nff, K});
  auto u = grouped_gemm_sm80(xin, w3d, sizes, 0);
  w3d.reset(); xin.reset();
  auto mid = (at::silu(g.clamp_max(10.0)) * u.clamp(-10.0, 10.0)).contiguous();
  g.reset(); u.reset();
  auto w2d = dequant_fp4_selected_bf16(w2w, w2s, act, Nff).reshape({na, K, Nff});
  auto dn = grouped_gemm_sm80(mid, w2d, sizes, 0);
  mid.reset(); w2d.reset();
  auto pos = slot_d;
  auto y = torch::empty({T, K}, xq.options().dtype(torch::kBFloat16));
  auto rwb = rw.to(torch::kBFloat16).contiguous();  // [nsel,1] -> same rounding as the old mul_
  auto stream = at::cuda::getCurrentCUDAStream();
  moe_slot_sum_w_kernel<<<(int)T, 256, 0, stream>>>(
      (const __nv_bfloat16*)dn.data_ptr(), (const __nv_bfloat16*)rwb.data_ptr(),
      pos.data_ptr<int32_t>(), (__nv_bfloat16*)y.data_ptr(), (int)K);
  return y;
}







// Whole moe_rank in one C++ call: hc_pre + rms_norm + route + qdq128 +
// grouped routed experts + shared expert FFN. Numerics mirror model/prefill.py.

// row rsqrt over hc*dim (bf16 in), no fp32 materialization
__global__ void hc_rsqrt_kernel(const __nv_bfloat16* __restrict__ x,
                                float* __restrict__ out, int N, float eps) {
  extern __shared__ float sm[];
  const long base = (long)blockIdx.x * N;
  float a = 0.f;
  for (int i = threadIdx.x; i < N; i += blockDim.x) {
    float v = __bfloat162float(x[base + i]);
    a += v * v;
  }
  sm[threadIdx.x] = a;
  __syncthreads();
  for (int st = blockDim.x >> 1; st; st >>= 1) {
    if (threadIdx.x < st) sm[threadIdx.x] += sm[threadIdx.x + st];
    __syncthreads();
  }
  if (threadIdx.x == 0) out[blockIdx.x] = rsqrtf(sm[0] / N + eps);
}


// U: fused hc_pre = rsqrt + pre-gate + post/sinkhorn + mix + rmsnorm, one launch.
// Numerics bit-identical to the 3-kernel path (same reduction trees, same scalar code).
template <int HC>
__global__ void hc_fused_pre_kernel(const __nv_bfloat16* __restrict__ x4,
                                    const __nv_bfloat16* __restrict__ mixes,
                                    const float* __restrict__ hs,
                                    const float* __restrict__ hb,
                                    const __nv_bfloat16* __restrict__ nw,
                                    float* __restrict__ post,
                                    float* __restrict__ comb,
                                    __nv_bfloat16* __restrict__ y,
                                    __nv_bfloat16* __restrict__ xraw,
                                    int dim, float eps) {
  extern __shared__ float sm[];
  float* pre = sm;             // HC
  float* red = sm + HC;        // blockDim.x
  __shared__ float s_rv;
  __shared__ float s_m[HC * HC];
  const int t = blockIdx.x;
  const int N = HC * dim;
  const long base = (long)t * N;
  // pass1: row rsqrt (bit-same as hc_rsqrt_kernel)
  float a = 0.f;
  const bool vec8 = ((dim & 7) == 0);
  if (vec8) {
    const float4* xv = reinterpret_cast<const float4*>(x4 + base);
    const int N8 = N >> 3;
    for (int i = threadIdx.x; i < N8; i += blockDim.x) {
      float4 q = xv[i];
      const __nv_bfloat16* hh = reinterpret_cast<const __nv_bfloat16*>(&q);
#pragma unroll
      for (int j = 0; j < 8; ++j) { float v = __bfloat162float(hh[j]); a += v * v; }
    }
  } else {
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
      float v = __bfloat162float(x4[base + i]);
      a += v * v;
    }
  }
  red[threadIdx.x] = a;
  __syncthreads();
  for (int st = blockDim.x >> 1; st; st >>= 1) {
    if (threadIdx.x < st) red[threadIdx.x] += red[threadIdx.x + st];
    __syncthreads();
  }
  if (threadIdx.x == 0) s_rv = rsqrtf(red[0] / N + eps);
  __syncthreads();
  const float rv = s_rv;
  const __nv_bfloat16* src = mixes + (long)t * (2 * HC + HC * HC);
  if (threadIdx.x < HC) {
    float m = __bfloat162float(src[threadIdx.x]) * rv;
    pre[threadIdx.x] = 1.f / (1.f + __expf(-(m * hs[0] + hb[threadIdx.x]))) + eps;
  }
  if (threadIdx.x == blockDim.x - 1) {
    // post + sinkhorn (bit-same as hc_post_sinkhorn_kernel), one lane
    const float hs1 = hs[1], hs2 = hs[2];
#pragma unroll
    for (int i = 0; i < HC; ++i)
      post[(long)t * HC + i] =
          2.f / (1.f + expf(-(__bfloat162float(src[HC + i]) * rv * hs1 + hb[HC + i])));
    float m[HC * HC];
#pragma unroll
    for (int i = 0; i < HC * HC; ++i)
      m[i] = __bfloat162float(src[2 * HC + i]) * rv * hs2 + hb[2 * HC + i];
#pragma unroll
    for (int r = 0; r < HC; ++r) {
      float mx = m[r * HC];
#pragma unroll
      for (int c = 1; c < HC; ++c) mx = fmaxf(mx, m[r * HC + c]);
      float s = 0.f;
#pragma unroll
      for (int c = 0; c < HC; ++c) { m[r * HC + c] = __expf(m[r * HC + c] - mx); s += m[r * HC + c]; }
#pragma unroll
      for (int c = 0; c < HC; ++c) m[r * HC + c] = m[r * HC + c] / s + eps;
    }
#pragma unroll
    for (int c = 0; c < HC; ++c) {
      float s = 0.f;
#pragma unroll
      for (int r = 0; r < HC; ++r) s += m[r * HC + c];
      s += eps;
#pragma unroll
      for (int r = 0; r < HC; ++r) m[r * HC + c] /= s;
    }
#pragma unroll
    for (int i = 0; i < HC * HC; ++i) s_m[i] = m[i];
  }
  __syncthreads();
  // sinkhorn 19 iters, one lane per matrix element (bit-same accumulation order)
  if (threadIdx.x < HC * HC) {
    const int rr = threadIdx.x / HC, cc = threadIdx.x % HC;
    const unsigned msk = (unsigned)((1ull << (HC * HC)) - 1ull);
    float mv = s_m[threadIdx.x];
    for (int it = 0; it < 19; ++it) {
      float s = 0.f;
#pragma unroll
      for (int j = 0; j < HC; ++j) s += __shfl_sync(msk, mv, rr * HC + j);
      s += eps;
      mv /= s;
      float s2 = 0.f;
#pragma unroll
      for (int j = 0; j < HC; ++j) s2 += __shfl_sync(msk, mv, j * HC + cc);
      s2 += eps;
      mv /= s2;
    }
    comb[(long)t * HC * HC + threadIdx.x] = mv;
  }
  __syncthreads();
  // pass2: mix + rmsnorm (bit-same as hc_mix_norm_kernel)
  float acc = 0.f;
  if (vec8) {
    const int dim8 = dim >> 3;
    for (int cb = threadIdx.x; cb < dim8; cb += blockDim.x) {
      float4 q[HC];
#pragma unroll
      for (int h = 0; h < HC; ++h)
        q[h] = reinterpret_cast<const float4*>(x4 + base + (long)h * dim)[cb];
      __nv_bfloat16 ob[8];
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        float v = 0.f;
#pragma unroll
        for (int h = 0; h < HC; ++h)
          v += pre[h] * __bfloat162float(reinterpret_cast<const __nv_bfloat16*>(&q[h])[j]);
        acc += v * v;
        ob[j] = __float2bfloat16(v);
      }
      const float4 ov = *reinterpret_cast<const float4*>(ob);
      reinterpret_cast<float4*>(y + (long)t * dim)[cb] = ov;
      if (xraw) reinterpret_cast<float4*>(xraw + (long)t * dim)[cb] = ov;
    }
  } else {
    for (int c = threadIdx.x; c < dim; c += blockDim.x) {
      float v = 0.f;
#pragma unroll
      for (int h = 0; h < HC; ++h)
        v += pre[h] * __bfloat162float(x4[base + (long)h * dim + c]);
      acc += v * v;
      const __nv_bfloat16 vb = __float2bfloat16(v);
      y[(long)t * dim + c] = vb;
      if (xraw) xraw[(long)t * dim + c] = vb;
    }
  }
  red[threadIdx.x] = acc;
  __syncthreads();
  for (int st = blockDim.x >> 1; st; st >>= 1) {
    if (threadIdx.x < st) red[threadIdx.x] += red[threadIdx.x + st];
    __syncthreads();
  }
  const float r = rsqrtf(red[0] / dim + eps);
  if (vec8) {
    const int dim8 = dim >> 3;
    for (int cb = threadIdx.x; cb < dim8; cb += blockDim.x) {
      float4 qy = reinterpret_cast<float4*>(y + (long)t * dim)[cb];
      const float4 qw = reinterpret_cast<const float4*>(nw)[cb];
      __nv_bfloat16* yh = reinterpret_cast<__nv_bfloat16*>(&qy);
      const __nv_bfloat16* wh = reinterpret_cast<const __nv_bfloat16*>(&qw);
#pragma unroll
      for (int j = 0; j < 8; ++j)
        yh[j] = __float2bfloat16(__bfloat162float(yh[j]) * r * __bfloat162float(wh[j]));
      reinterpret_cast<float4*>(y + (long)t * dim)[cb] = qy;
    }
  } else {
    for (int c = threadIdx.x; c < dim; c += blockDim.x) {
      float v = __bfloat162float(y[(long)t * dim + c]) * r
                * __bfloat162float(nw[c]);
      y[(long)t * dim + c] = __float2bfloat16(v);
    }
  }
}

// V2: hc_fused_pre split into (a) 256-thread main kernel with the 1024-lane
// reduction tree emulated (bit-same pairing) and y kept in registers, and
// (b) a tiny per-warp route kernel (post + sinkhorn) off the critical path.
// Numerics bit-identical to hc_fused_pre_kernel<4>.
#define HCP_THREADS 256

// Emulates the 1024-entry tree `for st=512..1: red[i]+=red[i+st]` with 256
// threads: levels 512..32 via smem, levels 16..1 inside warp 0 with shfl_down
// (same operand pairing, same order). Returns red[0] to all threads.
__device__ __forceinline__ float hcp_tree1024(float* red) {
  const int tid = threadIdx.x;
  __syncthreads();
#pragma unroll
  for (int st = 512; st >= 32; st >>= 1) {
    for (int i = tid; i < st; i += HCP_THREADS) red[i] += red[i + st];
    __syncthreads();
  }
  if (tid < 32) {
    float v = red[tid];
#pragma unroll
    for (int st = 16; st >= 1; st >>= 1) {
      const float o = __shfl_down_sync(0xffffffffu, v, st);
      if (tid < st) v += o;
    }
    if (tid == 0) red[0] = v;
  }
  __syncthreads();
  return red[0];
}

// main: pass1 rsqrt + pre + pass2 mix + rmsnorm. dim must be 4096 (dim8=512 ->
// 2 float4 per thread), checked on host. grid=T, block=256.
template <int HC>
__global__ void __launch_bounds__(HCP_THREADS)
hc_pre_main_kernel(const __nv_bfloat16* __restrict__ x4,
                   const __nv_bfloat16* __restrict__ mixes,
                   const float* __restrict__ hs,
                   const float* __restrict__ hb,
                   const __nv_bfloat16* __restrict__ nw,
                   float* __restrict__ rv_out,
                   __nv_bfloat16* __restrict__ y,
                   __nv_bfloat16* __restrict__ xraw,
                   int dim, float eps) {
  __shared__ float red[1024];
  __shared__ float pre[HC];
  const int tid = threadIdx.x;
  const int t = blockIdx.x;
  const int N = HC * dim;
  const long base = (long)t * N;
  const float4* xv = reinterpret_cast<const float4*>(x4 + base);
  const int N8 = N >> 3;  // 2048
  // pass1: virtual lane v = tid + 256*k accumulates i = v, v+1024, ...
#pragma unroll
  for (int k = 0; k < 4; ++k) {
    const int v = tid + HCP_THREADS * k;
    float a = 0.f;
    for (int i = v; i < N8; i += 1024) {
      float4 q = xv[i];
      const __nv_bfloat16* hh = reinterpret_cast<const __nv_bfloat16*>(&q);
#pragma unroll
      for (int j = 0; j < 8; ++j) { float vv = __bfloat162float(hh[j]); a += vv * vv; }
    }
    red[v] = a;
  }
  const float tot = hcp_tree1024(red);
  const float rv = rsqrtf(tot / N + eps);
  if (tid == 0) rv_out[t] = rv;
  const __nv_bfloat16* src = mixes + (long)t * (2 * HC + HC * HC);
  if (tid < HC) {
    float m = __bfloat162float(src[tid]) * rv;
    pre[tid] = 1.f / (1.f + __expf(-(m * hs[0] + hb[tid]))) + eps;
  }
  __syncthreads();
  // pass2: mix; virtual lanes v = tid (k=0), tid+256 (k=1) cover cb < 512.
  float4 yv[2];
#pragma unroll
  for (int k = 0; k < 2; ++k) {
    const int cb = tid + HCP_THREADS * k;
    float acc = 0.f;
    float4 q[HC];
#pragma unroll
    for (int h = 0; h < HC; ++h)
      q[h] = reinterpret_cast<const float4*>(x4 + base + (long)h * dim)[cb];
    __nv_bfloat16 ob[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      float v = 0.f;
#pragma unroll
      for (int h = 0; h < HC; ++h)
        v += pre[h] * __bfloat162float(reinterpret_cast<const __nv_bfloat16*>(&q[h])[j]);
      acc += v * v;
      ob[j] = __float2bfloat16(v);
    }
    yv[k] = *reinterpret_cast<const float4*>(ob);
    if (xraw) reinterpret_cast<float4*>(xraw + (long)t * dim)[cb] = yv[k];
    red[cb] = acc;
  }
  red[tid + 512] = 0.f;
  red[tid + 768] = 0.f;
  const float tot2 = hcp_tree1024(red);
  const float r = rsqrtf(tot2 / dim + eps);
#pragma unroll
  for (int k = 0; k < 2; ++k) {
    const int cb = tid + HCP_THREADS * k;
    float4 qy = yv[k];
    const float4 qw = reinterpret_cast<const float4*>(nw)[cb];
    __nv_bfloat16* yh = reinterpret_cast<__nv_bfloat16*>(&qy);
    const __nv_bfloat16* wh = reinterpret_cast<const __nv_bfloat16*>(&qw);
#pragma unroll
    for (int j = 0; j < 8; ++j)
      yh[j] = __float2bfloat16(__bfloat162float(yh[j]) * r * __bfloat162float(wh[j]));
    reinterpret_cast<float4*>(y + (long)t * dim)[cb] = qy;
  }
}

// route: post + sinkhorn, one 16-lane half-warp per token (2 tokens/warp),
// 8 tokens per 128-thread block. Bit-same serial prelude on lane 0 of each half.
template <int HC>
__global__ void __launch_bounds__(128)
hc_route_kernel(const __nv_bfloat16* __restrict__ mixes,
                const float* __restrict__ hs,
                const float* __restrict__ hb,
                const float* __restrict__ rv_in,
                float* __restrict__ post,
                float* __restrict__ comb,
                int T, float eps) {
  static_assert(HC * HC == 16, "HC must be 4");
  const int half = threadIdx.x >> 4;          // 0..7 within block
  const int l = threadIdx.x & 15;             // 0..15
  const int t = blockIdx.x * 8 + half;
  const unsigned msk = (threadIdx.x & 16) ? 0xffff0000u : 0x0000ffffu;
  if (t >= T) return;                          // whole half-warp exits together
  const float rv = rv_in[t];
  const __nv_bfloat16* src = mixes + (long)t * (2 * HC + HC * HC);
  float mv;
  if (l == 0) {
    const float hs1 = hs[1];
#pragma unroll
    for (int i = 0; i < HC; ++i)
      post[(long)t * HC + i] =
          2.f / (1.f + expf(-(__bfloat162float(src[HC + i]) * rv * hs1 + hb[HC + i])));
  }
  {
    // serial prelude replicated on every lane (identical scalar code -> identical result)
    const float hs2 = hs[2];
    float m[HC * HC];
#pragma unroll
    for (int i = 0; i < HC * HC; ++i)
      m[i] = __bfloat162float(src[2 * HC + i]) * rv * hs2 + hb[2 * HC + i];
#pragma unroll
    for (int r = 0; r < HC; ++r) {
      float mx = m[r * HC];
#pragma unroll
      for (int c = 1; c < HC; ++c) mx = fmaxf(mx, m[r * HC + c]);
      float s = 0.f;
#pragma unroll
      for (int c = 0; c < HC; ++c) { m[r * HC + c] = __expf(m[r * HC + c] - mx); s += m[r * HC + c]; }
#pragma unroll
      for (int c = 0; c < HC; ++c) m[r * HC + c] = m[r * HC + c] / s + eps;
    }
#pragma unroll
    for (int c = 0; c < HC; ++c) {
      float s = 0.f;
#pragma unroll
      for (int r = 0; r < HC; ++r) s += m[r * HC + c];
      s += eps;
#pragma unroll
      for (int r = 0; r < HC; ++r) m[r * HC + c] /= s;
    }
    mv = m[l];
  }
  const int rr = l / HC, cc = l % HC;
  const int lb = threadIdx.x & 16;  // lane base of this half
  for (int it = 0; it < 19; ++it) {
    float s = 0.f;
#pragma unroll
    for (int j = 0; j < HC; ++j) s += __shfl_sync(msk, mv, lb + rr * HC + j);
    s += eps;
    mv /= s;
    float s2 = 0.f;
#pragma unroll
    for (int j = 0; j < HC; ++j) s2 += __shfl_sync(msk, mv, lb + j * HC + cc);
    s2 += eps;
    mv /= s2;
  }
  comb[(long)t * HC * HC + l] = mv;
}


// ---- hc_mix small-N GEMV (decode): mixes[T,N] = x[T,K] . fn[N,K]^T, bf16 in/out, fp32 acc.
// Replaces cuBLAS splitK (2 kernels + non-deterministic order) for T<=HCM_MAXT.
// grid (N, HCM_KS), 256 threads; each block: one n, one K-chunk, all T rows.
// Reduction order fixed (thread-sequential -> warp shfl tree -> warp order -> split order)
// => deterministic (run-to-run bit-same), though not bit-same with cuBLAS.
#define HCM_KS 8
#define HCM_MAXT 64
__global__ void __launch_bounds__(256) hc_mix_gemv_kernel(
    const __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ w,
    float* __restrict__ part, unsigned int* __restrict__ cnt,
    __nv_bfloat16* __restrict__ y, int T, int N, int K) {
  const int n = blockIdx.x, ks = blockIdx.y;
  const int kc = K / HCM_KS;              // chunk len (K % (HCM_KS*256*8) == 0 checked host-side)
  const int per = kc / 256;               // elems per thread (multiple of 8)
  const int k0 = ks * kc + threadIdx.x * per;
  __shared__ float red[8][HCM_MAXT];
  __shared__ bool last;
  // load w slice once
  float wv[64];
#pragma unroll
  for (int i = 0; i < 64; i += 8) {
    if (i < per) {
      uint4 u = *reinterpret_cast<const uint4*>(w + (size_t)n * K + k0 + i);
      const __nv_bfloat162* p = reinterpret_cast<const __nv_bfloat162*>(&u);
#pragma unroll
      for (int j = 0; j < 4; ++j) { float2 f = __bfloat1622float2(p[j]); wv[i + 2*j] = f.x; wv[i + 2*j + 1] = f.y; }
    }
  }
  const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
  for (int t = 0; t < T; ++t) {
    float acc = 0.f;
#pragma unroll
    for (int i = 0; i < 64; i += 8) {
      if (i < per) {
        uint4 u = *reinterpret_cast<const uint4*>(x + (size_t)t * K + k0 + i);
        const __nv_bfloat162* p = reinterpret_cast<const __nv_bfloat162*>(&u);
#pragma unroll
        for (int j = 0; j < 4; ++j) { float2 f = __bfloat1622float2(p[j]); acc += f.x * wv[i + 2*j]; acc += f.y * wv[i + 2*j + 1]; }
      }
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
    if (lane == 0) red[wid][t] = acc;
  }
  __syncthreads();
  if (threadIdx.x < T) {
    float s = 0.f;
#pragma unroll
    for (int w8 = 0; w8 < 8; ++w8) s += red[w8][threadIdx.x];
    part[((size_t)ks * T + threadIdx.x) * N + n] = s;
  }
  __threadfence();
  __syncthreads();
  if (threadIdx.x == 0) {
    unsigned int prev = atomicAdd(&cnt[n], 1u);
    last = (prev == HCM_KS - 1);
  }
  __syncthreads();
  if (last) {
    __threadfence();
    if (threadIdx.x < T) {
      float s = 0.f;
#pragma unroll
      for (int k = 0; k < HCM_KS; ++k) s += __ldcg(&part[((size_t)k * T + threadIdx.x) * N + n]);
      y[(size_t)threadIdx.x * N + n] = __float2bfloat16_rn(s);
    }
    if (threadIdx.x == 0) cnt[n] = 0u;  // reset for next call (same stream)
  }
}

static torch::Tensor hc_mix_gemv(torch::Tensor x, torch::Tensor w, cudaStream_t stream) {
  const int64_t T = x.size(0), K = x.size(1), N = w.size(0);
  auto y = at::empty({T, N}, x.options());
  auto part = at::empty({HCM_KS, T, N}, x.options().dtype(torch::kFloat32));
  auto cnt = at::zeros({N}, x.options().dtype(torch::kInt32));
  dim3 grid((int)N, HCM_KS);
  hc_mix_gemv_kernel<<<grid, 256, 0, stream>>>(
      (const __nv_bfloat16*)x.data_ptr(), (const __nv_bfloat16*)w.data_ptr(),
      part.data_ptr<float>(), (unsigned int*)cnt.data_ptr<int>(),
      (__nv_bfloat16*)y.data_ptr(), (int)T, (int)N, (int)K);
  return y;
}

static std::vector<torch::Tensor> hc_fused_pre(
    torch::Tensor x4, torch::Tensor fn, torch::Tensor hs, torch::Tensor hb,
    torch::Tensor norm_w, double eps, bool want_xraw) {
  const int64_t T = x4.size(0), hc = x4.size(1), dim = x4.size(2);
  TORCH_CHECK(hc == 4, "hc_fused_pre requires hc==4");
  auto stream0 = at::cuda::getCurrentCUDAStream();
  auto x4c = x4.contiguous();
  auto fnb = get_or_cast(fn, torch::kBFloat16).contiguous();
  const int64_t Kmix = hc * dim;
  torch::Tensor mixes;
  if (T <= HCM_MAXT && Kmix % (HCM_KS * 256 * 8) == 0 && Kmix / HCM_KS / 256 <= 64 &&
      fn.size(0) <= 1024) {
    mixes = hc_mix_gemv(x4c.reshape({T, Kmix}), fnb, stream0);
  } else {
    mixes = at::linear(x4c.reshape({T, Kmix}), fnb).contiguous();
  }
  auto hsf = get_or_cast(hs, torch::kFloat32);
  auto hbf = get_or_cast(hb, torch::kFloat32);
  auto nwb = get_or_cast(norm_w, torch::kBFloat16);
  auto post = at::empty({T, hc}, x4.options().dtype(torch::kFloat32));
  auto comb = at::empty({T, hc, hc}, x4.options().dtype(torch::kFloat32));
  auto x = at::empty({T, dim}, x4.options().dtype(torch::kBFloat16));
  auto xraw = want_xraw
                  ? at::empty({T, dim}, x4.options().dtype(torch::kBFloat16))
                  : torch::Tensor();
  if (dim == 4096) {
    // V2 split path (bit-same): 256-thread main + per-halfwarp route kernel.
    auto rv = at::empty({T}, x4.options().dtype(torch::kFloat32));
    hc_pre_main_kernel<4><<<(int)T, HCP_THREADS, 0, stream0>>>(
        (const __nv_bfloat16*)x4c.data_ptr(),
        (const __nv_bfloat16*)mixes.data_ptr(),
        hsf.data_ptr<float>(), hbf.data_ptr<float>(),
        (const __nv_bfloat16*)nwb.data_ptr(),
        rv.data_ptr<float>(), (__nv_bfloat16*)x.data_ptr(),
        want_xraw ? (__nv_bfloat16*)xraw.data_ptr() : nullptr,
        (int)dim, (float)eps);
    hc_route_kernel<4><<<(int)((T + 7) / 8), 128, 0, stream0>>>(
        (const __nv_bfloat16*)mixes.data_ptr(),
        hsf.data_ptr<float>(), hbf.data_ptr<float>(),
        rv.data_ptr<float>(), post.data_ptr<float>(), comb.data_ptr<float>(),
        (int)T, (float)eps);
  } else {
    hc_fused_pre_kernel<4><<<(int)T, 1024, (hc + 1024) * sizeof(float), stream0>>>(
        (const __nv_bfloat16*)x4c.data_ptr(),
        (const __nv_bfloat16*)mixes.data_ptr(),
        hsf.data_ptr<float>(), hbf.data_ptr<float>(),
        (const __nv_bfloat16*)nwb.data_ptr(),
        post.data_ptr<float>(), comb.data_ptr<float>(),
        (__nv_bfloat16*)x.data_ptr(),
        want_xraw ? (__nv_bfloat16*)xraw.data_ptr() : nullptr,
        (int)dim, (float)eps);
  }
  if (want_xraw) return {post, comb, x, xraw};
  return {post, comb, x};
}




// Fused router: gemv(E=256) + sqrt(softplus) + (+bias top6 | ids_pre) + renorm*1.5.
// One block per token, 256 threads = one expert each.
__global__ void router_topk_kernel(
    const __nv_bfloat16* __restrict__ logits, // [T, E]
    const float* __restrict__ rb,             // [E] or null
    const int64_t* __restrict__ ids_pre,      // [T, 6] or null
    int64_t* __restrict__ ids, float* __restrict__ wts) {  // [T,6]
  constexpr int E = 256, K = 6;
  __shared__ float sscore[E];
  __shared__ float sbias[E];
  const int t = blockIdx.x, e = threadIdx.x;
  const float acc = __bfloat162float(logits[(int64_t)t * E + e]);
  // sqrt(softplus(acc))
  float sp = (acc > 20.f) ? acc : log1pf(expf(acc));
  float sc = sqrtf(sp);
  sscore[e] = sc;
  sbias[e] = rb ? (sc + rb[e]) : sc;
  __shared__ float sval[E];
  __shared__ int sidx[E];
  __shared__ int ssel[K];
  __syncthreads();
  if (ids_pre) {
    if (e < K) ssel[e] = (int)ids_pre[(int64_t)t * K + e];
    __syncthreads();
  } else {
    // parallel argmax x K; strict '>' keeps lowest index on ties (matches serial scan)
    for (int k = 0; k < K; ++k) {
      sval[e] = sbias[e]; sidx[e] = e;
      __syncthreads();
      for (int st = E >> 1; st; st >>= 1) {
        if (e < st && sval[e + st] > sval[e]) { sval[e] = sval[e + st]; sidx[e] = sidx[e + st]; }
        __syncthreads();
      }
      if (e == 0) { ssel[k] = sidx[0]; sbias[sidx[0]] = -FLT_MAX; }
      __syncthreads();
    }
  }
  if (e == 0) {
    float sum = 0.f;
    for (int k = 0; k < K; ++k) sum += sscore[ssel[k]];
    float inv = 1.5f / sum;
    for (int k = 0; k < K; ++k) {
      ids[(int64_t)t * K + k] = ssel[k];
      wts[(int64_t)t * K + k] = sscore[ssel[k]] * inv;
    }
  }
}


torch::Tensor moe_rank_routed_fp4(
    torch::Tensor xq, torch::Tensor ids, torch::Tensor wts,
    torch::Tensor w1w, torch::Tensor w1s, torch::Tensor w3w, torch::Tensor w3s,
    torch::Tensor w2w, torch::Tensor w2s) {
  at::cuda::CUDAGuard guard(xq.device());
  return moe_rank_decode_fp4(xq.contiguous(), ids.contiguous(),
                             wts.to(torch::kFloat32).contiguous(),
                             w1w, w1s, w3w, w3s, w2w, w2s);
}

__global__ void shared_glu_fused_kernel(const __nv_bfloat16* __restrict__ g,
                                        const __nv_bfloat16* __restrict__ u,
                                        __nv_bfloat16* __restrict__ out,
                                        int64_t n) {
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float gv = fminf(__bfloat162float(g[i]), 10.0f);
  float uv = fminf(fmaxf(__bfloat162float(u[i]), -10.0f), 10.0f);
  float sv = gv / (1.0f + __expf(-gv));
  out[i] = __float2bfloat16(sv * uv);
}


static std::vector<torch::Tensor> moe_rank_fused_fp4_impl(
    torch::Tensor x4, torch::Tensor fn, torch::Tensor hs, torch::Tensor hb,
    torch::Tensor ffn_norm,
    torch::Tensor router_w, torch::Tensor router_b, torch::Tensor ids_pre,
    torch::Tensor w1w, torch::Tensor w1s, torch::Tensor w3w, torch::Tensor w3s,
    torch::Tensor w2w, torch::Tensor w2s,
    torch::Tensor sw1, torch::Tensor sw3, torch::Tensor sw2, double eps,
    bool prefill_phase) {
  at::cuda::CUDAGuard guard(x4.device());
  const int64_t T = x4.size(0), hc = x4.size(1), dim = x4.size(2);
  // ---- hc_pre (same as attn_rank_fused_fp8) ----
  auto stream0 = at::cuda::getCurrentCUDAStream();
  auto hcp = hc_fused_pre(x4, fn, hs, hb, ffn_norm, eps, /*want_xraw=*/false);
  auto post = hcp[0];
  auto comb = hcp[1];
  auto x = hcp[2];
  // ---- route (sqrt(softplus(x @ Wr^T)); topk+bias or precomputed ids) ----
  auto rwb = get_or_cast(router_w, torch::kBFloat16);
  auto rbf = get_or_cast(router_b, torch::kFloat32);
  const bool use_pre = ids_pre.numel() > 0;
  auto ids = use_pre ? ids_pre.contiguous()
                     : at::empty({T, 6}, x4.options().dtype(torch::kLong));
  auto wts = at::empty({T, 6}, x4.options().dtype(torch::kFloat32));
  auto logits = at::linear(x, rwb);
  router_topk_kernel<<<(int)T, 256, 0, stream0>>>(
      (const __nv_bfloat16*)logits.data_ptr(),
      rbf.numel() > 0 ? rbf.data_ptr<float>() : nullptr,
      use_pre ? ids.data_ptr<int64_t>() : nullptr,
      ids.data_ptr<int64_t>(), wts.data_ptr<float>());
  // ---- routed experts ----
  auto xq = qdq128(x);
  auto y = prefill_phase
      ? moe_rank_grouped_prefill_fp4(xq, ids, wts, w1w, w1s, w3w, w3s, w2w, w2s)
      : moe_rank_decode_fp4(xq.contiguous(), ids.contiguous(),
                            wts.to(torch::kFloat32).contiguous(),
                            w1w, w1s, w3w, w3s, w2w, w2s);
  // ---- shared expert (same clamp cadence as python) ----
  auto smid = silu_mul_clamp_bf16(bf16_gemm(xq, sw1), bf16_gemm(xq, sw3));
  auto sout = bf16_gemm(qdq128(smid), sw2).to(torch::kFloat32);
  return {post, comb, y, sout, ids};
}

std::vector<torch::Tensor> moe_rank_fused_fp4(
    torch::Tensor x4, torch::Tensor fn, torch::Tensor hs, torch::Tensor hb,
    torch::Tensor ffn_norm,
    torch::Tensor router_w, torch::Tensor router_b, torch::Tensor ids_pre,
    torch::Tensor w1w, torch::Tensor w1s, torch::Tensor w3w, torch::Tensor w3s,
    torch::Tensor w2w, torch::Tensor w2s,
    torch::Tensor sw1, torch::Tensor sw3, torch::Tensor sw2, double eps) {
  return moe_rank_fused_fp4_impl(
      x4, fn, hs, hb, ffn_norm, router_w, router_b, ids_pre,
      w1w, w1s, w3w, w3s, w2w, w2s,
      sw1, sw3, sw2, eps,
      /*prefill_phase=*/false);
}

std::vector<torch::Tensor> moe_rank_fused_prefill_fp4(
    torch::Tensor x4, torch::Tensor fn, torch::Tensor hs, torch::Tensor hb,
    torch::Tensor ffn_norm,
    torch::Tensor router_w, torch::Tensor router_b, torch::Tensor ids_pre,
    torch::Tensor w1w, torch::Tensor w1s, torch::Tensor w3w, torch::Tensor w3s,
    torch::Tensor w2w, torch::Tensor w2s,
    torch::Tensor sw1, torch::Tensor sw3, torch::Tensor sw2, double eps) {
  return moe_rank_fused_fp4_impl(
      x4, fn, hs, hb, ffn_norm, router_w, router_b, ids_pre,
      w1w, w1s, w3w, w3s, w2w, w2s,
      sw1, sw3, sw2, eps,
      /*prefill_phase=*/true);
}


// ---- embed gather (masked TP shard) direct to f32 AR buffer
__global__ void embed_gather_f32_kernel(const int* __restrict__ ids,
                                        const __nv_bfloat16* __restrict__ embed,
                                        float* __restrict__ out,
                                        int rows, int lo, int dim) {
  const int q = blockIdx.x;
  const long id = (long)ids[q] - lo;
  const bool ok = (id >= 0) && (id < rows);
  const __nv_bfloat16* src = embed + (ok ? id : 0) * (long)dim;
  float* dst = out + (long)q * dim;
  for (int i = threadIdx.x; i < dim; i += blockDim.x)
    dst[i] = ok ? __bfloat162float(src[i]) : 0.f;
}

// ---- f32 [Q,dim] -> bf16 [Q,hc,dim] broadcast
__global__ void bcast_f32_bf16_kernel(const float* __restrict__ x,
                                      __nv_bfloat16* __restrict__ y,
                                      int hc, int dim) {
  const int q = blockIdx.x;
  const float* src = x + (long)q * dim;
  __nv_bfloat16* dst = y + (long)q * hc * dim;
  for (int i = threadIdx.x; i < dim; i += blockDim.x) {
    const __nv_bfloat16 v = __float2bfloat16(src[i]);
    for (int h = 0; h < hc; ++h) dst[(long)h * dim + i] = v;
  }
}

void embed_gather_f32(torch::Tensor ids, torch::Tensor embed, torch::Tensor out, long lo) {
  auto stream = at::cuda::getCurrentCUDAStream();
  const int q = out.size(0), rows = embed.size(0), dim = embed.size(1);
  embed_gather_f32_kernel<<<q, 256, 0, stream>>>(
      ids.data_ptr<int>(),
      reinterpret_cast<const __nv_bfloat16*>(embed.data_ptr()),
      out.data_ptr<float>(), rows, (int)lo, dim);
}

torch::Tensor bcast_f32_bf16(torch::Tensor x, long hc) {
  auto stream = at::cuda::getCurrentCUDAStream();
  const int q = x.size(0), dim = x.size(1);
  auto y = torch::empty({q, hc, dim}, x.options().dtype(torch::kBFloat16));
  bcast_f32_bf16_kernel<<<q, 256, 0, stream>>>(
      x.data_ptr<float>(),
      reinterpret_cast<__nv_bfloat16*>(y.data_ptr()), (int)hc, dim);
  return y;
}


// ---- out[i] = (float)a[i] + (float)b[i], no intermediate f32 materialization

__global__ void hc_post_bf16_kernel(const __nv_bfloat16* __restrict__ branch,
                                    const __nv_bfloat16* __restrict__ residual,
                                    const float* __restrict__ post,
                                    const float* __restrict__ comb,
                                    __nv_bfloat16* __restrict__ y,
                                    int hc, int dim, bool comb_t) {
  const int t = blockIdx.x, h = blockIdx.y;
  const float p = post[(long)t * hc + h];
  const float* cb = comb + (long)t * hc * hc;
  float cw[8];
  // comb_t: comb given as original [t, i, j] (contract i) -> read column h without a transpose copy
  for (int j = 0; j < hc; ++j) cw[j] = comb_t ? cb[j * hc + h] : cb[h * hc + j];
  const __nv_bfloat16* br = branch + (long)t * dim;
  const __nv_bfloat16* rs = residual + (long)t * hc * dim;
  __nv_bfloat16* out = y + ((long)t * hc + h) * dim;
  for (int c = threadIdx.x; c < dim; c += blockDim.x) {
    float v = p * __bfloat162float(br[c]);
    for (int j = 0; j < hc; ++j)
      v += cw[j] * __bfloat162float(rs[(long)j * dim + c]);
    out[c] = __float2bfloat16(v);
  }
}

// Fixed production HC=4: one block per token reuses each residual row for all
// four outputs. Explicit round-to-nearest operations preserve the generic
// kernel's per-output arithmetic order under --use_fast_math.
__global__ void hc_post_bf16_hc4_kernel(
    const void* __restrict__ branch_v,
    const __nv_bfloat16* __restrict__ residual,
    const float* __restrict__ post,
    const float* __restrict__ comb,
    __nv_bfloat16* __restrict__ y, int dim, bool comb_t, bool br_f32) {
  const int t = blockIdx.x;
  float p[4], cw[4][4];
#pragma unroll
  for (int h = 0; h < 4; ++h) {
    p[h] = post[(long)t * 4 + h];
#pragma unroll
    for (int j = 0; j < 4; ++j)
      cw[h][j] = comb_t ? comb[((long)t * 4 + j) * 4 + h] : comb[((long)t * 4 + h) * 4 + j];
  }
  const int dim2 = dim >> 1;
  const auto* br = br_f32 ? nullptr
      : reinterpret_cast<const __nv_bfloat162*>((const __nv_bfloat16*)branch_v + (long)t * dim);
  const float2* brf = br_f32
      ? reinterpret_cast<const float2*>((const float*)branch_v + (long)t * dim) : nullptr;
  const auto* rs = reinterpret_cast<const __nv_bfloat162*>(residual + (long)t * 4 * dim);
  auto* out = reinterpret_cast<__nv_bfloat162*>(y + (long)t * 4 * dim);
  for (int c = blockIdx.y * blockDim.x + threadIdx.x; c < dim2;
       c += blockDim.x * gridDim.y) {
    float2 rv[4];
#pragma unroll
    for (int j = 0; j < 4; ++j)
      rv[j] = __bfloat1622float2(rs[(long)j * dim2 + c]);
    float2 b;
    if (br_f32) {   // identical to a .to(bf16) cast followed by the bf16 load
      const float2 bb = brf[c];
      b.x = __bfloat162float(__float2bfloat16_rn(bb.x));
      b.y = __bfloat162float(__float2bfloat16_rn(bb.y));
    } else {
      b = __bfloat1622float2(br[c]);
    }
#pragma unroll
    for (int h = 0; h < 4; ++h) {
      float x = __fmul_rn(p[h], b.x), z = __fmul_rn(p[h], b.y);
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        x = __fmaf_rn(cw[h][j], rv[j].x, x);
        z = __fmaf_rn(cw[h][j], rv[j].y, z);
      }
      out[(long)h * dim2 + c] = __floats2bfloat162_rn(x, z);
    }
  }
}

torch::Tensor hc_post_fused_bf16(torch::Tensor branch, torch::Tensor residual,
                                 torch::Tensor post, torch::Tensor comb, bool comb_t) {
  at::cuda::CUDAGuard guard(branch.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  const int64_t T = residual.size(0), hc = residual.size(1), dim = residual.size(2);
  TORCH_CHECK(hc <= 8, "hc_post_fused_bf16: hc must be <= 8");
  const bool br_f32 = branch.scalar_type() == torch::kFloat32;
  TORCH_CHECK((br_f32 || branch.scalar_type() == torch::kBFloat16) && branch.is_contiguous());
  auto rc = residual.contiguous();
  auto pc = post.to(torch::kFloat32).contiguous();
  auto cc = comb.to(torch::kFloat32).contiguous();
  auto y = at::empty_like(rc);
  if (hc == 4 && (dim & 1) == 0) {
    // split the dim axis across blocks: 8 blocks cannot fill 108 SMs.
    const int d2 = (int)(dim >> 1), thr = 256;
    unsigned nsp = (unsigned)((d2 + thr - 1) / thr);
    if (nsp < 1u) nsp = 1u;
    if (nsp > 32u) nsp = 32u;
    hc_post_bf16_hc4_kernel<<<dim3((unsigned)T, nsp), thr, 0, stream>>>(
        branch.data_ptr(), (const __nv_bfloat16*)rc.data_ptr(),
        pc.data_ptr<float>(), cc.data_ptr<float>(),
        (__nv_bfloat16*)y.data_ptr(), (int)dim, comb_t, br_f32);
  } else {
    dim3 grid((unsigned)T, (unsigned)hc);
    auto bb = br_f32 ? branch.to(torch::kBFloat16).contiguous() : branch;
    hc_post_bf16_kernel<<<grid, 256, 0, stream>>>(
        (const __nv_bfloat16*)bb.data_ptr(), (const __nv_bfloat16*)rc.data_ptr(),
        pc.data_ptr<float>(), cc.data_ptr<float>(),
        (__nv_bfloat16*)y.data_ptr(), (int)hc, (int)dim, comb_t);
  }
  return y;
}



__global__ void add2_bf16_f32_kernel(const __nv_bfloat16* __restrict__ a,
                                     const __nv_bfloat16* __restrict__ b,
                                     float* __restrict__ y, long n) {
  const long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] = __bfloat162float(a[i]) + __bfloat162float(b[i]);
}

void add2_bf16_f32(torch::Tensor a, torch::Tensor b, torch::Tensor y) {
  auto stream = at::cuda::getCurrentCUDAStream();
  const long n = y.numel();
  add2_bf16_f32_kernel<<<(int)((n + 255) / 256), 256, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(a.data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(b.data_ptr()),
      y.data_ptr<float>(), n);
}

// ============================================================================
// Fused decode attention leaf, base layers (ratio 0) -- new-past ABI (ring, no carry).
// Mirrors broken attn_rank_sparse_decode_fp8(ratio=0) minus hc_pre (done by hc_pre_norm
// leaf in Block.forward) and minus the paged main pool (new past keeps a per-slot ring).
//   x        [T, dim]   bf16  (after hc_pre_norm)
//   freq     [T, 32]    complex64 contiguous (freqs_cis rows of the absolute tokens)
//   tok      [T]        int64 absolute token positions (device)
//   ring_kv  [ring, KD] bf16  slot main_kv, written in place at tok % ring
//   wqa/wqb/wkv/wob     bf16 pre-dequant Linear weights (fp8_linear ABI)
//   q_norm/kv_norm      RMSNorm weights; woa [G, R, D] bf16; attn_sink [h]
// returns wo_b partial [T, dim] (TP all_reduce stays in python, as RowParallelLinear).
// ============================================================================
static torch::Tensor attn_decode_fused_r0(
    torch::Tensor x, torch::Tensor freq, torch::Tensor tok, torch::Tensor ring_kv,
    torch::Tensor wqa, torch::Tensor q_norm, torch::Tensor wqb,
    torch::Tensor wkv, torch::Tensor kv_norm,
    torch::Tensor woa, torch::Tensor wob, torch::Tensor attn_sink,
    int64_t n_heads, int64_t n_groups, int64_t W, double scale, double eps) {
  at::cuda::CUDAGuard guard(x.device());
  const int64_t T = x.size(0);
  const int64_t RD = 64, HD = 512;
  const int64_t ring = ring_kv.size(0), KD = ring_kv.size(1);
  TORCH_CHECK(x.dim() == 2 && x.scalar_type() == torch::kBFloat16, "r0 leaf: x must be bf16 [T,dim]");
  TORCH_CHECK(tok.is_cuda() && tok.scalar_type() == torch::kInt64 && tok.numel() == T,
              "r0 leaf: tok must be CUDA int64 [T]");
  TORCH_CHECK(ring_kv.dim() == 2 && ring_kv.is_contiguous() && ring_kv.scalar_type() == torch::kBFloat16,
              "r0 leaf: ring_kv must be contiguous bf16 [ring,KD]");
  TORCH_CHECK(freq.size(0) == T && freq.scalar_type() == torch::kComplexFloat, "r0 leaf: freq [T,32] c64");
  TORCH_CHECK(W > 0 && W <= ring, "r0 leaf: window must fit in ring");

  auto xc = x.contiguous();
  // ---- q: wq_a -> q_norm -> wq_b -> per-head rms -> rope(last 64) ----
  auto qr = rms_norm_at(fp8_linear(xc, wqa), q_norm, eps);
  auto q = fp8_linear(qr.contiguous(), wqb).reshape({T, n_heads, HD});
  {
    auto z = q.to(torch::kFloat32);
    q = (z * at::rsqrt(z.square().mean(-1, true) + eps)).to(torch::kBFloat16);
  }
  rotary_at(q.slice(-1, HD - RD, HD), freq, false);

  // ---- kv: wkv -> kv_norm -> qdq64(nope) | rope(rope) -> ring write ----
  auto kv = rms_norm_at(fp8_linear(xc, wkv), kv_norm, eps);
  TORCH_CHECK(kv.size(1) == KD, "r0 leaf: kv width ", kv.size(1), " != ring KD ", KD);
  kv = at::cat({qdq64(kv.slice(1, 0, KD - RD).contiguous()), kv.slice(1, KD - RD)}, -1);
  rotary_at(kv.slice(-1, KD - RD, KD), freq, false);
  ring_kv.index_copy_(0, at::remainder(tok, ring), kv);

  // ---- sliding-window ids over the ring: [T, W], -1 for tokens < 0 ----
  auto t = tok.unsqueeze(1) - W + 1 + at::arange(W, tok.options());
  auto idxs = at::where(t < 0, at::full_like(t, -1), at::remainder(t, ring)).unsqueeze(0).contiguous();

  // ---- attention on the ring (pool = ring as one page; total = ring) ----
  auto pool = ring_kv.unsqueeze(0);
  auto table = at::zeros({1}, tok.options());
  auto sink = attn_sink.to(torch::kFloat32).contiguous();
  auto q4 = q.reshape({1, T, n_heads, HD});
  torch::Tensor o;
  if (n_heads <= 8) {
    o = sparse_attn_paged(q4.contiguous(), pool, table, pool, table, sink, idxs, ring, scale)
            .view({1, T, n_heads, HD});
  } else {
    std::vector<torch::Tensor> outs;
    for (int64_t i = 0; i < n_heads; i += 8) {
      const int64_t j = std::min<int64_t>(i + 8, n_heads);
      outs.push_back(sparse_attn_paged(q4.slice(2, i, j).contiguous(), pool, table, pool, table,
                                       sink.slice(0, i, j).contiguous(), idxs, ring, scale)
                         .view({1, T, j - i, HD}));
    }
    o = at::cat(outs, 2);
  }

  // ---- o: rope^-1(last 64) -> wo_a grouped -> wo_b (partial) ----
  o = o.reshape({T, n_heads, HD});
  rotary_at(o.slice(-1, HD - RD, HD), freq, true);
  o = o.reshape({T, n_groups, -1}).contiguous();
  o = wo_a_grouped(o, woa);                       // [T, G, R]
  return fp8_linear(o.reshape({T, -1}).contiguous(), wob);
}


// --- peer_ar_ipc.cu (one-shot NVLink all-reduce for the multi-process TP group) ---
torch::Tensor peer_ar_ipc_alloc(int64_t rank, int64_t world, int64_t n);
void peer_ar_ipc_open(torch::Tensor all_handles);
void peer_ar_ipc_run2(torch::Tensor y);
int64_t peer_ar_ipc_numel();
void peer_ar_ipc_close();


// ---- fused 3-way row scatter (res_x bf16 + derived kv/score f32 rings) ----
// Replaces three separate index_copy_ launches that all use the same row map.
__global__ void scatter3_rows_kernel(__nv_bfloat16* __restrict__ res,
                                     float* __restrict__ kvr,
                                     float* __restrict__ scr,
                                     const long* __restrict__ rows,
                                     const __nv_bfloat16* __restrict__ x,
                                     const float* __restrict__ kv,
                                     const float* __restrict__ sc,
                                     int d1, int d2, int d3) {
  const long n = blockIdx.x;
  const long r = rows[n];
  if (r < 0) return;
  // 2-D grid: blockIdx.y splits the row so we do not launch with only `n` blocks
  const int base = blockIdx.y * blockDim.x + threadIdx.x;
  const int step = blockDim.x * gridDim.y;
  for (int i = base; i < d1; i += step) res[r * d1 + i] = x[n * d1 + i];
  for (int i = base; i < d2; i += step) kvr[r * d2 + i] = kv[n * d2 + i];
  for (int i = base; i < d3; i += step) scr[r * d3 + i] = sc[n * d3 + i];
}

void scatter3_rows(torch::Tensor res, torch::Tensor kvr, torch::Tensor scr,
                   torch::Tensor rows, torch::Tensor x,
                   torch::Tensor kv, torch::Tensor sc) {
  const int n = (int)rows.size(0);
  const int d1 = (int)res.size(1), d2 = (int)kvr.size(1), d3 = (int)scr.size(1);
  TORCH_CHECK(x.size(0) == n && kv.size(0) == n && sc.size(0) == n, "scatter3 n mismatch");
  TORCH_CHECK(x.size(1) == d1 && kv.size(1) == d2 && sc.size(1) == d3, "scatter3 width mismatch");
  auto st = at::cuda::getCurrentCUDAStream();
  const int wmax = d1 > d2 ? (d1 > d3 ? d1 : d3) : (d2 > d3 ? d2 : d3);
  int ny = (wmax + 255) / 256; if (ny > 16) ny = 16; if (ny < 1) ny = 1;
  scatter3_rows_kernel<<<dim3(n, ny), 256, 0, st>>>(
      (__nv_bfloat16*)res.data_ptr(), kvr.data_ptr<float>(), scr.data_ptr<float>(),
      rows.data_ptr<long>(), (const __nv_bfloat16*)x.data_ptr(),
      kv.data_ptr<float>(), sc.data_ptr<float>(), d1, d2, d3);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fp8_linear", &fp8_linear, "fp8 Linear leaf: act qdq128 + bf16 gemm (pure)", pybind11::call_guard<pybind11::gil_scoped_release>());
  m.def("attn_decode_fused_r0", &attn_decode_fused_r0,
        "fused decode attention leaf, base layers (ring ABI): q/kv/ring-write/window attn/o; returns wo_b partial");
  m.def("wo_a_grouped", &wo_a_grouped, "o [T,G,D] bf16 x w [G,R,D] bf16 -> [T,G,R], pure");
  m.def("hc_fused_pre", &hc_fused_pre,
        "x4 [T,hc,d] bf16, fn, hs, hb, norm_w, eps, want_xraw -> {post f32, comb f32, x_normed bf16[, xraw]}, pure");
  m.def("rope_inplace", &rope_inplace, "in-place RoPE on last dim (bf16 x, complex64 freqs [S,rd/2])");
  m.def("rms_norm", &rms_norm, "bf16 x [T,D] * fp32 w [D] -> bf16, pure");
  m.def("bf16_gemm", &bf16_gemm, "pure BF16 GEMM (pre-dequantized weight)", pybind11::call_guard<pybind11::gil_scoped_release>());
  m.def("dequant_fp8_bf16", &dequant_fp8_bf16, "load-time fp8->bf16 dequant (pure)", pybind11::call_guard<pybind11::gil_scoped_release>());
  m.def("moe_rank_routed_fp4", &moe_rank_routed_fp4, "routed experts only", pybind11::call_guard<pybind11::gil_scoped_release>());
  m.def("moe_rank_fused_fp4", &moe_rank_fused_fp4, "fused decode MoE loop (fp4)", pybind11::call_guard<pybind11::gil_scoped_release>());
  m.def("moe_rank_fused_prefill_fp4", &moe_rank_fused_prefill_fp4, "fused prefill MoE preserving grouped arithmetic", pybind11::call_guard<pybind11::gil_scoped_release>());
  m.def("sgemm_skinny2_f32", &sgemm_skinny2_f32, "v2", pybind11::call_guard<pybind11::gil_scoped_release>());
  m.def("hc_post_fused_bf16", &hc_post_fused_bf16, "bf16 hc post", pybind11::call_guard<pybind11::gil_scoped_release>());
  m.def("sinkhorn_hc", &sinkhorn_hc, "fused 4x4 sinkhorn", pybind11::call_guard<pybind11::gil_scoped_release>());
  m.def("grouped_gemm_sm80", &grouped_gemm_sm80, "grouped gemm probe", pybind11::call_guard<pybind11::gil_scoped_release>());
  m.def("compressor_rows", &compressor_rows, "graph-safe per-query compressor rows (last row of each window)");
  m.def("compressor_rows_ring", &compressor_rows_ring, "graph-safe compressor rows gathered from derived rings via rowmap (no torch gather/cat/where)");
  m.def("paged_scatter_positions_masked", &paged_scatter_positions_masked, "graph-safe masked paged scatter with CUDA positions");
  m.def("embed_gather_f32", &embed_gather_f32, "masked TP embed gather to f32", pybind11::call_guard<pybind11::gil_scoped_release>());
  m.def("bcast_f32_bf16", &bcast_f32_bf16, "f32->bf16 hc broadcast", pybind11::call_guard<pybind11::gil_scoped_release>());
  m.def("add2_bf16_f32", &add2_bf16_f32, "bf16+bf16->f32 add", pybind11::call_guard<pybind11::gil_scoped_release>());
  m.def("scatter3_rows", &scatter3_rows, "fused 3-way row scatter", pybind11::call_guard<pybind11::gil_scoped_release>());
  m.def("index_score_reduce", &index_score_reduce, "relu*w sum_h + causal mask (leaf)");
  m.def("topk_select_post", &topk_select_post, "exact radix top-K + mask/offset (leaf)");
  m.def("index_score_fused", &index_score_fused, "fused gather+score+relu*w+sum_h, device-length early exit (graph-safe)");
  m.def("index_score_reduce_positions", &index_score_reduce_positions, "relu*w sum_h + causal mask, per-row device positions (graph-safe)");
  m.def("topk_select_post_positions", &topk_select_post_positions, "exact radix top-K + mask/offset, per-row device positions (graph-safe)");
  m.def("fp8_qdq64_", &fp8_qdq64_, "in-place fp8 e4m3 qdq block=64 pow2 scale (act_quant inplace), graph-safe");
  m.def("silu_mul_clamp_bf16", &silu_mul_clamp_bf16, "bf16 silu(clamp_max(g,10))*clamp(u,-10,10) -> bf16, torch f32 cadence");
  m.def("rms_scale_", &rms_scale_, "in-place x *= rsqrt(mean(x^2)+eps) per last dim, torch-bf16-step-rounding emulation");
  m.def("had_fp4_qdq_", &had_fp4_qdq_, "in-place fused hadamard rotate + fp4 e2m1 qdq (32-block), graph-safe");
  m.def("peer_ar_ipc_alloc", &peer_ar_ipc_alloc, "alloc local mailbox, return IPC handles");
  m.def("peer_ar_ipc_open", &peer_ar_ipc_open, "map peer mailboxes from gathered IPC handles");
  m.def("peer_ar_ipc_run2", &peer_ar_ipc_run2, "two-shot peer all-reduce (fp32, in place)");
  m.def("peer_ar_ipc_numel", &peer_ar_ipc_numel, "registered numel, 0 if not ready");
  m.def("peer_ar_ipc_close", &peer_ar_ipc_close, "tear down mailboxes");
  m.def("sparse_attn_paged", &sparse_attn_paged, "TC sparse paged MLA attention (leaf)", pybind11::call_guard<pybind11::gil_scoped_release>());
}
