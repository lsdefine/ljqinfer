
// compressor_tail: fuse (ape+ot+softmax+weighted-sum+rmsnorm+rope+hadamard+fp4q)
// One block per compressed row; blockDim.x == d (128 or 512). Zero ATen ops.
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

__device__ __forceinline__ float pow2_ceil_f(float x) {
  int e; float m = frexpf(x, &e);
  return ldexpf(m == 0.5f ? 0.5f : 1.0f, e);
}

static __device__ __forceinline__ float blk_sum(float v, float* sm, int n) {
  int t = threadIdx.x;
  sm[t] = v; __syncthreads();
  for (int s = n >> 1; s; s >>= 1) {
    if (t < s) sm[t] += sm[t + s];
    __syncthreads();
  }
  float r = sm[0]; __syncthreads();
  return r;
}

// kvin/scin: [b, S, C] bf16 / float   (C = d, or 2d when overlap)
// ape: [ratio, C] float ; norm_w: [d] fp32 (same as RMSNorm.weight / rms_norm op) ; freqs: [nb, rd/2] complex(float2)
// out: [b, nb, d] bf16
__device__ __forceinline__ float ldv(__nv_bfloat16 x) { return __bfloat162float(x); }
__device__ __forceinline__ float ldv(float x) { return x; }

template <typename T>
__global__ void compressor_tail_kernel(
    const T* __restrict__ kvin,
    const T* __restrict__ scin,
    const float* __restrict__ ape,
    const float* __restrict__ norm_w,
    const float2* __restrict__ freqs,
    __nv_bfloat16* __restrict__ out,
    int S, int nb, int ratio, int d, int rd, float eps,
    int overlap, int rotate, int noq, int fpb,
    const int64_t* __restrict__ rowmap, const int32_t* __restrict__ hasprev,
    const int32_t* __restrict__ validp) {
  extern __shared__ float sh[];
  float* red = sh;          // d floats (reduction / hadamard workspace)
  const int c = threadIdx.x;
  const int i = blockIdx.x;             // compressed row
  const int b = blockIdx.y;
  const int C = overlap ? 2 * d : d;
  const int R = overlap ? 2 * ratio : ratio;
  // rowmap mode (graph decode): kvin/scin are the [*, C] rings, rowmap[b*R + j] is the
  // ring row of window row j (prev window first when overlap), nb == 1; invalid queries
  // write zeros; a missing prev window (score -inf, weight 0) is skipped -- exact.
  if (validp && !validp[b]) { out[(long)b * d + c] = __float2bfloat16(0.f); return; }
  const long base = rowmap ? 0 : (long)b * S * C;
  const int64_t* rm = rowmap ? rowmap + (long)b * R : nullptr;

  // Row j -> (src, ch); j < jlo rows are skipped (overlap && i == 0: no previous window).
  // Loops are unrolled for ILP (independent loads); accumulation order is unchanged
  // (sequential j) so results stay bit-identical to the unroll-1 version.
  const int jlo = (overlap && (rowmap ? !hasprev[b] : i == 0)) ? ratio : 0;
  #define CT_OFF(j, src, ch) \
    if (rm)              { src = (int)rm[j]; ch = (overlap && (j) >= ratio) ? d + c : c; } \
    else if (!overlap)   { src = i * ratio + (j);         ch = c; } \
    else if ((j) >= ratio) { src = i * ratio + ((j)-ratio); ch = d + c; } \
    else                 { src = (i-1) * ratio + (j);     ch = c; }
  // ---- pass 1: max over R (per channel) ----
  float mx = -INFINITY;
  #pragma unroll 8
  for (int j = jlo; j < R; ++j) {
    int src, ch; CT_OFF(j, src, ch)
    float s = ldv(scin[base + (long)src * C + ch]) + ape[(long)(j % ratio) * C + ch];
    mx = fmaxf(mx, s);
  }
  // ---- pass 2: softmax-weighted sum ----
  float num = 0.f, den = 0.f;
  #pragma unroll 8
  for (int j = jlo; j < R; ++j) {
    int src, ch; CT_OFF(j, src, ch)
    long off = base + (long)src * C + ch;
    den += __expf(ldv(scin[off]) + ape[(long)(j % ratio) * C + ch] - mx);
  }
  // pass 3: fp32 weights and fp32 kv, exactly like official (kv * score.softmax(2)) in fp32
  float v = 0.f;
  #pragma unroll 8
  for (int j = jlo; j < R; ++j) {
    int src, ch; CT_OFF(j, src, ch)
    long off = base + (long)src * C + ch;
    float s = __expf(ldv(scin[off]) + ape[(long)(j % ratio) * C + ch] - mx) / den;
    v += s * ldv(kvin[off]);
  }
  #undef CT_OFF

  // ---- rmsnorm: identical rounding to prefill path (Compressor.forward -> kv.to(bf16) -> ops.rms_norm):
  //      xb = bf16(pooled); inv = rsqrt(mean(xb^2)+eps); y = bf16(w_f32 * (xb * inv)) ----
  float xb = __bfloat162float(__float2bfloat16(v));
  float ss = blk_sum(xb * xb, red, d);
  v = __bfloat162float(__float2bfloat16(norm_w[c] * (xb * rsqrtf(ss / d + eps))));

  // ---- rope on last rd channels ----
  if (c >= d - rd) {
    red[c] = v;
  }
  __syncthreads();
  if (c >= d - rd) {
    int k = c - (d - rd);
    int pair = k & ~1;
    float2 f = freqs[(long)(fpb ? ((long)b * nb + i) : i) * (rd / 2) + (k >> 1)];
    float re = red[d - rd + pair], im = red[d - rd + pair + 1];
    v = (k & 1) ? (re * f.y + im * f.x) : (re * f.x - im * f.y);
    v = __bfloat162float(__float2bfloat16(v));
  }
  __syncthreads();

  if (noq) { out[(long)b*nb*d + (long)i*d + c] = __float2bfloat16(v); return; }
  if (rotate) {
    // ---- hadamard butterfly over d, then scale d^-0.5 ----
    red[c] = v; __syncthreads();
    for (int hlen = 1; hlen < d; hlen <<= 1) {
      int g = c / (2 * hlen), p = (c / hlen) & 1, k = c % hlen;
      int i0 = g * 2 * hlen + k, i1 = i0 + hlen;
      float a = red[i0], bb = red[i1];
      float r = p ? (a - bb) : (a + bb);
      __syncthreads();
      red[c] = r;  __syncthreads();
    }
    v = red[c] * rsqrtf((float)d);
    __syncthreads();
    // ---- fp4 qdq, block 32 (amax within each 32-channel group) ----
    red[c] = fabsf(v); __syncthreads();
    int g0 = c & ~31;
    float amax = 0.f;
    for (int k = 0; k < 32; ++k) amax = fmaxf(amax, red[g0 + k]);
    __syncthreads();
    float sc = pow2_ceil_f(fmaxf(amax, 1e-38f) / 6.0f);
    float zn = fabsf(v) / sc;
    const float BND[7] = {0.25f,0.75f,1.25f,1.75f,2.5f,3.5f,5.0f};
    const float LUT[8] = {0.f,0.5f,1.f,1.5f,2.f,3.f,4.f,6.f};
    int idx = 0;
    #pragma unroll
    for (int k = 0; k < 7; ++k) idx += (zn >= BND[k]);
    v = copysignf(LUT[idx], v) * sc;
  } else {
    // ---- mxfp8 e4m3 qdq, block 64, only on the first d-rd channels ----
    red[c] = fabsf(v); __syncthreads();
    if (c < d - rd) {
      int g0 = c & ~63;
      float amax = 0.f;
      for (int k = 0; k < 64; ++k) amax = fmaxf(amax, red[g0 + k]);
      amax = fmaxf(amax, 1.0e-4f);
      float sc = exp2f(ceilf(log2f(amax / 448.0f)));
      float z = fminf(fmaxf(v / sc, -448.0f), 448.0f);
      __nv_fp8_e4m3 qv; qv.__x = __nv_cvt_float_to_fp8(z, __NV_SATFINITE, __NV_E4M3);
      v = (float)qv * sc;
    }
    __syncthreads();
  }
  out[((long)b * nb + i) * d + c] = __float2bfloat16(v);
}

static torch::Tensor compressor_tail_launch(
    torch::Tensor kvin, torch::Tensor scin, torch::Tensor ape,
    torch::Tensor norm_w, torch::Tensor freqs,
    int64_t ratio, int64_t d, int64_t rd, double eps,
    bool overlap, bool rotate, bool noq, bool fpb,
    const int64_t* rowmap = nullptr, const int32_t* hasprev = nullptr,
    const int32_t* validp = nullptr, int64_t rows_q = 0) {
  int b = rowmap ? (int)rows_q : kvin.size(0), S = rowmap ? 0 : kvin.size(1);
  int nb = rowmap ? 1 : S / ratio;
  auto out = torch::empty({b, nb, d}, kvin.options().dtype(torch::kBFloat16));
  dim3 grid(nb, b);
  size_t shm = d * sizeof(float);
  auto stream = c10::cuda::getCurrentCUDAStream();
  TORCH_CHECK(kvin.scalar_type() == scin.scalar_type(), "kvin/scin dtype mismatch");
  TORCH_CHECK(norm_w.scalar_type() == at::kFloat && norm_w.is_contiguous() && norm_w.numel() == d,
              "compressor norm_w must be contiguous fp32 [d] (got ", norm_w.scalar_type(), " ", norm_w.sizes(), ")");
  TORCH_CHECK(ape.scalar_type() == at::kFloat && ape.is_contiguous(), "compressor ape must be contiguous fp32");
  #define CT_LAUNCH(T) compressor_tail_kernel<T><<<grid, d, shm, stream>>>( \
      reinterpret_cast<const T*>(kvin.data_ptr()), \
      reinterpret_cast<const T*>(scin.data_ptr()), \
      ape.data_ptr<float>(), \
      norm_w.data_ptr<float>(), \
      reinterpret_cast<const float2*>(freqs.data_ptr()), \
      reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), \
      S, nb, (int)ratio, (int)d, (int)rd, (float)eps, overlap ? 1 : 0, rotate ? 1 : 0, noq ? 1 : 0, fpb ? 1 : 0, \
      rowmap, hasprev, validp)
  if (kvin.scalar_type() == at::kFloat) { CT_LAUNCH(float); }
  else { CT_LAUNCH(__nv_bfloat16); }
  #undef CT_LAUNCH
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}











// Per-query graph-safe wrapper that directly reuses compressor_tail.  Keeping
// the original leaf avoids a second floating-point implementation and preserves
// its exact operation order. Q is graph-static and at most the resident width.
torch::Tensor compressor_rows(
    torch::Tensor kvin, torch::Tensor scin, torch::Tensor ape,
    torch::Tensor norm_w, torch::Tensor freqs, torch::Tensor valid,
    int64_t ratio, int64_t d, int64_t rd, double eps,
    bool overlap, bool rotate, bool noq) {
  TORCH_CHECK(kvin.is_cuda() && scin.is_cuda() && ape.is_cuda() &&
              norm_w.is_cuda() && freqs.is_cuda() && valid.is_cuda(),
              "compressor_rows expects CUDA tensors");
  TORCH_CHECK(kvin.dim() == 3 && scin.sizes() == kvin.sizes() &&
              valid.numel() == kvin.size(0) && freqs.size(0) == kvin.size(0),
              "bad compressor_rows shapes");
  // Batched: single launch over all T query rows (grid = (nb, T)); row t keeps
  // only its last compressed row — identical math to the original per-row loop.
  auto kv = kvin.contiguous(), sc = scin.contiguous();
  const int64_t nbr = kvin.size(1) / ratio;
  // original loop passed the row's freq (duplicated when overlap) indexed by i;
  // batched kernel indexes freqs[b * nb + i], so repeat each row nb times.
  auto fr = (nbr > 1 ? at::repeat_interleave(freqs, nbr, 0) : freqs).contiguous();
  auto y = compressor_tail_launch(kv, sc, ape, norm_w, fr,
                                  ratio, d, rd, eps, overlap, rotate, noq, true);
  auto out = y.select(1, y.size(1) - 1).contiguous();
  return at::where(valid.to(torch::kBool).unsqueeze(1), out,
                   at::zeros_like(out));
}

// Graph-safe decode variant: gathers the window rows straight from the derived
// rings (kvr/scr [*, C] fp32/bf16) through rowmap [Q, R] (prev window rows first
// when overlap; ignored for rows with hasprev == 0) and masks invalid queries to
// zero in-kernel.  Same math/order as compressor_rows on the materialised windows.
torch::Tensor compressor_rows_ring(
    torch::Tensor kvr, torch::Tensor scr, torch::Tensor rowmap, torch::Tensor hasprev,
    torch::Tensor ape, torch::Tensor norm_w, torch::Tensor freqs, torch::Tensor valid,
    int64_t ratio, int64_t d, int64_t rd, double eps,
    bool overlap, bool rotate, bool noq) {
  TORCH_CHECK(kvr.is_cuda() && scr.is_cuda() && rowmap.is_cuda() && ape.is_cuda() &&
              norm_w.is_cuda() && freqs.is_cuda() && valid.is_cuda(),
              "compressor_rows_ring expects CUDA tensors");
  const int64_t R = overlap ? 2 * ratio : ratio, C = overlap ? 2 * d : d;
  TORCH_CHECK(kvr.dim() == 2 && kvr.sizes() == scr.sizes() && kvr.size(1) == C &&
              kvr.is_contiguous() && scr.is_contiguous() &&
              kvr.scalar_type() == scr.scalar_type(),
              "bad compressor_rows_ring ring shapes");
  const int64_t Q = rowmap.size(0);
  TORCH_CHECK(rowmap.dim() == 2 && rowmap.size(1) == R && rowmap.is_contiguous() &&
              rowmap.scalar_type() == torch::kInt64 && valid.numel() == Q &&
              valid.scalar_type() == torch::kInt32 && valid.is_contiguous() &&
              freqs.size(0) == Q && freqs.is_contiguous() &&
              (!overlap || (hasprev.numel() == Q && hasprev.scalar_type() == torch::kInt32 &&
                            hasprev.is_contiguous())),
              "bad compressor_rows_ring shapes");
  c10::cuda::CUDAGuard guard(kvr.device());
  auto y = compressor_tail_launch(kvr, scr, ape, norm_w, freqs,
                                  ratio, d, rd, eps, overlap, rotate, noq, true,
                                  rowmap.data_ptr<int64_t>(),
                                  overlap ? hasprev.data_ptr<int32_t>() : nullptr,
                                  valid.data_ptr<int32_t>(), Q);
  return y.select(1, 0);
}

#ifndef DSV4_NO_PYBIND
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("compressor_rows", &compressor_rows, "graph-safe per-query compressor rows");
}
#endif

