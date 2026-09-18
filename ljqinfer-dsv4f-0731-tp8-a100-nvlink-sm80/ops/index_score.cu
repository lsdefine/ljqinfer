#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cmath>

__global__ void index_score_reduce_kernel(const __nv_bfloat16* __restrict__ sc,
                                          const float* __restrict__ w,
                                          float* __restrict__ out,
                                          int H, int N, int ratio, long pos0) {
  extern __shared__ float sw[];
  const long s = blockIdx.x;
  for (int h = threadIdx.x; h < H; h += blockDim.x) sw[h] = w[s * H + h];
  __syncthreads();
  const long base = s * (long)H * N;
  const int lim = (int)((pos0 + s + 1) / ratio);
  for (int t = threadIdx.x; t < N; t += blockDim.x) {
    if (t >= lim) { out[s * (long)N + t] = -INFINITY; continue; }
    float acc = 0.f;
    for (int h = 0; h < H; ++h) {
      float v = __bfloat162float(sc[base + (long)h * N + t]);
      if (v > 0.f) acc += v * sw[h];
    }
    out[s * (long)N + t] = acc;
  }
}

torch::Tensor index_score_reduce(torch::Tensor score, torch::Tensor weights,
                                 int64_t ratio, int64_t pos0) {
  at::cuda::CUDAGuard guard(score.device());
  auto sc = score.contiguous();           // [B,S,H,N]
  auto wf = weights.to(torch::kFloat32).contiguous();
  const int64_t B = sc.size(0), S = sc.size(1), H = sc.size(2), N = sc.size(3);
  auto out = at::empty({B, S, N}, sc.options().dtype(torch::kFloat32));
  auto stream = at::cuda::getCurrentCUDAStream();
  index_score_reduce_kernel<<<(int)(B * S), 256, H * sizeof(float), stream>>>(
      (const __nv_bfloat16*)sc.data_ptr(), wf.data_ptr<float>(),
      out.data_ptr<float>(), (int)H, (int)N, (int)ratio, (long)pos0);
  return out;
}



__global__ void index_score_reduce_positions_kernel(
    const __nv_bfloat16* __restrict__ sc,
    const float* __restrict__ w,
    const int64_t* __restrict__ positions,
    float* __restrict__ out,
    int H, int N, int ratio) {
  extern __shared__ float sw[];
  const long s = blockIdx.x;
  for (int h = threadIdx.x; h < H; h += blockDim.x) sw[h] = w[s * H + h];
  __syncthreads();
  const long base = s * (long)H * N;
  const int lim = (int)((positions[s] + 1) / ratio);
  for (int t = threadIdx.x; t < N; t += blockDim.x) {
    if (t >= lim) { out[s * (long)N + t] = -INFINITY; continue; }
    float acc = 0.f;
    for (int h = 0; h < H; ++h) {
      float v = __bfloat162float(sc[base + (long)h * N + t]);
      if (v > 0.f) acc += v * sw[h];
    }
    out[s * (long)N + t] = acc;
  }
}

torch::Tensor index_score_reduce_positions(
    torch::Tensor score, torch::Tensor weights, int64_t ratio,
    torch::Tensor positions) {
  at::cuda::CUDAGuard guard(score.device());
  TORCH_CHECK(score.is_cuda() && weights.is_cuda() && positions.is_cuda(),
              "score, weights, and positions must be CUDA tensors");
  TORCH_CHECK(positions.scalar_type() == torch::kInt64 && positions.is_contiguous(),
              "positions must be contiguous CUDA int64");
  TORCH_CHECK(ratio > 0, "ratio must be positive");
  auto sc = score.contiguous();
  auto wf = weights.to(torch::kFloat32).contiguous();
  const int64_t B = sc.size(0), S = sc.size(1), H = sc.size(2), N = sc.size(3);
  TORCH_CHECK(positions.numel() >= B * S, "one position is required per score row");
  auto out = at::empty({B, S, N}, sc.options().dtype(torch::kFloat32));
  auto stream = at::cuda::getCurrentCUDAStream();
  index_score_reduce_positions_kernel<<<(int)(B * S), 256, H * sizeof(float), stream>>>(
      (const __nv_bfloat16*)sc.data_ptr(), wf.data_ptr<float>(),
      positions.data_ptr<int64_t>(), out.data_ptr<float>(),
      (int)H, (int)N, (int)ratio);
  return out;
}



// ---- single-kernel exact top-K select fused with mask+offset post ----
// Replaces at::topk (mbtopk multi-kernel chain) + topk_post[_positions].
// Output set is exact top-K; emission order is deterministic for stable FP reduction.
__device__ __forceinline__ unsigned f2key(float v) {
  unsigned u = __float_as_uint(v);
  return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

__global__ void topk_select_post_kernel(
    const float* __restrict__ score,        // [S, N]
    int* __restrict__ out,                  // [S, K]
    const int64_t* __restrict__ offsets,    // per-row, or null -> offset0
    const int64_t* __restrict__ positions,  // per-row, or null -> pos0 + s
    int N, int K, int ratio, long offset0, long pos0) {
  const long s = blockIdx.x;
  const float* row = score + s * (long)N;
  const int lim = (int)(((positions ? positions[s] : pos0 + s) + 1) / ratio);
  // Live prefix: the tail [lim, N) is -INF by construction, so skipping it is
  // exact.  This is what keeps one static graph cheap on a 1M-capacity pool.
  const int NL = (lim <= 0) ? 0 : ((lim < N) ? lim : N);
  const long off = offsets ? offsets[s] : offset0;

  if (NL <= K) {
    for (int t = threadIdx.x; t < K; t += blockDim.x)
      out[s * (long)K + t] = (t < NL) ? (int)(t + off) : -1;
    return;
  }

  __shared__ int hist[256];
  __shared__ unsigned s_prefix;
  __shared__ int s_k;
  constexpr int EMIT_ITEMS = 8;
  __shared__ int warp_gt[32 * EMIT_ITEMS], warp_eq[32 * EMIT_ITEMS];
  __shared__ int base_gt, base_eq, chunk_gt, chunk_eq;
  if (threadIdx.x == 0) { s_prefix = 0u; s_k = K; }
  __syncthreads();

  for (int pass = 3; pass >= 0; --pass) {
    for (int i = threadIdx.x; i < 256; i += blockDim.x) hist[i] = 0;
    __syncthreads();
    const unsigned mask_hi = (pass == 3) ? 0u : (0xFFFFFFFFu << ((pass + 1) * 8));
    const unsigned prefix = s_prefix;
    const int N4 = ((N & 3) == 0) ? (NL >> 2) : 0;  // row offset s*N must stay 16B-aligned for float4
    const float4* row4 = reinterpret_cast<const float4*>(row);
    for (int t = threadIdx.x; t < N4; t += blockDim.x) {
      const float4 v = row4[t];
      const float vv[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const unsigned u = f2key(vv[j]);
        if ((u & mask_hi) == prefix) atomicAdd(&hist[(u >> (pass * 8)) & 0xFF], 1);
      }
    }
    for (int t = (N4 << 2) + threadIdx.x; t < NL; t += blockDim.x) {
      const unsigned u = f2key(row[t]);
      if ((u & mask_hi) == prefix) atomicAdd(&hist[(u >> (pass * 8)) & 0xFF], 1);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      int kk = s_k, b = 255;
      for (; b > 0; --b) { if (hist[b] >= kk) break; kk -= hist[b]; }
      s_k = kk;
      s_prefix = s_prefix | ((unsigned)b << (pass * 8));
    }
    __syncthreads();
  }

  // Emit the exact same top-K set in a deterministic order.  The previous
  // atomic slot allocator made the output permutation depend on warp timing;
  // sparse attention is set-equivalent but its floating-point reduction is not
  // permutation invariant.  Integer radix histograms above remain deterministic.
  const unsigned thr = s_prefix;
  const int need_tie = s_k;
  const int n_gt = K - need_tie;
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const unsigned lane_mask = (lane == 0) ? 0u : ((1u << lane) - 1u);
  if (threadIdx.x == 0) { base_gt = 0; base_eq = 0; }
  __syncthreads();
  for (int begin = 0; begin < NL; begin += blockDim.x * EMIT_ITEMS) {
    unsigned mask_gt[EMIT_ITEMS], mask_eq[EMIT_ITEMS];
#pragma unroll
    for (int item = 0; item < EMIT_ITEMS; ++item) {
      const int t = begin + item * blockDim.x + threadIdx.x;
      const unsigned u = (t < NL) ? f2key(row[t]) : 0u;
      mask_gt[item] = __ballot_sync(0xffffffffu, t < NL && u > thr);
      mask_eq[item] = __ballot_sync(0xffffffffu, t < NL && u == thr);
      if (lane == 0) {
        const int segment = item * 32 + warp;
        warp_gt[segment] = __popc(mask_gt[item]);
        warp_eq[segment] = __popc(mask_eq[item]);
      }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      int pg = 0, pe = 0;
      for (int segment = 0; segment < 32 * EMIT_ITEMS; ++segment) {
        const int cg = warp_gt[segment], ce = warp_eq[segment];
        warp_gt[segment] = pg; warp_eq[segment] = pe;
        pg += cg; pe += ce;
      }
      chunk_gt = pg; chunk_eq = pe;
    }
    __syncthreads();
#pragma unroll
    for (int item = 0; item < EMIT_ITEMS; ++item) {
      const int t = begin + item * blockDim.x + threadIdx.x;
      const int segment = item * 32 + warp;
      if (mask_gt[item] & (1u << lane)) {
        const int rank = base_gt + warp_gt[segment] + __popc(mask_gt[item] & lane_mask);
        if (rank < n_gt)
          out[s * (long)K + rank] = (t >= lim) ? -1 : (int)(t + off);
      }
      if (mask_eq[item] & (1u << lane)) {
        const int rank = base_eq + warp_eq[segment] + __popc(mask_eq[item] & lane_mask);
        if (rank < need_tie)
          out[s * (long)K + n_gt + rank] = (t >= lim) ? -1 : (int)(t + off);
      }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      base_gt += chunk_gt;
      base_eq += chunk_eq;
    }
    __syncthreads();
  }
}

torch::Tensor topk_select_post(torch::Tensor score, int64_t K, int64_t ratio,
                               int64_t offset, int64_t pos0) {
  at::cuda::CUDAGuard guard(score.device());
  auto sc = score.contiguous();  // [B, S, N] float32
  const int64_t B = sc.size(0), S = sc.size(1), N = sc.size(2);
  auto out = at::full({B, S, K}, -1, sc.options().dtype(torch::kInt32));
  auto stream = at::cuda::getCurrentCUDAStream();
  topk_select_post_kernel<<<(int)(B * S), 1024, 0, stream>>>(
      sc.data_ptr<float>(), out.data_ptr<int>(), nullptr, nullptr,
      (int)N, (int)K, (int)ratio, (long)offset, (long)pos0);
  return out;
}

torch::Tensor topk_select_post_positions(torch::Tensor score, int64_t K,
                                         int64_t ratio, torch::Tensor offsets,
                                         torch::Tensor positions) {
  at::cuda::CUDAGuard guard(score.device());
  auto sc = score.contiguous();
  const int64_t B = sc.size(0), S = sc.size(1), N = sc.size(2);
  auto out = at::full({B, S, K}, -1, sc.options().dtype(torch::kInt32));
  auto stream = at::cuda::getCurrentCUDAStream();
  topk_select_post_kernel<<<(int)(B * S), 1024, 0, stream>>>(
      sc.data_ptr<float>(), out.data_ptr<int>(),
      offsets.data_ptr<int64_t>(), positions.data_ptr<int64_t>(),
      (int)N, (int)K, (int)ratio, 0L, 0L);
  return out;
}

// ---- fused rotate_activation (fp32 Hadamard butterfly, scale d^-0.5, bf16 round) +
//      fp4_act_quant_sim (32-block e2m1 qdq, pow2 scale exp2(ceil(log2(amax/6)))), in place.
//      Bitwise mirror of model/qmath.py rotate_activation + fp4_act_quant_sim.
__global__ void had_fp4_qdq_kernel(__nv_bfloat16* __restrict__ x, int d, float scale) {
  extern __shared__ float sh[];
  float* a = sh;
  float* b = sh + d;
  const long row = blockIdx.x;
  const int j = threadIdx.x;
  __nv_bfloat16* xr = x + row * (long)d;
  a[j] = __bfloat162float(xr[j]);
  __syncthreads();
  for (int h = 1; h < d; h <<= 1) {
    const int g = j / (2 * h), k = j - g * 2 * h;
    const int base = g * 2 * h;
    float v;
    if (k < h) v = a[base + k] + a[base + k + h];
    else       v = a[base + k - h] - a[base + k];
    b[j] = v;
    __syncthreads();
    float* t = a; a = b; b = t;
  }
  // rotate output is materialised as bf16 before quant (matches .to(x.dtype))
  const float z = __bfloat162float(__float2bfloat16_rn(a[j] * scale));
  float am = fabsf(z);
  #pragma unroll
  for (int o = 16; o > 0; o >>= 1) am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, o));
  am = fmaxf(am, 6.0f * 1.1754943508222875e-38f);   // 6 * 2^-126
  // exact pow2 scale under --use_fast_math: t = am/6 (IEEE div), s = 2^ceil(log2 t) via frexp
  const float t = __fdiv_rn(am, 6.0f);
  int e; const float mant = frexpf(t, &e);       // t = mant * 2^e, mant in [0.5,1)
  const float s = ldexpf(1.0f, (mant == 0.5f) ? e - 1 : e);
  float q = __fdiv_rn(z, s);
  q = fminf(fmaxf(q, -6.0f), 6.0f);
  const float aq = fabsf(q);
  // torch.bucketize(right=False) over midpoints of [0,.5,1,1.5,2,3,4,6]
  const float bnd[7] = {0.25f, 0.75f, 1.25f, 1.75f, 2.5f, 3.5f, 5.0f};
  const float lut[8] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
  int idx = 0;
  #pragma unroll
  for (int k = 0; k < 7; ++k) idx += (bnd[k] < aq) ? 1 : 0;
  const float sg = (q > 0.f) ? 1.f : ((q < 0.f) ? -1.f : 0.f);
  xr[j] = __float2bfloat16_rn(lut[idx] * sg * s);
}

torch::Tensor had_fp4_qdq_(torch::Tensor x) {
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == torch::kBFloat16 && x.is_contiguous());
  const int d = (int)x.size(-1);
  TORCH_CHECK(d == 64 || d == 128 || d == 256 || d == 512, "had_fp4_qdq_: d must be pow2 in [64,512]");
  const long rows = x.numel() / d;
  if (rows == 0) return x;
  const c10::cuda::CUDAGuard guard(x.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  had_fp4_qdq_kernel<<<(int)rows, d, 2 * d * sizeof(float), stream>>>(
      reinterpret_cast<__nv_bfloat16*>(x.data_ptr()), d, (float)std::pow((double)d, -0.5));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return x;
}


// >>> FUSED_IDX_BEGIN
// ---- fused gather + indexer score with device-length early exit (FA-style) ----
// Replaces: pool.flat.index_select(0, prow) -> einsum("bshd,btd->bsht") ->
//           index_score_reduce_positions(...).  Same math, but rows beyond the
//           real sequence length (lim = (pos+1)/ratio) are never touched: the
//           work scales with the actual length, not with the pool capacity.
// Fast path (H==8, D==128): one warp per compressed row, lanes split 8 heads x
// 4 segments of 32 elements -> 16B vector loads, 7 shuffles per row.
template <int H_, int D_>
__global__ void index_score_fused_w8_kernel(
    const __nv_bfloat16* __restrict__ q,      // [S, H, D]
    const __nv_bfloat16* __restrict__ pool,   // [R, D]
    const int64_t* __restrict__ prow,         // [N] shared, or [B, N] (prow_stride = N)
    const float* __restrict__ w,              // [B*S, H]
    const int64_t* __restrict__ positions,    // [B*S]
    float* __restrict__ out,                  // [B*S, N], prefilled with -inf
    long N, int ratio, int rows_per_block, int S, long prow_stride) {
  constexpr int SEG = D_ / 4;                 // elements handled by one lane
  const int s = blockIdx.y;                   // flat query row over B*S
  prow += (long)(s / S) * prow_stride;        // A9: per-batch row ids (0 stride = shared)
  const int lim = (int)((positions[s] + 1) / ratio);
  const long t0 = (long)blockIdx.x * rows_per_block;
  if (t0 >= lim) return;                      // <<< early exit, nothing to do

  extern __shared__ char smem_raw[];
  __nv_bfloat16* sq = reinterpret_cast<__nv_bfloat16*>(smem_raw);
  float* sw = reinterpret_cast<float*>(sq + H_ * D_);
  for (int i = threadIdx.x; i < H_ * D_; i += blockDim.x) sq[i] = q[(long)s * H_ * D_ + i];
  for (int i = threadIdx.x; i < H_; i += blockDim.x) sw[i] = w[(long)s * H_ + i];
  __syncthreads();

  const unsigned msk = 0xffffffffu;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int nwarp = blockDim.x >> 5;
  const int g = lane >> 2;                    // head for this lane
  const int j = lane & 3;                     // segment for this lane
  const __nv_bfloat16* qp = sq + g * D_ + j * SEG;

  for (int u = warp; u < rows_per_block; u += nwarp) {
    const long t = t0 + u;
    if (t >= lim) break;
    const __nv_bfloat16* kvp = pool + prow[t] * (long)D_ + j * SEG;
    float p = 0.f;
    #pragma unroll
    for (int e = 0; e < SEG; e += 8) {
      const float4 a = *reinterpret_cast<const float4*>(kvp + e);
      const float4 b = *reinterpret_cast<const float4*>(qp + e);
      const __nv_bfloat16* av = reinterpret_cast<const __nv_bfloat16*>(&a);
      const __nv_bfloat16* bv = reinterpret_cast<const __nv_bfloat16*>(&b);
      #pragma unroll
      for (int k = 0; k < 8; ++k) p += __bfloat162float(av[k]) * __bfloat162float(bv[k]);
    }
    p += __shfl_xor_sync(msk, p, 1);
    p += __shfl_xor_sync(msk, p, 2);          // lanes of a head now share the dot
    const float v = __bfloat162float(__float2bfloat16(p));  // match bf16 einsum out
    float c = (j == 0 && v > 0.f) ? v * sw[g] : 0.f;
    #pragma unroll
    for (int o = 4; o < 32; o <<= 1) c += __shfl_xor_sync(msk, c, o);
    if (lane == 0) out[(long)s * N + t] = c;
  }
}

// Multi-query fast path (B==1, S==8, H==8, D==128): a compressed row is loaded
// once into registers and reused by all S queries -> 8x less KV traffic and
// enough ILP to hide the load latency.
template <int H_, int D_, int S_>
__global__ void index_score_fused_ms_kernel(
    const __nv_bfloat16* __restrict__ q,      // [S, H, D]
    const __nv_bfloat16* __restrict__ pool,   // [R, D]
    const int64_t* __restrict__ prow,         // [N] shared, or [B, N] (prow_stride = N)
    const float* __restrict__ w,              // [B, S, H]
    const int64_t* __restrict__ positions,    // [B, S]
    float* __restrict__ out,                  // [B, S, N], prefilled with -inf
    long N, int ratio, int rows_per_block, long prow_stride) {
  constexpr int SEG = D_ / 4;                 // elements per lane
  constexpr int NV = SEG / 8;                 // 16B vectors per lane
  // A9: blockIdx.y = batch row.  Each batch row is an independent copy of the
  // B==1 problem (own q/w/positions/out slice, own row-id table); the per-output
  // reduction order is untouched, so B==1 stays bitwise identical to before.
  const int b = blockIdx.y;
  q += (long)b * S_ * H_ * D_;
  w += (long)b * S_ * H_;
  positions += (long)b * S_;
  out += (long)b * S_ * N;
  prow += (long)b * prow_stride;
  extern __shared__ char smem_raw[];
  __nv_bfloat16* sq = reinterpret_cast<__nv_bfloat16*>(smem_raw);
  float* sw = reinterpret_cast<float*>(sq + S_ * H_ * D_);
  int* slim = reinterpret_cast<int*>(sw + S_ * H_);
  if (threadIdx.x < S_) slim[threadIdx.x] = (int)((positions[threadIdx.x] + 1) / ratio);
  __syncthreads();

  int limmax = 0;
  #pragma unroll
  for (int s = 0; s < S_; ++s) limmax = max(limmax, slim[s]);
  const long t0 = (long)blockIdx.x * rows_per_block;
  if (t0 >= limmax) return;                   // <<< early exit BEFORE smem q load
  for (int i = threadIdx.x; i < S_ * H_ * D_; i += blockDim.x) sq[i] = q[i];
  for (int i = threadIdx.x; i < S_ * H_; i += blockDim.x) sw[i] = w[i];
  __syncthreads();

  const unsigned msk = 0xffffffffu;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int nwarp = blockDim.x >> 5;
  const int g = lane >> 2, j = lane & 3;

  for (int u = warp; u < rows_per_block; u += nwarp) {
    const long t = t0 + u;
    if (t >= limmax) break;
    const __nv_bfloat16* kvp = pool + prow[t] * (long)D_ + j * SEG;
    float4 kv4[NV];
    #pragma unroll
    for (int e = 0; e < NV; ++e) kv4[e] = *reinterpret_cast<const float4*>(kvp + e * 8);
    #pragma unroll
    for (int s = 0; s < S_; ++s) {
      if (t >= slim[s]) continue;
      const __nv_bfloat16* qp = sq + ((long)s * H_ + g) * D_ + j * SEG;
      float p = 0.f;
      #pragma unroll
      for (int e = 0; e < NV; ++e) {
        const float4 b = *reinterpret_cast<const float4*>(qp + e * 8);
        const __nv_bfloat16* av = reinterpret_cast<const __nv_bfloat16*>(&kv4[e]);
        const __nv_bfloat16* bv = reinterpret_cast<const __nv_bfloat16*>(&b);
        #pragma unroll
        for (int k = 0; k < 8; ++k) p += __bfloat162float(av[k]) * __bfloat162float(bv[k]);
      }
      p += __shfl_xor_sync(msk, p, 1);
      p += __shfl_xor_sync(msk, p, 2);
      const float v = __bfloat162float(__float2bfloat16(p));
      float c = (j == 0 && v > 0.f) ? v * sw[s * H_ + g] : 0.f;
      #pragma unroll
      for (int o = 4; o < 32; o <<= 1) c += __shfl_xor_sync(msk, c, o);
      if (lane == 0) out[(long)s * N + t] = c;
    }
  }
}

// Generic fallback: one warp per row, D/32 elements per lane.
__global__ void index_score_fused_kernel(
    const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ pool,
    const int64_t* __restrict__ prow, const float* __restrict__ w,
    const int64_t* __restrict__ positions, float* __restrict__ out,
    int H, int D, long N, int ratio, int rows_per_block, int S, long prow_stride) {
  const int s = blockIdx.y;                   // flat query row over B*S
  prow += (long)(s / S) * prow_stride;        // A9: per-batch row ids (0 stride = shared)
  const int lim = (int)((positions[s] + 1) / ratio);
  const long t0 = (long)blockIdx.x * rows_per_block;
  if (t0 >= lim) return;

  extern __shared__ char smem_raw[];
  __nv_bfloat16* sq = reinterpret_cast<__nv_bfloat16*>(smem_raw);
  float* sw = reinterpret_cast<float*>(sq + (long)H * D);
  for (int i = threadIdx.x; i < H * D; i += blockDim.x) sq[i] = q[(long)s * H * D + i];
  for (int i = threadIdx.x; i < H; i += blockDim.x) sw[i] = w[(long)s * H + i];
  __syncthreads();

  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int nwarp = blockDim.x >> 5;
  const int E = D >> 5;
  for (int u = warp; u < rows_per_block; u += nwarp) {
    const long t = t0 + u;
    if (t >= lim) break;
    const __nv_bfloat16* kv = pool + prow[t] * (long)D + lane * E;
    float acc = 0.f;
    for (int h = 0; h < H; ++h) {
      const __nv_bfloat16* qh = sq + (long)h * D + lane * E;
      float p = 0.f;
      for (int e = 0; e < E; ++e) p += __bfloat162float(kv[e]) * __bfloat162float(qh[e]);
      #pragma unroll
      for (int o = 16; o > 0; o >>= 1) p += __shfl_down_sync(0xffffffffu, p, o);
      if (lane == 0) {
        const float v = __bfloat162float(__float2bfloat16(p));
        if (v > 0.f) acc += v * sw[h];
      }
    }
    if (lane == 0) out[(long)s * N + t] = acc;
  }
}

torch::Tensor index_score_fused(torch::Tensor q, torch::Tensor pool, torch::Tensor prow,
                                torch::Tensor weights, torch::Tensor positions,
                                int64_t ratio) {
  at::cuda::CUDAGuard guard(q.device());
  TORCH_CHECK(q.is_cuda() && q.scalar_type() == torch::kBFloat16, "q must be cuda bf16");
  TORCH_CHECK(pool.is_cuda() && pool.scalar_type() == torch::kBFloat16 && pool.is_contiguous(),
              "pool must be contiguous cuda bf16");
  TORCH_CHECK(prow.is_cuda() && prow.scalar_type() == torch::kInt64 && prow.is_contiguous(),
              "prow must be contiguous cuda int64");
  TORCH_CHECK(positions.is_cuda() && positions.scalar_type() == torch::kInt64 &&
              positions.is_contiguous(), "positions must be contiguous cuda int64");
  TORCH_CHECK(ratio > 0, "ratio must be positive");
  auto qc = q.contiguous();
  auto wf = weights.to(torch::kFloat32).contiguous();
  const int64_t B = qc.size(0), S = qc.size(1), H = qc.size(2), D = qc.size(3);
  // A9: prow is either [N] (one row-id table shared by every batch row) or
  // [B, N] (one table per batch row: batched multi-slot indexer score).
  TORCH_CHECK(prow.dim() == 1 || (prow.dim() == 2 && prow.size(0) == B),
              "prow must be [N] or [B, N]");
  const int64_t N = prow.size(-1);
  const long prow_stride = prow.dim() == 2 ? (long)N : 0L;
  TORCH_CHECK(D % 32 == 0 && D <= 256, "D must be a multiple of 32 and <= 256");
  TORCH_CHECK(pool.size(1) == D, "pool row width must equal head dim");
  TORCH_CHECK(positions.numel() >= B * S, "one position per query row");
  auto out = at::full({B, S, N}, -std::numeric_limits<float>::infinity(),
                      qc.options().dtype(torch::kFloat32));
  const int block = 256, rpb = 32;
  dim3 grid((unsigned)((N + rpb - 1) / rpb), (unsigned)(B * S));
  const size_t shm = (size_t)H * D * sizeof(__nv_bfloat16) + (size_t)H * sizeof(float);
  auto stream = at::cuda::getCurrentCUDAStream();
  const __nv_bfloat16* qptr = (const __nv_bfloat16*)qc.data_ptr();
  const __nv_bfloat16* pptr = (const __nv_bfloat16*)pool.data_ptr();
  // ---- PERF FORK (numerically equivalent, verified) ----
  // Why fork: B==1&&S==8 lets one block own all 8 query rows, so the KV pool line is read once
  // instead of once per row. It is a *scheduling* specialization only.
  // Why it is safe: each output's reduction order (E-dim shfl_down, then h=0..H-1 accumulate)
  // is identical in ms/w8/generic kernels, so results are BITWISE equal, not just close.
  // Verified 2026-09-05: same q fed as B=1 (ms path) vs B=2 (w8 path) -> max_abs_diff=0.0, n_diff=0/2048.
  // RULE: any future fork here must keep per-output reduction order identical, else batch invariance breaks.
  // A9 (2026-09-05): the ms path now takes any B (grid.y = batch); each batch row
  // is an independent B==1 problem, so B==1 is bitwise unchanged and B>1 gets the
  // same 8x KV-traffic saving instead of falling to the per-row w8 kernel.
  if (S == 8 && H == 8 && D == 128) {
    const size_t shm_ms = (size_t)S * H * D * sizeof(__nv_bfloat16) +
                          (size_t)S * H * sizeof(float) + (size_t)S * sizeof(int);
    dim3 grid1((unsigned)((N + rpb - 1) / rpb), (unsigned)B);
    index_score_fused_ms_kernel<8, 128, 8><<<grid1, block, shm_ms, stream>>>(
        qptr, pptr, prow.data_ptr<int64_t>(), wf.data_ptr<float>(),
        positions.data_ptr<int64_t>(), out.data_ptr<float>(), (long)N, (int)ratio, rpb,
        prow_stride);
  } else if (H == 8 && D == 128) {
    index_score_fused_w8_kernel<8, 128><<<grid, block, shm, stream>>>(
        qptr, pptr, prow.data_ptr<int64_t>(), wf.data_ptr<float>(),
        positions.data_ptr<int64_t>(), out.data_ptr<float>(), (long)N, (int)ratio, rpb,
        (int)S, prow_stride);
  } else {
    index_score_fused_kernel<<<grid, block, shm, stream>>>(
        qptr, pptr, prow.data_ptr<int64_t>(), wf.data_ptr<float>(),
        positions.data_ptr<int64_t>(), out.data_ptr<float>(), (int)H, (int)D, (long)N,
        (int)ratio, rpb, (int)S, prow_stride);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}
// <<< FUSED_IDX_END

#ifndef DSV4_NO_PYBIND
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("index_score_fused", &index_score_fused,
        "fused gather+score+relu*w+sum_h with device-length early exit");
  m.def("had_fp4_qdq_", &had_fp4_qdq_, "in-place fused hadamard rotate + fp4 e2m1 qdq (32-block)");
  m.def("index_score_reduce", &index_score_reduce, "fused relu+weight+sum_h+mask");
  m.def("index_score_reduce_positions", &index_score_reduce_positions,
        "graph-safe fused index score with per-row CUDA positions");
  m.def("topk_select_post", &topk_select_post,
        "single-kernel unordered radix top-K fused with mask+offset");
  m.def("topk_select_post_positions", &topk_select_post_positions,
        "single-kernel unordered radix top-K with per-row CUDA metadata");
}
#endif
