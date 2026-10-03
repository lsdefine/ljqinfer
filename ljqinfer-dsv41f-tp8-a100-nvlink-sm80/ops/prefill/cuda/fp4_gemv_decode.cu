
// FP4 MoE decode GEMV (drop-in for fp4_grouped_gemm_device).
// Decode shape: few rows per expert. Per warp: COLS output columns; lanes split K.
// Dequant uses the SAME [256][16] LUT as the reference path -> bit-identical numerics.
// Deterministic: fixed K order + warp shuffle tree reduction, no atomics.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>

#define SCAN_THREADS 512
#define TILE_ROWS 2
#define COLS 4
#define WARPS 8
#define NPB (COLS * WARPS)

__global__ void fp4_dec_build_tiles(const int64_t* __restrict__ counts,
                                    const int64_t* __restrict__ offsets,
                                    int* __restrict__ te, int* __restrict__ tr0,
                                    int* __restrict__ trn, int E, int maxT) {
  __shared__ int sh[SCAN_THREADS];
  const int t = threadIdx.x;
  for (int i = t; i < maxT; i += SCAN_THREADS) te[i] = -1;
  const int n = (t < E) ? (int)((counts[t] + (TILE_ROWS - 1)) / TILE_ROWS) : 0;
  sh[t] = n;
  __syncthreads();
  for (int off = 1; off < SCAN_THREADS; off <<= 1) {
    int v = (t >= off) ? sh[t - off] : 0;
    __syncthreads();
    sh[t] += v;
    __syncthreads();
  }
  const int start = sh[t] - n;
  if (t < E && n > 0) {
    const int r0 = (int)offsets[t], c = (int)counts[t];
    for (int i = 0; i < n; ++i) {
      const int idx = start + i;
      if (idx >= maxT) break;
      te[idx] = t;
      tr0[idx] = r0 + i * TILE_ROWS;
      trn[idx] = min(TILE_ROWS, c - i * TILE_ROWS);
    }
  }
}

template <int BPL>
__global__ __launch_bounds__(WARPS * 32, 4) void fp4_gemv_dec(
    const __nv_bfloat16* __restrict__ x, const unsigned char* __restrict__ w,
    const unsigned char* __restrict__ s, const __nv_bfloat16* __restrict__ lut,
    const int* __restrict__ te, const int* __restrict__ tr0, const int* __restrict__ trn,
    __nv_bfloat16* __restrict__ out, int N, int K) {
  // ue8m0 scales are pure powers of two (fp4_unpack: ldexpf(v, s-127)), so the
  // [256][16] table factorises into 16 base values times 2^(s-127). Keeping only
  // the 16 bases kills the 8 KB shared LUT and its random-access bank conflicts.
  // one private copy of the 16 bases per lane, stride 17 (coprime with 32 banks)
  // so all 32 lanes of a warp hit distinct banks -> conflict-free random indexing
  __shared__ float fpb[32 * 17];
  const int tid = threadIdx.x;
  const int e = te[blockIdx.x];
  if (e < 0) return;                 // empty tile: bail out before any load
  for (int i = tid; i < 32 * 17; i += WARPS * 32) {   // 544 slots > 256 threads
    const int j = i % 17;
    fpb[i] = (j < 16) ? __bfloat162float(lut[127 * 16 + j]) : 0.f;
  }
  __syncthreads();
  const float* fp = fpb + (threadIdx.x & 31) * 17;

  const int r0 = tr0[blockIdx.x], rn = trn[blockIdx.x];
  const int warp = tid >> 5, lane = tid & 31;
  const size_t wrow = (size_t)K >> 1, srow = (size_t)K >> 5;
  const unsigned char* wE = w + (size_t)e * N * wrow;
  const unsigned char* sE = s + (size_t)e * N * srow;
  const int col0 = blockIdx.y * NPB + warp * COLS;

  constexpr int KPL = BPL * 2;
  constexpr int KPR = KPL * 32;
  float acc[COLS][TILE_ROWS];
#pragma unroll
  for (int c = 0; c < COLS; ++c)
#pragma unroll
    for (int m = 0; m < TILE_ROWS; ++m) acc[c][m] = 0.f;

  for (int base = 0; base < K; base += KPR) {
    const int k = base + lane * KPL;
    if (k >= K) break;
    // weights + one power-of-two scale per column (KPL <= 32 elems share a scale group)
    float wv[COLS][KPL];
    float sc[COLS];
#pragma unroll
    for (int c = 0; c < COLS; ++c) {
      const unsigned char* wp = wE + (size_t)(col0 + c) * wrow + (k >> 1);
      sc[c] = ldexpf(1.f, (int)sE[(size_t)(col0 + c) * srow + (k >> 5)] - 127);
#pragma unroll
      for (int b2 = 0; b2 < BPL; ++b2) {
        const unsigned int byte = wp[b2];
        wv[c][2 * b2] = fp[byte & 15];
        wv[c][2 * b2 + 1] = fp[byte >> 4];
      }
    }
    // activations: loaded ONCE per row, then reused across all COLS columns
#pragma unroll
    for (int m = 0; m < TILE_ROWS; ++m) {
      if (m >= rn) break;
      const __nv_bfloat16* xp = x + (size_t)(r0 + m) * K + k;
      float xv[KPL];
      if (KPL == 8) {                       // k is 8-aligned -> one 16B vector load
        const float4 r4 = *reinterpret_cast<const float4*>(xp);
        const __nv_bfloat162* xb = reinterpret_cast<const __nv_bfloat162*>(&r4);
#pragma unroll
        for (int j = 0; j < KPL / 2; ++j) {
          const float2 f2 = __bfloat1622float2(xb[j]);
          xv[2 * j] = f2.x;
          xv[2 * j + 1] = f2.y;
        }
      } else {
#pragma unroll
        for (int j = 0; j < KPL; ++j) xv[j] = __bfloat162float(xp[j]);
      }
#pragma unroll
      for (int c = 0; c < COLS; ++c) {
        float p = 0.f;
#pragma unroll
        for (int j = 0; j < KPL; ++j) p = fmaf(xv[j], wv[c][j], p);
        acc[c][m] = fmaf(p, sc[c], acc[c][m]);
      }
    }
  }

#pragma unroll
  for (int c = 0; c < COLS; ++c)
#pragma unroll
    for (int m = 0; m < TILE_ROWS; ++m) {
      if (m >= rn) break;
      float v = acc[c][m];
#pragma unroll
      for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffffu, v, off);
      if (lane == 0) out[(size_t)(r0 + m) * N + col0 + c] = __float2bfloat16(v);
    }
}

void fp4_grouped_gemm_device(torch::Tensor x, torch::Tensor w, torch::Tensor s,
                             torch::Tensor lut, torch::Tensor counts,
                             torch::Tensor offsets, torch::Tensor tile_e,
                             torch::Tensor tile_r0, torch::Tensor tile_rn,
                             torch::Tensor out) {
  at::cuda::CUDAGuard guard(x.device());
  const int64_t K = x.size(1), E = w.size(0), N = w.numel() / (E * (K / 2));
  TORCH_CHECK(x.dtype() == torch::kBFloat16 && out.dtype() == torch::kBFloat16, "BF16 activations");
  TORCH_CHECK(lut.dtype() == torch::kBFloat16 && lut.numel() == 256 * 16, "fp4 lut");
  TORCH_CHECK(N % NPB == 0 && K % 32 == 0, "fp4 dec gemv: N%32, K%32");
  TORCH_CHECK(out.size(0) == x.size(0) && out.size(1) == N, "out shape");
  TORCH_CHECK(E <= SCAN_THREADS, "expert count above scan width");
  const int maxT = (int)tile_e.numel();
  auto stream = at::cuda::getCurrentCUDAStream();
  fp4_dec_build_tiles<<<1, SCAN_THREADS, 0, stream>>>(
      counts.data_ptr<int64_t>(), offsets.data_ptr<int64_t>(), tile_e.data_ptr<int>(),
      tile_r0.data_ptr<int>(), tile_rn.data_ptr<int>(), (int)E, maxT);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  dim3 grid((unsigned)maxT, (unsigned)(N / NPB));
  auto X = (const __nv_bfloat16*)x.data_ptr();
  auto W = reinterpret_cast<const unsigned char*>(w.data_ptr());
  auto S = reinterpret_cast<const unsigned char*>(s.data_ptr());
  auto L = (const __nv_bfloat16*)lut.data_ptr();
  auto O = (__nv_bfloat16*)out.data_ptr();
  auto* TE = tile_e.data_ptr<int>(); auto* T0 = tile_r0.data_ptr<int>(); auto* TN = tile_rn.data_ptr<int>();
  fp4_gemv_dec<4><<<grid, WARPS * 32, 0, stream>>>(X, W, S, L, TE, T0, TN, O, (int)N, (int)K);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fp4_grouped_gemm_device", &fp4_grouped_gemm_device, "FP4 decode GEMV (SM80)");
}
