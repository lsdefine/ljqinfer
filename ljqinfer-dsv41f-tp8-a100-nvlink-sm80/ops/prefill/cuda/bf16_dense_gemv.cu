// BF16 dense decode GEMV (M<=8): out[m,n] = sum_k a[m,k] * w[n,k].
//
// Why this kernel exists: at decode the dense projections run with M=6 rows, so
// they are pure weight-bandwidth problems (5120-wide K, N of 640..5120 per
// rank).  cuBLAS answers them with 64x64/128x64 tensor-core tiles plus a
// split-K reduce pass, which reads the weight at ~200 GB/s -- 13% of HBM -- and
// leaves 234 extra splitKreduce launches per round.
//
// Mapping follows the lesson already written down in fp4_gemv_decode.cu: one
// lane must own SEVERAL output rows so that several independent uint4 weight
// loads are in flight (ILP), and the 32 lanes of a warp must split K so every
// warp step is one perfectly coalesced 32*16 B = 512 B burst per owned row.
//
//   * COLS output rows per lane-group, one warp = 32 lanes splitting K
//   * each lane reads uint4 = 8 bf16 of every owned row  -> COLS loads in flight
//   * the activation slice (M rows x 8 bf16) is read once per step and reused
//     across all COLS rows; it is tiny and stays resident in L1/L2
//   * fp32 accumulation, warp shuffle reduce, fixed-order in-block K-split
//     reduce (no atomics) so the result is deterministic run to run
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>

#define WARPS 8

template <int COLS, int KS, int MT>
__global__ __launch_bounds__(WARPS * 32) void bf16_gemv(
    const __nv_bfloat16* __restrict__ A, const __nv_bfloat16* __restrict__ W,
    __nv_bfloat16* __restrict__ O, int N, int K) {
  // warp w owns rows [base + (w/KS)*COLS, +COLS), K-split group ks = w%KS
  constexpr int RW = WARPS / KS;                  // row groups per block
  __shared__ float red[RW * COLS][KS][MT];
  const int wid = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int rg = wid / KS, ks = wid % KS;
  const int row0 = (blockIdx.x * RW + rg) * COLS;

  float acc[COLS][MT];
#pragma unroll
  for (int c = 0; c < COLS; ++c)
#pragma unroll
    for (int m = 0; m < MT; ++m) acc[c][m] = 0.f;

  const int step = KS * 32 * 8;
  for (int k = (ks * 32 + lane) * 8; k < K; k += step) {
    uint4 wv[COLS];
#pragma unroll
    for (int c = 0; c < COLS; ++c) {
      const int row = row0 + c;
      const size_t off = (size_t)(row < N ? row : N - 1) * K + k;
      wv[c] = *reinterpret_cast<const uint4*>(W + off);
    }
#pragma unroll
    for (int m = 0; m < MT; ++m) {
      const uint4 av = *reinterpret_cast<const uint4*>(A + (size_t)m * K + k);
      const __nv_bfloat162* a2 = reinterpret_cast<const __nv_bfloat162*>(&av);
      // convert the activation slice once and reuse it for every owned row:
      // inside the row loop it would be COLS times the cvt traffic and the
      // kernel turns ALU-bound before it saturates HBM.
      float2 af[4];
#pragma unroll
      for (int j = 0; j < 4; ++j) af[j] = __bfloat1622float2(a2[j]);
#pragma unroll
      for (int c = 0; c < COLS; ++c) {
        const __nv_bfloat162* w2 = reinterpret_cast<const __nv_bfloat162*>(&wv[c]);
        float s = 0.f;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          const float2 wf = __bfloat1622float2(w2[j]);
          s += af[j].x * wf.x + af[j].y * wf.y;
        }
        acc[c][m] += s;
      }
    }
  }

#pragma unroll
  for (int c = 0; c < COLS; ++c)
#pragma unroll
    for (int m = 0; m < MT; ++m) {
      float v = acc[c][m];
#pragma unroll
      for (int off = 16; off; off >>= 1) v += __shfl_down_sync(0xffffffffu, v, off);
      if (lane == 0) red[rg * COLS + c][ks][m] = v;
    }

  if (KS == 1) {
    if (lane < COLS) {
      const int row = row0 + lane;
      if (row < N)
#pragma unroll
        for (int m = 0; m < MT; ++m)
          O[(size_t)m * N + row] = __float2bfloat16(red[rg * COLS + lane][0][m]);
    }
    return;
  }
  __syncthreads();
  const int tid = threadIdx.x;
  const int total = RW * COLS * MT;
  for (int t = tid; t < total; t += WARPS * 32) {
    const int rr = t / MT, mm = t - rr * MT;
    const int row = (blockIdx.x * RW) * COLS + rr;
    if (row < N) {
      float s = 0.f;
#pragma unroll
      for (int q = 0; q < KS; ++q) s += red[rr][q][mm];   // fixed order
      O[(size_t)mm * N + row] = __float2bfloat16(s);
    }
  }
}

#define LAUNCH(COLS, KS, MT)                                                      \
  do {                                                                            \
    constexpr int RW = WARPS / (KS);                                              \
    const int blocks = (N + RW * (COLS) - 1) / (RW * (COLS));                     \
    bf16_gemv<COLS, KS, MT><<<blocks, WARPS * 32, 0, stream>>>(                    \
        (const __nv_bfloat16*)a.data_ptr(), (const __nv_bfloat16*)w.data_ptr(),    \
        (__nv_bfloat16*)out.data_ptr(), N, K);                                    \
  } while (0)

#define DISPATCH_M(COLS, KS)                                                      \
  do {                                                                            \
    switch (M) {                                                                  \
      case 1: LAUNCH(COLS, KS, 1); break;                                         \
      case 2: LAUNCH(COLS, KS, 2); break;                                         \
      case 3: LAUNCH(COLS, KS, 3); break;                                         \
      case 4: LAUNCH(COLS, KS, 4); break;                                         \
      case 5: LAUNCH(COLS, KS, 5); break;                                         \
      case 6: LAUNCH(COLS, KS, 6); break;                                         \
      case 7: LAUNCH(COLS, KS, 7); break;                                         \
      default: LAUNCH(COLS, KS, 8); break;                                        \
    }                                                                             \
  } while (0)

// out is [M, N]: the caller hands in the row-major activation [M, K] and the
// row-major weight [N, K] exactly as the dequantized bank stores them.
void dense_gemv_bf16(torch::Tensor a, torch::Tensor w, torch::Tensor out) {
  at::cuda::CUDAGuard guard(a.device());
  TORCH_CHECK(a.is_cuda() && a.scalar_type() == torch::kBFloat16 && a.dim() == 2 && a.is_contiguous(),
              "contiguous CUDA BF16 activation required");
  TORCH_CHECK(w.scalar_type() == torch::kBFloat16 && w.dim() == 2 && w.is_contiguous(),
              "contiguous BF16 weight required");
  TORCH_CHECK(out.scalar_type() == torch::kBFloat16 && out.is_contiguous(), "BF16 out");
  const int M = (int)a.size(0), K = (int)a.size(1), N = (int)w.size(0);
  TORCH_CHECK(w.size(1) == K && M >= 1 && M <= 8 && (K & 7) == 0, "shape/alignment");
  TORCH_CHECK(out.numel() == (int64_t)M * N, "out must hold M*N");
  auto stream = at::cuda::getCurrentCUDAStream();
  // Pick the K-split so that the launch fills the device: A100 has 108 SMs, so
  // aim for >= 216 blocks before giving rows away to extra warps.
  const int rows_per_block_k1 = WARPS * 4;
  if (N >= 216 * rows_per_block_k1) {
    DISPATCH_M(4, 1);
  } else if (N >= 216 * (WARPS / 2) * 4) {
    DISPATCH_M(4, 2);
  } else if (N >= 216 * (WARPS / 4) * 2) {
    DISPATCH_M(2, 4);
  } else {
    DISPATCH_M(1, 8);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("dense_gemv_bf16", &dense_gemv_bf16,
        "BF16 dense decode GEMV for M<=8 (weight-bandwidth bound, deterministic)");
}
