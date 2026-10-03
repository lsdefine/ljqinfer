// FP4 MoE decode GEMV v3 -- ILP-oriented rewrite.
//
// v2 lesson: widening the load (uint4) without raising per-lane load count left
// ILP at 1 and the kernel stayed at ~30% HBM.  V4's kRows trick is that ONE lane
// owns several output rows, so several independent uint4 loads are in flight.
//
// v3 layout:
//   * all 32 lanes split K   -> lane reads q = lane, lane+32, ...  (each column
//     is read as 32 consecutive uint4 = 512B per warp step, perfectly coalesced)
//   * each lane owns COLS=4 output columns -> 4 independent W loads + 4 x loads
//     in flight every step (ILP ~ 8)
//   * the 4 columns share the same activation row, so x is loaded once per step
//
// Numerics identical to the reference tiled kernel: 16 base values taken from
// lut[127] scaled by 2^(s-127), low nibble = even element, fp32 accumulation.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>

#define WARPS 8
#define COLS 4
#define NPB (WARPS * COLS)   // 32 output columns per block
#define SCANT 512

__global__ void fp4_dec3_build_tiles(const int64_t* __restrict__ counts,
                                     const int64_t* __restrict__ offsets,
                                     int* __restrict__ te, int* __restrict__ tr0,
                                     int E, int maxT) {
  __shared__ int sh[SCANT];
  const int t = threadIdx.x;
  for (int i = t; i < maxT; i += SCANT) te[i] = -1;
  const int n = (t < E) ? (int)counts[t] : 0;
  sh[t] = n;
  __syncthreads();
  for (int off = 1; off < SCANT; off <<= 1) {
    const int v = (t >= off) ? sh[t - off] : 0;
    __syncthreads();
    sh[t] += v;
    __syncthreads();
  }
  const int start = sh[t] - n;
  if (t < E && n > 0) {
    const int r0 = (int)offsets[t];
    for (int i = 0; i < n; ++i) {
      const int idx = start + i;
      if (idx >= maxT) break;
      te[idx] = t;
      tr0[idx] = r0 + i;
    }
  }
}

__global__ __launch_bounds__(WARPS * 32) void fp4_gemv_dec3(
    const __nv_bfloat16* __restrict__ x, const unsigned char* __restrict__ w,
    const unsigned char* __restrict__ s, const __nv_bfloat16* __restrict__ lut,
    const int* __restrict__ te, const int* __restrict__ tr0,
    __nv_bfloat16* __restrict__ out, int N, int K) {
  const int e = te[blockIdx.x];
  if (e < 0) return;

  __shared__ float fpb[32 * 17];
  const int tid = threadIdx.x;
  for (int i = tid; i < 32 * 17; i += WARPS * 32) {
    const int j = i % 17;
    fpb[i] = (j < 16) ? __bfloat162float(lut[127 * 16 + j]) : 0.f;
  }
  __syncthreads();

  const int lane = tid & 31;
  const int warp = tid >> 5;
  const float* __restrict__ fp = fpb + lane * 17;

  const int row = tr0[blockIdx.x];
  const int n0 = (blockIdx.y * WARPS + warp) * COLS;
  const size_t wrow = (size_t)K >> 1;     // bytes per weight row
  const size_t srow = (size_t)K >> 5;     // scale bytes per row
  const int nvec4 = K >> 5;               // uint4 chunks per row (32 fp4 each)

  const unsigned char* __restrict__ wE = w + (size_t)e * (size_t)N * wrow;
  const unsigned char* __restrict__ sE = s + (size_t)e * (size_t)N * srow;
  const __nv_bfloat16* __restrict__ xrow = x + (size_t)row * (size_t)K;

  int nc[COLS];
#pragma unroll
  for (int c = 0; c < COLS; ++c) {
    const int n = n0 + c;
    nc[c] = (n < N) ? n : (N - 1);        // clamp: keeps loads branch-free
  }

  float acc[COLS];
#pragma unroll
  for (int c = 0; c < COLS; ++c) acc[c] = 0.f;

  for (int q = lane; q < nvec4; q += 32) {
    // one activation chunk (32 bf16 = 64B) shared by all COLS columns
    uint4 xv[4];
    const uint4* xp = reinterpret_cast<const uint4*>(xrow + ((size_t)q << 5));
#pragma unroll
    for (int j = 0; j < 4; ++j) xv[j] = xp[j];

    // COLS independent weight loads in flight
    uint4 wv[COLS];
    float sc[COLS];
#pragma unroll
    for (int c = 0; c < COLS; ++c) {
      wv[c] = *reinterpret_cast<const uint4*>(wE + (size_t)nc[c] * wrow + ((size_t)q << 4));
      sc[c] = ldexpf(1.f, (int)sE[(size_t)nc[c] * srow + (size_t)q] - 127);
    }

#pragma unroll
    for (int c = 0; c < COLS; ++c) {
      const unsigned int* pw = reinterpret_cast<const unsigned int*>(&wv[c]);
      float part = 0.f;
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const unsigned int word = pw[j];
        const __nv_bfloat162* xb = reinterpret_cast<const __nv_bfloat162*>(&xv[j]);
#pragma unroll
        for (int b = 0; b < 4; ++b) {
          const unsigned int byte = (word >> (b * 8)) & 0xFFu;
          const float2 f2 = __bfloat1622float2(xb[b]);
          part = fmaf(f2.x, fp[byte & 15], part);
          part = fmaf(f2.y, fp[byte >> 4], part);
        }
      }
      acc[c] = fmaf(part, sc[c], acc[c]);
    }
  }

#pragma unroll
  for (int c = 0; c < COLS; ++c) {
#pragma unroll
    for (int off = 16; off; off >>= 1)
      acc[c] += __shfl_down_sync(0xffffffffu, acc[c], off);
  }

  if (lane == 0) {
#pragma unroll
    for (int c = 0; c < COLS; ++c) {
      const int n = n0 + c;
      if (n < N) out[(size_t)row * (size_t)N + n] = __float2bfloat16_rn(acc[c]);
    }
  }
}

void fp4_grouped_gemm_device(torch::Tensor x, torch::Tensor w, torch::Tensor s,
                             torch::Tensor lut, torch::Tensor counts,
                             torch::Tensor offsets, torch::Tensor tile_e,
                             torch::Tensor tile_r0, torch::Tensor tile_rn,
                             torch::Tensor out) {
  at::cuda::CUDAGuard guard(x.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  const int64_t N = w.size(1), K = x.size(1), E = w.size(0);
  const int maxT = (int)tile_e.numel();
  TORCH_CHECK(E <= SCANT, "expert count exceeds scan block");
  TORCH_CHECK((K & 31) == 0, "K must be a multiple of 32");

  fp4_dec3_build_tiles<<<1, SCANT, 0, stream>>>(
      counts.data_ptr<int64_t>(), offsets.data_ptr<int64_t>(),
      tile_e.data_ptr<int>(), tile_r0.data_ptr<int>(), (int)E, maxT);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  dim3 grid(maxT, (int)((N + NPB - 1) / NPB));
  fp4_gemv_dec3<<<grid, WARPS * 32, 0, stream>>>(
      (const __nv_bfloat16*)x.data_ptr(),
      reinterpret_cast<const unsigned char*>(w.data_ptr()),
      reinterpret_cast<const unsigned char*>(s.data_ptr()),
      (const __nv_bfloat16*)lut.data_ptr(),
      tile_e.data_ptr<int>(), tile_r0.data_ptr<int>(),
      (__nv_bfloat16*)out.data_ptr(), (int)N, (int)K);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fp4_grouped_gemm_device", &fp4_grouped_gemm_device, "FP4 decode GEMV v3 (SM80)");
}
