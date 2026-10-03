// Fused FP4 grouped GEMM for the TP8 MoE workspace (V4 kernel shape, V4.1 banks).
// B operand is consumed straight from the packed FP4 bank: no BF16 materialisation,
// so the per layer 3.4 GB expand-and-reload round trip disappears.
// Dequantisation uses the same [256][16] LUT the unpack kernel uses, so the
// numerics are bit identical to the unpacked path by construction.
// Tiles are built on device from the routing counts: fixed grid, capture safe.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <mma.h>

#define TILE_THREADS 512

__global__ void fp4_build_tiles_kernel(const int64_t* __restrict__ counts,
                                       const int64_t* __restrict__ offsets,
                                       int* __restrict__ te, int* __restrict__ tr0,
                                       int* __restrict__ trn, int E, int maxT) {
  __shared__ int sh[TILE_THREADS];
  const int t = threadIdx.x;
  for (int i = t; i < maxT; i += TILE_THREADS) te[i] = -1;
  const int n = (t < E) ? (int)((counts[t] + 63) / 64) : 0;
  sh[t] = n;
  __syncthreads();
  for (int off = 1; off < TILE_THREADS; off <<= 1) {
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
      tr0[idx] = r0 + i * 64;
      trn[idx] = min(64, c - i * 64);
    }
  }
}

__global__ void fp4_grouped_gemm_kernel(
    const __nv_bfloat16* __restrict__ x,
    const uint8_t* __restrict__ w,
    const uint8_t* __restrict__ s,
    const __nv_bfloat16* __restrict__ lut,
    const int* __restrict__ tile_e,
    const int* __restrict__ tile_r0,
    const int* __restrict__ tile_rn,
    __nv_bfloat16* __restrict__ out,
    int N, int K) {
  const int t = blockIdx.x;
  const int e = tile_e[t];
  if (e < 0) return;                      // padded tile: fixed grid, empty work
  const int r0 = tile_r0[t], rn = tile_rn[t];
  const int col0 = blockIdx.y * 64;
  const int tid = threadIdx.x, warp = tid >> 5;

  __shared__ __nv_bfloat16 Bs[64 * 64];   // [ncol=64][k=64]: 16B vector stores
  __shared__ __nv_bfloat16 As[64 * 64];   // [m=64][k=64]
  __shared__ __nv_bfloat16 Ls[256 * 16];  // scale byte x nibble -> BF16
  for (int i = tid; i < 256 * 16; i += 256) Ls[i] = lut[i];

  const size_t wrow = (size_t)K >> 1;     // bytes per weight row
  const size_t srow = (size_t)K >> 5;     // scale entries per row
  const uint8_t* wE = w + (size_t)e * N * wrow;
  const uint8_t* sE = s + (size_t)e * N * srow;

  using namespace nvcuda::wmma;
  fragment<accumulator, 16, 16, 16, float> c_frag[2];
  fill_fragment(c_frag[0], 0.f);
  fill_fragment(c_frag[1], 0.f);
  const int mb = warp & 3;
  const int nb0 = (warp >> 2) * 2;

  const int brow = tid >> 3;              // 0..31 -> two passes for 64 rows
  const int kseg = (tid & 7) * 8;         // 8 elements per thread
  // Rows beyond this tile's occupancy stay zero for the whole K sweep: routed
  // tiles are near empty (768 rows over 384 experts), so refilling them every step is
  // the dominant waste.
  for (int i = tid; i < 64 * 64; i += 256) As[i] = __float2bfloat16(0.f);
  __syncthreads();

  for (int k0 = 0; k0 < K; k0 += 64) {
    const int kk = k0 + kseg;
    const bool live = kk < K;             // K need only be a multiple of 8
    #pragma unroll
    for (int p = 0; p < 2; ++p) {
      const int nc = brow + p * 32;
      __nv_bfloat16 v[8];
      if (!live) {
        #pragma unroll
        for (int j = 0; j < 8; ++j) v[j] = __float2bfloat16(0.f);
      } else {
        const uint32_t four =
            *reinterpret_cast<const uint32_t*>(wE + (size_t)(col0 + nc) * wrow + (kk >> 1));
        const uint32_t sb = sE[(size_t)(col0 + nc) * srow + (kk >> 5)];
        const __nv_bfloat16* row = Ls + sb * 16;
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
          const uint32_t byte = (four >> (8 * j)) & 0xFF;
          v[2 * j] = row[byte & 15];
          v[2 * j + 1] = row[byte >> 4];
        }
      }
      *reinterpret_cast<uint4*>(&Bs[nc * 64 + kseg]) = *reinterpret_cast<const uint4*>(v);
    }
    for (int i = tid; i < rn * 64; i += 256) {
      const int m = i >> 6, k = i & 63;
      As[i] = (k0 + k < K) ? x[(size_t)(r0 + m) * K + k0 + k]
                           : __float2bfloat16(0.f);
    }
    __syncthreads();
    if (mb * 16 < rn) {                   // warps owning only padding rows idle
      fragment<matrix_a, 16, 16, 16, __nv_bfloat16, row_major> a_frag;
      fragment<matrix_b, 16, 16, 16, __nv_bfloat16, col_major> b_frag;
      #pragma unroll
      for (int kk2 = 0; kk2 < 4; ++kk2) {
        load_matrix_sync(a_frag, As + mb * 16 * 64 + kk2 * 16, 64);
        #pragma unroll
        for (int j = 0; j < 2; ++j) {
          load_matrix_sync(b_frag, Bs + (nb0 + j) * 16 * 64 + kk2 * 16, 64);
          mma_sync(c_frag[j], a_frag, b_frag, c_frag[j]);
        }
      }
    }
    __syncthreads();
  }
  __shared__ float Cs[64 * 64];
  #pragma unroll
  for (int j = 0; j < 2; ++j)
    store_matrix_sync(Cs + mb * 16 * 64 + (nb0 + j) * 16, c_frag[j], 64,
                      nvcuda::wmma::mem_row_major);
  __syncthreads();
  for (int i = tid; i < 64 * 64; i += 256) {
    const int m = i >> 6, c = i & 63;
    if (m < rn) out[(size_t)(r0 + m) * N + col0 + c] = __float2bfloat16(Cs[i]);
  }
}

// x [rows, K] BF16 gathered in routed order; w [E, N, K/2]; s [E, N, K/32];
// lut [256, 16] BF16; counts/offsets [E] int64 on device; tile_* [maxT] int32
// scratch; out [rows, N] BF16. Shapes are static: the grid never depends on data.
void fp4_grouped_gemm_device(torch::Tensor x, torch::Tensor w, torch::Tensor s,
                             torch::Tensor lut, torch::Tensor counts,
                             torch::Tensor offsets, torch::Tensor tile_e,
                             torch::Tensor tile_r0, torch::Tensor tile_rn,
                             torch::Tensor out) {
  at::cuda::CUDAGuard guard(x.device());
  const int64_t K = x.size(1), E = w.size(0), N = w.numel() / (E * (K / 2));
  TORCH_CHECK(x.dtype() == torch::kBFloat16 && out.dtype() == torch::kBFloat16,
              "fp4 grouped gemm: BF16 activations");
  TORCH_CHECK(lut.dtype() == torch::kBFloat16 && lut.numel() == 256 * 16, "fp4 lut");
  TORCH_CHECK(N % 64 == 0 && K % 8 == 0 && K % 32 == 0, "fp4 grouped gemm: N%64, K%32");
  TORCH_CHECK(out.size(0) == x.size(0) && out.size(1) == N, "fp4 grouped gemm: out shape");
  TORCH_CHECK(counts.numel() >= E && offsets.numel() >= E, "fp4 grouped gemm: routing");
  TORCH_CHECK(counts.dtype() == torch::kInt64 && offsets.dtype() == torch::kInt64,
              "fp4 grouped gemm: int64 routing");
  TORCH_CHECK(tile_e.dtype() == torch::kInt32 && tile_e.numel() == tile_r0.numel() &&
                  tile_e.numel() == tile_rn.numel(), "fp4 grouped gemm: tiles");
  TORCH_CHECK(E <= TILE_THREADS, "fp4 grouped gemm: expert count above scan width");
  const int maxT = (int)tile_e.numel();
  auto stream = at::cuda::getCurrentCUDAStream();
  fp4_build_tiles_kernel<<<1, TILE_THREADS, 0, stream>>>(
      counts.data_ptr<int64_t>(), offsets.data_ptr<int64_t>(), tile_e.data_ptr<int>(),
      tile_r0.data_ptr<int>(), tile_rn.data_ptr<int>(), (int)E, maxT);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  dim3 grid((unsigned)maxT, (unsigned)(N / 64));
  fp4_grouped_gemm_kernel<<<grid, 256, 0, stream>>>(
      (const __nv_bfloat16*)x.data_ptr(),
      reinterpret_cast<const uint8_t*>(w.data_ptr()),
      reinterpret_cast<const uint8_t*>(s.data_ptr()),
      (const __nv_bfloat16*)lut.data_ptr(),
      tile_e.data_ptr<int>(), tile_r0.data_ptr<int>(), tile_rn.data_ptr<int>(),
      (__nv_bfloat16*)out.data_ptr(), (int)N, (int)K);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fp4_grouped_gemm_device", &fp4_grouped_gemm_device,
        "FP4 grouped GEMM consuming packed banks directly (SM80)");
}
