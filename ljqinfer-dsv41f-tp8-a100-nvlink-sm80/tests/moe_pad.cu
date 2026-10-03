// V4.1 MoE decode: direct port of V4's fused moe_rank_decode_fp4 (gu + down).
//
// Ported verbatim from ljqinfer_dsv4f_tp8/ops/dsv4_wgemm.cu.  The MoE decode
// math is identical between V4 and V4.1 (same e2m1 + e8m0-per-32 banks, same
// topk=6, same swiglu_limit=10.0); the ONLY structural difference is that V4.1
// stores gate and up in one merged w13 bank of 2*Nff rows per expert, so w1 is
// row n and w3 is row Nff + n of that bank.  Nothing else is changed.
//
// Why this replaces the hand-written GEMV attempts: the decode fp4 path is
// ALU-pipe bound, not bandwidth bound (V4 ncu: 83% ALU at T=32).  A shared-LUT
// lookup per element loses regardless of load width -- v2 (uint4 loads) and v3
// (4 independent accumulators per lane) both measured >= baseline.  PRMT bit
// assembly (13 ALU ops per 8 elements) plus __hfma2 bf16x2 accumulation is the
// actual win, together with fusing gate/up/SiLU so the elementwise glue kernels
// disappear from the graph.
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <stdint.h>

__device__ __forceinline__ float e8m0_scale(uint8_t v) {
  // E8M0 is a bare fp32 exponent field: 2^(v-127) == bits (v << 23).
  return v == 255 ? __int_as_float(0x7fffffff) : __int_as_float((uint32_t)v << 23);
}

// Vectorised E2M1 -> bf16x8.  The 8 e2m1 magnitudes are exactly representable
// in bf16, so PRMT table-looks-up the low/high bytes of the bf16 pattern
// (index = 3-bit magnitude) and the sign nibble is OR-ed back in afterwards.
// Bit-identical to a per-element e2m1_to_float.
__device__ __forceinline__ void fp4x8_to_bf16x8(uint32_t p, uint32_t o[4]) {
  const uint32_t LO_A = 0xC0800000u, LO_B = 0xC0804000u;
  const uint32_t HI_A = 0x3F3F3F00u, HI_B = 0x40404040u;
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

// gate/up GEMV + fused SiLU, one warp per output column, 32 lanes split K.
// V4.1 change: w1 = w13 row (e, n), w3 = w13 row (e, Nff + n).
template<int NVEC_C>
__global__ void moe_decode_gu_fp4_kernel(
    const __nv_bfloat16* __restrict__ x,
    const int64_t* __restrict__ ids,
    const uint8_t* __restrict__ w13, const uint8_t* __restrict__ s13,
    __nv_bfloat16* __restrict__ mid,
    int N, int K, int packed_stride, int topk) {
  const int slot = blockIdx.y;
  const int t = slot / topk;
  const int64_t e = ids[slot];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int n = blockIdx.x * (blockDim.x >> 5) + warp;
  if (n >= N) return;
  const __nv_bfloat16* __restrict__ xrow = x + (size_t)t * K;
  // Merged bank: 2*N rows per expert, gate first then up.
  const size_t row1 = (size_t)e * (2 * N) + n;
  const size_t row3 = row1 + N;
  const uint4* w1v = (const uint4*)(w13 + row1 * packed_stride);
  const uint4* w3v = (const uint4*)(w13 + row3 * packed_stride);
  const uint8_t* s1r = s13 + row1 * (K >> 5);
  const uint8_t* s3r = s13 + row3 * (K >> 5);
  float ag = 0.f, au = 0.f;
  const int nvec = K >> 5;  // uint4 == 32 nibbles == one e8m0 scale block
  const int nvec_u = NVEC_C ? NVEC_C : nvec;
#pragma unroll
  for (int i = lane; i < nvec_u; i += 32) {
    const uint4 q1 = w1v[i], q3 = w3v[i];
    const float f1 = e8m0_scale(s1r[i]);
    const float f3 = e8m0_scale(s3r[i]);
    const uint32_t v1[4] = {q1.x, q1.y, q1.z, q1.w};
    const uint32_t v3[4] = {q3.x, q3.y, q3.z, q3.w};
    const int kb = i << 5;
    // Two independent 8-FMA chains merged into the FP32 outer accumulator:
    // the validated GU precision/speed tradeoff from V4.
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

// down GEMV + routing-weight reduction.  Verbatim V4: 8 lanes cover one row via
// uint4 loads and a warp processes kRows=4 rows concurrently, so 4 rows x 6
// experts = 24 independent 128 B transactions are in flight per warp.  The
// 32-lane shuffle tree is replayed over virtual lanes v = 4*L + j so the
// reduction order (and therefore the bits) matches the scalar version.
template <int TOPK>
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
  constexpr int kRows = 4;
  const int rsel = lane >> 3;   // row within the warp's 4-row group
  const int L = lane & 7;       // lane within the 8-lane row group
  const int n_base = (blockIdx.x * (blockDim.x >> 5) + warp) * kRows;
  extern __shared__ __nv_bfloat16 ms[];
  // Pad each 32-value block to 40 bf16 (80 B).  The 8 distinct shm addresses a
  // warp touches then map onto all 32 banks (20*L % 32) instead of only two
  // groups (16*L % 32 -> 4-way conflict), and 80 B stays 16 B aligned so the
  // four ms2[jj] reads still fuse into one LDS.128.  Layout only: the order of
  // every float op is untouched, so the result is bit-identical.
  const int MSB = 40;
  const int MSK = (M >> 5) * MSB;
  for (int i = threadIdx.x; i < TOPK * M; i += blockDim.x) {
    const int kt = i / M, r = i - kt * M;
    ms[kt * MSK + (r >> 5) * MSB + (r & 31)] = mid[(size_t)t * TOPK * M + i];
  }
  __syncthreads();
  if (n_base >= Nout) return;
  const int n = n_base + rsel;
  const bool nvalid = n < Nout;
  const int nvec4 = M >> 5;
  float acc[TOPK][4];
#pragma unroll
  for (int ktop = 0; ktop < TOPK; ++ktop) {
    const size_t row = (size_t)ids[t * TOPK + ktop] * Nout + (nvalid ? n : (Nout - 1));
    const uint4* wv4 = (const uint4*)(w2 + row * packed_stride);
    const uint8_t* sr = s2 + row * (M >> 5);
    float a[4] = {0.f, 0.f, 0.f, 0.f};
    for (int q = L; q < nvec4; q += 8) {
      const uint4 p4 = wv4[q];
      // word index i = 4q + j -> scale byte index i>>2 == q for all j
      const float sf = e8m0_scale(sr[q]);
      const uint32_t pw[4] = {p4.x, p4.y, p4.z, p4.w};
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const int i = (q << 2) + j;
        const int mb = i << 3;
        uint32_t ow[4];
        fp4x8_to_bf16x8(pw[j], ow);
        __nv_bfloat162 hb = __float2bfloat162_rn(0.f);
        const __nv_bfloat162* ms2 =
            (const __nv_bfloat162*)(ms + ktop * MSK + q * MSB + (j << 3));
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
    for (int ktop = 0; ktop < TOPK; ++ktop)
#pragma unroll
      for (int j = 0; j < 4; ++j)
        acc[ktop][j] += __shfl_down_sync(0xffffffffu, acc[ktop][j], d);
  if (L == 0 && nvalid) {
    float total = 0.f;
#pragma unroll
    for (int ktop = 0; ktop < TOPK; ++ktop) {
      // d=2: j += j+2 ; d=1: j0 += j1
      const float v0 = acc[ktop][0] + acc[ktop][2];
      const float v1 = acc[ktop][1] + acc[ktop][3];
      const float dn = __bfloat162float(__float2bfloat16_rn(v0 + v1));
      const float rw =
          __bfloat162float(__float2bfloat16_rn(wts[t * TOPK + ktop]));
      total += __bfloat162float(__float2bfloat16_rn(dn * rw));
    }
    y[(size_t)t * Nout + n] = __float2bfloat16_rn(total);
  }
}

// x: (T,K) bf16 ; ids/wts: (T,6) ; w13w: [E,2*Nff,K/2] ; w2w: [E,Nout,Nff/2]
static torch::Tensor moe_rank_decode_fp4(
    torch::Tensor x, torch::Tensor ids, torch::Tensor wts,
    torch::Tensor w13w, torch::Tensor w13s,
    torch::Tensor w2w, torch::Tensor w2s) {
  at::cuda::CUDAGuard guard(x.device());
  TORCH_CHECK(x.is_contiguous() && x.scalar_type() == torch::kBFloat16, "x bf16 contiguous");
  TORCH_CHECK(ids.scalar_type() == torch::kLong && wts.scalar_type() == torch::kFloat,
              "ids int64 / wts float32");
  const int topk = (int)ids.size(1);
  TORCH_CHECK(topk == 6 || topk == 3, "decode MoE kernel is built for topk 3 (draft) or 6 (trunk)");
  TORCH_CHECK(wts.size(1) == topk, "ids/wts topk mismatch");
  // The merged bank ships as [E,2,Nff,K/2] (gate plane first, up plane second),
  // which is byte-identical to [E,2*Nff,K/2]; accept the flattened form too.
  TORCH_CHECK(w13w.dim() == 4 ? w13w.size(1) == 2 : w13w.size(1) % 2 == 0,
              "w13 bank must hold gate then up");
  const int64_t T = x.size(0), K = x.size(1);
  const int64_t Nff = w13w.dim() == 4 ? w13w.size(2) : w13w.size(1) / 2;
  const int64_t Nout = w2w.size(1), nsel = T * topk;
  const int w13_stride = (int)w13w.size(w13w.dim() - 1);
  auto mid = torch::empty({nsel, Nff}, x.options().dtype(torch::kBFloat16));
  auto y = torch::empty({T, Nout}, x.options().dtype(torch::kBFloat16));
  auto stream = at::cuda::getCurrentCUDAStream();
  constexpr int kWarps = 8;
#define V41_LAUNCH_GU(NV)                                                      \
  moe_decode_gu_fp4_kernel<NV><<<dim3((unsigned)((Nff + kWarps - 1) / kWarps), \
                                     (unsigned)nsel),                          \
                                 kWarps * 32, 0, stream>>>(                    \
      (const __nv_bfloat16*)x.data_ptr(), ids.contiguous().data_ptr<int64_t>(), \
      (const uint8_t*)w13w.data_ptr(), (const uint8_t*)w13s.data_ptr(),         \
      (__nv_bfloat16*)mid.data_ptr(), (int)Nff, (int)K, w13_stride, topk)
  // Every production shape has K == 5120; the literal lets ptxas unroll the
  // five k-steps and issue their loads up front (+3.5% on the fused decode).
  if (K == 5120) { V41_LAUNCH_GU(160); } else { V41_LAUNCH_GU(0); }
#undef V41_LAUNCH_GU
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  constexpr int kDownRows = 4;  // must match kRows in the down kernel
  const dim3 dgrid((unsigned)((Nout + kWarps * kDownRows - 1) / (kWarps * kDownRows)), (unsigned)T);
  const size_t dshm = (size_t)(topk * ((Nff >> 5) * 40)) * sizeof(__nv_bfloat16);
#define V41_LAUNCH_DOWN(TK)                                                       \
  moe_decode_down_fp4_kernel<TK><<<dgrid, kWarps * 32, dshm, stream>>>(           \
      (const __nv_bfloat16*)mid.data_ptr(), ids.contiguous().data_ptr<int64_t>(), \
      wts.contiguous().data_ptr<float>(), (const uint8_t*)w2w.data_ptr(),         \
      (const uint8_t*)w2s.data_ptr(), (__nv_bfloat16*)y.data_ptr(),               \
      (int)Nout, (int)Nff, (int)w2w.size(2))
  if (topk == 6) { V41_LAUNCH_DOWN(6); } else { V41_LAUNCH_DOWN(3); }
#undef V41_LAUNCH_DOWN
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("moe_rank_decode_fp4", &moe_rank_decode_fp4,
        "V4.1 fused MoE decode (gate/up + SiLU + down + routing reduce), FP4 SM80");
}
