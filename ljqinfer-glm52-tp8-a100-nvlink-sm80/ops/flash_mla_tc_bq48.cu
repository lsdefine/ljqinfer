// TC multi-Q flash-MLA: BQ=48 rows (nq<=6 query tokens x nh=8 heads) share one K scan.
// ABI-compatible with flash_mla_sm80_splitk_k0_mq_kernel (same params, same ws layout,
// combine kernel reused). B (=nq) is runtime, rows masked. Split-K over grid.x = n_split.
// ws slot per (q,h): n_split*(2+L) floats [m, l, o0..o511].
#include <cuda_fp16.h>
#include <mma.h>
#include <cfloat>

namespace ljq_tc48 {
using namespace nvcuda;

constexpr int TC_L = 512;
constexpr int TC_R = 64;
constexpr int TC_D = TC_L + TC_R;      // 576
constexpr int TC_SD = TC_D + 8;        // padded row stride (584) for smem
constexpr int TC_BQ = 48;              // 6q * 8h max rows
constexpr int TC_BK = 32;
constexpr int TC_WARPS = 8;
constexpr int TC_THREADS = TC_WARPS * 32;
constexpr int TC_MT = 3;               // 48/16 M tiles
constexpr int TC_NTW = TC_BK / 16;     // 2 N tiles -> 6 S-tiles, warps 0..5 active
constexpr int TC_SP = TC_BK + 8;       // P tile row stride
constexpr float TC_SCALE = 0.0625f;

__device__ __forceinline__ uint32_t tc_pk(half a, half b) {
  half2 h2 = __halves2half2(a, b);
  return *reinterpret_cast<uint32_t*>(&h2);
}

__device__ __forceinline__ void tc_mma16816(float* d, const uint32_t* a, const uint32_t* b) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}


__device__ __forceinline__ void tc_ldsm_x4(uint32_t* r, const half* p) {
  const uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(p));
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
      : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}
__device__ __forceinline__ void tc_ldsm_x2t(uint32_t* r, const half* p) {
  const uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(p));
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
      : "=r"(r[0]), "=r"(r[1]) : "r"(a));
}

__device__ __forceinline__ const half* tc_paged_row(
    const half* cache, const int64_t* pt, int page_size, int k) {
  const int64_t page = pt[k / page_size];
  return cache + (page * page_size + (k % page_size)) * TC_D;
}

// grid: (n_split). One CTA handles all nq*nh rows for its K segment.
extern "C" __global__ __launch_bounds__(TC_THREADS)
void flash_mla_sm80_tc_bq48_splitk_kernel(
    const half* __restrict__ ql, const half* __restrict__ qr,
    const half* __restrict__ cache, const int64_t* __restrict__ page_table,
    int page_size, float* __restrict__ ws,
    int nq, int nk, int nh, int n_split, int stride_qh,
    const int* __restrict__ k0_ptr, int pt_stride) {
  const int sid = blockIdx.x;
  const int b = blockIdx.y;
  ql += (size_t)b * nq * nh * TC_L;
  qr += (size_t)b * nq * nh * TC_R;
  page_table += (size_t)b * pt_stride;
  ws += (size_t)b * nq * nh * (size_t)stride_qh;
  const int tid = threadIdx.x, w = tid >> 5, lane = tid & 31;
  const int nrows = nq * nh;               // <= 48
  const int q_start = k0_ptr ? k0_ptr[b] : 0;
  const int kend_max = min(nk, q_start + nq);   // last row sees q_start+nq
  // Split over the true prefix (device-side kend), not pool capacity: under
  // CUDA graph nk is total pool capacity and would leave most splits empty.
  const int chunk = (kend_max + n_split - 1) / n_split;
  const int k0s = sid * chunk;
  const int k1 = min(kend_max, min(nk, k0s + chunk));

  extern __shared__ half smem[];
  half* sQ = smem;                                   // [BQ][SD]
  half* sK = sQ + TC_BQ * TC_SD;                     // [2][BK][SD]
  half* sP = sK + 2 * TC_BK * TC_SD;                 // [BQ][SP] half P
  float* sS = reinterpret_cast<float*>(sP + TC_BQ * TC_SP);  // [BQ][BK] scores
  float* mrow = sS + TC_BQ * TC_BK;
  float* lrow = mrow + TC_BQ;
  float* arow = lrow + TC_BQ;

  // ---- load Q (rows: r = tq*nh+h; ql 512 + qr 64), zero-pad inactive rows ----
  for (int r = w; r < TC_BQ; r += TC_WARPS) {
    half* dst = sQ + r * TC_SD;
    if (r < nrows) {
      const half* srcL = ql + (size_t)r * TC_L;
      const half* srcR = qr + (size_t)r * TC_R;
      for (int c = lane * 8; c < TC_L; c += 32 * 8)
        *reinterpret_cast<int4*>(dst + c) = *reinterpret_cast<const int4*>(srcL + c);
      if (lane < 8)
        *reinterpret_cast<int4*>(dst + TC_L + lane * 8) =
            *reinterpret_cast<const int4*>(srcR + lane * 8);
    } else {
      for (int c = lane * 8; c < TC_D; c += 32 * 8)
        *reinterpret_cast<int4*>(dst + c) = make_int4(0, 0, 0, 0);
    }
  }
  for (int r = tid; r < TC_BQ; r += TC_THREADS) {
    mrow[r] = -FLT_MAX; lrow[r] = 0.f; arow[r] = 0.f;
  }

  // register accumulator for O = P*V: per warp 3 M-tiles x (512/8warps... )
  // O columns: 512 split across 8 warps -> 64 cols/warp -> NT=8 (8 cols per mma n=8)
  constexpr int NT = 8;
  const int c0 = w * 64;                  // this warp's O column base
  const int g = lane >> 2;                // 0..7
  const int tg = lane & 3;                // 0..3
  float acc[TC_MT][NT][4];
#pragma unroll
  for (int m = 0; m < TC_MT; ++m)
#pragma unroll
    for (int j = 0; j < NT; ++j)
      acc[m][j][0] = acc[m][j][1] = acc[m][j][2] = acc[m][j][3] = 0.f;

  const bool empty = (k0s >= k1);
  __syncthreads();

  // ---- K tile loader: 32 rows x 576 halves via cp.async; invalid rows zeroed ----
  auto load_tile = [&](int kb, int buf) {
    half* dst = sK + buf * TC_BK * TC_SD;
    // 72 16B-chunks per row, 32 rows = 2304 chunks over 256 threads
    for (int idx = tid; idx < TC_BK * 72; idx += TC_THREADS) {
      const int row = idx / 72, cc = (idx % 72) * 8;
      half* d = dst + row * TC_SD + cc;
      const int k = kb + row;
      if (k < k1) {
        const half* s = tc_paged_row(cache, page_table, page_size, k) + cc;
        const uint32_t sm = static_cast<uint32_t>(__cvta_generic_to_shared(d));
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(sm), "l"(s));
      } else {
        *reinterpret_cast<int4*>(d) = make_int4(0, 0, 0, 0);
      }
    }
    asm volatile("cp.async.commit_group;\n" ::);
  };

  const int mt = w / TC_NTW, nt = w % TC_NTW;
  const bool sact = (w < TC_MT * TC_NTW);   // warps 0..5 compute S tiles

  if (!empty) {
    load_tile(k0s, 0);
    for (int kb = k0s; kb < k1; kb += TC_BK) {
      const int buf = ((kb - k0s) / TC_BK) & 1;
      const half* ks = sK + buf * TC_BK * TC_SD;
      if (kb + TC_BK < k1) {
        load_tile(kb + TC_BK, buf ^ 1);   // prefetch next before waiting on current
        asm volatile("cp.async.wait_group 1;\n" ::);
      } else {
        asm volatile("cp.async.wait_group 0;\n" ::);
      }
      __syncthreads();

      // ---- S = Q * K^T : 3x2 tiles of 16x16, warps 0..5 ----
      wmma::fragment<wmma::accumulator, 16, 16, 16, float> c;
      wmma::fill_fragment(c, 0.f);
      if (sact) {
        for (int kk = 0; kk < TC_D; kk += 16) {
          wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a;
          wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b;
          wmma::load_matrix_sync(a, sQ + mt * 16 * TC_SD + kk, TC_SD);
          wmma::load_matrix_sync(b, ks + nt * 16 * TC_SD + kk, TC_SD);
          wmma::mma_sync(c, a, b, c);
        }
        wmma::store_matrix_sync(sS + mt * 16 * TC_BK + nt * 16, c, TC_BK, wmma::mem_row_major);
      }
      __syncthreads();

      // ---- online softmax: 8 threads/row, 2 passes (rows 0..31, 32..47) ----
      {
        const int rb = tid >> 3, sub = tid & 7;   // 32 row-groups, 8 threads each
        float vreg[2][4], mnw[2], mod[2];
#pragma unroll
        for (int p = 0; p < 2; ++p) {
          const int r = p * 32 + rb;
          const bool ok = (r < TC_BQ) && (r < nrows);
          const int rr = (r < TC_BQ) ? r : 0;
          const int tq_r = rr / nh;
          const int kend_r = min(k1, q_start + tq_r + 1);
          float mloc = -FLT_MAX;
#pragma unroll
          for (int t = 0; t < 4; ++t) {
            const int j = sub + t * 8;
            const float v = (ok && (kb + j) < kend_r)
                ? sS[rr * TC_BK + j] * TC_SCALE : -FLT_MAX;
            vreg[p][t] = v;
            mloc = fmaxf(mloc, v);
          }
#pragma unroll
          for (int off = 1; off < 8; off <<= 1)
            mloc = fmaxf(mloc, __shfl_xor_sync(0xffffffffu, mloc, off));
          const float mold = ok ? mrow[rr] : -FLT_MAX;
          mod[p] = mold;
          mnw[p] = fmaxf(mold, mloc);
        }
        __syncthreads();   // sS consumed
#pragma unroll
        for (int p = 0; p < 2; ++p) {
          const int r = p * 32 + rb;
          const bool okr = (r < TC_BQ);
          const bool ok = okr && (r < nrows);
          const int rr = okr ? r : 0;
          const float mnew = mnw[p], mold = mod[p];
          float lloc = 0.f;
#pragma unroll
          for (int t = 0; t < 4; ++t) {
            const float v = vreg[p][t];
            const float pv = (v == -FLT_MAX) ? 0.f : __expf(v - mnew);
            if (okr) sP[rr * TC_SP + sub + t * 8] = __float2half(pv);
            lloc += pv;
          }
#pragma unroll
          for (int off = 1; off < 8; off <<= 1)
            lloc += __shfl_xor_sync(0xffffffffu, lloc, off);
          const float alpha = (mold == -FLT_MAX) ? 0.f : __expf(mold - mnew);
          if (okr && sub == 0) {
            mrow[rr] = ok ? mnew : -FLT_MAX;
            lrow[rr] = ok ? (lrow[rr] * alpha + lloc) : 0.f;
            arow[rr] = ok ? alpha : 0.f;
          }
        }
      }
      __syncthreads();

      // ---- rescale acc, then O += P*V (V = first 512 cols of K tile) ----
      {
#pragma unroll
        for (int m = 0; m < TC_MT; ++m) {
          const float a0 = arow[m * 16 + g], a1 = arow[m * 16 + g + 8];
#pragma unroll
          for (int j = 0; j < NT; ++j) {
            acc[m][j][0] *= a0; acc[m][j][1] *= a0;
            acc[m][j][2] *= a1; acc[m][j][3] *= a1;
          }
        }
#pragma unroll
        for (int kk = 0; kk < TC_BK; kk += 16) {
          uint32_t af[TC_MT][4];
#pragma unroll
          for (int m = 0; m < TC_MT; ++m)
            tc_ldsm_x4(af[m], sP + (m * 16 + (lane & 15)) * TC_SP + kk + (lane >> 4) * 8);
#pragma unroll
          for (int j = 0; j < NT; ++j) {
            uint32_t bf[2];
            tc_ldsm_x2t(bf, ks + (kk + (lane & 7) + ((lane & 8) ? 8 : 0)) * TC_SD + c0 + j * 8);
#pragma unroll
            for (int m = 0; m < TC_MT; ++m) tc_mma16816(acc[m][j], af[m], bf);
          }
        }
      }
      __syncthreads();
    }
  }

  // ---- epilogue: write ws slots [m, l, o0..o511] per active row ----
#pragma unroll
  for (int m = 0; m < TC_MT; ++m) {
    const int r0 = m * 16 + g, r1 = m * 16 + g + 8;
#pragma unroll
    for (int j = 0; j < NT; ++j) {
      const int col = c0 + j * 8 + 2 * tg;
      if (r0 < nrows) {
        float* o0 = ws + (size_t)r0 * stride_qh + (size_t)sid * (2 + TC_L) + 2 + col;
        o0[0] = acc[m][j][0]; o0[1] = acc[m][j][1];
      }
      if (r1 < nrows) {
        float* o1 = ws + (size_t)r1 * stride_qh + (size_t)sid * (2 + TC_L) + 2 + col;
        o1[0] = acc[m][j][2]; o1[1] = acc[m][j][3];
      }
    }
  }
  for (int r = tid; r < nrows; r += TC_THREADS) {
    float* slot = ws + (size_t)r * stride_qh + (size_t)sid * (2 + TC_L);
    slot[0] = mrow[r]; slot[1] = lrow[r];
  }
}

constexpr int TC_SMEM_BYTES =
    (TC_BQ * TC_SD + 2 * TC_BK * TC_SD + TC_BQ * TC_SP) * (int)sizeof(half) +
    (TC_BQ * TC_BK + 3 * TC_BQ) * (int)sizeof(float);


// ---- v2: column-parallel combine. grid(nh, nq, CS); block 64 threads owns 64 cols ----
extern "C" __global__ void tc48_combine_v2_kernel(
    const float* __restrict__ ws, half* __restrict__ out,
    int nq, int nh, int n_split, int stride_qh) {
  const int h = blockIdx.x, q = blockIdx.y, cs = blockIdx.z;
  const int r = q * nh + h;
  const float* base = ws + (size_t)r * stride_qh;
  float M = -FLT_MAX;
  for (int s = 0; s < n_split; ++s) M = fmaxf(M, base[(size_t)s * (2 + TC_L)]);
  const int col = cs * 64 + threadIdx.x;
  float num = 0.f, den = 0.f;
  if (M != -FLT_MAX) {
    for (int s = 0; s < n_split; ++s) {
      const float* slot = base + (size_t)s * (2 + TC_L);
      const float w = __expf(slot[0] - M);
      den += w * slot[1];
      num += w * slot[2 + col];
    }
  }
  const float inv = (den > 0.f) ? 1.f / den : 0.f;
  out[((size_t)q * nh + h) * TC_L + col] = __float2half(num * inv);
}

}  // namespace ljq_tc48
