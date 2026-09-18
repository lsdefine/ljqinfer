#include <torch/extension.h>
#include <mutex>
#include <unordered_map>
#include <string>
#include <vector>
#include <limits>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cfloat>
#include <algorithm>

using namespace nvcuda;

namespace {
constexpr int L = 512;
constexpr int R = 64;
constexpr int D = L + R;        // 576
constexpr int BQ = 64;
constexpr int BK = 64;
constexpr int WARPS = 16;          // v19: 2 groups x 8 warps, one head each
constexpr int THREADS = WARPS * 32;
constexpr int SD = D + 8;       // 584, pad to break bank conflicts
constexpr int SPLITV = 3;          // uint4 groups in the first cp.async batch
constexpr int DSPLIT = SPLITV * 8 * 8;  // 320 halves, 16-aligned
constexpr int HPB = 1;          // heads per block, sharing ONE loaded K/V tile
constexpr int SP = BK + 8;      // 72
constexpr int WPG = WARPS / HPB;   // 8 warps per head-group
constexpr int RPP = (THREADS / HPB) / 8;
constexpr int NPASS = (BQ + RPP - 1) / RPP;
constexpr int NCOL = L / WPG;   // 64 output cols per warp
constexpr int NT = NCOL / 8;    // 8 n-tiles of 8
constexpr float QSCALE = 0.0625f;
constexpr int MT = BQ / 16;   // m-tiles of 16 rows
constexpr int NSPLIT = 2;   // split-K: KV segments per (q-tile, head-group)

__device__ __forceinline__ const half* kv_row(const half* pool, const long long* table,
                                              int page_size, long long k) {
  const long long lp = k / page_size;
  const int off = (int)(k - lp * page_size);
  return pool + ((table[lp] * (long long)page_size + off) * (long long)D);
}

__device__ __forceinline__ uint32_t pk(half a, half b) {
  __half2 h = __halves2half2(a, b);
  return *reinterpret_cast<uint32_t*>(&h);
}

__device__ __forceinline__ void mma16816(float (&d)[4], const uint32_t (&a)[4],
                                         const uint32_t (&b)[2]) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

__global__ __launch_bounds__(THREADS)
void paged_prefill_mla_kernel(
    const half* __restrict__ ql, const half* __restrict__ qr,
    const half* __restrict__ pool, const long long* __restrict__ table,
    float* __restrict__ po, float* __restrict__ pm, float* __restrict__ pl,
    int Q, int H, int page_size, int q_start) {
  extern __shared__ char smem_raw[];
  half*  qs   = reinterpret_cast<half*>(smem_raw);
  half*  ks   = qs + HPB * BQ * SD;
  float* scb  = reinterpret_cast<float*>(ks + BK * SD);   // HPB x (BQ*BK) floats
  float* mrow = scb + HPB * BQ * BK;                       // ps is overlaid on scb
  float* lrow = mrow + HPB * BQ;
  float* arow = lrow + HPB * BQ;

  const int h0 = blockIdx.y * HPB;      // v17: this block owns heads [h0, h0+HPB)
  const int q0 = blockIdx.x * BQ;
  const int tid = threadIdx.x;
  const int warp = tid >> 5, lane = tid & 31;
  const int hh = warp / WPG;      // v19: which head this warp serves
  const int w  = warp % WPG;      // warp index inside the group
  const int tg_id = tid & (THREADS / HPB - 1);   // thread id inside the head group
  const int g = lane >> 2, tg = lane & 3;
  const int nq = min(BQ, Q - q0);
  float* sc = scb + hh * (BQ * BK);
  half*  ps = reinterpret_cast<half*>(sc);   // overlay: sc read into regs first
  half*  qsh   = qs   + hh * BQ * SD;
  float* mrowh = mrow + hh * BQ;
  float* lrowh = lrow + hh * BQ;
  float* arowh = arow + hh * BQ;
  if (nq <= 0) return;

  float acc[MT][NT][4];
#pragma unroll
  for (int m = 0; m < MT; ++m)
#pragma unroll
    for (int j = 0; j < NT; ++j)
#pragma unroll
      for (int e = 0; e < 4; ++e) acc[m][j][e] = 0.f;

  for (int i = tid; i < HPB * BQ * SD; i += THREADS) qs[i] = __float2half(0.f);
  if (tid < HPB * BQ) { mrow[tid] = -FLT_MAX; lrow[tid] = 0.f; arow[tid] = 1.f; }
  __syncthreads();
  for (int idx = tid; idx < HPB * nq * D; idx += THREADS) {
    const int hh = idx / (nq * D), rest = idx - hh * (nq * D);
    const int r = rest / D, d = rest - r * D;
    const int h = h0 + hh;
    const half* src = (d < L) ? (ql + ((long long)(q0 + r) * H + h) * L + d)
                              : (qr + ((long long)(q0 + r) * H + h) * R + (d - L));
    qs[hh * BQ * SD + r * SD + d] = __float2half(__half2float(*src) * QSCALE);
  }
  __syncthreads();

  const int kend = q_start + q0 + nq;
  const int c0 = w * NCOL;
  const int sp = blockIdx.z;
  // chunk rounded up to a BK multiple so every segment is tile aligned
  const int chunk = ((kend + NSPLIT - 1) / NSPLIT + BK - 1) / BK * BK;
  const int kbeg = sp * chunk;
  const int kstop = min(kend, kbeg + chunk);

  for (int k0 = kbeg; k0 < kstop; k0 += BK) {
    const int nk = min(BK, kstop - k0);
    // Tail block only: zero the rows that have no KV so masked lanes read 0.
    if (nk < BK) {
      for (int i = tid + nk * SD; i < BK * SD; i += THREADS) ks[i] = __float2half(0.f);
    }
    __syncthreads();
    // One page-table lookup per row (not per element), 128-bit vector copy.
    // SD*2 and the pool row stride are both 16B aligned, so uint4 is safe.
    // v5: 8 threads per row x exactly 9 uint4 = 72 = D/8, no remainder pass.
    {
      // v6: cp.async global->shared, 16B per instruction, no register round trip.
      // v9: split the K tile load along d into two cp.async groups so that the
      // second half streams in while the S matmul already works on the first.
      const int sub = tid & 7;
#pragma unroll 1
      for (int r = tid >> 3; r < nk; r += (THREADS >> 3)) {
        const uint4* s4 = reinterpret_cast<const uint4*>(
            kv_row(pool, table, page_size, (long long)k0 + r));
        uint4* d4 = reinterpret_cast<uint4*>(ks + r * SD);
#pragma unroll
        for (int v = 0; v < SPLITV; ++v) {
          const int u = sub + v * 8;
          unsigned sm = static_cast<unsigned>(__cvta_generic_to_shared(d4 + u));
          asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                       :: "r"(sm), "l"(s4 + u));
        }
      }
      asm volatile("cp.async.commit_group;\n" ::);
#pragma unroll 1
      for (int r = tid >> 3; r < nk; r += (THREADS >> 3)) {
        const uint4* s4 = reinterpret_cast<const uint4*>(
            kv_row(pool, table, page_size, (long long)k0 + r));
        uint4* d4 = reinterpret_cast<uint4*>(ks + r * SD);
#pragma unroll
        for (int v = SPLITV; v < 9; ++v) {
          const int u = sub + v * 8;
          unsigned sm = static_cast<unsigned>(__cvta_generic_to_shared(d4 + u));
          asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                       :: "r"(sm), "l"(s4 + u));
        }
      }
      asm volatile("cp.async.commit_group;\n" ::);
      // only wait for the FIRST group; the second keeps flowing in background
      asm volatile("cp.async.wait_group 1;\n" ::);
    }
    __syncthreads();

    // v17: the K/V tile in smem is head-independent -> reuse it for HPB heads.

    const int NTW = BK / 16;
    const int mt = w / NTW, nt = w % NTW;
    const bool sact = (w < MT * NTW);   // S = Q*K^T : MT x (BK/16) tiles of 16x16
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c;
    wmma::fill_fragment(c, 0.f);
    if (sact) {
      for (int kk = 0; kk < DSPLIT; kk += 16) {
        wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b;
        wmma::load_matrix_sync(a, qsh + mt * 16 * SD + kk, SD);
        wmma::load_matrix_sync(b, ks + nt * 16 * SD + kk, SD);
        wmma::mma_sync(c, a, b, c);
      }
    }
    // FIX(v55): uniform across ALL warps -> no divergent barrier
    asm volatile("cp.async.wait_group 0;\n" ::);
    __syncthreads();
    if (sact) {
      for (int kk = DSPLIT; kk < D; kk += 16) {
        wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b;
        wmma::load_matrix_sync(a, qsh + mt * 16 * SD + kk, SD);
        wmma::load_matrix_sync(b, ks + nt * 16 * SD + kk, SD);
        wmma::mma_sync(c, a, b, c);
      }
      wmma::store_matrix_sync(sc + mt * 16 * BK + nt * 16, c, BK, wmma::mem_row_major);
    }
    __syncthreads();

    {   // online softmax, 8 threads/row, NPASS passes (two-phase: read-all then write-all)
      const int rb = tg_id >> 3, sub = tg_id & 7;
      float vreg[NPASS][BK / 8];
      float mnw[NPASS], mod[NPASS];
#pragma unroll
      for (int p = 0; p < NPASS; ++p) {
        const int r = p * RPP + rb;
        const bool ok = (r < BQ);
        const int rr = ok ? r : 0;
        const int qg = q_start + q0 + rr;
        float mloc = -FLT_MAX;
#pragma unroll
        for (int t = 0; t < BK / 8; ++t) {
          // Rotate 8-col groups by row-group within a warp to spread score banks.
          const int j = sub + (((t + (rb & 3)) & (BK / 8 - 1)) * 8);
          const float v = (ok && j < nk && (k0 + j) <= qg) ? sc[rr * BK + j] : -FLT_MAX;
          vreg[p][t] = v;
          mloc = fmaxf(mloc, v);
        }
#pragma unroll
        for (int off = 1; off < 8; off <<= 1)
          mloc = fmaxf(mloc, __shfl_xor_sync(0xffffffff, mloc, off));
        const float mold = ok ? mrowh[rr] : -FLT_MAX;
        mod[p] = mold;
        mnw[p] = fmaxf(mold, mloc);
      }
      __syncthreads();   // sc fully consumed into vreg; ps may now overwrite it
#pragma unroll
      for (int p = 0; p < NPASS; ++p) {
        const int r = p * RPP + rb;
        const bool ok = (r < BQ);
        const int rr = ok ? r : 0;
        const float mnew = mnw[p], mold = mod[p];
        float lloc = 0.f;
#pragma unroll
        for (int t = 0; t < BK / 8; ++t) {
          const int j = sub + (((t + (rb & 3)) & (BK / 8 - 1)) * 8);
          const float v = vreg[p][t];
          const float pv = (v == -FLT_MAX) ? 0.f : __expf(v - mnew);
          if (ok) ps[rr * SP + j] = __float2half(pv);
          lloc += pv;
        }
#pragma unroll
        for (int off = 1; off < 8; off <<= 1) lloc += __shfl_xor_sync(0xffffffff, lloc, off);
        const float alpha = (mold == -FLT_MAX) ? 0.f : __expf(mold - mnew);
        if (ok && sub == 0) { mrowh[rr] = mnew; lrowh[rr] = lrowh[rr] * alpha + lloc; arowh[rr] = alpha; }
      }
    }
    __syncthreads();

    {   // rescale register accumulator, then O += P*V  (all in registers)
#pragma unroll
      for (int m = 0; m < MT; ++m) {
        const float a0 = arowh[m * 16 + g], a1 = arowh[m * 16 + g + 8];
#pragma unroll
        for (int j = 0; j < NT; ++j) {
          acc[m][j][0] *= a0; acc[m][j][1] *= a0;
          acc[m][j][2] *= a1; acc[m][j][3] *= a1;
        }
      }
#pragma unroll
      for (int kk = 0; kk < BK; kk += 16) {
        uint32_t af[MT][4];
#pragma unroll
        for (int m = 0; m < MT; ++m) {
          const half* pr = ps + (m * 16) * SP + kk;
          af[m][0] = pk(pr[g * SP + 2 * tg],       pr[g * SP + 2 * tg + 1]);
          af[m][1] = pk(pr[(g + 8) * SP + 2 * tg], pr[(g + 8) * SP + 2 * tg + 1]);
          af[m][2] = pk(pr[g * SP + 2 * tg + 8],   pr[g * SP + 2 * tg + 9]);
          af[m][3] = pk(pr[(g + 8) * SP + 2 * tg + 8], pr[(g + 8) * SP + 2 * tg + 9]);
        }
#pragma unroll
        for (int j = 0; j < NT; ++j) {
          const int col = c0 + j * 8 + g;
          const half* vp = ks + kk * SD + col;
          uint32_t bf[2];
          bf[0] = pk(vp[(2 * tg) * SD],       vp[(2 * tg + 1) * SD]);
          bf[1] = pk(vp[(2 * tg + 8) * SD],   vp[(2 * tg + 9) * SD]);
#pragma unroll
          for (int m = 0; m < MT; ++m) mma16816(acc[m][j], af[m], bf);
        }
      }
    }
    __syncthreads();
  }

  const int h = h0 + hh;
  const long long spoff = (long long)sp * Q * H;
  for (int m = 0; m < MT; ++m) {
    const int r0 = m * 16 + g, r1 = m * 16 + g + 8;
#pragma unroll
    for (int j = 0; j < NT; ++j) {
      const int col = c0 + j * 8 + 2 * tg;
      if (r0 < nq) {
        float* o = po + (spoff + (long long)(q0 + r0) * H + h) * L + col;
        o[0] = acc[m][j][0];
        o[1] = acc[m][j][1];
      }
      if (r1 < nq) {
        float* o = po + (spoff + (long long)(q0 + r1) * H + h) * L + col;
        o[0] = acc[m][j][2];
        o[1] = acc[m][j][3];
      }
    }
  }
  // publish running max / denominator for EVERY (head, row) this block owns.
  // NOTE: warp 0 only belongs to hh==0, so this must be driven by tid, not by w.
  __syncthreads();
  for (int e = tid; e < HPB * BQ; e += THREADS) {
    const int hx = e / BQ, r = e - hx * BQ;
    if (r < nq) {
      const long long o = spoff + (long long)(q0 + r) * H + (h0 + hx);
      pm[o] = mrow[hx * BQ + r];
      pl[o] = lrow[hx * BQ + r];
    }
  }
}

// ---- split-K second pass: combine NSPLIT partials with log-sum-exp rescaling ----
__global__ void ppm_reduce_kernel(const float* __restrict__ po,
                                  const float* __restrict__ pm,
                                  const float* __restrict__ pl,
                                  half* __restrict__ out, int Q, int H) {
  const long long row = (long long)blockIdx.x;      // one block per (q,h) row
  const long long QH = (long long)Q * H;
  float m = -FLT_MAX;
#pragma unroll 1
  for (int s = 0; s < NSPLIT; ++s) m = fmaxf(m, pm[s * QH + row]);
  float den = 0.f;
  float sc[NSPLIT];
#pragma unroll 1
  for (int s = 0; s < NSPLIT; ++s) {
    const float ms = pm[s * QH + row];
    const float f = (ms == -FLT_MAX || m == -FLT_MAX) ? 0.f : __expf(ms - m);
    sc[s] = f;
    den += pl[s * QH + row] * f;
  }
  const float inv = (den > 0.f) ? 1.f / den : 0.f;
  for (int d = threadIdx.x; d < L; d += blockDim.x) {
    float a = 0.f;
#pragma unroll 1
    for (int s = 0; s < NSPLIT; ++s) a += po[(s * QH + row) * L + d] * sc[s];
    out[row * L + d] = __float2half(a * inv);
  }
}

constexpr int SMEM_BYTES = HPB * BQ * SD * 2 + BK * SD * 2
                         + HPB * BQ * BK * 4 + 3 * HPB * BQ * 4;
}  // namespace

// Large append fast path: materialize one logical KV sequence, then let cuBLAS
// execute both dense MLA GEMMs in bounded query chunks. The existing online
// softmax kernel remains the no-allocation fallback.
static torch::Tensor paged_tc_mask(const torch::Device& dev) {
  constexpr int B = 1024;
  static std::mutex mu;
  static std::unordered_map<std::string, torch::Tensor> masks;
  const std::string key = std::to_string(dev.index()) + ":ppm_tc_1024";
  std::lock_guard<std::mutex> lock(mu);
  auto it = masks.find(key);
  if (it != masks.end()) return it->second;
  auto i = torch::arange(B, torch::TensorOptions().device(dev).dtype(torch::kInt64));
  return masks.emplace(key, i.unsqueeze(0) > i.unsqueeze(1)).first->second;
}

static bool paged_tc_fastpath(
    const torch::Tensor& ql, const torch::Tensor& qr,
    const torch::Tensor& pool, const torch::Tensor& table,
    int64_t K0, torch::Tensor* result) {
  constexpr int64_t B = 1024;
  const int64_t Q = ql.size(0), H = ql.size(1), K = K0 + Q;
  // The dense path wins consistently for long cached prefixes; below 3K the
  // fixed gather/cuBLAS overhead can exceed the online kernel.
  if (H != 8 || Q < 64 || K0 < 3072 || K <= 0) return false;
  const int64_t page_size = pool.size(1);
  const int64_t npages = (K + page_size - 1) / page_size;
  if (npages <= 0 || table.numel() < npages) return false;

  // Score is the dominant live temporary.  Measurements include allocator
  // transients; keep 384 MiB beyond score+gather and otherwise use flash.
  const uint64_t score_bytes = uint64_t(H) * uint64_t(std::min<int64_t>(B, Q)) *
                               uint64_t(K) * sizeof(at::Half);
  const uint64_t gather_bytes = uint64_t(K) * 576ull * sizeof(at::Half);
  const uint64_t need = score_bytes + gather_bytes + (384ull << 20);
  size_t free_bytes = 0, total_bytes = 0;
  if (cudaMemGetInfo(&free_bytes, &total_bytes) != cudaSuccess ||
      uint64_t(free_bytes) < need) return false;

  auto cache = pool.index_select(0, table.narrow(0, 0, npages))
                   .reshape({-1, 576}).narrow(0, 0, K);
  auto latent = cache.narrow(1, 0, 512);
  auto key = cache.transpose(0, 1);
  auto full_mask = paged_tc_mask(ql.device());
  std::vector<torch::Tensor> chunks;
  chunks.reserve((Q + B - 1) / B);
  for (int64_t q0 = 0; q0 < Q; q0 += B) {
    const int64_t n = std::min<int64_t>(B, Q - q0);
    const int64_t kend = K0 + q0 + n;
    auto query = torch::cat({ql.narrow(0, q0, n).transpose(0, 1),
                             qr.narrow(0, q0, n).transpose(0, 1)}, 2);
    query.mul_(0.0625);
    auto score = torch::matmul(query, key.narrow(1, 0, kend));
    score.narrow(2, kend - n, n).masked_fill_(
        full_mask.narrow(0, 0, n).narrow(1, 0, n),
        -std::numeric_limits<float>::infinity());
    at::softmax_out(score, score, -1);
    chunks.push_back(torch::matmul(score, latent.narrow(0, 0, kend))
                         .transpose(0, 1).contiguous());
  }
  *result = chunks.size() == 1 ? chunks[0] : torch::cat(chunks, 0);
  return true;
}

torch::Tensor paged_prefill_mla(torch::Tensor q_latent, torch::Tensor q_rope,
                               torch::Tensor pool, torch::Tensor page_table,
                               int64_t q_start) {
  TORCH_CHECK(q_latent.is_cuda() && q_rope.is_cuda() && pool.is_cuda() && page_table.is_cuda(), "CUDA tensors required");
  TORCH_CHECK(q_latent.scalar_type() == at::kHalf && q_rope.scalar_type() == at::kHalf && pool.scalar_type() == at::kHalf, "fp16 required");
  TORCH_CHECK(q_latent.dim() == 3 && q_latent.size(2) == L, "q_latent must be [Q,H,512]");
  TORCH_CHECK(q_rope.dim() == 3 && q_rope.size(2) == R && q_rope.size(0) == q_latent.size(0) && q_rope.size(1) == q_latent.size(1), "q_rope must be [Q,H,64]");
  TORCH_CHECK(pool.dim() == 3 && pool.size(2) == D, "pool must be [pages,page_size,576]");
  TORCH_CHECK(page_table.dim() == 1 && page_table.scalar_type() == at::kLong, "page_table must be int64[logical_pages]");
  TORCH_CHECK(q_latent.is_contiguous() && q_rope.is_contiguous() && pool.is_contiguous() && page_table.is_contiguous(), "contiguous required");
  c10::cuda::CUDAGuard guard(q_latent.device());

  const int Q = (int)q_latent.size(0), H = (int)q_latent.size(1);
  const int page_size = (int)pool.size(1);
  TORCH_CHECK(q_start >= 0, "q_start must be >= 0");
  TORCH_CHECK(q_start + Q <= page_table.size(0) * (int64_t)page_size, "page table too small for prefix");

  torch::Tensor tc_result;
  if (paged_tc_fastpath(q_latent, q_rope, pool, page_table, q_start, &tc_result))
    return tc_result;

  auto out = torch::empty_like(q_latent);
  auto stream = at::cuda::getCurrentCUDAStream();
  // cudaFuncSetAttribute is per-device: opt in once on EVERY device, not once globally.
  {
    static std::once_flag ppm_attr_once[64];
    int dev = q_latent.device().index();
    TORCH_CHECK(dev >= 0 && dev < 64, "unexpected device index");
    std::call_once(ppm_attr_once[dev], [&]() {
      C10_CUDA_CHECK(cudaFuncSetAttribute(paged_prefill_mla_kernel,
          cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_BYTES));
    });
  }
  TORCH_CHECK(H % HPB == 0, "H must be divisible by HPB");
  auto fo = q_latent.options().dtype(at::kFloat);
  auto po = torch::empty({(long long)NSPLIT, (long long)Q, (long long)H, (long long)L}, fo);
  auto pm = torch::empty({(long long)NSPLIT, (long long)Q, (long long)H}, fo);
  auto pl = torch::empty({(long long)NSPLIT, (long long)Q, (long long)H}, fo);
  dim3 grid((Q + BQ - 1) / BQ, H / HPB, NSPLIT);
  paged_prefill_mla_kernel<<<grid, THREADS, SMEM_BYTES, stream>>>(
      reinterpret_cast<const half*>(q_latent.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(q_rope.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(pool.data_ptr<at::Half>()),
      reinterpret_cast<const long long*>(page_table.data_ptr<int64_t>()),
      po.data_ptr<float>(), pm.data_ptr<float>(), pl.data_ptr<float>(),
      Q, H, page_size, (int)q_start);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  ppm_reduce_kernel<<<Q * H, 128, 0, stream>>>(
      po.data_ptr<float>(), pm.data_ptr<float>(), pl.data_ptr<float>(),
      reinterpret_cast<half*>(out.data_ptr<at::Half>()), Q, H);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

int64_t paged_prefill_mla_smem() { return SMEM_BYTES; }

#ifndef LJQ_PPM_EMBED
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &paged_prefill_mla, "Fused paged MLA prefill v2",
        pybind11::call_guard<pybind11::gil_scoped_release>());
  m.def("smem", &paged_prefill_mla_smem, "shared memory bytes per block");
}
#endif  // LJQ_PPM_EMBED
