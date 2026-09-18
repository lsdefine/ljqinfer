// TensorCore (mma.m16n8k16 bf16) sparse paged MLA prefill attention.
// 1 block = 1 token, 8 warps, TILE=32 keys staged in shared. V == K (MLA).
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <mutex>
#include <unordered_map>

#define DIM 512
#define NWARP 8
#define TILE 32
#define LDK (DIM + 8)
#define LDQ (DIM + 8)
#define LDP (TILE + 8)

typedef __nv_bfloat16 bf16;

__device__ __forceinline__ void mma16816(float* d, const unsigned* a,
                                         const unsigned* b, const float* c) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
      : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]),
        "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
}

__device__ __forceinline__ unsigned pk(const bf16* p) {
  return *(const unsigned*)p;  // two contiguous bf16
}

__global__ __launch_bounds__(NWARP * 32) void sparse_attn_paged_tc_kernel(
    const bf16* __restrict__ q, const bf16* __restrict__ pool,
    const int64_t* __restrict__ page_table, const bf16* __restrict__ cpool,
    const int64_t* __restrict__ ctable, const float* __restrict__ sink,
    const int64_t* __restrict__ idxs, bf16* __restrict__ out, int K, int total,
    const int64_t* __restrict__ totals, int page, int cpage, float scale,
    int nhead, float* __restrict__ pacc, float* __restrict__ pml, int kchunk,
    int Q, long page_table_stride, long ctable_stride,
    const int64_t* __restrict__ keff) {
  extern __shared__ char smem[];
  bf16* kvs = (bf16*)smem;                  // [TILE][LDK]
  bf16* ps = kvs + TILE * LDK;              // [8][LDP]
  float* spart = (float*)(ps + 8 * LDP);    // [2][8][TILE]
  float* sh_corr = spart + 2 * 8 * TILE;    // [8]
  float* sh_m = sh_corr + 8;                // [8]
  float* sh_l = sh_m + 8;                   // [8]
  int* ok = (int*)(sh_l + 8);               // [TILE]

  const int token = blockIdx.x;
  const int batch = token / Q;
  page_table += (long)batch * page_table_stride;
  ctable += (long)batch * ctable_stride;
  if (totals) total = (int)totals[token];
  const int tid = threadIdx.x;
  const int w = tid >> 5;
  const int lane = tid & 31;
  const int gid = lane >> 2;   // 0..7
  const int tig = lane & 3;    // 0..3

  const int kg = w >> 2;             // contraction half (0/1)
  const int keybase = (w & 3) * 8;   // this warp's 8 keys for QK^T
  const int dimbase = w * 64;        // this warp's 64 dims for PV

  // ---- Q fragments in registers (rows 8..15 are zero -> a1=a3=0) ----
  unsigned qa[16][2];
  {
    const bf16* qp = q + ((size_t)token * nhead + gid) * DIM + kg * 256;
    const bool live = (gid < nhead);
#pragma unroll
    for (int ks = 0; ks < 16; ++ks) {
      qa[ks][0] = live ? pk(qp + ks * 16 + tig * 2) : 0u;
      qa[ks][1] = live ? pk(qp + ks * 16 + tig * 2 + 8) : 0u;
    }
  }

  float acc[8][4];
#pragma unroll
  for (int t = 0; t < 8; ++t)
#pragma unroll
    for (int j = 0; j < 4; ++j) acc[t][j] = 0.f;
  if (tid < 8) { sh_m[tid] = -INFINITY; sh_l[tid] = 0.f; }

  const int64_t* myidx = idxs + (size_t)token * K;

  // Interleaved split-K (was: contiguous [blockIdx.y*kchunk, +kchunk)).
  // The live ids form ONE contiguous run (window part may be -1-padded in FRONT
  // when tok < W, the compressed/topk part is -1-padded at the BACK), so with
  // contiguous chunking every live row landed in chunk 0: one block did all the
  // work while the other S-1 scanned their whole chunk of dead tiles, making the
  // cost grow with the STATIC K (= 128 + max_seq/128) instead of the real
  // context length.  Interleaving gives tile t to block t % S, so the live tiles
  // are spread evenly over all S blocks and each block still sees its own tiles
  // in increasing order (the `seen` early-out below stays valid).
  // keff (optional, per token): real upper bound of the live run.  Blocks whose
  // interleaved tiles all sit past it would otherwise scan the whole STATIC K
  // (= 128 + max_seq/128) before the `seen` guard can fire -- and since the
  // T*S blocks run concurrently, the kernel costs as much as that idle scan.
  const int kstride = (int)gridDim.y * TILE;
  const int Kend = keff ? min(K, (int)keff[token]) : K;
  int seen = 0;
  for (int kb = (int)blockIdx.y * TILE; kb < Kend; kb += kstride) {
    __syncthreads();
    // ---- stage TILE keys (=values) ----
    for (int slot = w; slot < TILE; slot += NWARP) {
      const int kk = kb + slot;
      int good = 0;
      const bf16* src = nullptr;
      if (kk < K) {
        const int64_t id = myidx[kk];
        if (id >= 0) {
          if (id < total) {
            const int64_t lp = id / page;
            src = pool + (page_table[lp] * page + (id - lp * page)) * DIM;
          } else {
            const int64_t c = id - total;
            const int64_t lp = c / cpage;
            src = cpool + (ctable[lp] * cpage + (c - lp * cpage)) * DIM;
          }
          good = 1;
        }
      }
#pragma unroll
      for (int j = 0; j < 2; ++j) {
        uint4 v = make_uint4(0, 0, 0, 0);
        if (good) v = *(const uint4*)(src + j * 256 + lane * 8);
        *(uint4*)(kvs + slot * LDK + j * 256 + lane * 8) = v;
      }
      if (lane == 0) ok[slot] = good;
    }
    __syncthreads();

    {
      int any = 0;
#pragma unroll
      for (int i = 0; i < TILE; ++i) any |= ok[i];
      if (!any) {
        if (seen) break;
        continue;
      }
      seen = 1;
    }

    // ---- QK^T : 16(head) x 8(key) x 256(dim half) ----
    {
      float c[4] = {0.f, 0.f, 0.f, 0.f};
      const bf16* kp = kvs + (keybase + gid) * LDK + kg * 256;
#pragma unroll 4
      for (int ks = 0; ks < 16; ++ks) {
        unsigned a[4] = {qa[ks][0], 0u, qa[ks][1], 0u};
        unsigned b[2];
        b[0] = pk(kp + ks * 16 + tig * 2);
        b[1] = pk(kp + ks * 16 + tig * 2 + 8);
        mma16816(c, a, b, c);
      }
      // c0/c1: head=gid, key=keybase+2*tig{,+1}
      spart[(kg * 8 + gid) * TILE + keybase + tig * 2] = c[0];
      spart[(kg * 8 + gid) * TILE + keybase + tig * 2 + 1] = c[1];
    }
    __syncthreads();

    // ---- online softmax: warp w owns head w, lane = key slot ----
    if (w < 8) {
      float s = -INFINITY;
      if (ok[lane] && (kb + lane) < K)
        s = (spart[w * TILE + lane] + spart[(8 + w) * TILE + lane]) * scale;
      float mx = s;
#pragma unroll
      for (int o = 16; o; o >>= 1)
        mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, o));
      const float mold = sh_m[w];
      const float mnew = fmaxf(mold, mx);
      const float corr = (mold == -INFINITY) ? 0.f : __expf(mold - mnew);
      float p = (s == -INFINITY) ? 0.f : __expf(s - mnew);
      float sum = p;
#pragma unroll
      for (int o = 16; o; o >>= 1) sum += __shfl_xor_sync(0xffffffff, sum, o);
      ps[w * LDP + lane] = __float2bfloat16(p);
      if (lane == 0) {
        sh_corr[w] = corr;
        sh_m[w] = mnew;
        sh_l[w] = sh_l[w] * corr + sum;
      }
    }
    __syncthreads();

    // ---- rescale accumulator ----
    {
      const float corr = sh_corr[gid];
#pragma unroll
      for (int t = 0; t < 8; ++t) {
        acc[t][0] *= corr;
        acc[t][1] *= corr;
      }
    }

    // ---- PV : 16(head) x 64(dim) x 32(key), V == K ----
#pragma unroll
    for (int t = 0; t < 8; ++t) {
      const int dcol = dimbase + t * 8 + gid;  // n index -> dim
#pragma unroll
      for (int ks = 0; ks < 2; ++ks) {
        const int k0 = ks * 16;
        unsigned a[4], b[2];
        a[0] = pk(ps + gid * LDP + k0 + tig * 2);
        a[1] = 0u;
        a[2] = pk(ps + gid * LDP + k0 + 8 + tig * 2);
        a[3] = 0u;
        const bf16* vp = kvs + (k0 + tig * 2) * LDK + dcol;
        b[0] = *(unsigned*)&__halves2bfloat162(vp[0], vp[LDK]);
        const bf16* vq = vp + 8 * LDK;
        b[1] = *(unsigned*)&__halves2bfloat162(vq[0], vq[LDK]);
        mma16816(acc[t], a, b, acc[t]);
      }
    }
  }

  // ---- epilogue ----
  __syncthreads();
  if (pacc) {  // splitK: dump raw acc + (m,l); sink applied in combine
    if (gid < nhead) {
      const size_t part = (size_t)(token * gridDim.y + blockIdx.y) * nhead + gid;
      float* pp = pacc + part * DIM;
#pragma unroll
      for (int t = 0; t < 8; ++t) {
        const int d = dimbase + t * 8 + tig * 2;
        pp[d] = acc[t][0];
        pp[d + 1] = acc[t][1];
      }
    }
    if (tid < nhead) {
      const size_t pp2 = ((size_t)(token * gridDim.y + blockIdx.y) * nhead + tid) * 2;
      pml[pp2] = sh_m[tid];
      pml[pp2 + 1] = sh_l[tid];
    }
    return;
  }
  if (gid < nhead) {
    const float l = sh_l[gid] + __expf(sink[gid] - sh_m[gid]);
    const float inv = 1.f / l;
    bf16* op = out + ((size_t)token * nhead + gid) * DIM;
#pragma unroll
    for (int t = 0; t < 8; ++t) {
      const int d = dimbase + t * 8 + tig * 2;
      op[d] = __float2bfloat16(acc[t][0] * inv);
      op[d + 1] = __float2bfloat16(acc[t][1] * inv);
    }
  }
}

// combine: 1 block = 1 (token, head); S partials -> out
__global__ void sparse_attn_splitk_combine_kernel(
    const float* __restrict__ pacc, const float* __restrict__ pml,
    const float* __restrict__ sink, bf16* __restrict__ out, int S, int nhead) {
  const int token = blockIdx.x, h = blockIdx.y, tid = threadIdx.x;
  const size_t base = (size_t)(token * S) * nhead + h;
  float m = -INFINITY;
  for (int s = 0; s < S; ++s) m = fmaxf(m, pml[(base + (size_t)s * nhead) * 2]);
  float l = __expf(sink[h] - m);
  float f[16];  // S<=16
  for (int s = 0; s < S; ++s) {
    const size_t p = (base + (size_t)s * nhead) * 2;
    const float ms = pml[p];
    const float fs = (ms == -INFINITY) ? 0.f : __expf(ms - m);
    f[s] = fs;
    l += pml[p + 1] * fs;
  }
  const float inv = 1.f / l;
  bf16* op = out + ((size_t)token * nhead + h) * DIM;
  for (int d = tid; d < DIM; d += blockDim.x) {
    float v = 0.f;
    for (int s = 0; s < S; ++s)
      v += pacc[(base + (size_t)s * nhead) * DIM + d] * f[s];
    op[d] = __float2bfloat16(v * inv);
  }
}

at::Tensor sparse_attn_paged(at::Tensor q, at::Tensor pool,
                             at::Tensor page_table, at::Tensor cpool,
                             at::Tensor ctable, at::Tensor sink,
                             at::Tensor idxs, int64_t total, double scale,
                             c10::optional<at::Tensor> keff) {
  const int h = q.size(-2), K = idxs.size(-1);
  const int T = q.numel() / ((int64_t)h * DIM);
  const int B = page_table.dim() == 2 ? (int)page_table.size(0) : 1;
  TORCH_CHECK(T % B == 0 && (ctable.dim() == 1 || ctable.size(0) == B),
              "attention table batch mismatch");
  const int Q = T / B;
  const long pts = page_table.dim() == 2 ? page_table.stride(0) : 0L;
  const long cts = ctable.dim() == 2 ? ctable.stride(0) : 0L;
  auto out = at::empty_like(q);
  const int64_t* keff_p = nullptr;
  if (keff.has_value() && keff->numel() > 0) {
    TORCH_CHECK(keff->scalar_type() == at::kLong && keff->is_cuda() &&
                    keff->numel() >= T,
                "keff must be an int64 cuda tensor with >= T entries");
    keff_p = (const int64_t*)keff->data_ptr();
  }
  size_t sm = (size_t)TILE * LDK * 2 + (size_t)8 * LDP * 2 +
              (2 * 8 * TILE + 24) * 4 + TILE * 4;
  auto st = at::cuda::getCurrentCUDAStream();
  cudaFuncSetAttribute(sparse_attn_paged_tc_kernel,
                       cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm);
  // split-K over the K selected rows: decode has T*h blocks only (B1Q8 -> 8),
  // so one block per token leaves the SMs idle.  S chunks per token + combine.
  // NOTE: S (split-K chunks) changes the order of the float combine reduction,
  // i.e. it is a NUMERICS knob, not a tuning knob.  Fixed at 8 on purpose --
   // do not turn this back into an env var.
  // Prefill (T >= 1024 tokens -> >= 1024 blocks, ~10x the SM count) does not
  // need split-K for occupancy at all; with S=8 each block only saw ~2.5 tiles
  // (K=640), and the partial acc round-trip (T*S*h*512*4 B = 1.6 GB/layer at
  // 12k tokens) + combine kernel were pure overhead (~140 ms/chunk of 49k).
  // S=1 keeps the ONE online-softmax pass (no combine), decode path (small T)
  // keeps S=8 bit-for-bit as before.
  int S = (T >= 1024) ? 1 : 8;
  if (S > 1) {
    const int ntile = (K + TILE - 1) / TILE;
    if (S > ntile) S = ntile;
    if (S > 16) S = 16;
  }
  if (S <= 1) {
    sparse_attn_paged_tc_kernel<<<T, NWARP * 32, sm, st>>>(
        (const bf16*)q.data_ptr(), (const bf16*)pool.data_ptr(),
        (const int64_t*)page_table.data_ptr(), (const bf16*)cpool.data_ptr(),
        (const int64_t*)ctable.data_ptr(), (const float*)sink.data_ptr(),
        (const int64_t*)idxs.data_ptr(), (bf16*)out.data_ptr(), K, (int)total,
        nullptr, (int)pool.size(1), (int)cpool.size(1), (float)scale, h,
        nullptr, nullptr, K, Q, pts, cts, keff_p);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
  }
  const int kchunk = ((K + S - 1) / S + TILE - 1) / TILE * TILE;
  // CUDA graphs freeze raw workspace pointers: keep one owned allocation per
  // (device, size) forever instead of growing a shared buffer (which would
  // leave earlier captured graphs pointing at freed memory).
  static std::mutex ws_mu;
  using Workspace = std::pair<at::Tensor, at::Tensor>;
  static std::unordered_map<int, std::unordered_map<int64_t, Workspace>> ws_map;
  float *pacc_p, *pml_p;
  Workspace tmp_ws;   // transient (eager path): freed on return, no per-shape leak
  {
    const int64_t need = (int64_t)T * 16 * h * DIM;
    auto opt = at::TensorOptions().dtype(at::kFloat).device(q.device());
    cudaStreamCaptureStatus cap = cudaStreamCaptureStatusNone;
    cudaStreamIsCapturing(st, &cap);
    if (cap != cudaStreamCaptureStatusNone) {
      std::lock_guard<std::mutex> lk(ws_mu);
      auto& dws = ws_map[q.get_device()];
      auto it = dws.find(need);
      if (it == dws.end()) {
        it = dws.emplace(need, std::make_pair(at::zeros({need}, opt),
                                              at::zeros({(int64_t)T * 16 * h * 2}, opt))).first;
      }
      pacc_p = it->second.first.data_ptr<float>();
      pml_p = it->second.second.data_ptr<float>();
    } else {
      tmp_ws = std::make_pair(at::empty({need}, opt), at::empty({(int64_t)T * 16 * h * 2}, opt));
      pacc_p = tmp_ws.first.data_ptr<float>();
      pml_p = tmp_ws.second.data_ptr<float>();
    }
  }
  dim3 grid(T, S);
  sparse_attn_paged_tc_kernel<<<grid, NWARP * 32, sm, st>>>(
      (const bf16*)q.data_ptr(), (const bf16*)pool.data_ptr(),
      (const int64_t*)page_table.data_ptr(), (const bf16*)cpool.data_ptr(),
      (const int64_t*)ctable.data_ptr(), (const float*)sink.data_ptr(),
      (const int64_t*)idxs.data_ptr(), (bf16*)out.data_ptr(), K, (int)total,
      nullptr, (int)pool.size(1), (int)cpool.size(1), (float)scale, h,
      pacc_p, pml_p, kchunk, Q, pts, cts, keff_p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  dim3 cgrid(T, h);
  sparse_attn_splitk_combine_kernel<<<cgrid, 128, 0, st>>>(
      pacc_p, pml_p, (const float*)sink.data_ptr(), (bf16*)out.data_ptr(), S, h);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}


#ifndef DSV4_NO_PYBIND
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("sparse_attn_paged", &sparse_attn_paged, "TC fused sparse paged MLA",
        pybind11::call_guard<pybind11::gil_scoped_release>());
}
#endif
