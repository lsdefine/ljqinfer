// v4.1 fused sparse latent attention (decode).  Ported from the v4 paged
// tensor-core kernel: 1 block = 1 token, 8 warps, TILE=32 keys staged in
// shared, V == K (latent MLA), online softmax with a sink denominator.
// v4.1 differences from v4: no per-token `totals`, no `keff`, and the rows
// this decode window has not committed to the paged pool live in a `tail`
// bank instead of v4's paged compressor pool -- ids >= total index that tail
// directly.  A call may carry several requests: `qwin` rows belong to one
// request and `pts`/`tstride` are its page-table / tail-bank strides, so
// seq = token / qwin picks whose state a block reads.  Single-sequence
// callers pass qwin = T, pts = tstride = 0 and every expression collapses
// to the pre-batch form with no branch.
// Rows with id < 0 are dead (the caller pads both the top-k and the sliding
// window that way), exactly as in v4.
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

__global__ __launch_bounds__(NWARP * 32) void sparse_attn_decode_kernel(
    const bf16* __restrict__ q, const bf16* __restrict__ pool,
    const int64_t* __restrict__ page_table, const bf16* __restrict__ tail,
    const float* __restrict__ sink, const int64_t* __restrict__ idxs,
    bf16* __restrict__ out, int K, const int64_t* __restrict__ pos, int ratio,
    int page, float scale, int nhead,
    float* __restrict__ pacc, float* __restrict__ pml, int qwin, int64_t pts,
    int64_t tstride, const int64_t* __restrict__ rowmap) {
  extern __shared__ char smem[];
  bf16* kvs = (bf16*)smem;                  // [TILE][LDK]
  bf16* ps = kvs + TILE * LDK;              // [8][LDP]
  float* spart = (float*)(ps + 8 * LDP);    // [2][8][TILE]
  float* sh_corr = spart + 2 * 8 * TILE;    // [8]
  float* sh_m = sh_corr + 8;                // [8]
  float* sh_l = sh_m + 8;                   // [8]
  int* ok = (int*)(sh_l + 8);               // [TILE]

  const int token = blockIdx.x;
  const int tid = threadIdx.x;
  const int w = tid >> 5;
  const int lane = tid & 31;
  const int gid = lane >> 2;   // 0..7
  const int tig = lane & 3;    // 0..3
  // Paged/tail split is read on device so the launch is graph-replayable:
  // pos[seq*qwin] is that request's live write cursor, total = its compressed
  // rows in the pool.  seq is 0 and the two strides are 0 for a single
  // sequence, so this is identical to the pre-batch indexing.
  const int seq = token / qwin;
  const int64_t total = pos[(int64_t)seq * qwin] / ratio;
  // Requests need not own neighbouring rows of the state pools, so rowmap maps
  // request -> pool row and a batch reads the pools in place instead of first
  // gathering its rows into a packed copy.  Absent it, request b is row b.
  const int64_t row = rowmap ? rowmap[seq] : (int64_t)seq;
  const int64_t* __restrict__ mypt = page_table + row * pts;
  const bf16* __restrict__ mytail = tail + row * tstride * DIM;

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

  // Interleaved split-K spreads live tiles across SMs.
  // IDs may contain interior padding between compressed top-k and SWA.
  // An empty tile is not an end marker: later tiles can still be live.
  const int kstride = (int)gridDim.y * TILE;
  for (int kb = (int)blockIdx.y * TILE; kb < K; kb += kstride) {
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
            src = pool + (mypt[lp] * page + (id - lp * page)) * DIM;
          } else {
            src = mytail + (id - total) * DIM;
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
      if (!any) continue;
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
  if (pacc) {  // split-K: dump raw acc + (m,l); sink applied in combine
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
      const size_t pp2 =
          ((size_t)(token * gridDim.y + blockIdx.y) * nhead + tid) * 2;
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
__global__ void sparse_attn_decode_combine_kernel(
    const float* __restrict__ pacc, const float* __restrict__ pml,
    const float* __restrict__ sink, bf16* __restrict__ out, int S, int nhead) {
  const int token = blockIdx.x, h = blockIdx.y, tid = threadIdx.x;
  const size_t base = (size_t)(token * S) * nhead + h;
  float m = -INFINITY;
  for (int s = 0; s < S; ++s) m = fmaxf(m, pml[(base + (size_t)s * nhead) * 2]);
  float l = __expf(sink[h] - m);
  float f[16];  // S <= 16
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

void sparse_attn_decode(at::Tensor q, at::Tensor pool, at::Tensor page_table,
                        at::Tensor tail, at::Tensor sink, at::Tensor idxs,
                        at::Tensor out, at::Tensor pos, int64_t ratio, double scale,
                        int64_t qwin, at::Tensor rowmap) {
  const int h = q.size(-2), K = idxs.size(-1);
  const int T = q.numel() / ((int64_t)h * DIM);
  TORCH_CHECK(idxs.scalar_type() == at::kLong, "idxs must be int64");
  TORCH_CHECK(page_table.scalar_type() == at::kLong, "page_table must be int64");
  TORCH_CHECK(q.is_cuda() && pool.is_cuda() && tail.is_cuda(), "cuda tensors");
  TORCH_CHECK(idxs.size(0) == T, "one row of ids per query token");
  TORCH_CHECK(pos.scalar_type() == at::kLong && pos.is_cuda(), "pos must be cuda int64");
  // One request owns `qwin` consecutive rows.  A 2-D page table / 3-D tail
  // carry one entry per request; the 1-D/2-D single-sequence forms keep
  // stride 0 so the kernel indexes them exactly as before.
  TORCH_CHECK(qwin > 0 && T % qwin == 0, "qwin must divide the row count");
  TORCH_CHECK(h <= NWARP,
              "this kernel maps one head per warp: nhead must be <= 8");
  const int64_t nreq = T / qwin;
  const int64_t pts = page_table.dim() == 2 ? page_table.stride(0) : 0;
  const int64_t tstride = tail.dim() == 3 ? tail.size(1) : 0;
  TORCH_CHECK(pts == 0 || page_table.size(0) >= nreq, "page table rows < requests");
  TORCH_CHECK(tstride == 0 || tail.size(0) >= nreq, "tail rows < requests");
  TORCH_CHECK(nreq == 1 || (pts != 0 && tstride != 0),
              "batched calls need a per-request page table and tail bank");
  // An empty rowmap means "request b owns row b"; a given one lets the batch
  // address arbitrary slots of the full pools.
  const int64_t* rowmap_p = nullptr;
  if (rowmap.numel()) {
    TORCH_CHECK(rowmap.scalar_type() == at::kLong && rowmap.is_cuda(),
                "rowmap must be a cuda int64 tensor");
    TORCH_CHECK(rowmap.numel() >= nreq, "rowmap shorter than the request count");
    rowmap_p = (const int64_t*)rowmap.data_ptr();
  }

  const size_t sm = (size_t)TILE * LDK * 2 + (size_t)8 * LDP * 2 +
                    (2 * 8 * TILE + 24) * 4 + TILE * 4;
  auto st = at::cuda::getCurrentCUDAStream();
  cudaFuncSetAttribute(sparse_attn_decode_kernel,
                       cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm);

  // S (split-K chunks) changes the order of the float combine reduction: it is
  // a NUMERICS knob, not a tuning knob.  Fixed at 8, as in v4.
  int S = 8;
  const int ntile = (K + TILE - 1) / TILE;
  if (S > ntile) S = ntile;
  if (S < 1) S = 1;

  if (S <= 1) {
    sparse_attn_decode_kernel<<<T, NWARP * 32, sm, st>>>(
        (const bf16*)q.data_ptr(), (const bf16*)pool.data_ptr(),
        (const int64_t*)page_table.data_ptr(), (const bf16*)tail.data_ptr(),
        (const float*)sink.data_ptr(), (const int64_t*)idxs.data_ptr(),
        (bf16*)out.data_ptr(), K, (const int64_t*)pos.data_ptr(), (int)ratio, (int)pool.size(1), (float)scale,
        h, nullptr, nullptr, (int)qwin, pts, tstride, rowmap_p);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return;
  }

  // CUDA graphs freeze raw workspace pointers: keep one owned allocation per
  // (device, size) forever instead of reusing a shared buffer that could be
  // freed while an earlier captured graph still points at it.
  static std::mutex ws_mu;
  using Workspace = std::pair<at::Tensor, at::Tensor>;
  static std::unordered_map<int, std::unordered_map<int64_t, Workspace>> ws_map;
  float *pacc_p, *pml_p;
  Workspace tmp_ws;  // eager path: freed on return, no per-shape leak
  {
    const int64_t need = (int64_t)T * S * h * DIM;
    const int64_t need_ml = (int64_t)T * S * h * 2;
    auto opt = at::TensorOptions().dtype(at::kFloat).device(q.device());
    cudaStreamCaptureStatus cap = cudaStreamCaptureStatusNone;
    cudaStreamIsCapturing(st, &cap);
    if (cap != cudaStreamCaptureStatusNone) {
      std::lock_guard<std::mutex> lk(ws_mu);
      auto& dws = ws_map[q.get_device()];
      auto it = dws.find(need);
      if (it == dws.end())
        it = dws.emplace(need, std::make_pair(at::zeros({need}, opt),
                                              at::zeros({need_ml}, opt)))
                 .first;
      pacc_p = it->second.first.data_ptr<float>();
      pml_p = it->second.second.data_ptr<float>();
    } else {
      tmp_ws = std::make_pair(at::empty({need}, opt), at::empty({need_ml}, opt));
      pacc_p = tmp_ws.first.data_ptr<float>();
      pml_p = tmp_ws.second.data_ptr<float>();
    }
  }
  dim3 grid(T, S);
  sparse_attn_decode_kernel<<<grid, NWARP * 32, sm, st>>>(
      (const bf16*)q.data_ptr(), (const bf16*)pool.data_ptr(),
      (const int64_t*)page_table.data_ptr(), (const bf16*)tail.data_ptr(),
      (const float*)sink.data_ptr(), (const int64_t*)idxs.data_ptr(),
      (bf16*)out.data_ptr(), K, (const int64_t*)pos.data_ptr(), (int)ratio, (int)pool.size(1), (float)scale, h,
      pacc_p, pml_p, (int)qwin, pts, tstride, rowmap_p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  dim3 cgrid(T, h);
  sparse_attn_decode_combine_kernel<<<cgrid, 128, 0, st>>>(
      pacc_p, pml_p, (const float*)sink.data_ptr(), (bf16*)out.data_ptr(), S, h);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("sparse_attn_decode", &sparse_attn_decode,
        "v4.1 fused sparse paged latent attention (decode)",
        pybind11::call_guard<pybind11::gil_scoped_release>());
}
