// peer_ar_rows.cu -- one-shot NVLink peer AllReduce over the LIVE PREFIX of each
// row of a row-major fp32 [rows, C] tensor (the indexer score).
//
// Why: NCCL all_reduce inside a CUDA graph has a host-side count frozen at
// capture (= rows*C_max), so the indexer paid ~1.1-1.6 ms/step for empty
// capacity at MAXSEQ=393k.  Here the per-row length lim_s = (pos[s]+1)/ratio is
// read ON DEVICE, so the same captured graph moves only the live columns.
// Columns >= lim_s are -inf on every rank by construction (index_score_fused),
// and sum(-inf, ...) == -inf, so leaving them untouched is value-identical.
//
// Protocol per call (same mailbox/flag/seq scheme as peer_ar_ipc.cu; separate
// state so that lineage file is untouched):
//   1. every lane (block) pushes its column-slice of every live row into slot r
//      of every peer's mailbox (double-buffered by the lane's seq parity),
//   2. raises its flag on every peer, 3. spins on its own flag row until all R
//      delivered, 4. sums the R slots in fixed order j=0..R-1 (deterministic)
//      and writes back in place.
// Buffers are raw cudaMalloc'd (torch allocator pointers are not IPC bases) and
// exchanged as cudaIpcMemHandle_t blobs through torch.distributed.

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <cstring>

namespace {

constexpr int MAXR = 8;
constexpr int MAXLANES = 128;             // max blocks per call = flag rows
static int g_lanes = 16;
constexpr int MAXROWS = 64;
static int g_oneshot_max4 = -1;   // flattened two-shot wins at every L measured (2026-09-08); one-shot kept for A/B
static int g_dbg = 0;
static int g_per_lane4 = 4096;  // adaptive active lanes: ceil(tot4 / per_lane4), clamped to [1, gridDim]   // debug: bit0/1/2 skip two-shot phase 1/2/3 data (timing only)   // total live float4 (~0.65MB) below which one-shot (8x traffic, 1 sync) wins

struct RowsTab {
  float* mail[MAXR];
  unsigned* flags[MAXR];
};

struct RowsState {
  bool ready = false;
  int r = 0, R = 0, dev = -1;
  int64_t n = 0;                          // max numel per call (registered)
  float* mail = nullptr;                  // 2 * (R+1) * n floats: parity x (R slots + result)
  unsigned* flags = nullptr;              // 2 phases * LANES * MAXR
  unsigned* seq = nullptr;                // LANES (private)
  void* opened[MAXR][2] = {};
  RowsTab* tab_dev = nullptr;
};
RowsState g_rs;

__device__ __forceinline__ void st_release_sys(unsigned* p, unsigned v) {
  asm volatile("st.release.sys.global.u32 [%0], %1;" :: "l"(p), "r"(v) : "memory");
}
__device__ __forceinline__ unsigned ld_acquire_sys(const unsigned* p) {
  unsigned v; asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory"); return v;
}

// segment p of a row's live4 (owner rank p), then lane's share of that segment
__device__ __forceinline__ void seg_of(int p, int R, int lane, int L, int live4, int& lo, int& hi) {
  const int perR = (live4 + R - 1) / R;
  const int slo = p * perR, shi = min(live4, slo + perR);
  const int perL = (shi - slo + L - 1) / L;
  lo = slo + lane * perL; hi = min(shi, lo + perL);
}

// Two-shot: (1) push segment p of my data to owner p's slot r; (2) owner reduces
// its segment in fixed order j=0..R-1 and pushes the result to every peer's
// result slot (index R); (3) everybody copies its result slot back in place.
// Traffic per rank = 2x live (one-shot was 8x -> lost to NCCL at 393k).
__global__ void __launch_bounds__(256)
ar_rows_f32_ipc_kernel(int r, int R, int rows, int C, int64_t n, int ratio, int oneshot_max4, int dbg, int per_lane4,
                       const RowsTab* __restrict__ tab,
                       float* __restrict__ buf,
                       const int64_t* __restrict__ pos,
                       unsigned* __restrict__ myseq) {
  const int lane = blockIdx.x;
  const int tid = threadIdx.x, nt = blockDim.x;

  // total live float4 over all rows decides algorithm + active lane count (identical on all ranks)
  int tot4 = 0;
  for (int s = 0; s < rows; ++s) {
    const long long lim = (pos[s] + 1) / ratio;
    if (lim > 0) tot4 += (int)min((long long)C, (lim + 3) & ~3LL) >> 2;
  }
  const int L = max(1, min((int)gridDim.x, (tot4 + per_lane4 - 1) / max(1, per_lane4)));
  if (lane >= L) return;   // inactive lane: no seq bump, no flags (every rank agrees on L)

  __shared__ unsigned s_seq;
  if (tid == 0) s_seq = atomicAdd(myseq + lane, 1u) + 1u;
  __syncthreads();
  const unsigned seq = s_seq;
  const int64_t par = (int64_t)(seq & 1u) * (int64_t)(R + 1) * n;   // (R slots + result) per parity
  if (tot4 <= oneshot_max4) {
    // ---- one-shot: push my live rows to every peer (slot r), sum R slots locally.
    for (int s = 0; s < rows; ++s) {
      const long long lim = (pos[s] + 1) / ratio;
      if (lim <= 0) continue;
      const int live4 = (int)min((long long)C, (lim + 3) & ~3LL) >> 2;
      int lo, hi; seg_of(0, 1, lane, L, live4, lo, hi);
      const int4* src = reinterpret_cast<const int4*>(buf + (int64_t)s * C);
      for (int i = lo + tid; i < hi; i += nt) {
        const int4 v = src[i];
        for (int p = 0; p < R; ++p)
          reinterpret_cast<int4*>(tab->mail[p] + par + (int64_t)r * n + (int64_t)s * C)[i] = v;
      }
    }
    __syncthreads();
    if (tid < R) st_release_sys(tab->flags[tid] + lane * MAXR + r, seq);
    if (tid < R) { while (ld_acquire_sys(tab->flags[r] + lane * MAXR + tid) < seq) { } }
    __syncthreads();
    const float* m = tab->mail[r] + par;
    for (int s = 0; s < rows; ++s) {
      const long long lim = (pos[s] + 1) / ratio;
      if (lim <= 0) continue;
      const int live4 = (int)min((long long)C, (lim + 3) & ~3LL) >> 2;
      int lo, hi; seg_of(0, 1, lane, L, live4, lo, hi);
      const int64_t row = (int64_t)s * C;
      for (int i = lo + tid; i < hi; i += nt) {
        float4 acc = reinterpret_cast<const float4*>(m + row)[i];
        for (int j = 1; j < R; ++j) {
          const float4 v = reinterpret_cast<const float4*>(m + (int64_t)j * n + row)[i];
          acc.x += v.x; acc.y += v.y; acc.z += v.z; acc.w += v.w;
        }
        reinterpret_cast<float4*>(buf + row)[i] = acc;
      }
    }
    return;
  }
  // ---- two-shot (large live): reduce-scatter + all-gather, 2x traffic.
  // Single flattened index space G = sum_s perR_s over (row s, in-segment offset k); owner p holds
  // element i = p*perR_s + k (valid if i < live4_s). ALL phases split G across lanes identically on
  // every rank, so lane k on rank A only ever depends on lane k of rank B -> per-lane flags suffice.
  // (Earlier variant split phase 1 over tot and phases 2/3 over the local segment: cross-lane
  // dependency not covered by the per-lane flags -> silent corruption at large L.)
  __shared__ int s_poff[MAXROWS + 1], s_live[MAXROWS], s_perR[MAXROWS];
  if (tid == 0) {
    int acc = 0;
    for (int s = 0; s < rows; ++s) {
      const long long lim = (pos[s] + 1) / ratio;
      const int live4 = lim > 0 ? ((int)min((long long)C, (lim + 3) & ~3LL) >> 2) : 0;
      const int perR = (live4 + R - 1) / R;
      s_live[s] = live4; s_perR[s] = perR; s_poff[s] = acc; acc += perR;
    }
    s_poff[rows] = acc;
  }
  __syncthreads();
  const int G = s_poff[rows];
  const int per_lane = (G + L - 1) / L, g_lo = lane * per_lane, g_hi = min(G, g_lo + per_lane);

  // 1. scatter: (s,k) of owner p -> mail[p] slot r
  if (!(dbg & 1)) {
    int s = 0;
    for (int g = g_lo + tid; g < g_hi; g += nt) {
      while (g >= s_poff[s + 1]) ++s;
      const int k = g - s_poff[s], perR = s_perR[s], live4 = s_live[s];
      const int64_t row = (int64_t)s * C;
      for (int p = 0; p < R; ++p) {
        const int i = p * perR + k;
        if (i < live4)
          reinterpret_cast<int4*>(tab->mail[p] + par + (int64_t)r * n + row)[i] =
              reinterpret_cast<const int4*>(buf + row)[i];
      }
    }
  }
  __syncthreads();
  if (tid < R) st_release_sys(tab->flags[tid] + lane * MAXR + r, seq);
  if (tid < R) { while (ld_acquire_sys(tab->flags[r] + lane * MAXR + tid) < seq) { } }
  __syncthreads();

  // 2. reduce my segment (fixed order j=0..R-1), push result to every peer's result region
  if (!(dbg & 2)) {
    const float* m = tab->mail[r] + par;
    int s = 0;
    for (int g = g_lo + tid; g < g_hi; g += nt) {
      while (g >= s_poff[s + 1]) ++s;
      const int i = r * s_perR[s] + (g - s_poff[s]);
      if (i >= s_live[s]) continue;
      const int64_t row = (int64_t)s * C;
      float4 acc = reinterpret_cast<const float4*>(m + row)[i];
      for (int j = 1; j < R; ++j) {
        const float4 v = reinterpret_cast<const float4*>(m + (int64_t)j * n + row)[i];
        acc.x += v.x; acc.y += v.y; acc.z += v.z; acc.w += v.w;
      }
      for (int p = 0; p < R; ++p)
        reinterpret_cast<float4*>(tab->mail[p] + par + (int64_t)R * n + row)[i] = acc;
    }
  }
  __syncthreads();
  if (tid < R) st_release_sys(tab->flags[tid] + MAXLANES * MAXR + lane * MAXR + r, seq);   // phase-2 flag region
  if (tid < R) { while (ld_acquire_sys(tab->flags[r] + MAXLANES * MAXR + lane * MAXR + tid) < seq) { } }
  __syncthreads();

  // 3. gather: copy every owner's (s,k) from my result region into buf
  if (!(dbg & 4)) {
    const float* res = tab->mail[r] + par + (int64_t)R * n;
    int s = 0;
    for (int g = g_lo + tid; g < g_hi; g += nt) {
      while (g >= s_poff[s + 1]) ++s;
      const int k = g - s_poff[s], perR = s_perR[s], live4 = s_live[s];
      const int64_t row = (int64_t)s * C;
      for (int p = 0; p < R; ++p) {
        const int i = p * perR + k;
        if (i < live4)
          reinterpret_cast<float4*>(buf + row)[i] = reinterpret_cast<const float4*>(res + row)[i];
      }
    }
  }
}

void rows_free() {
  if (g_rs.dev >= 0) cudaSetDevice(g_rs.dev);
  for (int p = 0; p < MAXR; ++p)
    for (int k = 0; k < 2; ++k)
      if (g_rs.opened[p][k]) { cudaIpcCloseMemHandle(g_rs.opened[p][k]); g_rs.opened[p][k] = nullptr; }
  if (g_rs.tab_dev) { cudaFree(g_rs.tab_dev); g_rs.tab_dev = nullptr; }
  if (g_rs.mail)  { cudaFree(g_rs.mail);  g_rs.mail = nullptr; }
  if (g_rs.flags) { cudaFree(g_rs.flags); g_rs.flags = nullptr; }
  if (g_rs.seq)   { cudaFree(g_rs.seq);   g_rs.seq = nullptr; }
  g_rs.ready = false; g_rs.n = 0; g_rs.R = 0;
}

}  // namespace

torch::Tensor peer_ar_rows_alloc(int64_t rank, int64_t world, int64_t n_max) {
  TORCH_CHECK(world >= 2 && world <= MAXR, "peer_ar_rows: world out of range");
  TORCH_CHECK(rank >= 0 && rank < world, "peer_ar_rows: bad rank");
  TORCH_CHECK(n_max > 0 && n_max % 4 == 0, "peer_ar_rows: n_max must be x4");
  if (g_rs.ready) rows_free();
  int dev = -1;
  C10_CUDA_CHECK(cudaGetDevice(&dev));
  g_rs.dev = dev; g_rs.r = (int)rank; g_rs.R = (int)world; g_rs.n = n_max;
  const size_t mail_bytes = sizeof(float) * (size_t)(2 * (world + 1) * n_max);
  C10_CUDA_CHECK(cudaMalloc(&g_rs.mail, mail_bytes));
  C10_CUDA_CHECK(cudaMemset(g_rs.mail, 0, mail_bytes));
  C10_CUDA_CHECK(cudaMalloc(&g_rs.flags, sizeof(unsigned) * 2 * MAXLANES * MAXR));
  C10_CUDA_CHECK(cudaMemset(g_rs.flags, 0, sizeof(unsigned) * 2 * MAXLANES * MAXR));
  C10_CUDA_CHECK(cudaMalloc(&g_rs.seq, sizeof(unsigned) * MAXLANES));
  C10_CUDA_CHECK(cudaMemset(g_rs.seq, 0, sizeof(unsigned) * MAXLANES));
  cudaIpcMemHandle_t h[2];
  C10_CUDA_CHECK(cudaIpcGetMemHandle(&h[0], g_rs.mail));
  C10_CUDA_CHECK(cudaIpcGetMemHandle(&h[1], g_rs.flags));
  auto out = torch::empty({(int64_t)sizeof(h)}, torch::TensorOptions().dtype(torch::kUInt8));
  std::memcpy(out.data_ptr(), h, sizeof(h));
  return out;
}

void peer_ar_rows_open(torch::Tensor all_handles) {
  TORCH_CHECK(g_rs.mail && g_rs.flags, "peer_ar_rows_open before alloc");
  const int R = g_rs.R;
  TORCH_CHECK(all_handles.dim() == 2 && all_handles.size(0) == R &&
                  all_handles.size(1) == (int64_t)(2 * sizeof(cudaIpcMemHandle_t)) &&
                  all_handles.scalar_type() == torch::kUInt8 &&
                  all_handles.is_contiguous() && all_handles.device().is_cpu(),
              "peer_ar_rows_open: bad handle blob");
  C10_CUDA_CHECK(cudaSetDevice(g_rs.dev));
  RowsTab tab_h{};
  const uint8_t* base = all_handles.data_ptr<uint8_t>();
  for (int p = 0; p < R; ++p) {
    if (p == g_rs.r) { tab_h.mail[p] = g_rs.mail; tab_h.flags[p] = g_rs.flags; continue; }
    cudaIpcMemHandle_t h[2];
    std::memcpy(h, base + (size_t)p * 2 * sizeof(cudaIpcMemHandle_t), sizeof(h));
    void *pm = nullptr, *pf = nullptr;
    C10_CUDA_CHECK(cudaIpcOpenMemHandle(&pm, h[0], cudaIpcMemLazyEnablePeerAccess));
    C10_CUDA_CHECK(cudaIpcOpenMemHandle(&pf, h[1], cudaIpcMemLazyEnablePeerAccess));
    g_rs.opened[p][0] = pm; g_rs.opened[p][1] = pf;
    tab_h.mail[p] = (float*)pm; tab_h.flags[p] = (unsigned*)pf;
  }
  C10_CUDA_CHECK(cudaMalloc(&g_rs.tab_dev, sizeof(RowsTab)));
  C10_CUDA_CHECK(cudaMemcpy(g_rs.tab_dev, &tab_h, sizeof(RowsTab), cudaMemcpyHostToDevice));
  g_rs.ready = true;
}

// y: contiguous fp32 [..., C] cuda (rows = numel / C); pos: int64 cuda [rows].
void peer_ar_rows_run(torch::Tensor y, torch::Tensor pos, int64_t ratio) {
  TORCH_CHECK(g_rs.ready, "peer_ar_rows: not registered");
  TORCH_CHECK(y.is_cuda() && y.scalar_type() == torch::kFloat32 && y.is_contiguous(),
              "peer_ar_rows_run expects contiguous fp32 cuda");
  const int64_t C = y.size(-1);
  const int64_t rows = y.numel() / C;
  TORCH_CHECK(C % 4 == 0 && rows * C <= g_rs.n, "peer_ar_rows_run: shape ", rows, "x", C,
              " exceeds registered ", g_rs.n, " or C not x4");
  TORCH_CHECK(pos.is_cuda() && pos.scalar_type() == torch::kInt64 && pos.is_contiguous() &&
                  pos.numel() == rows, "peer_ar_rows_run: pos must be int64 [rows]");
  auto st = at::cuda::getCurrentCUDAStream();
  TORCH_CHECK(rows <= MAXROWS, "peer_ar_rows: rows > MAXROWS");
  ar_rows_f32_ipc_kernel<<<g_lanes, 256, 0, st>>>(
      g_rs.r, g_rs.R, (int)rows, (int)C, g_rs.n, (int)ratio, g_oneshot_max4, g_dbg, g_per_lane4,
      g_rs.tab_dev, y.data_ptr<float>(), pos.data_ptr<int64_t>(), g_rs.seq);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void peer_ar_rows_set_lanes(int64_t l) { TORCH_CHECK(l >= 1 && l <= MAXLANES); g_lanes = (int)l; }
void peer_ar_rows_set_oneshot_max(int64_t v) { g_oneshot_max4 = (int)v; }
void peer_ar_rows_set_dbg(int64_t v) { g_dbg = (int)v; }
void peer_ar_rows_set_per_lane(int64_t v) { g_per_lane4 = (int)v; }
int64_t peer_ar_rows_numel() { return g_rs.ready ? g_rs.n : 0; }
void peer_ar_rows_close() { rows_free(); }

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("alloc", &peer_ar_rows_alloc);
  m.def("open", &peer_ar_rows_open);
  m.def("run", &peer_ar_rows_run);
  m.def("numel", &peer_ar_rows_numel);
  m.def("set_lanes", &peer_ar_rows_set_lanes);
  m.def("set_oneshot_max", &peer_ar_rows_set_oneshot_max);
  m.def("set_dbg", &peer_ar_rows_set_dbg);
  m.def("set_per_lane", &peer_ar_rows_set_per_lane);
  m.def("close", &peer_ar_rows_close);
}
