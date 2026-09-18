// peer_ar_ipc.cu -- one-shot peer AllReduce over NVLink for a *multi-process* TP group.
//
// Lineage: ar_oneshot_f32_kernel from ljqinfer_dsv4f_tp8_broken/ops/peer_ar_os.cu.
// That version lived in a single process that owned all 8 devices, so it could
// just take the peer pointers straight out of the tensor list.  Here every rank
// is its own process, so the mailbox/flag buffers are raw cudaMalloc'd (a torch
// caching-allocator pointer is not the base of its IPC handle) and exchanged as
// cudaIpcMemHandle_t blobs through the existing torch.distributed group.
//
// Protocol per call (unchanged from the original):
//   1. every rank pushes its own shard of the input into slot `r` of *every*
//      peer's mailbox,
//   2. raises its flag on every peer,
//   3. spins on its own flag row until all R ranks have delivered,
//   4. sums the R slots locally and writes the result back in place.
// A per-lane sequence counter with a double-buffered mailbox makes back-to-back
// calls safe without any host-side synchronisation, which is what lets the whole
// thing be captured into the decode CUDA graph.

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <vector>

namespace {

constexpr int MAXR = 8;
constexpr int LANES = 8;   // one-shot grid
constexpr int LPR = 4;     // two-shot blocks per owned shard
constexpr int FLAG_B = LANES * MAXR;          // second flag bank base
constexpr int FLAG_A2 = FLAG_B + MAXR * LPR;   // two-shot shard-arrival bank
constexpr int FLAG_WORDS = FLAG_A2 + MAXR * LPR;
constexpr int SEQ_WORDS = LANES + MAXR * LPR;  // fixed grid, so the flag layout is static

struct ArTab {
  float* mail[MAXR];
  unsigned* flags[MAXR];
};

struct ArState {
  bool ready = false;
  int r = 0, R = 0;
  int64_t n = 0;
  int dev = -1;
  float* mail = nullptr;    // 2 * R * n floats, owned
  unsigned* flags = nullptr; // LANES * MAXR unsigned, owned
  unsigned* seq = nullptr;   // LANES unsigned, owned (never shared)
  void* opened[MAXR][2] = {};  // peer mappings to close on teardown
  ArTab* tab_dev = nullptr;
};

ArState g_st;

__global__ void __launch_bounds__(512)
ar_oneshot_f32_ipc_kernel(int r, int R, int64_t n,
                          const ArTab* __restrict__ tab,
                          float* __restrict__ mybuf,
                          unsigned* __restrict__ myseq) {
  const int lane = blockIdx.x, lanes = gridDim.x;
  const int64_t chunk = n / lanes, chunk_off = (int64_t)lane * chunk;
  __shared__ unsigned s_seq;
  if (threadIdx.x == 0) s_seq = atomicAdd(myseq + lane, 1u) + 1u;
  __syncthreads();
  const unsigned seq = s_seq;
  const int tid = threadIdx.x, nt = blockDim.x;
  const int64_t float_off = (int64_t)(seq & 1u) * (int64_t)R * n;  // double buffer
  const int64_t n4 = chunk >> 2;
  const int4* src = reinterpret_cast<const int4*>(mybuf + chunk_off);

  // 1. push my shard into slot r of every peer's mailbox
  for (int p = 0; p < R; ++p) {
    int4* dst = reinterpret_cast<int4*>(
        tab->mail[p] + float_off + (int64_t)r * n + chunk_off);
    for (int64_t i = tid; i < n4; i += nt) dst[i] = src[i];
  }
  __syncthreads();
  __threadfence_system();

  // 2. announce delivery on every peer
  if (tid < R) atomicExch_system(tab->flags[tid] + lane * MAXR + r, seq);

  // 3. wait for all R deliveries into my own row
  if (tid < R) {
    volatile unsigned* f = (volatile unsigned*)(tab->flags[r] + lane * MAXR);
    while (f[tid] < seq) { __nanosleep(64); }
  }
  __syncthreads();
  __threadfence();

  // 4. reduce locally, in place
  const float* m = tab->mail[r] + float_off + chunk_off;
  for (int64_t i = tid; i < chunk; i += nt) {
    float acc = 0.f;
    for (int j = 0; j < R; ++j) acc += m[(int64_t)j * n + i];
    mybuf[chunk_off + i] = acc;
  }
}


// Two-shot: every block owns one sub-shard of one rank's slice.
//   phase 1  push my copy of shard j into peer j's region A slot r
//   phase 2  the owner (j == r) sums the R copies in place
//   phase 3  the owner pushes the reduced sub-shard into region B of every peer
__global__ void __launch_bounds__(512)
ar_twoshot_f32_ipc_kernel(int r, int R, int64_t n,
                          const ArTab* __restrict__ tab,
                          float* __restrict__ mybuf,
                          unsigned* __restrict__ myseq) {
  const int b = blockIdx.x;
  const int j = b / LPR;            // rank owning this shard
  const int lane = b % LPR;
  const int64_t shard = n / R;
  const int64_t sub = shard / LPR;
  const int64_t off = (int64_t)j * shard + (int64_t)lane * sub;
  const int tid = threadIdx.x, nt = blockDim.x;
  __shared__ unsigned s_seq;
  if (tid == 0) s_seq = atomicAdd(myseq + LANES + b, 1u) + 1u;
  __syncthreads();
  const unsigned seq = s_seq;
  const int64_t par = (int64_t)(seq & 1u) * n;   // double buffer offset
  const int64_t base2 = 2LL * R * n;            // past the one-shot region
  const int64_t n4 = sub >> 2;

  {
    const int4* src = reinterpret_cast<const int4*>(mybuf + off);
    int4* dst = reinterpret_cast<int4*>(
        tab->mail[j] + base2 + par + (int64_t)r * shard + (int64_t)lane * sub);
    for (int64_t i = tid; i < n4; i += nt) dst[i] = src[i];
  }
  __syncthreads();
  __threadfence_system();
  if (tid == 0) atomicExch_system(tab->flags[j] + FLAG_A2 + lane * MAXR + r, seq);

  if (j == r) {
    if (tid < R) {
      volatile unsigned* f = (volatile unsigned*)(tab->flags[r] + FLAG_A2 + lane * MAXR);
      while (f[tid] < seq) { __nanosleep(64); }
    }
    __syncthreads();
    __threadfence();
    const float* m = tab->mail[r] + base2 + par + (int64_t)lane * sub;
    float* out = mybuf + off;
    for (int64_t i = tid; i < sub; i += nt) {
      float acc = 0.f;
      for (int q = 0; q < R; ++q) acc += m[(int64_t)q * shard + i];
      out[i] = acc;
    }
    __syncthreads();
    __threadfence();
    const int4* s2 = reinterpret_cast<const int4*>(out);
    for (int p = 0; p < R; ++p) {
      if (p == r) continue;
      int4* d2 = reinterpret_cast<int4*>(tab->mail[p] + base2 + 2 * n + par + off);
      for (int64_t i = tid; i < n4; i += nt) d2[i] = s2[i];
    }
    __syncthreads();
    __threadfence_system();
    if (tid < R) atomicExch_system(tab->flags[tid] + FLAG_B + (j * LPR + lane), seq);
  } else {
    if (tid == 0) {
      volatile unsigned* f =
          (volatile unsigned*)(tab->flags[r] + FLAG_B + (j * LPR + lane));
      while (*f < seq) { __nanosleep(64); }
    }
    __syncthreads();
    __threadfence();
    const int4* s2 = reinterpret_cast<const int4*>(tab->mail[r] + base2 + 2 * n + par + off);
    int4* d2 = reinterpret_cast<int4*>(mybuf + off);
    for (int64_t i = tid; i < n4; i += nt) d2[i] = s2[i];
  }
}

void free_state() {
  if (g_st.dev >= 0) cudaSetDevice(g_st.dev);
  for (int p = 0; p < MAXR; ++p)
    for (int k = 0; k < 2; ++k)
      if (g_st.opened[p][k]) { cudaIpcCloseMemHandle(g_st.opened[p][k]); g_st.opened[p][k] = nullptr; }
  if (g_st.tab_dev) { cudaFree(g_st.tab_dev); g_st.tab_dev = nullptr; }
  if (g_st.mail)  { cudaFree(g_st.mail);  g_st.mail = nullptr; }
  if (g_st.flags) { cudaFree(g_st.flags); g_st.flags = nullptr; }
  if (g_st.seq)   { cudaFree(g_st.seq);   g_st.seq = nullptr; }
  g_st.ready = false; g_st.n = 0; g_st.R = 0;
}

}  // namespace

// Allocate the local mailbox/flags and hand back the two IPC handles as bytes.
torch::Tensor peer_ar_ipc_alloc(int64_t rank, int64_t world, int64_t n) {
  TORCH_CHECK(world >= 2 && world <= MAXR, "peer_ar_ipc: world out of range");
  TORCH_CHECK(rank >= 0 && rank < world, "peer_ar_ipc: bad rank");
  TORCH_CHECK(n % (LANES * 4) == 0,
              "peer_ar_ipc: numel must be divisible by ", LANES * 4, ", got ", n);
  if (g_st.ready) free_state();

  int dev = -1;
  C10_CUDA_CHECK(cudaGetDevice(&dev));
  g_st.dev = dev; g_st.r = (int)rank; g_st.R = (int)world; g_st.n = n;

  const size_t mail_bytes = sizeof(float) * (2 * world * n + 4 * n);
  C10_CUDA_CHECK(cudaMalloc(&g_st.mail, mail_bytes));
  C10_CUDA_CHECK(cudaMemset(g_st.mail, 0, mail_bytes));
  C10_CUDA_CHECK(cudaMalloc(&g_st.flags, sizeof(unsigned) * FLAG_WORDS));
  C10_CUDA_CHECK(cudaMemset(g_st.flags, 0, sizeof(unsigned) * FLAG_WORDS));
  C10_CUDA_CHECK(cudaMalloc(&g_st.seq, sizeof(unsigned) * SEQ_WORDS));
  C10_CUDA_CHECK(cudaMemset(g_st.seq, 0, sizeof(unsigned) * SEQ_WORDS));

  cudaIpcMemHandle_t h[2];
  C10_CUDA_CHECK(cudaIpcGetMemHandle(&h[0], g_st.mail));
  C10_CUDA_CHECK(cudaIpcGetMemHandle(&h[1], g_st.flags));
  auto out = torch::empty({(int64_t)sizeof(h)}, torch::TensorOptions().dtype(torch::kUInt8));
  std::memcpy(out.data_ptr(), h, sizeof(h));
  return out;
}

// all_handles: [world, 2*sizeof(cudaIpcMemHandle_t)] uint8 CPU tensor, row r = rank r.
void peer_ar_ipc_open(torch::Tensor all_handles) {
  TORCH_CHECK(g_st.mail && g_st.flags, "peer_ar_ipc_open before alloc");
  const int R = g_st.R;
  TORCH_CHECK(all_handles.dim() == 2 && all_handles.size(0) == R &&
                  all_handles.size(1) == (int64_t)(2 * sizeof(cudaIpcMemHandle_t)) &&
                  all_handles.scalar_type() == torch::kUInt8 &&
                  all_handles.is_contiguous() && all_handles.device().is_cpu(),
              "peer_ar_ipc_open: bad handle blob");
  C10_CUDA_CHECK(cudaSetDevice(g_st.dev));

  ArTab tab_h{};
  const uint8_t* base = all_handles.data_ptr<uint8_t>();
  for (int p = 0; p < R; ++p) {
    if (p == g_st.r) {                     // never IPC-open your own allocation
      tab_h.mail[p] = g_st.mail;
      tab_h.flags[p] = g_st.flags;
      continue;
    }
    cudaIpcMemHandle_t h[2];
    std::memcpy(h, base + (size_t)p * 2 * sizeof(cudaIpcMemHandle_t), sizeof(h));
    void *pm = nullptr, *pf = nullptr;
    C10_CUDA_CHECK(cudaIpcOpenMemHandle(&pm, h[0], cudaIpcMemLazyEnablePeerAccess));
    C10_CUDA_CHECK(cudaIpcOpenMemHandle(&pf, h[1], cudaIpcMemLazyEnablePeerAccess));
    g_st.opened[p][0] = pm; g_st.opened[p][1] = pf;
    tab_h.mail[p] = (float*)pm;
    tab_h.flags[p] = (unsigned*)pf;
  }
  C10_CUDA_CHECK(cudaMalloc(&g_st.tab_dev, sizeof(ArTab)));
  C10_CUDA_CHECK(cudaMemcpy(g_st.tab_dev, &tab_h, sizeof(ArTab), cudaMemcpyHostToDevice));
  g_st.ready = true;
}


void peer_ar_ipc_run2(torch::Tensor y) {
  TORCH_CHECK(g_st.ready, "peer_ar_ipc: not registered");
  TORCH_CHECK(y.is_cuda() && y.scalar_type() == torch::kFloat32 && y.is_contiguous(),
              "peer_ar_ipc_run2 expects a contiguous fp32 cuda tensor");
  TORCH_CHECK(y.numel() == g_st.n, "peer_ar_ipc_run2: numel mismatch");
  auto st = at::cuda::getCurrentCUDAStream();
  ar_twoshot_f32_ipc_kernel<<<g_st.R * LPR, 512, 0, st>>>(
      g_st.r, g_st.R, g_st.n, g_st.tab_dev, y.data_ptr<float>(), g_st.seq);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

int64_t peer_ar_ipc_numel() { return g_st.ready ? g_st.n : 0; }

void peer_ar_ipc_close() { free_state(); }
