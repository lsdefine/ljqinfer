// peer_ar_os.cu — one-shot symmetric peer allreduce for tiny fp16 messages (TP8 NVLink).
// Design: per-rank single-block kernel launched on that rank's current stream (graph-capturable).
//   rank r: push own [n] halfs into mail slot r on EVERY device -> fence -> set flag[r]=seq on every device
//           -> spin own flags[j]>=seq for all j -> sum R slots (fp32 acc, fixed order) -> write inplace to buf[r].
// Graph-safe: seq comes from per-device counter incremented inside kernel (replay-monotonic).
// Handles: multiple registrations (Q=1/Q=2 graphs) via handle id.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math_constants.h>
#include <vector>
#include <cstdint>

#define MAXR 8
#define MAXH 32

struct ARSet {
  bool used = false;
  int R = 0;
  int64_t n = 0;          // halfs per message
  int devs[MAXR];
  __half* buf[MAXR];      // user tensors (inplace in/out), stable addresses
  __half* mail[MAXR];     // per-device mailbox [R*n]
  unsigned* flags[MAXR];  // per-device [R]
  unsigned* seqc[MAXR];   // per-device [1]
  std::vector<torch::Tensor> holders;
};
static ARSet H[MAXH];
static int g_nh = 0;

// device-side pointer tables (one copy per device, indexed by handle)
struct DevTab { __half* mail[MAXR]; unsigned* flags[MAXR]; };
static DevTab* g_tab[MAXH][MAXR]; // tab uploaded to each device

__global__ void __launch_bounds__(512)
ar_oneshot_kernel(int r, int R, int64_t n,
                  const DevTab* __restrict__ tab,   // on this device
                  __half* __restrict__ mybuf,
                  unsigned* __restrict__ myseq) {
  const int lane = blockIdx.x, lanes = gridDim.x;
  const int64_t chunk = n / lanes, chunk_off = (int64_t)lane * chunk;
  __shared__ unsigned s_seq;
  if (threadIdx.x == 0) s_seq = atomicAdd(myseq + lane, 1u) + 1u;
  __syncthreads();
  const unsigned seq = s_seq;
  const int tid = threadIdx.x, nt = blockDim.x;
  const int64_t half_off = (int64_t)(seq & 1u) * (int64_t)R * n;
  // 1) push my payload into slot r of every device's mailbox (vector int4 = 8 halfs)
  const int64_t n8 = chunk >> 3;
  const int4* src = reinterpret_cast<const int4*>(mybuf + chunk_off);
  for (int p = 0; p < R; ++p) {
    int4* dst = reinterpret_cast<int4*>(tab->mail[p] + half_off + (int64_t)r * n + chunk_off);
    for (int64_t i = tid; i < n8; i += nt) dst[i] = src[i];
  }
  __syncthreads();
  __threadfence_system();
  // 2) publish flag[r]=seq on every device
  if (tid < R) {
    unsigned* f = tab->flags[tid] + lane * MAXR + r;
    atomicExch_system(f, seq);
  }
  // 3) spin on OWN device flags until all peers' payload landed (tab->flags[r] = flags array on my device)
  if (tid < R) {
    volatile unsigned* f = (volatile unsigned*)(tab->flags[r] + lane * MAXR);
    while (f[tid] < seq) { __nanosleep(64); }
  }
  __syncthreads();
  __threadfence(); // acquire: peer payload visible before local sum
  // 4) local sum fixed order, fp32 accumulate; write inplace to mybuf
  const __half* m = tab->mail[r] + half_off + chunk_off;
  const int64_t n2 = chunk >> 1;
  const __half2* m2 = reinterpret_cast<const __half2*>(m);
  __half2* out2 = reinterpret_cast<__half2*>(mybuf);
  for (int64_t i = tid; i < n2; i += nt) {
    float ax = 0.f, ay = 0.f;
    for (int j = 0; j < R; ++j) {
      const __half2 v = m2[(int64_t)j * (n >> 1) + i];
      ax += __half2float(__low2half(v));
      ay += __half2float(__high2half(v));
    }
    out2[(chunk_off >> 1) + i] = __floats2half2_rn(ax, ay);
  }
}


int64_t peer_ar_register(std::vector<torch::Tensor> ts) {
  int hid = -1;
  for (int i = 0; i < MAXH; ++i) if (!H[i].used) { hid = i; break; }
  TORCH_CHECK(hid >= 0, "too many live peer_ar handles");
  ARSet& S = H[hid];
  const int R = (int)ts.size();
  TORCH_CHECK(R >= 2 && R <= MAXR);
  const int64_t n = ts[0].numel();
  TORCH_CHECK(n % 8 == 0, "numel%8");
  // Validate the complete public contract before reserving a slot or allocating.
  for (int r = 0; r < R; ++r) {
    auto& t = ts[r];
    TORCH_CHECK(t.is_cuda() && t.is_contiguous() &&
                t.scalar_type() == torch::kHalf && t.numel() == n);
  }
  S.used = true; S.R = R; S.n = n;
  try {
    DevTab tab_h{};
    for (int r = 0; r < R; ++r) {
      auto& t = ts[r];
      S.devs[r] = t.get_device();
      S.buf[r] = reinterpret_cast<__half*>(t.data_ptr());
      auto opts_h = torch::TensorOptions().dtype(torch::kHalf).device(t.device());
      auto opts_u = torch::TensorOptions().dtype(torch::kInt32).device(t.device());
      auto mail = torch::zeros({(int64_t)2 * R * n}, opts_h);
      auto flg  = torch::zeros({2 * 8 * MAXR}, opts_u);
      auto sq   = torch::zeros({8}, opts_u);
      S.holders.push_back(mail); S.holders.push_back(flg); S.holders.push_back(sq);
      S.mail[r]  = reinterpret_cast<__half*>(mail.data_ptr());
      S.flags[r] = reinterpret_cast<unsigned*>(flg.data_ptr());
      S.seqc[r]  = reinterpret_cast<unsigned*>(sq.data_ptr());
      tab_h.mail[r] = S.mail[r]; tab_h.flags[r] = S.flags[r];
    }
    // Enable p2p + upload dev tables.
    for (int i = 0; i < R; ++i) {
      cudaSetDevice(S.devs[i]);
      for (int j = 0; j < R; ++j) if (i != j) {
        cudaError_t e = cudaDeviceEnablePeerAccess(S.devs[j], 0);
        TORCH_CHECK(e == cudaSuccess || e == cudaErrorPeerAccessAlreadyEnabled,
                    "p2p enable fail: ", cudaGetErrorString(e));
        if (e == cudaErrorPeerAccessAlreadyEnabled) cudaGetLastError();
      }
      cudaError_t e = cudaMalloc(&g_tab[hid][i], sizeof(DevTab));
      TORCH_CHECK(e == cudaSuccess, "peer_ar table allocation failed: ", cudaGetErrorString(e));
      e = cudaMemcpy(g_tab[hid][i], &tab_h, sizeof(DevTab), cudaMemcpyHostToDevice);
      TORCH_CHECK(e == cudaSuccess, "peer_ar table upload failed: ", cudaGetErrorString(e));
    }
  } catch (...) {
    // Registration is transactional: no failed attempt may consume a slot or
    // retain CUDA allocations in the resident process.
    for (int r = 0; r < R; ++r) {
      if (g_tab[hid][r]) {
        cudaSetDevice(S.devs[r]);
        cudaFree(g_tab[hid][r]);
        g_tab[hid][r] = nullptr;
      }
    }
    S.holders.clear();
    S.R = 0; S.n = 0; S.used = false;
    throw;
  }
  g_nh++;
  return hid;
}

void peer_ar_unregister(int64_t hid) {
  TORCH_CHECK(hid >= 0 && hid < MAXH && H[hid].used, "invalid peer_ar handle");
  ARSet& S = H[hid];
  // Caller must reset every graph using this handle and synchronize all ranks.
  for (int r = 0; r < S.R; ++r) {
    cudaSetDevice(S.devs[r]);
    if (g_tab[hid][r]) { cudaFree(g_tab[hid][r]); g_tab[hid][r] = nullptr; }
  }
  S.holders.clear();
  S.R = 0; S.n = 0; S.used = false;
  --g_nh;
}

// launch for rank r on CURRENT device/stream (caller sets torch.cuda.device+stream) — graph-capturable
void peer_ar_run(int64_t hid, int64_t r) {
  TORCH_CHECK(hid >= 0 && hid < MAXH && H[hid].used, "invalid peer_ar handle");
  ARSet& S = H[hid];
  cudaSetDevice(S.devs[r]);
  cudaStream_t st = at::cuda::getCurrentCUDAStream(S.devs[r]).stream();
  int blocks = 8;
  while (blocks > 1 && (S.n % blocks || (S.n / blocks) % 8)) blocks >>= 1;
  TORCH_CHECK(S.n % blocks == 0 && (S.n / blocks) % 8 == 0,
              "unsupported peer AR shape");
  ar_oneshot_kernel<<<blocks, 512, 0, st>>>(
      (int)r, S.R, S.n, g_tab[hid][r], S.buf[r], S.seqc[r]);
}

// Graph-safe TP8 metadata broadcast.  Rank 0 computes routing once; each
// rank-local graph calls peer_meta_bcast_run on its own stream.  A replay-
// monotonic device counter + peer-visible flags avoids cross-capture events.
struct MBTab {
  int64_t* ei[MAXR];
  float* ew[MAXR];
  unsigned* flags[MAXR];
};
struct MBSet {
  bool used = false;
  int R = 0;
  int64_t n = 0;
  int devs[MAXR];
  int64_t* ei[MAXR];
  float* ew[MAXR];
  unsigned* flags[MAXR];
  unsigned* seqc[MAXR];
  std::vector<torch::Tensor> holders;
};
static MBSet MB[MAXH];
static MBTab* g_mbtab[MAXH][MAXR];

__global__ void meta_bcast_kernel(int r, int R, int64_t n,
                                  const MBTab* __restrict__ tab,
                                  const int64_t* __restrict__ my_ei,
                                  const float* __restrict__ my_ew,
                                  unsigned* __restrict__ myseq) {
  __shared__ unsigned s_seq;
  if (threadIdx.x == 0) s_seq = atomicAdd(myseq, 1u) + 1u;
  __syncthreads();
  const unsigned seq = s_seq;
  const int tid = threadIdx.x;
  if (r == 0) {
    for (int p = 1; p < R; ++p) {
      for (int64_t i = tid; i < n; i += blockDim.x) {
        tab->ei[p][i] = my_ei[i];
        tab->ew[p][i] = my_ew[i];
      }
    }
    __syncthreads();
    __threadfence_system();
    if (tid < R) atomicExch_system(tab->flags[tid], seq);
  } else {
    if (tid == 0) {
      volatile unsigned* f = (volatile unsigned*)tab->flags[r];
      while (*f < seq) { __nanosleep(64); }
      __threadfence_system();
    }
  }
}

int64_t peer_meta_bcast_register(std::vector<torch::Tensor> eis,
                                 std::vector<torch::Tensor> ews) {
  int hid = -1;
  for (int i = 0; i < MAXH; ++i) if (!MB[i].used) { hid = i; break; }
  TORCH_CHECK(hid >= 0, "too many live peer metadata handles");
  const int R = (int)eis.size();
  TORCH_CHECK(R >= 2 && R <= MAXR && (int)ews.size() == R,
              "metadata register requires matching rank vectors");
  const int64_t n = eis[0].numel();
  for (int r = 0; r < R; ++r) {
    TORCH_CHECK(eis[r].is_cuda() && eis[r].is_contiguous() &&
                eis[r].scalar_type() == torch::kInt64 && eis[r].numel() == n,
                "ei must be contiguous CUDA int64 with equal shape");
    TORCH_CHECK(ews[r].is_cuda() && ews[r].is_contiguous() &&
                ews[r].scalar_type() == torch::kFloat32 && ews[r].numel() == n,
                "ew must be contiguous CUDA float32 with equal shape");
    TORCH_CHECK(eis[r].get_device() == ews[r].get_device(),
                "ei/ew device mismatch");
  }
  MBSet& S = MB[hid];
  S.used = true; S.R = R; S.n = n;
  try {
    MBTab tab_h{};
    for (int r = 0; r < R; ++r) {
      S.devs[r] = eis[r].get_device();
      S.ei[r] = eis[r].data_ptr<int64_t>();
      S.ew[r] = ews[r].data_ptr<float>();
      auto opts = torch::TensorOptions().dtype(torch::kInt32).device(eis[r].device());
      auto flg = torch::zeros({1}, opts);
      auto seq = torch::zeros({1}, opts);
      S.holders.push_back(flg); S.holders.push_back(seq);
      S.flags[r] = reinterpret_cast<unsigned*>(flg.data_ptr());
      S.seqc[r] = reinterpret_cast<unsigned*>(seq.data_ptr());
      tab_h.ei[r] = S.ei[r]; tab_h.ew[r] = S.ew[r]; tab_h.flags[r] = S.flags[r];
    }
    for (int i = 0; i < R; ++i) {
      cudaSetDevice(S.devs[i]);
      for (int j = 0; j < R; ++j) if (i != j) {
        cudaError_t e = cudaDeviceEnablePeerAccess(S.devs[j], 0);
        TORCH_CHECK(e == cudaSuccess || e == cudaErrorPeerAccessAlreadyEnabled,
                    "metadata p2p enable fail: ", cudaGetErrorString(e));
        if (e == cudaErrorPeerAccessAlreadyEnabled) cudaGetLastError();
      }
      cudaError_t e = cudaMalloc(&g_mbtab[hid][i], sizeof(MBTab));
      TORCH_CHECK(e == cudaSuccess, "metadata table allocation failed: ", cudaGetErrorString(e));
      e = cudaMemcpy(g_mbtab[hid][i], &tab_h, sizeof(MBTab), cudaMemcpyHostToDevice);
      TORCH_CHECK(e == cudaSuccess, "metadata table upload failed: ", cudaGetErrorString(e));
    }
  } catch (...) {
    for (int r = 0; r < R; ++r) if (g_mbtab[hid][r]) {
      cudaSetDevice(S.devs[r]); cudaFree(g_mbtab[hid][r]); g_mbtab[hid][r] = nullptr;
    }
    S.holders.clear(); S.R = 0; S.n = 0; S.used = false;
    throw;
  }
  return hid;
}


void peer_meta_bcast_run(int64_t hid, int64_t r) {
  TORCH_CHECK(hid >= 0 && hid < MAXH && MB[hid].used, "invalid peer metadata handle");
  MBSet& S = MB[hid];
  TORCH_CHECK(r >= 0 && r < S.R, "invalid metadata rank");
  cudaSetDevice(S.devs[r]);
  cudaStream_t st = at::cuda::getCurrentCUDAStream(S.devs[r]).stream();
  meta_bcast_kernel<<<1, 128, 0, st>>>((int)r, S.R, S.n, g_mbtab[hid][r],
                                      S.ei[r], S.ew[r], S.seqc[r]);
}


// Graph-safe TP8 all-gather for replicated MLA latent projections.
struct LatentTab {
  __half* q[MAXR];
  __half* kv[MAXR];
  unsigned* flags[MAXR];
};
struct LatentSet {
  bool used = false;
  int R = 0, T = 0;
  int devs[MAXR];
  __half* q_local[MAXR];
  __half* kv_local[MAXR];
  __half* q_full[MAXR];
  __half* kv_full[MAXR];
  unsigned* flags[MAXR];
  unsigned* seq[MAXR];
  std::vector<torch::Tensor> holders;
};
static LatentSet LG[MAXH];
static LatentTab* g_lgtab[MAXH][MAXR]{};

__global__ void latent_gather_kernel(
    int r, int R, int T, const __half* __restrict__ q_local,
    const __half* __restrict__ kv_local, const LatentTab* __restrict__ tab,
    unsigned* __restrict__ seq) {
  __shared__ unsigned s_seq;
  if (threadIdx.x == 0) s_seq = atomicAdd(seq, 1u) + 1u;
  __syncthreads();
  const unsigned z = s_seq;
  // q shard = 256 halves = 32 int4; kv shard = 72 halves = 9 int4.
  const int qv = T * 32, kvv = T * 9, nv = qv + kvv;
  const int4* q4 = reinterpret_cast<const int4*>(q_local);
  const int4* kv4 = reinterpret_cast<const int4*>(kv_local);
  for (int p = 0; p < R; ++p) {
    for (int i = threadIdx.x; i < nv; i += blockDim.x) {
      if (i < qv) {
        const int t = i / 32, j = i - t * 32;
        reinterpret_cast<int4*>(tab->q[p] + t * 2048 + r * 256)[j] = q4[i];
      } else {
        const int a = i - qv, t = a / 9, j = a - t * 9;
        reinterpret_cast<int4*>(tab->kv[p] + t * 576 + r * 72)[j] = kv4[a];
      }
    }
  }
  __syncthreads();
  __threadfence_system();
  if (threadIdx.x < R) atomicExch_system(tab->flags[threadIdx.x] + r, z);
  if (threadIdx.x < R) {
    volatile unsigned* f = tab->flags[r];
    while (f[threadIdx.x] < z) __nanosleep(64);
  }
  __syncthreads();
  __threadfence();
}

int64_t latent_gather_register(
    std::vector<torch::Tensor> q_local,
    std::vector<torch::Tensor> kv_local,
    std::vector<torch::Tensor> q_full,
    std::vector<torch::Tensor> kv_full) {
  const int R = (int)q_local.size();
  TORCH_CHECK(R == MAXR && (int)kv_local.size() == R &&
              (int)q_full.size() == R && (int)kv_full.size() == R,
              "latent gather requires TP8 tensors");
  int hid = -1;
  for (int i = 0; i < MAXH; ++i) if (!LG[i].used) { hid = i; break; }
  TORCH_CHECK(hid >= 0, "too many live latent gather handles");
  LatentSet& S = LG[hid];
  S.used = true;
  S.R = R;
  S.T = (int)q_full[0].size(0);
  try {
    LatentTab host{};
    for (int r = 0; r < R; ++r) {
      auto valid = [](const torch::Tensor& t) {
        return t.is_cuda() && t.is_contiguous() &&
               t.scalar_type() == torch::kHalf;
      };
      TORCH_CHECK(valid(q_local[r]) &&
                  q_local[r].sizes() == torch::IntArrayRef({S.T, 256}),
                  "invalid local q latent");
      TORCH_CHECK(valid(kv_local[r]) &&
                  kv_local[r].sizes() == torch::IntArrayRef({S.T, 72}),
                  "invalid local kv latent");
      TORCH_CHECK(valid(q_full[r]) &&
                  q_full[r].sizes() == torch::IntArrayRef({S.T, 2048}),
                  "invalid full q latent");
      TORCH_CHECK(valid(kv_full[r]) &&
                  kv_full[r].sizes() == torch::IntArrayRef({S.T, 576}),
                  "invalid full kv latent");
      const int dev = q_full[r].get_device();
      TORCH_CHECK(q_local[r].get_device() == dev &&
                  kv_local[r].get_device() == dev &&
                  kv_full[r].get_device() == dev,
                  "latent gather device mismatch");
      S.devs[r] = dev;
      S.q_local[r] = reinterpret_cast<__half*>(q_local[r].data_ptr());
      S.kv_local[r] = reinterpret_cast<__half*>(kv_local[r].data_ptr());
      S.q_full[r] = reinterpret_cast<__half*>(q_full[r].data_ptr());
      S.kv_full[r] = reinterpret_cast<__half*>(kv_full[r].data_ptr());
      cudaSetDevice(dev);
      auto opts = torch::TensorOptions().dtype(torch::kInt32).device(q_full[r].device());
      auto flags = torch::zeros({R}, opts);
      auto seq = torch::zeros({1}, opts);
      S.holders.push_back(flags);
      S.holders.push_back(seq);
      S.flags[r] = reinterpret_cast<unsigned*>(flags.data_ptr());
      S.seq[r] = reinterpret_cast<unsigned*>(seq.data_ptr());
      host.q[r] = S.q_full[r];
      host.kv[r] = S.kv_full[r];
      host.flags[r] = S.flags[r];
    }
    for (int i = 0; i < R; ++i) {
      cudaSetDevice(S.devs[i]);
      for (int j = 0; j < R; ++j) if (i != j) {
        cudaError_t e = cudaDeviceEnablePeerAccess(S.devs[j], 0);
        TORCH_CHECK(e == cudaSuccess || e == cudaErrorPeerAccessAlreadyEnabled,
                    "latent gather p2p enable failed: ", cudaGetErrorString(e));
        if (e == cudaErrorPeerAccessAlreadyEnabled) cudaGetLastError();
      }
      C10_CUDA_CHECK(cudaMalloc(&g_lgtab[hid][i], sizeof(LatentTab)));
      C10_CUDA_CHECK(cudaMemcpy(g_lgtab[hid][i], &host, sizeof(LatentTab),
                                cudaMemcpyHostToDevice));
    }
  } catch (...) {
    for (int r = 0; r < R; ++r) if (g_lgtab[hid][r]) {
      cudaSetDevice(S.devs[r]);
      cudaFree(g_lgtab[hid][r]);
      g_lgtab[hid][r] = nullptr;
    }
    S.holders.clear();
    S.R = 0;
    S.T = 0;
    S.used = false;
    throw;
  }
  return hid;
}

void latent_gather_run(int64_t hid, int64_t r) {
  TORCH_CHECK(hid >= 0 && hid < MAXH && LG[hid].used,
              "invalid latent gather handle");
  LatentSet& S = LG[hid];
  TORCH_CHECK(r >= 0 && r < S.R, "invalid latent gather rank");
  cudaSetDevice(S.devs[r]);
  cudaStream_t st = at::cuda::getCurrentCUDAStream(S.devs[r]).stream();
  latent_gather_kernel<<<1, 256, 0, st>>>(
      (int)r, S.R, S.T, S.q_local[r], S.kv_local[r],
      g_lgtab[hid][r], S.seq[r]);
}

void latent_gather_unregister(int64_t hid) {
  TORCH_CHECK(hid >= 0 && hid < MAXH && LG[hid].used,
              "invalid latent gather handle");
  LatentSet& S = LG[hid];
  for (int r = 0; r < S.R; ++r) {
    cudaSetDevice(S.devs[r]);
    if (g_lgtab[hid][r]) {
      cudaFree(g_lgtab[hid][r]);
      g_lgtab[hid][r] = nullptr;
    }
  }
  S.holders.clear();
  S.R = 0;
  S.T = 0;
  S.used = false;
}

struct GatherTab {
  uint16_t* full[MAXR];
  unsigned* flags[MAXR];
};
struct GatherSet {
  bool used = false;
  int R = 0;
  int64_t n = 0;
  int devs[MAXR];
  const uint16_t* local[MAXR];
  uint16_t* full[MAXR];
  unsigned* flags[MAXR];
  unsigned* seq[MAXR];
  std::vector<torch::Tensor> holders;
};
static GatherSet G[MAXH];
static GatherTab* g_tabs[MAXH][MAXR]{};

__global__ void peer_gather_bf16_kernel(
    int r, int R, int64_t n, const uint16_t* __restrict__ local,
    const GatherTab* __restrict__ tab, unsigned* __restrict__ seq) {
  __shared__ unsigned s_seq;
  if (threadIdx.x == 0) s_seq = atomicAdd(seq, 1u) + 1u;
  __syncthreads();
  const unsigned z = s_seq;
  const int64_t n8 = n >> 3;
  const int4* src = reinterpret_cast<const int4*>(local);
  for (int p = 0; p < R; ++p) {
    int4* dst = reinterpret_cast<int4*>(tab->full[p] + (int64_t)r * n);
    for (int64_t i = threadIdx.x; i < n8; i += blockDim.x) dst[i] = src[i];
  }
  __syncthreads();
  __threadfence_system();
  if (threadIdx.x < R) atomicExch_system(tab->flags[threadIdx.x] + r, z);
  if (threadIdx.x < R) {
    volatile unsigned* f = tab->flags[r];
    while (f[threadIdx.x] < z) __nanosleep(64);
  }
  __syncthreads();
  __threadfence();
}

int64_t peer_gather_bf16_register(std::vector<torch::Tensor> local,
                                  std::vector<torch::Tensor> full) {
  const int R = (int)local.size();
  TORCH_CHECK(R == MAXR && (int)full.size() == R, "peer gather requires TP8");
  int hid = -1;
  for (int i = 0; i < MAXH; ++i) if (!G[i].used) { hid = i; break; }
  TORCH_CHECK(hid >= 0, "too many live peer gather handles");
  GatherSet& S = G[hid];
  const int64_t n = local[0].numel();
  TORCH_CHECK(n > 0 && n % 8 == 0, "local numel must be positive and divisible by 8");
  for (int r = 0; r < R; ++r) {
    TORCH_CHECK(local[r].is_cuda() && local[r].is_contiguous() &&
                local[r].scalar_type() == torch::kBFloat16 && local[r].numel() == n,
                "invalid local bf16 shard");
    TORCH_CHECK(full[r].is_cuda() && full[r].is_contiguous() &&
                full[r].scalar_type() == torch::kBFloat16 && full[r].numel() == R * n,
                "invalid full bf16 gather buffer");
    TORCH_CHECK(local[r].get_device() == full[r].get_device(), "rank device mismatch");
  }
  S.used = true; S.R = R; S.n = n;
  try {
    GatherTab host{};
    for (int r = 0; r < R; ++r) {
      S.devs[r] = local[r].get_device();
      S.local[r] = reinterpret_cast<const uint16_t*>(local[r].data_ptr());
      S.full[r] = reinterpret_cast<uint16_t*>(full[r].data_ptr());
      cudaSetDevice(S.devs[r]);
      auto opts = torch::TensorOptions().dtype(torch::kInt32).device(local[r].device());
      auto flags = torch::zeros({R}, opts);
      auto seq = torch::zeros({1}, opts);
      S.holders.push_back(flags); S.holders.push_back(seq);
      S.flags[r] = reinterpret_cast<unsigned*>(flags.data_ptr());
      S.seq[r] = reinterpret_cast<unsigned*>(seq.data_ptr());
      host.full[r] = S.full[r]; host.flags[r] = S.flags[r];
    }
    for (int i = 0; i < R; ++i) {
      cudaSetDevice(S.devs[i]);
      for (int j = 0; j < R; ++j) if (i != j) {
        cudaError_t e = cudaDeviceEnablePeerAccess(S.devs[j], 0);
        TORCH_CHECK(e == cudaSuccess || e == cudaErrorPeerAccessAlreadyEnabled,
                    "P2P enable failed: ", cudaGetErrorString(e));
        if (e == cudaErrorPeerAccessAlreadyEnabled) cudaGetLastError();
      }
      C10_CUDA_CHECK(cudaMalloc(&g_tabs[hid][i], sizeof(GatherTab)));
      C10_CUDA_CHECK(cudaMemcpy(g_tabs[hid][i], &host, sizeof(GatherTab), cudaMemcpyHostToDevice));
    }
  } catch (...) {
    for (int r = 0; r < R; ++r) if (g_tabs[hid][r]) {
      cudaSetDevice(S.devs[r]); cudaFree(g_tabs[hid][r]); g_tabs[hid][r] = nullptr;
    }
    S.holders.clear(); S.R = 0; S.n = 0; S.used = false;
    throw;
  }
  return hid;
}

void peer_gather_bf16_run(int64_t hid, int64_t r) {
  TORCH_CHECK(hid >= 0 && hid < MAXH && G[hid].used, "invalid peer gather handle");
  GatherSet& S = G[hid];
  TORCH_CHECK(r >= 0 && r < S.R, "invalid rank");
  cudaSetDevice(S.devs[r]);
  cudaStream_t st = at::cuda::getCurrentCUDAStream(S.devs[r]).stream();
  peer_gather_bf16_kernel<<<1, 256, 0, st>>>(
      (int)r, S.R, S.n, S.local[r], g_tabs[hid][r], S.seq[r]);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void peer_gather_bf16_unregister(int64_t hid) {
  TORCH_CHECK(hid >= 0 && hid < MAXH && G[hid].used, "invalid peer gather handle");
  GatherSet& S = G[hid];
  for (int r = 0; r < S.R; ++r) {
    cudaSetDevice(S.devs[r]);
    if (g_tabs[hid][r]) { cudaFree(g_tabs[hid][r]); g_tabs[hid][r] = nullptr; }
  }
  S.holders.clear(); S.R = 0; S.n = 0; S.used = false;
}

struct BcastTab { uint16_t* dst[MAXR]; unsigned* flags[MAXR]; };
struct BcastSet {
  bool used = false; int R = 0, root = 0; int64_t n = 0; int devs[MAXR];
  uint16_t* dst[MAXR]; unsigned* flags[MAXR]; unsigned* seq[MAXR];
  std::vector<torch::Tensor> holders;
};
static BcastSet B[MAXH];
static BcastTab* b_tabs[MAXH][MAXR]{};

__global__ void peer_bcast_fp16_kernel(int r, int root, int R, int64_t n,
                                       BcastTab* tab, uint16_t* src,
                                       unsigned* seq) {
  __shared__ unsigned z;
  if (threadIdx.x == 0) z = atomicAdd(seq, 1u) + 1u;
  __syncthreads();
  if (r == root) {
    const int64_t n8 = n >> 3;
    const int4* s = reinterpret_cast<const int4*>(src);
    for (int p = 0; p < R; ++p) if (p != root) {
      int4* d = reinterpret_cast<int4*>(tab->dst[p]);
      for (int64_t i = threadIdx.x; i < n8; i += blockDim.x) d[i] = s[i];
    }
    __syncthreads(); __threadfence_system();
    if (threadIdx.x < R) atomicExch_system(tab->flags[threadIdx.x], z);
  } else if (threadIdx.x == 0) {
    volatile unsigned* f = tab->flags[r];
    while (*f < z) __nanosleep(64);
  }
  __syncthreads(); __threadfence();
}

int64_t peer_bcast_fp16_register(std::vector<torch::Tensor> dst, int64_t root) {
  const int R = (int)dst.size();
  TORCH_CHECK(R == MAXR && root >= 0 && root < R, "fp16 bcast requires TP8 and valid root");
  int hid = -1; for (int i = 0; i < MAXH; ++i) if (!B[i].used) { hid = i; break; }
  TORCH_CHECK(hid >= 0, "too many live fp16 bcast handles");
  BcastSet& S = B[hid]; const int64_t n = dst[0].numel();
  TORCH_CHECK(n > 0 && n % 8 == 0, "bcast numel must be divisible by 8");
  S.used = true; S.R = R; S.root = (int)root; S.n = n;
  BcastTab host{};
  for (int r = 0; r < R; ++r) {
    TORCH_CHECK(dst[r].is_cuda() && dst[r].is_contiguous() &&
                dst[r].scalar_type() == torch::kFloat16 && dst[r].numel() == n,
                "invalid fp16 bcast tensor");
    S.devs[r] = dst[r].get_device(); S.dst[r] = reinterpret_cast<uint16_t*>(dst[r].data_ptr());
    cudaSetDevice(S.devs[r]); auto o = torch::TensorOptions().dtype(torch::kInt32).device(dst[r].device());
    auto f = torch::zeros({1}, o); auto q = torch::zeros({1}, o);
    S.holders.push_back(f); S.holders.push_back(q);
    S.flags[r] = reinterpret_cast<unsigned*>(f.data_ptr()); S.seq[r] = reinterpret_cast<unsigned*>(q.data_ptr());
    host.dst[r] = S.dst[r]; host.flags[r] = S.flags[r];
  }
  for (int r = 0; r < R; ++r) {
    cudaSetDevice(S.devs[r]);
    for (int j = 0; j < R; ++j) if (r != j) {
      cudaError_t e = cudaDeviceEnablePeerAccess(S.devs[j], 0);
      TORCH_CHECK(e == cudaSuccess || e == cudaErrorPeerAccessAlreadyEnabled,
                  "bcast P2P enable failed: ", cudaGetErrorString(e));
      if (e == cudaErrorPeerAccessAlreadyEnabled) cudaGetLastError();
    }
    C10_CUDA_CHECK(cudaMalloc(&b_tabs[hid][r], sizeof(BcastTab)));
    C10_CUDA_CHECK(cudaMemcpy(b_tabs[hid][r], &host, sizeof(BcastTab), cudaMemcpyHostToDevice));
  }
  return hid;
}

void peer_bcast_fp16_run(int64_t hid, int64_t r) {
  TORCH_CHECK(hid >= 0 && hid < MAXH && B[hid].used, "invalid bcast handle");
  BcastSet& S = B[hid]; TORCH_CHECK(r >= 0 && r < S.R, "invalid rank");
  cudaSetDevice(S.devs[r]); auto st = at::cuda::getCurrentCUDAStream(S.devs[r]).stream();
  peer_bcast_fp16_kernel<<<1, 256, 0, st>>>((int)r, S.root, S.R, S.n,
      b_tabs[hid][r], S.dst[S.root], S.seq[r]);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void peer_bcast_fp16_unregister(int64_t hid) {
  TORCH_CHECK(hid >= 0 && hid < MAXH && B[hid].used, "invalid bcast handle");
  BcastSet& S = B[hid];
  for (int r = 0; r < S.R; ++r) { cudaSetDevice(S.devs[r]); cudaFree(b_tabs[hid][r]); b_tabs[hid][r] = nullptr; }
  S.holders.clear(); S.used = false; S.R = 0; S.n = 0;
}



// ---------------- vocab-parallel fused argmax (MTP lm_tail) ----------------
struct AMTab {
  const float* part[MAXR];   // dev r: fp32 partial, row b at b*Tstride
  __half*   mail[MAXR];      // dev r: [2 * R * B * Vs] halves (ring of 2)
  unsigned* flags[MAXR];     // dev r: [MAXR] slice-arrival seq
  __half*   cand_v[MAXR];    // dev r: [R*B] fp16 (only root's is read)
  int*      cand_i[MAXR];    // dev r: [R*B] int32 global idx
  unsigned* cflag[MAXR];     // dev r: [MAXR] candidate-arrival seq
  int*      tok[MAXR];       // dev r: [B] int32 final token out
  unsigned* tflag[MAXR];     // dev r: [1] token-ready seq
};
struct AMSet {
  bool used = false; int R = 0, B = 0; int64_t V = 0, Vs = 0, Tstride = 0;
  int devs[MAXR]{};
  unsigned* seqc[MAXR]{};    // dev r: [MAXR] per-block seq counters
  std::vector<torch::Tensor> holders;
};
static AMSet AM[MAXH];
static AMTab* am_tabs[MAXH][MAXR]{};

__global__ void __launch_bounds__(512)
peer_argmax_kernel(int r, int R, int B, int64_t Vs, int64_t Tstride,
                   AMTab* tab, unsigned* seqc) {
  const int p = blockIdx.x;            // this block pushes slice p to rank p
  const int tid = threadIdx.x, nt = blockDim.x;
  __shared__ unsigned s_seq;
  if (tid == 0) s_seq = atomicAdd(seqc + p, 1u) + 1u;
  __syncthreads();
  const unsigned seq = s_seq;
  const int64_t half_off = (int64_t)(seq & 1u) * (int64_t)R * B * Vs;
  // 1) cast my fp32 slice p to fp16, push into rank p's mail slot r
  {
    const float* src = tab->part[r];
    __half* dst = tab->mail[p] + half_off + (int64_t)r * B * Vs;
    for (int b = 0; b < B; ++b) {
      const float* s = src + (int64_t)b * Tstride + (int64_t)p * Vs;
      __half* d = dst + (int64_t)b * Vs;
      // clamp to fp16 finite range: fp32 partials can exceed 65504 and cast
      // to +/-inf, whose cross-rank sum becomes NaN and poisons the argmax.
      for (int64_t i = tid; i < Vs; i += nt)
        d[i] = __float2half(fminf(fmaxf(s[i], -65504.f), 65504.f));
    }
  }
  __syncthreads();
  __threadfence_system();
  if (tid == 0) atomicExch_system(tab->flags[p] + r, seq);
  if (p != r) return;                  // only the resident block continues
  // 2) wait until all R slices landed on my device
  if (tid < R) {
    volatile unsigned* f = (volatile unsigned*)tab->flags[r];
    while (f[tid] < seq) { __nanosleep(64); }
  }
  __syncthreads();
  __threadfence();
  // 3) per row: fixed-order fp32 sum -> fp16 round -> block argmax
  __shared__ float sv[512];
  __shared__ int   si[512];
  const __half* m = tab->mail[r] + half_off;
  for (int b = 0; b < B; ++b) {
    float bestv = -CUDART_INF_F; int besti = 0x7fffffff;
    for (int64_t i = tid; i < Vs; i += nt) {
      float acc = 0.f;
      for (int j = 0; j < R; ++j)
        acc += __half2float(m[((int64_t)j * B + b) * Vs + i]);
      const float v0 = __half2float(__float2half(acc));
      const float v = isfinite(v0) ? v0 : -CUDART_INF_F;  // NaN-safe
      if (v > bestv || (v == bestv && (int)i < besti)) { bestv = v; besti = (int)i; }
    }
    sv[tid] = bestv; si[tid] = besti;
    __syncthreads();
    for (int s = nt >> 1; s > 0; s >>= 1) {
      if (tid < s && (sv[tid + s] > sv[tid] ||
          (sv[tid + s] == sv[tid] && si[tid + s] < si[tid]))) {
        sv[tid] = sv[tid + s]; si[tid] = si[tid + s];
      }
      __syncthreads();
    }
    if (tid == 0) {
      tab->cand_v[0][(int64_t)r * B + b] = __float2half(sv[0]);
      tab->cand_i[0][(int64_t)r * B + b] = (int)((int64_t)r * Vs + si[0]);
    }
    __syncthreads();
  }
  __threadfence_system();
  if (tid == 0) atomicExch_system(tab->cflag[0] + r, seq);
  if (r == 0) {
    // 4) root: wait all candidates, final select, push tokens everywhere
    if (tid < R) {
      volatile unsigned* f = (volatile unsigned*)tab->cflag[0];
      while (f[tid] < seq) { __nanosleep(64); }
    }
    __syncthreads();
    __threadfence();
    if (tid == 0) {
      for (int b = 0; b < B; ++b) {
        float bv = -CUDART_INF_F; int bi = 0x7fffffff;
        for (int j = 0; j < R; ++j) {
          const float v0 = __half2float(tab->cand_v[0][(int64_t)j * B + b]);
          const float v = isfinite(v0) ? v0 : -CUDART_INF_F;  // NaN-safe
          const int   i = tab->cand_i[0][(int64_t)j * B + b];
          if (v > bv || (v == bv && i < bi)) { bv = v; bi = i; }
        }
        for (int q = 0; q < R; ++q) tab->tok[q][b] = bi;
      }
      __threadfence_system();
      for (int q = 0; q < R; ++q) atomicExch_system(tab->tflag[q], seq);
    }
    __syncthreads();
  } else {
    // 5) non-root: wait for final token from root
    if (tid == 0) {
      volatile unsigned* f = (volatile unsigned*)tab->tflag[r];
      while (*f < seq) { __nanosleep(64); }
    }
    __syncthreads();
    __threadfence();
  }
}

int64_t peer_argmax_register(std::vector<torch::Tensor> parts,
                             std::vector<torch::Tensor> toks, int64_t B) {
  const int R = (int)parts.size();
  TORCH_CHECK(R == MAXR && (int)toks.size() == R, "peer_argmax requires TP8");
  int hid = -1; for (int i = 0; i < MAXH; ++i) if (!AM[i].used) { hid = i; break; }
  TORCH_CHECK(hid >= 0, "too many live peer_argmax handles");
  AMSet& S = AM[hid];
  const int64_t V = parts[0].size(-1);
  TORCH_CHECK(V % R == 0, "V must divide R");
  const int64_t Vs = V / R;
  S.used = true; S.R = R; S.B = (int)B; S.V = V; S.Vs = Vs;
  S.Tstride = parts[0].stride(0);
  AMTab host{};
  for (int r = 0; r < R; ++r) {
    auto& t = parts[r];
    TORCH_CHECK(t.is_cuda() && t.scalar_type() == torch::kFloat &&
                t.dim() == 2 && t.size(1) == V && t.size(0) >= B &&
                t.stride(1) == 1 && t.stride(0) == S.Tstride, "bad part");
    auto& k = toks[r];
    TORCH_CHECK(k.is_cuda() && k.is_contiguous() &&
                k.scalar_type() == torch::kInt && k.numel() >= B &&
                k.get_device() == t.get_device(), "bad tok");
    S.devs[r] = t.get_device();
    host.part[r] = t.data_ptr<float>();
    host.tok[r] = k.data_ptr<int>();
    auto opts_h = torch::TensorOptions().dtype(torch::kHalf).device(t.device());
    auto opts_u = torch::TensorOptions().dtype(torch::kInt32).device(t.device());
    auto mail = torch::zeros({(int64_t)2 * R * B * Vs}, opts_h);
    auto flg  = torch::zeros({MAXR}, opts_u);
    auto cv   = torch::zeros({(int64_t)R * B}, opts_h);
    auto ci   = torch::zeros({(int64_t)R * B}, opts_u);
    auto cf   = torch::zeros({MAXR}, opts_u);
    auto tf   = torch::zeros({1}, opts_u);
    auto sq   = torch::zeros({MAXR}, opts_u);
    for (auto& h : {mail, flg, cv, ci, cf, tf, sq}) S.holders.push_back(h);
    host.mail[r]   = reinterpret_cast<__half*>(mail.data_ptr());
    host.flags[r]  = reinterpret_cast<unsigned*>(flg.data_ptr());
    host.cand_v[r] = reinterpret_cast<__half*>(cv.data_ptr());
    host.cand_i[r] = ci.data_ptr<int>();
    host.cflag[r]  = reinterpret_cast<unsigned*>(cf.data_ptr());
    host.tflag[r]  = reinterpret_cast<unsigned*>(tf.data_ptr());
    S.seqc[r]      = reinterpret_cast<unsigned*>(sq.data_ptr());
  }
  for (int i = 0; i < R; ++i) {
    cudaSetDevice(S.devs[i]);
    for (int j = 0; j < R; ++j) if (i != j) {
      cudaError_t e = cudaDeviceEnablePeerAccess(S.devs[j], 0);
      TORCH_CHECK(e == cudaSuccess || e == cudaErrorPeerAccessAlreadyEnabled,
                  "p2p enable fail: ", cudaGetErrorString(e));
      if (e == cudaErrorPeerAccessAlreadyEnabled) cudaGetLastError();
    }
    cudaError_t e = cudaMalloc(&am_tabs[hid][i], sizeof(AMTab));
    TORCH_CHECK(e == cudaSuccess, "peer_argmax tab alloc: ", cudaGetErrorString(e));
    e = cudaMemcpy(am_tabs[hid][i], &host, sizeof(AMTab), cudaMemcpyHostToDevice);
    TORCH_CHECK(e == cudaSuccess, "peer_argmax tab copy: ", cudaGetErrorString(e));
  }
  return hid;
}

void peer_argmax_run(int64_t hid, int64_t r) {
  AMSet& S = AM[hid];
  cudaStream_t st = at::cuda::getCurrentCUDAStream(S.devs[r]).stream();
  peer_argmax_kernel<<<S.R, 512, 0, st>>>((int)r, S.R, S.B, S.Vs, S.Tstride,
                                          am_tabs[hid][r], S.seqc[r]);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void peer_argmax_unregister(int64_t hid) {
  TORCH_CHECK(hid >= 0 && hid < MAXH && AM[hid].used, "invalid argmax handle");
  AMSet& S = AM[hid];
  for (int r = 0; r < S.R; ++r) {
    cudaSetDevice(S.devs[r]); cudaFree(am_tabs[hid][r]); am_tabs[hid][r] = nullptr;
  }
  S.holders.clear(); S.used = false; S.R = 0; S.B = 0;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("peer_ar_register", &peer_ar_register);
  m.def("peer_ar_unregister", &peer_ar_unregister);
  m.def("peer_ar_run", &peer_ar_run);
  m.def("latent_gather_register", &latent_gather_register);
  m.def("latent_gather_run", &latent_gather_run);
  m.def("latent_gather_unregister", &latent_gather_unregister);
  m.def("peer_meta_bcast_register", &peer_meta_bcast_register);
  m.def("peer_meta_bcast_run", &peer_meta_bcast_run);
  m.def("peer_gather_bf16_register", &peer_gather_bf16_register);
  m.def("peer_gather_bf16_run", &peer_gather_bf16_run);
  m.def("peer_gather_bf16_unregister", &peer_gather_bf16_unregister);
  m.def("peer_bcast_fp16_register", &peer_bcast_fp16_register);
  m.def("peer_argmax_register", &peer_argmax_register);
  m.def("peer_argmax_run", &peer_argmax_run);
  m.def("peer_argmax_unregister", &peer_argmax_unregister);
  m.def("peer_bcast_fp16_run", &peer_bcast_fp16_run);
  m.def("peer_bcast_fp16_unregister", &peer_bcast_fp16_unregister);
}
