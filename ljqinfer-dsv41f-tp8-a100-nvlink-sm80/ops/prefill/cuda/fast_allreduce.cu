// One-shot NVLink all-reduce for the decode-sized activations.
//
// Decode issues ~83 all-reduces per round, each only [6,5120] BF16 = 61KB.  At
// that size NCCL is pure latency (~35us on this box) because it still runs its
// ring in two phases.  With eight GPUs on a full NVLink mesh the cheap shape is
// one shot: every rank publishes its own vector into a peer-visible buffer,
// ranks handshake once, then every rank reads all eight copies and sums them
// locally.  Each rank moves 8x more bytes than a ring would, but 8 x 61KB over
// NVLink is far below the latency floor, so the trade is free.
//
// Buffers come from cudaMalloc + cudaIpcGetMemHandle (not the caching
// allocator, whose blocks are not IPC-exportable at arbitrary offsets).
// The handshake uses monotonic counters rather than flag resets, so the kernel
// is safe to capture in a CUDA graph and replay forever.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>

#define MAXB 64  // flag slots per source rank, one per block

struct Ptrs {
  unsigned* f[8];
  uint4* b[8];
};

__device__ __forceinline__ float2 bf2f(unsigned u) {
  unsigned lo = (u & 0xFFFFu) << 16, hi = u & 0xFFFF0000u;
  return make_float2(__int_as_float(lo), __int_as_float(hi));
}

__device__ __forceinline__ unsigned f2bf(float2 v) {
  unsigned a = __float_as_uint(v.x), b = __float_as_uint(v.y);
  a = (a + 0x7FFFu + ((a >> 16) & 1u)) >> 16;
  b = (b + 0x7FFFu + ((b >> 16) & 1u)) & 0xFFFF0000u;
  return (a & 0xFFFFu) | b;
}

__global__ void oneshot_ar(const uint4* __restrict__ in, uint4* __restrict__ out,
                           Ptrs p, int rank, int world, int nvec) {
  const int b = blockIdx.x, nb = gridDim.x;
  __shared__ unsigned sc;
  if (threadIdx.x == 0) sc = atomicAdd(&p.f[rank][8 * MAXB + b], 1u) + 1u;
  __syncthreads();
  const unsigned c = sc;

  uint4* mine = p.b[rank];
  const int stride = nb * blockDim.x;
  for (int i = b * blockDim.x + threadIdx.x; i < nvec; i += stride) mine[i] = in[i];
  __threadfence_system();

  // One thread per peer: a remote atomic costs ~2us of NVLink round trip, so
  // issuing all eight serially from one thread used to dominate the kernel.
  if (threadIdx.x < (unsigned)world) {
    const int q = threadIdx.x;
    atomicAdd_system(&p.f[q][rank * MAXB + b], 1u);
    volatile unsigned* fp = (volatile unsigned*)&p.f[rank][q * MAXB + b];
    while (*fp < c) __nanosleep(64);
  }
  __syncthreads();

  for (int i = b * blockDim.x + threadIdx.x; i < nvec; i += stride) {
    float acc[8];
    uint4 v = p.b[rank][i];  // rotate: rank0 buffer was an 8-way hot spot
    float2 a0 = bf2f(v.x), a1 = bf2f(v.y), a2 = bf2f(v.z), a3 = bf2f(v.w);
    acc[0] = a0.x; acc[1] = a0.y; acc[2] = a1.x; acc[3] = a1.y;
    acc[4] = a2.x; acc[5] = a2.y; acc[6] = a3.x; acc[7] = a3.y;
    for (int q = 1; q < world; ++q) {
      uint4 u = p.b[(rank + q) % world][i];
      float2 x0 = bf2f(u.x), x1 = bf2f(u.y), x2 = bf2f(u.z), x3 = bf2f(u.w);
      acc[0] += x0.x; acc[1] += x0.y; acc[2] += x1.x; acc[3] += x1.y;
      acc[4] += x2.x; acc[5] += x2.y; acc[6] += x3.x; acc[7] += x3.y;
    }
    uint4 o;
    o.x = f2bf(make_float2(acc[0], acc[1]));
    o.y = f2bf(make_float2(acc[2], acc[3]));
    o.z = f2bf(make_float2(acc[4], acc[5]));
    o.w = f2bf(make_float2(acc[6], acc[7]));
    out[i] = o;
  }
}

// ---- host side -------------------------------------------------------------

std::vector<int64_t> ipc_alloc(int64_t bytes) {
  void* ptr = nullptr;
  C10_CUDA_CHECK(cudaMalloc(&ptr, bytes));
  C10_CUDA_CHECK(cudaMemset(ptr, 0, bytes));
  cudaIpcMemHandle_t h;
  C10_CUDA_CHECK(cudaIpcGetMemHandle(&h, ptr));
  std::vector<int64_t> out;
  out.push_back((int64_t)ptr);
  const unsigned char* raw = (const unsigned char*)&h;
  for (size_t i = 0; i < sizeof(h); ++i) out.push_back((int64_t)raw[i]);
  return out;
}

int64_t ipc_open(std::vector<int64_t> blob) {
  TORCH_CHECK(blob.size() == sizeof(cudaIpcMemHandle_t), "bad handle blob");
  cudaIpcMemHandle_t h;
  unsigned char* raw = (unsigned char*)&h;
  for (size_t i = 0; i < blob.size(); ++i) raw[i] = (unsigned char)blob[i];
  void* ptr = nullptr;
  C10_CUDA_CHECK(cudaIpcOpenMemHandle(&ptr, h, cudaIpcMemLazyEnablePeerAccess));
  return (int64_t)ptr;
}

void all_reduce(torch::Tensor inp, torch::Tensor out, std::vector<int64_t> bases,
                int64_t flag_bytes, int64_t rank, int64_t blocks) {
  TORCH_CHECK(inp.scalar_type() == torch::kBFloat16, "bf16 only");
  TORCH_CHECK(inp.is_contiguous() && out.is_contiguous(), "contiguous only");
  const int world = (int)bases.size();
  const int64_t n = inp.numel();
  TORCH_CHECK(n % 8 == 0, "numel must be a multiple of eight");
  TORCH_CHECK(out.numel() == n, "shape mismatch");
  const int nvec = (int)(n / 8);
  Ptrs p;
  for (int q = 0; q < world; ++q) {
    p.f[q] = (unsigned*)bases[q];
    p.b[q] = (uint4*)((char*)bases[q] + flag_bytes);
  }
  const int nb = (int)blocks;
  TORCH_CHECK(nb <= MAXB, "too many blocks");
  oneshot_ar<<<nb, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      (const uint4*)inp.data_ptr(), (uint4*)out.data_ptr(), p, (int)rank, world, nvec);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("ipc_alloc", &ipc_alloc, "cudaMalloc plus an exportable IPC handle");
  m.def("ipc_open", &ipc_open, "map a peer's IPC handle");
  m.def("all_reduce", &all_reduce, "one-shot NVLink all-reduce");
}
