// Fused Engram gate: one block per (token, copy). Streams x and the reduced
// kv projection once, keeps norms/dot in fp32 registers, writes bf16 back.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>

typedef __nv_bfloat16 bf16;

__global__ void engram_gate_kernel(const bf16* __restrict__ x,
                                   const bf16* __restrict__ kv,
                                   const bf16* __restrict__ qw,
                                   const bf16* __restrict__ kw,
                                   bf16* __restrict__ out,
                                   int H, int D, float eps) {
  const int token = blockIdx.x / H;
  const int copy = blockIdx.x - token * H;
  const bf16* xp = x + ((size_t)token * H + copy) * D;
  const bf16* kp = kv + (size_t)token * (H + 1) * D + (size_t)copy * D;
  const bf16* vp = kv + (size_t)token * (H + 1) * D + (size_t)H * D;
  const bf16* qwp = qw + (size_t)copy * D;
  const bf16* kwp = kw + (size_t)copy * D;
  bf16* op = out + ((size_t)token * H + copy) * D;

  float sn = 0.f, kn = 0.f, dot = 0.f;
  for (int d = threadIdx.x; d < D; d += blockDim.x) {
    const float s = __bfloat162float(xp[d]);
    const float k = __bfloat162float(kp[d]);
    sn += s * s;
    kn += k * k;
    dot += s * k * __bfloat162float(qwp[d]) * __bfloat162float(kwp[d]);
  }
#pragma unroll
  for (int off = 16; off; off >>= 1) {
    sn += __shfl_down_sync(0xffffffff, sn, off);
    kn += __shfl_down_sync(0xffffffff, kn, off);
    dot += __shfl_down_sync(0xffffffff, dot, off);
  }
  __shared__ float red[3][32];
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  if (lane == 0) { red[0][warp] = sn; red[1][warp] = kn; red[2][warp] = dot; }
  __syncthreads();
  const int warps = blockDim.x >> 5;
  __shared__ float gate_s;
  if (threadIdx.x == 0) {
    float a = 0.f, b = 0.f, c = 0.f;
    for (int w = 0; w < warps; ++w) { a += red[0][w]; b += red[1][w]; c += red[2][w]; }
    const float inv = 1.f / (float)D;
    const float g = c * rsqrtf(a * inv + eps) * rsqrtf(b * inv + eps) * rsqrtf((float)D);
    float m = fabsf(g);
    m = fmaxf(m, 1e-6f);
    m = sqrtf(m);
    gate_s = 1.f / (1.f + __expf(-copysignf(m, g)));
  }
  __syncthreads();
  const float gate = gate_s;
  for (int d = threadIdx.x; d < D; d += blockDim.x)
    op[d] = __float2bfloat16(__bfloat162float(xp[d]) + gate * __bfloat162float(vp[d]));
}

torch::Tensor engram_gate_cuda(torch::Tensor x, torch::Tensor kv,
                               torch::Tensor q_weight, torch::Tensor k_weight,
                               double eps) {
  TORCH_CHECK(x.is_cuda() && x.is_contiguous() && kv.is_contiguous());
  TORCH_CHECK(x.scalar_type() == at::kBFloat16 && kv.scalar_type() == at::kBFloat16 &&
              q_weight.scalar_type() == at::kBFloat16 &&
              k_weight.scalar_type() == at::kBFloat16,
              "engram_gate is bf16-only; other dtypes would be reinterpreted");
  const int T = x.size(0), H = x.size(1), D = x.size(2);
  TORCH_CHECK(kv.size(-1) == (H + 1) * D);
  TORCH_CHECK(q_weight.numel() >= (int64_t)H * D && k_weight.numel() >= (int64_t)H * D,
              "engram_gate gate weights must hold H rows of D");
  auto out = torch::empty_like(x);
  const int threads = 512;
  engram_gate_kernel<<<T * H, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
      (const bf16*)x.data_ptr(), (const bf16*)kv.data_ptr(),
      (const bf16*)q_weight.data_ptr(), (const bf16*)k_weight.data_ptr(),
      (bf16*)out.data_ptr(), H, D, (float)eps);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

#ifndef DSV41_NO_PYBIND
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("engram_gate", &engram_gate_cuda, "fused engram gate",
        pybind11::call_guard<pybind11::gil_scoped_release>());
}
#endif
