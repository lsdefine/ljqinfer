#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>






__global__ void paged_scatter_positions_masked_kernel(
    __nv_bfloat16* __restrict__ pool,
    const int64_t* __restrict__ ptab,
    const __nv_bfloat16* __restrict__ src,
    const int64_t* __restrict__ positions,
    const int32_t* __restrict__ valid,
    int PAGE, int D, int Q, long table_stride) {
  const int i = blockIdx.x;
  if (!valid[i]) return;
  const int b = i / Q;
  const int64_t* table = ptab + (long)b * table_stride;
  const int64_t pos = positions[i];
  const long dst = ((long)table[pos / PAGE] * PAGE + (pos % PAGE)) * D;
  const long s = (long)i * D;
  for (int d = threadIdx.x; d < D; d += blockDim.x) pool[dst + d] = src[s + d];
}

void paged_scatter_positions_masked(torch::Tensor pool, torch::Tensor ptab,
                                    torch::Tensor positions, torch::Tensor valid,
                                    torch::Tensor src) {
  TORCH_CHECK(pool.is_cuda() && ptab.is_cuda() && positions.is_cuda() &&
              valid.is_cuda() && src.is_cuda(), "masked paged scatter inputs must be CUDA");
  TORCH_CHECK(pool.scalar_type() == torch::kBFloat16 &&
              src.scalar_type() == torch::kBFloat16,
              "masked paged scatter pool/src must be bf16");
  TORCH_CHECK(ptab.scalar_type() == torch::kInt64 &&
              positions.scalar_type() == torch::kInt64 &&
              valid.scalar_type() == torch::kInt32,
              "masked paged scatter metadata dtype mismatch");
  TORCH_CHECK(src.dim() == 2 && positions.numel() == src.size(0) &&
              valid.numel() == src.size(0) && src.size(1) == pool.size(2),
              "masked paged scatter shape mismatch");
  at::cuda::CUDAGuard guard(pool.device());
  auto s = src.contiguous(), pt = ptab.contiguous();
  auto pos = positions.contiguous(), mask = valid.contiguous();
  const int n = (int)s.size(0);
  const int B = pt.dim() == 2 ? (int)pt.size(0) : 1;
  TORCH_CHECK(n % B == 0, "masked paged scatter batch mismatch");
  const int Q = n / B;
  const long table_stride = pt.dim() == 2 ? pt.stride(0) : 0L;
  auto stream = at::cuda::getCurrentCUDAStream();
  paged_scatter_positions_masked_kernel<<<n, 256, 0, stream>>>(
      (__nv_bfloat16*)pool.data_ptr(), pt.data_ptr<int64_t>(),
      (const __nv_bfloat16*)s.data_ptr(), pos.data_ptr<int64_t>(),
      mask.data_ptr<int32_t>(), (int)pool.size(1), (int)pool.size(2),
      Q, table_stride);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

#ifndef DSV4_NO_PYBIND
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("paged_scatter_positions_masked", &paged_scatter_positions_masked,
        "graph-safe masked paged scatter with CUDA positions");
}
#endif
