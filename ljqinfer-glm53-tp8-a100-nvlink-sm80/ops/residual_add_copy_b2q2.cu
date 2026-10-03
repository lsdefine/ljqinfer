#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

__global__ void residual_add_copy_f32_kernel(
    __half* __restrict__ x,
    const __half* __restrict__ partial,
    float* __restrict__ out,
    int64_t n) {
  for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
       i < n;
       i += (int64_t)blockDim.x * gridDim.x) {
    const __half h = __hadd_rn(x[i], partial[i]);
    x[i] = h;
    out[i] = __half2float(h);
  }
}

void residual_add_copy_f32_cuda(
    torch::Tensor x,
    torch::Tensor partial,
    torch::Tensor out) {
  TORCH_CHECK(x.is_cuda() && partial.is_cuda() && out.is_cuda());
  TORCH_CHECK(x.scalar_type() == torch::kFloat16);
  TORCH_CHECK(partial.scalar_type() == torch::kFloat16);
  TORCH_CHECK(out.scalar_type() == torch::kFloat32);
  TORCH_CHECK(x.is_contiguous() && partial.is_contiguous() && out.is_contiguous());
  TORCH_CHECK(x.sizes() == partial.sizes() && x.sizes() == out.sizes());

  c10::cuda::CUDAGuard guard(x.device());
  const int64_t n = x.numel();
  const int blocks = static_cast<int>((n + 255) / 256);
  const auto stream = at::cuda::getCurrentCUDAStream(x.device().index());
  residual_add_copy_f32_kernel<<<blocks, 256, 0, stream>>>(
      reinterpret_cast<__half*>(x.data_ptr<at::Half>()),
      reinterpret_cast<const __half*>(partial.data_ptr<at::Half>()),
      out.data_ptr<float>(), n);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}


// ---- fused: xs += partial (fp16); h_norm = rmsnorm(xs)*w (fp16 out) ----
// One block per row. D known small (7168). fp32 accumulation for mean(x^2).
__global__ void fused_add_rmsnorm_kernel(
    __half* __restrict__ x,
    const __half* __restrict__ partial,
    const __half* __restrict__ w,
    __half* __restrict__ h_out,
    int d, float eps) {
  extern __shared__ float sh[];          // d floats: committed fp32 row
  const int64_t row = blockIdx.x;
  __half* xr = x + row * d;
  const __half* pr = partial + row * d;
  __half* hr = h_out + row * d;
  float acc = 0.f;
  for (int i = threadIdx.x; i < d; i += blockDim.x) {
    const __half h = __hadd_rn(xr[i], pr[i]);
    xr[i] = h;
    const float f = __half2float(h);
    sh[i] = f;
    acc += f * f;
  }
  // block reduce acc
  __shared__ float red[32];
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, o);
  if (lane == 0) red[warp] = acc;
  __syncthreads();
  if (warp == 0) {
    acc = (lane < (blockDim.x >> 5)) ? red[lane] : 0.f;
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, o);
    if (lane == 0) red[0] = acc;
  }
  __syncthreads();
  const float inv = rsqrtf(red[0] / (float)d + eps);
  for (int i = threadIdx.x; i < d; i += blockDim.x) {
    const float y = sh[i] * inv * __half2float(w[i]);
    hr[i] = __float2half_rn(y);
  }
}

void fused_add_rmsnorm_cuda(
    torch::Tensor x, torch::Tensor partial, torch::Tensor w, torch::Tensor h_out) {
  TORCH_CHECK(x.is_cuda() && partial.is_cuda() && w.is_cuda() && h_out.is_cuda());
  TORCH_CHECK(x.scalar_type() == torch::kFloat16 && partial.scalar_type() == torch::kFloat16);
  TORCH_CHECK(w.scalar_type() == torch::kFloat16 && h_out.scalar_type() == torch::kFloat16);
  TORCH_CHECK(x.is_contiguous() && partial.is_contiguous() && w.is_contiguous() && h_out.is_contiguous());
  TORCH_CHECK(x.sizes() == partial.sizes() && x.numel() == h_out.numel());
  const int64_t d = x.size(-1);
  TORCH_CHECK(w.numel() == d);
  const int64_t rows = x.numel() / d;
  c10::cuda::CUDAGuard guard(x.device());
  const auto stream = at::cuda::getCurrentCUDAStream(x.device().index());
  const size_t shmem = (size_t)d * sizeof(float);
  fused_add_rmsnorm_kernel<<<(int)rows, 512, shmem, stream>>>(
      reinterpret_cast<__half*>(x.data_ptr<at::Half>()),
      reinterpret_cast<const __half*>(partial.data_ptr<at::Half>()),
      reinterpret_cast<const __half*>(w.data_ptr<at::Half>()),
      reinterpret_cast<__half*>(h_out.data_ptr<at::Half>()),
      (int)d, 1e-5f);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}


// Preserve the exact old sequence:
// out.copy_(routed_fp32); shared_half = shared_fp32.to(fp16); out.add_(shared_half).
__global__ void combine_moe_f32_to_f16_kernel(
    const float* __restrict__ routed,
    const float* __restrict__ shared,
    __half* __restrict__ out,
    int64_t n) {
  for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
       i < n; i += (int64_t)blockDim.x * gridDim.x) {
    const __half r = __float2half_rn(routed[i]);
    const __half s = __float2half_rn(shared[i]);
    out[i] = __hadd_rn(r, s);
  }
}

void combine_moe_f32_to_f16_cuda(
    torch::Tensor routed, torch::Tensor shared, torch::Tensor out) {
  TORCH_CHECK(routed.is_cuda() && shared.is_cuda() && out.is_cuda());
  TORCH_CHECK(routed.scalar_type() == torch::kFloat32 &&
              shared.scalar_type() == torch::kFloat32 &&
              out.scalar_type() == torch::kFloat16);
  TORCH_CHECK(routed.is_contiguous() && shared.is_contiguous() && out.is_contiguous());
  TORCH_CHECK(routed.sizes() == shared.sizes() && routed.sizes() == out.sizes());
  TORCH_CHECK(routed.device() == shared.device() && routed.device() == out.device());
  c10::cuda::CUDAGuard guard(routed.device());
  const int64_t n = routed.numel();
  const int blocks = static_cast<int>((n + 255) / 256);
  const auto stream = at::cuda::getCurrentCUDAStream(routed.device().index());
  combine_moe_f32_to_f16_kernel<<<blocks, 256, 0, stream>>>(
      routed.data_ptr<float>(), shared.data_ptr<float>(),
      reinterpret_cast<__half*>(out.data_ptr<at::Half>()), n);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
