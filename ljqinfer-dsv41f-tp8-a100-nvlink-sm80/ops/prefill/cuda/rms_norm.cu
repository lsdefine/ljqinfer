// Fused RMSNorm for prefill: one block per token row, FP32 accumulation.
// Matches ops/prefill/residual.rms: y = x * rsqrt(mean(x^2) + eps) * weight,
// computed in FP32 and rounded once on store. Caller owns the output buffer.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>

namespace {

template <typename T>
__device__ inline float to_float(T v);
template <>
__device__ inline float to_float<float>(float v) { return v; }
template <>
__device__ inline float to_float<__nv_bfloat16>(__nv_bfloat16 v) { return __bfloat162float(v); }

template <typename T>
__device__ inline T from_float(float v);
template <>
__device__ inline float from_float<float>(float v) { return v; }
template <>
__device__ inline __nv_bfloat16 from_float<__nv_bfloat16>(float v) { return __float2bfloat16(v); }

template <typename T, typename W>
__global__ void rms_norm_kernel(const T* __restrict__ x, const W* __restrict__ weight,
                                T* __restrict__ out, const int width, const float eps,
                                const int has_weight) {
    const long row = blockIdx.x;
    const T* src = x + row * (long)width;
    T* dst = out + row * (long)width;

    float sum = 0.f;
    for (int i = threadIdx.x; i < width; i += blockDim.x) {
        const float v = to_float<T>(src[i]);
        sum += v * v;
    }
    for (int off = warpSize / 2; off > 0; off >>= 1) sum += __shfl_down_sync(0xffffffffu, sum, off);

    __shared__ float partial[32];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    if (lane == 0) partial[warp] = sum;
    __syncthreads();
    if (warp == 0) {
        const int warps = (blockDim.x + 31) >> 5;
        float v = lane < warps ? partial[lane] : 0.f;
        for (int off = warpSize / 2; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffffu, v, off);
        if (lane == 0) partial[0] = rsqrtf(v / (float)width + eps);
    }
    __syncthreads();
    const float rstd = partial[0];

    for (int i = threadIdx.x; i < width; i += blockDim.x) {
        float v = to_float<T>(src[i]) * rstd;
        if (has_weight) v *= to_float<W>(weight[i]);
        dst[i] = from_float<T>(v);
    }
}

}  // namespace

torch::Tensor rms_norm(torch::Tensor x, c10::optional<torch::Tensor> weight,
                       torch::Tensor out, double eps) {
    TORCH_CHECK(x.is_cuda() && out.is_cuda(), "CUDA tensors required");
    TORCH_CHECK(x.is_contiguous() && out.is_contiguous(), "contiguous tensors required");
    TORCH_CHECK(x.sizes() == out.sizes() && x.scalar_type() == out.scalar_type(),
                "out must match x shape and dtype");
    const int width = x.size(-1);
    const long rows = x.numel() / width;
    TORCH_CHECK(rows > 0 && width > 0, "empty input");

    const void* wptr = nullptr;
    bool weight_bf16 = false;
    if (weight.has_value()) {
        auto w = weight.value();
        TORCH_CHECK(w.is_cuda() && w.is_contiguous() && w.numel() == width, "weight must be [width]");
        TORCH_CHECK(w.scalar_type() == torch::kFloat32 || w.scalar_type() == torch::kBFloat16,
                    "weight must be fp32 or bf16");
        weight_bf16 = w.scalar_type() == torch::kBFloat16;
        wptr = w.data_ptr();
    }
    const int threads = width >= 1024 ? 1024 : 256;
    auto stream = at::cuda::getCurrentCUDAStream();

#define LAUNCH(T, W)                                                                    \
    rms_norm_kernel<T, W><<<rows, threads, 0, stream>>>(                                 \
        reinterpret_cast<const T*>(x.data_ptr()), reinterpret_cast<const W*>(wptr),      \
        reinterpret_cast<T*>(out.data_ptr()), width, (float)eps, wptr != nullptr ? 1 : 0)

    if (x.scalar_type() == torch::kFloat32) {
        if (weight_bf16) { LAUNCH(float, __nv_bfloat16); } else { LAUNCH(float, float); }
    } else if (x.scalar_type() == torch::kBFloat16) {
        if (weight_bf16) { LAUNCH(__nv_bfloat16, __nv_bfloat16); } else { LAUNCH(__nv_bfloat16, float); }
    } else {
        TORCH_CHECK(false, "x must be fp32 or bf16");
    }
#undef LAUNCH
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rms_norm", &rms_norm, "Fused RMSNorm (prefill)");
}
