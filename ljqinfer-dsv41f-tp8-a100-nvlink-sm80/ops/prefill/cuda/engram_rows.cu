// Engram row dequantization: FP8 E4M3 values with per-32 E8M0 scales -> BF16.
// One block per gathered row; caller owns all buffers (no allocation here).
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp8.h>
#include <cuda_bf16.h>

namespace {

__global__ void engram_dequant_kernel(const __nv_fp8_storage_t* __restrict__ values,
                                      const unsigned char* __restrict__ scales,
                                      __nv_bfloat16* __restrict__ out,
                                      const long rows) {
    const long row = blockIdx.x;
    if (row >= rows) return;
    const int lane = threadIdx.x;
    const float scale = exp2f(static_cast<float>(scales[row * 8 + (lane >> 5)]) - 127.0f);
    const __half_raw raw = __nv_cvt_fp8_to_halfraw(values[row * 256 + lane], __NV_E4M3);
    const float value = __half2float(*reinterpret_cast<const __half*>(&raw));
    out[row * 256 + lane] = __float2bfloat16(value * scale);
}

}  // namespace

torch::Tensor engram_dequant(torch::Tensor values, torch::Tensor scales, torch::Tensor out) {
    TORCH_CHECK(values.is_cuda() && scales.is_cuda() && out.is_cuda(), "CUDA tensors required");
    TORCH_CHECK(values.is_contiguous() && scales.is_contiguous() && out.is_contiguous(),
                "contiguous tensors required");
    TORCH_CHECK(values.dim() == 2 && values.size(1) == 256, "values must be [rows,256]");
    TORCH_CHECK(scales.dim() == 2 && scales.size(1) == 8, "scales must be [rows,8]");
    TORCH_CHECK(out.sizes() == values.sizes(), "out must match values shape");
    TORCH_CHECK(values.scalar_type() == torch::kUInt8 && scales.scalar_type() == torch::kUInt8,
                "raw uint8 bytes required");
    TORCH_CHECK(out.scalar_type() == torch::kBFloat16, "bf16 output required");
    TORCH_CHECK(scales.size(0) == values.size(0), "row count mismatch");
    const long rows = values.size(0);
    if (rows == 0) return out;
    engram_dequant_kernel<<<rows, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __nv_fp8_storage_t*>(values.data_ptr<uint8_t>()),
        scales.data_ptr<uint8_t>(),
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()), rows);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("engram_dequant", &engram_dequant, "Engram FP8 row dequantization");
}
