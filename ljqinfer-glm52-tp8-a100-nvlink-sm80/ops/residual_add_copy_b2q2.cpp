#include <torch/extension.h>

void residual_add_copy_f32_cuda(
    torch::Tensor x,
    torch::Tensor partial,
    torch::Tensor out);

void fused_add_rmsnorm_cuda(
    torch::Tensor x, torch::Tensor partial, torch::Tensor w, torch::Tensor h_out);

void combine_moe_f32_to_f16_cuda(
    torch::Tensor routed, torch::Tensor shared, torch::Tensor out);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &residual_add_copy_f32_cuda,
        "FP16 residual add plus exact FP32 copy");
  m.def("fused_add_rmsnorm", &fused_add_rmsnorm_cuda,
        "FP16 residual add + RMSNorm(w) fp16 out, one kernel");
  m.def("combine_moe_f32_to_f16", &combine_moe_f32_to_f16_cuda,
        "Round two FP32 MoE branches independently, add in FP16, write partial");
}
