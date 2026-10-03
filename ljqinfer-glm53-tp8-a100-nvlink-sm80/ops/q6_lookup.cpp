#include <torch/extension.h>

torch::Tensor q6_lookup_cuda(torch::Tensor packed, torch::Tensor ids);
torch::Tensor q6_matvec_cuda(torch::Tensor packed, torch::Tensor x);
void q6_dequant_fp16_out_cuda(torch::Tensor packed, torch::Tensor out);
void q6_gemm_fp16_out_cuda(torch::Tensor w, torch::Tensor x, torch::Tensor out);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("lookup", &q6_lookup_cuda, "Q6_K row lookup/dequant (CUDA)");
    m.def("matvec", &q6_matvec_cuda, "Q6_K matrix-vector product (CUDA)");
    m.def("dequant_fp16_out", &q6_dequant_fp16_out_cuda,
          "Q6_K dequantize into caller-owned FP16 output (CUDA)");
    m.def("gemm_fp16_out", &q6_gemm_fp16_out_cuda,
          "FP16 x FP16 LM-head GEMM into caller-owned FP32 output (CUDA)");
}
