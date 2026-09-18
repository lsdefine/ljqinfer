#include <torch/extension.h>

torch::Tensor q8_cublas_forward_cuda(torch::Tensor x, torch::Tensor packed, int64_t in_features);
torch::Tensor q8_cublas_forward_out_cuda(torch::Tensor x, torch::Tensor packed, int64_t in_features, torch::Tensor y);
torch::Tensor q8_cublas_forward_grouped_cuda(torch::Tensor x, torch::Tensor packed, int64_t in_features);
void q8_cublas_clear_weight_cache();
void q8_cublas_pin_weight_cache();

static void check_common(torch::Tensor x, torch::Tensor packed, int64_t K) {
    TORCH_CHECK(x.is_cuda() && packed.is_cuda(), "x and packed must be CUDA");
    TORCH_CHECK(x.dtype() == torch::kFloat16, "x must be fp16");
    TORCH_CHECK(packed.dtype() == torch::kUInt8, "packed must be uint8");
    TORCH_CHECK(x.is_contiguous() && packed.is_contiguous(), "x and packed must be contiguous");
    TORCH_CHECK(K % 32 == 0, "K must be multiple of 32 (Q8_0 block)");
}

// forward(x[T,K], packed[N, K/32*34], K) -> [T,N]
torch::Tensor q8_cublas_forward(torch::Tensor x, torch::Tensor packed, int64_t in_features) {
    check_common(x, packed, in_features);
    TORCH_CHECK(x.dim() == 2 && packed.dim() == 2, "forward expects 2D x and 2D packed");
    TORCH_CHECK(x.size(1) == in_features, "x.size(1) must equal in_features");
    int64_t row_bytes = in_features / 32 * 34;
    TORCH_CHECK(packed.size(1) == row_bytes, "packed.size(1) must equal K/32*34");
    return q8_cublas_forward_cuda(x, packed, in_features);
}

torch::Tensor q8_cublas_forward_out(torch::Tensor x, torch::Tensor packed, int64_t in_features, torch::Tensor y) {
    check_common(x, packed, in_features);
    TORCH_CHECK(x.dim() == 2 && packed.dim() == 2 && y.dim() == 2, "forward_out expects 2D x, packed, y");
    TORCH_CHECK(x.size(1) == in_features && y.size(0) == x.size(0) && y.size(1) == packed.size(0), "forward_out shape mismatch");
    return q8_cublas_forward_out_cuda(x, packed, in_features, y);
}

// forward_grouped(x[T,H,K], packed[H,N, K/32*34], K) -> [T,H,N]

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &q8_cublas_forward, "cuBLAS Q8_0 linear (CUDA)", pybind11::call_guard<pybind11::gil_scoped_release>());
    m.def("forward_out", &q8_cublas_forward_out, "cuBLAS Q8_0 linear into output (CUDA)", pybind11::call_guard<pybind11::gil_scoped_release>());
    m.def("clear_weight_cache", &q8_cublas_clear_weight_cache, "Drop non-pinned dequantized Q8 weights");
    m.def("pin_weight_cache", &q8_cublas_pin_weight_cache, "Pin current Q8 weights for CUDA graph lifetime");
}
