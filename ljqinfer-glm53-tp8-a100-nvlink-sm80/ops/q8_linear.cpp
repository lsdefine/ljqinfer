#include <torch/extension.h>

torch::Tensor q8_linear_cuda(torch::Tensor x, torch::Tensor packed, int64_t in_features);
torch::Tensor q8_linear_grouped_cuda(torch::Tensor x, torch::Tensor packed, int64_t in_features);

static void check_common(torch::Tensor x, torch::Tensor packed, int64_t K) {
    TORCH_CHECK(x.is_cuda() && packed.is_cuda(), "x and packed must be CUDA");
    TORCH_CHECK(x.scalar_type() == at::kHalf, "x must be float16");
    TORCH_CHECK(packed.scalar_type() == at::kByte, "packed must be uint8");
    TORCH_CHECK(x.is_contiguous() && packed.is_contiguous(), "inputs must be contiguous");
    TORCH_CHECK(K > 0 && K % 32 == 0, "K must be divisible by 32");
    TORCH_CHECK(x.size(-1) == K, "x K mismatch");
    TORCH_CHECK(packed.size(-1) == K / 32 * 34, "Q8_0 row_bytes mismatch");
    TORCH_CHECK(x.device() == packed.device(), "device mismatch");
}

torch::Tensor q8_linear(torch::Tensor x, torch::Tensor packed, int64_t K) {
    check_common(x, packed, K);
    TORCH_CHECK(x.dim() == 2 && packed.dim() == 2, "expected x[T,K], packed[N,row_bytes]");
    return q8_linear_cuda(x, packed, K);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &q8_linear, "packed GGML Q8_0 linear (CUDA)");
}
