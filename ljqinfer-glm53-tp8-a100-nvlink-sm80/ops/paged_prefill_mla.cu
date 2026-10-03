// GLM53 prefill: fixed dense TC algorithm, no OOM fallback.
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <mutex>
#include <unordered_map>
#include <string>
#include <vector>
#include <limits>
#include <algorithm>
namespace { constexpr int L=512, R=64, D=576; }

static torch::Tensor paged_tc_mask(const torch::Device& dev) {
  constexpr int B = 1024;
  static std::mutex mu;
  static std::unordered_map<std::string, torch::Tensor> masks;
  const std::string key = std::to_string(dev.index()) + ":ppm_tc_1024";
  std::lock_guard<std::mutex> lock(mu);
  auto it = masks.find(key);
  if (it != masks.end()) return it->second;
  auto i = torch::arange(B, torch::TensorOptions().device(dev).dtype(torch::kInt64));
  return masks.emplace(key, i.unsqueeze(0) > i.unsqueeze(1)).first->second;
}

static torch::Tensor paged_tc_dense(
    const torch::Tensor& ql, const torch::Tensor& qr,
    const torch::Tensor& pool, const torch::Tensor& table,
    int64_t K0) {
  constexpr int64_t B = 1024;
  const int64_t Q = ql.size(0), H = ql.size(1), K = K0 + Q;
  const int64_t page_size = pool.size(1);
  const int64_t npages = (K + page_size - 1) / page_size;
  // Fixed algorithm; allocation failures propagate, never switch kernels.
  auto cache = pool.index_select(0, table.narrow(0, 0, npages))
                   .reshape({-1, 576}).narrow(0, 0, K);
  auto latent = cache.narrow(1, 0, 512);
  auto key = cache.transpose(0, 1);
  auto full_mask = paged_tc_mask(ql.device());
  std::vector<torch::Tensor> chunks;
  chunks.reserve((Q + B - 1) / B);
  for (int64_t q0 = 0; q0 < Q; q0 += B) {
    const int64_t n = std::min<int64_t>(B, Q - q0);
    const int64_t kend = K0 + q0 + n;
    auto query = torch::cat({ql.narrow(0, q0, n).transpose(0, 1),
                             qr.narrow(0, q0, n).transpose(0, 1)}, 2);
    query.mul_(0.0625);
    auto score = torch::matmul(query, key.narrow(1, 0, kend));
    score.narrow(2, kend - n, n).masked_fill_(
        full_mask.narrow(0, 0, n).narrow(1, 0, n),
        -std::numeric_limits<float>::infinity());
    at::softmax_out(score, score, -1);
    chunks.push_back(torch::matmul(score, latent.narrow(0, 0, kend))
                         .transpose(0, 1).contiguous());
  }
  return chunks.size() == 1 ? chunks[0] : torch::cat(chunks, 0);
}

torch::Tensor paged_prefill_mla(torch::Tensor q_latent, torch::Tensor q_rope,
                               torch::Tensor pool, torch::Tensor page_table,
                               int64_t q_start) {
  TORCH_CHECK(q_latent.is_cuda() && q_rope.is_cuda() && pool.is_cuda() && page_table.is_cuda(), "CUDA tensors required");
  TORCH_CHECK(q_latent.scalar_type() == at::kHalf && q_rope.scalar_type() == at::kHalf && pool.scalar_type() == at::kHalf, "fp16 required");
  TORCH_CHECK(q_latent.dim() == 3 && q_latent.size(2) == L, "q_latent must be [Q,H,512]");
  TORCH_CHECK(q_rope.dim() == 3 && q_rope.size(2) == R && q_rope.size(0) == q_latent.size(0) && q_rope.size(1) == q_latent.size(1), "q_rope must be [Q,H,64]");
  TORCH_CHECK(pool.dim() == 3 && pool.size(2) == D, "pool must be [pages,page_size,576]");
  TORCH_CHECK(page_table.dim() == 1 && page_table.scalar_type() == at::kLong, "page_table must be int64[logical_pages]");
  TORCH_CHECK(q_latent.is_contiguous() && q_rope.is_contiguous() && pool.is_contiguous() && page_table.is_contiguous(), "contiguous required");
  c10::cuda::CUDAGuard guard(q_latent.device());

  const int Q = (int)q_latent.size(0), H = (int)q_latent.size(1);
  const int page_size = (int)pool.size(1);
  TORCH_CHECK(Q > 0 && H > 0 && page_size > 0, "positive Q/H/page_size required");
  TORCH_CHECK(q_start >= 0, "q_start must be >= 0");
  TORCH_CHECK(q_start + Q <= page_table.size(0) * (int64_t)page_size, "page table too small for prefix");

  return paged_tc_dense(q_latent, q_rope, pool, page_table, q_start);
}

int64_t paged_prefill_mla_smem() { return 0; }
#ifndef LJQ_PPM_EMBED
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &paged_prefill_mla, "Fixed dense paged MLA prefill (no fallback)",
        pybind11::call_guard<pybind11::gil_scoped_release>());
  m.def("smem", &paged_prefill_mla_smem, "shared memory bytes per block");
}
#endif  // LJQ_PPM_EMBED
