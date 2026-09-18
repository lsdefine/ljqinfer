#include <torch/extension.h>

torch::Tensor moe_decode_iq_down_cache_out_cuda(
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor);
torch::Tensor dequant_iq4_selected_cuda(
    torch::Tensor, torch::Tensor, int64_t);
torch::Tensor materialize_q8_selected_cuda(
    torch::Tensor, torch::Tensor, int64_t);

torch::Tensor decode(
    torch::Tensor gate, torch::Tensor up, torch::Tensor down,
    torch::Tensor x, torch::Tensor expert_ids, torch::Tensor expert_weights,
    torch::Tensor hidden_workspace, torch::Tensor output_workspace,
    torch::Tensor down_cache, torch::Tensor expert_to_slot) {
  return moe_decode_iq_down_cache_out_cuda(
      gate, up, down, x, expert_ids, expert_weights, hidden_workspace,
      output_workspace, down_cache, expert_to_slot);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("dequant_selected", &materialize_q8_selected_cuda,
        "Materialize selected IQ4_XS rows as exact LUT int8 plus fp32 group scales");
  m.def("materialize_q8_selected", &materialize_q8_selected_cuda,
        "Materialize selected IQ4_XS rows as exact LUT int8 plus fp32 group scales");
  m.def("decode", &decode,
        "Routed IQ MoE decode with a persistent selected-expert exact-LUT Q8 Down cache");
}
