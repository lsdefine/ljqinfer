#include <aclnn/acl_meta.h>
#include <ATen/ATen.h>
#include <torch/library.h>
#include "aclnn_torch_adapter/op_api_common.h"

namespace vllm_ascend {
thread_local char g_hashBuf[kHashBufSize];
thread_local int g_hashOffset = 0;

std::tuple<at::Tensor, at::Tensor, at::Tensor> chunk_h(
    const at::Tensor &k, const at::Tensor &w, const at::Tensor &u,
    const c10::optional<at::Tensor> &g,
    const c10::optional<at::Tensor> &initial_state,
    bool output_final_state, int64_t chunk_size,
    c10::optional<at::IntArrayRef> cu_seqlens,
    c10::optional<at::IntArrayRef> chunk_indices)
{
    const at::Tensor &g_ = c10::value_or_else(g, [] { return at::Tensor(); });
    const at::Tensor &initial_state_ = c10::value_or_else(initial_state, [] { return at::Tensor(); });
    at::Tensor empty;
    auto ks = k.sizes();
    auto us = u.sizes();
    int64_t b = ks[0], t = ks[2], key_dim = ks[3];
    int64_t value_heads = us[1], value_dim = us[3];
    int64_t nt = chunk_indices.has_value() ? chunk_indices->size() / 2 : (t + chunk_size - 1) / chunk_size;
    at::Tensor h = at::zeros({b, value_heads, nt, key_dim, value_dim}, k.options());
    at::Tensor v_new = at::zeros(u.sizes(), u.options());
    at::Tensor final_state;
    if (output_final_state) {
        int64_t n = cu_seqlens.has_value() ? cu_seqlens->size() - 1 : b;
        auto options = initial_state.has_value() ? initial_state->options() : h.options();
        final_state = at::empty({n, value_heads, key_dim, value_dim}, options);
    } else {
        final_state = at::empty({1}, k.options());
    }
    bool head_first = true;
    bool use_qmap = false;
    bool use_chunk_offsets = false;
    EXEC_NPU_CMD(aclnnChunkGatedDeltaRuleFwdH,
        k, w, u, g_, empty, initial_state_, output_final_state, chunk_size, head_first,
        cu_seqlens, chunk_indices, use_qmap, use_chunk_offsets, h, v_new, final_state);
    return std::make_tuple(h, v_new, output_final_state ? final_state : at::Tensor());
}

at::Tensor chunk_o(
    const at::Tensor &q, const at::Tensor &k, const at::Tensor &v,
    const at::Tensor &h, double scale,
    const c10::optional<at::Tensor> &g,
    c10::optional<at::IntArrayRef> cu_seqlens,
    c10::optional<at::IntArrayRef> chunk_indices,
    int64_t chunk_size)
{
    const at::Tensor &g_ = c10::value_or_else(g, [] { return at::Tensor(); });
    at::Tensor out = at::zeros(v.sizes(), v.options());
    EXEC_NPU_CMD(aclnnChunkFwdO,
        q, k, v, h, g_, cu_seqlens, chunk_indices, scale, chunk_size, out);
    return out;
}
}

TORCH_LIBRARY(ljq_chunk, m) {
    m.def("chunk_h(Tensor k, Tensor w, Tensor u, Tensor? g=None, Tensor? initial_state=None, bool output_final_state=True, int chunk_size=64, int[]? cu_seqlens=None, int[]? chunk_indices=None) -> (Tensor, Tensor, Tensor)");
    m.impl("chunk_h", c10::kPrivateUse1, &vllm_ascend::chunk_h);
    m.def("chunk_o(Tensor q, Tensor k, Tensor v, Tensor h, float scale, Tensor? g=None, int[]? cu_seqlens=None, int[]? chunk_indices=None, int chunk_size=64) -> Tensor");
    m.impl("chunk_o", c10::kPrivateUse1, &vllm_ascend::chunk_o);
}
