"""CUDA FA2 calls over the existing BSND/paged model layouts.

Masks retain the model contract (True means excluded). There is no dense
attention fallback. Plans are prepared only in eager prefill, never in decode.
"""
from __future__ import annotations
import torch
import flashinfer


def attention(q, k, v, *, mask=None, causal=False, scale=None):
    """BSND GQA; fixed B is unrolled while each row uses fused FA2 attention."""
    rows = []
    for row in range(q.shape[0]):
        allowed = None
        if mask is not None:
            m = mask[0 if mask.shape[0] == 1 else row]
            allowed = ~m.reshape(q.shape[1], k.shape[1])
        rows.append(flashinfer.single_prefill_with_kv_cache(
            q[row], k[row], v[row], custom_mask=allowed, causal=causal,
            kv_layout="NHD", sm_scale=scale, backend="fa2"))
    return torch.stack(rows, dim=0)


def paged_attention(q, cache, layer_slot, sequence_id, total_tokens):
    """Run directly on the original physical page pool, without gathering KV."""
    page_size = cache.spec.page_size
    pages = (int(total_tokens) + page_size - 1) // page_size
    indices = cache.host_page_table[sequence_id][:pages]
    if any(i < 0 for i in indices):
        raise RuntimeError("prefill attention references an unallocated KV page")
    signature = (sequence_id, int(q.shape[0]), int(total_tokens), tuple(indices))
    state = getattr(cache, "_cuda_prefill_attention", None)
    if state is None:
        # FA2 eager auto-split bounds tasks by 2*SM/kv_heads; no split uses no scratch.
        # Tile Q <=64 for D>=256, else <=128. FP32 partial V plus softmax sums.
        sm = torch.cuda.get_device_properties(q.device).multi_processor_count
        tile_q = 64 if q.shape[-1] >= 256 else 128
        tasks = 2 * sm // cache.k.shape[-2]
        required = tasks * tile_q * q.shape[1] * (q.shape[-1] + 1) * 4 + 32
        workspace_bytes = 1 << (required - 1).bit_length()
        workspace = torch.empty(workspace_bytes, dtype=torch.uint8, device=q.device)
        wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
            workspace, kv_layout="NHD", backend="fa2")
        state = cache._cuda_prefill_attention = [workspace, wrapper, None]
    wrapper = state[1]
    if state[2] != signature:
        wrapper.plan(
            torch.tensor([0, q.shape[0]], dtype=torch.int32),
            torch.tensor([0, pages], dtype=torch.int32),
            torch.tensor(indices, dtype=torch.int32),
            torch.tensor([(total_tokens - 1) % page_size + 1], dtype=torch.int32),
            q.shape[1], cache.k.shape[-2], q.shape[-1], page_size,
            causal=True, q_data_type=q.dtype, kv_data_type=cache.k.dtype,
            sm_scale=q.shape[-1] ** -0.5, non_blocking=False)
        state[2] = signature
    return wrapper.run(q, (cache.k[layer_slot], cache.v[layer_slot]))


def attention_packed(q, k, v, packed_mask, *, scale):
    """Same FA2 kernel with the per-step prepacked Q8 mask."""
    return torch.stack([
        flashinfer.single_prefill_with_kv_cache(
            q[row], k[row], v[row], packed_custom_mask=packed_mask[row],
            causal=False, kv_layout="NHD", sm_scale=scale, backend="fa2")
        for row in range(q.shape[0])], dim=0)
