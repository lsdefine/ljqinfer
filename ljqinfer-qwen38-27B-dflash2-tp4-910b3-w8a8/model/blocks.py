"""Readable TP4 decoder blocks for Qwen3.5-27B.

The implementation deliberately uses ordinary Torch/Torch-NPU operators.  It is
not the final fast kernel path, but it is a numerically meaningful reference
behind a small ABI that optimized attention/GDN kernels can replace later.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable
import math

from .config import CONFIG
from .weights import FullAttentionWeights, GDNWeights, LayerWeights
from ops.kernels import K


@dataclass
class LayerContext:
    """Per-call state. Token-major tensors may contain several sequences.

    ``sequence_ids`` selects rows in the hot GDN cache.  Full attention accepts
    either an explicit dense causal chunk (no cache) or lists of past local K/V
    tensors, one list entry per sequence.  The orchestration layer owns paged
    cache addressing; blocks only consume tensors.
    """

    positions: Any
    sequence_ids: Any | None = None
    gdn_conv: Any | None = None
    gdn_recurrent: Any | None = None
    # Host-known unique sequence IDs avoid repeated device-to-host extraction in
    # every layer.  Orchestration must only provide values matching sequence_ids.
    host_sids: tuple[int, ...] | None = None
    past_k: list[Any] | None = None
    past_v: list[Any] | None = None
    kv_cache: Any | None = None
    kv_layer_slot: int = -1
    kv_old_lengths: dict[int, int] | None = None
    update_kv: bool = True
    gdn_absolute_start: int = 0
    gdn_checkpoint_offsets: tuple[int, ...] = ()
    gdn_checkpoint_conv: Any | None = None
    gdn_checkpoint_recurrent: Any | None = None


def _identity(x):
    return x


def _all_reduce(x, fn: Callable[[Any], Any] | None):
    return x if fn is None else fn(x)




def gdn_attention(hidden, w: GDNWeights, ctx: LayerContext | None = None,
                  all_reduce: Callable[[Any], Any] | None = None):
    """Run one contiguous prefill row or one decode token per sequence."""
    import torch

    if ctx is None or ctx.gdn_conv is None or ctx.gdn_recurrent is None:
        raise ValueError("GDN requires resident convolution and recurrent state")
    tokens = hidden.shape[0]
    seq = ctx.sequence_ids
    if seq is None or seq.numel() != tokens:
        raise ValueError("sequence_ids must have one entry per token")

    host_sids = ctx.host_sids
    if host_sids is None:
        host_sids = tuple(dict.fromkeys(int(x) for x in seq.detach().cpu().tolist()))
    single_sid = host_sids[0] if len(host_sids) == 1 else None
    if single_sid is None:
        if len(host_sids) != tokens or len(set(host_sids)) != tokens:
            raise ValueError("multi-sequence GDN requires exactly one decode row per sequence")
        batch, sequence_tokens = tokens, 1
        state_indices = seq.long()
        conv = ctx.gdn_conv.index_select(0, state_indices)
        rec = ctx.gdn_recurrent.index_select(0, state_indices)
        if ctx.gdn_checkpoint_offsets:
            raise ValueError("GDN checkpoint collection requires one sequence")
    else:
        batch, sequence_tokens = 1, tokens
        state_indices = None
        conv = ctx.gdn_conv[single_sid:single_sid + 1]
        rec = ctx.gdn_recurrent[single_sid:single_sid + 1]

    if w.qkvz is not None:
        packed = K.w8a8_linear(hidden, w.qkvz)
        qkv_width = (2 * CONFIG.local_gdn_k_heads * CONFIG.linear_key_head_dim +
                     CONFIG.local_gdn_v_heads * CONFIG.linear_value_head_dim)
        qkv, z = packed.split((qkv_width, packed.shape[-1] - qkv_width), dim=-1)
    else:
        qkv = K.w8a8_linear(hidden, w.qkv)
        z = K.w8a8_linear(hidden, w.z)
    ab = K.bf16_linear(hidden, w.ab)
    a, b = ab.chunk(2, dim=-1)

    offsets = ctx.gdn_checkpoint_offsets
    for slot, offset in enumerate(offsets):
        if offset <= 0 or offset > sequence_tokens or offset % 64:
            raise ValueError("GDN checkpoints must be 64-token aligned inside the call")
        target = ctx.gdn_checkpoint_conv[slot]
        if offset >= target.shape[0]:
            target.copy_(qkv[offset - target.shape[0]:offset])
        else:
            keep = target.shape[0] - offset
            target[:keep].copy_(conv[0, offset:])
            target[keep:].copy_(qkv[:offset])

    qkv_rows = qkv.reshape(batch, sequence_tokens, -1)
    mixed = K.silu(K.causal_conv_prefill(qkv_rows, conv, w.conv))
    qn = CONFIG.local_gdn_k_heads * CONFIG.linear_key_head_dim
    vn = CONFIG.local_gdn_v_heads * CONFIG.linear_value_head_dim
    q, k, v = mixed.split((qn, qn, vn), dim=-1)
    q = q.reshape(batch, sequence_tokens, CONFIG.local_gdn_k_heads,
                  CONFIG.linear_key_head_dim)
    k = k.reshape_as(q)
    v = v.reshape(batch, sequence_tokens, CONFIG.local_gdn_v_heads,
                  CONFIG.linear_value_head_dim)
    g = (-torch.exp(w.A_log.float())[None] * torch.nn.functional.softplus(
        a.float() + w.dt_bias.float()[None])).reshape(batch, sequence_tokens, -1)
    beta = torch.sigmoid(b.float()).reshape(batch, sequence_tokens, -1)

    # The native prefill operator consumes 64-token chunks.  Padding happens
    # after the convolution, so decode advances each convolution state exactly
    # once. beta=0 and g=0 make recurrent padding exact no-ops.
    padded_tokens = (sequence_tokens + 63) // 64 * 64
    padding = padded_tokens - sequence_tokens
    if padding:
        q = torch.cat((q, q.new_zeros((batch, padding, *q.shape[2:]))), dim=1)
        k = torch.cat((k, k.new_zeros((batch, padding, *k.shape[2:]))), dim=1)
        v = torch.cat((v, v.new_zeros((batch, padding, *v.shape[2:]))), dim=1)
        g = torch.cat((g, g.new_zeros((batch, padding, g.shape[2]))), dim=1)
        beta = torch.cat((beta, beta.new_zeros((batch, padding, beta.shape[2]))), dim=1)

    y, chunk_states = K.chunk_gated_delta_sequence(q, k, v, g, beta, rec)
    for slot, offset in enumerate(offsets):
        if offset == padded_tokens:
            ctx.gdn_checkpoint_recurrent[slot].copy_(rec[0])
        else:
            ctx.gdn_checkpoint_recurrent[slot].copy_(
                chunk_states[0, :, offset // 64].transpose(-1, -2))

    if state_indices is not None:
        ctx.gdn_conv.index_copy_(0, state_indices, conv)
        ctx.gdn_recurrent.index_copy_(0, state_indices, rec)

    y = K.rmsnorm_gated(
        y[:, :sequence_tokens].reshape(
            tokens, CONFIG.local_gdn_v_heads, CONFIG.linear_value_head_dim),
        z.reshape(tokens, CONFIG.local_gdn_v_heads,
                  CONFIG.linear_value_head_dim),
        w.norm, CONFIG.rms_norm_eps).reshape(tokens, vn)
    return _all_reduce(K.bf16_linear(y, w.out), all_reduce)


_FUSED_CAUSAL_MASKS = {}


def _fused_causal_attention(q, k, v):
    """Native prompt FlashAttention for an equal-length causal GQA chunk."""
    import torch
    import torch_npu

    device_key = str(q.device)
    mask = _FUSED_CAUSAL_MASKS.get(device_key)
    if mask is None:
        mask = torch.ones((2048, 2048), dtype=torch.bool, device=q.device).triu_(1)
        _FUSED_CAUSAL_MASKS[device_key] = mask
    out, _ = torch_npu.npu_fused_infer_attention_score(
        q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0),
        atten_mask=mask,
        num_heads=CONFIG.local_q_heads,
        num_key_value_heads=CONFIG.local_kv_heads,
        scale=CONFIG.head_dim ** -0.5,
        input_layout="BSND",
        sparse_mode=2,
        inner_precise=0,
    )
    return out.squeeze(0)


def _paged_attention(q, cache, layer_slot: int, sequence_id: int,
                     total_tokens: int):
    """Run FIA PageAttention directly over the preallocated 2048-token KV pool."""
    import torch
    import torch_npu

    query_tokens = int(q.shape[0])
    mask = None
    sparse_mode = 0
    if query_tokens != 1:
        device_key = str(q.device)
        mask = _FUSED_CAUSAL_MASKS.get(device_key)
        if mask is None:
            mask = torch.ones((2048, 2048), dtype=torch.bool, device=q.device).triu_(1)
            _FUSED_CAUSAL_MASKS[device_key] = mask
        sparse_mode = 3
    key, value, block_table = cache.paged_attention_view(
        layer_slot, sequence_id, total_tokens)
    out, _ = torch_npu.npu_fused_infer_attention_score(
        q.unsqueeze(0), key, value,
        atten_mask=mask,
        actual_seq_lengths=[query_tokens],
        actual_seq_lengths_kv=[total_tokens],
        block_table=block_table,
        block_size=128,
        num_heads=CONFIG.local_q_heads,
        num_key_value_heads=CONFIG.local_kv_heads,
        scale=CONFIG.head_dim ** -0.5,
        input_layout="BSND",
        sparse_mode=sparse_mode,
        inner_precise=0,
    )
    return out.squeeze(0)


def _dense_attention(q, k, v, causal: bool):
    """Local GQA scaled-dot-product attention in a transparent formulation."""
    import torch

    # Local TP geometry is one KV head serving six query heads.
    k = k.repeat_interleave(CONFIG.local_q_heads // CONFIG.local_kv_heads, dim=1)
    v = v.repeat_interleave(CONFIG.local_q_heads // CONFIG.local_kv_heads, dim=1)
    score = torch.einsum("thd,shd->hts", q.float(), k.float()) / math.sqrt(CONFIG.head_dim)
    if causal:
        t, s = q.shape[0], k.shape[0]
        # When K includes a past prefix, query i can see prefix + [0..i].
        prefix = s - t
        qi = torch.arange(t, device=q.device)[:, None]
        kj = torch.arange(s, device=q.device)[None, :]
        score.masked_fill_(kj[None] > (prefix + qi)[None, :, :], float("-inf"))
    prob = torch.softmax(score, dim=-1).to(v.dtype)
    return torch.einsum("hts,shd->thd", prob, v)


def _packed_sequence_attention(q, k, v, ctx: LayerContext | None):
    """Write current K/V to the page pool, then attend through its block table."""
    import torch

    if ctx is None or ctx.sequence_ids is None:
        if q.device.type == "npu":
            return _fused_causal_attention(q, k, v)
        return _dense_attention(q, k, v, causal=True)
    if ctx.kv_cache is None or ctx.kv_old_lengths is None:
        # Transparent compatibility/reference path. The NPU engine always gives
        # this function a page pool and never reaches these concatenations.
        seq = ctx.sequence_ids.long()
        sids = (list(ctx.host_sids) if ctx.host_sids is not None else
                list(dict.fromkeys(int(x) for x in seq.detach().cpu().tolist())))
        y = torch.empty_like(q)
        for sid in sids:
            index = torch.nonzero(seq == sid, as_tuple=False).flatten()
            sq, sk, sv = (q.index_select(0, index), k.index_select(0, index),
                          v.index_select(0, index))
            pk = None if ctx.past_k is None else ctx.past_k.get(sid)
            pv = None if ctx.past_v is None else ctx.past_v.get(sid)
            kk = sk if pk is None else torch.cat((pk, sk), dim=0)
            vv = sv if pv is None else torch.cat((pv, sv), dim=0)
            y.index_copy_(0, index, _dense_attention(sq, kk, vv, causal=True))
            if ctx.update_kv and ctx.past_k is not None and ctx.past_v is not None:
                ctx.past_k[sid], ctx.past_v[sid] = kk, vv
        return y

    seq = ctx.sequence_ids.long()
    sids = (list(ctx.host_sids) if ctx.host_sids is not None else
            list(dict.fromkeys(int(x) for x in seq.detach().cpu().tolist())))
    y = torch.empty_like(q)
    for sid in sids:
        index = torch.nonzero(seq == sid, as_tuple=False).flatten()
        sq = q if len(sids) == 1 else q.index_select(0, index)
        sk = k if len(sids) == 1 else k.index_select(0, index)
        sv = v if len(sids) == 1 else v.index_select(0, index)
        old = int(ctx.kv_old_lengths[sid])
        ctx.kv_cache.write_kv(ctx.kv_layer_slot, sid, old, sk, sv)
        total = old + int(sk.shape[0])
        if sq.device.type == "npu":
            out = _paged_attention(sq, ctx.kv_cache, ctx.kv_layer_slot, sid, total)
        else:
            kk, vv = ctx.kv_cache.read_kv_range(
                ctx.kv_layer_slot, sid, 0, total)
            out = _dense_attention(sq, kk, vv, causal=True)
        if len(sids) == 1:
            return out
        y.index_copy_(0, index, out)
    return y

def full_attention(hidden, w: FullAttentionWeights, ctx: LayerContext | None = None,
                   all_reduce: Callable[[Any], Any] | None = None):
    """Full-attention reference for one or several packed sequences."""
    import torch

    if w.qkv is not None:
        packed = K.w8a8_linear(hidden, w.qkv)
        qw = CONFIG.local_q_heads * 2 * CONFIG.head_dim
        kw = CONFIG.local_kv_heads * CONFIG.head_dim
        q_gate, k, v = packed.split((qw, kw, kw), dim=-1)
    else:
        q_gate = K.w8a8_linear(hidden, w.q)
        k = K.w8a8_linear(hidden, w.k)
        v = K.w8a8_linear(hidden, w.v)
    k = k.reshape(-1, CONFIG.local_kv_heads, CONFIG.head_dim)
    v = v.reshape(-1, CONFIG.local_kv_heads, CONFIG.head_dim)
    q_gate = q_gate.reshape(-1, CONFIG.local_q_heads, 2 * CONFIG.head_dim)
    q, gate = q_gate.chunk(2, dim=-1)
    q = K.rms_norm(q, w.q_norm, CONFIG.rms_norm_eps)
    k = K.rms_norm(k, w.k_norm, CONFIG.rms_norm_eps)
    positions = (torch.arange(hidden.shape[0], device=hidden.device) if ctx is None
                 else ctx.positions)
    q, k = K.rope(q, k, positions, CONFIG.rotary_dim, CONFIG.rope_theta)

    # A whole causal chunk removes the historical O(tokens) growing-prefix
    # launch sequence. Packed batches still get one independent call per stream.
    y = _packed_sequence_attention(q, k, v, ctx)

    y = y * torch.sigmoid(gate)
    y = K.w8a8_linear(y.reshape(hidden.shape[0], -1), w.o)
    return _all_reduce(y, all_reduce)


def block_forward(hidden, layer_idx: int, weights: LayerWeights | None = None,
                  ctx: LayerContext | None = None,
                  all_reduce: Callable[[Any], Any] | None = None):
    """One pre-norm decoder block.

    Passing no weights retains the old mock contract for scaffold tests.
    """
    if weights is None:
        if layer_idx % CONFIG.full_attention_interval == CONFIG.full_attention_interval - 1:
            hidden = K.full_attention(hidden, layer_idx=layer_idx, cache=ctx)
        else:
            hidden = K.gated_delta_net(hidden, layer_idx=layer_idx, cache=ctx)
        return K.mlp(hidden, layer_idx=layer_idx)

    normed = K.rms_norm(hidden, weights.input_norm, CONFIG.rms_norm_eps)
    if isinstance(weights.attention, FullAttentionWeights):
        attn = full_attention(normed, weights.attention, ctx, all_reduce)
    else:
        attn = gdn_attention(normed, weights.attention, ctx, all_reduce)
    hidden = hidden + attn
    normed = K.rms_norm(hidden, weights.post_norm, CONFIG.rms_norm_eps)
    mlp = K.swiglu_mlp(normed, weights.mlp, all_reduce=all_reduce)
    return hidden + mlp
