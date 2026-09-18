# -*- coding: utf-8 -*-
"""BnQ8 decode step (eager reference), see docs/DECODE_BnQ8_CONTRACT.md.

One step = the Q tokens [start_pos, start_pos + Q) of one slot (B == 1 here;
batching is the graph layer's job).  start_pos is arbitrary (not r/128-aligned),
so every write is idempotent: the raw x rows and kv rows of the Q tokens are
written into the rings (overwriting whatever a rejected draft left there), then
every window closed inside the step is recomputed from res_x (pure function of
the ring).  Reads are causal per row (window ids by _chunk_ctx, compressed ids
by get_compress_topk_idxs / index_topk with pos0 gating).

Row 0 of the output is the exact single-token decode result for start_pos
(oracle == token 0), rows 1..Q-1 are the verify rows for draft tokens.
"""
import torch
import model.arch as A
import model.kernels_torch as KT
from ops import sparse_attn_flat, index_topk
from ops.attn_ref import (_bind, _compress_step, _chunk_ctx, _q, _kv, _o)


def _compress_steps(m, comp, write, slot: int, start_pos: int, n: int) -> None:
    """Close (recompute) every window whose last token lies in [start_pos, start_pos + n)."""
    for p in range(start_pos, start_pos + n):
        _compress_step(m, comp, write, slot, p)


def _indexer_q(m, x, qr, start_pos, offset, slot):
    """attn_ref._indexer for an unaligned Q-token step (window closes via step recompute)."""
    ix = m.indexer
    bsz, seqlen, _ = x.size()
    ratio, rd = ix.compress_ratio, ix.rope_head_dim
    end_pos = start_pos + seqlen
    if ix.compressor.freqs_cis is None:
        ix.compressor.freqs_cis = ix.freqs_cis
    q = ix.wq_b(qr).unflatten(-1, (ix.n_local_heads, ix.head_dim))
    A.apply_rotary_emb(q[..., -rd:], ix.freqs_cis[start_pos:end_pos])
    q = KT.rotate_activation(q)
    A.fp4_act_quant(q, A.fp4_block_size, True)
    _compress_steps(m, ix.compressor, m.past.write_ickv, slot, start_pos, seqlen)
    weights = ix.weights_proj(x) * (ix.softmax_scale * ix.n_heads ** -0.5)
    return index_topk(q, m.past.ickv(slot, end_pos).unsqueeze(0), weights, ratio,
                      start_pos, offset, ix.index_topk)


def attn_base_decode_q(m, x: torch.Tensor, start_pos: int, slot: int = 0):
    bsz, seqlen, _ = x.size()
    assert bsz == 1 and start_pos > 0
    freqs_cis = m.freqs_cis[start_pos:start_pos + seqlen]
    q, _ = _q(m, x, freqs_cis)
    kv = _kv(m, x, freqs_cis)
    ctx, topk_idxs, _ = _chunk_ctx(m, kv, slot, start_pos)
    m.past.write_kv(slot, start_pos, kv[0])
    o = sparse_attn_flat(q, ctx, m.attn_sink, topk_idxs, m.softmax_scale)
    return _o(m, o, freqs_cis, bsz, seqlen)


def _compressed_decode_q(m, x: torch.Tensor, start_pos: int, use_indexer: bool, slot: int):
    bsz, seqlen, _ = x.size()
    assert bsz == 1 and start_pos > 0
    end_pos = start_pos + seqlen
    freqs_cis = m.freqs_cis[start_pos:end_pos]
    _bind(m, use_indexer, x.device)
    q, qr = _q(m, x, freqs_cis)
    kv = _kv(m, x, freqs_cis)
    # writes first (idempotent), then window closes recomputed from the ring
    m.past.write_kv(slot, start_pos, kv[0])
    m.past.write_res(slot, start_pos, x[0])
    _compress_steps(m, m.compressor, m.past.write_ckv, slot, start_pos, seqlen)
    ctx, topk_idxs, offset = _chunk_ctx(m, kv, slot, start_pos)
    if use_indexer:
        c_idxs = _indexer_q(m, x, qr, start_pos, offset, slot)
    else:
        c_idxs = A.get_compress_topk_idxs(m.compress_ratio, bsz, seqlen, start_pos, offset)
    topk_idxs = torch.cat([topk_idxs, c_idxs], dim=-1)
    kv_compress = m.past.ckv(slot, end_pos).unsqueeze(0)
    o = sparse_attn_flat(q, ctx, m.attn_sink, topk_idxs, m.softmax_scale, kv_compress)
    return _o(m, o, freqs_cis, bsz, seqlen)


def attn_indexed_decode_q(m, x, start_pos, slot=0):
    return _compressed_decode_q(m, x, start_pos, True, slot)


def attn_compressed_decode_q(m, x, start_pos, slot=0):
    return _compressed_decode_q(m, x, start_pos, False, slot)


OPS_Q = {0: attn_base_decode_q, 4: attn_indexed_decode_q, 128: attn_compressed_decode_q}
