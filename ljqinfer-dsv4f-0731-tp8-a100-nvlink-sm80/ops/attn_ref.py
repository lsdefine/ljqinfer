# Reference (pure torch) attention ops: 3 layer types x {prefill, decode}.
#
# ABI (all six):   op(m, x, start_pos, slot=0) -> o_proj   [1, S, dim]
#   m         Attention module (weights, norms, compressor/indexer, freqs_cis, past)
#             m.past is this layer's SlotPool entry (LayerPast/CompressedPast/IndexedPast)
#   slot      sequence slot in the SlotPool (ops are per-sequence: B == 1)
#   x         [B, S, dim] layer input (S == 1 for decode ops)
#   start_pos absolute position of x[:, 0] (0 for prefill ops)
# Side effects: the op owns ALL writes to m.past (main_kv / main_ckv / idx_ckv /
# res_x). Compressor/Indexer are pure functions of x (+ res_x); no other state.
#
# These are the numeric judges for the fused replacements: each fused op must
# match its ref op here (per-op maxdiff), not just the final token stream.
#
# NOTE: arch helpers are accessed lazily via the module object (A.xxx) so this
# file can be imported from model.arch without a circular-import failure.
import torch
import model.arch as A
import model.kernels_torch as KT
from ops import sparse_attn_flat, index_topk, _mod  # broken TC leaves (tests/test_*_leaf.py)


# ---------------------------------------------------------------- shared leaves
def _bind(m, use_indexer: bool, device) -> None:
    """Lazy freqs_cis binding of the compressors."""
    if m.compressor.freqs_cis is None:
        m.compressor.freqs_cis = m.freqs_cis
        if use_indexer:
            m.indexer.freqs_cis = m.freqs_cis


def _prev_window(m, comp, slot: int, start_pos: int):
    """Raw x of the closed window before start_pos (overlap compressors only)."""
    r = comp.compress_ratio
    if not comp.overlap or start_pos == 0:
        return None
    return m.past.res(slot, start_pos - r, start_pos).unsqueeze(0)


def _compress_chunk(m, comp, write, x: torch.Tensor, slot: int, start_pos: int) -> None:
    """Block mode: close every full window of x (r-aligned start); past.res_x
    already holds the previous window."""
    kvc = comp(x, start_pos, _prev_window(m, comp, slot, start_pos))
    if kvc is not None:
        write(slot, start_pos // comp.compress_ratio, kvc[0])


def _compress_step(m, comp, write, slot: int, pos: int) -> None:
    """Decode: token pos is already in past.res_x; when it closes a window,
    compress that window (pure function of res_x)."""
    r = comp.compress_ratio
    if (pos + 1) % r:
        return
    w0 = pos + 1 - r
    xw = m.past.res(slot, w0, pos + 1).unsqueeze(0)
    write(slot, w0 // r, comp(xw, w0, _prev_window(m, comp, slot, w0))[0])


def _window_ctx(m, slot: int, start_pos: int):
    """Decode window: materialized [n, kd] of tokens [lo, start_pos] (oldest->newest)
    plus local ids padded to window_size (same reduction order as the flat past)."""
    lo = max(0, start_pos - m.window_size + 1)
    ctx = m.past.kv(slot, lo, start_pos + 1)
    n = ctx.size(0)
    idxs = torch.full((1, 1, m.window_size), -1, dtype=torch.int32, device=ctx.device)
    idxs[0, 0, :n] = torch.arange(n, dtype=torch.int32, device=ctx.device)
    return ctx.unsqueeze(0), idxs, n


def _chunk_ctx(m, kv: torch.Tensor, slot: int, start_pos: int):
    """Prefill window context for a chunk starting at `start_pos` (128-aligned).
    Returns (ctx [1, p + S, kd], window ids [1, S, K], n_ctx) where ctx = the last
    window_size - 1 tokens before start_pos (from the past ring) followed by this
    chunk's kv, and row i attends flat ids [p + i - W + 1, p + i] (-1 below 0).
    start_pos == 0 reduces exactly to the whole-sequence prefill."""
    seqlen = kv.size(1)
    lo = max(0, start_pos - m.window_size + 1)
    prev = m.past.kv(slot, lo, start_pos)  # [p, kd], p == 0 when start_pos == 0
    p = prev.size(0)
    ctx = torch.cat([prev.unsqueeze(0), kv], dim=1) if p else kv
    n = p + seqlen
    dev = kv.device
    base = torch.arange(p, n, device=dev)                       # flat id of each row
    k = min(n, m.window_size)
    idxs = (base - m.window_size + 1).clamp_min(0).unsqueeze(1) + torch.arange(k, device=dev)
    idxs = torch.where(idxs > base.unsqueeze(1), -1, idxs)
    return ctx, idxs.int().unsqueeze(0), n


def _write_res(m, x: torch.Tensor, slot: int, start_pos: int) -> None:
    """Keep the last 2r raw x rows of the chunk in past.res_x (ring)."""
    n = min(x.size(1), 2 * m.past.ratio)
    m.past.write_res(slot, start_pos + x.size(1) - n, x[0, x.size(1) - n:])


def _q(m, x: torch.Tensor, freqs_cis: torch.Tensor):
    """Low-rank Q: wq_a -> q_norm -> wq_b -> per-head rms -> rope(rope part).
    Returns (q [B,S,H,hd], qr [B,S,q_lora_rank]) — qr feeds the indexer."""
    rd = m.rope_head_dim
    qr = m.q_norm(m.wq_a(x))
    q = m.wq_b(qr).unflatten(-1, (m.n_local_heads, m.head_dim))
    _mod().rms_scale_(q, m.eps)  # == q *= rsqrt(mean(q^2)+eps), bitwise (bf16 step rounding emulated); 1 kernel vs 5
    A.apply_rotary_emb(q[..., -rd:], freqs_cis)
    return q, qr


def _kv(m, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """Latent KV: wkv -> kv_norm -> rope(rope part) -> fp8 act_quant(nope part)."""
    rd = m.rope_head_dim
    kv = m.kv_norm(m.wkv(x))
    A.apply_rotary_emb(kv[..., -rd:], freqs_cis)
    _mod().fp8_qdq64_(kv[..., :-rd])  # == act_quant(..., 64, inplace=True), bitwise; 1 kernel vs ~12
    return kv


def _o(m, o: torch.Tensor, freqs_cis: torch.Tensor, bsz: int, seqlen: int):
    """Inverse rope on the rope part, then grouped low-rank O projection."""
    rd = m.rope_head_dim
    A.apply_rotary_emb(o[..., -rd:], freqs_cis, True)
    o = o.view(bsz, seqlen, m.n_local_groups, -1)
    from ops import wo_a_grouped
    o = wo_a_grouped(o, m.wo_a.weight)
    return m.wo_b(o.flatten(2))


# ---------------------------------------------------------------- Base (r=0)
def attn_base_prefill(m, x: torch.Tensor, start_pos: int = 0, slot: int = 0):
    assert start_pos % 128 == 0, "chunk start must be 128-aligned"
    bsz, seqlen, _ = x.size()
    assert bsz == 1
    freqs_cis = m.freqs_cis[start_pos:start_pos + seqlen]
    q, _ = _q(m, x, freqs_cis)
    kv = _kv(m, x, freqs_cis)
    ctx, topk_idxs, _ = _chunk_ctx(m, kv, slot, start_pos)
    m.past.write_kv(slot, start_pos, kv[0])
    o = sparse_attn_flat(q, ctx, m.attn_sink, topk_idxs, m.softmax_scale)
    return _o(m, o, freqs_cis, bsz, seqlen)


def attn_base_decode(m, x: torch.Tensor, start_pos: int, slot: int = 0):
    bsz, seqlen, _ = x.size()
    assert bsz == 1 and seqlen == 1 and start_pos > 0
    freqs_cis = m.freqs_cis[start_pos:start_pos + 1]
    q, _ = _q(m, x, freqs_cis)
    kv = _kv(m, x, freqs_cis)
    m.past.write_kv(slot, start_pos, kv[0])
    ctx, topk_idxs, _ = _window_ctx(m, slot, start_pos)
    o = sparse_attn_flat(q, ctx, m.attn_sink, topk_idxs, m.softmax_scale)
    return _o(m, o, freqs_cis, bsz, 1)


# ---------------------------------------------------------------- compressed core

def _indexer(m, x, qr, start_pos, offset, slot):
    """arch.Indexer.forward with the score-reduce/top-k tail on broken leaves."""
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
    if seqlen > 1:
        _compress_chunk(m, ix.compressor, m.past.write_ickv, x, slot, start_pos)
    else:
        _compress_step(m, ix.compressor, m.past.write_ickv, slot, start_pos)
    weights = ix.weights_proj(x) * (ix.softmax_scale * ix.n_heads ** -0.5)
    return index_topk(q, m.past.ickv(slot, end_pos).unsqueeze(0), weights, ratio,
                      start_pos, offset, ix.index_topk)


def _compressed_prefill(m, x: torch.Tensor, use_indexer: bool, slot: int, start_pos: int = 0):
    """Whole-sequence prefill (start_pos == 0) or a 128-aligned chunk (start_pos > 0).
    The compressors' carry (open window + prev closed window) is per-slot canonical
    state owned by the past, so slots never share carry."""
    bsz, seqlen, _ = x.size()
    assert bsz == 1 and start_pos % 128 == 0, "chunk start must be 128-aligned"
    end_pos = start_pos + seqlen
    freqs_cis = m.freqs_cis[start_pos:end_pos]
    _bind(m, use_indexer, x.device)
    q, qr = _q(m, x, freqs_cis)
    kv = _kv(m, x, freqs_cis)
    # indices: window part addresses cat([prev window, kv]); compressed part is
    # offset past it and addresses all closed windows of the sequence
    ctx, topk_idxs, offset = _chunk_ctx(m, kv, slot, start_pos)
    if use_indexer:
        c_idxs = _indexer(m, x, qr, start_pos, offset, slot)
    else:
        c_idxs = A.get_compress_topk_idxs(m.compress_ratio, bsz, seqlen, start_pos, offset)
    topk_idxs = torch.cat([topk_idxs, c_idxs], dim=-1)
    # past writes: main_kv, raw-x residue, then compressor (-> main_ckv)
    m.past.write_kv(slot, start_pos, kv[0])
    _compress_chunk(m, m.compressor, m.past.write_ckv, x, slot, start_pos)  # needs prev window from res_x
    _write_res(m, x, slot, start_pos)
    capture = getattr(m, "_cold_kv_v2_tail", None)
    if capture is not None:
        capture(m.layer_id, start_pos, x[0], m.past.ratio)
    kv_compress = m.past.ckv(slot, end_pos).unsqueeze(0)  # all closed windows so far
    o = sparse_attn_flat(q, ctx, m.attn_sink, topk_idxs, m.softmax_scale, kv_compress)
    return _o(m, o, freqs_cis, bsz, seqlen)


def _compressed_decode(m, x: torch.Tensor, start_pos: int, use_indexer: bool, slot: int):
    bsz, seqlen, _ = x.size()
    assert bsz == 1 and seqlen == 1 and start_pos > 0
    freqs_cis = m.freqs_cis[start_pos:start_pos + 1]
    _bind(m, use_indexer, x.device)
    q, qr = _q(m, x, freqs_cis)
    kv = _kv(m, x, freqs_cis)
    # past writes first so the materialized window/ckv include this token
    m.past.write_kv(slot, start_pos, kv[0])
    m.past.write_res(slot, start_pos, x[0])
    _compress_step(m, m.compressor, m.past.write_ckv, slot, start_pos)  # window boundary -> ckv
    # in-flight cat([window ctx, ckv]); compressed ids offset by the window length
    ctx, topk_idxs, offset = _window_ctx(m, slot, start_pos)
    if use_indexer:
        c_idxs = _indexer(m, x, qr, start_pos, offset, slot)
    else:
        c_idxs = A.get_compress_topk_idxs(m.compress_ratio, bsz, 1, start_pos, offset)
    topk_idxs = torch.cat([topk_idxs, c_idxs], dim=-1)
    kv_flat = torch.cat([ctx, m.past.ckv(slot, start_pos + 1).unsqueeze(0)], dim=1)
    o = sparse_attn_flat(q, kv_flat, m.attn_sink, topk_idxs, m.softmax_scale)
    return _o(m, o, freqs_cis, bsz, 1)


# ---------------------------------------------------------------- Indexed (r=4)
def attn_indexed_prefill(m, x: torch.Tensor, start_pos: int = 0, slot: int = 0):
    return _compressed_prefill(m, x, True, slot, start_pos)


def attn_indexed_decode(m, x: torch.Tensor, start_pos: int, slot: int = 0):
    return _compressed_decode(m, x, start_pos, True, slot)


# ---------------------------------------------------------------- Compressed (r=128)
def attn_compressed_prefill(m, x: torch.Tensor, start_pos: int = 0, slot: int = 0):
    return _compressed_prefill(m, x, False, slot, start_pos)


def attn_compressed_decode(m, x: torch.Tensor, start_pos: int, slot: int = 0):
    return _compressed_decode(m, x, start_pos, False, slot)


OPS = {
    (0, "prefill"): attn_base_prefill,       (0, "decode"): attn_base_decode,
    (4, "prefill"): attn_indexed_prefill,    (4, "decode"): attn_indexed_decode,
    (128, "prefill"): attn_compressed_prefill, (128, "decode"): attn_compressed_decode,
}