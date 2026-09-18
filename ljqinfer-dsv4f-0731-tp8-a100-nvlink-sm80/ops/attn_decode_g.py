# -*- coding: utf-8 -*-
"""BnQ8 decode step, graph-safe variant of ops/attn_decode_q.py (step 1b-1).

Same contract as attn_decode_q (one slot, Q tokens at start_pos, idempotent
writes), but every position-dependent quantity is a device tensor and every
shape is static in Q, so the step can later be captured by torch.cuda.graph.
`start_pos` may be an int (converted once) or an int64 device tensor [1].

Base layer (r == 0): ring rows are gathered as a fixed [Q + W - 1] context
(row = (pos - W + 1 + i) % ring); window ids are static (row i -> [i, i + W)),
masked -1 only where the absolute token would be < 0.
Compressed layers: still delegated to attn_decode_q until their device-pos
versions land (gate must stay green at every sub-step).
"""
import torch
import os as _os
# reduce-scatter indexer path changes the collective/reduction order -> numerics.
# Hard-off by default; only the explicit setter below may flip it (benchmarks).
_IDX_RS = False


_IDX_RS_HITS = [0, 0]
_IDX_BF16 = False


def set_idx_bf16(v):
    global _IDX_BF16
    _IDX_BF16 = bool(v)



def set_idx_rs(v):
    global _IDX_RS
    _IDX_RS = bool(v)

from ops import sparse_attn_flat, sparse_attn_ring_paged, sparse_attn_pool_paged, _mod
from ops.attn_ref import (_q, _kv, _o)
import ops.attn_decode_q as DQ
import model.arch as A
import model.kernels_torch as KT
from ops.attn_ref import _bind
import torch.distributed as dist


# Per-step cache of the pure position math (tok, ring rows, window ids, compressor
# rows, page-table rows...).  Every layer of one step recomputed the same ~100 tiny
# int64 kernels; now they run once per (pos tensor, Q, slot, r).  Cleared at the
# start of every graph step (arch.forward_q_g / forward_spec_g / step_g).
_CTX = {}


def step_ctx_reset():
    _CTX.clear()


def _ctx(m, pos: torch.Tensor, seqlen: int, slot: int) -> dict:
    key = (pos.data_ptr(), seqlen, slot, m.compress_ratio)
    c = _CTX.get(key)
    if c is None:
        dev = pos.device
        tok = pos + torch.arange(seqlen, device=dev)
        c = {"tok": tok, "ring_rows": tok % m.past.ring,
             "freqs_cis": m.freqs_cis.index_select(0, tok),
             "widx": _window_ids_g(tok, m.window_size, m.past.ring)}
        _CTX[key] = c
    return c


def _all_rows_g(m, pool, slot: int, c: dict):
    """All compressed rows of the slot at static capacity (kv [1, C_max, D], C_max).
    C_max = max_seq // ratio is shape-static (graph-safe); the page table is read on
    device at replay time, unallocated pages (-1) clamp to page 0 and are masked by
    the per-row causal mask (ids >= (p + 1) // r) downstream."""
    _all_rows_ids_init(m, pool, slot, c)
    return pool.flat.index_select(0, c["idx_prow"]).unsqueeze(0), c["idx_C_max"]


def _all_rows_ids_init(m, pool, slot: int, c: dict):
    if "idx_prow" not in c:
        rpp = pool.rpp
        C_max = pool.max_rows
        ar_c = torch.arange(C_max, device=pool.flat.device)
        table = m.past.pt.table[slot].long()
        prow = table[ar_c // rpp] * rpp + ar_c % rpp
        c["idx_prow"], c["idx_C_max"] = prow.clamp_min(0), C_max


def _all_row_ids_g(m, pool, slot: int, c: dict):
    """Row ids of every compressed slot row (int64 [C_max]) without materialising
    the gathered KV: the fused indexer kernel reads the pool through these ids."""
    _all_rows_ids_init(m, pool, slot, c)
    return c["idx_prow"], c["idx_C_max"]


def _indexer_pre(m, x, qr, tok):
    """Token-wise indexer prologue (wq_b + rope + fp4 qdq, weights_proj).  Depends
    only on the flat token stream (tok = per-token positions), so a batch of rows
    runs it ONCE on the concatenated stream (A3)."""
    ix = m.indexer
    rd = ix.rope_head_dim
    if ix.compressor.freqs_cis is None:
        ix.compressor.freqs_cis = ix.freqs_cis
    q = ix.wq_b(qr).unflatten(-1, (ix.n_local_heads, ix.head_dim))
    A.apply_rotary_emb(q[..., -rd:], ix.freqs_cis.index_select(0, tok))
    _mod().had_fp4_qdq_(q)  # fused rotate_activation + fp4_act_quant(inplace), bitwise vs torch
    weights = ix.weights_proj(x) * (ix.softmax_scale * ix.n_heads ** -0.5)
    return q, weights


def _indexer_write(m, x, c: dict, slot: int):
    """Slot-bound indexer pool write (compress + write) of ONE slot.  Returns the
    slot's row-id table (int64 [C_max]) and C_max."""
    ix = m.indexer
    ickv, valid = _compress_g(m, ix.compressor, slot, c, x.dtype)
    _write_ckv_g(m, m.past.idx_pool, slot, c, ickv, valid)
    return _all_row_ids_g(m, m.past.idx_pool, slot, c)


def _indexer_score(m, x, q, weights, c: dict, slot: int):
    """Slot-bound indexer part: compress + pool write + fused score over ONE slot's
    indexer pool.  Returns (score f32 [1,Q,C_max] pre-reduce, k)."""
    ix = m.indexer
    tok = c["tok"]
    prow, C_max = _indexer_write(m, x, c, slot)
    # single kernel: gather rows + q@kv + relu * w + sum over heads + causal mask.
    # Rows past the device-side length (ids >= (p+1)//r) are never read, so the
    # cost follows the real sequence length inside one static CUDA graph.
    score = _mod().index_score_fused(
        q, m.past.idx_pool.flat, prow, weights, tok, ix.compress_ratio)   # f32 [1,Q,C]
    return score, min(ix.index_topk, C_max)


def _indexer_topk(m, score, k, c: dict, offset: int):
    """Post-reduce top-k (score already TP-summed) -> ids + offset (-1 masked)."""
    if "offs" not in c:
        c["offs"] = torch.full_like(c["tok"], offset)
    return _mod().topk_select_post_positions(score, k, m.indexer.compress_ratio, c["offs"], c["tok"])


def _indexer_reduce_(score, pos=None, ratio=None):
    """In-place TP all-reduce of the score (fp32 or bf16 experiment).
    pos/ratio given -> live-prefix peer AR (ops.peer_ar_rows): only columns
    [0, (pos[s]+1)//ratio) of each row are reduced, length read on device
    (graph-safe); the tail is -INF on every rank already.  PEER_AR_ROWS=0 -> NCCL."""
    if not (dist.is_initialized() and dist.get_world_size() > 1):
        return score
    _IDX_RS_HITS[1] += 1
    if _IDX_BF16:
        _sb = score.to(torch.bfloat16)
        dist.all_reduce(_sb)
        return _sb.float()
    if pos is not None and score.dtype == torch.float32 and score.is_contiguous():
        from . import peer_ar_rows
        if peer_ar_rows.ready():
            peer_ar_rows.all_reduce_rows_(score.view(-1, score.size(-1)), pos, int(ratio))
            return score
    dist.all_reduce(score)
    return score


def _indexer_g(m, x, qr, c: dict, offset: int, slot: int):
    """Device-pos indexer (single row): score every allocated indexer row, causal
    mask per query row (ids >= (p + 1) // r), TP all-reduce, top-k -> ids + offset."""
    q, weights = _indexer_pre(m, x, qr, c["tok"])
    score, k = _indexer_score(m, x, q, weights, c, slot)
    _ws = dist.get_world_size() if dist.is_initialized() else 1
    _Q = score.size(1)
    if _ws > 1 and _IDX_RS and _Q % _ws == 0:
        _IDX_RS_HITS[0] += 1
        # score rows are independent: reduce_scatter gives each rank the FULL
        # (head-summed) score of its own Q rows -> local exact top-k -> gather ids.
        if "offs" not in c:
            c["offs"] = torch.full_like(c["tok"], offset)
        tok = c["tok"]; ratio = m.indexer.compress_ratio
        _r = dist.get_rank(); _ql = _Q // _ws
        _flat = score.reshape(-1)
        _loc = torch.empty(_flat.numel() // _ws, dtype=_flat.dtype, device=_flat.device)
        dist.reduce_scatter_tensor(_loc, _flat)
        _s0 = _r * _ql
        _o = _mod().topk_select_post_positions(
            _loc.view(1, _ql, -1), k, ratio, c["offs"][_s0:_s0 + _ql], tok[_s0:_s0 + _ql])
        _out = torch.empty((1, _Q, k), dtype=_o.dtype, device=_o.device)
        dist.all_gather_into_tensor(_out.view(-1), _o.reshape(-1))
        return _out
    score = _indexer_reduce_(score, c["tok"], m.indexer.compress_ratio)
    return _indexer_topk(m, score, k, c, offset)


def _window_ids_g(tok: torch.Tensor, W: int, ring: int) -> torch.Tensor:
    """Sliding-window ids for sparse_attn_ring_paged: row i -> ring rows of
    absolute tokens [tok_i - W + 1, tok_i], -1 where the token is < 0."""
    t = tok.unsqueeze(1) - W + 1 + torch.arange(W, device=tok.device)   # [Q, W] absolute
    return torch.where(t < 0, -1, t % ring)


def _pos_t(start_pos, device):
    if torch.is_tensor(start_pos):
        return start_pos.to(device=device, dtype=torch.int64).reshape(1)
    return torch.tensor([start_pos], device=device, dtype=torch.int64)


def _res_rows(m, c):
    """Cached (tok % 4r) row map for the residual/derived rings (per _ctx entry)."""
    k = ("res_rows", m.past.ratio)
    r = c.get(k)
    if r is None:
        r = c[k] = c["tok"] % (4 * m.past.ratio)
    return r



# ---------------------------------------------------------------------------
# Batched-decode layout (see temp/amdahl_b_report.md, A1/A2):
#   every variant = _q/_kv (token-wise)  ->  _core_* (row-bound: KV ring/page
#   table of ONE slot)  ->  _o (token-wise, contains the RowParallel all_reduce).
# attn_decode_g_batch runs _q/_kv/_o ONCE on the flat [1, B*Q] stream and only
# the _core_* per row, so the projection GEMMs read their weights once and the
# hidden all_reduce is issued once per layer instead of B times.  The single-row
# entries (OPS_G) are the B == 1 instance of the very same op sequence.
# ---------------------------------------------------------------------------

def _base_fused_partial(m, x, c):
    """Fused leaf (broken attn_rank_sparse_decode_fp8 idea on the ring ABI): q/kv proj
    + norms + rope + ring write + window attention + wo_a/wo_b partial.  Returns the
    fp32 PARTIAL (pre all_reduce) so the caller can merge rows before one all_reduce."""
    from ops import attn_decode_fused_r0
    assert m.q_norm.eps == m.eps and m.kv_norm.eps == m.eps
    seqlen = x.size(1)
    main = m.past.main_kv[c["slot"]]                               # [ring, kd]
    return attn_decode_fused_r0(x.reshape(seqlen, -1), c["freqs_cis"], c["tok"], main, m, m.window_size).float()


def _base_fused_ok(m):
    return not isinstance(m.wo_b.weight, list) and m.wo_b.bias is None


def _core_base_pre(m, x, q, qr, kv, c, slot):
    """r == 0 core, phase 1 (slot-bound): ring write.  Returns idxs [1, Q, K]."""
    main = m.past.main_kv[slot]                                    # [ring, kd]
    main.index_copy_(0, c["ring_rows"], kv[0].to(main.dtype))     # idempotent write
    return c["widx"].unsqueeze(0)


def _core_base(m, x, q, qr, kv, c, slot):
    dev = x.device
    idxs = _core_base_pre(m, x, q, qr, kv, c, slot)
    main = m.past.main_kv[slot]
    return sparse_attn_ring_paged(q, main, main.unsqueeze(0), torch.zeros(1, dtype=torch.int64, device=dev),
                                  m.attn_sink, idxs, m.softmax_scale)


def attn_base_decode_g(m, x: torch.Tensor, start_pos, slot: int = 0):
    bsz, seqlen, _ = x.size()
    assert bsz == 1
    dev = x.device
    pos = _pos_t(start_pos, dev)                                   # [1]
    c = _ctx(m, pos, seqlen, slot)
    freqs_cis = c["freqs_cis"]
    if _base_fused_ok(m):
        c["slot"] = slot
        y = _base_fused_partial(m, x, c)
        from . import peer_ar
        peer_ar.all_reduce(y)                                      # RowParallelLinear semantics
        return y.to(x.dtype).view(bsz, seqlen, -1)
    q, _ = _q(m, x, freqs_cis)
    kv = _kv(m, x, freqs_cis)
    o = _core_base(m, x, q, None, kv, c, slot)
    return _o(m, o, freqs_cis, bsz, seqlen)


def _compress_g(m, comp, slot: int, c: dict, dtype):
    """Compressor for the Q candidate windows ending at each token of `tok`
    ([p + 1 - r, p + 1), gathered from the derived projection rings).  Same math as
    Compressor.forward (overlap with the previous window when comp.overlap,
    softmax over the window, norm, rope, act/fp4 quant) but with the window
    start as a device tensor.  Returns (kv [Q, D], valid [Q]) where
    valid = the window really closes at p ((p + 1) % r == 0)."""
    tok = c["tok"]
    dev = tok.device
    r, rd, d = comp.compress_ratio, comp.rope_head_dim, comp.head_dim
    _, kvr, scr = m.past.derived[comp.derived_name]                # fp32 [4r, coff*d] rings
    kvr, scr = kvr[slot], scr[slot]
    cc = _cmp_cc(comp, c)
    # single launch: gathers window rows from the rings in-kernel (no torch gather/cat/where)
    kv = _mod().compressor_rows_ring(kvr, scr, cc["rowmap"], cc["has_prev"],
                                     comp.ape.float(), comp.norm.weight, cc["freqs"],
                                     cc["valid_i"], r, d, rd, comp.norm.eps,
                                     comp.overlap, comp.rotate, False)  # [Q, D] bf16
    return kv, cc["valid"]


def _cmp_cc(comp, c: dict):
    """Per-ctx (per ratio) static compressor tables: window row map, has_prev, valid,
    freqs, logical compressed row.  Shared by the single-row and batched paths."""
    tok = c["tok"]
    dev = tok.device
    r = comp.compress_ratio
    cc = c.get(("cmp", r))                                         # per-ratio sub-cache
    if cc is None:
        w0 = tok + 1 - r                                           # [Q] window starts
        ar_r = torch.arange(r, device=dev)
        rows = (w0.unsqueeze(1) + ar_r) % (4 * r)
        if comp.overlap:
            # kernel overlap layout: window rows [0,r) = previous window (channels [0,d)),
            # rows [r,2r) = current window (channels [d,2d)); has_prev == 0 skips the
            # previous half (its softmax weight would be exactly 0).
            rows = torch.cat([(w0.unsqueeze(1) - r + ar_r) % (4 * r), rows], dim=1)
        cc = c[("cmp", r)] = {
            "rowmap": rows.contiguous(),                           # [Q, R] ring rows
            "has_prev": (w0 >= r).int(),
            "valid": (tok + 1) % r == 0,
            "freqs": comp.freqs_cis.index_select(0, w0.clamp_min(0)),   # [Q, rd/2] complex
            "c": ((tok + 1) // r - 1).clamp_min(0).long()}         # [Q] logical compressed row
        cc["valid_i"] = cc["valid"].int()
    return cc


def _write_ckv_g(m, pool, slot: int, c: dict, kv: torch.Tensor, valid: torch.Tensor):
    """Masked paged write of closed window c = (p + 1) // r - 1 (valid rows only).
    broken leaf paged_scatter_positions_masked: rows with valid == 0 are skipped in-kernel,
    so no fallback-row trick / no D2H sync is needed (graph-safe)."""
    cc = c[("cmp", pool.ratio)]                                    # filled by _compress_g
    if "pt_table" not in c:
        c["pt_table"] = m.past.pt.table[slot].long()
    _mod().paged_scatter_positions_masked(pool.data, c["pt_table"], cc["c"],
                                          cc["valid_i"], kv.to(pool.data.dtype).contiguous())



# ---------------------------------------------------------------------------
# A10: slot-bound WRITES of all rows in ONE launch each (main ring, res/derived
# rings, compressor tail, paged ckv write, indexer pool).  Every kernel below
# already addresses rows through an explicit row map / page table, so a batch of
# slots is expressed purely as flat views of the whole pools + per-row offsets
# (slot * ring rows / stacked page tables).  Pure copies and row-independent
# kernels: bit-identical to the per-row sequence, only fewer launches.
# Static index tables are cached in cs[0] under key ("a10", tuple(slots), ...).
def _a10_key(slots, name):
    return ("a10", tuple(slots), name)


def _a10_rows(cs, slots, name, stride, c0, key):
    """cat over rows of cs[b][name] (int64 row ids) + slots[b] * stride."""
    t = c0.get(key)
    if t is None:
        t = c0[key] = torch.cat([c[name] + slots[b] * stride for b, c in enumerate(cs)])
    return t


def _compress_rows_g(m, comp, cs, slots, c0, dtype):
    """compressor_rows_ring over all rows in ONE launch: rings viewed flat
    [n_slots*4r, C], row map = per-row map + slot*4r.  Returns (kv [B*Q, D],
    valid_i [B*Q], c_all [B*Q])."""
    r, rd, d = comp.compress_ratio, comp.rope_head_dim, comp.head_dim
    _, kvr, scr = m.past.derived[comp.derived_name]                # fp32 [n_slots, 4r, C]
    C = kvr.size(-1)
    ccs = [_cmp_cc(comp, c) for c in cs]
    k = _a10_key(slots, ("cmp", r))
    t = c0.get(k)
    if t is None:
        t = c0[k] = (
            torch.cat([cc["rowmap"] + slots[b] * (4 * r) for b, cc in enumerate(ccs)]).contiguous(),
            torch.cat([cc["has_prev"] for cc in ccs]),
            torch.cat([cc["freqs"] for cc in ccs]).contiguous(),
            torch.cat([cc["valid_i"] for cc in ccs]),
            torch.cat([cc["c"] for cc in ccs]))
    rowmap, has_prev, freqs, valid_i, c_all = t
    kv = _mod().compressor_rows_ring(kvr.view(-1, C), scr.view(-1, C), rowmap, has_prev,
                                     comp.ape.float(), comp.norm.weight, freqs,
                                     valid_i, r, d, rd, comp.norm.eps,
                                     comp.overlap, comp.rotate, False)  # [B*Q, D] bf16
    return kv, valid_i, c_all


def _write_ckv_rows_g(m, pool, cs, slots, c0, kv, valid_i, c_all):
    """Masked paged write of all rows: page tables stacked [B, T] (kernel infers
    B from ptab.dim() == 2 and Q = n / B)."""
    k = _a10_key(slots, ("ptab",))
    pt = c0.get(k)
    if pt is None:
        for b, c in enumerate(cs):
            if "pt_table" not in c:
                c["pt_table"] = m.past.pt.table[slots[b]].long()
        pt = c0[k] = torch.stack([c["pt_table"] for c in cs]).contiguous()
    _mod().paged_scatter_positions_masked(pool.data, pt, c_all, valid_i,
                                          kv.to(pool.data.dtype).contiguous())


def _rows_write_g(m, x, kv, cs, slots, B, Q):
    """Phase-1 writes of ALL rows (main ring, res/derived rings, compressor +
    paged ckv) in one launch each.  Same ops as _core_*_write per row, batched."""
    c0 = cs[0]
    past = m.past
    ring, r = past.ring, m.compress_ratio
    # main KV ring: index_copy on the flat [n_slots*ring, kv_dim] view
    rows = _a10_rows(cs, slots, "ring_rows", ring, c0, _a10_key(slots, "ring_rows"))
    main = past.main_kv
    main.view(-1, main.size(-1)).index_copy_(0, rows, kv[0].to(main.dtype))
    # residual + derived rings (write_res_t fused path, batched): the derived
    # projection was computed ONCE on the flat stream (A5) -> use it whole.
    for c in cs:
        _res_rows(m, c)
    rrows = _a10_rows(cs, slots, ("res_rows", past.ratio), 4 * past.ratio, c0, _a10_key(slots, "res_rows"))
    x0 = x[0]
    if len(past.derived) == 1 and x0.dtype == past.res_x.dtype and x0.is_cuda and c0.get("a10_derived_flat") is not None:
        (name, (_, kvr, scr)), = past.derived.items()
        dkv, dsc = kvr.view(-1, kvr.size(-1)), scr.view(-1, scr.size(-1))
        kvs, scs = c0["a10_derived_flat"][name]                    # [B*Q, w] flat (A5), no cat
        from ops import scatter3_rows
        scatter3_rows(past.res_x.view(-1, past.res_x.size(-1)), dkv, dsc, rrows,
                      x0.contiguous(), kvs.contiguous(), scs.contiguous())
    else:
        # generic (non-fused) layout: keep the per-row write_res_t op sequence
        # (only reachable when Past has != 1 derived ring or dtype mismatch).
        for b, c in enumerate(cs):
            past.write_res_t(slots[b], c["tok"], x0[b * Q:(b + 1) * Q], c[("res_rows", past.ratio)], c.get("derived"))
    # compressor tail + paged ckv write (one launch each over all rows)
    ckv, valid_i, c_all = _compress_rows_g(m, m.compressor, cs, slots, c0, x.dtype)
    _write_ckv_rows_g(m, past.ckv_pool, cs, slots, c0, ckv, valid_i, c_all)


def _indexer_write_rows_g(m, x, cs, slots, B, Q):
    """Batched _indexer_write: indexer compressor + pool write in ONE launch each;
    returns the per-row id tables (as _indexer_write does per row)."""
    ix = m.indexer
    c0 = cs[0]
    ickv, valid_i, c_all = _compress_rows_g(m, ix.compressor, cs, slots, c0, x.dtype)
    _write_ckv_rows_g(m, m.past.idx_pool, cs, slots, c0, ickv, valid_i, c_all)
    return [_all_row_ids_g(m, m.past.idx_pool, slots[b], cs[b])[0] for b in range(B)]


def _core_compressed_idxs(m, c, slot):
    """r == 128: static id table (window + all closed compressed rows)."""
    ring, r = m.past.ring, m.compress_ratio
    cpool = m.past.ckv_pool
    if "topk_idxs" not in c:
        tok = c["tok"]
        ar_c = torch.arange(cpool.max_rows, device=tok.device)         # static capacity
        n_closed = (tok + 1) // r                                      # [Q]
        c_idxs = torch.where(ar_c.unsqueeze(0) < n_closed.unsqueeze(1), ar_c.unsqueeze(0) + ring, -1)
        c["topk_idxs"] = torch.cat([c["widx"], c_idxs], dim=-1).unsqueeze(0)
        c["keff"] = (c["widx"].shape[-1] + n_closed).contiguous()
    return c["topk_idxs"]


def _core_compressed_pre(m, x, q, qr, kv, c, slot):
    """r == 128 core, phase 1 (slot-bound): ring write + residual ring + compressor +
    paged write of ONE slot.  Returns idxs [1, Q, K] (window + all closed compressed
    rows, static K = |widx| + cpool.max_rows)."""
    dev = x.device
    ring, r = m.past.ring, m.compress_ratio
    tok = c["tok"]
    main = m.past.main_kv[slot]
    main.index_copy_(0, c["ring_rows"], kv[0].to(main.dtype))
    m.past.write_res_t(slot, tok, x[0], _res_rows(m, c), c.get("derived"))
    ckv, valid = _compress_g(m, m.compressor, slot, c, x.dtype)
    _write_ckv_g(m, m.past.ckv_pool, slot, c, ckv, valid)
    # compressed context: all allocated rows read in place from the paged pool,
    # causal mask per row; ids >= ring address compressed row (id - ring)
    cpool = m.past.ckv_pool
    if "topk_idxs" not in c:
        ar_c = torch.arange(cpool.max_rows, device=dev)                # static capacity
        n_closed = (tok + 1) // r                                  # [Q]
        c_idxs = torch.where(ar_c.unsqueeze(0) < n_closed.unsqueeze(1), ar_c.unsqueeze(0) + ring, -1)
        c["topk_idxs"] = torch.cat([c["widx"], c_idxs], dim=-1).unsqueeze(0)
        # Live ids stop at |widx| + n_closed (the compressed half is a strict
        # prefix): hand that bound to the kernel so its split-K blocks stop at
        # the real context instead of the static capacity (128 + max_seq/128).
        c["keff"] = (c["widx"].shape[-1] + n_closed).contiguous()
    return c["topk_idxs"]


def _core_compressed(m, x, q, qr, kv, c, slot):
    """r == 128 layer core: phase 1 + window/compressed sparse attention of ONE slot."""
    idxs = _core_compressed_pre(m, x, q, qr, kv, c, slot)
    return sparse_attn_ring_paged(q, m.past.main_kv[slot], m.past.ckv_pool.data, c["pt_table"],
                                  m.attn_sink, idxs, m.softmax_scale, c.get("keff"))


def attn_compressed_decode_g(m, x: torch.Tensor, start_pos, slot: int = 0):
    """r == 128 layer: base window + all closed compressed windows (no indexer).
    kv_compress covers the slot's static row capacity (max_seq // r) and each row
    masks ids >= (p + 1) // r."""
    bsz, seqlen, _ = x.size()
    assert bsz == 1
    dev = x.device
    pos = _pos_t(start_pos, dev)
    c = _ctx(m, pos, seqlen, slot)
    freqs_cis = c["freqs_cis"]
    _bind(m, False, dev)
    q, _ = _q(m, x, freqs_cis)
    kv = _kv(m, x, freqs_cis)
    o = _core_compressed(m, x, q, None, kv, c, slot)
    return _o(m, o, freqs_cis, bsz, seqlen)


def _core_indexed_write(m, x, kv, c, slot):
    """r == 4 core, phase 1 (slot-bound): KV/res/ckv/indexer-pool writes of ONE
    slot.  Returns the slot's indexer row-id table (int64 [C_max]) and C_max."""
    tok = c["tok"]
    main = m.past.main_kv[slot]
    main.index_copy_(0, c["ring_rows"], kv[0].to(main.dtype))
    m.past.write_res_t(slot, tok, x[0], _res_rows(m, c), c.get("derived"))
    ckv, valid = _compress_g(m, m.compressor, slot, c, x.dtype)
    _write_ckv_g(m, m.past.ckv_pool, slot, c, ckv, valid)
    return _indexer_write(m, x, c, slot)


def _core_indexed_pre(m, x, q_ix, w_ix, kv, c, slot):
    """r == 4 core, phase 1 (slot-bound): KV/res/ckv writes + indexer score of
    ONE slot.  Returns (score pre-reduce, k)."""
    ix = m.indexer
    prow, C_max = _core_indexed_write(m, x, kv, c, slot)
    score = _mod().index_score_fused(
        q_ix, m.past.idx_pool.flat, prow, w_ix, c["tok"], ix.compress_ratio)   # f32 [1,Q,C]
    return score, min(ix.index_topk, C_max)


def _indexer_rows_score(m, q_ix, w_ix, tok_all, cs, slots, prows, B, Q):
    """A9: ONE fused indexer-score launch for B rows (prow [B, C_max], one row-id
    table per slot; C_max = idx_pool.max_rows is the same for every slot).  Each
    batch row is an independent copy of the single-row problem inside the kernel
    (grid.y = row), so per-output reduction order == the per-row call.  Returns
    score f32 [B, Q, C_max] pre-reduce and k."""
    ix = m.indexer
    c0 = cs[0]
    key = tuple(slots)
    if c0.get("idx_prow_rows_key") != key:                  # captured once per graph (like idx_prow)
        c0["idx_prow_rows"], c0["idx_prow_rows_key"] = torch.stack(prows), key
    C_max = m.past.idx_pool.max_rows
    h = ix.n_local_heads
    score = _mod().index_score_fused(
        q_ix.view(B, Q, h, -1), m.past.idx_pool.flat, c0["idx_prow_rows"],
        w_ix.reshape(B, Q, h), tok_all, ix.compress_ratio)
    return score, min(ix.index_topk, C_max)


def _core_indexed_idxs(m, score, k, c, slot):
    """r == 4 core, phase 2a: top-k on the TP-summed score.  Returns idxs [1, Q, K]
    (window + index_topk, static K)."""
    c_idxs = _indexer_topk(m, score, k, c, m.past.ring)                 # ids + ring (-1 masked)
    return torch.cat([c["widx"].unsqueeze(0), c_idxs.long()], dim=-1)


def _core_indexed_post(m, q, score, k, c, slot):
    """r == 4 core, phase 2: top-k on the TP-summed score + sparse attention."""
    topk_idxs = _core_indexed_idxs(m, score, k, c, slot)
    return sparse_attn_ring_paged(q, m.past.main_kv[slot], m.past.ckv_pool.data, c["pt_table"],
                                  m.attn_sink, topk_idxs, m.softmax_scale)


_CORE_PRE_G = {0: _core_base_pre, 128: _core_compressed_pre}
_SLOT_TBL = {}


def _slot_table(slots, dev):
    """[B, 1] int64 page table selecting ring page `slot` of the whole main_kv
    pool for each row.  Built from fill kernels (no H2D copy -> graph-safe) and
    cached per (slots, device)."""
    key = (tuple(int(s) for s in slots), str(dev))
    t = _SLOT_TBL.get(key)
    if t is None:
        t = torch.stack([torch.full((1,), int(s), dtype=torch.int64, device=dev) for s in slots])
        _SLOT_TBL[key] = t
    return t


def _attn_rows_g(m, q, cs, slots, idxs):
    """A8: ONE sparse attention launch for all B rows.  The kernel is batch-aware
    (token -> batch = token // Q; page_table / ctable indexed per batch), so the
    B per-row launches (pool = main_kv[slot]) become one launch over the whole
    main_kv pool [n_slots, ring, 512] with page_table[b] = [slots[b]] (page = ring)
    and ctable[b] = the slot's compressed page table.  Per token the kernel does
    exactly the per-row work -> identical values; K is static per layer kind so
    the rows concatenate.  B == 1 goes through the same path (no fork)."""
    B = len(slots)
    ptab = _slot_table(slots, q.device)                            # [B, 1]
    if m.ratio == 0:                                               # no compressed ids: cpool/ctable unused (alias the ring pool)
        cpool, ctab = m.past.main_kv, ptab
    else:
        cpool = m.past.ckv_pool.data
        ctab = cs[0]["pt_table"].unsqueeze(0) if B == 1 else torch.stack([c["pt_table"] for c in cs])
    # keff: per-token upper bound of the live id run (r == 128 layers only, where
    # the id table is [window | closed compressed rows] and the tail is static
    # capacity padding).  Without it every split-K block walks the full static
    # K (128 + max_seq/128), i.e. work grows with the graph's max_seq instead of
    # the real context.  None -> kernel keeps the old full-K behaviour.
    # keff (per-row K upper bound) is ALWAYS applied -- no env switch, no implicit
    # fallback: a missing keff would silently restore the full-K scan.
    keffs = [c.get("keff") for c in cs]
    keff = None
    if all(k is not None for k in keffs):
        keff = keffs[0] if B == 1 else torch.cat(keffs)
    return sparse_attn_pool_paged(q, m.past.main_kv, ptab, cpool, ctab, m.attn_sink,
                                  torch.cat(idxs, dim=1) if B > 1 else idxs[0], m.softmax_scale,
                                  keff)


def _core_indexed(m, x, q, qr, kv, c, slot):
    """r == 4 layer core: same as compressed but the compressed ids come from the
    device-pos indexer (which scores ONE slot's indexer pool)."""
    ring = m.past.ring
    tok = c["tok"]
    main = m.past.main_kv[slot]
    main.index_copy_(0, c["ring_rows"], kv[0].to(main.dtype))
    m.past.write_res_t(slot, tok, x[0], _res_rows(m, c), c.get("derived"))
    ckv, valid = _compress_g(m, m.compressor, slot, c, x.dtype)
    _write_ckv_g(m, m.past.ckv_pool, slot, c, ckv, valid)
    c_idxs = _indexer_g(m, x, qr, c, ring, slot)                   # ids + ring (-1 masked)
    topk_idxs = torch.cat([c["widx"].unsqueeze(0), c_idxs.long()], dim=-1)
    return sparse_attn_ring_paged(q, main, m.past.ckv_pool.data, c["pt_table"],
                                  m.attn_sink, topk_idxs, m.softmax_scale)


def attn_indexed_decode_g(m, x: torch.Tensor, start_pos, slot: int = 0):
    """r == 4 layer: base window + indexer top-k over all closed compressed
    windows (overlap compressor).  Same layout as the r == 128 layer, but the
    compressed ids come from the device-pos indexer instead of a causal range."""
    bsz, seqlen, _ = x.size()
    assert bsz == 1
    dev = x.device
    pos = _pos_t(start_pos, dev)
    c = _ctx(m, pos, seqlen, slot)
    freqs_cis = c["freqs_cis"]
    _bind(m, True, dev)
    q, qr = _q(m, x, freqs_cis)
    kv = _kv(m, x, freqs_cis)
    o = _core_indexed(m, x, q, qr, kv, c, slot)
    return _o(m, o, freqs_cis, bsz, seqlen)


_CORE_G = {0: _core_base, 4: _core_indexed, 128: _core_compressed}


def attn_decode_g_batch(m, x: torch.Tensor, pos_list, slots):
    """Batched BnQ8 decode: x [1, B*Q, D] is the flat row stream (row b = tokens
    [b*Q, (b+1)*Q) at pos_list[b] in slot slots[b]).  Token-wise stages (_q, _kv,
    _o incl. its all_reduce) run once on the flat stream; only the slot-bound core
    runs per row.  B == 1 executes exactly the OPS_G[m.ratio] op sequence."""
    B = len(slots)
    bsz, S, _ = x.size()
    assert bsz == 1 and S % B == 0, (x.shape, B)
    Q = S // B
    dev = x.device
    cs = [_ctx(m, _pos_t(pos_list[b], dev), Q, slots[b]) for b in range(B)]
    if m.ratio == 0 and _base_fused_ok(m):
        ys = []
        for b in range(B):
            cs[b]["slot"] = slots[b]
            ys.append(_base_fused_partial(m, x[:, b * Q:(b + 1) * Q], cs[b]))
        y = ys[0] if B == 1 else torch.cat(ys, dim=0)             # [B*Q, D] fp32 partial
        from . import peer_ar
        peer_ar.all_reduce(y)                                      # ONE all_reduce for all rows
        return y.to(x.dtype).view(1, S, -1)
    _bind(m, m.ratio == 4, dev)
    freqs_cis = cs[0]["freqs_cis"] if B == 1 else torch.cat([c["freqs_cis"] for c in cs], dim=0)
    q, qr = _q(m, x, freqs_cis)                                    # one GEMM chain, M = B*Q
    kv = _kv(m, x, freqs_cis)
    # A5: compressor derived projection (wkv/wgate skinny GEMM, batch-invariant)
    # ONCE on the flat stream instead of per row inside write_res_t.  Same
    # condition as the fused path of write_res_t; B == 1 -> identical op sequence.
    # cs[] dicts are cached across layers (_CTX): always overwrite, never inherit.
    if m.ratio != 0 and m.past.derived:
        dd = m.past.derive(x[0])                                   # {name: (kv [B*Q,w], sc [B*Q,w])}
        cs[0]["a10_derived_flat"] = dd                             # A10: whole flat tensors for the batched ring write
        for b in range(B):
            sl = slice(b * Q, (b + 1) * Q)
            cs[b]["derived"] = {k: (kv[sl], sc[sl]) for k, (kv, sc) in dd.items()}
    else:
        for c in cs:
            c["derived"] = None
        cs[0]["a10_derived_flat"] = None
    if m.ratio == 4 and B > 1:
        # A3: indexer prologue once on the flat stream; per-row slot-bound score;
        # ONE all_reduce over the concatenated (flattened, variable C_max) scores
        # instead of B; then per-row top-k + sparse attention.  all_reduce is
        # element-wise, so merging rows into one call changes no value.  (The
        # IDX_RS reduce_scatter experiment path stays single-row only.)
        # A9: per-row slot-bound writes only; the indexer score is ONE launch over
        # all rows (prow [B, C_max]) instead of B (index_score was 1.8 ms of the
        # B4-B1 gap and the B>1 kernel re-read the pool once per query row).
        tok_all = torch.cat([c["tok"] for c in cs])
        q_ix, w_ix = _indexer_pre(m, x, qr, tok_all)
        # A10: all slot-bound writes batched (one launch per pool instead of B)
        _rows_write_g(m, x, kv, cs, slots, B, Q)
        prows = _indexer_write_rows_g(m, x, cs, slots, B, Q)
        score, k = _indexer_rows_score(m, q_ix, w_ix, tok_all, cs, slots, prows, B, Q)  # [B, Q, C]
        score = _indexer_reduce_(score, tok_all, m.indexer.compress_ratio)   # element-wise
        idxs = [_core_indexed_idxs(m, score[b:b + 1], k, cs[b], slots[b]) for b in range(B)]
        o = _attn_rows_g(m, q, cs, slots, idxs)                    # A8: one attention launch
        return _o(m, o, freqs_cis, 1, S)
    # A8: per-row slot-bound writes (+ top-k ids), then ONE sparse attention launch
    # over all rows instead of B (sparse_attn was 3.2 ms of the B4-B1 gap).
    if m.ratio == 4:                                               # B == 1 (A3 path is B > 1 only): one row = one launch already
        o = _core_indexed(m, x, q, qr, kv, cs[0], slots[0])
        return _o(m, o, freqs_cis, 1, S)
    if B == 1:
        core_pre = _CORE_PRE_G[m.ratio]
        idxs = [core_pre(m, x, q, qr, kv, cs[0], slots[0])]
    else:
        # A10 (r == 128, B > 1): batched slot-bound writes, then per-row static id tables
        _rows_write_g(m, x, kv, cs, slots, B, Q)
        idxs = [_core_compressed_idxs(m, cs[b], slots[b]) for b in range(B)]
    o = _attn_rows_g(m, q, cs, slots, idxs)
    return _o(m, o, freqs_cis, 1, S)                               # one wo_a/wo_b + one all_reduce


def _delegate(fn):
    def op(m, x, start_pos, slot=0):
        if torch.is_tensor(start_pos):
            start_pos = int(start_pos.reshape(-1)[0])
        return fn(m, x, start_pos, slot)
    return op


OPS_G = {0: attn_base_decode_g,
         4: attn_indexed_decode_g,
         128: attn_compressed_decode_g}
