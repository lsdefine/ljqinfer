"""Fused candidate row selection for decode sparse attention.

Replaces the ATen chain ``gather -> masked_fill -> topk -> gather -> isfinite
-> masked_fill`` (six launches plus a cub radix sort over [6,16384]) with five
Triton launches that keep the whole candidate block parallel instead of
serialising one row per program:

  1. ``_cand_hist``  gather + validity + 256-bin histogram of bits[31:24] of
     the order-preserving score key, one program per (row, 1k chunk).
  2. ``_cand_b1``    per row: walk that histogram down to the bin holding the
     ``width``-th best score; publish bin + remaining quota; reset; fill -1.
  3. ``_cand_hist2`` refine: histogram of bits[23:16] for the candidates that
     landed exactly in the boundary bin.  Eight exponent bits alone cannot
     separate scores (they are all the same magnitude), so the second pass is
     what makes the selection meaningful.
  4. ``_cand_b2``    finish the 16-bit threshold and the slot counters.
  5. ``_cand_emit``  write ids: all keys above the 16-bit threshold, then
     boundary-equal keys until the row is full.

The consumer treats the row as a *set*, so this is selection, never ordering;
candidates whose top 16 key bits tie at the boundary are interchangeable.
Buffers are cached per shape and every launch resets what the next step needs,
so the sequence is CUDA-graph safe.
"""
import torch
import triton
import triton.language as tl

_BUF = {}


@triton.jit
def _keys(SC, CS, BAD, q, SN, CN, C, BS: tl.constexpr, b):
    j = b * BS + tl.arange(0, BS)
    m = j < C
    ids = tl.load(CS + q * CN + j, mask=m, other=0)
    bad = tl.load(BAD + q * CN + j, mask=m, other=1)
    s = tl.load(SC + q * SN + ids, mask=m, other=0.).to(tl.float32)
    ok = m & (bad == 0) & (s == s) & (s < float('inf')) & (s > float('-inf'))
    b32 = s.to(tl.int32, bitcast=True)
    key = b32 ^ ((b32 >> 31) & 0x7FFFFFFF)
    u = tl.where(ok, key.to(tl.int64) + 2147483648, 0)
    return ids, ok, u


@triton.jit
def _cand_hist(SC, CS, BAD, HIST, SN, CN, C, BS: tl.constexpr):
    q = tl.program_id(0)
    _, ok, u = _keys(SC, CS, BAD, q, SN, CN, C, BS, tl.program_id(1))
    h = tl.histogram(tl.where(ok, (u >> 24).to(tl.int32), 0), 256)
    tl.atomic_add(HIST + q * 256 + tl.arange(0, 256), h)


@triton.jit
def _cand_b1(HIST, META, OUT, ON, W, TOPK, BT: tl.constexpr):
    q = tl.program_id(0)
    bi = tl.arange(0, 256)
    h = tl.load(HIST + q * 256 + bi)
    h = tl.where(bi == 0, 0, h)                      # bin 0 holds the invalids
    ge = tl.sum(tl.where(bi[None, :] >= bi[:, None], h[None, :], 0), 1)
    thr = tl.max(tl.where(ge >= W, bi, 0))
    room = W - tl.sum(tl.where(bi > thr, h, 0))
    tl.store(HIST + q * 256 + bi, tl.zeros([256], tl.int32))
    ix = tl.arange(0, 4)
    tl.store(META + q * 4 + ix,
             tl.where(ix == 0, thr, 0) + tl.where(ix == 1, room, 0))
    t = tl.arange(0, BT)
    tl.store(OUT + q * ON + t, tl.full([BT], -1, tl.int64), mask=t < TOPK)


@triton.jit
def _cand_hist2(SC, CS, BAD, META, HIST2, SN, CN, C, BS: tl.constexpr):
    q = tl.program_id(0)
    _, ok, u = _keys(SC, CS, BAD, q, SN, CN, C, BS, tl.program_id(1))
    at = ok & ((u >> 24).to(tl.int32) == tl.load(META + q * 4))
    h = tl.histogram(tl.where(at, ((u >> 16) & 255).to(tl.int32), 0), 256)
    n0 = tl.sum((at & (((u >> 16) & 255) == 0)).to(tl.int32))
    h = tl.where(tl.arange(0, 256) == 0, n0, h)      # drop the masked-out lanes
    tl.atomic_add(HIST2 + q * 256 + tl.arange(0, 256), h)


@triton.jit
def _cand_b2(HIST2, META):
    q = tl.program_id(0)
    bi = tl.arange(0, 256)
    h = tl.load(HIST2 + q * 256 + bi)
    thr1 = tl.load(META + q * 4)
    room1 = tl.load(META + q * 4 + 1)
    ge = tl.sum(tl.where(bi[None, :] >= bi[:, None], h[None, :], 0), 1)
    thr2 = tl.max(tl.where(ge >= room1, bi, 0))
    above2 = tl.sum(tl.where(bi > thr2, h, 0))
    tl.store(HIST2 + q * 256 + bi, tl.zeros([256], tl.int32))
    ix = tl.arange(0, 4)
    tl.store(META + q * 4 + ix,
             tl.where(ix == 0, thr1 * 256 + thr2, 0) +
             tl.where(ix == 1, room1 - above2, 0) +
             tl.where(ix == 3, 0, 0))


@triton.jit
def _cand_cnt(SC, CS, BAD, META, CNT, SN, CN, C, NBP: tl.constexpr,
              BS: tl.constexpr):
    """Count what each program will emit, so emit can skip the atomics."""
    q, p = tl.program_id(0), tl.program_id(1)
    _, ok, u = _keys(SC, CS, BAD, q, SN, CN, C, BS, p)
    thr16 = tl.load(META + q * 4)
    k16 = (u >> 16).to(tl.int32)
    tl.store(CNT + q * 2 * NBP + p, tl.sum((ok & (k16 > thr16)).to(tl.int32)))
    tl.store(CNT + q * 2 * NBP + NBP + p,
             tl.sum((ok & (k16 == thr16)).to(tl.int32)))


@triton.jit
def _cand_scan(CNT, NB, NBP: tl.constexpr):
    """Exclusive prefix sums: program p owns the slots its predecessors left."""
    q = tl.program_id(0)
    i = tl.arange(0, NBP)
    m = i < NB
    hi = tl.load(CNT + q * 2 * NBP + i, mask=m, other=0)
    eq = tl.load(CNT + q * 2 * NBP + NBP + i, mask=m, other=0)
    tl.store(CNT + q * 2 * NBP + i, tl.cumsum(hi, 0) - hi, mask=m)
    tl.store(CNT + q * 2 * NBP + NBP + i, tl.cumsum(eq, 0) - eq, mask=m)


@triton.jit
def _cand_emit(SC, CS, BAD, META, CNT, OUT, SN, CN, ON, C, NBP: tl.constexpr,
               BS: tl.constexpr):
    q, p = tl.program_id(0), tl.program_id(1)
    ids, ok, u = _keys(SC, CS, BAD, q, SN, CN, C, BS, p)
    thr16 = tl.load(META + q * 4)
    room = tl.load(META + q * 4 + 1)
    k16 = (u >> 16).to(tl.int32)
    hi = ok & (k16 > thr16)
    base = tl.load(CNT + q * 2 * NBP + p)
    pos = base + tl.cumsum(hi.to(tl.int32), 0) - 1
    tl.store(OUT + q * ON + pos, ids, mask=hi)
    eq = ok & (k16 == thr16)
    be = tl.load(CNT + q * 2 * NBP + NBP + p)
    pe = be + tl.cumsum(eq.to(tl.int32), 0) - 1
    tl.store(OUT + q * ON + ON - 1 - pe, ids, mask=eq & (pe < room))


def cand_select(scores, csc, bad, topk):
    """scores [Q,N] fp32, csc [Q,C] int64 candidate ids, bad [Q,C] bool.

    Returns [Q,topk] int64 row ids, -1 padded, in candidate order.
    """
    q, c = csc.shape
    width = min(topk, c)
    bs = 1024 if c >= 1024 else triton.next_power_of_2(c)
    nb = triton.cdiv(c, bs)
    nbp = triton.next_power_of_2(nb)
    grid = (q, nb)
    key = (q, topk, c, scores.device)
    buf = _BUF.get(key)
    if buf is None:
        z = lambda w: torch.zeros((q, w), dtype=torch.int32,
                                  device=scores.device)
        buf = (z(256), z(256), z(4), z(2 * nbp))
        _BUF[key] = buf
    h1, h2, meta, cnt = buf
    out = torch.empty((q, topk), dtype=torch.int64, device=scores.device)
    sn, cn, on = scores.stride(0), csc.stride(0), out.stride(0)
    _cand_hist[grid](scores, csc, bad, h1, sn, cn, c, bs, num_warps=4)
    _cand_b1[(q,)](h1, meta, out, on, width, topk,
                   triton.next_power_of_2(topk), num_warps=4)
    _cand_hist2[grid](scores, csc, bad, meta, h2, sn, cn, c, bs, num_warps=4)
    _cand_b2[(q,)](h2, meta, num_warps=4)
    _cand_cnt[grid](scores, csc, bad, meta, cnt, sn, cn, c, nbp, bs,
                    num_warps=4)
    _cand_scan[(q,)](cnt, nb, nbp)
    _cand_emit[grid](scores, csc, bad, meta, cnt, out, sn, cn, on, c, nbp, bs,
                     num_warps=4)
    return out
