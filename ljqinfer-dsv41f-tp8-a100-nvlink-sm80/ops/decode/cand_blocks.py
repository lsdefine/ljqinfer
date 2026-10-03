"""Fused candidate-block selection for the decode candidate source layer.

The source layer used to build its block prefilter with an ATen chain --
pad, unflatten, amax, arange, masked_fill, topk, gather, isfinite,
masked_fill, cat, arange, mul, flatten, masked_fill -- and then hand the
result to ``prepare_candidates``, which added a second ``torch.sort``.  On a
384k window that is a cub radix sort over [t, 12288] plus another over
[t, 16384] plus ~20 elementwise launches, and it showed up as the single
most expensive layer in the decode trace.

This module does the same selection in five Triton launches using the
histogram threshold trick already used by ``cand_select``: two passes over
the block maxima narrow a 16-bit threshold on the order-preserving score
key, then one emit pass expands the surviving blocks straight into the
(ids, invalid) pair the attention kernel consumes.  No ordering is produced
because the consumer treats each row as a set, and no duplicate mask is
needed because the forced newest block is excluded from the selection.

Buffers are cached per shape and each launch clears what the next step
needs, so the sequence is CUDA-graph safe.
"""
import torch
import triton
import triton.language as tl

_BUF = {}


@triton.jit
def _bkey(SC, POS, KEY, SN, KN, N, NBLK, RATIO, MINK,
          BLK: tl.constexpr, BS: tl.constexpr):
    """Materialise one order-preserving int32 key per block.

    The key is written to scratch rather than recomputed by every pass:
    the later passes then read a plain int32 vector, which also keeps
    ``tl.histogram`` away from a value derived from a ``tl.max`` reduction
    (that combination miscounts on triton 3.4).  Invalid blocks get MINK.
    """
    q = tl.program_id(0)
    blk = tl.program_id(1) * BS + tl.arange(0, BS)
    j = blk[:, None] * BLK + tl.arange(0, BLK)[None, :]
    s = tl.load(SC + q * SN + j, mask=j < N, other=float('-inf')).to(tl.float32)
    m = tl.max(s, 1)
    lens = (tl.load(POS + q) + 1) // RATIO
    newest = (lens - 1) // BLK
    ok = (blk < newest) & (m == m) & (m < float('inf')) & (m > float('-inf'))
    b32 = m.to(tl.int32, bitcast=True)
    key = b32 ^ ((b32 >> 31) & 0x7FFFFFFF)
    tl.store(KEY + q * KN + blk, tl.where(ok, key, MINK), mask=blk < NBLK)


@triton.jit
def _bh1(KEY, HIST, KN, NBLK, MINK, BS: tl.constexpr):
    q = tl.program_id(0)
    blk = tl.program_id(1) * BS + tl.arange(0, BS)
    key = tl.load(KEY + q * KN + blk, mask=blk < NBLK, other=MINK)
    ok = key != MINK
    b1 = (key >> 24) + 128
    h = tl.histogram(tl.where(ok, b1, 0), 256)
    n0 = tl.sum((ok & (b1 == 0)).to(tl.int32))
    h = tl.where(tl.arange(0, 256) == 0, n0, h)   # bin 0 is a real score bin
    tl.atomic_add(HIST + q * 256 + tl.arange(0, 256), h)


@triton.jit
def _bs1(HIST, META, W):
    q = tl.program_id(0)
    bi = tl.arange(0, 256)
    h = tl.load(HIST + q * 256 + bi)
    ge = tl.sum(tl.where(bi[None, :] >= bi[:, None], h[None, :], 0), 1)
    thr = tl.max(tl.where(ge >= W, bi, 0))
    room = W - tl.sum(tl.where(bi > thr, h, 0))
    tl.store(HIST + q * 256 + bi, tl.zeros([256], tl.int32))
    ix = tl.arange(0, 4)
    tl.store(META + q * 4 + ix,
             tl.where(ix == 0, thr, 0) + tl.where(ix == 1, room, 0))


@triton.jit
def _bh2(KEY, META, HIST2, KN, NBLK, MINK, BS: tl.constexpr):
    q = tl.program_id(0)
    blk = tl.program_id(1) * BS + tl.arange(0, BS)
    key = tl.load(KEY + q * KN + blk, mask=blk < NBLK, other=MINK)
    at = (key != MINK) & (((key >> 24) + 128) == tl.load(META + q * 4))
    b2 = (key >> 16) & 255
    h = tl.histogram(tl.where(at, b2, 0), 256)
    n0 = tl.sum((at & (b2 == 0)).to(tl.int32))
    h = tl.where(tl.arange(0, 256) == 0, n0, h)
    tl.atomic_add(HIST2 + q * 256 + tl.arange(0, 256), h)


@triton.jit
def _bs2(HIST2, META):
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
             tl.where(ix == 1, room1 - above2, 0))


@triton.jit
def _bcnt(KEY, META, CNT, KN, NBLK, MINK, NBP: tl.constexpr, BS: tl.constexpr):
    """Count the survivors each program will emit, so emit needs no atomics."""
    q, p = tl.program_id(0), tl.program_id(1)
    blk = p * BS + tl.arange(0, BS)
    key = tl.load(KEY + q * KN + blk, mask=blk < NBLK, other=MINK)
    ok = key != MINK
    k16 = (key >> 16) + 32768
    thr16 = tl.load(META + q * 4)
    tl.store(CNT + q * 2 * NBP + p, tl.sum((ok & (k16 > thr16)).to(tl.int32)))
    tl.store(CNT + q * 2 * NBP + NBP + p,
             tl.sum((ok & (k16 == thr16)).to(tl.int32)))


@triton.jit
def _bscan(CNT, NB, NBP: tl.constexpr):
    """Exclusive prefix sum over the per-program counts, one row per query."""
    q = tl.program_id(0)
    i = tl.arange(0, NBP)
    m = i < NB
    hi = tl.load(CNT + q * 2 * NBP + i, mask=m, other=0)
    eq = tl.load(CNT + q * 2 * NBP + NBP + i, mask=m, other=0)
    tl.store(CNT + q * 2 * NBP + i, tl.cumsum(hi, 0) - hi, mask=m)
    tl.store(CNT + q * 2 * NBP + NBP + i, tl.cumsum(eq, 0) - eq, mask=m)


@triton.jit
def _bemit(KEY, POS, META, CNT, CS, BAD, KN, CN, NBLK, RATIO, TAKE, MINK,
           BLK: tl.constexpr, NBP: tl.constexpr, BS: tl.constexpr):
    """Expand the surviving blocks into row ids and their invalid mask."""
    q, p = tl.program_id(0), tl.program_id(1)
    blk = p * BS + tl.arange(0, BS)
    key = tl.load(KEY + q * KN + blk, mask=blk < NBLK, other=MINK)
    ok = key != MINK
    thr16 = tl.load(META + q * 4)
    room = tl.load(META + q * 4 + 1)
    k16 = (key >> 16) + 32768
    # Both ranks come from the prescan, so a block's slot depends only on its
    # index -- the scheduler cannot reorder the row or change which tied
    # blocks survive the ``room`` cut.
    hi = ok & (k16 > thr16)
    base = tl.load(CNT + q * 2 * NBP + p)
    slot = base + tl.cumsum(hi.to(tl.int32), 0) - 1
    eq = ok & (k16 == thr16)
    be = tl.load(CNT + q * 2 * NBP + NBP + p)
    pe = be + tl.cumsum(eq.to(tl.int32), 0) - 1
    slot = tl.where(eq, TAKE - 1 - pe, slot)
    keep = (hi & (slot < TAKE)) | (eq & (pe < room))
    lens = (tl.load(POS + q) + 1) // RATIO
    o = slot[:, None] * BLK + tl.arange(0, BLK)[None, :]
    row = blk[:, None] * BLK + tl.arange(0, BLK)[None, :]
    tl.store(CS + q * CN + o, row, mask=keep[:, None])
    tl.store(BAD + q * CN + o, (row >= lens).to(tl.int1), mask=keep[:, None])


@triton.jit
def _bnew(POS, CS, BAD, CN, TAKE, RATIO, BLK: tl.constexpr):
    """Force the newest block into the last slot; it is never selected above."""
    q = tl.program_id(0)
    lens = (tl.load(POS + q) + 1) // RATIO
    newest = (lens - 1) // BLK
    i = tl.arange(0, BLK)
    row = newest * BLK + i
    bad = (row >= lens) | (lens <= 0)
    tl.store(CS + q * CN + TAKE * BLK + i, tl.where(bad, 0, row))
    tl.store(BAD + q * CN + TAKE * BLK + i, bad.to(tl.int1))


def candidate_prep(score, pos, ratio, block_size, top_blocks):
    """Fused replacement for ``candidate_rows`` + ``prepare_candidates``.

    Returns (ids [t, top_blocks*block_size] int64, invalid [same] bool).
    Rows are emitted in block-index order and ties at the cut are broken the
    same way every step: attention sums over these rows in fp32, so a
    scheduler-dependent order would be visible in the logits.
    """
    t, n = score.shape
    nblk = (n + block_size - 1) // block_size
    take = min(top_blocks - 1, nblk)
    c = top_blocks * block_size
    bs = 512 if nblk >= 512 else triton.next_power_of_2(nblk)
    nb = triton.cdiv(nblk, bs)
    grid = (t, nb)
    mink = -2147483648
    key = (t, c, nblk, score.device)
    buf = _BUF.get(key)
    if buf is None:
        z = lambda w, d: torch.zeros((t, w), dtype=d, device=score.device)
        buf = (z(256, torch.int32), z(256, torch.int32), z(4, torch.int32),
               z(nb * bs, torch.int32), z(2 * triton.next_power_of_2(nb),
                                          torch.int32))
        _BUF[key] = buf
    h1, h2, meta, kb, cnt = buf
    nbp = triton.next_power_of_2(nb)
    cs = torch.zeros((t, c), dtype=torch.int64, device=score.device)
    bad = torch.ones((t, c), dtype=torch.bool, device=score.device)
    sn, cn, kn = score.stride(0), cs.stride(0), kb.stride(0)
    _bkey[grid](score, pos, kb, sn, kn, n, nblk, ratio, mink,
                block_size, bs, num_warps=4)
    _bh1[grid](kb, h1, kn, nblk, mink, bs, num_warps=4)
    _bs1[(t,)](h1, meta, take)
    _bh2[grid](kb, meta, h2, kn, nblk, mink, bs, num_warps=4)
    _bs2[(t,)](h2, meta)
    _bcnt[grid](kb, meta, cnt, kn, nblk, mink, nbp, bs, num_warps=4)
    _bscan[(t,)](cnt, nb, nbp)
    _bemit[grid](kb, pos, meta, cnt, cs, bad, kn, cn, nblk, ratio, take, mink,
                 block_size, nbp, bs, num_warps=4)
    _bnew[(t,)](pos, cs, bad, cn, take, ratio, block_size)
    return cs, bad
