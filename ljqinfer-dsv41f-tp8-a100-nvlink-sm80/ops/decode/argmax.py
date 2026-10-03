"""Row-wise argmax for the decode verify window.

torch.argmax over a [T, V] logit block puts one program on each of the six
rows, so 3MB of vocabulary streamed at ~0.08 TB/s and the reduction cost as
much as a real projection. The vocabulary is split across programs instead
and the per-split winners are folded by a second, tiny launch. Ties resolve
to the lowest index, which is what the verify comparison assumes.

Under TP the vocabulary is already sliced across the eight ranks, so the
caller can hand in its own slice and skip rebuilding the whole row: pass the
``VocabShard`` it belongs to and the fold runs twice, once over the local
splits and once over the eight rank winners. Only two floats per row cross
the wire instead of a full vocabulary, and the answer is unchanged because a
max is associative and the tie rule stays "lowest global index".
"""
import torch
import triton
import triton.language as tl

_SPLIT = 32
_BLK = 1024
_BIG = tl.constexpr(2147483647)  # triton jit only reads constexpr globals


class VocabShard:
    """One rank's vocabulary slice: rows [offset, offset + local) of `full`.

    `parallel` only has to expose `world` and `gather_rows`, the same
    all_gather_into_tensor wrapper the trunk uses for row-major gathers.
    """
    __slots__ = ('offset', 'full', 'parallel')

    def __init__(self, offset, full, parallel):
        self.offset, self.full, self.parallel = int(offset), int(full), parallel


@triton.jit
def _part(X, MV, MI, N, OFF, CH: tl.constexpr, BLK: tl.constexpr, SPLIT: tl.constexpr):
    r = tl.program_id(0)
    s = tl.program_id(1)
    best = tl.full((BLK,), float('-inf'), tl.float32)
    bidx = tl.full((BLK,), _BIG, tl.int32)
    for o in range(0, CH, BLK):
        i = s * CH + o + tl.arange(0, BLK)
        v = tl.load(X + r * N + i, mask=i < N, other=float('-inf')).to(tl.float32)
        take = v > best
        # indices are emitted in global vocabulary space so the cross-rank
        # fold can break ties the same way a single-rank scan would
        bidx = tl.where(take, (i + OFF).to(tl.int32), bidx)
        best = tl.where(take, v, best)
    m = tl.max(best, 0)
    tl.store(MV + r * SPLIT + s, m)
    tl.store(MI + r * SPLIT + s, tl.min(tl.where(best == m, bidx, _BIG), 0))


@triton.jit
def _join(MV, MI, OUT, SPLIT: tl.constexpr):
    r = tl.program_id(0)
    s = tl.arange(0, SPLIT)
    v = tl.load(MV + r * SPLIT + s)
    m = tl.max(v, 0)
    idx = tl.min(tl.where(v == m, tl.load(MI + r * SPLIT + s), _BIG), 0)
    tl.store(OUT + r, idx.to(tl.int64))


@triton.jit
def _join_pair(MV, MI, OUT, SEED, SPLIT: tl.constexpr, BUMP: tl.constexpr):
    """Fold the local splits into one (value, index) pair per row.

    The pair travels as two float32 lanes: a vocabulary index is below 2**24,
    so float32 carries it exactly and one gather moves both halves.
    """
    r = tl.program_id(0)
    s = tl.arange(0, SPLIT)
    v = tl.load(MV + r * SPLIT + s)
    m = tl.max(v, 0)
    idx = tl.min(tl.where(v == m, tl.load(MI + r * SPLIT + s), _BIG), 0)
    tl.store(OUT + r * 2, m)
    tl.store(OUT + r * 2 + 1, idx.to(tl.float32))
    if BUMP:
        if r == 0:
            tl.store(SEED, tl.load(SEED) + 1)


@triton.jit
def _join_ranks(P, OUT, T, WORLD: tl.constexpr):
    """Fold the eight gathered rank winners into the row's global argmax."""
    r = tl.program_id(0)
    w = tl.arange(0, WORLD)
    off = w * (T * 2) + r * 2
    v = tl.load(P + off)
    idx = tl.load(P + off + 1)
    m = tl.max(v, 0)
    best = tl.min(tl.where(v == m, idx, float('inf')), 0)
    tl.store(OUT + r, best.to(tl.int64))


def _seed_for(device):
    seed = _SEED.get(device)
    if seed is None:
        seed = torch.zeros((), device=device, dtype=torch.int32)
        _SEED[device] = seed
    return seed


def _fold_ranks(pair, rows, shard):
    """gather [T,2] from every rank and reduce to one int64 id per row."""
    par = shard.parallel
    buf = torch.empty((par.world * rows, 2), device=pair.device, dtype=torch.float32)
    par.gather_rows(pair, buf)
    out = torch.empty((rows,), device=pair.device, dtype=torch.int64)
    _join_ranks[(rows,)](buf, out, rows, par.world, num_warps=1)
    return out


def argmax_rows(x, shard=None):
    """x [T, V] -> int64 [T]; same answer as x.argmax(-1) including ties.

    With `shard`, x holds only that rank's vocabulary slice and every rank
    returns the same global ids.
    """
    x = x.contiguous()
    rows, n = x.shape
    ch = triton.cdiv(triton.cdiv(n, _SPLIT), _BLK) * _BLK
    mv = torch.empty((rows, _SPLIT), device=x.device, dtype=torch.float32)
    mi = torch.empty((rows, _SPLIT), device=x.device, dtype=torch.int32)
    off = 0 if shard is None else shard.offset
    _part[(rows, _SPLIT)](x, mv, mi, n, off, ch, _BLK, _SPLIT, num_warps=4)
    if shard is None:
        out = torch.empty((rows,), device=x.device, dtype=torch.int64)
        _join[(rows,)](mv, mi, out, _SPLIT, num_warps=1)
        return out
    pair = torch.empty((rows, 2), device=x.device, dtype=torch.float32)
    _join_pair[(rows,)](mv, mi, pair, _seed_for(x.device), _SPLIT, False, num_warps=1)
    return _fold_ranks(pair, rows, shard)


# --- sampling ---------------------------------------------------------------
# The released sampler spends four launches per draft position (softmax,
# exponential_, div_, argmax) on a [T, V] block; the argmax alone ran at
# ~0.08 TB/s. Gumbel-max only needs the *order* of log p - log E, and the
# softmax denominator is a per-row constant, so the whole chain collapses into
# the split argmax below: score = logit + temperature*gumbel(u), u from a
# counter-based RNG whose seed lives in a device tensor (bumped inside the
# join kernel so a captured graph keeps drawing fresh numbers).
_GSPLIT = 128
_SEED = {}


@triton.jit
def _part_gumbel(X, MV, MI, SEED, TEMP, N, NFULL, OFF, CH: tl.constexpr,
                 BLK: tl.constexpr, SPLIT: tl.constexpr):
    r = tl.program_id(0)
    s = tl.program_id(1)
    seed = tl.load(SEED)
    # Temperature is read from a device tensor, not baked in as a launch
    # argument: the verify graph is captured once and a scalar would freeze
    # whatever value the capture happened to see.  Row-wise, so one batch can
    # mix temperatures.
    t = tl.load(TEMP + r)
    best = tl.full((BLK,), float('-inf'), tl.float32)
    bidx = tl.full((BLK,), 2147483647, tl.int32)
    for o in range(0, CH, BLK):
        i = s * CH + o + tl.arange(0, BLK)
        v = tl.load(X + r * N + i, mask=i < N, other=float('-inf')).to(tl.float32)
        # the RNG is addressed in global vocabulary space, so a sharded draw
        # reproduces the unsharded one draw for draw
        u = tl.rand(seed, r * NFULL + i + OFF)
        # -log(-log u) is the Gumbel sample; u is in (0, 1) so both logs are safe.
        # scaling the gumbel instead of the logit keeps T == 0 meaningful:
        # argmax(x + T*g) == argmax(x/T + g) for T > 0, and is a plain argmax
        # at T == 0.
        v = tl.where(i < N, v - t * tl.log(-tl.log(u)), float('-inf'))
        take = v > best
        bidx = tl.where(take, (i + OFF).to(tl.int32), bidx)
        best = tl.where(take, v, best)
    m = tl.max(best, 0)
    tl.store(MV + r * SPLIT + s, m)
    tl.store(MI + r * SPLIT + s, tl.min(tl.where(best == m, bidx, 2147483647), 0))


@triton.jit
def _join_bump(MV, MI, OUT, SEED, SPLIT: tl.constexpr):
    r = tl.program_id(0)
    s = tl.arange(0, SPLIT)
    v = tl.load(MV + r * SPLIT + s)
    m = tl.max(v, 0)
    idx = tl.min(tl.where(v == m, tl.load(MI + r * SPLIT + s), 2147483647), 0)
    tl.store(OUT + r, idx.to(tl.int64))
    if r == 0:
        tl.store(SEED, tl.load(SEED) + 1)


def temp_rows(rows, value, device):
    """A [rows] temperature buffer to hand `sample_rows` inside a graph.

    Captured graphs read the temperature through this tensor, so the caller
    keeps the handle and writes it between replays.
    """
    return torch.full((rows,), float(value), device=device, dtype=torch.float32)


def sample_rows(x, temperature, shard=None):
    """x [T, V] -> int64 [T]. Greedy at temperature 0, else Gumbel-max.

    `temperature` is either a python float or a [T] float32 device tensor; the
    tensor form is what a captured graph needs, since a float is frozen into
    the launch at capture time.
    """
    tensor_temp = torch.is_tensor(temperature)
    if not tensor_temp and temperature == 0:
        return argmax_rows(x, shard)
    x = x.contiguous()
    rows, n = x.shape
    if tensor_temp:
        assert temperature.dtype == torch.float32 and temperature.numel() >= rows, \
            'temperature must be a float32 tensor with one entry per row'
        temp = temperature.contiguous()
    else:
        temp = temp_rows(rows, temperature, x.device)
    # one program per 128th of the vocabulary keeps every SM busy even for the
    # single-row draft blocks; the join then folds the partial winners.
    ch = triton.cdiv(triton.cdiv(n, _GSPLIT), _BLK) * _BLK
    seed = _seed_for(x.device)
    mv = torch.empty((rows, _GSPLIT), device=x.device, dtype=torch.float32)
    mi = torch.empty((rows, _GSPLIT), device=x.device, dtype=torch.int32)
    nfull = n if shard is None else shard.full
    off = 0 if shard is None else shard.offset
    _part_gumbel[(rows, _GSPLIT)](x, mv, mi, seed, temp, n, nfull, off,
                                  ch, _BLK, _GSPLIT, num_warps=4)
    if shard is None:
        out = torch.empty((rows,), device=x.device, dtype=torch.int64)
        _join_bump[(rows,)](mv, mi, out, seed, _GSPLIT, num_warps=1)
        return out
    pair = torch.empty((rows, 2), device=x.device, dtype=torch.float32)
    _join_pair[(rows,)](mv, mi, pair, seed, _GSPLIT, True, num_warps=1)
    return _fold_ranks(pair, rows, shard)
