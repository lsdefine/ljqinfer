"""Split Gumbel-max adapted from ljqinfer_dsv41f_tp8 ops/decode/argmax.py.
No penalties; two launches, no full-vocabulary intermediate or extra collective.
"""
import secrets
import torch
import triton
import triton.language as tl

_BIG = tl.constexpr(2147483647)
_SPLIT = 128
_STATES = {}

@triton.jit
def _part_gumbel(X, MV, MI, SEED, TEMP, N, NFULL, OFF, CH: tl.constexpr,
                 BLK: tl.constexpr, SPLIT: tl.constexpr):
    r = tl.program_id(0)
    s = tl.program_id(1)
    seed = tl.load(SEED)
    # Row-wise temperature supports mixed requests. This sampler runs after
    # verify graph replay; graph capture is tested only for fixed temperature.
    t = tl.load(TEMP + r)
    best = tl.full((BLK,), float('-inf'), tl.float32)
    bidx = tl.full((BLK,), 2147483647, tl.int32)
    for o in range(0, CH, BLK):
        i = s * CH + o + tl.arange(0, BLK)
        v = tl.load(X + r * N + i, mask=i < N, other=float('-inf')).to(tl.float32)
        # Global vocabulary counters; each rank also has an independent seed.
        # Distribution matches unsharded sampling, not its exact draw sequence.
        u = tl.maximum(tl.rand(seed, r * NFULL + i + OFF), 1.0e-7)
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



def sample_pairs(logits, temps, rank, world):
    rows, width = logits.shape
    key = (logits.device, rank, world)
    state = _STATES.get(key)
    if state is None:
        # Independent shard streams, not torch's possibly identical TP seeds.
        seed = torch.tensor([secrets.randbits(31)], dtype=torch.int64, device=logits.device)
        state = _STATES[key] = [seed, None, None, {}]
    expanded = tuple(t for t in temps for _ in range(rows // len(temps)))
    if state[1] != expanded:
        state[2] = torch.tensor(expanded, dtype=torch.float32, device=logits.device)
        state[1] = expanded
    buffers = state[3].get(rows)
    if buffers is None:
        buffers = (torch.empty((rows, _SPLIT), device=logits.device, dtype=torch.float32),
                   torch.empty((rows, _SPLIT), device=logits.device, dtype=torch.int32),
                   torch.empty((rows, 2), device=logits.device, dtype=torch.float32))
        state[3][rows] = buffers
    mv, mi, pairs = buffers
    chunk = triton.cdiv(width, _SPLIT)
    block = min(triton.next_power_of_2(chunk), 1024)
    _part_gumbel[(rows, _SPLIT)](logits, mv, mi, state[0], state[2], width,
        width * world, rank * width, chunk, block, _SPLIT, num_warps=4)
    _join_pair[(rows,)](mv, mi, pairs, state[0], _SPLIT, True, num_warps=1)
    return pairs
