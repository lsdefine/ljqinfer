"""Decode MoE router: two Triton launches per call.

Replaces ops/prefill/residual.route (x.float, weight cast, F.linear, div,
softplus, sqrt, add bias, topk, gather, sum, div, mul: ~11 launches) with a
(token, column-block) GEMV that also applies temperature and the score
function, then one program per token that runs the biased top-k selection,
gathers, normalises and scales. Outputs match the routed() ABI: prob FP32
[T, topk] contiguous, ids int64 [T, topk].
"""
import torch
import triton
import triton.language as tl

_CONST = {}
_TP_MAX = 32  # token block: caps the unrolled PTX body (7.2 MB -> ~0.4 MB)
_CFG = (10, 32, 64, 4)  # ksplit, BN, BK, warps: BN>=16 so the tile rides tensor cores


def _const(t, dtype):
    got = _CONST.get((t.data_ptr(), dtype))
    if got is None or got.shape != t.shape:
        got = t.to(dtype).contiguous()
        _CONST[(t.data_ptr(), dtype)] = got
    return got


@triton.jit
def _route_gemv(X, W, Z, inv_temp, T, K: tl.constexpr, N: tl.constexpr, TP: tl.constexpr,
                BN: tl.constexpr, BK: tl.constexpr, KS: tl.constexpr, SCORE: tl.constexpr):
    # grid (N/BN, K/KS): each program streams a [BN, KS] weight tile exactly once
    # and applies it to every token; partial sums land in Z[split, T, N] and are
    # reduced in fixed order by _route_topk (deterministic, no atomics).
    n = tl.program_id(0) * BN + tl.arange(0, BN)
    ks = tl.program_id(1)
    # Pad the token axis to TP = max(16, next_pow2(rows)): 16 is the smallest
    # M the MMA shape wants, and one TP=32 launch beats two TP=16 launches at
    # B=4 (T=24).  The outer-product form spilled [TP, BN, BK] floats through
    # registers and ran the gate GEMV at ~0.19 TB/s.
    t = tl.arange(0, TP)
    mn = n < N
    mt = t < T
    acc = tl.zeros((TP, BN), dtype=tl.float32)
    for k0 in range(ks * KS, (ks + 1) * KS, BK):
        k = k0 + tl.arange(0, BK)
        x = tl.load(X + t[:, None] * K + k[None, :], mask=mt[:, None], other=0.)
        w = tl.load(W + n[:, None] * K + k[None, :], mask=mn[:, None], other=0.)
        acc += tl.dot(x, tl.trans(w), out_dtype=tl.float32)
    tl.store(Z + ks * T * N + t[:, None] * N + n[None, :], acc, mask=mt[:, None] & mn[None, :])


@triton.jit
def _route_topk(Z, BIAS, PROB, IDS, scale, inv_temp, T, N: tl.constexpr, BNP: tl.constexpr, NSPLIT: tl.constexpr, SCORE: tl.constexpr,
                TOPK: tl.constexpr, NORMALIZE: tl.constexpr, SOFTMAX: tl.constexpr):
    row = tl.program_id(0)
    n = tl.arange(0, BNP)
    m = n < N
    z = tl.zeros((BNP,), dtype=tl.float32)
    for sk in tl.static_range(NSPLIT):
        z += tl.load(Z + sk * T * N + row * N + n, mask=m, other=0.)
    z = z * inv_temp
    if SCORE == 0:  # sqrtsoftplus
        z = tl.sqrt(tl.log(1. + tl.exp(z)))
    elif SCORE == 1:  # sigmoid
        z = tl.sigmoid(z)
    if SOFTMAX:
        e = tl.exp(z - tl.max(z, 0))
        e = tl.where(m, e, 0.)
        z = e / tl.sum(e, 0)
    b = tl.load(BIAS + n, mask=m, other=0.)
    s = tl.where(m, z + b, -float('inf'))
    total = 0.
    for i in tl.static_range(TOPK):
        best = tl.max(s, 0)
        idx = tl.min(tl.where(s == best, n, N), 0)
        p = tl.sum(tl.where(n == idx, z, 0.), 0)
        tl.store(IDS + row * TOPK + i, idx.to(tl.int64))
        tl.store(PROB + row * TOPK + i, p)
        total += p
        s = tl.where(n == idx, -float('inf'), s)
    if NORMALIZE and TOPK > 1:
        inv = 1. / (total + 1e-20)
        p = tl.load(PROB + row * TOPK + tl.arange(0, 8), mask=tl.arange(0, 8) < TOPK, other=0.)
        tl.store(PROB + row * TOPK + tl.arange(0, 8), p * inv * scale, mask=tl.arange(0, 8) < TOPK)
    else:
        p = tl.load(PROB + row * TOPK + tl.arange(0, 8), mask=tl.arange(0, 8) < TOPK, other=0.)
        tl.store(PROB + row * TOPK + tl.arange(0, 8), p * scale, mask=tl.arange(0, 8) < TOPK)


_SCORES = {'sqrtsoftplus': 0, 'sigmoid': 1, 'softmax': 2}


def route(x, weight, bias, *, topk, temperature, scale, normalize=True, score='sqrtsoftplus'):
    """x [T, K] (any float dtype); weight [N, K]; bias [N]. Returns (prob f32, ids i64)."""
    assert topk <= 8
    x = x.contiguous()
    rows, k = x.shape
    if rows > _TP_MAX:
        # Each token's routing is independent, so blocking the token dimension
        # is bit-exact; it keeps TP a small constant instead of next_pow2(rows),
        # which is what makes the unrolled [TP, BN, BK] body compile in seconds.
        prob = torch.empty((rows, topk), device=x.device, dtype=torch.float32)
        ids = torch.empty((rows, topk), device=x.device, dtype=torch.int64)
        for i in range(0, rows, _TP_MAX):
            p, d = route(x[i:i + _TP_MAX], weight, bias, topk=topk,
                         temperature=temperature, scale=scale,
                         normalize=normalize, score=score)
            prob[i:i + p.shape[0]] = p
            ids[i:i + d.shape[0]] = d
        return prob, ids
    n = weight.shape[0]
    dev = x.device
    ksplit, bn, bk, nw = _CFG
    assert k % (ksplit * bk) == 0, k
    tp = max(16, triton.next_power_of_2(rows))
    z = torch.empty((ksplit, rows, n), device=dev, dtype=torch.float32)
    prob = torch.empty((rows, topk), device=dev, dtype=torch.float32)
    ids = torch.empty((rows, topk), device=dev, dtype=torch.int64)
    _route_gemv[(triton.cdiv(n, bn), ksplit)](x, _const(weight, x.dtype), z, 1. / temperature,
                                              rows, k, n, tp, bn, bk, k // ksplit,
                                              _SCORES[score], num_warps=nw)
    _route_topk[(rows,)](z, _const(bias, torch.float32), prob, ids, float(scale),
                         1. / temperature, rows, n, triton.next_power_of_2(n), ksplit,
                         _SCORES[score], topk, bool(normalize), score == 'softmax', num_warps=4)
    return prob, ids
