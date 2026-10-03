"""Decode hyper-connection gate: two Triton launches per call.

Replaces the ATen chain in ops/prefill/residual.mixes (vector_norm, mul, div,
linear, float, rsqrt, mul) plus the gates kernel: 8 launches -> 2: a (token, column) GEMV grid so
all 2H+H*H rows of fn stream in parallel, then one program per token for
the gates. The mixing
GEMV (x[T, H*D] . fn[W, H*D]^T, W = 2H + H*H) accumulates in FP32 and is
rounded to BF16 exactly where F.linear rounded its output, then scaled by the
RMS of the row and pushed through the same sigmoid/Sinkhorn solve as
ops/prefill/hc_mix._gates. Scale and base are frozen parameters and are
cached as FP32 like hc_mix._const_f32.
"""
import torch
import triton
import triton.language as tl

_CONST = {}
_CFG = (16, 256, 2)  # ksplit, BK, warps


def _const(t, dtype):
    got = _CONST.get((t.data_ptr(), dtype))
    if got is None or got.shape != t.shape:
        got = t.to(dtype).contiguous()
        _CONST[(t.data_ptr(), dtype)] = got
    return got


def _tile(rows):
    """Rows per program.

    acc and ss are each [TP, BK] FP32, so TP is what decides whether the
    program fits in registers: at TP=32 that is 64KB of live state and the
    kernel spills.  Measured on one H100 (K=28672, W=24, BK=256, 2 warps,
    microseconds per launch, bitwise identical across TP):

        rows     tp=2    tp=4    tp=8    tp=16   tp=32
           6     39.5    39.8    38.9    23.9    384.1
          18     23.0    23.0    22.9    23.4    383.2
          24     23.3    22.9    22.8    22.9    386.9
         128     62.3    51.9    53.8    59.8   1646.3
         512    270.9   255.6   237.8   266.7   6696.3
        2048   1067.9  1004.7   938.5  1043.3  26867.0

    next_power_of_2(rows) reached 32 at seventeen rows and stayed there, so
    every call with more than sixteen rows -- MTP verify past b=2, and all
    of prefill -- ran the spilled variant.  Eight rows a program is at or
    inside the noise of the best column everywhere except the very short
    tiles, which prefer sixteen.
    """
    return 16 if rows <= 8 else 8

@triton.jit
def _hc_gemv(X, FN, Z, SS, T, K: tl.constexpr, W: tl.constexpr, TP: tl.constexpr,
             BK: tl.constexpr, KS: tl.constexpr):
    # grid (W, K/KS): each program reads one weight slice once and applies it to
    # all T tokens, so the 1MB fn weight is streamed once per call instead of
    # once per token. Partial sums land in Z[ks, T, W] / SS[ks, T] and are
    # reduced in fixed order by _hc_gates (deterministic, no atomics).
    col = tl.program_id(0)
    ks = tl.program_id(1)
    t = tl.program_id(2) * TP + tl.arange(0, TP)
    tm = t < T
    acc = tl.zeros([TP, BK], tl.float32)
    ss = tl.zeros([TP, BK], tl.float32)
    for k0 in range(ks * KS, (ks + 1) * KS, BK):
        ko = k0 + tl.arange(0, BK)
        f = tl.load(FN + col * K + ko).to(tl.float32)
        x = tl.load(X + t[:, None] * K + ko[None, :], mask=tm[:, None], other=0.).to(tl.float32)
        acc += f[None, :] * x
        ss += x * x
    tl.store(Z + (ks * T + t) * W + col, tl.sum(acc, 1), mask=tm)
    if col == 0:
        tl.store(SS + ks * T + t, tl.sum(ss, 1), mask=tm)


@triton.jit
def _hc_gates(Z, SS, SCALE, BASE, PRE, POST, COMB, norm_eps, hc_eps, T, K,
              W: tl.constexpr, H: tl.constexpr, BH: tl.constexpr, ITERS: tl.constexpr,
              NSPLIT: tl.constexpr):
    row = tl.program_id(0)
    off = tl.arange(0, BH)
    m = off < H
    grid = off[:, None] * H + off[None, :]
    mm = m[:, None] & m[None, :]
    ssum = tl.load(SS + row)
    z0 = tl.load(Z + row * W + off, mask=m, other=0.)
    z1 = tl.load(Z + row * W + H + off, mask=m, other=0.)
    z2 = tl.load(Z + row * W + 2 * H + grid, mask=mm, other=0.)
    for i in tl.static_range(1, NSPLIT):
        ssum += tl.load(SS + i * T + row)
        z0 += tl.load(Z + (i * T + row) * W + off, mask=m, other=0.)
        z1 += tl.load(Z + (i * T + row) * W + H + off, mask=m, other=0.)
        z2 += tl.load(Z + (i * T + row) * W + 2 * H + grid, mask=mm, other=0.)
    # F.linear on bf16 operands rounds its FP32 accumulator to bf16 on store.
    z0 = z0.to(tl.bfloat16).to(tl.float32)
    z1 = z1.to(tl.bfloat16).to(tl.float32)
    z2 = z2.to(tl.bfloat16).to(tl.float32)
    rstd = tl.rsqrt(ssum / K + norm_eps)
    s0 = tl.load(SCALE).to(tl.float32)
    s1 = tl.load(SCALE + 1).to(tl.float32)
    s2 = tl.load(SCALE + 2).to(tl.float32)
    zz = z0 * rstd
    b = tl.load(BASE + off, mask=m, other=0.).to(tl.float32)
    tl.store(PRE + row * H + off, tl.sigmoid(zz * s0 + b) + hc_eps, mask=m)
    zz = z1 * rstd
    b = tl.load(BASE + H + off, mask=m, other=0.).to(tl.float32)
    tl.store(POST + row * H + off, 2. * tl.sigmoid(zz * s1 + b), mask=m)
    v = z2 * rstd
    b = tl.load(BASE + 2 * H + grid, mask=mm, other=0.).to(tl.float32)
    v = tl.where(mm, v * s2 + b, -float('inf'))
    v = tl.exp(v - tl.max(v, 1)[:, None])
    c = v / tl.sum(v, 1)[:, None] + hc_eps
    c = tl.where(mm, c, 0.)
    c = c / (tl.sum(c, 0)[None, :] + hc_eps)
    for _ in tl.static_range(ITERS - 1):
        c = c / (tl.sum(c, 1)[:, None] + hc_eps)
        c = c / (tl.sum(c, 0)[None, :] + hc_eps)
    tl.store(COMB + row * H * H + grid, c, mask=mm)


def hc_pre(x, fn, scale, base, *, norm_eps, hc_eps, iters):
    """x [T, H, D] contiguous CUDA bf16; fn [2H+H*H, H*D]. Returns pre, post [T,H], comb [T,H,H] FP32."""
    rows, h, d = x.shape
    k = h * d
    w = fn.shape[0]
    if fn.shape[1] != k or w != 2 * h + h * h:
        raise ValueError('hc_pre: fn shape does not match the residual width')
    dev = x.device
    ksplit, bk, nw = _CFG
    if k % (ksplit * bk):
        raise ValueError('hc_pre: K=%d not divisible by ksplit*bk' % k)
    tp = _tile(rows)
    z = torch.empty((ksplit, rows, w), device=dev, dtype=torch.float32)
    ss = torch.empty((ksplit, rows), device=dev, dtype=torch.float32)
    pre = torch.empty((rows, h), device=dev, dtype=torch.float32)
    post = torch.empty((rows, h), device=dev, dtype=torch.float32)
    comb = torch.empty((rows, h, h), device=dev, dtype=torch.float32)
    _hc_gemv[(w, ksplit, triton.cdiv(rows, tp))](x, _const(fn, x.dtype), z, ss, rows, k, w, tp, bk,
                          k // ksplit, num_warps=nw)
    _hc_gates[(rows,)](z, ss, _const(scale, torch.float32), _const(base, torch.float32),
                       pre, post, comb, norm_eps, hc_eps, rows, k, w, h,
                       triton.next_power_of_2(h), iters, ksplit, num_warps=1)
    return pre, post, comb


@triton.jit
def _hc_gates_collapse(Z, SS, SCALE, BASE, PRE, POST, COMB,
                       X, PPRE, NW, OUT, norm_eps, hc_eps, T, K,
                       W: tl.constexpr, H: tl.constexpr, BH: tl.constexpr,
                       ITERS: tl.constexpr, NSPLIT: tl.constexpr,
                       D: tl.constexpr, BD: tl.constexpr):
    # One program per token does both halves of the mHC preamble: the gate
    # reduction over Z/SS (this sublayer's mix) and the collapse+RMS of the
    # SAME residual x under the PREVIOUS sublayer's pre gates.  The two halves
    # read x/Z independently -- pre belongs to the next sublayer, so there is
    # no dependency between them -- which is why the launch can be shared.
    row = tl.program_id(0)
    off = tl.arange(0, BH)
    m = off < H
    grid = off[:, None] * H + off[None, :]
    mm = m[:, None] & m[None, :]
    ssum = tl.load(SS + row)
    z0 = tl.load(Z + row * W + off, mask=m, other=0.)
    z1 = tl.load(Z + row * W + H + off, mask=m, other=0.)
    z2 = tl.load(Z + row * W + 2 * H + grid, mask=mm, other=0.)
    for i in tl.static_range(1, NSPLIT):
        ssum += tl.load(SS + i * T + row)
        z0 += tl.load(Z + (i * T + row) * W + off, mask=m, other=0.)
        z1 += tl.load(Z + (i * T + row) * W + H + off, mask=m, other=0.)
        z2 += tl.load(Z + (i * T + row) * W + 2 * H + grid, mask=mm, other=0.)
    z0 = z0.to(tl.bfloat16).to(tl.float32)
    z1 = z1.to(tl.bfloat16).to(tl.float32)
    z2 = z2.to(tl.bfloat16).to(tl.float32)
    rstd = tl.rsqrt(ssum / K + norm_eps)
    s0 = tl.load(SCALE).to(tl.float32)
    s1 = tl.load(SCALE + 1).to(tl.float32)
    s2 = tl.load(SCALE + 2).to(tl.float32)
    zz = z0 * rstd
    b = tl.load(BASE + off, mask=m, other=0.).to(tl.float32)
    tl.store(PRE + row * H + off, tl.sigmoid(zz * s0 + b) + hc_eps, mask=m)
    zz = z1 * rstd
    b = tl.load(BASE + H + off, mask=m, other=0.).to(tl.float32)
    tl.store(POST + row * H + off, 2. * tl.sigmoid(zz * s1 + b), mask=m)
    v = z2 * rstd
    b = tl.load(BASE + 2 * H + grid, mask=mm, other=0.).to(tl.float32)
    v = tl.where(mm, v * s2 + b, -float('inf'))
    v = tl.exp(v - tl.max(v, 1)[:, None])
    c = v / tl.sum(v, 1)[:, None] + hc_eps
    c = tl.where(mm, c, 0.)
    c = c / (tl.sum(c, 0)[None, :] + hc_eps)
    for _ in tl.static_range(ITERS - 1):
        c = c / (tl.sum(c, 1)[:, None] + hc_eps)
        c = c / (tl.sum(c, 0)[None, :] + hc_eps)
    tl.store(COMB + row * H * H + grid, c, mask=mm)
    # collapse + RMS of x[row] under PPRE (previous sublayer's pre gates)
    dof = tl.arange(0, BD)
    dm = dof < D
    acc = tl.zeros([BD], tl.float32)
    for i in tl.static_range(H):
        wgt = tl.load(PPRE + row * H + i).to(tl.float32)
        part = tl.load(X + row * H * D + i * D + dof, mask=dm, other=0.).to(tl.float32)
        acc += wgt * part
    sc = tl.rsqrt(tl.sum(acc * acc, 0) / D + norm_eps)
    gain = tl.load(NW + dof, mask=dm, other=0.).to(tl.float32)
    tl.store(OUT + row * D + dof, (acc * sc * gain).to(OUT.dtype.element_ty), mask=dm)


def hc_pre_collapse(x, fn, scale, base, prev_pre, norm_weight, *, norm_eps, hc_eps, iters):
    """Decode mHC preamble in two launches instead of three.

    Returns (pre, post, comb, hidden): the gates of THIS sublayer plus the
    collapsed+normalised hidden state of the previous sublayer's pre gates,
    which is what ops.prefill.residual.collapse_norm used to produce in a
    third launch reading the same x.
    """
    rows, h, d = x.shape
    k = h * d
    w = fn.shape[0]
    if fn.shape[1] != k or w != 2 * h + h * h:
        raise ValueError('hc_pre_collapse: fn shape does not match the residual width')
    if prev_pre.shape != (rows, h):
        raise ValueError('hc_pre_collapse: pre gates do not match the residual')
    dev = x.device
    ksplit, bk, nw = _CFG
    if k % (ksplit * bk):
        raise ValueError('hc_pre_collapse: K=%d not divisible by ksplit*bk' % k)
    tp = _tile(rows)
    z = torch.empty((ksplit, rows, w), device=dev, dtype=torch.float32)
    ss = torch.empty((ksplit, rows), device=dev, dtype=torch.float32)
    pre = torch.empty((rows, h), device=dev, dtype=torch.float32)
    post = torch.empty((rows, h), device=dev, dtype=torch.float32)
    comb = torch.empty((rows, h, h), device=dev, dtype=torch.float32)
    out = torch.empty((rows, d), device=dev, dtype=x.dtype)
    _hc_gemv[(w, ksplit, triton.cdiv(rows, tp))](x, _const(fn, x.dtype), z, ss, rows, k, w, tp, bk,
                          k // ksplit, num_warps=nw)
    _hc_gates_collapse[(rows,)](z, ss, _const(scale, torch.float32), _const(base, torch.float32),
                       pre, post, comb, x, prev_pre.contiguous().float(), norm_weight, out,
                       norm_eps, hc_eps, rows, k, w, h,
                       triton.next_power_of_2(h), iters, ksplit,
                       d, triton.next_power_of_2(d), num_warps=8)
    return pre, post, comb, out
