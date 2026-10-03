"""Fused hyper-connection gate kernel.

The gates are tiny (H copies plus an H*H mixing matrix) but the Sinkhorn
normalisation needs ``iters`` sequential passes; issued as torch ops that is
~80 kernel launches per call, which dominates short prefill segments. One
Triton program per token keeps the whole solve in registers. Scale and base
stay on the device so no host synchronisation is introduced.
"""
import torch
import triton
import triton.language as tl
_CONST_F32 = {}


def _const_f32(t):
    """The sinkhorn scale and base are frozen parameters.

    Widening and compacting them is a pure function of the parameter, so
    doing it on every call only buys two extra kernels per layer per step.
    Keyed on data_ptr the way the norm leaves already cache their weights.
    """
    got = _CONST_F32.get(t.data_ptr())
    if got is None or got.shape != t.shape:
        got = t.float().contiguous()
        _CONST_F32[t.data_ptr()] = got
    return got




@triton.jit
def _gates(Z, SCALE, BASE, PRE, POST, COMB, hc_eps,
           H: tl.constexpr, BH: tl.constexpr, ITERS: tl.constexpr):
    row = tl.program_id(0)
    off = tl.arange(0, BH)
    m = off < H
    width = 2 * H + H * H
    s0 = tl.load(SCALE).to(tl.float32)
    s1 = tl.load(SCALE + 1).to(tl.float32)
    s2 = tl.load(SCALE + 2).to(tl.float32)

    z = tl.load(Z + row * width + off, mask=m, other=0.).to(tl.float32)
    b = tl.load(BASE + off, mask=m, other=0.).to(tl.float32)
    tl.store(PRE + row * H + off, tl.sigmoid(z * s0 + b) + hc_eps, mask=m)

    z = tl.load(Z + row * width + H + off, mask=m, other=0.).to(tl.float32)
    b = tl.load(BASE + H + off, mask=m, other=0.).to(tl.float32)
    tl.store(POST + row * H + off, 2. * tl.sigmoid(z * s1 + b), mask=m)

    grid = off[:, None] * H + off[None, :]
    mm = m[:, None] & m[None, :]
    v = tl.load(Z + row * width + 2 * H + grid, mask=mm, other=0.).to(tl.float32)
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


def gates(z, scale, base, *, h, hc_eps=1e-6, iters=20):
    """z is [T, 2H+H*H] FP32 contiguous; returns pre, post [T,H], comb [T,H,H]."""
    rows = z.shape[0]
    pre = torch.empty((rows, h), device=z.device, dtype=torch.float32)
    post = torch.empty((rows, h), device=z.device, dtype=torch.float32)
    comb = torch.empty((rows, h, h), device=z.device, dtype=torch.float32)
    _gates[(rows,)](z, _const_f32(scale), _const_f32(base),
                    pre, post, comb, hc_eps, h, triton.next_power_of_2(h),
                    iters, num_warps=1)
    return pre, post, comb
