"""Eager Tensor compositions over the registered optimized NPU kernels."""

import torch

import torch.nn.functional as F

from .native import ops as native


def _cast_w(w, dtype):
    return w if w.dtype == dtype else w.to(dtype)


def rms(x, weight=None, eps=1e-6):
    """RMSNorm accumulated in FP32 with a single rounding on store."""
    y = x.float()
    y = y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + eps)
    if weight is not None:
        y = y * weight.float()
    return y.to(x.dtype)


collapse = native.hc_collapse


def collapse_norm(x, pre, weight, eps=1e-6):
    return rms(collapse(x, pre), weight, eps)


def mixes(x, fn, scale, base, *, norm_eps=1e-6, hc_eps=1e-6, iters=20):
    """Hyper-connection gates (pre[T,h], post[T,h], comb[T,h,h]).

    The BF16->FP32 cast with its RMS statistic and the Sinkhorn sweep are the
    two released kernels; both take the live row count, so only the rows that
    the caller passes in are ever touched.
    """
    t, h, d = x.shape
    flat, stats = native.hc_cast_stats(x, norm_eps)
    z = (flat @ fn.t()) * stats
    pre = torch.sigmoid(z[:, :h] * scale[0] + base[:h]) + hc_eps
    post = torch.sigmoid(z[:, h:2 * h] * scale[1] + base[h:2 * h]) * 2
    comb = (z[:, 2 * h:] * scale[2] + base[2 * h:]).view(t, h, h)
    comb = torch.softmax(comb, -1) + hc_eps
    comb = comb / (comb.sum(-2, keepdim=True) + hc_eps)
    if iters > 1:
        comb = native.hc_sinkhorn(comb, iters - 1)
    return pre, post, comb

expand = native.hc_expand


def engram_gate(x, kv, weight, rotation, eps=1e-20):
    """Gate in the original basis; weight is the released FP32 q*k product.

    Only the gate query is rotated back through the 32x32 block: V inside kv is
    already rotated, so the residual stream adds the untouched value.
    """
    n, c, d = x.shape
    query = (x.float().reshape(-1, 32) @ rotation.t()).view(n, c, d)
    key, value = kv.float().split([c * d, d], -1)
    key = key.view(n, c, d)
    rstd = (torch.rsqrt(query.pow(2).mean(-1, keepdim=True) + eps)
            * torch.rsqrt(key.pow(2).mean(-1, keepdim=True) + eps))
    dot = (query * weight * key).sum(-1, keepdim=True) * rstd * d**-0.5
    magnitude = dot.abs().clamp_min(1e-6).sqrt()
    # copysign falls back to the CPU on the NPU; read the IEEE sign bit instead,
    # which also keeps -0. on the negative side the way copysign does.
    negative = dot.contiguous().view(torch.int32) < 0
    gate = torch.sigmoid(torch.where(negative, -magnitude, magnitude))
    return (x.float() + gate * value.unsqueeze(-2)).to(x.dtype)


def swiglu(gate, up, limit, route_weight=None):
    g, u = gate.float(), up.float()
    if limit > 0:
        g, u = g.clamp(max=limit), u.clamp(-limit, limit)
    y = F.silu(g) * u
    if route_weight is not None:
        y = y * route_weight
    return y.to(gate.dtype)


def route(x, weight, bias, *, topk, temperature, scale,
          normalize=True, score='sqrtsoftplus'):
    z = F.linear(x.float(), _cast_w(weight, torch.float32)) / temperature
    if score == 'softmax':
        z = z.softmax(-1)
    elif score == 'sigmoid':
        z = z.sigmoid()
    elif score == 'sqrtsoftplus':
        z = F.softplus(z).sqrt()
    else:
        raise ValueError(score)
    ids = (z + bias).topk(topk, dim=-1).indices
    p = z.gather(-1, ids)
    if normalize and topk > 1:
        p = p / (p.sum(-1, keepdim=True) + 1e-20)
    return p * scale, ids


def moe_add(routed, shared, dtype=torch.bfloat16):
    """Released finish: shared is rounded to BF16 before the FP32 accumulate."""
    return (routed.float() + shared.to(dtype).float()).to(dtype)
