"""Eager Tensor compositions over the registered optimized NPU kernels."""

import torch
import torch.nn.functional as F

from .native import ops as native


def _cast_w(w, dtype):
    return w if w.dtype == dtype else w.to(dtype)


def rms(x, weight=None, eps=1e-06):
    """RMSNorm accumulated in FP32 with a single rounding on store."""
    # Only owned temporaries are mutated; FP32 inputs may alias .float().
    y = x.float()
    variance = y.square().mean(-1, keepdim=True)
    variance.add_(eps).rsqrt_()
    out = y * variance
    if weight is not None:
        out.mul_(weight.float())
    return out.to(x.dtype)


collapse = native.hc_collapse


def collapse_norm(x, pre, weight, eps=1e-6):
    return rms(collapse(x, pre), weight, eps)


def mixes(x, fn, scale, base, *, norm_eps=1e-06, hc_eps=1e-06, iters=20):
    """Hyper-connection gates (pre[T,h], post[T,h], comb[T,h,h]).

    The BF16->FP32 cast with its RMS statistic and the Sinkhorn sweep are the
    two released kernels; both take the live row count, so only the rows that
    the caller passes in are ever touched.
    """
    t, h, d = x.shape
    flat, stats = native.hc_cast_stats(x, norm_eps)
    z = flat @ fn.t()
    z.mul_(stats)
    pre = z[:, :h] * scale[0]
    pre.add_(base[:h]).sigmoid_().add_(hc_eps)
    post = z[:, h:2 * h] * scale[1]
    post.add_(base[h:2 * h]).sigmoid_().mul_(2)
    comb = z[:, 2 * h:] * scale[2]
    comb.add_(base[2 * h:])
    comb = comb.view(t, h, h).softmax(-1)
    comb.add_(hc_eps)
    denom = comb.sum(-2, keepdim=True)
    denom.add_(hc_eps)
    comb.div_(denom)
    if iters > 1:
        comb = native.hc_sinkhorn(comb, iters - 1)
    return (pre, post, comb)

expand = native.hc_expand


def engram_gate(x, kv, weight, rotation, eps=1e-20):
    """Gate in the original basis; weight is the released FP32 q*k product.

    Only the gate query is rotated back through the 32x32 block: V inside kv is
    already rotated, so the residual stream adds the untouched value.
    """
    n, c, d = x.shape
    xf = x.float()
    query = (xf.reshape(-1, 32) @ rotation.t()).view(n, c, d)
    key, value = kv.float().split([c * d, d], -1)
    key = key.view(n, c, d)
    qr = query.square().mean(-1, keepdim=True)
    qr.add_(eps).rsqrt_()
    kr = key.square().mean(-1, keepdim=True)
    kr.add_(eps).rsqrt_()
    qr.mul_(kr)
    dot = query * weight
    dot.mul_(key)
    dot = dot.sum(-1, keepdim=True)
    dot.mul_(qr).mul_(d ** (-0.5))
    magnitude = dot.abs()
    magnitude.clamp_min_(1e-06).sqrt_()
    negative = dot.contiguous().view(torch.int32) < 0
    gate = torch.where(negative, -magnitude, magnitude)
    gate.sigmoid_()
    out = gate * value.unsqueeze(-2)
    out.add_(xf)
    return out.to(x.dtype)


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
    if (dtype == torch.bfloat16 and routed.dtype == shared.dtype == torch.float32
            and routed.device.type == "npu" and routed.ndim == 2
            and routed.shape[-1] == 5120):
        return native.moe_finish(routed, shared)
    return (routed.float() + shared.to(dtype).float()).to(dtype)
