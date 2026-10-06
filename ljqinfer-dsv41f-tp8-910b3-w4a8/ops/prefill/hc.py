"""Hyper-connection semantic operators; next_pre gates the NEXT sublayer.

h BF16[T,H,D], incoming_pre FP32[T,H]; result x BF16[T,D],
next_pre/post FP32[T,H], comb FP32[T,H,H]. Inputs are read-only.
Eager v1 returns owned tensors; current-stream scratch is allocator-managed.
"""
from . import residual as r


def hc_prepare(h, incoming_pre, fn, scale, base, norm_weight, *, norm_eps, hc_eps, iters):
    next_pre, post, comb = r.mixes(h, fn.float(), scale.float(), base.float(),
                                 norm_eps=norm_eps, hc_eps=hc_eps, iters=iters)
    x = r.collapse_norm(h, incoming_pre, norm_weight, norm_eps)
    return x, next_pre, post, comb


def hc_finish(y, h, post, comb):
    """BF16[T,D] sublayer result + BF16[T,H,D] residual -> BF16[T,H,D]."""
    return r.expand(y, h, post, comb)
