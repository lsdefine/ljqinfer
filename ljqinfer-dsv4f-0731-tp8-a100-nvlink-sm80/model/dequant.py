"""In-place dequant matmul primitives (oracle stage).

Weights stay fp8/fp4 resident (wcache untouched); dequant happens inside the
matmul call, temporary bf16/fp32 weight lives only for one F.linear.
Semantics ported verbatim from /tmp/tlinear.py, which was bitwise-vetted while
building the official gold (gold_cpage).  Activations are NOT quantized here:
oracle precision >= official fp8 kernel by construction.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

_E2M1 = torch.tensor(
    [0., .5, 1., 1.5, 2., 3., 4., 6., -0., -.5, -1., -1.5, -2., -3., -4., -6.])


def _e8m0(s: torch.Tensor) -> torch.Tensor:
    return torch.exp2(s.view(torch.uint8).float() - 127.0)


def deq_fp8(w: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """e4m3 weight + e8m0 scale, 128x128 blocks -> fp32."""
    W = w.float()
    S = _e8m0(s).repeat_interleave(128, 0).repeat_interleave(128, 1)
    return W * S[: W.size(0), : W.size(1)]


def deq_fp4(w: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """e2m1x2 packed weight + e8m0 scale, 32-wide blocks -> fp32."""
    b = w.view(torch.uint8)
    lo = (b & 0xF).long()
    hi = (b >> 4).long()
    t = _E2M1.to(w.device)
    out = torch.stack([t[lo], t[hi]], -1).reshape(b.size(0), -1)
    S = _e8m0(s).repeat_interleave(32, 1)[:, : out.size(1)]
    return out * S


def dequant(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    if weight.dtype == torch.float8_e4m3fn:
        return deq_fp8(weight, scale)
    if weight.dtype == torch.float4_e2m1fn_x2:
        return deq_fp4(weight, scale)
    raise TypeError(f"unexpected weight dtype {weight.dtype}")


def qlinear(x: torch.Tensor, weight: torch.Tensor,
            scale: torch.Tensor | None = None) -> torch.Tensor:
    """x @ dequant(weight).T in fp32, returned in x.dtype.

    `scale` defaults to weight.scale attribute (wcache convention).
    """
    if scale is None:
        scale = getattr(weight, "scale", None)
    if scale is None:  # plain bf16/fp32 weight
        return F.linear(x, weight)
    W = dequant(weight, scale)
    return F.linear(x.float(), W).to(x.dtype)
