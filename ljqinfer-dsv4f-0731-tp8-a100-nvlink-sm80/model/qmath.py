"""Pure-torch quantization math for the DSV4 oracle engine.

Two roles:
1. qlinear / dequant_*: on-the-fly dequant of resident fp8/fp4 weights (wcache untouched).
2. act_quant_sim / fp4_act_quant_sim / rotate_activation: QAT fake-quant ops that are
   part of the model's *semantics* (official model.py applies them even in bf16 path).
Faithful to official inference/kernel.py; slow is fine.
"""
from __future__ import annotations

import os
import torch

_E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                      -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0])

_E2M1_DEV = {}


def _e2m1_on(device):
    """Per-device cached LUT (no H2D inside CUDA graph capture)."""
    k = str(device)
    t = _E2M1_DEV.get(k)
    if t is None:
        t = _E2M1_DEV[k] = _E2M1.to(device)
    return t



def _e8m0_to_float(s: torch.Tensor) -> torch.Tensor:
    return torch.exp2(s.view(torch.uint8).float() - 127.0)


def _expand_scale(scale_f: torch.Tensor, rows: int, cols: int, block: int = 128) -> torch.Tensor:
    return scale_f.repeat_interleave(block, 0).repeat_interleave(block, 1)[:rows, :cols]


def dequant_fp8(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """fp8 e4m3 weight [N,K] + e8m0/fp32 block scale -> fp32 weight."""
    w = weight.float()
    s = _e8m0_to_float(scale) if scale.dtype == torch.float8_e8m0fnu else scale.float()
    return w * _expand_scale(s, w.size(0), w.size(1))


def dequant_fp4(weight: torch.Tensor, scale: torch.Tensor, block: int = 32) -> torch.Tensor:
    """fp4 e2m1x2 packed weight [N,K//2] + e8m0 block scale -> fp32 weight [N,K]."""
    b = weight.view(torch.uint8)
    lo, hi = b & 0xF, b >> 4
    lut = _e2m1_on(weight.device)
    w = torch.stack([lut[lo.long()], lut[hi.long()]], dim=-1).flatten(-2)
    s = _e8m0_to_float(scale)
    s = s.repeat_interleave(block, -1)[..., : w.size(-1)]
    return w * s


def qlinear(x: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor,
            bias: torch.Tensor | None = None) -> torch.Tensor:
    """x @ dequant(W).T. Pure reference (no kernel cache)."""
    if weight.dtype == torch.bfloat16:
        W = weight.float()
    elif weight.dtype == torch.float8_e4m3fn:
        W = dequant_fp8(weight, scale)
    else:  # packed fp4
        W = dequant_fp4(weight, scale)
    y = torch.nn.functional.linear(x.float(), W, bias)
    return y.to(x.dtype)


def _round_scale_pow2(amax: torch.Tensor, qmax: float) -> torch.Tensor:
    return torch.exp2(torch.ceil(torch.log2(amax / qmax)))


def act_quant_sim(x: torch.Tensor, block_size: int = 64) -> torch.Tensor:
    """In-place fused fp8 quant+dequant (QAT simulation). Power-of-2 scales (MXFP),
    amax clamp 1e-4, matching act_quant(round_scale=True, inplace=True)."""
    fp8_max = 448.0
    orig = x.shape
    z = x.reshape(-1, block_size).float()
    amax = z.abs().amax(dim=-1, keepdim=True).clamp_min(1e-4)
    s = _round_scale_pow2(amax, fp8_max)
    z = (z / s).clamp(-fp8_max, fp8_max).to(torch.float8_e4m3fn).float() * s
    x.copy_(z.reshape(orig).to(x.dtype))
    return x


def fp4_act_quant_sim(x: torch.Tensor, block_size: int = 32) -> torch.Tensor:
    """In-place fused fp4 e2m1 quant+dequant (QAT simulation). Power-of-2 scales,
    amax clamp 6*2^-126, matching fp4_act_quant(inplace=True)."""
    fp4_max = 6.0
    orig = x.shape
    z = x.reshape(-1, block_size).float()
    amax = z.abs().amax(dim=-1, keepdim=True).clamp_min(6.0 * 2.0 ** -126)
    s = _round_scale_pow2(amax, fp4_max)
    q = (z / s).clamp(-fp4_max, fp4_max)
    lut = _e2m1_on(x.device)[:8]  # positive magnitudes
    idx = torch.bucketize(q.abs(), (lut[:-1] + lut[1:]) / 2)
    deq = lut[idx] * q.sign() * s
    x.copy_(deq.reshape(orig).to(x.dtype))
    return x


def rotate_activation(x: torch.Tensor) -> torch.Tensor:
    """Randomized Hadamard rotation, scale d^-0.5 (official uses fast_hadamard_transform)."""
    d = x.size(-1)
    assert d & (d - 1) == 0, d
    y = x.float()
    h = 1
    while h < d:
        y = y.unflatten(-1, (-1, 2, h))
        y = torch.cat([y[..., 0, :] + y[..., 1, :], y[..., 0, :] - y[..., 1, :]], dim=-1)
        y = y.flatten(-2)
        h *= 2
    return (y * d ** -0.5).to(x.dtype)
