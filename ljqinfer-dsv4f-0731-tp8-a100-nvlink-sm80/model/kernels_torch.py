"""Pure-torch equivalents of the official tilelang kernels (oracle path, A100/sm80).

API mirrors official inference/kernel.py so model code can be ported verbatim:
    act_quant, fp4_act_quant, fp8_gemm, fp4_gemm, sparse_attn, hc_split_sinkhorn

Internally self-consistent: act_quant returns float32 scales (values are exact
powers of two when scale_fmt is set), gemms dequantize to fp32 and matmul.
Slow is fine; fidelity first.
"""
from __future__ import annotations

from typing import Optional

import torch

from .qmath import (_E2M1, _e2m1_on, _e8m0_to_float, act_quant_sim, fp4_act_quant_sim,
                    dequant_fp4, rotate_activation)

FP8_MAX = 448.0
FP4_MAX = 6.0


def _pow2_scale(amax: torch.Tensor, qmax: float) -> torch.Tensor:
    return torch.exp2(torch.ceil(torch.log2(amax / qmax)))


def act_quant(
    x: torch.Tensor, block_size: int = 128, scale_fmt: Optional[str] = None,
    scale_dtype: torch.dtype = torch.float32, inplace: bool = False,
):
    """Block-wise FP8 quantization along last dim.
    inplace=True: fused quant+dequant back into x (returns x).
    else: returns (q fp8_e4m3, s fp32) with s shaped [..., K//block_size]."""
    if inplace:
        return act_quant_sim(x, block_size)
    assert x.size(-1) % block_size == 0
    z = x.reshape(*x.shape[:-1], -1, block_size).float()
    amax = z.abs().amax(dim=-1, keepdim=True).clamp_min(1e-4)
    if scale_fmt is not None:
        s = _pow2_scale(amax, FP8_MAX)
    else:
        s = amax / FP8_MAX
    q = (z / s).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    return q.reshape(x.shape), s.squeeze(-1)


def fp4_act_quant(x: torch.Tensor, block_size: int = 32, inplace: bool = False):
    """Block-wise FP4 e2m1 quantization along last dim, pow2 (E8M0) scales.
    inplace=True: fused quant+dequant back into x."""
    if inplace:
        return fp4_act_quant_sim(x, block_size)
    assert x.size(-1) % block_size == 0
    z = x.reshape(*x.shape[:-1], -1, block_size).float()
    amax = z.abs().amax(dim=-1, keepdim=True).clamp_min(FP4_MAX * 2.0 ** -126)
    s = _pow2_scale(amax, FP4_MAX)
    q = (z / s).clamp(-FP4_MAX, FP4_MAX)
    lut = _e2m1_on(x.device)[:8]
    idx = torch.bucketize(q.abs(), (lut[:-1] + lut[1:]) / 2)
    qv = lut[idx] * q.sign()
    return qv.reshape(x.shape), s.squeeze(-1)


_E8M0 = getattr(torch, "float8_e8m0fnu", None)


def _scale_f(s: torch.Tensor) -> torch.Tensor:
    return _e8m0_to_float(s) if s.dtype is _E8M0 else s.float()


def fp8_gemm(a: torch.Tensor, a_s: torch.Tensor, b: torch.Tensor, b_s: torch.Tensor,
             scale_dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """C[M,N] = A_fp8[...,K] @ B_fp8[N,K]^T. a_s per 1x128 (K), b_s per 128x128."""
    K = a.size(-1)
    af = a.float() * _scale_f(a_s).repeat_interleave(128, -1)[..., :K]
    bs = _scale_f(b_s)
    bf = b.float() * bs.repeat_interleave(128, 0).repeat_interleave(128, 1)[
        : b.size(0), : b.size(1)]
    return (af @ bf.t()).to(torch.bfloat16)


def fp4_gemm(a: torch.Tensor, a_s: torch.Tensor, b: torch.Tensor, b_s: torch.Tensor,
             scale_dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """C[M,N] = A_fp8[...,K] @ B_fp4[N,K]^T. a_s per 1x128; b packed [N,K//2],
    b_s per 1x32 E8M0."""
    K = a.size(-1)
    af = a.float() * _scale_f(a_s).repeat_interleave(128, -1)[..., :K]
    bf = dequant_fp4(b, b_s)
    return (af @ bf.t()).to(torch.bfloat16)


def sparse_attn(q: torch.Tensor, kv: torch.Tensor, attn_sink: torch.Tensor,
                topk_idxs: torch.Tensor, softmax_scale: float) -> torch.Tensor:
    """q:[b,s,h,d], kv:[b,n,d] (shared across heads), topk_idxs:[b,s,topk] (-1 pad),
    attn_sink:[h]. Softmax over gathered positions with sink term in denominator."""
    b, s, h, d = q.shape
    idx = topk_idxs.long()
    valid = idx >= 0
    kvg = kv.gather(1, idx.clamp_min(0).reshape(b, -1, 1).expand(-1, -1, d))
    kvg = kvg.reshape(b, s, -1, d).float()                      # [b,s,topk,d]
    scores = torch.einsum("bshd,bskd->bshk", q.float(), kvg) * softmax_scale
    scores = scores.masked_fill(~valid.unsqueeze(2), float("-inf"))
    m = scores.amax(dim=-1)                                     # [b,s,h]
    m = torch.maximum(m, torch.full_like(m, -1e30))             # all-invalid guard
    e = torch.exp(scores - m.unsqueeze(-1))
    denom = e.sum(dim=-1) + torch.exp(attn_sink.float().view(1, 1, h) - m)
    o = torch.einsum("bshk,bskd->bshd", e, kvg) / denom.unsqueeze(-1)
    return o.to(q.dtype)


def hc_split_sinkhorn(mixes: torch.Tensor, hc_scale: torch.Tensor,
                      hc_base: torch.Tensor, hc_mult: int = 4,
                      sinkhorn_iters: int = 20, eps: float = 1e-6):
    """mixes:[b,s,(2+hc)*hc] -> pre:[b,s,hc], post:[b,s,hc], comb:[b,s,hc,hc]."""
    hc = hc_mult
    z = mixes.float()
    sc, bs = hc_scale.float(), hc_base.float()
    pre = torch.sigmoid(z[..., :hc] * sc[0] + bs[:hc]) + eps
    post = 2 * torch.sigmoid(z[..., hc:2 * hc] * sc[1] + bs[hc:2 * hc])
    comb = (z[..., 2 * hc:] * sc[2] + bs[2 * hc:]).reshape(*z.shape[:-1], hc, hc)
    comb = comb.softmax(-1) + eps
    comb = comb / (comb.sum(-2, keepdim=True) + eps)
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + eps)
        comb = comb / (comb.sum(-2, keepdim=True) + eps)
    return pre, post, comb
