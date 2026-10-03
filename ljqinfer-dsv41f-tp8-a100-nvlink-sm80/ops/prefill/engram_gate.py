"""Fused Engram gate kernel plus its dtype-agnostic oracle.

The fused kernel is bf16-only (released execution). `reference` is the same
math in float32 and serves the explicit FP32 diagnostic chain, so a precision
sweep never silently reinterprets float32 storage as bf16.
"""
from functools import lru_cache
from pathlib import Path
import os


@lru_cache(maxsize=1)
def extension():
    from torch.utils.cpp_extension import load
    os.environ.setdefault('TORCH_CUDA_ARCH_LIST', '8.0')
    return load(name='v41_engram_gate',
                sources=[str(Path(__file__).with_name('cuda') / 'engram_gate.cu')],
                extra_cuda_cflags=['-O3', '--use_fast_math'], verbose=False)


def reference(x, kv, q_weight, k_weight, eps):
    """x[T,H,D], kv[T,(H+1)*D] -> x + sigmoid(signed sqrt gate) * value row."""
    import torch
    T, H, D = x.shape
    if kv.shape[-1] != (H + 1) * D:
        raise ValueError('engram kv must hold H key copies plus one value row')
    rows = kv.reshape(T, H + 1, D).float()
    s, k, v = x.float(), rows[:, :H], rows[:, H:]
    dot = (s * k * q_weight.float() * k_weight.float()).sum(-1)
    gate = dot * torch.rsqrt(s.pow(2).sum(-1) / D + eps) \
               * torch.rsqrt(k.pow(2).sum(-1) / D + eps) * D ** -0.5
    mag = gate.abs().clamp_min(1e-6).sqrt()
    return (s + torch.sigmoid(torch.copysign(mag, gate)).unsqueeze(-1) * v).to(x.dtype)


def engram_gate(x, kv, q_weight, k_weight, eps):
    """Released bf16 path uses the fused kernel; diagnostics fall back to fp32."""
    import torch
    if x.dtype is torch.bfloat16:
        return extension().engram_gate(x, kv, q_weight, k_weight, eps)
    return reference(x, kv, q_weight.to(x.dtype), k_weight.to(x.dtype), eps)
