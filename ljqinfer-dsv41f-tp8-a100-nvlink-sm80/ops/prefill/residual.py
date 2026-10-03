"""Prefill-only residual operators. Tokens are packed as [T,H,D].

No decode import or sequence-length dispatch. Fusion may replace these whole
segments; intermediate mixes are transient and never stored in Past.
"""
import os
from functools import lru_cache
from pathlib import Path
import torch
import triton
import triton.language as tl
from . import hc_mix
import torch.nn.functional as F
_FUSED_GLU = None


def _fused_glu():
    """The decode SwiGLU leaf, resolved late.

    ops.decode imports this module, so binding the fused kernel at import
    time would close a cycle. The ATen expansion below stays as the
    fallback for shapes the kernel declines.
    """
    global _FUSED_GLU
    if _FUSED_GLU is None:
        from ops.decode.quant_fused import swiglu as fused
        _FUSED_GLU = fused
    return _FUSED_GLU




_WCAST = {}


def _cast_w(w, dtype):
    """Cast a CONSTANT weight once and keep it.

    These leaves re-cast the same parameter on every layer of every decode
    step. The value never changes, so the cast is pure launch overhead
    sitting inside the captured graph. Activations are deliberately NOT
    routed through here.
    """
    if w.dtype == dtype:
        return w
    got = _WCAST.get((w.data_ptr(), dtype))
    if got is None or got.shape != w.shape:
        got = w.to(dtype).contiguous()
        _WCAST[(w.data_ptr(), dtype)] = got
    return got



@lru_cache(maxsize=1)
def _norm_extension():
    from torch.utils.cpp_extension import load
    os.environ.setdefault('TORCH_CUDA_ARCH_LIST', '8.0')
    os.environ.setdefault('MAX_JOBS', '2')
    return load(name='v41_prefill_rms_norm',
                sources=[str(Path(__file__).parent/'cuda'/'rms_norm.cu')],
                extra_cuda_cflags=['-O3'], verbose=False)


def rms(x, weight=None, eps=1e-6):
    """RMSNorm in FP32 with a single rounding on store."""
    if x.is_cuda and x.is_contiguous() and x.dtype in (torch.float32, torch.bfloat16):
        out = torch.empty_like(x)
        return _norm_extension().rms_norm(x, weight, out, eps)
    y = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + eps)
    # Released RMSNorm multiplies in FP32 and rounds only the final result.
    if weight is not None:
        y = y * weight.float()
    return y.to(x.dtype)


def mixes(x, fn, scale, base, *, norm_eps=1e-6, hc_eps=1e-6, iters=20):
    h = x.shape[-2]
    # The width-sized reduction and GEMM read x in its stored precision and
    # accumulate in FP32; widening the activation first costs gigabytes of
    # traffic for no extra accuracy in the gates below.
    flat = x.flatten(-2)
    scale_ms = torch.linalg.vector_norm(flat, dim=-1, keepdim=True, dtype=torch.float32)
    scale_ms = scale_ms * scale_ms / flat.shape[-1]
    z = F.linear(flat, _cast_w(fn, flat.dtype)).float() * torch.rsqrt(scale_ms + norm_eps)
    if z.is_cuda and h <= 8 and torch.is_tensor(scale) and torch.is_tensor(base):
        return hc_mix.gates(z, scale, base, h=h, hc_eps=hc_eps, iters=iters)
    pre = torch.sigmoid(z[..., :h] * scale[0] + base[:h]) + hc_eps
    post = 2 * torch.sigmoid(z[..., h:2*h] * scale[1] + base[h:2*h])
    comb = (z[..., 2*h:] * scale[2] + base[2*h:]).unflatten(-1, (h, h))
    comb = comb.softmax(-1) + hc_eps
    comb = comb / (comb.sum(-2, keepdim=True) + hc_eps)
    for _ in range(iters - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + hc_eps)
        comb = comb / (comb.sum(-2, keepdim=True) + hc_eps)
    return pre, post, comb


def collapse(x, pre):
    return (x.float() * pre.unsqueeze(-1)).sum(-2).to(x.dtype)


@triton.jit
def _collapse_norm(X, PRE, W, O, eps, H: tl.constexpr, D: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    off = tl.arange(0, BLOCK)
    mask = off < D
    acc = tl.zeros([BLOCK], tl.float32)
    for i in tl.static_range(H):
        weight = tl.load(PRE + row * H + i).to(tl.float32)
        part = tl.load(X + row * H * D + i * D + off, mask=mask, other=0.).to(tl.float32)
        acc += weight * part
    scale = tl.rsqrt(tl.sum(acc * acc, 0) / D + eps)
    gain = tl.load(W + off, mask=mask, other=0.).to(tl.float32)
    tl.store(O + row * D + off, (acc * scale * gain).to(O.dtype.element_ty), mask=mask)


def collapse_norm(x, pre, weight, eps=1e-6):
    """Collapse the residual copies and RMS-normalise in one FP32 accumulation."""
    if not (x.is_cuda and x.is_contiguous() and x.ndim == 3 and weight is not None):
        return rms(collapse(x, pre), weight, eps)
    rows, copies, width = x.shape
    out = torch.empty((rows, width), device=x.device, dtype=x.dtype)
    _collapse_norm[(rows,)](x, _cast_w(pre.contiguous(), torch.float32), weight, out, eps,
                            copies, width, triton.next_power_of_2(width), num_warps=8)
    return out


def expand(x, residual, post, comb):
    # comb[source_copy, destination_copy], NOT the reverse orientation.
    y = post.unsqueeze(-1) * x.unsqueeze(-2)
    y = y + (comb.unsqueeze(-1) * residual.unsqueeze(-2)).sum(-3)
    return y.to(x.dtype)


def engram_gate(x, kv, q_weight, k_weight, *, eps=1e-6, mask=None):
    """kv is the ALL-REDUCED TP projection of prefetched rows, [T,(H+1)*D]."""
    h, d = x.shape[-2:]
    key, value = kv.split([h*d, d], dim=-1)
    key = key.float().unflatten(-1, (h, d))
    stream = x.float()
    sn = torch.linalg.vector_norm(stream, dim=-1)
    kn = torch.linalg.vector_norm(key, dim=-1)
    inv = 1.0 / stream.shape[-1]
    rstd = torch.rsqrt(sn * sn * inv + eps) * torch.rsqrt(kn * kn * inv + eps)
    dot = (stream * _cast_w(q_weight, torch.float32) * _cast_w(k_weight, torch.float32) * key).sum(-1) * rstd * d**-.5
    gate = torch.sigmoid(torch.copysign(dot.abs().clamp_min(1e-6).sqrt(), dot))
    if mask is not None:
        gate = gate.masked_fill(~mask.unsqueeze(-1), 0)
    return (stream + gate.unsqueeze(-1) * value.float().unsqueeze(-2)).to(x.dtype)


def swiglu_packed(gu, limit):
    """SwiGLU straight off a packed [T, 2n] gate/up GEMM, one launch."""
    if gu is None:
        return None
    try:
        from ops.decode.quant_fused import swiglu_packed as _packed
    except Exception:
        return None
    return _packed(gu, limit)


def swiglu(gate, up, limit, route_weight=None):
    y = _fused_glu()(gate, up, limit, route_weight)
    if y is not None:
        return y
    g, u = gate.float(), up.float()
    if limit > 0:
        g, u = g.clamp(max=limit), u.clamp(-limit, limit)
    y = F.silu(g) * u
    if route_weight is not None:
        y = y * route_weight
    return y.to(gate.dtype)


def route(x, weight, bias, *, topk, temperature, scale, normalize=True, score='sqrtsoftplus'):
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
