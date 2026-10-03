"""Decode kernels borrowed from the v4 tree (bit-exact verified against ops.prefill).

Loaded out-of-tree on purpose: the .cu sources stay in the v4 checkout until the
port is accepted, so this file is a thin, revertible shim. Set DSV4_OPS_DIR to
relocate.
"""
import functools
import os

import torch
from pathlib import Path

_DIR = Path('/mnt/data/kw/ljqinfer_dsv4f_tp8/ops')
_SRC = ['dsv4_wgemm.cu', 'prefill_moe_cutlass_gemm.cu', 'sparse_attn_paged.cu',
        'paged_io.cu', 'compressor_tail.cu', 'index_score.cu', 'peer_ar_ipc.cu']


@functools.lru_cache(maxsize=1)
def mod():
    os.environ.setdefault('TORCH_CUDA_ARCH_LIST', '8.0')
    os.environ.setdefault('MAX_JOBS', '8')
    from torch.utils.cpp_extension import load
    inc = _DIR / 'third_party' / 'cutlass' / 'include'
    if not (inc / 'cutlass' / 'cutlass.h').is_file():
        raise RuntimeError('cutlass headers missing at %s' % inc)
    return load(
        name='dsv4_wgemm',
        sources=[str(_DIR / s) for s in _SRC],
        extra_cuda_cflags=['-O3', '--use_fast_math', '-lineinfo',
                           '-DDSV4_NO_PYBIND', '-I' + str(inc)],
        build_directory=str(_DIR / '.build'),
        verbose=False,
    )


def rope_(x, freqs, inverse=False):
    """In-place RoPE over the trailing 2*freqs.shape[-1] lanes of x [T,H,D].

"""
    d = freqs.shape[-1] * 2
    view = x if x.shape[-1] == d else x[..., -d:]
    mod().rope_inplace(view.unsqueeze(0), freqs, inverse)
    return x


_F32_W = {}


def _f32_weight(weight):
    """Norm weights are stored bf16 but the leaf wants fp32.

    They are constants, so the widening happens once per tensor and is exact
    (bf16 -> fp32 never rounds); caching keeps it out of the captured graph.
    """
    if weight.dtype == torch.float32:
        return weight
    got = _F32_W.get(weight.data_ptr())
    if got is None or got.shape != weight.shape:
        got = weight.float().contiguous()
        _F32_W[weight.data_ptr()] = got
    return got


def rms(x, weight, eps):
    return mod().rms_norm(x, _f32_weight(weight), eps)


def rms_split2(x, w0, w1, eps):
    """Both halves of a packed projection, one launch (see quant_fused)."""
    from ops.decode.quant_fused import rms_split2 as _split
    return _split(x, _f32_weight(w0), _f32_weight(w1), eps)


_ZERO = {}


def topk(scores, k, ratio, positions):
    """Exact top-K over the live prefix, replacing a full [Q,N] sort.

    scores [Q,N] fp32 already carries -inf past ``(pos+1)//ratio``; the leaf
    re-derives that bound itself, so the two agree on which rows are legal.
    Returns logical row ids [Q,k], -1 padded, same ABI as the sort it replaces.
    Emission order differs from the sort (row order, not score order); the set
    is identical, and every consumer treats it as a set.
    """
    q = scores.shape[0]
    key = (scores.device, q)
    off = _ZERO.get(key)
    if off is None:
        off = _ZERO[key] = scores.new_zeros(q, dtype=torch.int64)
    pos = positions.to(torch.int64)
    out = mod().topk_select_post_positions(
        scores.unsqueeze(0).contiguous(), k, ratio, off, pos)
    return out[0].to(torch.int64)


def grouped_linear_bf16(x, weight):
    """[T,G,K] @ [G,N,K] -> [T,G,N] as one strided-batched GEMM.

    Decode-side replacement for the prefill grouped_linear, which looped over
    groups and materialised an FP32 copy of both the activation slice and the
    whole weight group on every call. Tensor cores already accumulate in FP32,
    so feeding BF16 straight through keeps the single output rounding while
    removing the per-group FP32 weight copies.
    """
    if x.ndim != 3 or weight.ndim != 3:
        raise ValueError('grouped GEMM requires rank-three tensors')
    if x.shape[1] != weight.shape[0] or x.shape[2] != weight.shape[2]:
        raise ValueError('grouped GEMM geometry')
    if x.dtype != weight.dtype:
        raise TypeError('grouped GEMM requires matching dtypes')
    return torch.bmm(x.transpose(0, 1), weight.transpose(1, 2)).transpose(0, 1)
