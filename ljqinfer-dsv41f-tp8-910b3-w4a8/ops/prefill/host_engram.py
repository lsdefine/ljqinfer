"""Synchronous CPU Engram gather/dequantize: INT8 + FP32 scales -> BF16.

Owns its output; no request cache or full-table materialization. Four workers
above 3072 selected rows, with round-to-nearest-even conversion. Build products
and the inter-process build lock live in Torch's external extension cache.
"""
import ctypes
from pathlib import Path
import torch
from torch.utils.cpp_extension import load


_library = ctypes.CDLL(load(
    name='ljq_prefill_host_engram',
    sources=[str(Path(__file__).resolve().parents[1] / 'kernels' / 'prefill_host_engram.cpp')],
    extra_cflags=['-O3', '-std=c++17', '-fopenmp'],
    extra_ldflags=['-fopenmp'], is_python_module=False, verbose=False))
_gather = _library.gather_bf16
_gather.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_int64] * 2
_gather.restype = ctypes.c_int


def gather(weight, scale, ids):
    """Contiguous CPU table + selected int64 IDs -> owned BF16[...,256]."""
    if (weight.device.type != 'cpu' or scale.device.type != 'cpu'
            or weight.dtype != torch.int8 or scale.dtype != torch.float32
            or weight.ndim != 2 or weight.shape[1] != 256
            or scale.shape != (weight.shape[0], 8)
            or not weight.is_contiguous() or not scale.is_contiguous()):
        raise ValueError('expected CPU contiguous int8 [rows,256], float32 [rows,8]')
    if ids.device.type != 'cpu' or ids.dtype != torch.int64:
        raise ValueError('expected CPU int64 IDs')
    ids = ids.contiguous()
    out = torch.empty((*ids.shape, 256), dtype=torch.bfloat16, device='cpu')
    if _gather(weight.data_ptr(), scale.data_ptr(), ids.data_ptr(),
               out.data_ptr(), ids.numel(), weight.shape[0]):
        raise ValueError('hash row out of range')
    return out
