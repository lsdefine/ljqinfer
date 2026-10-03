"""Caller-owned Engram row staging: host gather bytes -> pinned H2D -> BF16 rows.

The host tables stay mapped; only the gathered rows cross the bus, and the
dequantization runs on the device instead of the prefill critical path CPU.
"""
from functools import lru_cache
from pathlib import Path
import os
import torch


@lru_cache(maxsize=1)
def extension():
    from torch.utils.cpp_extension import load
    os.environ.setdefault('TORCH_CUDA_ARCH_LIST', '8.0')
    os.environ.setdefault('MAX_JOBS', '2')
    return load(name='v41_prefill_engram_rows',
                sources=[str(Path(__file__).parent/'cuda'/'engram_rows.cu')],
                extra_cuda_cflags=['-O3'], verbose=False)


class RowWorkspace:
    """Fixed capacity staging for `columns` hash rows per token."""

    def __init__(self, capacity, columns, device):
        if capacity <= 0 or columns <= 0:
            raise ValueError('positive capacity and columns required')
        rows = capacity*columns
        self.columns = columns
        self.values = torch.empty((rows, 256), device=device, dtype=torch.uint8)
        self.scales = torch.empty((rows, 8), device=device, dtype=torch.uint8)
        self.output = torch.empty((rows, 256), device=device, dtype=torch.bfloat16)
        self.host_values = torch.empty((rows, 256), dtype=torch.uint8, pin_memory=True)
        self.host_scales = torch.empty((rows, 8), dtype=torch.uint8, pin_memory=True)
        self.mod = extension()

    def __call__(self, values, scales):
        """values [T,C,256] FP8 bytes and scales [T,C,8] on CPU; returns [T,C,256] BF16."""
        if (values.ndim != 3 or scales.ndim != 3 or values.shape[:2] != scales.shape[:2]
                or values.shape[2] != 256 or scales.shape[2] != 8
                or values.shape[1] != self.columns):
            raise ValueError('expected CPU [T,C,256] rows with [T,C,8] scales')
        if values.device.type != 'cpu' or scales.device.type != 'cpu':
            raise ValueError('host gather output required')
        tokens = values.shape[0]
        rows = tokens*self.columns
        if rows > self.values.shape[0]:
            raise ValueError('gathered rows exceed workspace capacity')
        raw = values if values.dtype == torch.uint8 else values.view(torch.uint8)
        self.host_values[:rows].copy_(raw.reshape(rows, 256))
        self.host_scales[:rows].copy_(scales.reshape(rows, 8))
        self.values[:rows].copy_(self.host_values[:rows], non_blocking=True)
        self.scales[:rows].copy_(self.host_scales[:rows], non_blocking=True)
        self.mod.engram_dequant(self.values[:rows], self.scales[:rows], self.output[:rows])
        return self.output[:rows].view(tokens, self.columns, 256)
