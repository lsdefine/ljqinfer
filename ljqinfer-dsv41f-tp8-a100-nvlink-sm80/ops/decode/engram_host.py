"""Fused host collector for Engram rows (gather + FP8 dequant + bf16).

The table lives in host memory (~100GB per layer, mmap'd), so the gather can
never move to the GPU; what mattered was killing the per-call numpy/torch
overhead. One C++ pass writes straight into a pinned staging buffer so the
H2D can be async.
"""
import functools
import os

import torch
from pathlib import Path


@functools.lru_cache(maxsize=1)
def mod():
    from torch.utils.cpp_extension import load
    build = '/tmp/ljqinfer_engram_ext'
    Path(build).mkdir(parents=True, exist_ok=True)
    return load(name='engram_gather',
                sources=[str(Path(__file__).with_name('engram_gather.cpp'))],
                extra_cflags=['-O3', '-march=native', '-funroll-loops', '-fopenmp'],
                extra_ldflags=['-fopenmp'],
                build_directory=build, verbose=False)


@functools.lru_cache(maxsize=1)
def lut():
    """FP8 E4M3 byte -> float, built by torch so the values match .float()."""
    return torch.arange(256, dtype=torch.uint8).view(torch.float8_e4m3fn).float()
