"""Standalone FP4 grouped GEMM extension (SM80).

Separate from the CUTLASS grouped extension: one pybind module per build unit.
"""
from functools import lru_cache
from pathlib import Path
import os


@lru_cache(maxsize=1)
def extension():
    from torch.utils.cpp_extension import load
    root = Path(__file__).parent / 'cuda'
    os.environ.setdefault('TORCH_CUDA_ARCH_LIST', '8.0')
    return load(name='v41_prefill_fp4gemm',
                sources=[str(root / 'fp4_grouped_gemm.cu')],
                extra_cuda_cflags=['-O3'], verbose=False)
