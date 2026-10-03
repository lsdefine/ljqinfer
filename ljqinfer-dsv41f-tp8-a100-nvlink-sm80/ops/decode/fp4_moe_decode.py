"""Loader for the fused FP4 MoE decode kernel (port of V4's moe_rank_decode_fp4)."""
import functools
import pathlib
from torch.utils.cpp_extension import load


@functools.lru_cache(maxsize=1)
def extension():
    root = pathlib.Path(__file__).resolve().parent / 'cuda'
    return load(name='v41_moe_decode_fused',
                sources=[str(root / 'moe_decode_fp4.cu')],
                extra_cuda_cflags=['-O3', '--use_fast_math'], verbose=False)
