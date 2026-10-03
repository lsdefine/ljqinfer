"""Exact small-row FP16 RMS; ATen mean reduction order for 1..15 rows."""
from functools import lru_cache
from pathlib import Path
import os

@lru_cache(None)
def extension():
    from torch.utils.cpp_extension import load
    os.environ.setdefault('TORCH_CUDA_ARCH_LIST','8.0')
    os.environ.setdefault('MAX_JOBS','2')
    return load(name='glm53_decode_rms_small',
                sources=[str(Path(__file__).with_suffix('.cu'))],
                extra_cuda_cflags=['-O3','--use_fast_math','-lineinfo'],verbose=False)
