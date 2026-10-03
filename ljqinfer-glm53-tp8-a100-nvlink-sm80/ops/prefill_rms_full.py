"""Exact FP16 prefill RMS, retaining ATen mean reduction order for rows >=16."""
from functools import lru_cache
from pathlib import Path
import os

@lru_cache(None)
def extension():
    from torch.utils.cpp_extension import load
    os.environ.setdefault('TORCH_CUDA_ARCH_LIST','8.0')
    os.environ.setdefault('MAX_JOBS','2')
    return load(name='glm53_prefill_rms_full',
                sources=[str(Path(__file__).with_suffix('.cu'))],
                extra_cuda_cflags=['-O3','--use_fast_math','-lineinfo'],verbose=False)
