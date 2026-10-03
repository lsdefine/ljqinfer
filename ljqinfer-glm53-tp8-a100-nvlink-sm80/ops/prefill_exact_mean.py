"""Exact-order prefill square mean for FP16 inputs, SM80."""
import os
from pathlib import Path
from functools import lru_cache

@lru_cache(None)
def extension():
 from torch.utils.cpp_extension import load
 os.environ.setdefault('TORCH_CUDA_ARCH_LIST','8.0')
 os.environ.setdefault('MAX_JOBS','2')
 return load(name='glm53_prefill_exact_mean',sources=[str(Path(__file__).with_suffix('.cu'))],
             extra_cuda_cflags=['-O3','--fmad=false'],verbose=False)
