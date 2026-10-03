"""Fused sparse latent attention (prefill): gather happens inside the kernel."""
from functools import lru_cache
from pathlib import Path
import os
import torch


@lru_cache(maxsize=1)
def extension():
    from torch.utils.cpp_extension import load
    os.environ.setdefault('TORCH_CUDA_ARCH_LIST', '8.0')
    os.environ.setdefault('MAX_JOBS', '2')
    return load(name='v41_prefill_sparse_attn',
                sources=[str(Path(__file__).parent/'cuda'/'sparse_attn_flat.cu')],
                extra_cuda_cflags=['-O3'], verbose=False)


def attend(q, bank, ix, bad, sink, out, scale):
    extension().sparse_attn_flat(q, bank, ix, bad, sink, out, scale)
    return out
