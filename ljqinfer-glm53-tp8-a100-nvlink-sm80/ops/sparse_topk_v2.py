"""V2-style exact top-k, A100. Default selector for TP8 decode index scoring.
FP32 score rows, int64 inclusive positions, int32 unsorted selected IDs.
NaN scores are unsupported. Ordered IEEE keys distinguish -0 from +0.
Ties choose the lowest IDs. Short/empty rows pad with -1.
Workspace is explicitly allocated outside the invocation/graph.
"""
import os
from pathlib import Path
from functools import lru_cache
import torch

@lru_cache(None)
def extension():
    from torch.utils.cpp_extension import load
    os.environ.setdefault('TORCH_CUDA_ARCH_LIST', '8.0')
    os.environ.setdefault('MAX_JOBS', '4')
    return load(name='glm53_topk_v2_a100', sources=[str(Path(__file__).with_suffix('.cu'))], extra_cuda_cflags=['-O3'], verbose=False)

def workspace(rows, width, *, device):
    chunks = (width + 4095) // 4096
    return tuple(torch.empty(shape, dtype=torch.int32, device=device) for shape in
                 [(rows, chunks, 1024), (rows, 4), (rows, chunks, 2), (rows, width), (rows, width)])

def select_out(scores, positions, out, work):
    extension().select_out(scores, positions, out, *work)
    return out
