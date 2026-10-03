"""Caller-owned FP32 activation preparation on the construction stream."""
from functools import lru_cache
from pathlib import Path
import os
import torch


@lru_cache(maxsize=1)
def extension():
    from torch.utils.cpp_extension import load
    os.environ.setdefault('TORCH_CUDA_ARCH_LIST', '8.0')
    os.environ.setdefault('MAX_JOBS', '2')
    return load(name='v41_prefill_quant_workspace',
                sources=[str(Path(__file__).parent/'cuda'/'quant_workspace.cu')],
                extra_cuda_cflags=['-O3'], verbose=False)


class ActivationWorkspace:
    def __init__(self, capacity, width, device):
        if capacity <= 0 or width <= 0 or width % 32:
            raise ValueError('positive capacity and K32 width required')
        self.output = torch.empty((capacity, width), device=device, dtype=torch.float32)
        self.stream = torch.cuda.current_stream(self.output.device)
        self.mod = extension()
        self.calls = 0

    def __call__(self, x):
        if (x.ndim != 2 or x.shape[0] > self.output.shape[0]
                or x.shape[1] != self.output.shape[1]
                or x.device != self.output.device or x.dtype != torch.bfloat16
                or not x.is_contiguous()):
            raise ValueError('rank-local contiguous BF16 matrix within workspace capacity required')
        if (torch.cuda.current_stream(x.device) != self.stream
                and not torch.cuda.is_current_stream_capturing()):
            raise RuntimeError('activation workspace requires its construction stream')
        result = self.mod.activation(x, self.output[:x.shape[0]])
        self.calls += 1
        return result
