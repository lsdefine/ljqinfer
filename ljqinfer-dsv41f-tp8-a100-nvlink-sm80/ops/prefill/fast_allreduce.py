"""One-shot NVLink all-reduce for decode-sized activations.

NCCL costs ~35us for the 61KB [6,5120] BF16 reduction decode issues ~83 times
per round; the one-shot exchange over IPC-shared buffers costs ~24us.  Larger
tensors keep using NCCL, which wins again once bandwidth, not latency, rules.
"""
from functools import lru_cache
from pathlib import Path
import os
import torch
import torch.distributed as dist


@lru_cache(maxsize=1)
def extension():
    from torch.utils.cpp_extension import load
    os.environ.setdefault('TORCH_CUDA_ARCH_LIST', '8.0')
    os.environ.setdefault('MAX_JOBS', '2')
    return load(name='v41_fast_allreduce',
                sources=[str(Path(__file__).parent/'cuda'/'fast_allreduce.cu')],
                extra_cuda_cflags=['-O3'], verbose=False)


class FastAllReduce:
    FLAG_BYTES = 8192
    CAPACITY = 1 << 20   # bytes of payload the shared buffer can hold
    BLOCKS = 16          # measured optimum at 61KB on eight A100s

    def __init__(self, group=None, device=None):
        self.world, self.rank = dist.get_world_size(group), dist.get_rank(group)
        self.mod = extension()
        blob = self.mod.ipc_alloc(self.FLAG_BYTES + self.CAPACITY)
        base, handle = blob[0], blob[1:]
        handles = [None]*self.world
        dist.all_gather_object(handles, handle, group=group)
        self.bases = [base if p == self.rank else self.mod.ipc_open(handles[p])
                      for p in range(self.world)]
        self.device = torch.device('cuda', torch.cuda.current_device()
                                   if device is None else device)
        self.calls = 0

    def eligible(self, tensor):
        return (tensor.dtype == torch.bfloat16 and tensor.is_contiguous()
                and tensor.numel() % 8 == 0
                and tensor.numel()*2 <= self.CAPACITY
                and tensor.device == self.device)

    def __call__(self, tensor):
        self.mod.all_reduce(tensor, tensor, self.bases, self.FLAG_BYTES,
                            self.rank, self.BLOCKS)
        self.calls += 1
        return tensor
