"""One-process-per-visible-CUDA-device TP runtime; AF_UNIX control is unchanged."""
from __future__ import annotations
from datetime import timedelta
import os
from typing import Optional
import torch
import torch.distributed as dist
from .config import CONFIG


class SPMDRuntime:
    def __init__(self, rank: int, world: int = CONFIG.tp,
                 device_index: Optional[int] = None,
                 root_file: Optional[str] = None):
        self.rank, self.world = int(rank), int(world)
        if not 0 <= self.rank < self.world:
            raise ValueError(f"rank {rank} out of range for world {world}")
        self.device_index = self.rank if device_index is None else int(device_index)
        # Fixed numerical policy for this dedicated BF16 CUDA engine.
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
        torch.cuda.set_device(self.device_index)
        self.device = torch.device(f"cuda:{self.device_index}")
        self.stream = torch.cuda.current_stream(self.device)
        self._comm = None
        self._gather_buffers = {}
        self._root_file = root_file or os.environ.get(
            "LJQ_NCCL_ROOT", "/dev/shm/ljqinfer_qwen_tp4_nccl")

    @property
    def world_size(self):
        return self.world

    def communicator(self):
        if self._comm is None:
            if dist.is_initialized():
                raise RuntimeError("SPMDRuntime must own its NCCL process group")
            dist.init_process_group(
                "nccl", init_method="file://" + os.path.abspath(self._root_file),
                rank=self.rank, world_size=self.world,
                timeout=timedelta(seconds=180), device_id=self.device)
            self._comm = dist.group.WORLD
        return self._comm

    def activate(self):
        torch.cuda.set_device(self.device_index)
        return torch.cuda.current_stream(self.device)

    def all_reduce(self, tensor):
        dist.all_reduce(tensor, group=self.communicator())
        return tensor

    def all_gather(self, tensor):
        # Separate calls may keep same-shaped results alive simultaneously
        # (e.g. draft top-k values and indices). Shape-keyed reuse aliases them.
        # During capture, the graph allocator owns these distinct storages.
        output = torch.empty((self.world,) + tuple(tensor.shape),
                             dtype=tensor.dtype, device=tensor.device)
        dist.all_gather_into_tensor(output.view(-1), tensor.contiguous().view(-1),
                                    group=self.communicator())
        return output

    def barrier(self):
        dist.barrier(group=self.communicator(), device_ids=[self.device_index])
        self.synchronize()

    def synchronize(self):
        torch.cuda.current_stream(self.device).synchronize()

    def destroy(self):
        if self._comm is not None:
            dist.destroy_process_group(self._comm)
            self._comm = None
            self._gather_buffers.clear()
