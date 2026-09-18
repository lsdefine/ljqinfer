"""One-process-per-logical-NPU TP4 runtime."""
from __future__ import annotations
import ctypes
import os
import time
from typing import Optional
import torch

from . import hccl
from .config import CONFIG

ROOT_INFO_BYTES = 4108


class SPMDRuntime:
    def __init__(self, rank: int, world: int = CONFIG.tp,
                 device_index: Optional[int] = None,
                 root_file: Optional[str] = None):
        self.rank, self.world = int(rank), int(world)
        if not 0 <= self.rank < self.world:
            raise ValueError(f"rank {rank} out of range for world {world}")
        self.device_index = self.rank if device_index is None else int(device_index)
        torch.npu.set_device(self.device_index)
        self.device = torch.device(f"npu:{self.device_index}")
        self.stream = torch.npu.current_stream(self.device)
        self._comm = None
        self._gather_buffers = {}
        self._root_file = root_file or os.environ.get(
            "LJQ_HCCL_ROOT", "/dev/shm/ljqinfer_qwen_tp4_hccl.bin")

    @property
    def world_size(self):
        return self.world

    def _await_root(self, timeout: float = 120.0):
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                if os.path.getsize(self._root_file) == ROOT_INFO_BYTES:
                    with open(self._root_file, "rb") as handle:
                        blob = handle.read()
                    if len(blob) == ROOT_INFO_BYTES:
                        return blob
            except OSError:
                pass
            time.sleep(0.01)
        raise RuntimeError(f"timed out waiting for {self._root_file}")

    def communicator(self):
        if self._comm is not None:
            return self._comm
        lib = hccl._load()
        info = ctypes.create_string_buffer(ROOT_INFO_BYTES)
        if self.rank == 0:
            try:
                os.unlink(self._root_file)
            except OSError:
                pass
            rc = lib.HcclGetRootInfo(ctypes.byref(info))
            if rc:
                raise RuntimeError(f"HcclGetRootInfo rc={rc}")
            blob = ctypes.string_at(ctypes.byref(info), ROOT_INFO_BYTES)
            temporary = f"{self._root_file}.tmp{os.getpid()}"
            with open(temporary, "wb") as handle:
                handle.write(blob)
            os.replace(temporary, self._root_file)
        else:
            ctypes.memmove(info, self._await_root(), ROOT_INFO_BYTES)
        comm = ctypes.c_void_p()
        rc = lib.HcclCommInitRootInfo(
            ctypes.c_uint32(self.world), info, ctypes.c_uint32(self.rank),
            ctypes.byref(comm))
        if rc:
            raise RuntimeError(f"HcclCommInitRootInfo rank={self.rank} rc={rc}")
        hccl.ensure_deterministic()
        self._comm = comm
        return comm

    def activate(self):
        """Bind the calling thread to this rank's NPU before issuing work."""
        torch.npu.set_device(self.device_index)
        return torch.npu.current_stream(self.device)

    def all_reduce(self, tensor):
        hccl.all_reduce_comm(self.communicator(), tensor,
                             torch.npu.current_stream(tensor.device))
        return tensor

    def all_gather(self, tensor):
        key = (tuple(tensor.shape), tensor.dtype)
        output = self._gather_buffers.get(key)
        if output is None:
            output = torch.empty((self.world,) + tuple(tensor.shape),
                                 dtype=tensor.dtype, device=tensor.device)
            self._gather_buffers[key] = output
        hccl.all_gather_comm(self.communicator(), tensor, output,
                             torch.npu.current_stream(tensor.device))
        return output

    def barrier(self):
        probe = torch.zeros(1, dtype=torch.float16, device=self.device)
        self.all_reduce(probe)
        torch.npu.current_stream(self.device).synchronize()

    def synchronize(self):
        torch.npu.current_stream(self.device).synchronize()

    def destroy(self):
        if self._comm is not None:
            rc = hccl._load().HcclCommDestroy(self._comm)
            self._comm = None
            if rc:
                raise RuntimeError(f"HcclCommDestroy rank={self.rank} rc={rc}")
