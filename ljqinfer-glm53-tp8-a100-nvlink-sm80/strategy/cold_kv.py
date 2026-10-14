"""Torch payload/IO adapter; cache policy lives only in cold_kv_core.py."""
import torch
from strategy.cold_kv_core import (ColdCache as _ColdCache, Field, Lease, Store,
                                 Entry, Node, CacheCapacityError)


class TorchBackend:
    @staticmethod
    def nbytes(value):
        return value.numel() * value.element_size()

    @staticmethod
    def empty(value, pin_memory):
        return torch.empty(value.shape, dtype=value.dtype, device='cpu',
                           pin_memory=pin_memory)

    @staticmethod
    def copy(dst, src, waiters):
        if src.device.type != 'cpu':
            device_api = getattr(torch, src.device.type)
            # Register BEFORE enqueue: rollback must also await a failed copy.
            waiters.add(device_api.current_stream(src.device))
        dst.copy_(src, non_blocking=True)


class ColdCache(_ColdCache):
    def __init__(self, fields, budget_bytes, *, namespace, pin_memory=False,
                 shm=None):
        super().__init__(fields, budget_bytes, namespace=namespace,
                         pin_memory=pin_memory, shm=shm, backend=TorchBackend())
