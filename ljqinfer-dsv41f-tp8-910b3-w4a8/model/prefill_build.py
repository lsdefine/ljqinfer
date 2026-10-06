"""Lightweight eager prefill construction. No warmup, graph or device execution."""
import ctypes as C
import os
from pathlib import Path
import torch
import torch.distributed as dist
from ops.queued import queued
from model.prefill import PrefillModel
from model.prefill_block import PrefillBlock
from model.prefill_config import validate_config


class PrefillParallel:
    """One capture-safe TP8 communicator; caller closes after resetting graphs."""
    def __init__(self, group=None):
        self.group = group
        self.world, self.rank = dist.get_world_size(group), dist.get_rank(group)
        if self.world != 8:
            raise ValueError('canonical layout requires exactly TP8')
        self.device = torch.device('npu', torch.npu.current_device())
        root = Path(os.environ['ASCEND_HOME_PATH'])
        self.lib = C.CDLL(str(root / 'lib64/libhccl.so'))
        p, i, u = C.c_void_p, C.c_int32, C.c_uint32
        signatures = {
            'HcclGetRootInfo': [p],
            'HcclCommInitRootInfo': [u, p, u, C.POINTER(p)],
            'HcclAllReduce': [p, p, C.c_uint64, i, i, p, p],
            'HcclAllGather': [p, p, C.c_uint64, i, p, p],
            'HcclReduceScatter': [p, p, C.c_uint64, i, i, p, p],
            'HcclSetConfig': [i, i], 'HcclGetConfig': [i, C.POINTER(i)],
            'HcclCommDestroy': [p],
        }
        for name, args in signatures.items():
            fn = getattr(self.lib, name)
            fn.argtypes, fn.restype = args, i
        # Keep library symbols raw: native kernels borrow their function pointers.
        self._collectives = {name: queued(getattr(self.lib, name), signatures[name],
                                         name, check_status=True)
            for name in ('HcclAllReduce', 'HcclAllGather', 'HcclReduceScatter')}
        self._call('HcclSetConfig', 0, 1)
        actual = i()
        self._call('HcclGetConfig', 0, C.byref(actual))
        if actual.value != 1:
            raise RuntimeError('HCCL deterministic reduction not enabled')
        info = C.create_string_buffer(4108)
        if self.rank == 0:
            self._call('HcclGetRootInfo', info)
        bootstrap = torch.tensor(list(info.raw), dtype=torch.uint8, device=self.device)
        src = 0 if group is None else dist.get_global_rank(group, 0)
        dist.broadcast(bootstrap, src=src, group=group)
        C.memmove(info, bytes(bootstrap.cpu().tolist()), 4108)
        torch.npu.synchronize(self.device)
        self.comm = p()
        self._call('HcclCommInitRootInfo', self.world, info, self.rank, C.byref(self.comm))

    def _call(self, name, *args):
        fn = self._collectives.get(name) or getattr(self.lib, name)
        rc = fn(*args)
        if rc:
            raise RuntimeError(f'{name} failed: {rc}')

    def _buffer(self, tensor):
        if tensor.device != self.device or not tensor.is_contiguous():
            raise ValueError('HCCL requires a contiguous rank-local device buffer')
        return {torch.float16: 3, torch.float32: 4, torch.bfloat16: 11}[tensor.dtype]

    def index_scores(self, q, bank, weight, positions, valid, dot, scores, keys, ratio, scale, tile):
        """Borrow this communicator for the ordered native index scoring operator."""
        from ops.prefill.native import ops
        return ops.score_tiles(q, bank, weight, positions, valid, dot, scores,
                               keys, ratio, scale, tile,
                               C.cast(self.lib.HcclAllReduce, C.c_void_p).value, self.comm.value)

    def sum(self, tensor):
        dtype = self._buffer(tensor)
        self._call('HcclAllReduce', tensor.data_ptr(), tensor.data_ptr(),
                   tensor.numel(), dtype, 0, self.comm,
                   torch.npu.current_stream(self.device).npu_stream)

    def logits(self, tensor, *, out):
        dtype = self._buffer(tensor)
        if self._buffer(out) != dtype or out.numel() != self.world * tensor.numel():
            raise ValueError('rank-major gather output shape/dtype mismatch')
        self._call('HcclAllGather', tensor.data_ptr(), out.data_ptr(),
                   tensor.numel(), dtype, self.comm,
                   torch.npu.current_stream(self.device).npu_stream)
        return out

    def scatter(self, tensor, *, out):
        dtype = self._buffer(tensor)
        if self._buffer(out) != dtype or tensor.numel() != out.numel()*self.world:
            raise ValueError('invalid Engram reduce-scatter geometry')
        self._call('HcclReduceScatter', tensor.data_ptr(), out.data_ptr(),
                   out.numel(), dtype, 0, self.comm,
                   torch.npu.current_stream(self.device).npu_stream)
        return out

    def close(self):
        if self.comm:
            torch.npu.synchronize(self.device)
            self._call('HcclCommDestroy', self.comm)
            self.comm = C.c_void_p()


def build_prefill(config, weights, hasher, host_tables, *, device, length,
                  parallel, past, phase='encoder_append', capacity=None,
                  library=None, shared_from=None):
    validate_config(config)
    capacity = past.max_seq if capacity is None else capacity
    if phase not in ('encoder_append', 'encoder_replay'):
        raise ValueError('invalid prefill phase')
    limit = 128 if phase == 'encoder_replay' else 12288
    if type(length) is not int or not 1 <= length <= limit:
        raise ValueError('invalid encoder row limit')
    if not max(length, 2) <= capacity <= past.max_seq:
        raise ValueError('invalid context capacity')
    if shared_from is not None:
        if (shared_from.past is not past or shared_from.weights is not weights
                or shared_from.parallel is not parallel or shared_from.c != config):
            raise ValueError('incompatible prefill sharing')
        blocks = shared_from.blocks
    else:
        blocks = tuple(PrefillBlock(i, config, weights, parallel, hasher, host_tables, library)
                       for i in range(config['n_layers']))
    return PrefillModel(config, weights, blocks, parallel, past=past,
                        phase=phase, length=length, capacity=capacity,
                        device=device, library=library)
