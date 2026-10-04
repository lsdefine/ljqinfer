"""Vendor fused lightning indexer (aclnnLightningIndexer) via aclnn C API.

One dispatch replaces score -> reduce -> top-k: it scores all index heads
against paged keys and emits the top sparse_count key positions directly.
torch_npu exposes no binding for this op, so the two-stage aclnn contract is
driven through ctypes against libopapi.

Hard constraints discovered from the op tiling checks (not in any header):
  * block_table is mandatory and int32; keys are always paged.
  * layout_query='BSND', layout_key='PA_BSND'.
  * pre_tokens and next_tokens must be INT64_MAX, otherwise tiling rejects.
  * return_values must be False under PA_BSND; only indices come back.
  * sparse_indices must be 4-D [B, S, 1, sparse_count].
Causality is expressed by sparse_mode=3 plus actual_seq_lengths_{query,key}.
"""
import ctypes as C
from ops.queued import queued

_INT64_MAX = (1 << 63) - 1
_FORMAT_ND = 2
_DTYPE = {'torch.float32': 0, 'torch.float16': 1, 'torch.int8': 2,
          'torch.int32': 3, 'torch.bfloat16': 27}


class _Acl:
    """Lazily bound aclnn entry points, loaded once per process."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._bind()
        return cls._instance

    def _bind(self):
        self.base = C.CDLL('libnnopbase.so', mode=C.RTLD_GLOBAL)
        self.api = C.CDLL('libopapi.so', mode=C.RTLD_GLOBAL)
        self.base.aclCreateTensor.restype = C.c_void_p
        self.base.aclCreateTensor.argtypes = [
            C.POINTER(C.c_int64), C.c_uint64, C.c_int, C.POINTER(C.c_int64),
            C.c_int64, C.c_int, C.POINTER(C.c_int64), C.c_uint64, C.c_void_p]
        self.workspace = self.api.aclnnLightningIndexerGetWorkspaceSize
        self.workspace.restype = C.c_int
        self.workspace.argtypes = ([C.c_void_p] * 6 + [C.c_char_p, C.c_char_p]
                                   + [C.c_int64] * 4 + [C.c_bool]
                                   + [C.c_void_p] * 2
                                   + [C.POINTER(C.c_uint64), C.POINTER(C.c_void_p)])
        self.repeatable = self.base.aclSetAclOpExecutorRepeatable
        self.repeatable.restype = C.c_int
        self.repeatable.argtypes = [C.c_void_p]
        self.launch = self.api.aclnnLightningIndexer
        self.launch.restype = C.c_int
        self.launch.argtypes = [C.c_void_p, C.c_uint64, C.c_void_p, C.c_void_p]
        self.launch = queued(self.launch, self.launch.argtypes, check_status=True)

    def tensor(self, t):
        dims = (C.c_int64 * t.dim())(*t.shape)
        strides = (C.c_int64 * t.dim())(*t.stride())
        storage = (C.c_int64 * 1)(t.numel())
        handle = self.base.aclCreateTensor(
            dims, t.dim(), _DTYPE[str(t.dtype)], strides, 0, _FORMAT_ND,
            storage, 1, C.c_void_p(t.data_ptr()))
        if not handle:
            raise RuntimeError('aclCreateTensor returned null')
        return C.c_void_p(handle)


class VendorIndexer:
    """Bound operand set for one shape; rebuild when any address changes.

    query  [B, S, N, D] bf16, weights [B, S, N] bf16 or fp32,
    key    [num_blocks, block_size, 1, D] bf16, block_table [B, max_blocks] int32,
    actual_seq_lengths_* [B] int32, indices [B, S, 1, sparse_count] int32.
    """

    def __init__(self, query, key, weights, seq_query, seq_key, block_table,
                 indices, sparse_count, *, sparse_mode=3):
        import torch
        self.acl = _Acl()
        self.indices = indices
        self.sparse_count = int(sparse_count)
        self.sparse_mode = int(sparse_mode)
        self.operands = (query, key, weights, seq_query, seq_key, block_table)
        # Values are never produced under PA_BSND but the op still type-checks
        # the output descriptor, so a one-element bf16 stub is handed over.
        self.values = torch.zeros(1, 1, 1, 1, dtype=torch.bfloat16,
                                  device=query.device)
        self.workspace = None
        self.executor = C.c_void_p()
        self._prepare()

    def _prepare(self):
        import torch
        acl, size = self.acl, C.c_uint64(0)
        # aclnn tensor descriptors must outlive the executor, and the executor
        # is single-shot unless marked repeatable; both are kept on the plan so
        # one plan can be dispatched every decode step without rebuilding.
        self.handles = [acl.tensor(t) for t in self.operands]
        self.handles.extend((acl.tensor(self.indices), acl.tensor(self.values)))
        handles = self.handles[:6]
        status = acl.workspace(*handles, b'BSND', b'PA_BSND', self.sparse_count,
                               self.sparse_mode, _INT64_MAX, _INT64_MAX, False,
                               *self.handles[6:],
                               C.byref(size), C.byref(self.executor))
        if status != 0:
            raise RuntimeError(
                f'aclnnLightningIndexerGetWorkspaceSize failed: {status}; '
                'rerun with ASCEND_SLOG_PRINT_TO_STDOUT=1 ASCEND_GLOBAL_LOG_LEVEL=1 '
                'to read the tiling check that rejected the operands')
        status = acl.repeatable(self.executor)
        if status != 0:
            raise RuntimeError(f'aclSetAclOpExecutorRepeatable failed: {status}')
        self.size = size
        self.workspace = torch.empty(max(size.value, 1), dtype=torch.uint8,
                                     device=self.indices.device)

    def run(self, stream=None):
        import torch
        if stream is None:
            stream = torch.npu.current_stream().npu_stream
        status = self.acl.launch(C.c_void_p(self.workspace.data_ptr()), self.size,
                                 self.executor, C.c_void_p(stream))
        if status != 0:
            raise RuntimeError(f'aclnnLightningIndexer failed: {status}')
        return self.indices

    def close(self):
        destroy = self.acl.base.aclDestroyAclOpExecutor
        destroy.argtypes, destroy.restype = [C.c_void_p], C.c_int
        if self.executor:
            destroy(self.executor)
            self.executor = C.c_void_p()
        destroy = self.acl.base.aclDestroyTensor
        destroy.argtypes, destroy.restype = [C.c_void_p], C.c_int
        for handle in reversed(self.handles):
            destroy(handle)
        self.handles.clear()
