"""Decode CANN W4A8 HP projection; all device storage is caller-owned.

Build creates descriptors/executors only. bind supplies serial scratch. Execution
uses fixed outputs, with no torch_npu allocating wrappers or weight conversion.
Destroy captured graphs and synchronize before close. Concurrent streams need
independent scratch. NZ bytes are borrowed from the canonical weight cache.
"""
import ctypes as C
from pathlib import Path
import torch
from ops.queued import queued

P, I, U = C.c_void_p, C.c_int64, C.c_uint64
_LIB = C.CDLL('libopapi.so')
_DT = {torch.float32: 0, torch.float16: 1, torch.int8: 2,
       torch.int32: 3, torch.int64: 9, torch.bfloat16: 27}


def _fn(name, args, result=C.c_int):
    f = getattr(_LIB, name)
    f.argtypes, f.restype = args, result
    return f


def _overlaps(a, b):
    return (a.data_ptr() < b.data_ptr()+b.numel()*b.element_size()
            and b.data_ptr() < a.data_ptr()+a.numel()*a.element_size())


class W4A8GroupedMatmul:
    def __init__(self, x, weight, scale, bias, ends, out, *, quantized, counts,
                 raw_quantized, token_scale, projected):
        e, k, n = ends.numel(), x.shape[-1], out.shape[-1]
        m, kp = x.shape[0], (k+63)//64*64
        specs = ((x, (m,k), torch.bfloat16),
                 (weight, (e,kp,n//8), torch.int32),
                 (scale, (e,1,n), torch.int64),
                 (bias, (e,n), torch.float32),
                 (ends, (e,), torch.int64),
                 (out, (m,n), torch.bfloat16),
                 (quantized, (m,kp), torch.int8),
                 (counts, (e,), torch.int64),
                 (raw_quantized, (m,k), torch.int8),
                 (token_scale, (m,), torch.float32),
                 (projected, (m,n), torch.float16))
        if e not in (128,384) or (k,n) not in ((5120,576),(288,5120)) or m <= 0:
            raise ValueError('unsupported decode expert geometry')
        for tensor, shape, dtype in specs:
            if (tuple(tensor.shape)!=shape or tensor.dtype!=dtype
                    or tensor.device.type!='npu' or tensor.device!=x.device
                    or not tensor.is_contiguous() or tensor.data_ptr()%32):
                raise ValueError('invalid prepared decode projection storage')
        held = tuple(t for t,_,_ in specs)
        for i,t in enumerate(held):
            if any(_overlaps(t,other) for other in held[:i]):
                raise ValueError('projection storage overlaps')
        self.storage = (x,weight,out,ends)
        self.scale, self.bias = scale, bias
        self._held = held
        self.q, self.raw, self.projected = quantized, raw_quantized, projected
        if m > 144:
            raise ValueError('decode input preparation supports at most 144 routed rows')
        # Resolve fixed pointers before capture; no runtime compile or fallback.
        self._prepare_lib = C.CDLL(str(Path(__file__).with_name('libdecode_w4_prepare.so')))
        self._prepare = self._prepare_lib.dec_w4_prepare
        self._prepare.argtypes = [P]*5 + [C.c_int]*3
        self._prepare.restype = C.c_int
        self._prepare = queued(self._prepare, self._prepare.argtypes, check_status=True)
        self._prepare_args = tuple(P(t.data_ptr()) for t in
                                   (raw_quantized, quantized, ends, counts)) + (m,k,e)
        self.handles, self.executors, self.steps = [], [], []
        self.workspace, self.closed, self.workspace_bytes = None, False, 0
        try:
            self._plan('DynamicQuant', [P]*4,
                       self._tensor(x), P(), self._tensor(raw_quantized),
                       self._tensor(token_scale))
            # The prepared HP ABI exposes NZ bytes through an ND INT4 logical
            # descriptor, as the torch_npu int32-packed interface does.
            w = self._tensor(weight, shape=(e,kp,n), stride=(kp*n,n,1), dtype=29)
            tuning = _fn('aclCreateIntArray', [P,U], P)((I*2)(0,1), 2)
            if not tuning:
                raise RuntimeError('CANN tuning descriptor failed')
            self.handles.append(('aclDestroyIntArray', tuning))
            self._plan('GroupedMatmulV5', [P]*12+[I]*4+[P]*4,
                       self._list(self._tensor(quantized)), self._list(w),
                       self._list(self._tensor(bias)), self._list(self._tensor(scale)),
                       P(), P(), P(), self._list(self._tensor(token_scale)),
                       self._tensor(counts), P(), P(), P(),
                       3, 0, 1, 0, P(tuning), self._list(self._tensor(projected)),
                       P(), P())
        except BaseException:
            self.close()
            raise

    def _tensor(self, t, *, shape=None, stride=None, dtype=None):
        shape = tuple(t.shape) if shape is None else shape
        stride = tuple(t.stride()) if stride is None else stride
        dims, steps = (I*len(shape))(*shape), (I*len(stride))(*stride)
        h = _fn('aclCreateTensor', [P,U,C.c_int,P,I,C.c_int,P,U,P], P)(
            dims, len(shape), _DT[t.dtype] if dtype is None else dtype,
            steps, 0, 2, dims, len(shape), P(t.data_ptr()))
        if not h:
            raise RuntimeError('CANN projection tensor descriptor failed')
        self.handles.append(('aclDestroyTensor', h))
        return P(h)

    def _list(self, tensor):
        h = _fn('aclCreateTensorList', [P,U], P)((P*1)(tensor), 1)
        if not h:
            raise RuntimeError('CANN projection tensor list failed')
        self.handles.remove(('aclDestroyTensor', tensor.value))
        self.handles.append(('aclDestroyTensorList', h))
        return P(h)

    def _plan(self, name, signature, *args):
        size, executor = U(), P()
        status = _fn('aclnn'+name+'GetWorkspaceSize', signature+[P,P])(
            *args, C.byref(size), C.byref(executor))
        if executor:
            self.executors.append(executor)
        if status:
            raise RuntimeError(f'CANN {name} planning failed: {status}')
        status = _fn('aclSetAclOpExecutorRepeatable', [P])(executor)
        if status:
            raise RuntimeError(f'CANN {name} repeatable failed: {status}')
        launch = queued(_fn('aclnn'+name, [P,U,P,P]), [P,U,P,P], check_status=True)
        self.steps.append((launch, size.value, executor))
        self.workspace_bytes = max(self.workspace_bytes, size.value)


    def _launch(self, index, stream, workspace):
        fn, size, executor = self.steps[index]
        status = fn(P(workspace.data_ptr()), size, executor, stream)
        if status:
            raise RuntimeError(f'CANN projection step {index} failed: {status}')

    def __call__(self, workspace):
        if self.closed:
            raise RuntimeError('decode projection is not bound')
        stream = P(torch.npu.current_stream().npu_stream)
        self._launch(0, stream, workspace)
        # HP GMM may mutate scratch. Restore quantized input and its padding.
        status = self._prepare(stream, *self._prepare_args)
        if status:
            raise RuntimeError(f'decode input preparation failed: {status}')
        self._launch(1, stream, workspace)
        self.storage[2].copy_(self.projected)
        return self.storage[2]


    def close(self):
        for executor in reversed(self.executors):
            _fn('aclDestroyAclOpExecutor', [P])(executor)
        self.executors.clear()
        for name, handle in reversed(self.handles):
            _fn(name, [P])(P(handle))
        self.handles.clear()
        self.closed = True
        self.workspace = None
