"""Decode-only fixed-address BF16 or FP32 GEMM. Build once, bind scratch, capture.

Physical weight [N,K], x [B*6,K], out [B*6,N]. Same-dtype operands;
BF16 permits BF16/FP32 output, FP32 requires FP32 output. No implicit casts.
Weight unpacking is an explicit upstream leaf, never a hidden fallback.
Caller keeps this plan alive until its graphs are destroyed and synchronizes
before close(). Workspace cannot be shared across concurrently running plans.
"""
import ctypes as C
import torch
from ops.queued import queued

P, I, U = C.c_void_p, C.c_int64, C.c_uint64
_LIB = C.CDLL('libopapi.so')

def _fn(name, args, result=C.c_int):
    f = getattr(_LIB, name)
    f.argtypes, f.restype = args, result
    return f

class _Projection:
    """Own CANN descriptors/executors; caller owns tensors and serial scratch.

    Destroy captured graphs and drain streams before closing either projection.
    """
    def __init__(self):
        self.handles, self.executors, self.steps = [], [], []
        self.closed, self.workspace, self.workspace_bytes = False, None, 0

    def _tensor(self, t, shape=None, stride=None):
        shape = tuple(t.shape) if shape is None else shape
        stride = tuple(t.stride()) if stride is None else stride
        dims, strides = (I*len(shape))(*shape), (I*len(stride))(*stride)
        base = (I*t.ndim)(*t.shape)
        dtype = {torch.float32: 0, torch.bfloat16: 27, torch.int8: 2}[t.dtype]
        handle = _fn('aclCreateTensor', [P,U,C.c_int,P,I,C.c_int,P,U,P], P)(
            dims, len(shape), dtype, strides, 0, 2, base, t.ndim, P(t.data_ptr()))
        if not handle:
            raise RuntimeError('projection tensor descriptor creation failed')
        self.handles.append(P(handle))
        return P(handle)

    def _plan(self, name, args, values):
        size, executor = U(), P()
        status = _fn(name+'GetWorkspaceSize', args)(
            *values, C.byref(size), C.byref(executor))
        if executor:
            self.executors.append(executor)
        if status:
            raise RuntimeError(f'{name} planning failed: {status}')
        status = _fn('aclSetAclOpExecutorRepeatable', [P])(executor)
        if status:
            raise RuntimeError(f'{name} repeatable failed: {status}')
        launch = queued(_fn(name, [P,U,P,P]), [P,U,P,P], name,
                        check_status=True)
        self.steps.append((launch, size.value, executor))
        self.workspace_bytes = max(self.workspace_bytes, size.value)

    def __call__(self, workspace):
        if self.closed:
            raise RuntimeError('projection is closed')
        stream = P(torch.npu.current_stream(self.storage[0].device).npu_stream)
        for fn, size, executor in self.steps:
            status = fn(P(workspace.data_ptr()), size, executor, stream)
            if status:
                raise RuntimeError(f'projection launch failed: {status}')
        return self.storage[2]

    def close(self):
        for executor in reversed(self.executors):
            _fn('aclDestroyAclOpExecutor', [P])(executor)
        self.executors.clear()
        for handle in reversed(self.handles):
            _fn('aclDestroyTensor', [P])(handle)
        self.handles.clear()
        self.steps.clear()
        self.workspace, self.closed = None, True


class Matmul(_Projection):
    def __init__(self, x, weight, out):
        super().__init__()
        self.storage = (x, weight, out)
        if (x.ndim != 2 or weight.ndim != 2 or out.ndim != 2
                or x.shape[0] < 6 or x.shape[0] % 6
                or x.shape[1] != weight.shape[1]
                or out.shape != (x.shape[0], weight.shape[0])
                or x.dtype not in (torch.bfloat16, torch.float32)
                or weight.dtype != x.dtype
                or out.dtype not in (torch.bfloat16, torch.float32)
                or (x.dtype == torch.float32 and out.dtype != torch.float32)
                or x.device.type != 'npu'
                or any(t.device != x.device or not t.is_contiguous()
                       for t in self.storage)):
            raise ValueError('decode GEMM requires contiguous Q6 same-dtype BF16/FP32 operands')
        for t in (x, weight):
            if (out.data_ptr() < t.data_ptr()+t.numel()*t.element_size()
                    and t.data_ptr() < out.data_ptr()+out.numel()*out.element_size()):
                raise ValueError('output aliases an operand')
        try:
            a = self._tensor(x, tuple(x.shape), tuple(x.stride()))
            w = self._tensor(weight, (weight.shape[1], weight.shape[0]),
                             (1, weight.shape[1]))
            y = self._tensor(out, tuple(out.shape), tuple(out.stride()))
            self._plan('aclnnMatmul', [P,P,P,C.c_int,P,P], (a, w, y, 0))
        except BaseException:
            self.close()
            raise


class W8Matmul(_Projection):
    """CANN W8A16 projection with caller-owned scale, output and workspace.

    weight is INT8 [N,K], scale is persistent BF16 [N] or [N,1]. CANN
    consumes the transposed weight view without a full BF16 expansion.
    FP32 output explicitly uses a caller-supplied BF16 projected buffer and
    a Cast plan. No device allocation or conversion is hidden in this plan.
    Graphs must be destroyed and streams drained before close().
    """
    def __init__(self, x, weight, scale, out, *, projected=None):
        super().__init__()
        if (x.ndim != 2 or weight.ndim != 2 or out.ndim != 2
                or x.shape[0] < 6 or x.shape[0] % 6
                or x.dtype != torch.bfloat16 or weight.dtype != torch.int8
                or x.shape[1] != weight.shape[1]
                or out.shape != (x.shape[0], weight.shape[0])
                or out.dtype not in (torch.bfloat16, torch.float32)
                or scale.dtype != torch.bfloat16
                or tuple(scale.shape) not in ((weight.shape[0],), (weight.shape[0], 1))):
            raise ValueError('W8 requires BF16 X, INT8 [N,K], BF16 channel scale and BF16/FP32 output')
        y = out if out.dtype == torch.bfloat16 else projected
        if (y is None or y.dtype != torch.bfloat16 or y.shape != out.shape
                or (out.dtype == torch.bfloat16 and projected is not None)):
            raise ValueError('FP32 W8 output requires explicit BF16 projected storage')
        tensors = (x, weight, scale, out, y)
        if (x.device.type != 'npu'
                or any(t.device != x.device or not t.is_contiguous()
                       or t.data_ptr() % 32 for t in tensors)):
            raise ValueError('W8 buffers must be contiguous aligned storage on one NPU')
        if (any(self._overlap(out, t) for t in (x, weight, scale))
                or (y is not out and any(self._overlap(y, t)
                                        for t in (x, weight, scale, out)))):
            raise ValueError('W8 output/projected overlaps live storage')
        self.storage = (x, weight, out)  # Shared-expert binding contract.
        self._held = (scale, y)
        n, k = weight.shape
        try:
            a = self._tensor(x)
            w = self._tensor(weight, (k, n), (1, k))
            s = self._tensor(scale, (n,), (1,))
            o = self._tensor(y)
            self._plan('aclnnWeightQuantBatchMatmulV2', [P]*7 + [C.c_int, P, P, P],
                       (a, w, s, P(), P(), P(), P(), 0, o))
            if y is not out:
                self._plan('aclnnCast', [P, C.c_int, P, P, P],
                           (o, 0, self._tensor(out)))
        except BaseException:
            self.close()
            raise

    @staticmethod
    def _overlap(a, b):
        return (a.numel() and b.numel()
                and a.data_ptr() < b.data_ptr() + b.numel()*b.element_size()
                and b.data_ptr() < a.data_ptr() + a.numel()*a.element_size())
