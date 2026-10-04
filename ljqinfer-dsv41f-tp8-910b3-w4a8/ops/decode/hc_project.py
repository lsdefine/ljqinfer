"""Decode-only fused HC prepare+projection. Caller-owned storage; zero scratch."""
import ctypes as C
import math
from pathlib import Path
from ops.queued import queued

class HCProject:
    def __init__(self, x, weight, z, stats, *, eps):
        import torch
        rows = x.shape[0]
        expected = ((rows,20480),(24,20480),(rows,24),(rows,))
        self.storage = (x,weight,z,stats)
        if rows not in (6,12,18,24) or not math.isfinite(eps) or not 0 < eps <= 3.402823466e38:
            raise ValueError('Q6 B1..4 and finite positive epsilon required')
        for i,(t,shape) in enumerate(zip(self.storage,expected)):
            if (tuple(t.shape)!=shape or t.dtype!=(torch.bfloat16 if i==0 else torch.float32)
                or t.device!=x.device or t.device.type!='npu' or not t.is_contiguous() or t.data_ptr()%32):
                raise ValueError('invalid HC storage')
            for u in self.storage[:i]:
                if (t.data_ptr()<u.data_ptr()+u.numel()*u.element_size()
                        and u.data_ptr()<t.data_ptr()+t.numel()*t.element_size()):
                    raise ValueError('overlapping HC storage')
        self.library=C.CDLL(str(Path(__file__).with_name('libdecode_hc_project.so')))
        self.fn=self.library.dec_hc_project
        self.fn.argtypes=[C.c_void_p]*5+[C.c_uint32,C.c_float]
        self.fn.restype=C.c_int
        self.fn=queued(self.fn, self.fn.argtypes, check_status=True)
        self.args=tuple(C.c_void_p(t.data_ptr()) for t in self.storage)+(rows//6,eps)
        self.device=x.device
        self.workspace_bytes=0
        self.workspace, self.closed = None, False

    def close(self):
        self.closed = True

    def __call__(self, workspace):
        if self.closed:
            raise RuntimeError('HC projection is not bound and live')
        import torch
        status=self.fn(C.c_void_p(torch.npu.current_stream(self.device).npu_stream),*self.args)
        if status: raise RuntimeError(f'dec_hc_project: {status}')
