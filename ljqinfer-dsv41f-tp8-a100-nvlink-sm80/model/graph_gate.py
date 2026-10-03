"""Let a captured body wait on host work that cannot be captured.

Engram addressing depends only on the token window, so the row gather is free
to run while earlier layers compute.  A captured graph cannot contain the host
gather, but it can contain a stream memory-op that blocks until the host says
the rows have landed, which is what keeps the gather off the launch path.
"""
import ctypes

import torch

_WAIT_GEQ = 2  # CU_STREAM_WAIT_VALUE_GEQ
_lib = None


def _driver():
    """Bind the two stream memory-ops; torch exposes no wrapper for them."""
    global _lib
    if _lib is None:
        lib = ctypes.CDLL('libcuda.so')
        for name in ('cuStreamWaitValue32_v2', 'cuStreamWriteValue32_v2'):
            fn = getattr(lib, name)
            fn.restype = ctypes.c_int
            fn.argtypes = [ctypes.c_void_p, ctypes.c_ulonglong,
                           ctypes.c_uint, ctypes.c_uint]
        _lib = lib
    return _lib


def _call(fn, stream, ptr, value, flags):
    rc = fn(ctypes.c_void_p(stream), ctypes.c_ulonglong(ptr),
            ctypes.c_uint(value), ctypes.c_uint(flags))
    if rc:
        raise RuntimeError(f'{fn.__name__} failed with CUDA driver error {rc}')


class RowGate:
    """A one-shot flag: the captured body waits on it, the host raises it.

    The wait is recorded only while capturing, so eager callers keep reading
    the buffer directly and can never strand themselves on an unraised gate.
    """

    def __init__(self, device):
        self.flag = torch.zeros(1, dtype=torch.int32, device=device)
        self.stream = torch.cuda.Stream(device=device)

    def wait_in_graph(self):
        """Record 'block until raised, then re-arm' on the capturing stream."""
        if not torch.cuda.is_current_stream_capturing():
            return False
        lib, stream = _driver(), torch.cuda.current_stream().cuda_stream
        _call(lib.cuStreamWaitValue32_v2, stream, self.flag.data_ptr(), 1, _WAIT_GEQ)
        # Clearing inside the graph is what makes the next replay wait again;
        # the write orders after the wait, so the rows are already visible.
        _call(lib.cuStreamWriteValue32_v2, stream, self.flag.data_ptr(), 0, 0)
        return True

    def raise_on(self, stream):
        """Raise the gate behind everything already queued on `stream`."""
        _call(_driver().cuStreamWriteValue32_v2, stream.cuda_stream,
              self.flag.data_ptr(), 1, 0)

    def release(self):
        """Unblock a replay whose staging raised: the caller is failing out."""
        self.raise_on(torch.cuda.current_stream())
