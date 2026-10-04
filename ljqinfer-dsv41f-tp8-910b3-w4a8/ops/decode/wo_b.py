"""Fixed-address wo_b only: BF16 operands, FP32 accumulation/output, B=1..4.

No JIT, allocation, executor setup, synchronization or precision-policy changes
in __call__. Caller retains this plan until captured graphs are destroyed and
in-flight work is complete. Build libdecode_wo_b_cube.so explicitly before startup.
Caller supplies startup-owned immutable tiling matching X rows.
"""
import ctypes as C
from pathlib import Path
import torch
from ops.queued import queued

P = C.c_void_p


def _overlap(a, b):
    return (a.data_ptr() < b.data_ptr()+b.numel()*b.element_size()
            and b.data_ptr() < a.data_ptr()+a.numel()*a.element_size())


class WoB:
    _k, _n = 1024, 5120
    _rows = (6, 12, 18, 24)
    _library, _entry = "libdecode_wo_b_cube.so", "dec_wo_b_cube"

    def __init__(self, x, weight, out, tiling):
        if (x.ndim != 2 or tuple(x.shape) not in
                tuple((m, self._k) for m in self._rows)
                or tuple(weight.shape) != (self._n, self._k)
                or tuple(out.shape) != (x.shape[0], self._n)):
            raise ValueError(f'{type(self).__name__} requires X[M,{self._k}], '
                             f'W[{self._n},{self._k}], Y[M,{self._n}], M in {self._rows}')
        if x.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16 or out.dtype != torch.float32:
            raise ValueError('wo_b requires BF16/BF16 -> FP32; no implicit casts')
        if x.device.type != 'npu' or any(t.device != x.device for t in (weight, out)):
            raise ValueError('wo_b storage must share one owning NPU')
        if any(not t.is_contiguous() or t.data_ptr() % 32 for t in (x, weight, out)):
            raise ValueError('wo_b storage must be contiguous and 32-byte aligned')
        if _overlap(out, x) or _overlap(out, weight):
            raise ValueError('wo_b output overlaps an input')
        if (tiling is None or tiling.dtype != torch.uint8
                or tiling.device != x.device or tuple(tiling.shape) != (224,)
                or not tiling.is_contiguous() or tiling.data_ptr() % 32):
            raise ValueError('wo_b requires caller-owned aligned 224-byte tiling')
        if any(_overlap(tiling, t) for t in (x, weight, out)):
            raise ValueError('wo_b tiling overlaps operands')
        self.storage, self.device = (x, weight, out, tiling), x.device
        # Loading occurs during plan construction, never during capture/replay.
        self.library = C.CDLL(str(Path(__file__).with_name(self._library)))
        self.launch = getattr(self.library, self._entry)
        self.launch.argtypes = [P, P, P, P, P, C.c_uint32]
        self.launch.restype = C.c_int
        self.launch = queued(self.launch, self.launch.argtypes, check_status=True)
        self.args = (P(x.data_ptr()), P(weight.data_ptr()), P(out.data_ptr()),
                     P(tiling.data_ptr()), int(x.shape[0]))
        self.workspace_bytes = 0
        self.workspace, self.closed = None, False


    def __call__(self, workspace):
        if self.closed:
            raise RuntimeError('wo_b is not bound')
        status = self.launch(P(torch.npu.current_stream(self.device).npu_stream), *self.args)
        if status:
            raise RuntimeError(f'wo_b launch rejected: {status}')
        return self.storage[2]


    def close(self):
        # Same lifecycle rule as Matmul: destroy graphs/wait before closing.
        self.closed = True


class DraftVocab(WoB):
    """Draft-only BF16 [6*B,5120] x [16160,5120] -> FP32 vocabulary."""
    _k, _n = 5120, 16160
    _library, _entry = "libdecode_draft_vocab_cube.so", "dec_draft_vocab_cube"


class MarkovHead(WoB):
    """Draft Markov BF16 [B,256] x [16160,256] -> compact FP32 bias."""
    _rows = (1, 2, 3, 4)
    _k, _n = 256, 16160
    _library, _entry = "libdecode_markov_cube.so", "dec_markov_cube"


class DraftSharedDown(WoB):
    """Draft shared expert BF16 [6*B,288] x [5120,288] -> FP32."""
    _k, _n = 288, 5120
    _library, _entry = "libdecode_draft_down_cube.so", "dec_draft_down_cube"
