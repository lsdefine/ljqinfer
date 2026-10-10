"""TP-reduced output -> rounded BF16 branch -> HC residual expansion.

B1..4, Q6, D5120, HC4. Caller owns contiguous NPU storage:
  value/shared: FP32 [B*6,5120]; residual/out: BF16 [B*6,20480];
  post: FP32 [B*6,4]; comb: FP32 [B*6,4,4]. out may alias residual.
shared=None selects attention; MoE preserves BF16(shared) before addition.
No allocation, collective or synchronization. Launch on the current PTA stream;
caller must finish TP reduction and wait for HC gates before calling.
Build: bash ops/decode/build.sh ops/decode/hc_finish.cpp ops/decode/libdecode_hc_finish.so
"""
import ctypes as C
from pathlib import Path
from ops import qlaunch

_library = C.CDLL(str(Path(__file__).with_name('libdecode_hc_finish.so')))
_address = C.cast(_library.dec_hc_finish, C.c_void_p).value


def hc_finish(value, shared, residual, post, comb, out, batch):
    """Submit one fused leaf; C ABI validates batch, alignment and output aliases."""
    tensors = (value, shared, residual, post, comb, out)
    qlaunch.post(_address, [0, *(0 if t is None else t.data_ptr() for t in tensors), batch],
                 kinds='pppppppi', stream_slot=0, name='dec_hc_finish', check_status=True)
