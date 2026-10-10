"""Fixed-Q6 KV normalization, rotary embedding and simulated FP8 QDQ.

Caller owns contiguous NPU storage: x/out BF16 [B*6,512], weight FP32
[512], freqs FP32 [B*6,64], B=1..4. x may alias out. Both original BF16
rounding boundaries and QDQ signed zero are preserved. No allocation,
collective or synchronization; submit through the current-stream task queue.
"""
import ctypes as C
from pathlib import Path
from ops import qlaunch

_lib = C.CDLL(str(Path(__file__).with_name('libdecode_kv_prepare.so')))
_address = C.cast(_lib.dec_kv_prepare, C.c_void_p).value


def kv_prepare(x, weight, freqs, out, batch, eps):
    """Submit one leaf; C ABI validates alignment, aliases, batch and epsilon."""
    qlaunch.post(_address, [0, *(t.data_ptr() for t in (x, weight, freqs, out)), batch, eps],
                 kinds='pppppuf', stream_slot=0, name='dec_kv_prepare', check_status=True)
