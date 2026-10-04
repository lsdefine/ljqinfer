"""One-shot native launch through the PTA queue; no bound execution recipes."""
import ctypes as C
from ops import qlaunch


def native(lib, name, tensors, *scalars, check_status=False):
    fn = getattr(lib, name)
    pointers = [0 if t is None else t.data_ptr() for t in tensors]
    kinds = 'p' * (len(pointers) + 1)
    kinds += ''.join('f' if isinstance(x, float) else 'i' for x in scalars)
    qlaunch.post(C.cast(fn, C.c_void_p).value, [0, *pointers, *scalars],
                 kinds=kinds, stream_slot=0, name=name, check_status=check_status)
