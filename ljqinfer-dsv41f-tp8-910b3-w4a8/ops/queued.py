"""Post a hand-written kernel's launches to the torch_npu task queue.

The engine launches its own kernels by address, which bypasses the task queue:
a raw launch can overtake torch ops that are still sitting in it, and with
TASK_QUEUE_ENABLE=1 that corrupts attention (we measured it as an out of range
access from the stale pointers). Wrapping the entry point keeps every call site
unchanged -- the wrapper looks like the ctypes function it replaces -- while the
launch itself is handed to the queue, in order, with its arguments copied.
"""

import ctypes as C

from . import qlaunch

# How each ctypes argument travels to the kernel: a register word, or, for
# float and double, a floating point register, which is why the kinds string
# has to follow the signature rather than treat everything as a word.
_KINDS = {
    C.c_void_p: 'p', C.c_char_p: 'p',
    C.c_int: 'i', C.c_int32: 'i', C.c_uint32: 'u',
    C.c_int64: 'q', C.c_uint64: 'q', C.c_size_t: 'q',
    C.c_float: 'f', C.c_double: 'd',
}


class Queued:
    """One kernel entry point, called exactly like the ctypes function."""

    __slots__ = ('_fn', '_kinds', '_name', '_check_status')

    def __init__(self, fn, kinds, name='kernel', check_status=False):
        self._fn = C.cast(fn, C.c_void_p).value
        self._kinds = kinds
        self._name = name
        self._check_status = check_status

    def __call__(self, *args):
        # ctypes scalars carry their word in .value; a null pointer reports
        # None there, and the word it stands for is 0.
        values = [getattr(a, 'value', a) for a in args]
        values = [0 if v is None else v for v in values]
        qlaunch.post(self._fn, values, kinds=self._kinds, name=self._name,
                     check_status=self._check_status)
        return 0  # Submission only; checked ACL results are handled by PTA.


def queued(fn, argtypes, name=None, *, check_status=False):
    """Wrap fn, reading the kinds string off the signature it declares."""
    try:
        kinds = ''.join(_KINDS[t] for t in argtypes)
    except KeyError as exc:
        raise TypeError(f'queued: unsupported argument type {exc}') from exc
    return Queued(fn, kinds, name or getattr(fn, '__name__', 'kernel'), check_status)
