"""Post hand-written kernel launches through the torch_npu task queue.

The engine's own kernels are launched by address rather than through aclnn.
Issued straight onto the stream they can overtake work still sitting in the
torch_npu task queue, which is why the engine had to run with
TASK_QUEUE_ENABLE=0. A Launch freezes one argument list and posts the launch
to that same queue, so ordering holds and the queue can be switched back on.
"""
import os

import torch
import torch_npu

_HERE = os.path.dirname(os.path.abspath(__file__))
_TN = os.path.dirname(torch_npu.__file__)
_CANN = os.environ.get('ASCEND_HOME_PATH',
                       '/usr/local/Ascend/ascend-toolkit/latest')
_BUILD = os.environ.get('LJQ_QLAUNCH_BUILD', '/data/custom_ops/qbuild')

_module = None


def _load():
    """Build the extension once; later runs hit the ninja cache."""
    global _module
    if _module is None:
        from torch.utils.cpp_extension import load
        abi = int(torch._C._GLIBCXX_USE_CXX11_ABI)
        os.makedirs(_BUILD, exist_ok=True)
        _module = load(
            name='qlaunch',
            sources=[os.path.join(_HERE, 'kernels', 'qlaunch.cpp')],
            extra_include_paths=[f'{_TN}/include',
                                 f'{_TN}/include/third_party/op-plugin',
                                 f'{_CANN}/include'],
            extra_cflags=[f'-D_GLIBCXX_USE_CXX11_ABI={abi}',
                          '-std=c++17', '-O2'],
            extra_ldflags=[f'-L{_TN}/lib', '-ltorch_npu',
                           f'-Wl,-rpath,{_TN}/lib', '-lffi'],
            build_directory=_BUILD,
            verbose=False,
        )
    return _module


def _coerce(args, kinds):
    """Match each argument to the register class its kind asks for."""
    if kinds is None:
        kinds = 'q' * len(args)
    if len(kinds) != len(args):
        raise ValueError('qlaunch: kinds must describe every argument')
    values = [float(a) if k in 'fd' else int(a) for k, a in zip(kinds, args)]
    return values, kinds


def post(fn, args, *, kinds=None, stream_slot=-1, name='qlaunch', check_status=False):
    """Post once; check_status forwards an int32 ACL result to the task queue."""
    values, kinds = _coerce(args, kinds)
    return _load().post(int(fn), values, kinds,
                        stream_slot=stream_slot, name=name, check_status=check_status)


def bind(fn, args, *, kinds=None, stream_slot=-1, name='qlaunch'):
    """Freeze one launch. fn is the kernel address, args its arguments.

    kinds names the C type of each argument, one character each:
    p pointer, u uint32, i int32, q int64, f float, d double. It defaults to
    integer words, which is what most of the engine's launchers take.
    """
    values, kinds = _coerce(args, kinds)
    return _load().Launch(int(fn), values, kinds,
                          stream_slot=stream_slot, name=name)
