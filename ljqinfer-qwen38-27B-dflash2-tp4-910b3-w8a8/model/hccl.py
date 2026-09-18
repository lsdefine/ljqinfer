"""Minimal capture-safe raw HCCL bindings for TP4 SPMD."""
from __future__ import annotations
import ctypes
import glob
import os
import threading
import torch

_DTYPE = {torch.float16: 3, torch.float32: 4, torch.bfloat16: 11}
_SUM = 0
_lib = None
_lock = threading.Lock()
_deterministic_set = False


def _load():
    global _lib
    with _lock:
        if _lib is not None:
            return _lib
        root = (os.environ.get("ASCEND_HOME_PATH") or
                os.environ.get("ASCEND_TOOLKIT_HOME") or
                "/usr/local/Ascend/cann-9.0.1")
        # Respect the explicitly selected CANN provider.  A global sorted glob
        # may otherwise pick an older toolkit (8.2 before 9.0) whose HCCL ABI is
        # incompatible with the already-loaded runtime.
        preferred = root + "/aarch64-linux/lib64/libhccl.so"
        if not os.path.isfile(preferred):
            preferred = root + "/lib64/libhccl.so"
        if os.path.isfile(preferred):
            provider = preferred
        else:
            hits = glob.glob("/usr/local/Ascend/**/lib64/libhccl.so",
                             recursive=True)
            if not hits:
                raise RuntimeError("libhccl.so not found")
            provider = sorted(set(hits))[-1]
        lib = ctypes.CDLL(provider)
        vp = ctypes.c_void_p
        lib.HcclGetRootInfo.argtypes = [vp]
        lib.HcclGetRootInfo.restype = ctypes.c_int32
        lib.HcclCommInitRootInfo.argtypes = [
            ctypes.c_uint32, vp, ctypes.c_uint32, ctypes.POINTER(vp)]
        lib.HcclCommInitRootInfo.restype = ctypes.c_int32
        lib.HcclAllReduce.argtypes = [
            vp, vp, ctypes.c_uint64, ctypes.c_int32, ctypes.c_int32, vp, vp]
        lib.HcclAllReduce.restype = ctypes.c_int32
        lib.HcclAllGather.argtypes = [
            vp, vp, ctypes.c_uint64, ctypes.c_int32, vp, vp]
        lib.HcclAllGather.restype = ctypes.c_int32
        lib.HcclGroupStart.argtypes = []
        lib.HcclGroupStart.restype = ctypes.c_int32
        lib.HcclGroupEnd.argtypes = []
        lib.HcclGroupEnd.restype = ctypes.c_int32
        lib.HcclSetConfig.argtypes = [ctypes.c_int32, ctypes.c_int32]
        lib.HcclSetConfig.restype = ctypes.c_int32
        lib.HcclGetConfig.argtypes = [ctypes.c_int32,
                                      ctypes.POINTER(ctypes.c_int32)]
        lib.HcclGetConfig.restype = ctypes.c_int32
        lib.HcclCommDestroy.argtypes = [vp]
        lib.HcclCommDestroy.restype = ctypes.c_int32
        _lib = lib
        return lib


def ensure_deterministic():
    global _deterministic_set
    if _deterministic_set:
        return
    lib = _load()
    rc = lib.HcclSetConfig(0, ctypes.c_int32(1))
    got = ctypes.c_int32()
    rc2 = lib.HcclGetConfig(0, ctypes.byref(got))
    if rc or rc2 or got.value != 1:
        raise RuntimeError(f"HCCL deterministic config failed {rc=} {rc2=} value={got.value}")
    _deterministic_set = True


def group_start():
    rc = _load().HcclGroupStart()
    if rc:
        raise RuntimeError(f"HcclGroupStart rc={rc}")


def group_end():
    rc = _load().HcclGroupEnd()
    if rc:
        raise RuntimeError(f"HcclGroupEnd rc={rc}")


def all_reduce_comm(comm, tensor, stream=None):
    if tensor.dtype not in _DTYPE or not tensor.is_contiguous():
        raise ValueError("HCCL all_reduce needs contiguous fp16/bf16/fp32")
    ensure_deterministic()
    stream = stream or torch.npu.current_stream(tensor.device)
    ptr = ctypes.c_void_p(tensor.data_ptr())
    rc = _load().HcclAllReduce(ptr, ptr, tensor.numel(), _DTYPE[tensor.dtype],
                               _SUM, comm,
                               ctypes.c_void_p(stream.npu_stream))
    if rc:
        raise RuntimeError(f"HcclAllReduce rc={rc}")


def all_gather_comm(comm, send, recv, stream=None):
    if send.dtype not in _DTYPE or recv.dtype != send.dtype:
        raise ValueError("HCCL all_gather dtype mismatch")
    if not send.is_contiguous() or not recv.is_contiguous():
        raise ValueError("HCCL all_gather needs contiguous buffers")
    stream = stream or torch.npu.current_stream(send.device)
    rc = _load().HcclAllGather(
        ctypes.c_void_p(send.data_ptr()), ctypes.c_void_p(recv.data_ptr()),
        send.numel(), _DTYPE[send.dtype], comm,
        ctypes.c_void_p(stream.npu_stream))
    if rc:
        raise RuntimeError(f"HcclAllGather rc={rc}")
