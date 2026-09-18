"""NVLink peer all-reduce for the TP group -- a drop-in for dist.all_reduce.

Two-shot (reduce-scatter + broadcast) over cudaIpc mailboxes.  Only fp32 vectors
of one registered length take the fast path; everything else falls back to NCCL,
so callers can use this unconditionally.

Registration needs a collective (handle exchange) and cudaIpcOpenMemHandle, so it
happens lazily on the first eager call -- never inside a graph capture.  Set
PEER_AR=0 to force NCCL (A/B testing).
"""
import os

import torch
import torch.distributed as dist

from ops.build import load_wgemm

_ready = False
_failed = False
_n = 0
# one-shot peer all-reduce is an ALTERNATIVE reduction order -> numerics differ.
# Hard-off by default; flip only via the explicit setter (benchmarks).
_ENABLED = False
# only this length is registered: the decode hidden vector [8, 7168].
_N_TARGET = 57344   # fixed; was env PEER_AR_N


def init(n: int) -> bool:
    """Register a mailbox for length-n fp32 vectors.  Collective: all ranks, same n."""
    global _ready, _n
    if _ready:
        return _n == n
    ext = load_wgemm(False)
    r, w = dist.get_rank(), dist.get_world_size()
    h = ext.peer_ar_ipc_alloc(r, w, n).cuda()          # 2 handles as bytes
    parts = [torch.empty_like(h) for _ in range(w)]
    dist.all_gather(parts, h)
    ext.peer_ar_ipc_open(torch.stack(parts).cpu().contiguous())
    torch.cuda.synchronize()
    dist.barrier()
    _ready, _n = True, n
    return True


def _try_init(n: int) -> bool:
    global _failed
    if _failed:
        return False
    try:
        return init(n)
    except Exception as e:                              # noqa: BLE001 - stay on NCCL
        _failed = True
        if dist.get_rank() == 0:
            print(f"[peer_ar] disabled: {e}", flush=True)
        return False


def all_reduce(y: torch.Tensor) -> None:
    """In-place sum across the TP group."""
    if (_ENABLED and not _failed and y.dtype == torch.float32 and y.is_contiguous()
            and y.is_cuda and dist.is_initialized() and dist.get_world_size() > 1):
        if not _ready and y.numel() == _N_TARGET:
            if torch.cuda.is_current_stream_capturing():
                dist.all_reduce(y)                      # cannot register mid-capture
                return
            _try_init(y.numel())
        if _ready and y.numel() == _n:
            load_wgemm(False).peer_ar_ipc_run2(y)
            return
    dist.all_reduce(y)


def set_enabled(v: bool):
    global _ENABLED
    _ENABLED = bool(v)


def prewarm() -> bool:
    """Register the decode-sized buffer while still eager (pre-capture)."""
    if _ENABLED and not _ready and not _failed and dist.is_initialized() and dist.get_world_size() > 1:
        _try_init(_N_TARGET)
        if dist.get_rank() == 0:
            print(f"[peer_ar] prewarm n={_N_TARGET} ready={_ready} failed={_failed}", flush=True)
    return _ready


def ready() -> bool:
    return _ready
