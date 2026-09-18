"""Live-prefix peer all-reduce for the indexer score [rows, C] fp32.

Reduces only columns [0, lim_s) of each row, lim_s = (pos[s]+1)//ratio read on
device -> graph-safe, capacity-independent.  Fixed reduction order j=0..R-1
(deterministic, == sequential sum of the R inputs; NOT bit-eq to NCCL ring).
PEER_AR_ROWS=0 forces NCCL.  Registration is collective; never inside capture.
"""
import os
import torch
import torch.distributed as dist
from torch.utils.cpp_extension import load

_ROOT = os.path.dirname(os.path.abspath(__file__))
_ext = None
_ready = False
_failed = False
_n = 0
_ENABLED = os.environ.get("PEER_AR_ROWS", "1") == "1"


def _mod():
    global _ext
    if _ext is None:
        _ext = load(name="peer_ar_rows", sources=[os.path.join(_ROOT, "peer_ar_rows.cu")],
                    extra_cuda_cflags=["-O3", "-lineinfo"],
                    build_directory=os.path.join(_ROOT, ".build"), verbose=False)
    return _ext


def init(n_max: int) -> bool:
    """Register mailbox for up to n_max fp32 elements. Collective, same n on all ranks."""
    global _ready, _n, _failed
    if _ready:
        return _n >= n_max
    try:
        ext = _mod()
        r, w = dist.get_rank(), dist.get_world_size()
        h = ext.alloc(r, w, n_max).cuda()
        parts = [torch.empty_like(h) for _ in range(w)]
        dist.all_gather(parts, h)
        ext.open(torch.stack(parts).cpu().contiguous())
        torch.cuda.synchronize()
        dist.barrier()
        _ready, _n = True, n_max
    except Exception as e:  # noqa
        _failed = True
        if dist.get_rank() == 0:
            print(f"[peer_ar_rows] init failed -> NCCL fallback: {e}", flush=True)
    return _ready


def all_reduce_rows_(score: torch.Tensor, pos: torch.Tensor, ratio: int) -> torch.Tensor:
    """In-place. score fp32 [rows, C] contiguous; pos int64 [rows]."""
    if _ENABLED and _ready and score.numel() <= _n and score.dtype == torch.float32:
        _mod().run(score, pos, ratio)
    else:
        dist.all_reduce(score)
    return score


def ready() -> bool:
    return _ready


def prewarm(n_max: int) -> bool:
    """Register before graph capture (collective, eager only). No-op if disabled/failed/ready."""
    if not (_ENABLED and not _failed and not _ready and dist.is_initialized() and dist.get_world_size() > 1):
        return _ready
    if torch.cuda.is_current_stream_capturing():
        return False
    init(int(n_max))
    if dist.get_rank() == 0:
        print(f"[peer_ar_rows] prewarm n_max={n_max} ready={_ready} failed={_failed}", flush=True)
    return _ready


def nmax_for_pool(pool, rows: int) -> int:
    """Mailbox capacity: `rows` decode rows x max indexer C over pool.layers (4-aligned)."""
    cmax = 0
    for past in pool.layers.values():
        ip = getattr(past, "idx_pool", None)
        if ip is not None:
            cmax = max(cmax, int(ip.max_rows() if callable(ip.max_rows) else ip.max_rows))
    return max(4, (int(rows) * cmax + 3) // 4 * 4)
