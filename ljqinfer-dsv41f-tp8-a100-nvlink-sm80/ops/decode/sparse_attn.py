"""Fused sparse latent attention (decode): the gather happens in the kernel."""
from functools import lru_cache
from pathlib import Path
import os
import torch


@lru_cache(maxsize=1)
def extension():
    from torch.utils.cpp_extension import load
    os.environ.setdefault('TORCH_CUDA_ARCH_LIST', '8.0')
    os.environ.setdefault('MAX_JOBS', '2')
    return load(name='v41_decode_sparse_attn',
                sources=[str(Path(__file__).parent/'cuda'/'sparse_attn_paged.cu')],
                extra_cuda_cflags=['-O3'], verbose=False)


@lru_cache(maxsize=None)
def _no_rowmap(device):
    """Cached so a captured graph keeps pointing at the same empty buffer."""
    return torch.empty(0, dtype=torch.int64, device=device)


def attend(q, pool, page_table, tail, sink, idxs, out, pos, ratio, scale,
           qwin=None, rowmap=None):
    """q [T,H,512] against pool rows (id < total, paged) and tail rows
    (id >= total), total = pos[seq*qwin] // ratio read on device so the launch
    is graph-replayable.  Ids < 0 are dead slots.  Writes and returns out.

    `qwin` is how many consecutive rows of T belong to one request, so
    T // qwin requests share the call; page_table [nreq, pages] and tail
    [nreq, rows, 512] then carry one entry each.  The default keeps every row
    on one sequence with a flat page table and tail bank.

    Requests rarely own neighbouring rows of the state pools, so `rowmap`
    [nreq] maps request -> pool row: pass the whole page table / tail bank and
    the batch reads its slots in place, with no gather into a packed copy.
    Without it request b reads row b."""
    if qwin is None:
        qwin = q.shape[0]
    extension().sparse_attn_decode(q, pool, page_table, tail, sink, idxs, out,
                                   pos, int(ratio), float(scale), int(qwin),
                                   _no_rowmap(q.device) if rowmap is None else rowmap)
    return out
