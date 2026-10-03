"""Fused row-id construction for sparse_paged decode attention.

sparse_paged() used to build ``ids`` with ~12 ATen long-tensor kernels per
layer (arange/sub/ge/and/lt/remainder/add/where/full_like/cat/to); at 40
layers per step that is ~500 launches of pure index arithmetic.  This
kernel emits the identical [Q, topk+window] int64 tensor in one launch.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _attn_ids(POS, SEL, OUT, ns, pad, ratio,
              WINDOW: tl.constexpr, RING: tl.constexpr, TOPK: tl.constexpr, BLOCK: tl.constexpr,
              QWIN: tl.constexpr):
    # start/total/local_start are derived from the first position of this
    # row's own request on device, so the kernel is graph-replayable at any
    # position (no host ints baked into the graph).  QWIN rows belong to one
    # request; QWIN == the row count collapses this to the single-sequence
    # form where every row reads POS[0].
    row = tl.program_id(0)
    seq = row // QWIN
    start = tl.load(POS + seq * QWIN).to(tl.int64)
    total = start // ratio
    local_start = tl.maximum(start - (WINDOW - 1), 0)
    p = tl.load(POS + row).to(tl.int64)
    j = tl.arange(0, BLOCK)
    if TOPK > 0:
        sm = j < TOPK
        sel = tl.load(SEL + row * TOPK + j, mask=sm, other=-1).to(tl.int64)
        live = (sel >= 0) & (sel < (p + 1) // ratio) & (sel < total + ns)
        tl.store(OUT + row * (TOPK + WINDOW) + j, tl.where(live, sel, -1), mask=sm)
    wm = j < WINDOW
    swa = p - (WINDOW - 1 - j)
    ok = (swa >= local_start) & (swa >= 0)
    committed = total + pad + swa % RING
    staged = total + ns + swa - start
    v = tl.where(swa < start, committed, staged)
    v = tl.where(ok, v, -1)
    tl.store(OUT + row * (TOPK + WINDOW) + TOPK + j, v, mask=wm)


def attn_ids(positions, selected, *, ns, pad, window, ring, ratio, qwin=None):
    """Same values as the torch.where/cat chain in sparse_paged (int64, contiguous).

    positions must be arange(start, start+Q) per request on device; start,
    total=start//ratio and local_start=max(0, start-window+1) are read from
    positions[seq*qwin] inside the kernel, so the launch carries no per-step
    host scalars.  `qwin` is the rows per request (default: all of them, i.e.
    one sequence).
    """
    q = positions.shape[0]
    qwin = q if qwin is None else int(qwin)
    assert q % qwin == 0
    topk = 0 if selected is None else selected.shape[1]
    out = torch.empty((q, topk + window), dtype=torch.int64, device=positions.device)
    sel = positions if selected is None else selected
    assert ring >= window
    _attn_ids[(q,)](positions, sel, out, int(ns), int(pad), int(ratio), window, ring, topk,
                    triton.next_power_of_2(max(window, topk)), qwin, num_warps=4)
    return out
