"""Decode-only fused residual gather for Source.fold. One launch per batch.

Replaces the index shuffling of the eager decode branch in model.past.Source.fold
(the ``pos0 is not None and not write`` case): arange, sub, clamp, add, remainder,
two ring index_selects, two window index_selects, two casts and two wheres --
thirteen kernels that between them touch only a handful of rows.

One call serves a whole batch: request ``b`` owns projected rows
``[b*n, (b+1)*n)``, reads the residual ring of pool row ``SLOTS[b]`` and writes
output rows ``[b*rows, (b+1)*rows)``.  A lone request is a batch of one, so
there is no separate single-slot path.

This kernel moves bytes and casts; it performs no float arithmetic, so the fp32
rows it emits are bit-identical to the eager ones and the softmax/weighted sum
stay in torch (triton fp32 exp differs from CUDA expf by 1 ulp, which would
break bit-exactness against the prefill reference).
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _fold_gather(V, S, KVS, SCS, POS0, SLOTS, OV, OS,
                 vs0, vs1, kslot, ks0, ks1, os0, os1, ps,
                 n, rows, ring, d, R: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    o = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    b = tl.program_id(2)
    m = o < d
    pos0 = tl.load(POS0 + b * ps)
    slot = tl.load(SLOTS + b)
    rel = row - pos0 % R
    cur = tl.minimum(tl.maximum(rel, 0), n - 1)
    rr = (pos0 + rel) % ring
    take = rel >= 0
    vw = tl.load(V + (b * n + cur) * vs0 + o * vs1, mask=m, other=0.0).to(tl.float32)
    sw = tl.load(S + (b * n + cur) * vs0 + o * vs1, mask=m, other=0.0).to(tl.float32)
    vr = tl.load(KVS + slot * kslot + rr * ks0 + o * ks1, mask=m, other=0.0)
    sr = tl.load(SCS + slot * kslot + rr * ks0 + o * ks1, mask=m, other=0.0)
    tl.store(OV + (b * rows + row) * os0 + o * os1, tl.where(take, vw, vr), mask=m)
    tl.store(OS + (b * rows + row) * os0 + o * os1, tl.where(take, sw, sr), mask=m)


def fold_gather(values, scores, kv_state, score_state, pos0, slots, ratio,
                *, rows_per_request=None):
    """fp32 [B*groups*ratio, D] value/score rows, ring rows where rel < 0.

    ``values``/``scores`` carry ``B * n`` rows, request-major.  ``kv_state`` and
    ``score_state`` are the whole residual pools, indexed by ``slots`` (an int32
    device tensor of pool rows, one per request).  ``pos0`` holds each request's
    window start; its stride between requests is ``rows_per_request`` (the row
    count of one request's window in the live position vector).
    """
    assert values.stride() == scores.stride()
    assert kv_state.stride() == score_state.stride()
    assert kv_state.dim() == 3 and slots.numel() >= 1
    b = int(slots.numel())
    total, d = values.shape
    assert total % b == 0
    n = total // b
    rows = ((n + ratio - 1) // ratio) * ratio
    v = torch.empty((b * rows, d), dtype=torch.float32, device=values.device)
    sc = torch.empty_like(v)
    block = min(1024, triton.next_power_of_2(d))
    ps = n if rows_per_request is None else int(rows_per_request)
    _fold_gather[(rows, triton.cdiv(d, block), b)](
        values, scores, kv_state, score_state, pos0, slots, v, sc,
        values.stride(0), values.stride(1),
        kv_state.stride(0), kv_state.stride(1), kv_state.stride(2),
        v.stride(0), v.stride(1), ps,
        n, rows, 2 * ratio, d, R=ratio, BLOCK=block, num_warps=4)
    return v, sc
