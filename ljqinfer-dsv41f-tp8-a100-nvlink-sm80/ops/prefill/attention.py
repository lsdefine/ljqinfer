"""Prefill-only attention math; bounded query tiles, explicit absolute positions.

KV history/paging belongs to model state adapter, not these operators. Sparse
rows are logical global-history indices; -1 means masked. No decode dispatch.
"""
import torch
from .selection import topk as select_topk


def rope(x, freqs, *, inverse=False):
    d = freqs.shape[-1] * 2
    y = x.clone()
    z = torch.view_as_complex(x[..., -d:].float().reshape(*x.shape[:-1], d//2, 2))
    f = freqs
    while f.ndim < z.ndim:
        f = f.unsqueeze(-2)
    z = z * (f.conj() if inverse else f)
    y[..., -d:] = torch.view_as_real(z).flatten(-2).to(x.dtype)
    return y


def fp8_roundtrip(x, block=32):
    """Training's power-of-two block scaling, preserving BF16 storage."""
    if x.shape[-1] % block:
        raise ValueError('FP8 block alignment')
    z = x.float().unflatten(-1, (-1, block))
    scale = torch.exp2(torch.ceil(torch.log2(z.abs().amax(-1, keepdim=True).clamp_min(1e-4) / 448)))
    return ((z / scale).clamp(-448, 448).to(torch.float8_e4m3fn).float() * scale).flatten(-2).to(x.dtype)


def compress(values, scores, start, ratio, carry_values=None, carry_scores=None):
    """Return complete unnormalized groups and update caller-owned FP32 carry.

    A ratio-2 odd start consumes ONE prior row. All complete groups in a chunk
    execute together; only the final r positions update carry (no token loop).
    """
    if start < 0 or ratio not in (1, 2):
        raise ValueError('position/ratio')
    if ratio == 1:
        return values, start
    if scores.shape != values.shape or carry_values.shape != (ratio, values.shape[-1]) or carry_scores.shape != carry_values.shape:
        raise ValueError('compressor shape')
    lead = start % ratio
    v = torch.cat((carry_values[:lead], values.float()), dim=0)
    s = torch.cat((carry_scores[:lead], scores.float()), dim=0)
    n = len(v) // ratio * ratio
    groups = v[:n].reshape(-1, ratio, values.shape[-1])
    gates = s[:n].reshape_as(groups).softmax(1)
    out = (groups * gates).sum(1).to(values.dtype)
    # Match the official ring carry exactly, including the stale completed row.
    begin = max(0, len(values) - ratio)
    rows = (torch.arange(begin, len(values), device=values.device) + start) % ratio
    carry_values[rows] = values[begin:].float()
    carry_scores[rows] = scores[begin:].float()
    return out, start // ratio


def compress_groups(values, scores, ratio):
    """Pure compressor over whole groups: no carry, no hidden sequence state.

    values/scores are [G*ratio, D] rows gathered by absolute position, so the
    caller owns ordering and the result depends on nothing else.
    """
    if ratio == 1:
        return values
    assert values.shape == scores.shape and values.size(0) % ratio == 0
    g = values.float().reshape(-1, ratio, values.shape[-1])
    gates = scores.float().reshape_as(g).softmax(1)
    return (g * gates).sum(1).to(values.dtype)


def select(q, head_weight, keys, positions, ratio, topk, *,
           total_heads=None, reduce_scores=None, query_tile=3072, key_tile=6144):
    """Streaming ReLU index scores: O(query_tile*key_tile*heads), not T*history.

    reduce_scores sums TP partial scores in place, BEFORE top-k selection.
    total_heads is the global head count, never the local TP count.
    """
    t, h, d = q.shape
    total_heads = h if total_heads is None else total_heads
    if total_heads != h and reduce_scores is None:
        raise ValueError('TP index scores require sum reduction')
    result = torch.full((t, topk), -1, dtype=torch.long, device=q.device)
    if not len(keys):
        return result
    if (q.is_cuda and d == 128 and 0 < h <= 4
            and 0 < topk and 0 < key_tile and topk + key_tile <= 8192
            and query_tile > 0 and ratio > 0 and positions.is_contiguous()):
        from .index_merge import select_direct
        return select_direct(q, head_weight, keys, positions, ratio, topk,
                             total_heads=total_heads, reduce_scores=reduce_scores,
                             query_tile=query_tile, key_tile=key_tile, result=result)
    for lo in range(0, t, query_tile):
        hi = min(t, lo + query_tile)
        best = torch.full((hi-lo, topk), -torch.inf, device=q.device)
        ids = torch.full_like(best, -1, dtype=torch.long)
        width = len(keys)
        for a in range(0, width, key_tile):
            ix = torch.arange(a, min(a+key_tile, width), device=q.device).expand(hi-lo, -1)
            valid = (ix >= 0) & (ix < ((positions[lo:hi, None]+1)//ratio)) & (ix < len(keys))
            # The main path never scans future rows even though all chunk KV is written.
            k = keys[ix.clamp(0, max(0, len(keys)-1))].float()
            logits = torch.einsum('thd,tkd->thk', q[lo:hi].float(), k).relu_()
            scores = (logits * head_weight[lo:hi].float().unsqueeze(-1)).sum(1) * d**-.5 * total_heads**-.5
            if reduce_scores is not None:
                reduce_scores(scores)
            scores.masked_fill_(~valid, -torch.inf)
            merged = torch.cat((best, scores), dim=-1)
            if a == 0:
                # Ascending real IDs precede initial -1 padding in the ABI.
                ordered_scores = torch.cat((scores, best), dim=-1)
                ordered_ids = torch.cat((ix, ids), dim=-1)
                choice = ordered_scores.argsort(dim=-1, descending=True, stable=True)[..., :topk]
                best = ordered_scores.gather(-1, choice)
                ids = ordered_ids.gather(-1, choice)
            else:
                best, ids = select_topk(merged, torch.cat((ids, ix), dim=-1), topk)
        result[lo:hi] = ids.masked_fill(~torch.isfinite(best), -1)
    return result


def sparse(q, local_kv, local_start, global_kv, selected, positions, sink,
           *, window, ratio, scale, query_tile=16):
    """Latent KV attention with sink denominator; duplicates across SWA/global
    are intentional distinct entries as in training. No historical dense mask.
    """
    t, h, d = q.shape
    out = torch.empty_like(q)
    for lo in range(0, t, query_tile):
        hi = min(t, lo + query_tile)
        p = positions[lo:hi]
        swa = p[:, None] - torch.arange(window-1, -1, -1, device=q.device)
        valid = (swa >= local_start) & (swa >= 0)
        rows = local_kv[(swa-local_start).clamp(0, len(local_kv)-1)]
        if selected is not None:
            ix = selected[lo:hi]
            gv = (ix >= 0) & (ix < (p[:, None]+1)//ratio) & (ix < len(global_kv))
            if len(global_kv):
                grow = global_kv[ix.clamp(0, len(global_kv)-1)]
            else:
                grow = rows.new_zeros(hi-lo, ix.shape[-1], d)
            rows = torch.cat((rows, grow), dim=1)
            valid = torch.cat((valid, gv), dim=-1)
        logits = torch.einsum('thd,tkd->thk', q[lo:hi].float(), rows.float()) * scale
        logits.masked_fill_(~valid[:, None], -torch.inf)
        logits = torch.cat((logits, sink.float().expand(hi-lo, h).unsqueeze(-1)), dim=-1)
        prob = logits.softmax(-1)[..., :-1]
        out[lo:hi] = torch.einsum('thk,tkd->thd', prob, rows.float()).to(q.dtype)
    return out
