"""Live-shape Tensor compositions over registered optimized NPU operators."""
import torch
from .native import ops as native
from model.prefill_trace import span


def rope(x, freqs, *, inverse=False, library=None):
    return native.rope(x, freqs, inverse)


def qdq(x, kind, *, library=None):
    return native.qdq(x, {'local': 0, 'compressed': 1, 'index': 2}[kind])


def compress(values, scores, cv, cs, start, *, library=None):
    return native.compress(values, scores, cv, cs, start)


def frequencies(table, positions, *, library=None):
    return native.frequencies(table, positions)


@span("index")
def select(q, weight, keys, positions, valid, *, ratio, total_heads,
           parallel, workspace, topk=512, candidates=None, make_candidates=False, library=None):
    """Fixed Cube/TP tiles; grouped native NPU TopK."""
    rows, heads, dim = q.shape
    width = max(32, (len(keys)+31)//32*32)
    bank = keys
    if len(keys) != width:
        bank = workspace.bank[:width]
        bank.zero_()
        bank[:len(keys)].copy_(keys)
    tile, group = workspace.tile, workspace.group
    out = torch.empty((rows, topk), dtype=torch.int64, device=q.device)
    if make_candidates:
        candidates = out.new_empty((rows, 16384))
    for begin in range(0, rows, group):
        end = min(begin+group, rows)
        pos = positions[begin:end]
        dot, score, values, order = workspace.views(end-begin, heads, width, min(topk, width))
        # Four fixed compute tiles share one selection. Each tile
        # writes disjoint score rows; dot storage is reused on the same stream.
        for lo in range(begin, end, tile):
            hi = min(lo+tile, end)
            block = dot[:(hi-lo)*heads]
            native.matmul_fp32.out(q[lo:hi].reshape(-1, dim), bank.t(), block)
            native.score_reduce(block, weight[lo:hi], positions[lo:hi], valid,
                                len(keys), ratio, dim**-.5 * total_heads**-.5,
                                score[lo-begin:hi-begin])
            parallel.sum(score[lo-begin:hi-begin])
        if make_candidates:
            block_values, block_order = torch.sort(native.candidate_blocks(score, pos, ratio),
                                       descending=True, stable=True)
            candidates[begin:end] = native.candidate_expand(block_values, block_order, pos, valid, len(keys), ratio)
        if candidates is not None:
            score = native.candidate_mask(score, candidates[begin:end], len(keys))
        torch.topk(score, min(topk, score.shape[-1]), sorted=True, out=(values, order))
        native.sorted_ids(values, order, topk, out[begin:end])
    return (candidates, out) if make_candidates else out


def attend(q, local, bank, selected, positions, meta, sink, *, ratio, library=None):
    """Released fused attention with explicit sink/invalid-key correction."""
    if not ratio:
        out, maximum, total = native.flash_swa(q, local, sink)
        return native.swa_correct(out, maximum, total, positions, meta)
    packed, indices, missing = native.joint_pack(local, bank, selected, positions, meta, ratio)
    out, maximum, total = native.flash_sparse(q, packed, indices)
    return native.joint_correct(out, maximum, total, missing, sink, indices.shape[-1])


def build_paged_read(data, table, out, *, library=None):
    """Compatibility for Past.gather_into; dispatch is unbound and one-shot."""
    def run(slot, count):
        return native.paged_read(data, table, slot, count, out)
    return run


def compress_groups(values, scores, ratio):
    """Past.fold compatibility for complete, non-overlapping groups."""
    if ratio < 1 or len(values) % ratio or values.shape != scores.shape:
        raise ValueError('complete equal-shaped compression groups required')
    v = values.float().reshape(-1, ratio, values.shape[-1])
    s = scores.float().reshape_as(v)
    return (v * s.softmax(dim=1)).sum(dim=1)
