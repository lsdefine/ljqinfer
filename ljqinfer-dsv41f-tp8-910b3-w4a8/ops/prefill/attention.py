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
    """CANN fused scoring/TopK for a complete ratio-two eager prefix.

    Split query parity so right-aligned causality adds one compressed key per
    row. CED/decode use their own paged adapters; no legacy score fallback.
    """
    if ratio != 2 or total_heads != 32 or topk != 512 or candidates is not None or make_candidates:
        raise ValueError('eager index expects ratio2, 32 heads, Top512, no CED candidates')
    rows, heads, dim = q.shape
    limits = torch.minimum((positions + 1) // ratio, valid)
    if len(keys) <= topk:
        ids = torch.arange(topk, device=q.device).expand(rows, -1)
        return torch.where(ids < limits[:, None], ids, -1)
    start = int(positions[0].item())
    gq = q.new_empty((total_heads // heads, *q.shape))
    gw = weight.new_empty((total_heads // heads, *weight.shape))
    parallel.logits(q.contiguous(), out=gq)
    parallel.logits(weight.contiguous(), out=gw)
    query = gq.permute(1, 0, 2, 3).reshape(rows, total_heads, dim)
    weights = gw.permute(1, 0, 2).reshape(rows, total_heads)
    pages = (len(keys) + 1023) // 1024
    bank = torch.nn.functional.pad(keys, (0, 0, 0, pages * 1024 - len(keys)))
    bank = bank.view(pages, 1024, 1, dim)
    table = torch.arange(pages, device=q.device, dtype=torch.int32)[None]
    out = torch.full((rows, topk), -1, device=q.device, dtype=torch.int64)
    for offset in range(ratio):
        part = query[offset::ratio].contiguous()
        n = len(part)
        if not n:
            continue
        length = min(len(keys), (start + offset + ratio * (n - 1) + 1) // ratio)
        if not length:
            continue
        sq = torch.tensor([n], device=q.device, dtype=torch.int32)
        sk = torch.tensor([length], device=q.device, dtype=torch.int32)
        ids = native.paged_index(part[None], bank, weights[offset::ratio].contiguous()[None],
                                 sq, sk, table, topk).view(n, topk).long()
        out[offset::ratio] = torch.where((ids >= 0) & (ids < limits[offset::ratio, None]), ids, -1)
    return out


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
