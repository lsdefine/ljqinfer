"""Prefill attention state adapter. No allocation/commit/cold or decode path.

Q, local KV and index projections are supplied by the layer computation.
Only complete global rows are appended; source/index reuse is explicit through
chunk-local selections. Past is the sole owner of persistent state.
"""
from dataclasses import dataclass, field
import torch
from ops.prefill import attention as ops


@dataclass
class PrefillScratch:
    # Logical selected row IDs, scoped to one slot/chunk (never retained).
    selections: dict = field(default_factory=dict)
    read_global_only: bool = False
    replay_start: int | None = None


def attend(*, past, slot, start, layer, q, kv, sink, scale, scratch,
           compressed=None, index_keys=None, index_q=None, index_weight=None,
           topk=64, total_index_heads=None, reduce_scores=None,
           swa_workspace=None, sparse_workspace=None):
    """q[T,H,D], kv[T,D], compressed/index_keys are NEW complete source rows.

    All projection/normalization/quantization is done by the prefill layer.
    Compressed output uses row origin floor(start/ratio), including odd carry.
    """
    view = past.views[layer]
    t = len(q)
    replay = scratch.read_global_only
    positioned = (0 <= start < start+t <= past.pos[slot]) if replay else start == past.pos[slot]
    if t == 0 or len(kv) != t or not positioned:
        raise ValueError('prefill chunk position/length mismatch')
    if slot in past.free_slots or (slot in past.replay_pending and not replay):
        raise ValueError('prefill requires a live, hot slot')
    if view.mode == 'dspark':
        raise ValueError('DSpark has a separate execution path')
    window = past.windows[layer]
    local_start = max(0, start-window.window+1)
    if replay:
        if scratch.replay_start != start or t > window.window:
            raise ValueError('replay requires one bounded, SWA-truncated segment')
        local_start = start
    if swa_workspace is not None:
        if view.ratio or window.window != swa_workspace.w:
            raise ValueError('SWA workspace requires uncompressed sliding-window attention')
        history = window.read(slot, local_start, start)
        result = swa_workspace(q, history, kv, start=start,
            local_start=local_start, sink=sink, scale=scale)
        window.write(slot, start, kv)
        return result
    pos = torch.arange(start, start+t, device=q.device)
    # Must copy/gather old ring BEFORE writing a chunk longer than the ring.
    history = window.read(slot, local_start, start)
    local = None if sparse_workspace is not None else torch.cat((history, kv), dim=0)
    selected, global_kv = None, None
    if view.ratio:
        source = past.sources[view.kv_source_layer]
        if view.mode == 'full' and not replay:
            count = (start+t)//view.ratio-start//view.ratio
            if count:
                if compressed is None or index_keys is None or len(compressed) != count or len(index_keys) != count:
                    raise ValueError('source must provide exactly the newly complete rows')
                source.write_ckv(slot, start//view.ratio, compressed)
                source.write_index_k(slot, start//view.ratio, index_keys)
            elif len(compressed or ()) or len(index_keys or ()):
                # A chunk that completes no group (ratio 2 starting on an even
                # position) is folded into the carry: the source must stay silent.
                raise ValueError('source must provide exactly the newly complete rows')
        elif compressed is not None or index_keys is not None:
            raise ValueError('non-source layer must not write global rows')
        global_kv = source.ckv(slot, start+t)
        if view.mode in ('full', 'reindex'):
            if index_q is None or index_weight is None:
                raise ValueError('missing index query projection')
            selected = ops.select(index_q, index_weight, source.index_k(slot, start+t),
                                  pos, view.ratio, topk,
                                  total_heads=total_index_heads, reduce_scores=reduce_scores)
            scratch.selections[layer] = selected
        else:
            selected = scratch.selections[view.index_source_layer]
    if sparse_workspace is not None:
        if not view.ratio or window.window != sparse_workspace.w:
            raise ValueError('sparse workspace requires compressed attention')
        result = sparse_workspace(q, history, kv, global_kv, selected, pos,
            start=start, local_start=local_start, sink=sink, scale=scale, ratio=view.ratio)
    else:
        result = ops.sparse(q, local, local_start, global_kv, selected, pos, sink,
                            window=window.window, ratio=view.ratio, scale=scale)
    window.write(slot, start, kv)
    return result
