"""Decode attention state adapter: read-only over a speculative window.

Contract differences from model.prefill_attention.attend (they cannot merge):
  * prefill writes the ring and the paged source inside attend, because every
    prefill row is final. A verify window of Q rows may be rejected, so decode
    reads committed history plus caller-held *staged* rows and writes nothing.
    Committing is Past.commit_verify_prefix + WindowPast.write, done once the
    acceptance count is known.
  * prefill asserts start == past.pos[slot] for a growing chunk; decode asserts
    the same start but treats the Q rows as provisional.
  * DSpark layers are rejected by prefill; decode serves them (uncompressed
    sliding window, no source layer).
"""
from dataclasses import dataclass, field
from functools import lru_cache
import torch
from .past import WINDOW_TOKENS
from ops.decode import attention as ops
from ops.decode.publish import publish


@lru_cache(maxsize=None)
def _no_pages(device, rows=1):
    """Placeholder page table for layers with no compressed pool: every id
    lands in the tail, so the kernel never dereferences it.  It carries one
    row per *slot* (not per request) because the kernel addresses it through
    the slot -> row map, exactly like a real page table.  Cached so the
    captured graph carries no per-step allocation."""
    return torch.zeros(rows, 1, dtype=torch.int64, device=device)


@dataclass
class DecodeScratch:
    """Per-step scratch, scoped to one verify window, never retained."""
    selections: dict = field(default_factory=dict)
    # Rotation rows already selected this step, keyed by (table, positions).
    rope_rows: dict = field(default_factory=dict)
    # Projected source rows held by verify, keyed by source layer.  They are
    # graph-stable tensors; commit publishes only the accepted prefix to the
    # canonical residual ring.  Tuple: (values, scores, host_start, dtype).
    pending: dict = field(default_factory=dict)
    # Ring rows this window produced, per layer, awaiting commit.
    rows: dict = field(default_factory=dict)
    # Source rows completed inside this window, keyed by *source* layer: paged
    # sources are shared by every layer reading them, so a reuse layer must see
    # the rows its source layer staged.
    staged: dict = field(default_factory=dict)
    # Block prefilter rows built by the candidate source layer and reused by
    # every later indexing layer, exactly as prefill does.
    candidate_rows: object = None
    # (sorted ids, invalid mask) derived from candidate_rows once per window.
    candidate_prep: object = None
    # Device positions [Q] for this window (graph replay updates it in place);
    # None means derive from the host ``start``.
    pos_t: object = None

    def rebase(self, start: int):
        """Point the host bookkeeping at the window just scored.

        A replay reruns the captured body over live buffers, but the host
        ``start`` recorded in ``pending`` is the one the capture saw.  A graph
        runner calls this after each replay so commit's window check reads the
        window that actually ran.
        """
        for layer, (values, scores, _, dtype) in self.pending.items():
            self.pending[layer] = (values, scores, start, dtype)

def _requests(slot):
    """One call may carry several requests; a lone int is a batch of one."""
    return (slot,) if isinstance(slot, int) else tuple(slot)


@lru_cache(maxsize=None)
def _row_map(nreq, device):
    """The one row map every batch of ``nreq`` requests is read through.

    One buffer per batch width, at one address for the process lifetime, with
    a pinned mirror to fill it from without a sync.  The list carries the slots
    the device copy currently holds, so a repeat call costs a comparison.
    """
    return (torch.empty(nreq, dtype=torch.long, device=device),
            torch.empty(nreq, dtype=torch.long).pin_memory(), [])


def slot_rows(slots, device):
    """Which pool row each request of the batch owns.

    Slots are handed out as requests arrive, so a batch never owns adjacent
    rows and the pools must be read in place through this map.  The map is a
    buffer per batch width refilled with the slots of the call: a capture
    bakes in the *address*, never the slots, so one graph per batch width
    serves whatever batch the scheduler hands it.  A graph runner refills it
    before the replay, exactly as it stages Engram rows.
    """
    want = [int(x) for x in slots]
    rows, host, holds = _row_map(len(want), device)
    if holds == want or torch.cuda.is_current_stream_capturing():
        # Already pointing at this batch, or mid-capture where the graph only
        # needs the address: filling here would bake a copy into the graph,
        # and the runner refills it before each replay anyway.
        return rows
    for i, one in enumerate(want):
        host[i] = one
    # Synchronous: host is one reused staging row, and the next batch
    # overwrites it long before an async copy would have read it.
    rows.copy_(host)
    holds[:] = want
    return rows


def window_positions(past, slot, rows_per_request, device):
    """Absolute position of every row of the batch, read off the live cursor.

    Derived on device so a captured graph replayed at a later step ropes and
    attends at the position it actually holds, not the captured one.
    """
    rows = slot_rows(_requests(slot), device)
    return (past.pos_dev.index_select(0, rows).unsqueeze(1)
            + torch.arange(rows_per_request, device=device)).reshape(-1)


def decode_attend(*, past, slot, start, layer, q, kv, sink, scale, scratch,
                  staged_ckv=None, staged_index_k=None,
                  index_q=None, index_weight=None, topk=64, candidates=None,
                  total_index_heads=None, reduce_scores=None):
    """q[B*Q,H,D], kv[B*Q,D] are provisional rows at start..start+Q-1.

    ``slot`` and ``start`` may be sequences: the call then carries one window
    per request, stacked back to back, and every pool is read in place through
    a slot -> row map.  A lone int is that same path with a batch of one, so
    there is no second single-request implementation.

    staged_ckv/staged_index_k are source rows completed *within* this window
    and not yet in the pools; they are appended after the committed history so
    that logical row ids stay contiguous.  A batch carries them stacked the
    same way, an equal count per request.
    """
    view = past.views[layer]
    slots, starts = _requests(slot), _requests(start)
    nreq = len(slots)
    qwin = len(q) // nreq
    if len(starts) != nreq:
        raise ValueError('decode needs one start per slot')
    if qwin == 0 or len(q) != nreq * qwin or len(kv) != len(q):
        raise ValueError('decode window needs one KV row per query row')
    for one, begin in zip(slots, starts):
        if begin != past.pos[one]:
            raise ValueError('decode window must start at the committed position')
        if one in past.free_slots or one in past.replay_pending:
            raise ValueError('decode requires a live, replayed slot')
    window = past.windows[layer]
    if qwin > window.ring - window.window:
        # in-graph ring writes may only clobber rows older than the SWA span
        raise ValueError('verify window may not exceed the sliding window')
    rows = slot_rows(slots, q.device)
    if scratch.pos_t is None:
        # Every layer asks for the same positions; building it once per
        # step removes 39 launches from the captured graph.  It is derived
        # from the *device* cursor, not the host int, so a captured graph
        # replayed at a later step reads the advanced position instead of
        # the one frozen at capture time.
        scratch.pos_t = window_positions(past, slots, qwin, q.device)
    pos = scratch.pos_t
    selected = None
    if view.ratio:
        source = past.sources[view.kv_source_layer]
        if staged_ckv is not None or staged_index_k is not None:
            scratch.staged[view.kv_source_layer] = (staged_ckv, staged_index_k)
        held = scratch.staged.get(view.kv_source_layer, (None, None))
        if view.mode in ('full', 'reindex'):
            if index_q is None or index_weight is None:
                raise ValueError('missing index query projection')
            from ops.decode.live_index import paged_scores
            score = paged_scores(index_q, index_weight, source.index_pool,
                                 rows, pos, view.ratio,
                                 total_index_heads, reduce_scores)
            selected = ops.select(index_q, index_weight, None, pos, view.ratio,
                                  topk, total_heads=total_index_heads,
                                  candidates=candidates, precomputed_scores=score)
            scratch.selections[layer] = selected
        else:
            # Reuse layers must not recompute index scores; that is the whole
            # point of the 32/40 reuse budget.
            selected = scratch.selections[view.index_source_layer]
        staged_kv = held[0] if held[0] is not None and len(held[0]) else None
        ns = 0 if staged_kv is None else len(staged_kv) // nreq
        # Publish the window's rows into the ring now, inside the graph.  Rows
        # the commit rejects stay above pos and are never addressed (attn_ids
        # reads committed rows only below start); the next window overwrites
        # them.  This is the whole WindowPast commit: nothing is written later.
        publish(window, rows, pos, kv, staged_kv, nreq, qwin, ns)
        return ops.sparse_paged(
            q, source.ckv_pool.data, source.ckv_pool.pt.table, window.main_kv,
            ns, selected, pos, sink, window=WINDOW_TOKENS, ring=window.ring,
            ratio=view.ratio, scale=scale, pad=window.pad, qwin=qwin,
            rowmap=rows)
    # No compressed pool on this layer: every id the window needs lands in the
    # tail band, so the same fused kernel the paged branch uses reads the ring
    # in place.  The pool argument is never dereferenced (ids >= total).
    publish(window, rows, pos, kv, None, nreq, qwin, 0)
    return ops.sparse_paged(
        q, window.main_kv, _no_pages(q.device, window.main_kv.size(0)), window.main_kv, 0, None, pos,
        sink, window=WINDOW_TOKENS, ring=window.ring, ratio=1, scale=scale,
        pad=window.pad, qwin=qwin, rowmap=rows)
