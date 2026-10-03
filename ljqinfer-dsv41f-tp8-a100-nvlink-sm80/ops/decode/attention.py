"""Decode-only attention primitives: fixed small Q, gather-based, stateless.

Why these cannot be shared with prefill:
  * prefill streams query/key tiles (O(T*K) loop) because T reaches 12k; decode
    has Q <= 8 rows, so tiling is pure overhead and every score fits one pass.
  * prefill commits rows as it goes; a verify window may be rejected, so nothing
    here touches Past. Reads take already-committed history plus *staged* rows
    the caller holds outside the pools.
  * decode always scores the whole live index history in one shot, keeping the
    AllReduce shape constant, which is what makes graph capture possible.

Math is a restatement of ops.prefill.attention.select/sparse: same ReLU index
score, same sink denominator, same validity rules.
"""
import torch
from .attn_ids import attn_ids

from ops.decode import sparse_attn, v4k
from ops.decode.cand_select import cand_select


def prepare_candidates(candidates, positions, ratio, prefix_mask=None):
    """Layer-independent part of the candidate prefilter, computed once.

    The candidate ids and the live-prefix bound do not depend on the layer,
    so the sort plus live/duplicate masks are hoisted out of every reuse
    layer and cached by the candidate source layer.
    """
    cs, _ = torch.sort(candidates, dim=-1)
    live = (cs >= 0) & (cs < ((positions[:, None] + 1) // ratio))
    if prefix_mask is not None:
        live &= ~prefix_mask.gather(1, cs.clamp(min=0))
    dup = torch.zeros_like(live)
    dup[:, 1:] = cs[:, 1:] == cs[:, :-1]
    return cs.clamp(min=0), (~live) | dup


def select(q, head_weight, keys, positions, ratio, topk, *, candidates=None,
           total_heads=None, reduce_scores=None, prefix_mask=None, precomputed_scores=None):
    """ReLU index scores for Q rows against the whole index history.

    q [Q,H,D], head_weight [Q,H], keys [N,D], positions [Q].
    candidates [Q,C] is the optional block prefilter: unique logical row ids
    (-1 padded) built once by the candidate source layer and reused after it.
    Returns logical row ids [Q,topk] (-1 padded); same ABI as prefill.select.
    """
    t, h, d = q.shape
    total_heads = h if total_heads is None else total_heads
    if total_heads != h and reduce_scores is None and precomputed_scores is None:
        raise ValueError('TP index scores require sum reduction')
    result = torch.full((t, topk), -1, dtype=torch.long, device=q.device)
    n = len(keys) if precomputed_scores is None else precomputed_scores.shape[1]
    if not n:
        return result
    if precomputed_scores is None:
        logits = torch.einsum('thd,kd->thk', q.float(), keys.float()).relu_()
        scores = (logits * head_weight.float().unsqueeze(-1)).sum(1) * d**-.5 * total_heads**-.5
        if reduce_scores is not None:
            # Partial sums must be reduced BEFORE top-k or ranks disagree.
            reduce_scores(scores)
    else:
        scores = precomputed_scores
    if candidates is not None:
        # The prefilter restricts, it never rescores: identical scores over a
        # smaller legal set, so ranks inside the set match prefill exactly.
        # A tuple is the pre-sorted/masked form cached by the source layer.
        if isinstance(candidates, tuple):
            csc, bad = candidates
        else:
            csc, bad = prepare_candidates(candidates, positions, ratio, prefix_mask)
        # Selection, not ordering: the consumer treats the row as a set, so one
        # fused radix-select kernel replaces gather/mask/top-k/gather/isfinite.
        return cand_select(scores.contiguous(), csc, bad, topk)
    if prefix_mask is None:
        # The radix leaf re-derives the live-prefix bound from positions/ratio,
        # so materialising an [Q,N] mask over the whole pool capacity and then
        # filling -inf through it is pure waste; hand the raw scores straight on.
        return v4k.topk(scores.contiguous(), topk, ratio, positions)
    scores = scores.masked_fill(prefix_mask, -torch.inf)
    # No prefilter means the only -inf is the live-prefix bound, which the
    # radix leaf re-derives; a full [Q,N] sort to keep topk rows is waste.
    return v4k.topk(scores, topk, ratio, positions)


def sparse(q, local_kv, local_start, global_kv, selected, positions, sink,
           *, window, ratio, scale):
    """Latent KV attention over a sliding window plus selected global rows.

    q [Q,H,D]; local_kv [L,D] is ring history concatenated with this verify
    window's own rows, addressed from ``local_start``.
    """
    t, h, d = q.shape
    p = positions
    swa = p[:, None] - torch.arange(window - 1, -1, -1, device=q.device)
    valid = (swa >= local_start) & (swa >= 0)
    rows = local_kv[(swa - local_start).clamp(0, len(local_kv) - 1)]
    if selected is not None:
        ix = selected
        gv = (ix >= 0) & (ix < (p[:, None] + 1) // ratio) & (ix < len(global_kv))
        if len(global_kv):
            grow = global_kv[ix.clamp(0, max(0, len(global_kv) - 1))]
        else:
            grow = rows.new_zeros(t, ix.shape[-1], d)
        rows = torch.cat((rows, grow), dim=1)
        valid = torch.cat((valid, gv), dim=-1)
    logits = torch.einsum('thd,tkd->thk', q.float(), rows.float()) * scale
    logits.masked_fill_(~valid[:, None], -torch.inf)
    logits = torch.cat((logits, sink.float().expand(t, h).unsqueeze(-1)), dim=-1)
    prob = logits.softmax(-1)[..., :-1]
    return torch.einsum('thk,tkd->thd', prob, rows.float()).to(q.dtype)


_SINK_F32 = {}


def _sink_f32(sink, heads):
    """sink is a weight: widen+expand once per tensor, not per layer per step."""
    key = (sink.data_ptr(), heads)
    got = _SINK_F32.get(key)
    if got is None:
        got = sink.float().expand(heads).contiguous()
        _SINK_F32[key] = got
    return got

def sparse_paged(q, pool, page_table, tail, ns, selected,
                 positions, sink, *, window, ring, ratio, scale, pad, qwin=None,
                 rowmap=None):
    """sparse() without materialising the committed history.

    sparse() gathers the whole compressed history into a fresh tensor once per
    layer (40 O(N) copies per step) and scores it in fp32.  Here only row ids
    move: ids below ``positions[0] // ratio`` are read straight from the paged pool, the rest
    from ``tail``, which holds the rows this window staged followed by the
    sliding window.  Same SWA validity rule and sink denominator as sparse().

    ``qwin`` is the rows per request: with it, q holds len(q)//qwin requests
    back to back and page_table/tail carry a leading request dimension, so one
    launch serves a decode batch.  Left unset there is a single request.
    """
    ids = attn_ids(positions, selected, ns=ns, pad=pad, window=window, ring=ring,
                   ratio=ratio, qwin=qwin)
    return sparse_attn.attend(
        q, pool, page_table, tail, _sink_f32(sink, q.shape[1]),
        ids, torch.empty_like(q), positions, ratio, scale, qwin=qwin,
        rowmap=rowmap)
