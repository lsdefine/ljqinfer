"""DSpark draft attention (decode-only).

Contract differences from ops.decode.attention.sparse (they cannot merge):
  * sparse() is causal inside the verify window: row i sees rows <= i. A DSpark
    draft block is semi-autoregressive -- every draft query shares one visible
    set (the window history plus *all* draft rows), which is exactly what the
    reference implementation expresses by handing the same topk_idxs row to
    each of the block_size queries.
  * the ring only ever receives the main model's KV row; draft rows are
    provisional and live in this call alone.
"""
import torch

from ops.decode.argmax import sample_rows


def draft_attend(q, window_kv, draft_kv, sink, *, scale, valid=None):
    """q [B,Q,H,D] draft queries, window_kv [B,W,D] committed ring history
    in logical order, draft_kv [B,Q,D] this block's provisional rows, sink [H].
    A single request may drop the batch axis on all three.

    Row order inside the visible set is irrelevant (no positional mask is
    applied here; RoPE is already baked into q and kv), so reading the ring in
    logical order matches the reference gather over physical ring rows.
    """
    flat = q.dim() == 3
    if flat:
        q, window_kv, draft_kv = q[None], window_kv[None], draft_kv[None]
        valid = None if valid is None else valid[None]
    b, t, h, d = q.shape
    if tuple(draft_kv.shape[:2]) != (b, t):
        raise ValueError("one provisional KV row per draft query")
    if window_kv.shape[-1] != d or draft_kv.shape[-1] != d:
        raise ValueError("latent KV width must match the query head width")
    rows = torch.cat((window_kv, draft_kv), dim=1).float()
    logits = torch.einsum("bthd,bkd->bthk", q.float(), rows) * scale
    if valid is not None:
        # A captured caller gathers the whole ring; rows before the history
        # begins are folded onto the oldest row and dropped here.
        keep = torch.cat((valid, valid.new_ones(b, t)), dim=1)
        logits = logits.masked_fill(~keep[:, None, None, :], float("-inf"))
    logits = torch.cat((logits, sink.float().expand(b, t, h).unsqueeze(-1)), dim=-1)
    prob = logits.softmax(-1)[..., :-1]
    out = torch.einsum("bthk,bkd->bthd", prob, rows).to(q.dtype)
    return out[0] if flat else out


def sample(logits, temperature, shard=None):
    """Greedy at temperature 0; otherwise Gumbel-max, both as one split scan.

    The softmax/exponential/div/argmax chain this replaces cost four launches
    and ~104us per draft position on a [1, V] block of torch reductions.
    With `shard` the logits are this rank's vocabulary slice and the winner is
    folded across ranks on two floats per row.
    """
    return sample_rows(logits, temperature, shard)


def markov_refine(logits, token, markov, *, temperature=0.0, shard=None):
    """Turn one block of position-wise draft logits into a token chain.

    The three DSpark stages emit block_size sets of logits in a single pass, so
    nothing in them is conditioned on what the previous draft position ended up
    sampling. The Markov head supplies exactly that missing dependency: step i
    biases logits[i] with a low-rank term read from the token chosen at i-1.

    logits [B,S,V] is modified in place (it is a per-step scratch buffer);
    token [B] is the accepted main-model token that opens the chain;
    markov(ids) -> (bias [B,V], embed [B,R]).
    Under TP, V is the rank's vocabulary slice and `shard` describes it: the
    bias add stays elementwise on the slice and only the sampler reduces, so
    no step of the chain moves a full vocabulary between ranks.
    Returns (ids [B,S+1], logits [B,S,V], embeds [B,S,R]).
    """
    steps = logits.shape[1]
    ids = token.new_empty(token.shape[0], steps + 1)
    ids[:, 0] = token
    embeds = []
    for i in range(steps):
        bias, embed = markov(ids[:, i])
        logits[:, i].add_(bias)
        embeds.append(embed)
        ids[:, i + 1] = sample(logits[:, i], temperature, shard)
    return ids, logits, torch.stack(embeds, dim=1)


def confidence(x, embeds, proj):
    """Per-draft-position acceptance score, computed in fp32 because the
    scheduler compares it against a fixed threshold."""
    return torch.nn.functional.linear(
        torch.cat((x, embeds), dim=-1).float(), proj.float()).squeeze(-1)
