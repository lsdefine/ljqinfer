"""Prefill candidate prefilter, tiled across queries and source blocks."""
import torch
from .selection import topk as select_topk


def build(q, head_weight, keys, positions, ratio, *, block_size, top_blocks,
          total_heads=None, reduce_scores=None, query_tile=16, block_tile=64):
    t, h, d = q.shape
    total_heads = h if total_heads is None else total_heads
    if total_heads != h and reduce_scores is None:
        raise ValueError('TP index scores require reduction')
    # The forced newest block consumes one of top_blocks slots.
    if top_blocks < 1 or block_size < 1:
        raise ValueError('positive candidate geometry required')
    out = torch.full((t, top_blocks*block_size), -1, dtype=torch.long, device=q.device)
    if not len(keys):
        return out
    nblocks = (len(keys)+block_size-1)//block_size
    within = torch.arange(block_size, device=q.device)
    for lo in range(0, t, query_tile):
        hi = min(t, lo+query_tile)
        lens = (positions[lo:hi]+1)//ratio
        newest = (lens-1)//block_size
        best = torch.full((hi-lo, top_blocks-1), -torch.inf, device=q.device)
        ids = torch.full_like(best, -1, dtype=torch.long)
        for a in range(0, nblocks, block_tile):
            blocks = torch.arange(a, min(a+block_tile, nblocks), device=q.device)
            rows = (blocks[:, None]*block_size+within).flatten()
            k = keys[rows.clamp(max=len(keys)-1)].float()
            logits = torch.einsum('thd,kd->thk', q[lo:hi].float(), k).relu_()
            score = (logits*head_weight[lo:hi].float().unsqueeze(-1)).sum(1) * d**-.5 * total_heads**-.5
            if reduce_scores is not None:
                reduce_scores(score)
            score.masked_fill_(rows[None] >= lens[:, None], -torch.inf)
            bs = score.unflatten(-1, (-1, block_size)).amax(-1)
            bs.masked_fill_(blocks[None] >= newest[:, None], -torch.inf)
            merged = torch.cat((best, bs), dim=-1)
            best, ids = select_topk(merged, torch.cat((ids, blocks.expand(hi-lo, -1)), dim=-1), top_blocks-1)
        ids = ids.masked_fill(~torch.isfinite(best), -1)
        ids = torch.cat((ids, newest[:, None]), dim=-1)
        rows = (ids[..., None]*block_size+within).flatten(-2)
        out[lo:hi] = rows.masked_fill((rows < 0) | (rows >= lens[:, None]), -1)
    return out
