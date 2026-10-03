"""Direct index selection with bounded reusable score/merge buffers."""
import os
from functools import lru_cache
from pathlib import Path
import torch

@lru_cache(maxsize=1)
def extension():
    from torch.utils.cpp_extension import load
    os.environ.setdefault('TORCH_CUDA_ARCH_LIST', '8.0')
    os.environ.setdefault('MAX_JOBS', '2')
    return load(name='glm53_index_merge_port',
                sources=[str(Path(__file__).with_suffix('.cu'))],
                extra_cuda_cflags=['-O3', '--fmad=false'], verbose=False)


def select_direct(q, head_weight, keys, positions, ratio, topk, *,
                  total_heads, reduce_scores, query_tile, key_tile, result, weights_prescaled=False):
    """Top-k index selection; rows are sharded across ranks when possible.

    A rank-sharded reduction halves the score traffic of an all-reduce and
    leaves every rank merging only its own rows; the ids are gathered back at
    the end, which is three orders of magnitude smaller than the scores.
    """
    merge_module = extension()
    q, keys, weights = q.float().contiguous(), keys.float().contiguous(), head_weight.float().contiguous()
    rows = min(len(q), query_tile)
    width = len(keys)
    scale = float(q.size(-1) ** -0.5) * (1.0 if weights_prescaled else float(total_heads ** -0.5))
    world = getattr(reduce_scores, 'world', 1) if reduce_scores is not None else 1
    rank = getattr(reduce_scores, 'rank', 0) if reduce_scores is not None else 0
    sharded = (world > 1 and hasattr(reduce_scores, 'scatter_sum')
               and rows % world == 0 and len(q) % query_tile == 0)
    held = rows // world if sharded else rows
    tile = min(width, key_tile)
    score_storage = torch.empty(rows * tile, device=q.device, dtype=torch.float32)
    part_storage = torch.empty(rows * tile, device=q.device, dtype=torch.float32)
    local_storage = (torch.empty(held * tile, device=q.device, dtype=torch.float32)
                     if sharded else None)
    score_pairs = [torch.empty((held, topk), device=q.device, dtype=torch.float32) for _ in range(2)]
    id_pairs = [torch.empty((held, topk), device=q.device, dtype=torch.long) for _ in range(2)]
    for lo in range(0, len(q), query_tile):
        hi = min(len(q), lo + query_tile)
        n = hi - lo
        held_n = n // world if sharded else n
        first = lo + rank * held_n if sharded else lo
        current = 0
        best, ids = score_pairs[0][:held_n], id_pairs[0][:held_n]
        best.fill_(-torch.inf)
        ids.fill_(-1)
        for begin in range(0, width, key_tile):
            end = min(width, begin + key_tile)
            scores = score_storage[:n * (end-begin)].view(n, end-begin)
            part = part_storage[:n * (end-begin)].view(n, end-begin)
            keys_t, tile_q, tile_w = keys[begin:end].t(), q[lo:hi], weights[lo:hi]
            scores.zero_()
            for head in range(tile_q.size(1)):
                torch.mm(tile_q[:, head], keys_t, out=part)
                scores.addcmul_(part.relu_(), tile_w[:, head:head+1])
            scores.mul_(scale)
            if sharded:
                held_scores = local_storage[:held_n * (end-begin)].view(held_n, end-begin)
                reduce_scores.scatter_sum(scores, held_scores)
                scores = held_scores
            elif reduce_scores is not None:
                reduce_scores(scores)
            other = 1-current
            out_scores, out_ids = score_pairs[other][:held_n], id_pairs[other][:held_n]
            merge_module.merge(best, ids, scores, positions,
                               positions[first:first+held_n],
                               begin, width, ratio, True, end == width, out_scores, out_ids)
            best, ids, current = out_scores, out_ids, other
        if sharded:
            reduce_scores.gather_rows(ids.contiguous(), result[lo:hi])
        else:
            result[lo:hi].copy_(ids)
    return result
