"""Prefill selection ABI: descending score, ascending logical ID on exact ties.

No epsilon perturbation: distinct scores keep their mathematical order. Fused
selection kernels must preserve this rule to avoid chunk-length-dependent KV.
"""
import torch


def topk(scores, ids, count):
    order = ids.masked_fill(ids < 0, torch.iinfo(ids.dtype).max).argsort(dim=-1, stable=True)
    sorted_scores = scores.gather(-1, order)
    choice = sorted_scores.argsort(dim=-1, descending=True, stable=True)[..., :count]
    order = order.gather(-1, choice)
    return scores.gather(-1, order), ids.gather(-1, order)
