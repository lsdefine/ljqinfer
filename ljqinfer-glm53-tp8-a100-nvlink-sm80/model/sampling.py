"""Exact nucleus sampling: batched sort/CDF, one uniform per target position.

Rank 0 owns PyTorch RNG; TP peers receive only the selected token IDs.
Greedy bypasses sorting and RNG entirely. No model logits are modified.
"""
import torch
import torch.distributed as dist


def nucleus_cdf(scores):
    """Ascending FP32 CDF and original vocabulary IDs, for independent validation."""
    values, ids = scores.float().sort(dim=-1)
    return values.softmax(-1).cumsum(-1), ids


def _inverse_cdf_cpu(cdf, ids, uniforms, top_p):
    if not torch.isfinite(cdf).all():
        raise ValueError('sampling requires finite logits or -inf masks and a finite maximum')
    rows, vocab = cdf.shape
    cut = torch.full((rows, 1), 1 - top_p, dtype=cdf.dtype, device=cdf.device)
    first = torch.searchsorted(cdf, cut, right=True).clamp_max(vocab - 1)
    tail = cdf.gather(1, (first - 1).clamp_min(0))
    tail = torch.where(first > 0, tail, 0)
    total = cdf[:, -1:]
    target = tail + (total - tail) * uniforms.reshape(-1, 1)
    # Rounding near one must not select a trailing zero-probability interval.
    target = torch.minimum(target, torch.nextafter(total, tail))
    chosen = torch.searchsorted(cdf, target.contiguous(), right=True).clamp_max(vocab - 1)
    return ids.gather(1, chosen).flatten()


def inverse_cdf(cdf, ids, uniforms, top_p):
    """Uniforms must be in [0,1); production supplies torch.rand, not custom RNG."""
    if cdf.is_cuda:
        from .sampling_cuda import inverse_cdf_cuda
        return inverse_cdf_cuda(cdf, ids, uniforms, top_p)
    return _inverse_cdf_cpu(cdf, ids, uniforms, top_p)


def pick_tokens(logits, temperatures, *, top_p=0.95):
    """logits [B,...,V]; T in {0,1}, nucleus applied independently at each position."""
    if len(temperatures) != logits.shape[0] or any(t not in (0, 1) for t in temperatures):
        raise ValueError('one temperature (0 or 1) per row required')
    if isinstance(top_p, bool) or not 0 < top_p <= 1:
        raise ValueError('top_p must be in (0, 1]')
    distributed = dist.is_initialized()
    if not distributed or dist.get_rank() == 0:
        sampled = [i for i, t in enumerate(temperatures) if t == 1]
        if not sampled:
            tokens = logits.argmax(-1)
        else:
            all_sampled = len(sampled) == len(temperatures)
            if all_sampled:
                scores = logits
            else:
                index = torch.tensor(sampled, device=logits.device, dtype=torch.long)
                scores = logits.index_select(0, index)
            cdf, ids = nucleus_cdf(scores.reshape(-1, scores.shape[-1]))
            uniforms = torch.rand(cdf.shape[0], device=logits.device, dtype=torch.float32)
            selected = inverse_cdf(cdf, ids, uniforms, top_p).reshape(scores.shape[:-1])
            if all_sampled:
                tokens = selected
            else:
                tokens = logits.argmax(-1)
                tokens.index_copy_(0, index, selected)
    else:
        tokens = torch.empty(logits.shape[:-1], device=logits.device, dtype=torch.long)
    if distributed:
        dist.broadcast(tokens, src=0)
    return tokens
