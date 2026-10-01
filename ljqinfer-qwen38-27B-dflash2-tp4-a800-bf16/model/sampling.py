"""Top-k then top-p target sampling, one shard-candidate collective.

Only the candidate buffer is copied to the host. The random exponential for
 each candidate travels with it, so all ranks make the same decision without
 a second collective or synchronized generator state. T=0 is greedy.
"""
import math
import secrets
import torch

TOP_K = 20
TOP_P = 0.95
_GENERATORS = {}


def temperatures(value, count):
    values = ([float(value)] * count if isinstance(value, (int, float))
              else [float(x) for x in value])
    if len(values) != count or any(not math.isfinite(t) or t < 0 for t in values):
        raise ValueError('temperature must be finite, nonnegative and one per request')
    return values


def sample_pairs(logits, temps, rank, world, *, generator=None):
    """Return [row, local top-k, (logit, global token id, Exp(1))].

    The global top-k is a subset of the union of local top-k sets. Randomness
    is independent of candidate selection; filtering must use raw logits.
    """
    rows, width = logits.shape
    if not temps or rows % len(temps) or len(temps) > 4:
        raise ValueError('rows must be grouped by 1..4 requests')
    if width * world >= 2**24 or not 0 <= rank < world:
        raise ValueError('global token ids must be representable in float32')
    values, ids = logits.topk(min(TOP_K, width), dim=-1, sorted=False)
    values = values.float()
    if any(t == 0 for t in temps):
        # Backend topk can omit the lowest-id maximum when many logits tie.
        # Preserve the original argmax policy for greedy rows in mixed batches.
        greedy_ids = logits.argmax(dim=-1, keepdim=True)
        greedy_values = logits.gather(-1, greedy_ids).float()
        greedy_mask = torch.tensor(
            [t == 0 for t in temps for _ in range(rows // len(temps))],
            dtype=torch.bool, device=logits.device).unsqueeze(-1)
        ids = torch.where(greedy_mask, greedy_ids, ids)
        values = torch.where(greedy_mask, greedy_values, values)
    if any(temps):
        if generator is None:
            key = (str(logits.device), rank)
            if key not in _GENERATORS:
                _GENERATORS[key] = torch.Generator(device=logits.device).manual_seed(
                    secrets.randbits(63))
            generator = _GENERATORS[key]
        noise = torch.empty_like(values).exponential_(generator=generator)
    else:
        noise = torch.ones_like(values)
    return torch.stack((values, ids.float().add_(rank * width), noise), dim=-1)


def select_candidates(candidates, temps):
    """Merge host [world, rows, k, 3] candidates, truncate, then sample.

    Small Python lists deliberately avoid a CPU torch thread-pool launch on
    every decode step. Keep the token that crosses the nucleus threshold.
    Boundary logit ties inside local top-k follow the backend topk policy;
    ties among transmitted candidates use the lower global id first.
    """
    rows = len(candidates[0])
    if not temps or rows % len(temps):
        raise ValueError('rows must be grouped by request')
    rows_per_request = rows // len(temps)
    result = []
    for row in range(rows):
        pool = sorted((item for shard in candidates for item in shard[row]),
                      key=lambda item: (-item[0], item[1]))[:TOP_K]
        t = temps[row // rows_per_request]
        if t == 0:
            result.append(int(pool[0][1]))
            continue
        # Subtract before dividing, avoiding overflow for small temperatures.
        scaled = [(item[0] - pool[0][0]) / t for item in pool]
        weights = [math.exp(v) for v in scaled]
        cutoff = TOP_P * math.fsum(weights)
        cumulative = 0.0
        best_score, best_id = -math.inf, None
        for (logit, token, noise), value, weight in zip(pool, scaled, weights):
            score = value - math.log(max(noise, 1.1754943508222875e-38))
            if score > best_score:
                best_score, best_id = score, int(token)
            cumulative += weight
            if cumulative >= cutoff:
                break
        result.append(best_id)
    return result
