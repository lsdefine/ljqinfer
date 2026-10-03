"""Offline data-free symmetric INT4 quantizer; no activation/corpus dependency.

Weights [out, K]; per-row K groups (64/128); FP16 multiplicative scales.
Signed two's complement nibbles, even K low, odd K high, range [-7,7].
Search minimizes source FP32 weight SSE using the actual stored FP16 scale.
This is weight-only MSE fitting, not GPTQ or real-activation calibration.
"""
import torch


@torch.no_grad()
def quantize_mse(weight, group_size=128, *, chunk_rows=128):
    if weight.ndim != 2 or group_size not in (64, 128):
        raise ValueError('Expected a matrix and group_size 64 or 128')
    n, k = weight.shape
    if n == 0 or k == 0 or k % group_size or chunk_rows <= 0:
        raise ValueError('Invalid shape or chunk_rows')
    if not torch.isfinite(weight).all():
        raise ValueError('Non-finite source weights')
    packed = torch.empty((n, k//2), device=weight.device, dtype=torch.uint8)
    scales = torch.empty((n, k//group_size), device=weight.device, dtype=torch.float16)
    for start in range(0, n, chunk_rows):
        w = weight[start:start+chunk_rows].float().reshape(-1, k//group_size, group_size)
        raw = w.abs().amax(-1)/7
        if (raw > torch.finfo(torch.float16).max).any() or ((raw > 0) & (raw.half() == 0)).any():
            raise ValueError('Source scale is outside FP16 range')
        base = torch.where(raw == 0, torch.ones_like(raw), raw).half()

        def encode(s):
            q = (w/s.float().unsqueeze(-1)).round().clamp(-7, 7)
            error = (w-q*s.float().unsqueeze(-1)).square().sum(-1)
            return q, error

        best = base.clone()
        _, best_error = encode(best)
        # Include absmax RTN; every accepted change decreases group SSE.
        for step in range(1, 21):
            candidate = (base.float()*(1-0.02*step)).half()
            candidate = candidate.clamp_min(torch.finfo(torch.float16).smallest_normal)
            _, error = encode(candidate)
            take = error < best_error
            best = torch.where(take, candidate, best)
            best_error = torch.minimum(best_error, error)
        # Fixed-code least squares, then re-round codes and accept only improvements.
        for _ in range(4):
            q, _ = encode(best)
            denom = q.square().sum(-1)
            fitted = (w*q).sum(-1)/denom.clamp_min(1)
            candidate = torch.where(denom > 0, fitted, best.float()).half()
            candidate = candidate.clamp_min(torch.finfo(torch.float16).smallest_normal)
            _, error = encode(candidate)
            take = error < best_error
            best = torch.where(take, candidate, best)
            best_error = torch.minimum(best_error, error)
        q, _ = encode(best)
        q = q.to(torch.int8).reshape(-1, k)
        end = start+q.shape[0]
        packed[start:end] = (q[:, 0::2].to(torch.uint8)&15) | ((q[:, 1::2].to(torch.uint8)&15)<<4)
        scales[start:end] = best
    return packed, scales
