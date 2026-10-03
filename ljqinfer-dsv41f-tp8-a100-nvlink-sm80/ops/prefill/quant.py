"""Prefill activation quantization reference; replace whole calls with kernels."""
import torch


_E2M1_TABLES = {}


def _e2m1_table(device):
    """The eight E2M1 magnitudes, materialized once per device."""
    table = _E2M1_TABLES.get(device)
    if table is None:
        table = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6.], device=device)
        _E2M1_TABLES[device] = table
    return table


def fp4_roundtrip(x, *, block, e4m3_scale):
    if x.shape[-1] % block:
        raise ValueError('FP4 block alignment')
    z = x.float().unflatten(-1, (-1, block))
    amax = z.abs().amax(-1, keepdim=True)
    if e4m3_scale:
        s = (amax.clamp_min(6*2**-9)/6).to(torch.float8_e4m3fn).float()
    else:
        s = torch.exp2(torch.ceil(torch.log2(amax.clamp_min(6*2**-126)/6)))
    u = (z/s).clamp(-6, 6)
    # E2M1 round-to-nearest-even: even code wins exact midpoint ties.
    table = _e2m1_table(x.device)
    mid = (table[1:]+table[:-1])/2
    index = torch.bucketize(u.abs().contiguous(), mid)
    low = index.clamp(max=6)
    tie = (index < 7) & (u.abs() == mid[low]) & (index % 2 == 1)
    index = index + tie.long()
    return (torch.copysign(table[index], u)*s).flatten(-2).to(x.dtype)
