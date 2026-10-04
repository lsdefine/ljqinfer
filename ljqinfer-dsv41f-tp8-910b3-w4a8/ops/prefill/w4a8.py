"""Live-shape HP W4A8 composition; all launches use Tensor operators."""
import torch


def linear(x, weight, scale, bias, counts):
    """Released HP W4A8 composed through registered Tensor operators."""
    from .native import ops
    kp = weight.shape[1]
    if kp != (x.shape[-1] + 63) // 64 * 64:
        raise ValueError('prepared weight K must match padded input K')
    raw, token_scale = ops.dynamic_quant(x)
    quantized = raw
    if kp != x.shape[-1]:
        quantized = torch.zeros((len(x), kp), dtype=torch.int8, device=x.device)
        quantized[:, :x.shape[-1]].copy_(raw)
    return ops.grouped_int4(quantized, weight, scale, bias, token_scale, counts)
