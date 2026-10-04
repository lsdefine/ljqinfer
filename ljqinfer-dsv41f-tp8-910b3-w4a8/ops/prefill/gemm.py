"""Tensor GEMM over the released W8 dequant kernel."""
from .native import ops


def linear(x, weight, scale=None):
    if scale is None:
        return x @ weight.to(x.dtype).t()
    expanded = ops.w8_dequant(weight, scale.reshape(-1).float().contiguous())
    return x @ expanded.t()
