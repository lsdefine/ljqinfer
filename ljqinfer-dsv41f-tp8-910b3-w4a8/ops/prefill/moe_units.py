"""Replaceable eager W4A8 expert units. No model, Past or communication state.

All launches follow the current stream. Returned tensors are owned outputs;
intermediate allocations are explicit in these v1 compositions. Stable token-
major sorting and FP16 -> BF16 GMM rounding are part of the numerical ABI.
"""
import torch
import torch.nn.functional as F
from .native import ops
from .residual import route


def dispatch_quant(x, ids, *, experts):
    """BF16[T,D], INT64[T,K] -> (INT8 rows, FP32 scales, INT64 counts/order)."""
    if experts > 2 ** 24:
        raise ValueError('expert IDs exceed exact FP32 sort range')
    flat = ids.flatten()
    order = flat.float().argsort(stable=True)
    raw, scale = ops.dynamic_quant(x)
    rows = order // ids.shape[1]
    counts = torch.zeros(experts, dtype=torch.int64, device=x.device)
    counts.scatter_add_(0, flat, torch.ones_like(flat))
    return (raw.index_select(0, rows), scale.index_select(0, rows), counts, order)


def grouped_linear(quantized, weight, scale, bias, token_scale, counts):
    """INT8[P,Kpad] + packed NZ INT4 weights -> BF16[P,N].

    scale is immutable npu_trans_quant_param output; bias is HP correction.
    The kernel emits FP16; BF16 rounding is mandatory before activation/combine.
    """
    return ops.grouped_int4(quantized, weight, scale, bias, token_scale, counts).to(torch.bfloat16)


def activation_quant(hidden, probability, *, padded_dim):
    """BF16/FP16[P,2I] + FP32[P] -> INT8[P,Kpad], FP32[P].

    Released top6/limit10 SwiGLU includes routing weight and BF16 rounding
    before dynamic quantization. INT8 padding is zero, not re-quantized.
    """
    gated = ops.routed_swiglu(hidden, probability)
    raw, scale = ops.dynamic_quant(gated)
    if padded_dim < raw.shape[-1]:
        raise ValueError('padded_dim must cover the activation width')
    if padded_dim != raw.shape[-1]:
        raw = F.pad(raw, (0, padded_dim - raw.shape[-1]))
    return (raw, scale)


def grouped_linear_activation(quantized, weight, scale, bias, token_scale, counts,
                              probability, *, padded_dim):
    """W13 W4A8 -> quantized activation; preserve BF16 rounding inside SwiGLU."""
    hidden = ops.grouped_int4(quantized, weight, scale, bias, token_scale, counts)
    return activation_quant(hidden, probability, padded_dim=padded_dim)


def combine(values, order):
    """BF16[T*6,D] expert-major -> FP32[T,D] rank-local fixed-order sum.

    Caller performs TP reduction. No atomics or shared expert rounding here.
    """
    inverse = torch.empty_like(order)
    inverse.scatter_(0, order, torch.arange(order.numel(), device=order.device))
    return ops.routed_combine(values, inverse)


def grouped_linear_combine(quantized, weight, scale, bias, token_scale, counts, order):
    """W2 W4A8 -> FP32[T,5120]; preserve BF16 rounding inside fixed-order sum.

    Owns the GMM temporary; large chunks avoid BF16 materialization. TP stays outside.
    """
    # Keep short-chunk latency; fusion is validated for 4096/8192 tokens.
    if order.numel() < 4096 * 6:
        return combine(grouped_linear(quantized, weight, scale, bias, token_scale, counts), order)
    values = ops.grouped_int4(quantized, weight, scale, bias, token_scale, counts)
    inverse = torch.empty_like(order)
    inverse.scatter_(0, order, torch.arange(order.numel(), device=order.device))
    return ops.routed_combine_half(values, inverse)
