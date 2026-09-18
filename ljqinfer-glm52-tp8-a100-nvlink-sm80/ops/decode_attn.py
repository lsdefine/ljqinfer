"""Permanent decode-attention boundary shared by B=1 and B>1.

Every model path calls :func:`decode_attn` with explicit batched tensors.  The
public contract never changes with batch size or query width.  Dispatch and
all implementation-specific shape adaptation stay below this boundary.

Semantic contract:
- ``x`` is [B,Q,6144], ``positions`` is [B,Q];
- ``page_tables`` is [B,P], ``context_lengths`` is int32 [B];
- the only permitted externally visible mutation is appending Q KV rows per
  request to ``kv_workspace`` through its page table at its context length;
- metadata, activations and weights are read-only;
- the result is a non-input-aliasing [B,Q,6144] tensor.
"""
from __future__ import annotations

import torch


D = 6144
MAX_Q = 6
ABI = "decode_attn/v1"


def decode_attn(
    op,
    *,
    x: torch.Tensor,
    positions: torch.Tensor,
    kv_workspace: torch.Tensor,
    page_tables: torch.Tensor,
    context_lengths: torch.Tensor,
    attn_norm: torch.Tensor,
    q_a: torch.Tensor,
    q_a_norm: torch.Tensor,
    q_b: torch.Tensor,
    kv_a: torch.Tensor,
    kv_a_norm: torch.Tensor,
    k_b: torch.Tensor,
    v_b: torch.Tensor,
    attn_out: torch.Tensor,
) -> torch.Tensor:
    """Run decode attention through one B-agnostic, Q-agnostic interface."""
    if x.ndim != 3 or x.shape[2] != D:
        raise ValueError(f"{ABI}: x must be [B,Q,{D}], got {tuple(x.shape)}")
    B, Q, _ = x.shape
    if B < 1 or Q < 1 or Q > MAX_Q:
        raise ValueError(f"{ABI}: expected B>=1 and 1<=Q<={MAX_Q}, got B={B}, Q={Q}")
    if positions.shape != (B, Q):
        raise ValueError(f"{ABI}: positions must be [{B},{Q}]")
    if page_tables.ndim != 2 or page_tables.shape[0] != B:
        raise ValueError(f"{ABI}: page_tables must be [B,P] with B={B}")
    if context_lengths.shape != (B,):
        raise ValueError(f"{ABI}: context_lengths must be [{B}]")
    if x.dtype != torch.float16:
        raise TypeError(f"{ABI}: x must be float16")
    if positions.dtype != torch.int64 or page_tables.dtype != torch.int64:
        raise TypeError(f"{ABI}: positions/page_tables must be int64")
    if context_lengths.dtype != torch.int32:
        raise TypeError(f"{ABI}: context_lengths must be int32")
    if not (x.is_cuda and positions.is_cuda and kv_workspace.is_cuda
            and page_tables.is_cuda and context_lengths.is_cuda):
        raise ValueError(f"{ABI}: runtime tensors must be CUDA tensors")
    device = x.device
    if any(t.device != device for t in
           (positions, kv_workspace, page_tables, context_lengths)):
        raise ValueError(f"{ABI}: runtime tensors must share one device")
    if not all(t.is_contiguous() for t in
               (x, positions, kv_workspace, page_tables, context_lengths)):
        raise ValueError(f"{ABI}: runtime tensors must be contiguous")

    impl = _decode_attn_bn_impl
    result = impl(
        op, x=x, positions=positions, kv_workspace=kv_workspace,
        page_tables=page_tables, context_lengths=context_lengths,
        attn_norm=attn_norm, q_a=q_a, q_a_norm=q_a_norm, q_b=q_b,
        kv_a=kv_a, kv_a_norm=kv_a_norm, k_b=k_b, v_b=v_b,
        attn_out=attn_out)
    if not isinstance(result, torch.Tensor) or result.shape != x.shape:
        shape = getattr(result, "shape", None)
        raise AssertionError(f"{ABI}: implementation returned {shape}, expected {x.shape}")
    if result.device != device or result.dtype != x.dtype:
        raise AssertionError(f"{ABI}: implementation changed output device/dtype")
    for source in (x, positions, page_tables, context_lengths, kv_workspace):
        if result.data_ptr() == source.data_ptr():
            raise AssertionError(f"{ABI}: output aliases an input")
    return result


def _decode_attn_bn_impl(
    op, *, x, positions, kv_workspace, page_tables, context_lengths,
    attn_norm, q_a, q_a_norm, q_b, kv_a, kv_a_norm, k_b, v_b, attn_out,
):
    """Native Bn implementation; ragged paging stays below this boundary."""
    B, Q, D = x.shape
    returned_pool, partial = op.forward_rank_paged_batch_k0(
        x.reshape(B * Q, D), positions.reshape(B * Q), kv_workspace,
        list(page_tables.unbind(0)),
        [context_lengths[b:b + 1] for b in range(B)],
        attn_norm, q_a, q_a_norm, q_b, kv_a, kv_a_norm, k_b, v_b, attn_out)
    if returned_pool.data_ptr() != kv_workspace.data_ptr():
        raise AssertionError(f"{ABI}: implementation replaced KV workspace")
    return partial.reshape(B, Q, D)
