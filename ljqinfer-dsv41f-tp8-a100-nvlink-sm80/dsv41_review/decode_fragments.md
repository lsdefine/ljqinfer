# Decode fragment audit (B1Q6, TP8, A100 sm80)

The step time is not one hot kernel. A TorchDispatchMode probe over a single
eager step counted **7318 aten calls**, and almost all of them come from leaves
written for prefill that decode happens to reuse.

## How the probe works (reproducible)

`bench/decode_graph.py --dispatch` wraps one eager `model.forward` in a
TorchDispatchMode that keys a counter on `(op, python call site)`, prints the
top sites plus `DISP_TOTAL`, and returns before graph capture so the run costs
about a minute. The probe is a diagnostic, it is not committed.

## What it found (calls per step)

| calls | ops | site |
|-------|-----|------|
| 290 each | empty, view, t, **mm.out**, view | ops/prefill/projection_workspace.py:74/87 |
| 80 each | vector_norm, mul, div, rsqrt, add, mul, linear | ops/prefill/residual.py mixes |
| 80 each | float x2, clamp x2, silu, mul x2, to | ops/prefill/residual.py swiglu |
| ~20/layer | the engram gate expansion | ops/prefill/residual.py engram_gate |
| 160 | to.dtype | ops/prefill/hc_mix.py:54 |
| 80 | empty | ops/prefill/moe_workspace.py |

Every one of these files lives under `ops/prefill/`.

## Cuts landed against it

- **55e99e9** `_cast_w` caches the constant weight casts in the residual
  leaves (mixes gate, engram q/k, router weight), keyed on `data_ptr` the way
  `v4k._f32_weight` already did for norm weights. B1Q6 21.983 -> 21.174 ms,
  `LOGITS_EQUAL True maxdiff 0.0`.
- **(pending)** `swiglu` now calls the triton leaf in
  `ops/decode/quant_fused.py`, which was already in the tree and unused by this
  path, collapsing an eight kernel expansion into one launch.

## Kernels that exist but are not wired

`ops/decode/v4k.py` binds six entry points. The V4 tree exports far more that
answer fragments listed above: `silu_mul_clamp_bf16`, `rms_scale_`,
`hc_fused_pre`, `hc_post_fused_bf16`, `sinkhorn_hc`, `add2_bf16_f32`,
`scatter3_rows`, `fp8_linear`, `bf16_gemm`, `attn_decode_fused_r0`.
V4.1's own cuda sources already export `activation`, `glu_quant`,
`engram_gate`, `rms_norm`, `dense_projection`, `sparse_attn_decode`.

## Next targets, largest first

1. `projection_workspace` issues **290 separate `mm.out`** per step. This is
   the single largest fragment source left.
2. `engram_gate` and `mixes` are still ATen expansions with fused kernels
   available.
3. `hc_mix.py:54` casts 160 times per step.
