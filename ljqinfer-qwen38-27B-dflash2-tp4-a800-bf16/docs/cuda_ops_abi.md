# CUDA operator ABI notes (ops/kernels.py, ops/cuda_ops.py)

Single CUDA path; no env switch, no NPU fallback inside these files.

## Frozen signatures kept from 47e014b
`rope_frequencies(positions, rotary_dim, theta, dtype)` returns the **half-width**
`(cos, sin)` pair of shape `[T, rotary_dim/2]` in the model dtype; `apply_rope(x, frequencies, rotary_dim)`
takes a single tensor. `qk_rms_norm_rope_decode` is the fused leaf and requires exactly that layout.

## Integration requirements
- Weights must arrive **BF16** and same-device. `w8a8_linear` keeps its historical name and
  `(x, linear)` call shape, but INT8 payloads raise instead of being dequantized in the hot path.
- GDN Q8 decode: `q/k/v/beta/state/base_state` BF16, decay `g` **FP32**, `state` is the pending
  per-token snapshot buffer, `base_state` is read-only, `ssm_state_indices` selects the destination
  on the GPU (graph safe). `actual_seq_lengths` / `num_accepted_tokens` stay in the signature but
  all Q8 tokens execute, matching the base-pointer native kernel.
- Chunk prefill returns `(out, final_state, checkpoints)` with `h` as `[B, HV, NT, Dk, Dv]`;
  the hot cache layout is `[B, HV, Dv, Dk]` (transposed), matching `transpose_state_layout=True`.
