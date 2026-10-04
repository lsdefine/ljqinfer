# Operator contract, layer by layer

What the assembled model asks of `ops/`, derived from the code that calls it
(`model/prefill_layer.py`, `prefill_block.py`, `prefill_attention.py`,
`prefill_linears.py`, `prefill_build.py`) and from the released constants in
`model/v41_config.json`.  Nothing here is aspirational: every entry is a call
site that already exists.

## Three standing rules

1. No run-time branch.  Shapes come from the build, never from data.
2. No device-to-host copy on the compute path.  `.item() .cpu() .tolist()
   nonzero() masked_select() synchronize()` are build-time or audit-time only.
3. No allocation inside an operator.  Buffers are cut once at build time and
   passed in; an operator writes into `out`.

An operator that cannot honour all three is not finished, however correct its
numbers are.

## Frozen shapes (released V4.1, TP8)

| Name | Value | Per rank |
| --- | --- | --- |
| chunk tokens | 12288 | 12288 |
| dim | 5120 | 5120 |
| heads | 64 | 8 |
| head_dim / rope_head_dim | 512 / 64 | same |
| q_lora_rank / o_lora_rank | 1280 / 1024 | same |
| o_groups | 8 | 1 group |
| window_size | 128 | 128 |
| routed experts / topk | 384 / 6 | 48 experts |
| moe_inter_dim | 2304 | 2304 |
| shared experts | 1 | TP-split |
| backbone dtype / expert dtype | fp8 | fp4 |
| norm_eps / hc_eps | 1e-20 / 1e-6 | same |
| hc_mult / sinkhorn iters | 4 / 20 | same |

## Layer 0 role

`compress_ratios[0] == 0` so layer 0 is **pure sliding window**: no compressor,
no global rows, no index projection, no selection, no candidates.
`engram_layer_ids == [1, 14]` so layer 0 has no Engram.  Rotary uses
`rope_theta=10000` with no YaRN ramp (the compressed base 160000 starts at
layer 2).  The FFN is **not** dense - every one of the 40 layers routes over
384 FP4 experts plus one shared expert.  Layer 0 is therefore the smallest
*attention* layer but a full-weight *MoE* layer.

## Operators layer 0 needs

Reference column: `S` = CUDA implementation under `S:/ljqinfer_dsv41f_tp8/ops/prefill/`,
`P` = slow PyTorch reference in the model directory.  Both are read for
semantics; neither is transcribed.

| # | Entry | Signature as called | Shapes | Reference |
| --- | --- | --- | --- | --- |
| 1 | `gemm` / `PrefillLinear` | `lin(name, x) -> y` | fp8 weight GEMM, `(12288,5120)` against q_lora 1280, kv 512+64, o_lora 1024 | `S gemm.py`, `fp4_gemm.py` |
| 2 | `residual.rms` | `rms(x, w, eps) -> y` | `(12288,5120)`, eps 1e-20 | `S residual.py` |
| 3 | `residual.mixes` | `mixes(x, fn, scale, base, norm_eps, hc_eps, iters) -> (pre, post, comb)` | Sinkhorn 20 iters, hc_mult 4 | `S hc_mix.py` |
| 4 | `residual.collapse` | `collapse(h, pre) -> x` | head collapse `4*dim -> dim` | `S hc_mix.py` |
| 5 | `residual.expand` | `expand(x, h, post, comb) -> h` | inverse of 4 | `S hc_mix.py` |
| 6 | `attention.rope` | `rope(x, freqs) -> y` | `(12288,heads,64)` complex table | `S attention.py` |
| 7 | `quant.fp8_roundtrip` / `fp4_roundtrip` | `(x, block=32) -> x'` | activation quant, block 32 | `S quant.py` |
| 8 | `attention.sparse` (SWA path) | `sparse(q, local, local_start, None, None, pos, sink, window=128, scale, query_start)` | causal + 128 window + sink, no global rows | `S sparse_attn.py`, `swa_initial.py` |
| 9 | `residual.swiglu` | `swiglu(g, u, limit=10.0, prob) -> y` | shared expert and per-expert FFN | `S residual.py` |
| 10 | `residual.route` | `route(x, w, b, topk=6, temperature=1.0, scale=1.5, normalize=True, score='sqrtsoftplus') -> (prob, ids)` | gate over 384 | `S selection.py` |
| 11 | `npu_routed.PackedRouted` | `routed(x, ids, prob) -> out` | FP4 grouped MoE, dispatch+GEMM+combine complete on rank | `S grouped_moe.py`, `expand.py` |
| 12 | collective | `parallel.sum(t)` | TP8 all-reduce over `(12288,5120)` | `S fast_allreduce.py` |

Not needed at layer 0, deliberately absent: compressor, index projection and
scoring, top-k selection, candidate blocks, Engram gate and rows, paged or
cold KV.  They enter at layers 1 (Engram), 2 (source) and 20 (candidates).

### The one hard operator

`PackedRouted` is the only place where data decides work.  The invariant that
keeps it inside the rules: the number of token-expert pairs is exactly
`chunk * topk = 73728`, a build-time constant.  Sort by expert id on device,
cut into fixed blocks, let each block carry its expert id and reach the weight
through an id-indexed base address.  No capacity factor, no dropped token, no
host round trip, and the scratch is a fixed `73728 x moe_inter_dim`.

## Debts already visible in the model layer

Cleared on 2026-09-23 (commit below).  The three run-time branches the model
layer still carried were not in the CUDA reference at all; they grew during the
910B port, and each one is now a build-time decision:

- `prefill_attention.attend_swa` no longer probes `swa_initial.supported(...)`
  on the first chunk.  A fused cold-window path, if it is worth having, belongs
  inside the sparse operator where the empty history is a shape, not a test.
- `_bind_engram_gate(..., rows)` now takes the chunk width, so the sharded gate
  is chosen once; `_engram_sharded` asserts the promised geometry instead of
  silently falling back to the all-reduce path.
- `_moe_overlapped` no longer re-checks `len(x)`; the chunk width is fixed by
  the build, and `OVERLAP_ROWS` is gone.

What stays is what the CUDA reference also does: raise on a geometry the build
never promised.  An assertion is not a branch -- it has one live path.

## Two shape regimes, one operator set

`model/prefill.py` runs `blocks[:ENCODER_LAYERS]` (ENCODER_LAYERS = 20) over a
variable-width chunk, then evaluates only layer 20's global projection to write
`ckv` / `index_k`, and returns.  Layers 20..39 never execute a full block during
prefill: they run in the replay/decode pass over a fixed 128 rows.

So an operator is written once and accepted twice:

- encoder regime, layers 0..19, variable rows, eager.  Correctness only:
  parity against `/data/models/DeepSeek-V4.1-Flash/inference/model.py`.
- CED regime, layers 20..39, exactly 128 rows, graph captured.  Parity plus
  100 bitwise-identical replays and no growth in peak device memory.

The second regime is where the three rules are load-bearing.  The first is
where the reference answers come from.  An operator that passes only the first
is not done.

Note on the reference: the dataclass defaults in `inference/model.py`
(head_dim 128, 8 experts, rope_head_dim 32) are toy values for the file's own
smoke test.  The released geometry is `config.json` next to it, which agrees
with `model/v41_config.json`: head_dim 512, qk_rope_head_dim 64, 384 routed
experts, sliding_window 128.

## Order of work

Layer 0 first, then layer 1 reusing whatever layer 0 already produced and
adding only what is missing (Engram), then layer 2 (compressor, index,
selection).  Each operator ships with a correctness comparison against the
slow PyTorch reference and, before it counts as done, a capture-replay check:
graph captures, 100 replays are bitwise identical, and peak device memory does
not move between steps.
