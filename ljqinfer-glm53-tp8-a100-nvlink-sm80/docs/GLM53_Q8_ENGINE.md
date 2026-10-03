# GLM53 Q8 target-engine checkpoint — 2026-10-01

## Scope
Use model.glm53_engine.Engine under eight-rank torchrun in the only repository
/mnt/data/kw/ljqinfer_glm53_tp8. Full 78-layer target prefill, Q8 verify, prefix
commit and CUDA graph are implemented. Prefill/decode remain separate operators.
Q8 = anchor plus seven drafts. Decode uses existing ops.moe_decode.moe_decode,
without kernel changes. The temporary Q8 kernel and override hook are deleted.
The legacy model.model facade/service is NOT migrated.

MTP pool: six layers, one local KV head/rank, head dimension 128, paged storage,
sequence isolation, append/truncate/release. Features are post-layer/pre-final-norm
at layers 5,19,33,47,61,75. Only consumed-prefix features reach the commit callback.
DFlash weights, projection, draft network and B1 generation loop are implemented
in the current working tree; service integration remains unfinished.

## Restored existing decode kernel
- No changes to ops/moe_decode.py or ops/moe_prefill.py.
- Eight-rank real-weight DFlash generation: two prompts, 24 tokens each, repeated
  outputs identical. All eight restored_decode/rank*.json report PASS.
- Short-prefix full 78-layer Q8 verify wall time: median 59.034ms across 13 steps;
  range 58.776--61.924ms. Excludes draft and commit; not an 8K/64K benchmark.
- The previous approximately 288ms result used the removed slow kernel.
- Existing prefill/decode strict logits-equivalence gate remains failing
  (relative L2 approximately .126 in old_moe). Repeatability is not independent
  quality proof; the gate is not relaxed for this switch.
- Evidence: /mnt/data2/kw/glm53_int4_tp8/engine_q8_audit/restored_decode/.

## Historical validation before kernel removal (not current guarantees)
- Real model on eight ranks: same-Q prefill/verify logits and six features exact;
  causal prefix under suffix perturbation exact; partial commit/replay exact;
  eager/graph and dynamic IDs/positions exact. Eight production/rank*.json PASS.
- Pool: eight CPU tests plus one CUDA lifecycle test PASS.
- Independent Torch MoE oracle: eight records including prefill controls;
  INT4 and INT4/FP8, toy and target-width shapes, eager/graph/changed routing.
- Three natural texts: 16 identical-input repeats each have one logits/features
  hash on every rank. Cross-Q prefix recomputation: 24/24 top1 positions agree.
  Mean KL: 0.004934, 0.006534, 0.013970; relative L2: .03969, .04663, .07466.
  This is teacher-forced comparison, not generated text or independent model truth.
- Random-token cross-Q strict L2 gate remains FAILED (~.1806; full reduction .1654).
  First observed drift is layer-0 attention (~.000255); root cause not proven.
  CHECK_CROSS_Q=1 retains the strict gate. No complete quality PASS is claimed.
- Actual short-prefix full-model Q8 graph time ~288.1ms/step (five iterations).
  NOT a speedup claim, 8K/64K benchmark, or generated TPS. Performance remains open.

## API lifetime
Engine.load(capacity=...,prefill_chunk_tokens=...); all ranks call identically.
prefill returns last chunk; on_commit visits each chunk. verify requires eight IDs;
commit(0..8) resolves pending suffix. Callbacks must project/append committed features.
Graph results and commit slices alias reusable buffers: consume/clone before replay.
reset releases logical context/MTP slot zero, not the graph. Reset/destroy CUDA graphs
BEFORE destroying the process group (otherwise teardown can hang). Current engine is B1.

## Reproduce
Python/torchrun: /mnt/data/kw/anaconda3/bin/.
python -m pytest -q tests/test_glm53_mtp_pool.py
AUDIT_DIR=/fresh/path torchrun --standalone --nproc-per-node=8 tests/test_glm53_engine_q8.py
CHECK_CROSS_Q=1 AUDIT_DIR=/fresh/cross torchrun --standalone --nproc-per-node=8 tests/test_glm53_engine_q8.py
torchrun --standalone --nproc-per-node=8 tests/diag_q8_text.py
Evidence: /mnt/data2/kw/glm53_int4_tp8/engine_q8_audit/{production,text,full_accum}/,
verify8_oracle.json and cross_diag0.json. Do not overlap two real-model GPU runs.
