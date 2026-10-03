# Decode round optimization — 2026-10-02

Baseline 6f55a13; node09, A100 SM80 TP8, B1, Q8 DFlash2.

## Changes
- MLA block-wide vote skips wholly invalid tiles; still inspect each tile.
  No assumption that IDs are sorted/compact; no early break on holes.
- Reuse elementwise RMS/RoPE/query/cache packing fusion for decode and share
  per-step position state. Preserve prefill-only token projection gating.
- Capture append separately for accepted lengths 1..8 with original GEMM
  shapes. Retain offsets and inputs for graph lifetime. Save/restore the first
  KV page around capture; allocate/reserve pages outside replay.

## Integrated measurement
12 prompts x 384 output tokens, programming/explanation/writing/math/thinking.
Warm graphs, complete decode-loop wall time, not just verify, not HTTP TTFT.
Weighted round: 70.14256 -> 48.62456 ms (-30.68%).
Aggregate decode: 41.18 -> 59.41 token/s (+44.25%).
Each case mean: 48.33..49.04 ms, not a per-round maximum guarantee.

| Case | Before ms/round | After ms/round | Before tok/s | After tok/s |
|---|---:|---:|---:|---:|
| cpp_debug | 69.78 | 49.04 | 40.06 | 57.01 |
| cpp_debug_think | 70.98 | 48.54 | 47.33 | 69.22 |
| english_explain | 69.28 | 48.58 | 39.49 | 56.32 |
| explain_zh | 69.34 | 48.48 | 40.31 | 57.67 |
| explain_zh_think | 69.64 | 48.56 | 36.91 | 52.94 |
| python_lru | 70.39 | 48.33 | 49.92 | 72.70 |
| python_lru_think | 70.69 | 48.86 | 38.70 | 55.99 |
| reasoning_math | 69.71 | 48.38 | 51.35 | 73.99 |
| reasoning_math_think | 70.95 | 48.53 | 65.04 | 95.08 |
| sql_window | 71.32 | 48.99 | 41.63 | 60.60 |
| typescript_async | 69.79 | 48.54 | 44.62 | 64.15 |
| writing_zh | 70.27 | 48.57 | 24.44 | 35.36 |

Separate instrumented runs: draft3.71ms, verify42.72..42.75ms,
commit0.71..0.72ms. These do not include all other host work and are not
substitutes for the uninstrumented complete-loop measurements.

## Correctness and diagnostic history
- All 12 token sequences and accepted-draft sequences match baseline.
  All eight ranks have 12 result records.
- MLA leaf: 10 cases exact, including empty/holes/full2048 IDs.
- Production append: 8 lengths x4 positions including page boundaries,
  all 8 ranks KV exact versus eager; capture preserves resident KV.
- Original integrated harness stopped AFTER timing/profile results at an
  extra cold/hot equality assertion. The whole harness run did not pass.
- Separate baseline and optimized cold/hot/reset runs both exit0. For each
  same path, tokens AND acceptance match across versions. In both versions
  cold!=hot while cold==reset. This pre-existing difference is not fixed;
  its numerical root cause is not established. The saved harness records
  the distinction and supports same-path baseline comparison.

## Reproduction/evidence
Use /mnt/data/kw/anaconda3/bin/python -m torch.distributed.run
--standalone --nproc-per-node=8 with:
- tests/bench_decode_round.py (DECODE_AUDIT_DIR, optional DECODE_CACHE_BASELINE)
- tests/bench_dflash_append_graph.py (DECODE_AUDIT_DIR)
- tests/bench_decode_cache_ab.py (PROBE_REPO, PROBE_VARIANT, DECODE_AUDIT_DIR)

Raw: /mnt/data2/kw/glm53_int4_tp8/service_audit/decode50/
Files: decode_optimization_summary.json; integrated/*.json and rank*.jsonl;
baseline.cache.json / optimized.cache.json and .exit markers;
append_production.rank*.json; mla_skip_leaf.json.

## Limits
B1 short prompts, 384 output tokens, warmed graphs only. No new long-context,
concurrency, independent HF-truth, or HTTP benchmark. HTTP stays stopped.
Prefill path preserved; 12K prefill performance was not rebenchmarked.
