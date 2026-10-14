# A23 decode leaf optimization — 2026-10-14

## Scope and implementation
- Previous usable point: `6caae72`. A23 only, port 8001. A21/A22/R unchanged.
- Sampling finish: four rows per AIV task; a subsequent seed-update kernel avoids cross-task seed races. Full vocabulary inverse-CDF/top-p semantics and ABI unchanged.
- FP4 preparation: repartition independent quantization blocks for small row counts; preserve the original kernel, scaling, rounding, input/output buffers and overlap validation. Four host-side lines, no second production algorithm.
- Resource lifecycle: vendor indexer constructor unwinds partial ownership; library singleton published only after successful initialization; TargetLayer owns returned child plans immediately and closes all owned plans on failure. Borrowed weights/Past/TP/scratch remain caller-owned.
- No broad module split or new runtime framework. Failed tile-kernel variants remain outside the repository.

## Leaf validation
- Sampling: 96 eight-rank cases across B1–B4; mixed temperatures, seed wrap and consecutive graph replay; output and seed bit-exact against old code and across ranks.
- FP4: 630 cases covering dimensions 32–512, row counts 1–192, both scaling modes and in-place operation; bit-exact against old code.
- CPU lifecycle: 204/204 repository tests pass; fault injection includes partial construction, repeated close and cleanup failures. Test doubles validate Python ownership/calls, NOT native driver failure semantics.

| B | old full sampler ms | new ms | saved us |
|---|---:|---:|---:|
| 1 | 0.610146 | 0.597804 | 12.34 |
| 2 | 0.686109 | 0.645836 | 40.27 |
| 3 | 0.800665 | 0.731280 | 69.38 |
| 4 | 1.000140 | 0.907196 | 92.94 |

FP4 example (dim128, block16/e4): rows6 11.024→6.279us; rows12 11.259→8.081us; rows18 11.550→8.639us. Rows24 and above retain the old launch shape; observed near-baseline timing, not a claim of universal speedup.

## Whole-engine ABBA
- Four fresh independent engine loads: old_a → new_a → new_b → old_b. Old source and 44 native binaries were copied before integration, not symlinked.
- Both use temperature=1, top_p=.95, same environment and prompts. Each load: one warmup per context, then three measured requests per context, alternating order; 1024 output tokens each. 24 measured requests total (6/version/context), 8 warmups excluded.
- Actual input tokens: 36868 / 131076. All measured decode batch histograms are B1.
- Metric is complete `model_step_host_ms` per-request mean, not a kernel-only number. Each pair has only one independent load per version; loading/system drift remains significant.

| Nominal context | Old model ms | New model ms | delta ms | old wall ms | new wall ms |
|---|---:|---:|---:|---:|---:|
| 36864 | 34.2223 | 33.5066 | -0.7157 | 34.8048 | 34.0648 |
| 131072 | 34.5092 | 33.8149 | -0.6942 | 35.1035 | 34.4006 |

| Context | paired delta a / b ms | output TPS old / new | MTP accepted/step old / new |
|---|---|---|---|
| 36864 | -1.1364 / -0.2950 | 80.032 / 76.241 | 1.7817 / 1.5955 |
| 131072 | -1.1704 / -0.2181 | 75.240 / 77.717 | 1.6367 / 1.6705 |

Both load-pair comparisons favor the candidate step time, but the measured aggregate ~0.7ms is not a deterministic per-step guarantee. 36k output TPS is lower and MTP acceptance differs: sampling trajectories are not matched, so this experiment does not establish the cause of the TPS change or a general throughput improvement. No isolated per-change whole-engine attribution; no per-step P95 collection; B2–B4 end-to-end performance sweep not performed.

## Fresh restart / resource checks
- Final integrated source freshly loaded after ABBA; B1–B4 graph capture completed on eight ranks. Mixed temp0/temp1 requests observed B4=191 steps and B3/B2/B1 transitions; follow-up request reuse passed.
- Health HTTP200 with expected model; empty messages HTTP400 invalid_request_error.
- All five loads report decode_tensor_bytes=5,312,451,872 per rank: no measured persistent tensor reduction. Final post-capture free memory ranges 182,411,264–602,779,648 bytes; headroom is small and variable, no capacity increase claimed.
- Cleanup fault injection is CPU-only. Real native resource destroy failures have not been injected; native destroy errors are attempted once and do not prove successful release.

## Reproduction / rollback
- Evidence, full requests/responses, ABBA controller, engine logs, leaf variants and raw timings: `/data/decode_oct14_opt/`.
- Summary: `acceptance/summary.json`; final smoke: `acceptance/smoke.log` and `smoke_*.json`; API validation: `acceptance/api_validation.json`.
- Leaf scripts: `finish_parallel/bench.py`, `fp4_dispatch/bench.py`. They require exclusive devices and point to experiment artifacts; do not run alongside the service.
- CPU regression: `python3 -B scripts/test_decode_resource_lifecycle.py /tmp/resource_lifecycle.json`.
- Rebuild matching native libraries with `bash ops/decode/build.sh ops/decode/sampling.cpp ops/decode/libdecode_sampling.so` and analogous attention.cpp/libdecode_attention.so, using the saved CANN environment; restart engine.
- Rollback to 6caae72 requires matching source AND the two native libraries (rebuild or use `acceptance/baseline_6caae72` copies). External ABI is unchanged, but rolling back only source leaves the optimized implementation loaded.


## Follow-up: same-prompt MTP guardrail

User acceptance priority: full decode step time measures implementation efficiency; MTP is a same-prompt regression guardrail, not a required improvement.

- Fixed follow-up new/old/old/new: 4 fresh loads, one warmup plus six measured requests/load, identical 36,868-token input payload, temperature=1, 1024 output tokens. Request payload hashes also match the first experiment.
- Follow-up MTP accepted/step: old 1.71358, new 1.78392 (+4.10%), 12 requests/version. Earlier MTP decrease did not reproduce.
- Both experiments pooled (18 requests/version): old 1.73628, new 1.72111 (-0.87%). No persistent large MTP decrease observed for this prompt; not proof of universal equivalence.
- IMPORTANT correction to strength of performance claim above: follow-up step time old 33.31819ms, new 34.19427ms (+0.87608ms). First new load averaged 35.00185ms, second 33.38669ms. All measured slow samples retained; cause of drift unproven. Pooled step time old 33.61956ms, new 33.96504ms (+0.34548ms). Thus initial ~2% full-model improvement is not reproducibly established by the combined evidence; leaf gains remain independently measured.
- Original 131k results unchanged; this follow-up focused on the 36k MTP concern.
- Raw follow-up: `/data/decode_oct14_opt/mtp_recheck/{design.json,summary.json,controller.log}`. Restored engine health HTTP 200; no implementation change during this follow-up.
