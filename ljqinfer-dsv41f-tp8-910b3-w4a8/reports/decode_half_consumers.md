# Decode FP16 projection consumers

## Implementation
Keep official CANN W4A8 grouped matmul unchanged. Target-layer activation and combine consume its FP16 output directly, round FP16 -> FP32 -> BF16 (RINT) inside the consumer, then use the original arithmetic. This removes the two standalone output casts per target MoE layer and their intermediate BF16 buffers. This is fusion of our consumer boundary, not replacement of the vendor GEMM. Draft projections retain the BF16 output contract. No sampling-policy changes or runtime experiment flag.

## Full-engine evidence
Baseline: 5b36621. Same-process A/B, same prompt per context, temperature=1, unchanged production capture/sampling. ABBA/BAAB: 20 runs total, including one A/B warmup pair per context; four measured runs per label/context. Eight ranks each produced 20 complete, idle-after records without step errors.

| Context | A steady mean ms | B steady mean ms | A run-median average ms | B run-median average ms | A tokens/step | B tokens/step |
|---|---:|---:|---:|---:|---:|---:|
| 36k | 33.073970 | 32.831622 | 32.746413 | 32.490514 | 2.647671 | 2.747224 |
| 131k | 33.856279 | 33.384631 | 33.458573 | 33.186854 | 2.595566 | 2.681286 |

Steady means omit the first eight steps of each measured request and average request means equally. Including every step: 36k 33.078903 -> 32.832976 ms; 131k 33.860237 -> 33.406399 ms. The observed benefit is small (steady mean 0.242/0.472 ms, median-based about 0.26/0.27 ms), not a multi-ms improvement. MTP did not show a material decline; stochastic output variation is not attributed to this optimization.

A/B shared the same engine/model storage; candidate graphs bypassed the casts but retained baseline intermediate allocations. Final integration also removes those unused allocations. Its functional acceptance is below; independent-restart performance of the final memory layout has not been separately established. Do not claim a guaranteed speedup for all prompts or hardware, or full-engine bitwise output equivalence.

## Correctness and integration
- Production consumer regression: 28 cases, B1..4, random inputs plus all 63,488 finite FP16 bit patterns, eager/side-stream/graph replay. Previous native consumer after explicit BF16 cast is bit-exact. CPU combine reference is exact; CPU activation reference uses rtol=0.008, atol=1e-6 for exp/div differences. NaN/Inf inputs are not covered by the finite-pattern claim.
- Official GMM chain with final interface: up/down geometries each passed 16 baseline repetitions and graph exact checks; BF16 draft-compatible output remains supported. Synthetic chain is not an end-to-end semantic proof.
- Final integrated engine: arithmetic 323, Chinese answer Paris, JSON answer 42 passed using official chat encoding and localhost generation RPC. Concurrent 2/3/4-request runs produced non-error streams, observed actual decode widths 1/2/3/4, and each long request generated 384 tokens.
- Five existing CPU contract tests passed. Python syntax and git diff --check passed.
- Development backend is running; 8001 frontend health is OK. Authenticated frontend generation was not exercised; semantic tests used localhost backend RPC, without bypassing frontend authentication.

## Evidence / reproduction
Full raw evidence: /data/workspace/decode_half_consumers/
- engine_ab/{cases.json,worker.py,core.py,variants.py,results/,summary.json}
- integration/regression.log; pre_integration/ retains original sources/library
- production_chain.py and production_chain_{up,down}/
- service_acceptance.json, service_acceptance.log, production_engine.log
- reports/decode_half_consumers.json includes summaries and evidence hashes.

With engine stopped and the usual CANN environment:
`PYTHONPATH=. python scripts/test_decode_half_consumers.py --baseline-lib /data/workspace/decode_half_consumers/pre_integration/libdecode_norm.so`

Rebuild ops/decode/libdecode_norm.so together with Python changes: the two target consumer input ABIs now take FP16, not BF16. Do not mix old/new shared libraries.
