# Decode context scaling measurement

Measured on A2-3, code 6bc3c31. No inference implementation changes.

## End-to-end B1, 1024 output tokens, three rounds

| Input tokens | Model step median ms | Range ms | Wall step median ms | Decode steps/request |
|---:|---:|---:|---:|---|
| 90 | 31.5435 | 31.3870–32.0471 | 32.0575 | 368,360,352 |
| 8197 | 32.6486 | 32.1289–32.6517 | 33.1518 | 354,357,368 |
| 36868 | 33.1591 | 33.0727–33.3640 | 33.8260 | 347,352,354 |
| 131076 | 32.7451 | 32.6443–33.0044 | 33.2568 | 340,349,340 |

All 12 samples completed 1024 tokens with actual B1 decode. Long-context steady decode did not reproduce a +5ms increment. 36k vs 128k is non-monotonic; three rounds and changing generated tokens/routing do not isolate a pure length effect.

## Synthetic graph-replay diagnostic

| Context | ratio2 index us | ratio1 index us | 3x + 5x index ms (estimate) | ratio2 scattered CSA us | ratio1 scattered CSA us |
|---:|---:|---:|---:|---:|---:|
| 512 | 30.275 | 31.957 | 0.2506 | 82.632 | 127.409 |
| 8192 | 41.707 | 52.840 | 0.3893 | 128.777 | 127.439 |
| 36864 | 64.394 | 81.985 | 0.6031 | 129.716 | 128.049 |
| 131072 | 104.456 | 150.855 | 1.0676 | 132.812 | 131.267 |

Eight index sources/step: three ratio2 and five ratio1. 8k to 128k weighted leaf increment is 0.6783ms; this is score+reduction+selection, not topk alone. It is NOT a measured full-step attribution.

CSA at 8k/36k/128k has fixed 512 selected keys plus a fixed local window. Contiguous IDs: ratio2 approximately 120.7us, ratio1 approximately 118.7us, essentially flat. Scattered IDs: 8k to 128k increases approximately 4us per layer (38-layer weighted estimate 0.1492ms); fixed arithmetic does not guarantee fixed memory locality. At 512-token prefix, ratio2 has only 256 selected compressed keys, so comparing against near-empty prefixes also changes actual attention work.

## Limits / reproducibility

- No old-version A/B in this experiment
- Leaf uses synthetic tensors and repeated graph replay; not an additive production profile
- VendorIndexer includes score, reduction and top512; topk alone not isolated
- CSA synthetic IDs identical across six query rows, fixed per case; cache locality differs from real workloads
- Short context grows by 1024 tokens during output; context is initial prefix length

Raw harness, prompts, outputs and logs: /data/decode_fusion_dev/{context_bench.py,context_leaf.py,context_prompts.json,context_decode.json,context_leaf.json}. API helper api_ab.py and runtime_env.json remain there; do not run the standalone NPU leaf benchmark alongside the full eight-card server.

The 34–35ms of general service was not attributed by this test: concurrent batches, prompt/output differences, host work and scheduling require matching production samples, not subtraction across unrelated requests.
