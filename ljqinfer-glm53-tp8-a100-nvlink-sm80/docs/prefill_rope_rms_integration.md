# Production prefill RoPE/RMS integration

Baseline: 58dd597. Tested on node09 TP8 SM80, 2026-10-02.

- Engine explicitly prepares original FP64 RoPE angles once per prefill chunk;
  shared workspace scope invalidates on exit, including exceptions.
- Q and index Q rotation/packing write fixed contiguous output buffers.
  MLA KV and index K rotation/packing write the paged pools directly.
- FP16 RMS uses the existing FP32 mean reduction order without materializing
  the full squared matrix for widths 512/2048/6144 and rows >=16.
  Short inputs retain the original mean path; rsqrt/apply arithmetic is unchanged.
- First 12288-token chunk uses measured pair split 6648/5640. Other lengths
  and later chunks retain the previous equal/padded split.
- Existing MLA kernels, MoE, decode math, graph policy and 144K limit unchanged.

## Full engine A/B
Same deterministic 12288 input IDs, history=0, full 78 layers, Engine.prefill.
Two warmups, three clean samples; per-sample rank maximum then median.
No HTTP/tokenization; not a live-service throughput claim.

| Metric | 58dd597 | Integrated |
|---|---:|---:|
| Wall seconds | 2.505017008 | 2.415543750 |
| Tokens/s | 4905.356 | 5087.053 |

Latency -3.572%; throughput +3.704%. All eight ranks' logits bitwise equal.
This remains above the 2-second full-model target.

## Regressions
- tests/test_prefill_elementwise.py: 48 RMS cases, row counts 1/6/15/16/17/127/128/129,
  contiguous/strided inputs; direct Q/KV/index writes at positions 0/8192/135168,
  invalid-page guards, changed same-address positions and exception invalidation.
- tests/test_sparse_pair.py: all eight ranks pass six eager/dynamic-graph cases.
- tests/test_glm53_sparse_binding.py: all eight ranks pass real-weight full/shared
  prefill/decode reference and changed-input decode graph checks.

Run from repository root with PYTHONPATH=.:
```
python tests/test_prefill_elementwise.py
python -m torch.distributed.run --standalone --nproc-per-node=8 tests/test_sparse_pair.py --output /tmp/pair
REPORT=/tmp/binding python -m torch.distributed.run --standalone --nproc-per-node=8 tests/test_glm53_sparse_binding.py
```

Detailed evidence: /mnt/data2/kw/glm53_int4_tp8/service_audit/
engine_rope_rms_production_summary.json, engine_rope_rms_{baseline,candidate}.rank*.jsonl,
prefill_production_{pair,binding}.rank*.json. No full 144K run or HTTP restart in this change.

## V projection before pair output exchange

Project V before paired prefill exchange; full-model 2.41889 -> 2.40192 s

Gather per-layer peer V weights at initialization; sequential layers share
preallocated send/receive buffers. BMM writes token-major views directly,
halving pair output payload (512 -> 256). First 12288-token chunk only.

Measured A100 TP8, history=0, 12288 tokens, no HTTP:
- Candidate attention: 1215.77 -> 1199.86 ms (21 full + 57 shared calls).
- Engine candidate: 2.418889862 -> 2.401917676 s (-16.972 ms);
  5080.016 -> 5115.912 tok/s (+0.7066%), 2 warmups + 5 alternating
  rank-max wall samples. All 8 ranks: logits + 6 features bitwise equal.
- Formal binding regression: 1215.38 -> 1197.26 ms, 3 alternating samples;
  all 8 ranks full/shared output, IDs, MLA/index caches bitwise equal.

Keep decode, other chunks, MLA/MoE kernels and 144K capacity unchanged.
Full 144K sequence untested; attention <1s and full-model <2s not reached.
Future small improvements stay on the operator bench until gains accumulate.

Evidence: service_audit/v_direct_optimization_summary.json, engine_v_direct_memorysafe.py, v_projected_integrated.rank*.json. Regression: tests/test_prefill_projected_pair.py (torchrun, nproc=8).
