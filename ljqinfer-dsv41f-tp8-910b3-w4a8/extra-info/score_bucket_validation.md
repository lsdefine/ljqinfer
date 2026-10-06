# 8K score-width graph specialization

For max(start)+6 <= 8192, startup captures one additional verify graph per batch using packed score rows of width 8192/ratio. Other requests keep the full-capacity graph. Existing data buffers and workspace are shared; no additional runtime device allocation or synchronization is introduced.
Graph metadata/device graph storage is additional and allocated during startup capture; its full-engine memory cost has NOT been measured. This is not a claim of zero extra startup memory.

## Validation
TP8 synthetic BF16 integer inputs with independent exact CPU reference: B1/r1/start128, B1/r2/start8185, B4/r1/start8186 with mixed starts/inactive slot; missing pages, negative head weights, ties, pending overrides. All eight ranks passed exact score/TopK, buffer guards and alternating full/short captured graphs. Candidate SourcePlan.capture_score_width was exercised, not a duplicated implementation.
Host routing method passed seven boundary/fallback cases including starts 8186/8187 and mixed batches. Full model capture, graph-memory budget, arbitrary BF16 rounding effects of changed HCCL message size, generation acceptance and end-to-end timing remain unverified.

## Unprofiled reduce + TopK microbenchmark
Values below are medians across eight rank medians in microseconds. Synthetic score inputs, five alternating samples, ten graph replays per sample. Not whole-model savings.
```json
[
  {
    "batch": 1,
    "ratio": 1,
    "start": 128,
    "full_reduce_us": 342.4080014228821,
    "short_reduce_us": 110.27299761772156,
    "full_chain_us": 395.4480051994324,
    "short_chain_us": 165.03700017929077
  },
  {
    "batch": 1,
    "ratio": 1,
    "start": 8186,
    "full_reduce_us": 335.33400297164917,
    "short_reduce_us": 109.3030035495758,
    "full_chain_us": 1316.7099952697754,
    "short_chain_us": 1085.2700233459473
  },
  {
    "batch": 1,
    "ratio": 2,
    "start": 8186,
    "full_reduce_us": 240.42699337005615,
    "short_reduce_us": 75.48699975013733,
    "full_chain_us": 953.0580043792725,
    "short_chain_us": 792.0369863510132
  },
  {
    "batch": 4,
    "ratio": 1,
    "start": 8186,
    "full_reduce_us": 911.1479759216309,
    "short_reduce_us": 124.15800094604492,
    "full_chain_us": 1891.769027709961,
    "short_chain_us": 1105.869960784912
  }
]
```

Evidence: /tmp/prefill_space_ab/score_bucket_probe_v1, score_bucket_generation_v1 and score_bucket_candidate_v1/routing_result.json. Production parent: f315169. No full model run performed.
