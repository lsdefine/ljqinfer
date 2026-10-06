# Host Engram optimization acceptance

Baseline: `74da8ec`. A2-3 only, eager encoder first 20 layers, TP8.
No CED, serving, prefix cache, FP8 change or precomputed request results.
Same loaded model, alternating original/production method; max wall time over 8 ranks.
State snapshots are outside timing; first two pairs discarded.

| Run | Tokens | Warm pairs | Original s | New s | New tokens/s |
|---|---:|---:|---:|---:|---:|
| integrated_sequence | 1 | 4 | 0.179706814 | 0.178627248 | 5.6 |
| integrated_sequence | 127 | 4 | 0.203710050 | 0.207908355 | 610.8 |
| integrated_sequence | 1024 | 4 | 0.397444216 | 0.396787904 | 2580.7 |
| integrated_sequence | 8192 | 4 | 1.042296672 | 1.035719758 | 7909.5 |
| integrated_sequence | 36864 | 4 | 4.584804690 | 4.528435819 | 8140.6 |
| integrated_confirm | 127 | 10 | 0.191892009 | 0.189105547 | 671.6 |
| integrated_confirm | 8192 | 10 | 1.040121553 | 1.015962965 | 8063.3 |

36,864 tokens use `[8192,8192,8192,8192,4096]` continuously.
The 127-token slowdown in the first run did not reproduce in the 10-pair confirmation.
Absolute times vary between runs; do not cherry-pick the fastest sample. **Below 1 second is not yet achieved.**

## Implementation and validation

- Model computes only the bound layer/rank hash IDs; explicit raw-token history, no hidden request state.
- Independent CPU operator: contiguous INT8 rows and FP32 group scales to owned BF16 output, NEON nearest-even conversion.
- Four OpenMP workers at >=3072 selected rows; eight workers brought no chunk benefit.
- Query-splitting sparse attention was exact but slower and was rejected.
- Eight ranks: 48 real-table leaf cases each, plus five continuous-state cases and two confirmation cases, all exact.
- All integration/confirmation allocation retries are zero.
- CPU-only independent test: 12 gather cases, 450 hash comparisons, 8 invalid-input rejections.

## Reproduction and scope

CPU test (A2 environment, repository root):

```sh
PYTHONPATH=. /data/apps/ascend-cann9/bin/python scripts/test_prefill_host_engram.py
```

Raw rank reports: `host_engram_results.json`. Resident-worker integration harnesses and commands:
`/data/prefill_subsecond_round2/{baseline_resume12.py,integrated_sequence.py,integrated_confirm.py,command18.json,command19.json}`.
The harnesses require the resident loaded-model context; they are not standalone CLI tests.
Production-source methods were exercised in the resident TP8 model. Fresh service startup, HTTP generation,
full repository tests, A2-1/A2-2 deployment and full-model TTFT are not validated by this change.
