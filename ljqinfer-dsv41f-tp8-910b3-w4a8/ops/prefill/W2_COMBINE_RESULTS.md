# W2 FP16 rounding + expert combine acceptance

Baseline: `697dda7`. A2-3 only, TP8 eager encoder first 20 layers.
No FP8, CED, prefix-result cache, service traffic, TP communication or scheduler changes.

## Implementation

- Share one AscendC combine template for BF16 and FP16 input. The FP16 variant preserves FP16 -> FP32 -> BF16 RINT -> FP32, then the original sorted-bank six-term sum.
- `grouped_linear_combine` owns the W2 temporary and returns the original FP32 rank-local result; TP reduction remains at its previous boundary.
- A three-line shape branch keeps the original composition below 4096 tokens. Two all-fused fresh-process tests showed small-input regressions despite faster leaf timing; do not infer end-to-end wins from the leaf alone.
- At 8192, eliminate a 480 MiB BF16 intermediate. Measured leaf peak allocation decreases by 503316992 bytes.

## Final shape-dispatched acceptance

Fresh process. Alternating AB/BA, 12 pairs per length, discard first 2 pairs. Each sample is max wall time over all 8 ranks, then median across repetitions.
Snapshots are outside timing. Every timed call has branch counters and zero allocation retries.

| Tokens | Baseline seconds | Candidate seconds | Candidate tok/s | Median reduction |
|---:|---:|---:|---:|---:|
| 127 | 0.212842 | 0.208688 | 608.6 | 1.95% |
| 8192 | 1.013293 | 0.991782 | 8259.9 | 2.12% |
| 36864 | 4.491429 | 4.426610 | 8327.8 | 1.44% |

8192 candidate range: [0.9703707862645388, 1.0181830078363419] seconds. Median below one second, NOT a guarantee that every run is below one second.
Short-input improvement is not attributed to new computation: it uses the original composition and remains noise-sensitive.

## Repeatability and correctness

- Before the short-shape branch, independent process ABs measured 8192 medians 1.015708 -> 0.997794 and 1.012722 -> 1.006167 seconds. Keep this variability visible; do not report stable subsecond performance.
- All three TP8 processes passed five snapshot cases on every rank: 1, 127, 1024, 8192 and consecutive [8192,8192,8192,8192,4096].
- Independent regression passed finite FP16 bit-pattern coverage, fixed-order CPU oracle, empty/small/full shapes, strided input, non-default stream, and four invalid-input rejections.
- Production leaf at 8192 (FP16 -> BF16 -> combine vs fused): [1.591709442436695, 0.8421903476119041] ms; this excludes GMM, not end-to-end encoder timing.

## Reproduction

Source CANN environment and use `/data/apps/ascend-cann9/bin/python`.
Build the shared kernel explicitly:
```bash
bash ops/decode/build.sh ops/kernels/hc_residual.cpp ops/kernels/libhc.so
PYTHONPATH=. /data/apps/ascend-cann9/bin/python scripts/test_prefill_combine_half.py
```
Fresh-process native import rebuilds the torch registration extension.
Acceptance launch/environment: `/data/prefill_combine_round3/launch_combine_shape_ab.py`; actual standalone torchrun workload: `combine_shape_ab.py`.
Raw evidence: same directory, `combine_shape_ab_rank{0..7}.json`, `final_acceptance_summary.json`, `production_leaf.json`, `production_extended.json`, `regression.log`.
Earlier all-fused comparisons: `combine_ab_rank{0..7}.json`, `combine_confirm_rank{0..7}.json`.
These acceptance scripts load the full resident model but time only the 20-layer eager prefill encoder.

## Scope and unvalidated items

HTTP generation/full-model TTFT, decode generation, full repository tests, A2-1/A2-2 deployment are not validated here. No push or cross-machine deployment.
