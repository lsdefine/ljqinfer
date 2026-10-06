# W13 rounding / SwiGLU fusion

## Scope and decision

Fuse the FP16-to-BF16 rounding of W13 output into SwiGLU using one shared template. Keep BF16 rounding, clipping and arithmetic order; keep dynamic_quant and TP communication unchanged. Retained as an exact leaf-level optimization, NOT an established whole-prefill speedup. No claim of stable sub-second performance or proven absence of regression.

## Corrected production-shape leaf measurements

Input [rows,576], activation [rows,288], INT8 output padded to 320 (not 512). Timings exclude grouped GEMM. The earlier pad-to-512 chain benchmark is superseded here.

| Rows | Path | Baseline ms | Candidate ms | Reduction |
|---:|---|---:|---:|---:|
| 762 | cast + activation | 0.126214 | 0.054398 | 56.90% |
| 762 | activation + quant + pad | 0.258051 | 0.192412 | 25.44% |
| 24576 | cast + activation | 0.108720 | 0.080611 | 25.85% |
| 24576 | activation + quant + pad | 0.250950 | 0.190301 | 24.17% |
| 49152 | cast + activation | 0.206976 | 0.155501 | 24.87% |
| 49152 | activation + quant + pad | 0.385294 | 0.327545 | 14.99% |

## Whole-prefill AB

A2-3 TP8, first 20 eager layers; paired alternating baseline/candidate in each process. Take slowest rank for each sample, discard first two repetitions, then median. 36864 is consecutive 4.5 chunks. Two independent engine processes.

| Run | Tokens | Baseline s | Candidate s | Paired saving ms | Wins / pairs |
|---|---:|---:|---:|---:|---:|
| activation_ab | 127 | 0.208799 | 0.209942 | 1.424 | 5/10 |
| activation_ab | 8192 | 0.994803 | 0.997512 | 0.527 | 5/10 |
| activation_ab | 36864 | 4.414548 | 4.401506 | 5.811 | 7/10 |
| activation_confirm | 127 | 0.206551 | 0.202595 | 0.964 | 14/24 |
| activation_confirm | 8192 | 0.992416 | 0.991817 | -2.371 | 11/24 |
| activation_confirm | 36864 | 4.409354 | 4.430329 | -6.470 | 3/10 |

8192 and 36864 results change direction between processes. In the confirmation run, 36864 candidate is slower by 20.97 ms comparing medians (~0.48%); paired median is also slower. Whole-prefill benefit is unestablished; regression cannot be ruled out. Do not extrapolate the ~15% leaf-chain reduction to whole-prefill throughput.

## Validation / reproduction

- Both TP8 runs: eight ranks, five state cases bit-exact, candidate path hit checks and zero allocation retries.
- Independent regression: 14 cases (including finite FP16 bit patterns, tail sizes, noncontiguous inputs, nondefault stream), four invalid-input cases rejected. Quantized output and scales checked on nonempty shape cases.
- Regression command: `PYTHONPATH=. /data/apps/ascend-cann9/bin/python scripts/test_prefill_swiglu_half.py`.
- Raw evidence and replay scripts: `/data/prefill_activation_round4/production_leaf320.py`, `activation_ab.py`, `activation_confirm.py`, corresponding launcher scripts, per-rank JSON and exit files.
- Dynamic-quant fusion was rejected: explored formulas did not reproduce INT8 results exactly. Production dynamic_quant was not replaced.
- No HTTP TTFT, full-model generation speed, A2-1/A2-2 deployment or nonfinite-input correctness claim. No changes to service configuration in this round.
