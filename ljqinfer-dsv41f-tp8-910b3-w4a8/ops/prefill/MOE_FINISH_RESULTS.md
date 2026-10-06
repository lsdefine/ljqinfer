# MoE finish fusion: production validation

Fuses shared FP32 -> BF16 -> FP32 rounding, routed FP32 addition, and final BF16 cast. Existing libhc kernel; tensor-owning queued wrapper preserves current-stream lifetime. No orchestration or communication changes. residual.moe_add adds a four-line fast-path branch; fallback retained.

## Fresh-process production benchmark

Both methods use committed host Engram optimization a157cf4. Baseline is the independent original Torch expression. Alternate AB/BA for 12 pairs per size; discard first two pairs. Time synchronized whole forward/sequence, report maximum rank wall time; no per-layer synchronization. Fresh slot per run.

| Tokens | Reference median (s) | Production median (s) | Production tokens/s |
|---:|---:|---:|---:|
| 127 | 0.204288 | 0.204251 | 621.8 |
| 8192 | 1.022288 | 1.019720 | 8033.6 |
| 36864 | 4.545048 | 4.486632 | 8216.4 |

36864 = 4 x 8192 + 4096 with continuous history. Production 8192 range: 0.998608–1.035668 s. **Subsecond is NOT established**: minimum is not the acceptance metric. Single-chunk median gain is only 2.568 ms in the fresh process; do not substitute the earlier probe-wrapper result (1.028060 -> 1.002853 s). 36k median improves 58.416 ms; 9/10 paired runs faster. Short-input result essentially unchanged.

## Correctness and assets

- Production: all eight ranks, five state cases (1, 127, 1024, 8192, continuous 36864) bitwise equal; 72 timings/rank, zero allocator retries; subprocess returncode 0.
- Existing-kernel real-input leaf: 20 layers/rank exact on eight ranks; rank0 8192 leaf 1.034930 -> 0.376050 ms. Leaf improvement is not whole-model improvement.
- Independent production wrapper: 12 cases exact (empty, short, 8192, noncontiguous, side-stream, fallback), two invalid inputs rejected.
- Run independent test: `PYTHONPATH=. python scripts/test_prefill_moe_finish.py` using the configured Ascend environment.
- Raw rank records: `moe_finish_results.json`. Original scripts and logs: `/data/prefill_subsecond_round2/` on A2-3 (`moe_finish_production_ab.py`, `.log`, `.exit.json`; successful leaf log is `moe_finish_production_leaf_retry.log`, not the initial failed import log).
- A2-1/A2-2 untouched; A2-3 server remains stopped for development. No FP8, cached output, CED, or serving-speed claim.
