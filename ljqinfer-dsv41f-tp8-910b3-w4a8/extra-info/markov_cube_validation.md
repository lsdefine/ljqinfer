# Markov projection Cube replacement

Baseline: 121e59b. BF16 [6*B,256] x [16160,256] -> FP32. Shape, collective sequence and chain semantics unchanged; startup-owned tiling, no runtime large allocation.

Random and real rank0 weights passed CPU FP32/original Matmul comparison and tail guards. Production wrapper: 100 calls per graph replay, six alternating samples per implementation.

| M | Baseline us | Cube us | CPU max abs error |
|---:|---:|---:|---:|
| 6 | 35.8467 | 10.0095 | 1.52587891e-05 |
| 12 | 35.9803 | 10.3922 | 1.52587891e-05 |
| 18 | 36.5959 | 10.7810 | 2.28881836e-05 |
| 24 | 36.3877 | 11.5592 | 2.28881836e-05 |

Five-call B1 savings estimated from isolated tests: 0.1292 ms; NOT measured full-engine speedup. No full-engine or TP8 acceptance-rate test for this change. Profiled ~300us Matmul versus ~36us isolated baseline discrepancy unresolved. Draft 11.904ms contains cross-rank waits.

Reproduction artifacts: /tmp/prefill_space_ab/markov_cube_v1/{test.py,test_real.py,test_wrapper.py,test_result.json,real_result.json,wrapper_result.json} and logs.
