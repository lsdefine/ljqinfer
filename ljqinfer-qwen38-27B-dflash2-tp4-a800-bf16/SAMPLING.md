# Top-k/nucleus correction

Positive temperatures now use local top20 candidate triples (logit, id, Exp(1)), one all-gather, global top20 then p=.95. T0 reduction and MTP acceptance unchanged. No penalties. CPU regression: 107 passed, 14 skipped, 86 subtests; CUDA 12000 draws per temperature distribution and tail-exclusion checks passed. Sampler is outside verify graphs. Historical results below refer to earlier full-softmax revisions.

## Live truncation acceptance

4250c79 restarted on common service. Public :18084 and engine :62001 healthy;
four ranks alive, fatal scans clean. Three public T1 security articles finished
with stop. Severe mixed-language symptom not observed, but article_1 contains
"即使某一段落被攻破" (expected 网段/区域): residual lexical error, not a factual
accuracy issue. Do not claim the wrong-word problem is fully resolved.

Whole decode timing: same 84-token prompt, B1, 1024 output tokens, no foreign
submissions in measured intervals. Two drain/warmup runs excluded; no profiler.
T0 24.006/24.117 ms (mean24.0615); T1 24.367/24.386 ms (mean24.3765).
Difference +0.315 ms/step (+1.31%), not an isolated sampler cost measurement.
Facade resumed and health passed. Evidence: /tmp/cuda140_topk_live,
/tmp/cuda140_topk_step_81718860, /tmp/cuda140_topk_live_check.log.

# Historical CUDA temperature sampling (2026-09-29)

Baseline: b5d55e5 (both penalty versions reverted). This change adds only temperature sampling. No repetition penalties or thinking-length bias.

HTTP temperature defaults to 1; explicit 0 keeps greedy decoding. Internal model helpers retain greedy defaults. Request temperature propagates through remote strategy, control plane, initial prefill selection, fixed batches, dynamic boarding, and verification. Different request temperatures may share a batch.

Positive temperatures use split Triton Gumbel-max (adapted from ljqinfer_dsv41f_tp8); zero-only selections keep the original CUDA reduction. Two local kernel launches replace local max, with the same candidate all-gather and CPU materialization as before. No full vocabulary communication. Independent rank seeds advance on device. Sampler runs outside verify graphs; cached scratch belongs to the serialized model executor and must not be used concurrently on multiple streams. Drafting remains unchanged; sampled target tokens are accepted by longest matching draft prefix.

## Verified before service restart
- CPU baseline: 99 passed, 14 skipped, 86 subtests.
- Candidate: 103 passed, 14 skipped, 86 subtests (including mixed-temperature boarding).
- CUDA simulated two-shard distribution: 51200 draws for each T=0,.5,1,2, max absolute frequency error <=.00254. Mixed zero rows, nonmutation and fresh randomness on fixed-temperature graph replay passed.
- Local selection microbenchmark on A800, FP32 [rows,37984], rows=1/8/32: original ~38.5-39.0us, sampled ~56.2-57.6us (median host-loop wall time). About 17-19us added. This is NOT whole decode step latency or an isolated device-kernel cost; other GPU workloads were running. Does not establish model throughput/acceptance rate.

Reproduce from repository root with vllm-env Python, PYTHONPATH=.; optional GPU checks: scripts/check_temperature_cuda.py and scripts/bench_temperature_cuda.py. CPU suite uses CUDA_VISIBLE_DEVICES='' PYTHONPATH=/data/ljq/cuda_test_deps:.

## Live service acceptance after authorized restart

Implementation 448ce1c was loaded by a fresh TP4 supervisor; public API :18084 and engine :62001 are healthy. Both old penalty versions remain absent.

Seven HTTP non-stream requests generated complete merge_sort programs: three T0 thinking, three T1 thinking, one T1 non-thinking. All seven finished with stop; every program passed its generated asserts plus 1000 fixed-seed random cases checking sorted output, input nonmutation and new-list identity. This is one coding task, not a broad quality benchmark.

| Case | Tokens | Batch at prefill | Logged ms/step | Accepted draft tokens/step | API decode tok/s |
|---|---:|---:|---:|---:|---:|
| 0_t0_thinkTrue | 1253 | 3 | 45.976 | 4.614 | 122.160 |
| 1_t1_thinkTrue | 954 | 3 | 43.944 | 4.509 | 125.412 |
| 2_t1_thinkTrue | 1921 | 2 | 27.331 | 2.288 | 120.081 |
| 3_t0_thinkTrue | 858 | 1 | 384.513 | 4.713 | 14.874 |
| 4_t0_thinkTrue | 1253 | 3 | 46.028 | 4.614 | 121.988 |
| 5_t1_thinkTrue | 1718 | 3 | 72.334 | 3.380 | 60.574 |
| 6_t1_thinkFalse | 679 | 2 | 157.532 | 5.220 | 39.507 |

These are live per-request metrics, NOT a controlled T0/T1 comparison. Concurrent ~65k-token requests and dynamic boarding were present. Batch at prefill is not a constant batch size during decoding. Slow cases are retained above; no claim of stable >100 tok/s or unchanged whole-step cost is established. Accepted draft tokens/step is not an acceptance percentage.

T0 lengths were 1253/858/1253. Repeatability has not been established under identical execution conditions; the cause of the differing output is undiagnosed and must not be asserted to be harmless numerical drift without an A-A baseline.

Evidence: /tmp/cuda140_live_f5f0e2b7 (responses/programs/summaries), /tmp/cuda140_live_acceptance_v2.log (DONE), /tmp/qwen_tp4_serve_rank0.log (engine metrics); local copy cuda140_live_evidence. Verification found seven nonempty successful summaries and matched each engine RPC output count. Rank0 log scan contained no Traceback/RuntimeError/CUDA error/ERROR markers.

Remaining performance gate: repeat controlled same-workload T0/T1 steady-state runs without concurrent boarding, plus a controlled T0 repeatability baseline. This live run does not close those gates.

## Controlled whole-decode step comparison

Same running TP4 instance (448ce1c implementation), same 84-token prompt, batch=1, 1024 output tokens each, actual service CUDA graph path. Public facade was temporarily paused with an independent timed resume safeguard; first drain/warmup and second warmup excluded. Eight measured runs used alternating order 0,1,1,0,0,1,1,0; each had batch=1 and no foreign submission in its log interval. No model edits or phase profiler.

| Temperature | Four measured whole-decode ms/step | Mean ms/step |
|---|---|---|
| T0 | 24.077 / 24.059 / 24.050 / 24.026 | 24.053 |
| T1 | 24.209 / 24.192 / 24.003 / 24.047 | 24.11275 |

Mean difference +0.05975 ms/step (+0.2484%). In this controlled B1 short-context workload, whole-step time is essentially unchanged; do not interpret this small difference as precise isolated sampler overhead or extrapolate to all contexts/batches. Four T0 runs each took 306 decode steps; four T1 runs took 402/397/382/370 steps.

T0 measured token-sequence distinct count: 1 across four 1024-token outputs. This establishes repeatability for this test only, not the earlier dynamically batched workload.

Evidence: /tmp/cuda140_step_1c1c9f7f/{0..9}.json and results.json; local cuda140_step_evidence. Harness /tmp/cuda140_step_isolated.py. Test completed and facade resumed; public health returned status=ok.
