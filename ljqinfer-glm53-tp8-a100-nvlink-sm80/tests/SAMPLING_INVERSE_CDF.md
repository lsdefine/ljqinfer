# Batched nucleus inverse CDF (2026-10-14)

T=0 remains argmax, with lowest-ID ties and no RNG consumption. T=1 uses
FP32 ascending sorted probabilities, drops CDF <= 1-p (always keeps final
entry), draws one PyTorch uniform per target position, and searches the
remaining CDF. No logits are mutated. Rank 0 selects, all TP ranks receive
the same IDs. CUDA finish fuses two upper-bound searches and ID mapping;
CUDA graph RNG progression is tested, but production sampling remains eager.

Independent references: installed HF TopPLogitsWarper support, FP64 truncated
categorical quantiles, empirical frequencies. Old implementation used ONLY
for timing, not correctness. Floating-point CDF rounding remains; no bitwise
random-trajectory equivalence claimed. Temperature and top-p API policy unchanged.
No repetition penalty added. Long-task The failure is not claimed fixed.

Commands (from repo, Python=/mnt/data/kw/anaconda3/bin/python):
- CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 python -m pytest -q tests/test_sampling.py tests/test_temperature.py tests/test_generation_trace.py tests/test_service_template_contract.py
  109 passed, 1 skipped.
- CUDA_VISIBLE_DEVICES=0 SAMPLING_DEVICE=cuda OMP_NUM_THREADS=2 python -m pytest -q tests/test_sampling.py
  33 passed, 3 skipped (bad-distribution tests skipped on CUDA to avoid poisoned context).
- OMP_NUM_THREADS=1 python -m torch.distributed.run --standalone --nproc-per-node=8 tests/check_sampling_tp8.py
  all five pure/mixed temperature cases passed, B1/B2/B4 Q8 V154880.
- python -m pytest -q tests/test_openai_protocol.py tests/test_http_cancellation.py tests/test_chat_completions_route.py tests/test_api_repair.py
  22 passed.

Busy A100 microbenchmark, no full model loaded in test process. Real production
was concurrent. Each function 3 warmups, 3 alternating trials of 10 calls,
CUDA events around complete pick_tokens, local process (no TP broadcast):
B/Q/V | greedy median ms | old median ms | new median ms
1/8/154880 | .0168 | 1.9668 | 1.4859
2/8/154880 | .0194 | 4.3899 | 2.0466
4/8/154880 | .0264 | 9.1648 | 3.4144
This is NOT idle-GPU/end-to-end performance, nor near-argmax overhead.

Service trace: >=8192 observed raw output IDs ->
/mnt/data/kw/glm53_long_generations/<timestamp>_<uuid>.json, files 0600,
new directory 0700. Input token IDs, effective plan, raw output IDs/text,
terminal event, available decode stats and longest identical-token run.
No HTTP headers. Effective plan includes messages; input IDs are also retained.
This is not the full HTTP body.
Decoder-finally saves even on error/close once threshold was observed; already
cancelled/disconnected requests can only capture tokens consumed so far.
No per-step logits/RNG snapshots; this captures reproduction context, not root
cause by itself. Contains private prompt/output data: do not publish logs.
No automatic deletion/rotation; operator should monitor disk usage.
Normal/error/close, threshold and EOS retention tested with real file writes.

## Idle and real-model acceptance after authorized service stop

Idle A100, Q8/V154880, local pick_tokens without TP broadcast; CUDA events,
5 warmups then five trials of 30 calls (median):
B | greedy ms | old (performance only) ms | committed sampler ms
1 | .01833 | .92587 | .69144
2 | .01836 | 1.85450 | .94491
4 | .02560 | 3.71285 | 1.46988

Fresh TP8 engine running committed 57a78bf: eight short real requests completed.
T0 and T1 both answer 17*23=391, seasonal explanation is coherent in both.
For this short explanation alone, wall step T0=44.89ms, T1=45.52ms;
random trajectories/acceptance differ, so not a general throughput claim.
Four mixed-temperature requests demonstrably boarded concurrently (active=4),
with coherent responses. Two hit their deliberately small 384-token limit;
these are not complete-answer acceptance tests.

A ninth, real T1 non-thinking textbook request generated exactly 8192 output
IDs: finish_reason=length, decode_seconds=117.265871, decode_tps=69.858,
wall_ms_per_step=46.195409. Auto-trace exists, parses as JSON, mode 0600,
86 input IDs and 8192 output IDs, effective plan/messages, terminal and metrics.
Longest identical-token run=4; output starts with a Python introduction and
ends mid-example in Chapter 8. This verifies real trace capture, NOT rare
long-thinking The-loop reproduction or its resolution.
Trace: /mnt/data/kw/glm53_long_generations/1791974449469243212_09b40d71033b442bacfd8a9d8382c917.json
Raw local acceptance evidence: /mnt/data/kw/glm53_sampling_acceptance/.
Do not publish private prompt/output data. Public HTTP frontend remained
stopped during development; this used the ServiceLayer and private TP8 engine.

Additional block-scan optimization was NOT merged: FP64 accumulation matches
the high-precision prefix more closely, but uniform V154880,p=.5 changes one
boundary token versus installed HF FP32 cumsum (.5 vs .5000000596). No test
standard was relaxed to accommodate this experimental optimization.
