# GLM53 service integration and configuration repair

- Service model: glm-5.3; HTTP :8000; engine loopback :62001.
- Total budget: 144*1024; output budget: 8192. Physical KV: 147520.
- Prefill uses model.config.DEFAULT_PREFILL_CHUNK_TOKENS (12288), not the experimental 128 default.
- Existing MoE and sparse attention kernels are retained; existing paired-prefill binding is wired in.
- Sequential layers reuse scratch. Index scan plans use logical sequence/request bounds, not the entire KV reservation.
- Generation projects only the last prefill logit; DFlash append counts feature rows, not logit rows.
- Per-phase synchronous profiling is opt-in. RPC and OpenAI blocking/streaming preserve draft statistics.
- mtp_accept_rate = matched draft tokens / proposed draft tokens (7 per verify step). This measures proposals matched before output-budget/EOS truncation, not emitted-token fraction.

## Observed validation (2026-10-01, node09)
Same request: 1590 input tokens / 128 output budget, three runs each; first excluded as cold.
Before: TTFT 4.6568s, decode 34.1485 tok/s.
After: TTFT 1.2076s, decode 32.8095 tok/s.
TTFT improved about 74%; decode throughput did NOT improve in this comparison.
After: 82 matched / 315 proposed = 0.260317, 45 verify steps.
Repeated post-change outputs agree; before/after generated outputs differ. Chunk/attention scheduling changed; this is NOT independent numerical correctness certification.
13020-token input crosses the 12288 boundary and returns HTTP200. Streaming ends with DONE and real ljqinfer statistics; protocol tests: 4 passed.
144K allocation/startup and budget guard validated earlier; full 144K generation NOT validated. No claim of overall model-quality validation.

Evidence: /mnt/data2/kw/glm53_int4_tp8/service_audit/{fix_before.json,fix_after.json,fix_long.json,fix_stream_final.txt,service_fix.log}.

## Standard API repair after c5b0860

Reference: GLM52 decode_backend uses explicit per-device capture streams. GLM53 used an implicit global CUDA capture stream. On this A100/PyTorch build, 96 ordinary stream allocations aliased the implicit capture root at indices 31/63/95; a high-priority root had zero aliases. Repeated workspace allocations can therefore collide with the root. Cancellation was the observed context, not the independently established cause.

Changes:
- Verify warmup/capture use the same explicit high-priority root, distinct from ordinary MoE side streams. DFlash/append capture explicitly use their warmup streams. No kernel math changes.
- Load the repository template honoring the existing default thinking-off state. The external template always appended <think> while the parser started in TEXT. This restores configured behavior, not a claim of thinking-enabled quality improvement.
- Complete OpenAI models list/model object, created and owned_by fields; retain legacy extra fields.

Live TP8 validation, no restart between tests:
- 5 CPU protocol/template tests passed.
- 8 rounds: four unequal-budget concurrent requests -> singleton -> disconnect after content -> immediate singleton. All passed: 32 isolated cohort responses, 8 cancellations, 16 singleton responses. Blocking/streaming exact-string output passed.
- 8/8 small reasoning samples correct; blocking/streaming function calls and tool-result roundtrips passed; zero control-tag pollution. 30/30 performance requests ended with DONE. OpenAI SDK models/chat/stream passed. No error log lines.
- 256-token coding workload, two rounds per concurrency: C1/2/4/8 aggregate end-to-end throughput 48.02/60.56/65.87/62.36 tokens/s; TTFT medians 3.43/0.79/0.89/8.42 seconds. Single-request streaming decode estimate median 137.68 tokens/s. Cache/workload effects apply; changed thinking prefix means this is NOT a kernel-speedup comparison with the previous broken API.

Evidence: /mnt/data2/kw/glm53_int4_tp8/service_audit/api_fix/ (service.log, stream_pool_probe.json, lifecycle_results.json, results.json, raw/, perf_aggregate.json, sdk.json, unit_tests.log).

Run lifecycle regression explicitly: API_REPAIR_AUDIT=<output_dir> python tests/test_api_repair.py
Run CPU regression: python -m pytest -q tests/test_openai_protocol.py tests/test_service_template_contract.py

Scope excludes extended API fields, long-context generation, multi-hour soak and external-network connectivity.

Exact performance aggregate:
```json
[
  {
    "concurrency": 1,
    "success": 2,
    "throughput": 48.01580684152384,
    "ttft_median": 3.4304312414024025,
    "latency_median": 5.330027463845909,
    "decode_est_median": 137.675605944901
  },
  {
    "concurrency": 2,
    "success": 4,
    "throughput": 60.559680990984674,
    "ttft_median": 0.7899522178340703,
    "latency_median": 7.801854545483366,
    "decode_est_median": 37.06489920501636
  },
  {
    "concurrency": 4,
    "success": 8,
    "throughput": 65.87214172582986,
    "ttft_median": 0.8906141880434006,
    "latency_median": 14.283260729396716,
    "decode_est_median": 19.344852918927515
  },
  {
    "concurrency": 8,
    "success": 16,
    "throughput": 62.36194920473721,
    "ttft_median": 8.423579533584416,
    "latency_median": 23.180493822786957,
    "decode_est_median": 17.92742351519533
  }
]
```
