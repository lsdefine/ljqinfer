# Decode leaf fusion on A2-3

Baseline: f6bc462. API remains on 8001. No changes deployed to other machines or R archive; no push.

## Boundaries
- hc_finish.{cpp,py}: TP-reduced output -> BF16 rounding -> HC residual expansion. No collective inside. MoE preserves BF16(shared) and BF16(sum).
- kv_prepare.{cpp,py}: RMSNorm -> RoPE -> simulated FP8 QDQ. Preserves intermediate BF16 rounding and signed zero; KV storage format is unchanged.
- model/decode.py now calls semantic leaves; redundant KV normalization/rotation buffers removed. No runtime compilation or new fallback switch.

## Reproduction
Activate the existing Python venv and source CANN. Build explicitly:
```bash
bash ops/decode/build.sh ops/decode/hc_finish.cpp ops/decode/libdecode_hc_finish.so
bash ops/decode/build.sh ops/decode/kv_prepare.cpp ops/decode/libdecode_kv_prepare.so
python scripts/test_decode_hc_finish.py
python scripts/test_decode_kv_prepare.py
```
Leaf tests require an idle device. Shared libraries are generated artifacts, following repository convention.

## Evidence
- HC: B1..4, attention/MoE, three scales, out-of-place and residual alias: exact numerical comparison passed (not signed-zero bitwise proof).
- KV: B1..4, five seeds x five scales including zero: bitwise BF16 and in-place comparisons passed.
- Leaf timing uses warmed captured graphs, alternating A/B order and device events. It is not whole-step timing.
- Service A/B: three prompts x three repetitions, 512 output tokens, temperature zero, thinking off, B1; baseline restarted with the same environment.
- model_step_host_ms median: 30.915059 -> 30.407611 ms; saving 0.507448 ms (1.64%).
- wall_ms_per_step median: 31.514308 -> 31.037975 ms; saving 0.476333 ms (1.51%).
- Initial long-running baseline is excluded: its higher latency would exaggerate gains.
- 9/9 acceptance checks passed: concurrent B2/B4 (actual batch metrics captured), streaming/non-streaming arithmetic checks and approximately 8k/36k/128k prefix retrieval. Actual input lengths below:
  - context-8192: 8165 input tokens.
  - context-36864: 36837 input tokens.
  - context-131072: 131045 input tokens.

## Limits
The proposed 1-2 ms opportunity is NOT achieved by this patch. Sequential service A/B is a small sample, not randomized causal attribution. Semantic service checks are not full-model bitwise proof. Long-context answers are short: correctness checks, not steady-state decode performance. Detailed numeric samples are in decode_leaf_fusion.json; raw requests/logs remain in /data/decode_fusion_dev.
