# Draft BF16 activation type-contract fix

The target FP16 consumer change also affected `dec_ds_routed_act`, which reused the same C++ kernel while its caller still supplied BF16 `routed_hidden`. This was a dtype reinterpretation bug, not a top-p policy change.

Fix: a dedicated BF16 draft activation kernel; target fusion and sampling policy are unchanged.

Validation:
- Draft B1..4, three seeds: broken output differed in about 99.8% of elements; fixed output bitwise matched the prior library in all 12 cases.
- Integrated target/draft consumer regression: 61 cases, independent CPU oracle, prior-library bitwise comparison, side stream and graph replay passed. FP16 finite patterns are covered; this is not an exhaustive BF16-domain test.
- Fixed full-chain audit: 10 runs, 512 delivered tokens each, 36,868 / 131,076 input tokens; clean exit. Exact steps/timings are in the companion JSON.
- Fixed receipts: 2,053 steps, zero errors. 1,440 sampled draws: zero float64-reference mismatches, zero kernel-reference mismatches, zero outside-nucleus samples.
- Whole old norm restored acceptance; replacing only target consumers did not. Target-only leaf/tap tests had missed the internal C++ draft reuse.

Scope: measured workload results, not a guarantee of bitwise deterministic generation or universal throughput improvement. Delivered tokens/step includes the verifier token and is not the draft-only acceptance rate. Greedy and sampling use different trajectories. No top-p adjustment was made. A21/A22 were not modified.

Raw audit evidence: `/data/workspace/mtp_sampling_audit` (fixed_results, fixed_analysis.json, draft_dtype_regression.json, integrated_consumers_test.log). Service restart and live smoke verification are tracked separately.
