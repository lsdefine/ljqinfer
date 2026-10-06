# Draft attention tiles + remove duplicate prepare (A2-3)

Base: 209df373, /data/ljqinfer_dsv41f_tp8. Evidence: /tmp/draft_attention_opt_v1.

## Changes
- norm.cpp: draft attention uses 32-key tiles, vectorized score/exp/reductions and block online softmax. Receipt, slot, ring, pending-token mask and sink semantics unchanged; ABI and workspace allocation unchanged. FP32 reduction order changes (not bitwise equivalent).
- decode.py: remove outer duplicate dec_ds_prepare binding/call. NativeDSpark.propose still prepares itself. Seed preparation remains intact.

## Leaf validation
`bench.py`: 28 cases, B1-B4, base receipt positions 0/126/255/20000/999/34 (plus batch offsets), reverse-ordered slot mapping, duplicate/inactive/invalid receipts, circular history, direct warmup launches and captured graph replay after receipt mutation. Independent CPU attention reference and original shared library used. Candidate CPU max abs <=0.016 and relative L2 <0.003 asserted; original/candidate max abs <=0.032 and relative L2 <0.003 asserted. Observed max abs 0.0078125, relative L2 9.251e-5. Not exact equality.

Correct AIV build, 128 graph replays per sample, three alternating-order paired samples (microseconds, medians):

| B | original | candidate |
|---|---:|---:|
|1|93.418|18.116|
|2|184.086|32.860|
|3|274.612|48.186|
|4|365.582|65.568|

`prepare_check.py`: 25 tests across B sequence 1/4/2/1/3 and live/inactive/invalid slot/duplicate/negative history. One versus two prepare calls bitwise equal in captured graphs, plus CPU expected-id checks. Results: prepare_results.json.

## Integrated validation and limits
Single serialized TP8 process group, no profiler, two 192-output-token generations per process, 80 input tokens, B1. Full Engine.step host wall time includes normal synchronization; first two steps omitted for median. All eight ranks must finish, and source/library hashes must remain stable.
- base_run2 (earlier baseline): 32.830 / 32.753 ms median full step.
- base_run3 (independent repeat): 33.157 / 33.143 ms median full step.
- prepare_only_run1: 33.361 / 33.120 ms. No demonstrated end-to-end gain from prepare removal alone.
- candidate_vec_run1: 32.224 / 32.214 ms.
- candidate_vec_run2 (independent repeat): 32.539 / 32.610 ms; all eight ranks complete and hashes stable.

These short runs indicate a small improvement, not a statistically established percentage or token-throughput guarantee. Baseline drift and differing acceptance paths prevent strict paired end-to-end attribution. Leaf acceleration is much larger than whole-step improvement.

Original model already produces different token streams across repeated identical requests (first divergence observed at output index81). Candidate has the same limitation; some runs match original exactly, others do not. This is not a proof of exact end-to-end numerical equivalence or broad quality. No B2-B4 full-engine, long-context, cancellation, or acceptance-quality certification is claimed. Leaf coverage is broader than integrated coverage.

## Excluded experiments / reproduction
- candidate_run1 invalid: another session changed source during execution; never use for performance claims.
- candidate_run2 used wrong architecture: build.sh selects -vec by basename. A source named norm_candidate.cpp compiled dav-c220, affecting the whole library. Never use that library/result as the optimized AIV result.
- Correct build: `bash ops/decode/build.sh /tmp/draft_attention_opt_v1/vector_candidate/norm.cpp /tmp/draft_attention_opt_v1/libnorm_candidate_vec.so` (basename MUST be norm.cpp). Canonical in-repo build also selects -vec correctly.
- Final micro bench loads libnorm_base.so and libnorm_candidate.so under evidence root; latter is now the corrected AIV library. Before the run both source and installed .so were checked against candidate source/build.
- Full harness: `python -B /tmp/draft_attention_opt_v1/coordinator.py /tmp/draft_attention_opt_v1/<fresh-output-dir>`; requires idle eight NPUs, valid current cache and serialized access. Do not run concurrently with other device experiments.
