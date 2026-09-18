# Development history

按开发顺序排列的完整提交消息。

```text
init: scaffold Qwen3.8 W8A8 TP4 mock engine
```

---

```text
implement TP4 engine graph MTP and cold prefix cache
```

---

```text
implement B1Q8 verify graph and aggregate decode projections
```

---

```text
fuse TP4 GDN decode below 50ms
```

---

```text
fix Qwen3.5 RMSNorm weight semantics
```

---

```text
implement persistent TP4 inference service
```

---

```text
add pure-BF16 GDN chunk prefill path
```

---

```text
vectorize packed full-attention prefill
```

---

```text
perf: fuse SwiGLU activation quantization
```

---

```text
perf: defer GDN convolution state snapshots
```

---

```text
optimize speculative GDN causal convolution
```

---

```text
perf: fuse paired GDN L2 normalization
```

---

```text
perf: precompute static GDN decay parameters
```

---

```text
perf: use fused GQA full attention in decode graph
```

---

```text
perf: use fused causal attention for prefill
```

---

```text
perf: move GDN triangular inverse to AICore matmuls
```

---

```text
perf: avoid unfold broadcast in GDN prefill convolution
```

---

```text
perf: make optimized GDN prefill the default
```

---

```text
fix: materialize packed GDN input before native decode conv
```

---

```text
feat: serve DFlash2 through persistent Q8 engine
```

---

```text
fix: restore engine context and isolate server protocol
```

---

```text
feat: add radix cold KV restore for DFlash2
```

---

```text
fix: restore large GDN prefill kernels
```

---

```text
feat: stream 12k prefill checkpoints through pools
```

---

```text
Optimize batched cold-prefix export
```

---

```text
Optimize cold-prefix restore transfers
```

---

```text
refactor: collapse fixed GDN hot paths
```

---

```text
perf: capture DFlash2 draft graph
```

---

```text
perf: specialize DFlash2 B1Q8 grouped convolution
```

---

```text
perf: use fused NPU RMSNorm in DFlash2
```

---

```text
perf: share DFlash2 RoPE factors across layers
```

---

```text
perf: merge DFlash2 gate and up projections
```

---

```text
perf: share DFlash2 graph context indexing
```

---

```text
perf: fuse DFlash2 graph attention
```

---

```text
Fuse DFlash grouped decode convolution
```

---

```text
Shard DFlash2 drafter across TP4
```

---

```text
perf: capture fixed-Q8 DFlash append_context as NPUGraph

Decode used to eagerly project accepted auxiliary features each round,
leaving ~7ms of launch/sync fragmentation after the TP draft rewrite.
Capture one fixed-width append graph that always computes BLOCK_SIZE rows,
then write only the accepted prefix into the paged KV pool. Prefill keeps
the eager path. Protected A/B matched generated ids/acceptance exactly and
cut decode from 67.35 to 62.29 ms/round (29.69→32.11 TPS).
```

---

```text
perf: add opt-in engine phase NPU timing
```

---

```text
optimize fixed B1Q8 GDN recurrent decode
```

---

```text
optimize GDN packed projections and strided convolution
```

---

```text
Share RoPE frequencies across attention layers
```

---

```text
Fuse QK RMSNorm with partial RoPE
```

---

```text
Avoid redundant GDN value transfers
```

---

```text
Pair GDN QK normalization per core
```

---

```text
Overlap GDN recurrent state writeback
```

---

```text
Fuse GDN RMSNorm gate with SwiGLU
```

---

```text
Hoist shared full-attention mask construction
```

---

```text
Reuse contiguous full-attention KV workspace
```

---

```text
Store W8A8 weights in native NPU layout
```

---

```text
feat: stabilize TP4 streaming inference service

Add OpenAI-compatible streaming protocol handling, coordinated TP4 strategy execution, service metrics, tool-call normalization, and regression coverage. Ignore generated runtime state and accelerator cache artifacts.
```

---

```text
checkpoint: stable TP4 service before batched KV refactor

Includes strict FIFO admission, complete control-plane writes, and structured backend error responses. Verified streaming, non-streaming, tool protocol, concurrent long prompts, and production health.
```

---

```text
feat: scale TP4 KV pool to 800K tokens
```

---

```text
feat: add paged batched prefill handoff
```

---

```text
feat: add minimal TP4 batched decode path

Extend the existing decode graph and GDN leaf ABI from B=1 to a unified
B=1..4 implementation, with model-level batched DFlash APIs and transaction
coverage. This does not add an old B1 bypass and does not rewrite MTP, prefill,
KV-cache ownership, or the server scheduler.

Absolute B1 performance baseline on atlas-a2b, four TP ranks, using this
commit's own source and native SO:

- Full 64-layer DecodeGraph B1Q8, 10 warmups + 60 measured replays:
  median 21.9166 ms.
- Public DFlash2Sidecar.draft() B1 cycle:
  median 5.336356 ms.
- Complete generate_dflash with prompt [1, 198, 260, 17], 64 generated
  tokens, 2 warmups + 8 measured generations in two opposite-order runs:
  decode median 527.165 ms; full wall median 890.634 ms.

The parent d1cb23c was measured under the same conditions at 21.9575 ms for
full DecodeGraph replay, 5.336486 ms for the public B1 draft cycle,
531.609 ms for complete-generation decode, and 898.534 ms wall time.
Tokens and output/state fingerprints were exact on every valid rank/run; all
valid run RCs were zero, fatal scan was empty, and NPUs were released.

Absolute batched graph leaf medians on rank 0, seven measured samples, with
serial B1 sums measured in the same run:

- B2: batch 8.469789 ms; serial B1 sum 10.340978 ms.
- B3: batch 10.945255 ms; serial B1 sum 15.426482 ms.
- B4: batch 13.572233 ms; serial B1 sum 20.588997 ms.

All B2/B3/B4 paths and candidates matched their serial references and all
reported unary maximum differences were 0.0.

Scope note: this commit proves the unified model-level B=1..4 decode path and
establishes the absolute B1 baseline above. It does not yet provide HTTP
request microbatch scheduling or establish a service-level concurrency
baseline.
```

---

```text
perf: pre-capture fixed TP4 batch graphs

Move B=2..4 target-verify and DFlash draft graph capture out of request latency while preserving the unified B=1 path. Reuse the existing graph pool, keep packet membership fixed, and remove one redundant synchronize already owned by draft_batch. No MTP, prefill, KV ownership, memory-management, or scheduler rewrite.

Cold TP4 production validation on NPU 4-7 (4 ranks), prompt [1,2,3,4,5,6,7,8], 64 new tokens:

- B=1 median decode 522.124 ms and wall 884.984 ms on rank 0 (other ranks decode 526.566-527.055 ms), below absolute +5% gates 553.523/935.166 ms; exact repeat and rank fingerprints match.

- B=2 nonadjacent packet: 177.57 tok/s, 1.155x decode and 1.131x wall speedup over serial.

- B=3 reverse packet: 108.69 tok/s, 1.064x decode and 1.067x wall speedup.

- B=4 across four prompt/SID rotations: 121.71-122.78 tok/s, 1.126-1.136x decode and 1.099-1.119x wall speedup.

All rows exactly matched independent serial outputs; release checks, slot permutation, draft/verify ratio, startup bucket checks, py_compile, diff-check, and 8 batch transaction/paging tests passed. Startup was 35.668 s; NPU processes were clean after validation.
```

---

```text
feat: add dynamic sequence boarding to TP4 decode

Add ref-compatible dynamic boarding across the strategy, control plane,
and model adapter while retaining fixed B1-B4 captured verify graphs.
Completed rows leave the active batch without reclaiming epoch KV pages;
new rows board only at safe decode boundaries, and the epoch releases all
pages together. Legacy static follower callbacks remain supported, while
dynamic operations require explicit op support.

Acceptance on A2-2, NPU 4-7:
- B1 absolute per-step timing (64 new tokens, 2 warmups + 8 measured runs,
  17 MTP decode steps per run): decode 31.551 ms/step, including draft
  5.378 ms/step and verify 23.926 ms/step. Values are medians of each
  measured run's phase time divided by its actual returned round count.
- Dynamic TP4 model gate boarded active rows 1->2->3->4; output lengths
  [48,12,24,36] exactly matched independent B1 results on every rank.
  Token hashes and decode rounds matched across all four ranks, and both
  target and draft KV pools returned to zero resident pages.
- Persistent service E2E accepted four requests staggered by 50 ms with
  prefill active_batch_size 1/2/3/4; all streams ended normally.
- py_compile, git diff --check, focused dynamic contracts, and all 58 CPU
  tests passed (6.673 s).

Use float32 for HCCL control payloads because the raw collective binding
supports floating dtypes only; all encoded integers remain exactly below
float32's 2^24 integer limit.
```

---

```text
perf: batch cold GDN checkpoint host clones

Replace per-slot host clone operations in RuntimeCache._finalize_gdn_bank with two contiguous bank clones and publish per-slot immutable views. The transient staging banks remain independently owned and reusable.

Absolute Atlas A2-2 TP4 validation on physical NPU 4-7:

- 12,288-token production prefill critical median: 1.747655 s, 7031.13 tok/s.

- 98,304-token production prefill critical wall: 16.215449 s, 6062.37 tok/s; all four ranks finite with target/draft length 98,304.

- 262,144-token production prefill critical wall: 54.835774 s, 4780.53 tok/s; all four ranks finite with target/draft length 262,144.

- B=1 decode absolute median: 31.384789 ms/step; token hash 159e354183aef330526959f7c0de1c02cc970800400f6ea2d1dce4432fe832fd.

- 58 unittest cases passed in 4.522 s; direct cold-state ownership and staging-reuse isolation check passed.

Diagnosis: 98,304-token per-slot clone probe spent 2.70-4.59 s/rank in host clones while event waits totaled 14-26 ms. Contiguous-clone probe reduced clone work to 0.22-0.33 s/rank.
```

---

```text
chore: enforce CANN 9.0.1 as sole toolkit

Remove active CANN 8.2/latest fallbacks from HCCL loading, AscendC builds, runtime setup, and documentation. Host-wide startup now sources /usr/local/Ascend/cann-9.0.1 only; the legacy toolkit is disabled separately at the system level. Verified fresh-login environment, NPU4 torch execution, zero legacy process mappings, and a CANN9 AscendC build linked exclusively against cann-9.0.1 libraries.
```

---

```text
perf: replace prefill GDN inverse chain with frozen CANN9 AOT kernels

Route the production 64-token GDN prepare path through checked-in CANN 9.0.1 AOT kernels while retaining LJQ_GDN_PREPARE_BACKEND=torch only as an explicit A/B reference. Verify every frozen shared object and npubin by SHA-256 before loading, reject stale modules, unsupported shapes, non-NPU placement, and non-fp32 inputs without fallback.

Absolute TP4 hot 12K prefill results on NPU4-7, one weight load, eight ABBA/BAAB rounds, each round measured by the slowest rank:
- AOT median: 1.368057 s, 8,982.08 token/s
- torch median: 1.683583 s, 7,298.72 token/s
- absolute median saving: 0.315526 s
- AOT critical samples: 1.419587, 1.341761, 1.377760, 1.358353 s
- torch critical samples: 1.690734, 1.676432, 1.673866, 1.694208 s

Correctness gates:
- 12K TP4 prefill: all four ranks finite and internally identical; all eight runs selected token 82
- 32-token greedy continuation after 12K prefill: AOT and torch tokens and every top-2 margin exactly identical on all four ranks; final cache length 12,319 before release and zero live pages after release
- 64-token leaf regression: u max abs 2.3841858e-7, w max abs 1.4901161e-8
- deterministic AOT-vs-torch final hidden difference remains bounded but non-bitwise: max abs 1.0625, mean abs 0.08004, cosine 0.99802
- 61 CPU contract/unit tests, five NPU adversarial boundary cases, and all nine frozen asset hashes pass
```

---

```text
perf: batch recurrent GDN rows without changing B1 path

Add a dedicated batch recurrent GDN kernel for B=2..4 and dispatch to it
from the existing launcher. Keep B=1 on the original kernel launch and ABI.
No model, MTP, cache, scheduler, or memory-management code is changed.

CANN9 leaf ABBA medians, old -> new:
- B1: 0.023237500 ms -> 0.023150250 ms
- B2: 0.042864200 ms -> 0.036328625 ms
- B3: 0.064699400 ms -> 0.049312775 ms
- B4: 0.084089625 ms -> 0.061000650 ms

TP4 full-sequence decode, 64 new tokens, rank-max median of two cold
OLD/NEW runs:
- B1: 423.368309392 ms -> 423.471279908 ms
- B2: 612.293510931 ms -> 613.945783582 ms
- B3: 778.984062141 ms -> 765.278531471 ms
- B4: 931.105107535 ms -> 924.126562197 ms

Final four-rank heterogeneous cold-start diagnostics with the new artifact:
- B1 decode median: 413.227725..413.704070 ms
- B1 wall median: 742.162168..742.821004 ms
- B1 draft median: 69.316045..74.747664 ms
- B1 verify median: 307.883575..310.450342 ms
- startup: 39.569505..40.019004 s
- packet draft/verify ratio: 0.221819..0.254533

Validation:
- B1 repeated output/state hash was stable for 10/10 launches.
- Heterogeneous B2 and B4 output and recurrent state matched the old kernel
  bit-for-bit, including non-adjacent state indices and mixed accept counts.
- CPU reference maxima: output 9.54e-7, recurrent state 2.44e-4.
- Four-rank full-sequence token hashes matched for every B and every run.
- Slot permutations, accept/reject rollback, resource release, B1 absolute
  gates, and draft/verify proportion gates passed.
- 61 unit tests passed.

The legacy final-batch harness rule requiring every heterogeneous packet to
beat its serial aggregate also fails on the old artifact in the same cold
setup; its measurements were retained as diagnostics rather than attributed
to this leaf change.
```

---

```text
Group row-wise TP all-reduces for batched decode

Keep the B=1 decode path on its existing direct all-reduce. For B=2..4,
preserve each row's BF16 reduction shape and submission order, wrap the
row collectives in HcclGroupStart/HcclGroupEnd, update rows in place, and
remove the post-reduction torch.cat copy.

Validated on A2-2, CANN 9.0.1, TP4 devices 4..7:
- Four-rank leaf outputs were bit-exact for B=1..4.
- Leaf synchronized rank-max medians, old / grouped:
  B=1: 0.590 / 0.535 ms (production B=1 retains the direct branch)
  B=2: 0.820 / 0.753 ms
  B=3: 1.068 / 0.918 ms
  B=4: 1.316 / 1.049 ms
- 64-layer Q8 synchronized rank-max step medians, old1 / new / old2:
  B=1: 22.512 / 22.273 / 22.376 ms
  B=4: 48.633 / 47.270 / 48.816 ms
- Old/new/old output fingerprints matched exactly on every rank for B=1
  and B=4.
- Four heterogeneous sequences matched B=1 token-for-token with distinct
  acceptance rates and rollback counts; serial statistics and resource
  release checks passed on every rank.
- 61 unit/contract tests passed; git diff --check passed.
```

---

```text
Batch QK RMSNorm and RoPE across decode rows

Keep the B=1 DecodeGraphRunner branch unchanged. For B=2..4, invoke the
existing AscendC QK RMSNorm/RoPE kernel once over all Q8 rows instead of
launching it once per sequence and concatenating the outputs. Generalize
the native wrapper shape gate from 8 rows to 8/16/24/32 rows; the kernel
ABI already accepts dynamic row counts and strides.

Validated on A2-2 TP4 with Q8 decode:
- 200-iteration NPU leaf medians, row-wise / batched: B1 0.426 / 0.160 ms,
  B2 0.528 / 0.159 ms, B3 0.700 / 0.157 ms, B4 0.879 / 0.168 ms.
- Leaf Q and K outputs were bit-identical for B=1..4, max abs difference 0.
- Independent cold-process HEAD -> candidate -> HEAD synchronized rank-max
  step medians: B1 22.281 / 22.245 / 22.226 ms; B4 47.067 / 46.392 /
  47.189 ms. Output fingerprints matched on every rank for B1 and B4.
- Four-rank NPUGraph heterogeneous-sequence validation matched independent
  B1 tokens and serial MTP statistics, exercised rejection/rollback, and
  released all sequence resources.
- 61 unit/contract tests passed; git diff --check passed; NPUs 4-7 were
  idle after validation.
```

---

```text
Skip redundant DFlash context zero-fill before masked attention

Return the safely gathered K/V tensors directly for both B=1 and batched
DFlash decode graphs. Invalid context slots already have in-bounds gather
indices and are excluded by the FIAS attention mask, so the two per-layer
where/zeros_like operators do not affect attention results.

Validated on A2-2, CANN 9.0.1, TP4 devices 4..7:
- Direct FIAS leaf validation covered B=1..4, prefix and random validity
  masks, and 96 seeded cases; every output was bit-identical with max abs
  difference 0.
- Full Gather+FIAS NPUGraph old / new p50 latency:
  B=1: 0.331 / 0.296 ms
  B=2: 0.326 / 0.273 ms
  B=3: 0.363 / 0.298 ms
  B=4: 0.366 / 0.302 ms
- Cold candidate / old / candidate full-sequence runs completed on all four
  ranks with identical token hashes and successful resource release.
- Candidate-mean / old per-round draft, decode, and wall latency:
  B=1: 4.989 / 5.269 ms; 30.411 / 30.884 ms; 52.337 / 53.522 ms
  B=2: 7.526 / 7.906 ms; 43.818 / 44.404 ms; 87.761 / 90.114 ms
  B=3: 9.334 / 9.757 ms; 54.779 / 55.705 ms; 122.813 / 126.482 ms
  B=4: 11.030 / 11.479 ms; 66.333 / 66.397 ms; 155.198 / 157.267 ms
- Source-level four-rank heterogeneous validation matched independent B=1
  tokens and serial MTP acceptance/rejection statistics on every rank,
  exercised rejection rollback, and released all sequence resources.
- 61 unit/contract tests passed; py_compile and git diff --check passed;
  NPUs 4-7 were idle after validation.
```

---

```text
Collapse batched GDN convolution into one four-core launch

Keep the existing B=1 convolution path unchanged. For B=2..4, assign each
of four cores one disjoint channel tile and process all batch rows in that
core. Load the row-invariant convolution weights once per tile, preserve
explicit UB pipeline ordering, and update pending state with one captured
NPU copy after the native launch succeeds.

Validated on A2-2, CANN 9.0.1, TP4 devices 4..7:
- Varied B=1..4 direct and NPUGraph tests were bit-identical to the prior
  implementation for convolution output and pending state. Three cold
  production-artifact processes each passed 64 direct cases and 32 graph
  replays.
- Four-rank heterogeneous validation matched independent B=1 tokens and
  serial MTP statistics, exercised distinct acceptance/rejection and
  rollback behavior, and released all sequence resources.
- Cold old1 / new1 / old2 / new2 synchronized rank-max step medians:
  B=1: 22.3856 / 22.0101 / 22.2906 / 21.9925 ms
  B=4: 46.5541 / 44.8893 / 46.2992 / 44.8595 ms
  Output fingerprints matched exactly on every rank for B=1 and B=4.
- 61 unit/contract tests passed; py_compile and git diff --check passed;
  NPUs 4-7 were idle after validation.
```

---

```text
Remove unused speculative-token config

The production DFlash2 path has a fixed VERIFY_WIDTH=8 (one target plus
seven drafts). EngineConfig.max_speculative_tokens was an unreferenced
remnant of the original one-layer qwen_native MTP scaffold and never
controlled either path. Remove it to avoid implying that production
verify width is configurable.

Validation:
- py_compile: model/config.py, model/model.py, model/model_api.py, model/mtp.py
- tests.test_model_capacity: 4 passed
- full unittest discovery: 61 passed
- EngineConfig no longer exposes max_speculative_tokens
- model.model_api.VERIFY_WIDTH remains exactly 8
```

---

```text
Batch TP reductions with exact BF16 order

Replace B=2..4 per-sequence grouped all-reduces with one captured all-gather followed by the empirically equivalent TP4 small-message BF16 reduction tree ((rank0 + rank1) + rank3) + rank2. Keep the B=1 path unchanged. This removes batch-linear collective launches without changing target logits or speculative decisions.

Validated on A2-2 CANN 9.0.1 TP4 devices 4..7: three-seed B1..B4 leaf tests were bit exact; cold old/new/old 64-layer graph fingerprints matched for B1 and B4 while B4 fell from 45.13ms to 32.07ms; full-sequence B1..B4 token hashes, acceptance and resource release matched baseline; four heterogeneous rows matched serial B1 tokens/stats with distinct rejection/rollback on all ranks; 61 tests passed; py_compile and diff check passed.
```

---

```text
Batch exact TP reductions in DFlash graph

Replace B=2..4 per-row DFlash all-reduces with one captured TP4 all-gather and the exact small-message BF16 reduction order ((rank0 + rank1) + rank3) + rank2. The B=1 graph remains unchanged.

Validated on NPU4-7 with real resident DFlash KV/context:
- cold old/new/old B=4 draft medians were about 10.86 / 9.73 / 10.88 ms (-10.4%);
- full path, candidate, and unary tensors were byte-identical on every rank;
- cold full-sequence old/new/old B=1 and B=4 token hashes, speculative statistics, and release checks matched;
- four heterogeneous rows matched serial B=1 tokens and rollback statistics on every rank;
- 62 unit/contract tests, py_compile, and git diff --check passed.
```

---

```text
Keep DFlash draft handoff on device

Return resident anchor/path tensors from the captured DFlash graph and copy them directly into the target verify graph. Defer the single host path readback until target argmax has synchronized the round, while preserving the existing CPU fallback at topology boundaries for static and dynamic decode.

Validated on physical NPU4-7 with CANN 9.0.1. Balanced AB/BA synchronized step medians improved from 30.111/38.542/42.989/48.595 ms to 29.722/38.059/42.316/48.069 ms for B1-B4. Static heterogeneous and dynamic boarding cold gates were exact on all four ranks with full resource release; 65 unit tests passed.
```

---

```text
Eliminate DFlash batch convolution concatenations

Write each unchanged B1Q8 convolution directly into its preallocated batch row slice instead of allocating per-row outputs and concatenating them. Preserve the original current-stream kernel launch count and arithmetic, and validate caller-provided output buffers.

B4 device profiles drop from 614 to 594 kernels on every TP rank: the targeted Concat kernel falls from 50 to 30 while dflash_grouped_conv_b1q8 remains 80. Four-rank heterogeneous full-sequence tokens, acceptance/rollback statistics, release gates, NPUGraph replay, and 66 tests pass. Drift-normalized cand-old-cand decode improves B4 by 0.487-0.519 ms per step across ranks.
```

---

```text
Read batched GDN state directly from base slots
```

---

```text
Fuse GDN beta sigmoid into recurrent kernels
```

---

```text
Route batched GDN decode through AIV kernels
```

---

```text
Reset dynamic decode epochs and restore boarding grace
```

---

```text
Expose runtime cache page ownership for cleanup audit
```

---

```text
Return transient epoch allocations to NPU driver
```

---

```text
Preallocate common TP4 verify graph capacities
```

---

```text
Freeze dynamic boarding cohort after grace period
```

---

```text
Restore dynamic boarding after initial grace
```

---

```text
Raise max_tokens default to 8192 and cap to 12384
```

---

```text
Raise max_tokens cap to 16384
```

---

```text
Release v1.0.0
```

---

```text
chore: clean release tree and archive development artifacts
```

---

```text
Add request cancellation at dynamic safe points
```

---

```text
Optimize dynamic decode control synchronization
```

---

```text
Optimize batched GDN convolution with row-parallel AIV kernel

Parallelize independent (row, channel tile) blocks; preserve arithmetic.
Include reproducible CANN9 build entry and validated native SO.

Atlas A2 TP4 cold-repeat medians, ms per speculative round, NOT per token:
- b1_r: 29.056864 -> 28.944742 ms (-0.39%).
- b4_r: 46.147022 -> 45.004200 ms (-2.48%).
- mixed_r: 46.182762 -> 45.088358 ms (-2.37%).
B1 improvement is small; no significant gain claim.
8 micro cases bit-exact; 10 short + 6 long + 6 dynamic cases exact on all 4 ranks.
Contexts 2040/8190/16370; dynamic 4->3->2->1 verified.
Additional 10-case cold short repeat and trailing baseline exact.
Fresh build byte-identical to validated SO, SHA256:
fb0a4f9999b5dad8c03017e8c03c859f4c0784fe91adcb55db39fa57c06fbee1

Production health and four-rank provider mappings verified after restart.
B1 nonstream, SSE DONE, four concurrent 160-token completions passed.
Evidence and timing samples: docs/conv_aiv_experiment_20260909.json.
```

---

```text
Speed up B1 GDN recurrent decode with immutable base-state kernel

Batch-1 speculative decode copied the previous SSM state into the pending
buffer on every step, then let the recurrent kernel read that copy. The new
gdn_recurrent_b1base_direct kernel reads the immutable base state directly
through its own pointer, so the per-step D*D state copy disappears from the
hot path. Batch>=2 keeps the existing batched kernel unchanged.

Measured on Ascend 910B3, TP4 (NPU 4-7), Qwen3.8-27B-W8A8 (model/config.py MODEL_DIR), 4 ranks per run,
candidate vs control executed back to back with the service stopped:

  case      candidate   control    delta
  b1_r0     28.301 ms   28.930 ms  -2.18%
  b1_r1     28.276 ms   28.948 ms  -2.32%
  b1_r2     28.298 ms   28.922 ms  -2.16%
  warm_b1   28.555 ms   29.282 ms  -2.48%
  b4_r0/1/2 45.139/45.024/45.112   -0.22/-0.29/-0.19% (kernel unchanged, noise)

Correctness: 22/22 cases bit-exact against the recorded gold traces
(token ids, accepted draft tokens, step count, commits, targets, geometries;
all 4 ranks agree) - 10 short cases (b1/b4/mixed/warm), 6 long-context cases
(ctx 2040/8190/16370) and 6 dynamic-batch cases (batch2, batch3, 4-3-2-1
shrink). Full unit suite: 77 tests OK.

Dead code removed with the switch: the legacy base_state=None B1 route and
its LJQ_GDN_B1Q8_DISABLE environment fallback in ops/kernels.py (the only
caller always passes a base state), and the unused batch path inside the new
kernel source. Also refreshed a stale gdn_conv contract assertion left by
c68f003, which no longer matched the shipped kernel.
```

---

```text
docs: record decode follow-up experiments; no runtime promotion

Base 9db0a95; physical NPU0-3 sequential runs; production4-7 unchanged.
Append RMS: 78 tests, short/long/dynamic exact. B1 initial
28.662->28.561ms, repeat 28.604->28.635ms; B4 initial 44.672->44.674ms,
repeat 44.903->44.529ms. No reliable B1 gain demonstrated.
Ordered TP4 sum: 77 tests, 10 short cases exact; B4 44.903->44.776ms.
Separate-input gate: 32 finite microtests zero differences, 77 tests,
10 short cases exact; B1 28.604->28.573ms, B4 44.903->45.168ms.
Values are median-of-three steady decode-round medians, not kernel time.
Archive comparisons and zero-context candidate patches; activate none.
```
