# Development history

按开发顺序排列的完整提交消息。

```text
Initialize GLM-5.3 workspace from GLM-5.2 TP8 snapshot

Source snapshot: 1cff864874407ab7b30f25a4001da234064eaf56; includes copied working-tree changes. Runtime code unchanged; GLM-5.3 not yet adapted.
```

---

```text
Document operator migration boundaries and verified expert quantization

Audit all 1809 GGUF tensors and 13 selected operator roles; distinguish Q8 LUT cache from source precision and quantify FP8 exceptions. No runtime changes or GPU tests.
```

---

```text
Add real FP8 expert INT4 reference probe and measured RTN baseline

Three layer12 experts, G64/G128, all INT4 versus FP8 down; 72 synthetic output comparisons. Exact packing and artifact reload checks pass; peak Torch reserve 452 MiB. Not a production kernel, quality acceptance or speed result. Reproduction and scope in docs/INT4_PROBE.md.
```

---

```text
Keep documentation and tests local; track runtime code and required assets only
```

---

```text
Add data-free groupwise INT4 MSE scale optimization

Fixed signed nibble layout and FP16 scales; RTN candidate, clipping search and four least-squares refinements. Layer12 experts 0/127/255, G64/G128: group SSE non-regression, CPU unpack oracle, save/reload checks pass. Synthetic output mean relL2 reduced 10-13 percent versus matched RTN; 382/384 comparisons improved, two G64 hybrid regressions retained. Peak Torch reserve 742 MiB. Local evidence: /mnt/data2/kw/glm53_int4_datafree_v1/report.json; tests/probe_datafree.py untracked by policy. No full-model or production-kernel validation.
```

---

```text
Retire legacy GLM52 MoE and implement four GLM53 routed kernels

Hash-verified local snapshot of 70 tracked files in ignored archive/glm52_moe_pre_int4. Remove 16 legacy MoE/cache sources and stale bindings; preserve non-MoE recipes. Old engine entry explicitly blocked until FP8 loader and integration are ready. Add direct decode GEMV and GPU-dispatched prefill Tensor Core kernels for INT4 gate/up with INT4 or FP8 down, explicit preallocated workspaces. Independent Torch oracle: 44 cases plus 16 multiseed cases, eager and CUDA Graph replay with changed inputs/routes. Main run peak Torch reserve 516 MiB; max relL2 decode 0.000193, prefill 0.000797. Tests/docs/archive not tracked. No TP8 collective, whole-model, long-context or performance acceptance.
```

---

```text
Optimize routed INT4/FP8 MoE tiles and count-driven prefill dispatch

Preserve four entry points and workspace ABI. Bound decode K tiles and reuse expert weights with channel-first CTA order; prefill uses bounded expert worker grid and loops device counts, with reduced pipeline pressure. Remove unsuccessful experiments. A100 GPU7, E256 H6144 I256 top8 G64, warm CUDA graph median: INT4 decode T6/8/128 1.394/1.786/26.435 -> 0.546/0.723/11.185 ms; FP8down 1.223/1.555/22.771 -> 0.571/0.753/11.680 ms. INT4 prefill8K/16K 254.318/506.723 -> 41.695/77.834 ms; FP8down 312.891/623.178 ->45.971/86.959 ms. Two independent speed rounds. 44 regression cases plus 10 full-output target cases (G64/G128), eager/graph/changed-skewed routing; 16 repeated graphs bitwise stable. Target max relL2 0.000304. Repro ignored tests/bench_moe_speed.py --tokens 6,8,128,8192,16384; tests/test_moe_four.py; tests/test_moe_target.py. Tests/docs/archive stay untracked. No TP8 collective, shared expert, router, model quality or end-to-end TPS acceptance. Dense routing workspace retained.
```

---

```text
Optimize 12K routed prefill with per-call dequant and pipelined BF16 GEMM

Reuse 1.5GiB BF16 scratch across projections/layers (target rank shape); include dequant in graph timing, never cache expanded layer weights. Public calls unchanged; Workspace.create now provides scratch for large prefill. Generic/small shapes remain inline.
A100 TP8 local E256 H6144 I256 top8 G64 T12288: INT4 59.714 -> 11.771ms, FP8 down 66.483 -> 11.696ms; repeat agrees. Useful BF16 math efficiency about25.3% of312TF/s. Decode6 unchanged.
44 regression cases and four 12K full-output G64/G128 cases pass independent Torch reference, changed skewed graph routes and16 bitwise-identical replays. Tests/reports intentionally untracked. No whole-model or communication performance claim.
```

---

```text
Optimize small routed decode with paired INT4 unpack and split reductions
```

---

```text
Accelerate routed decode with K-major reductions and direct FP8 bit decoding
```

---

```text
Accelerate INT4 decode with packed loads and exact bitwise float conversion
```

---

```text
Optimize SM80 decode BF16 rounding with explicit PTX packing
```

---

```text
Optimize large prefill expert tile scheduling and down dequantization
```

---

```text
Optimize SM80 prefill packed INT4 and bitwise FP8 dequantization
```

---

```text
Add complete prefill and decode MoE layers with router and shared side stream

Select routed down precision via down_dtype; preallocate layer workspace, synchronize shared branch, and require explicit TP reduction callback. Keep router sigmoid rounding consistent at top-k ties. Validate 12 layer cases, 44 core regressions, H6144 verify6/12K outputs and graph replay, and four eight-rank NCCL graph cases. Rank-local timings: 12K 13.39/13.53 ms; verify6 0.212/0.255 ms (INT4 G64 / FP8 down G128). Full model loader/service integration and target-shape TP8 performance remain untested. Tests/docs remain untracked.
```

---

```text
Fuse shared expert addition into MoE combine after side-stream join

Keep public routed/layer ABIs and workspace size unchanged; join aux stream after routed down, then add shared output after routed reduction. Router arithmetic unchanged. Same-process ABBA 12K full layer: INT4 G64 13.406/13.415 -> 13.106/13.105 ms; FP8 G128 13.565/13.567 -> 13.265/13.261 ms (~2.25%). Verify6 remains within timing noise, no claimed gain.

Validation: test_moe_four.py 44 cases; test_moe_layer.py 12 cases; test_moe_layer_target.py verify6/12K both precisions; test_moe_layer_tp8.py 8-rank NCCL eager/graph/changed-input, all pass. bench_fusion_ab.py same-partial top8 fused combine equals separate add bitwise for all four targets. Test scripts/reports remain untracked per policy. No full-model/service integration or TP8 target-shape performance claim.
```

---

```text
Fuse exact sigmoid with single-warp grouped routing and optimize input cast

Keep FP32 projection and group-ranked tie order. Validate 1,228,800 routing rows, 96 boundary cases, 12 layer and 44 core cases, real-shape references and TP8 graph replay. Same-process prefill A/B saves 0.20-0.25 ms; no large decode gain claimed.
```

---

```text
Overlap prefill GU unpack with routing and fuse gate/up activation

Preserve BF16 boundaries and rebuild weight scratch on every graph replay. No extra workspace; decode unchanged. 16-round graph A/B at T12288: INT4 13.4466->12.9206 ms, FP8-down 13.6758->13.1696 ms. Passed 12 layer/44 core cases, real-shape oracle, weight/input/skew graph mutation exact A/B and TP8 T8192 graph tests. Tests/docs remain ignored.
```

---

```text
Increase large-prefill down GEMM row tile to 128

Keep BN128 BK64, four warps and three stages. Sixteen-round graph A/B against 443a2ce: INT4 12.9445 -> 12.5930 ms; FP8 down 13.1941 -> 12.7941 ms at T12288. Exact old/new replay after weight, routing skew and input mutations. Passed 12 layer, 44 core, real-shape references, and eight-rank small-hidden T8192 graph regressions. No workspace change or decode change.
```

---

```text
Integrate GLM53 TP8 non-sparse transformer block with native half attention

Decode source FP8 attention to FP16 without Q8 requantization; reuse paged KV/MLA and INT4 or FP8-down MoE. Fuse FFN norm/residual and release attention temporaries to fit the available 8GiB budget.

Layer12 real weights, synthetic hidden/prefix KV, TP8, K0=64512: T=12288 eager medians INT4 153.799ms / FP8-down 201.559ms; T=6 graph 0.560128ms / 0.603136ms. Independent attention reference covers permuted pages and changed graph inputs/context; full decode block reference relL2 <=5.47e-5, routing identical on all ranks.

Local reproduction: tests/build_half_attn.py; torchrun --standalone --nproc-per-node=8 tests/bench_transformer.py <prefill|decode> <int4|fp8> <12288|6> 64512; tests/validate_native_attention.py; torchrun tests/validate_native_block.py <int4|fp8>. Tests/artifacts remain local per repository ignore policy.

Limits: no full-model/text validation, no 12K independent full-block oracle; performance uses synthetic activations. Prefill timing spread is attention-dominated, not established as a down-format effect.
```

---

```text
prefill: use fixed dense MLA path and propagate allocation failures

Remove online paged fallback and memory/shape dispatch from native half prefill. Keep fixed 1024-query dense chunks. Validated layer12 TP8 INT4/FP8: fresh/permuted KV, 12K tail64 oracle, verify6 graph, explicit allocator-cap OOM; two benchmark runs. Reproduction archive: /mnt/data2/kw/glm53_layer12_tp8/fast_only_reproduction.tar.gz.
```

---

```text
sparse: checkpoint validated primitives and dsv41f-derived paged MLA
```

---

```text
sparse: port DSV41f exact TP8 index selection and fuse local scoring
```

---

```text
sparse: shard index queries to eliminate full-score TP reduction
```

---

```text
sparse: use direct async shared-memory loads in paged MLA
```

---

```text
docs: record rejected sparse MLA tuning experiments
```

---

```text
perf: pair TP query ownership to reuse sparse MLA KV across 16 heads
```

---

```text
perf: parallelize exact top-k emission scan for sparse decode
```

---

```text
perf: split prefill and decode index operators; parallelize decode radix
```

---

```text
perf: tune decode packed gather and prefill query tiles independently
```

---

```text
Add standalone A100 V2-style exact top-k and graph benchmarks
```

---

```text
Wire exact V2 into default TP8 sparse decode selection
```

---

```text
Bind TP8 sparse attention and shared index IDs into GLM53 model layers
```

---

```text
Add resumable TP8 full-model routed INT4 conversion
```

---

```text
Replace legacy weights cache with auto-built GLM53 rank-local shm snapshots
```

---

```text
Fix standalone cache audit entry and record eight-rank loading PASS
```

---

```text
Add TP8 Q8 target engine and independent DFlash KV pool with bounded validation
```

---

```text
Remove slow Q8 MoE kernel and restore existing decode implementation

Existing prefill/decode kernels unchanged. Eight-rank real-weight DFlash repeated generation passes; short-prefix full-model Q8 verify median 59.034ms versus prior ~288ms. Strict prefill/decode equivalence remains unresolved and documented.
```

---

```text
Wire GLM53 DFlash service; restore prefill configuration and preserve stream metrics
```

---

```text
Restore GLM53 prefix caching and lifetime-owned execution scratch
```

---

```text
Fuse GLM53 RMS pointwise tail while preserving reduction order

Reference: dsv41f_eptp8 3659e84 normalization fusion. Retain PyTorch square/mean/rsqrt; full Triton reduction changed real-model routes and was rejected. Fuse two multiplies and FP16 cast with contraction disabled.

12K history0 TP8 real weights eager forward: 3.188s to 3.091s; restored baseline 3.190s. Eight ranks bitwise-equal final logits; four micro shapes including Q8 strided input exact vs old expression, FP64 relative error <6e-6. No Graph, MoE/MLA main kernel, capacity or production constant changes. <=2s target not met; no whole-service latency claim.

Evidence: /mnt/data2/kw/glm53_int4_tp8/service_audit/{rmsexact12k_rank*.jsonl,rms_landed_micro.log}; reproduction rmsexact_prefill_audit.py and rms_landed_micro.py.
```

---

```text
Integrate SM80 paired prefill and fused elementwise kernels; improve engine throughput 23 percent
```

---

```text
Optimize production prefill with shared RoPE, direct paged writes and exact RMS reduction

Preserve MLA/MoE/decode and 144K policy. Balance only first 12K paired query chunk. Full TP8 Engine.prefill 2.5050s -> 2.4155s (+3.70% tokens/s), all-rank logits bitwise equal. Include boundary regressions and validation report.
```

---

```text
Project V before paired prefill exchange; full-model 2.41889 -> 2.40192 s

Gather per-layer peer V weights at initialization; sequential layers share
preallocated send/receive buffers. BMM writes token-major views directly,
halving pair output payload (512 -> 256). First 12288-token chunk only.

Measured A100 TP8, history=0, 12288 tokens, no HTTP:
- Candidate attention: 1215.77 -> 1199.86 ms (21 full + 57 shared calls).
- Engine candidate: 2.418889862 -> 2.401917676 s (-16.972 ms);
  5080.016 -> 5115.912 tok/s (+0.7066%), 2 warmups + 5 alternating
  rank-max wall samples. All 8 ranks: logits + 6 features bitwise equal.
- Formal binding regression: 1215.38 -> 1197.26 ms, 3 alternating samples;
  all 8 ranks full/shared output, IDs, MLA/index caches bitwise equal.

Keep decode, other chunks, MLA/MoE kernels and 144K capacity unchanged.
Full 144K sequence untested; attention <1s and full-model <2s not reached.
Future small improvements stay on the operator bench until gains accumulate.
```

---

```text
Integrate SM80 prefill optimizations; full-model 2.39826 -> 2.20094 s

Register softmax/max reduction, PAGE64 specialization and exact fused RMS.
TP8 Q / paired KV token-owned buffers shared across sequential layers.
Projection specialization restricted to the 12288-token first chunk.

Same-path Engine.prefill, 78 real layers, 2 warmups + 5 samples/version:
92045ca 2.398256 s / 5123.72 tok/s -> 2.200936 s / 5583.08 tok/s.
Latency -8.23%, throughput +8.97%, saved 197.32 ms.
All 8 ranks: logits and six feature tensors bitwise equal for 12K,
subsequent33, verify8, reset/first17; 48 RMS cases pass.
Rank0 allocated memory +170.75 MiB. No HTTP restart or constant changes.
Full-model 2-second target remains unmet.
Evidence: reports/engine_history_integration.json; service_audit engine_history_*
```

---

```text
Optimize complete Q8 decode round to 48.62ms on A100 TP8

Skip empty MLA tiles without assuming compact IDs; share decode RMS/RoPE
and cache packing fusion; capture DFlash committed-feature append for n1..8
with exact GEMM shapes, retained offsets and capture-time KV preservation.

Integrated B1 warmed-graph results, 12 prompts x384 tokens vs 6f55a13:
weighted complete round 70.14256 -> 48.62456ms (-30.68%); aggregate decode
41.18 -> 59.41 tok/s (+44.25%). Case means 48.33..49.04ms, not max latency.
Separate profile: verify42.72..42.75, draft3.71, commit0.71..0.72ms.
All output tokens and accepted-draft sequences exact on the 12 cases.
Production append leaf: n1..8 x4 positions x8 ranks KV exact, eager
2.76..3.07ms -> graph0.56..0.64ms. MLA10 leaf cases exact incl holes/full.

Original suite stopped after timings at extra cold/hot equality assertion.
Separate old/new cache diagnostics both exit0: cold/hot/reset individually
match tokens and acceptance across versions; cold==reset, cold!=hot in
both. Pre-existing cold/hot discrepancy not fixed or root-caused here.

Scope: B1 short prompts, warmed graphs,384 tokens. No new long-context,
HTTP/concurrency/HF-truth or12K prefill benchmark. HTTP remains stopped.
Evidence and reproducible probes: docs/decode_round_optimization.md.
```

---

```text
Optimize SM80 decode kernels: 48.62 -> 43.79 ms

Fuse small-row RMS with ATen reduction order and unaligned fallback; lop3 INT4 unpack and paired BF16 conversion; wider down tile; MLA live-tile prescan.

TP8 production-entry graphs, profiling off, 12 prompts x 384 output tokens: weighted complete round 48.624564 -> 43.788272 ms (-9.946%). RMS+MoE intermediate 45.605822 ms. All 12 token streams and acceptance sequences equal baseline 59ae8f1; all 8 ranks complete; cold/hot/reset match same-condition decode50/optimized.cache.json.

Regression PASS: 165 RMS shape/stride cases plus unaligned offsets and graph mutation; 44 MoE cases; 6 sparse cases with graph metadata/page-table mutation.

40ms target NOT met. Sparse-hole microbenchmarks can regress. Long-context performance and HTTP not revalidated; HTTP remains stopped. Evidence: /mnt/data2/kw/glm53_int4_tp8/service_audit/decode40/integration_summary.json, regressions.json and integrated/. No universal latency claim.
```

---

```text
Integrate bitwise DFlash fusion: engine round 44.10->43.62 ms

Share six-layer RoPE frequencies in draft and KV append, preserve BF16
product rounding and original RMS, fuse greedy path selection.
Fresh node09 TP8 12-case suite: 44.103706->43.616257 ms (-1.105%).
All 12 improved; all 8 ranks output hashes, steps, acceptance histograms
and cold/hot/reset cache outputs match baseline e6ab07c.
58 leaf checks and all-rank n=1..8 KV append graph regression passed.
Isolated draft 3.595->3.214 ms is separate from engine round timing.
HTTP not restarted, 40ms unmet, long-context/concurrency not certified.
See DFLASH_FUSION_INTEGRATION.md for evidence and scope.
```

---

```text
Implement fixed-cohort GLM53 TP8 batching with independent row commits

Share resident weights/KV pages; batch routed experts and TP on B*8 rows.
Collect up to four requests with capacity-aware admission and independent
EOS/cancellation. Keep attention/draft row-wise and sensitive GEMMs Q8
to preserve B1 rounding. Use explicit capture stream to avoid aux aliasing.

Experiments (node09 A100 TP8, short prompts, capacity=8192):
Seven model cases passed all 8 ranks, B2/B3/B4 128-token outputs exact vs B1;
mixed lengths, permutation, cancellation, anchor EOS and B1 return passed.
Real strategy queue B2/B3/B4 48-token outputs and cancellation passed all
8 ranks, no client errors. Full-batch max-rank round means:
model B2/B3/B4=76.08/111.89/146.09ms (35/34/34 samples);
strategy=74.45/109.38/142.82ms (11/9/9 samples).
Exclude first round, prefill, graph capture and queue/control overhead.
HTTP and full production 144K capacity startup not tested; service stopped.
Fixed cohort, no mid-cohort admission or batched prefix reuse.
Evidence: service_audit/batching_114dd0d/summary.json
```

---

```text
Batch GLM53 attention projections and DFlash compute

A100 TP8 complete-round B1/B2/B3/B4 rank-max means: 43.73/62.27/78.30/96.00ms; samples 30/30/32/34 after first 6 rounds. Strategy B2/B3/B4: 61.66/77.00/93.75ms. Historical B4 model/strategy: 146.09/142.82ms (not interleaved A/B).
Controlled B4 graph probe attention 1.218->0.565ms, draft 12.939->3.667ms, verify 125.070->85.223ms. Excludes prefill, capture and queue latency.
Eight model cases, B1-4 same-batch repeat determinism, strategy/cancel, sparse-index isolation and convolution boundary tests passed eight ranks. B1 probe logits exact; B>1 long greedy sequences differ. Real B4 same-input shadow top1 96.875-100%, mean KL .000849-.018644; no broad quality certification.
Joint B*8 attention/index projections, TP gathers/reduction, router/shared/head GEMMs and DFlash. Sparse KV and commits remain request-local, experts per-route GEMV, fixed cohorts. Capacity 8192 tested; 144K/full-model long context and HTTP untested.
Evidence: service_audit/joint_batch_6f477f3/summary.json and installed/. Details in docs/glm53_batching.md.
```

---

```text
Fix CUDA capture stream aliasing and standard API template mismatch

Reference GLM52 explicit capture streams. Verify uses high-priority root separated from normal MoE side streams; DFlash/append explicitly capture on their warmup streams. Use repository template matching thinking-off parser; complete models schema.

Validation: 5 CPU tests passed; 8 B4-to-B1/cancel cycles passed; 8/8 reasoning samples; both tool roundtrips; OpenAI SDK list/chat/stream; 30/30 load requests, no error logs.

C1/2/4/8 aggregate HTTP throughput: 48.02/60.56/65.87/62.36 tok/s for 256-token workload. Prefix semantics changed, not a kernel speedup comparison. CUDA probe reproduced implicit root aliases at 31/63/95; high-priority root had none. Evidence: service_audit/api_fix; details in docs/GLM53_SERVICE_FIX.md.
```

---

```text
Implement resident GLM53 batching with commit-point boarding

Reuse B1-B4 graphs across epochs; compact page maps without KV copies.
Add epoch page leases, all-rank boarding and initial/active cancellation.

Final TP8 resident and queue regressions PASS on eight ranks; HTTP chat,
streaming, multiturn, reasoning and tool audit PASS. Final restored-service
RPC late boarding, cancellation and 4K-to-2K reuse PASS; 144K startup PASS.

Same API workload, two repeats, mean per-request decode tokens/s excluding
TTFT: C1 137.68->109.40; C2 36.58->87.05; C4 19.75->54.11;
C8 18.19->47.40. E2E aggregate C4 65.87->157.45 tokens/s.
Historical baseline 22a46d2, not interleaved A/B. Eight-request epochs
required zero new Verify captures. Singleton decode regression remains.

Prefill stays eager; leases reclaimed at epoch end. Maximum-context and
many-bucket stress and independent HF quality parity remain unverified.
Evidence: docs/glm53_batching.md and service_audit/resident_boarding.
```

---

```text
Remove context buckets with live-length scoring and frozen batch graphs

TP8 resident/queue pass all 8 ranks; verify graph count stays 4 across contexts. Public API: 58 audit records, 8 reasoning cases, tool roundtrips and 1/2/4/8 clients pass.

Same-payload RPC B1/B2/B3/B4 mean decode tok/s: 70.500/58.259/45.108/35.798 -> 72.015/59.276/45.977/35.849. All 10 outputs token-exact; single-run samples, B4 gain within noise.

Leaf TopK 50 oracle cases and dynamic scores through 1M pass. 1M full-model memory/performance NOT tested. Prefix-cache lease integration withdrawn after cold/warm mismatch; prior cache behavior retained.

Evidence: /mnt/data2/kw/glm53_int4_tp8/service_audit/rectify; local untracked report: reports/context_bucket_rectification.md (repository policy excludes Markdown).
```

---

```text
Align GLM53 prefill cache lifecycle and service controls with v41f

Centralize restore/prefill/commit/store in Generator.prefill shared by
single, fixed-batch and resident requests. Preserve cold prefixes across
batch epochs; invalidate only aliased resident state. Keep GLM geometry
separate from unchanged ColdCache/HostArena. Align readiness, semantic-stop
fallback and strict max_tokens checks.

Experiments on node09 TP8 A100 (service_audit/align_v41f):
Candidate CPU suite 16 passed; merged control contract rerun 11 passed.
Resident/queue passed all 8 ranks, queue errors empty; cache geometry passed.
4000-token warm prompt hit 3999; isolation 8/8; generated-code execution 2/2.
API lifecycle 8 cohort/single/cancel/recovery cycles passed (60 records).
Strict semantic baseline 14/18 vs candidate 12/18: same arithmetic/logic
failures; two extra exact-format failures contain correct trace values
plus explanation. Preserve observations, do not claim accuracy improvement.

User selects aligned implementation as canonical production baseline.
Post-merge restart/smoke is recorded separately. No operator/weight changes.
```

---

```text
Cancel non-streaming generation on HTTP disconnect

Share an idempotent late-handle relay across both APIs and response modes.
Offload shielded cancellation RPC with independent worker capacity; log
failures without skipping local cleanup. Keep GPU/cache/epoch leases unchanged.

Validation: baseline 16 CPU tests; now 31 passing including ASGI routing,
late binding, failed RPC, event-loop responsiveness and full worker limiter.
Live TP8: both non-streaming APIs disconnect -> cancelled terminal;
B2 cancel-one/survivor completes; both APIs normal SSE and SSE cancellation.
Audit: service_audit/cancel_nonstream. Prefill cancellation points unchanged.
```

---

```text
Bound GLM53 prefill chunks and gate readiness on full-capacity execution

Use 8192-token chunks without reducing request capacity. Exercise cold B1-B4 leased contexts plus 16 generated tokens before readiness; check zero prefix hits, fixed graph count and released epoch state.

TP8 isolated B1-B4 passed on all eight ranks; peak allocated 73.58746 GiB. Deployed startup produced all 32 rank/batch records. Standard streaming API: 11 requests passed including 131072-token prompt, prefix reuse and four concurrent requests. 128K first/repeat TTFT 48.665/2.722 s (39/131071 cached tokens); 2684 first/repeat 1.840/0.297 s.

Scope: fixes reproduced 12K-chunk OOM; startup coverage is not a proof for every dynamic allocation pattern. Short-prefill dispatch optimization and latency targets remain unresolved.
```

---

```text
Use decode MoE below 128 rows and grouped prefill otherwise

Unify scratch allocation, GU preparation and dispatch at 128 actual rows; remove inline compressed prefill fallback. Keep cache, attention and batch policy unchanged. Syntax and diff checks pass. Prior 512-path oracle and real-text AB do not validate this new boundary; no new GPU benchmark run per user request.
```

---

```text
Keep prefill request lengths and strides runtime in 13 Triton kernels

Disable value/alignment scalar specialization without changing routing or math. CPU metadata guard passes; grouped GU/down and index scorer compile offline for SM80. No new GPU benchmark or numerical oracle run.
```

---

```text
Overlap prefix cache writeback with compute using bounded stream-local transactions
```

---

```text
Restore reference decode host, control and wall timing through existing logs
```

---

```text
Reuse split MLA for Q8 decode: 32K API host step 83.3 to 45.8ms; validate FP32 oracle and B4 smoke
```

---

```text
Handle physical cold-arena pressure with eviction and transactional admission rejection
```

---

```text
Store public cold KV once across TP ranks; restore via pipelined NCCL broadcast

Keep genuine draft shards local; agree on common prefix before restore collectives. Unique payload 768 to 117 KiB/token. Ten lifecycle regressions and eight-rank exact restore/eviction tests pass. Ten 64K API requests pass under eviction pressure; cold load 0.684-0.691s to 0.123-0.129s.
```

---

```text
Align GLM53 reasoning API controls with native chat template

Preserve low/high/max effort, accept output_config and adaptive thinking, reject unsupported controls without silent remapping. Restore model-shipped template. 44 CPU tests and both live API protocols in blocking/streaming modes pass.
```
