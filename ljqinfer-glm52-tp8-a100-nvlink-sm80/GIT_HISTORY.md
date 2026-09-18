# Development history

按开发顺序排列的完整提交消息。

```text
prefill freeze: slim engine + operator pin + 4-tier TPS baseline

- model.py (700L) public API: prefill(cache_start)->hidden; lm_head outside
- Frozen operators (not a menu): TC attn cold/cached-inplace, moe v10 scatter,
  special_moe v5, q8 cublas dense, q6 embed/head
- Verified: full-text SEMANTIC_OK; 0/8/32/48+8K TPS medians recorded in model.py
- Pin Attention SO under prebuilt/; archive/ and .torch_ext/ intentionally excluded
- decode_path eager Q1-3 smoke PASS (graph/MTP still TODO) — decode work starts from here
```

---

```text
decode: Q1 CUDA-graph path hits 32tps via special local-route

- Resident fp16 expert cache for special MoE (L8/75/76/77 x TP8)
- Graph-safe special FFN: route + index_select + bmm (no CSR dispatch in graph)
- Capture all FFN layers including special; drop eager special threads
- Q1: median 0.031s / 32.03 tps (target >=27 PASS)
- Q2: median 0.044s / step 22.9 tps, token 45.8 tps
- Progress notes in PROGRESS_SP_LR.md
```

---

```text
MTP: blk.78 nextn draft (mtp_forward T>=1, mtp_kv, prompt-shift prefill); accept@1=0.75/32-step greedy smoke
```

---

```text
bench: large-block prefill tps w/ hidden out + MTP layer overhead (2033/2034/1435 vs base 2199/2015/1467; mtp +0.18s/8k; expandable_segments fixes 32k OOM)
```

---

```text
decode.py: clean rewrite of decode graph path (Q1 32tps / Q2 22.4step-s verified); model.py switch imports
```

---

```text
decode: fill MTP hidden (base residuals [Q,D]) in-graph each step; zero speed cost (Q1 32.1tps / Q2 22.3step-s); Q1 numerics exact
```

---

```text
fix(decode): route phase-graph replays onto capture streams (rt.streams). Root cause of Q2 NaN: replays ran on default streams, NCCL on side streams, zero stream ordering -> all_reduce raced graph writes. Entry/exit default<->side wait_stream. Q2 now finite 100+ steps. Cost: Q1 32.1->27.3, Q2 22.3->19.8 step/s (racy overlap removed).
```

---

```text
mtp greedy verify: advance/retract/base_hidden compute API + greedy driver; decode logits widened to [Q,V]; probe: row-causal OK, Q1 27.5 steps/s kept
```

---

```text
mtp draft graph: blk.78 single-layer phase graphs (T=1/2) + mtp_step/ensure_mtp_graphs; e2e 13.1->21.4 tps (accept .81), draft ~45ms->graph replay; capture before mtp_kv prime (warm writes slots)
```

---

```text
attn mmvq: T=2 specialized kernels (t2/dual_t2/rms_t2/grouped_t2), numerics verified vs old ext, e2e neutral
```

---

```text
MONO decode: in-graph one-shot peer allreduce (78-layer single graph/rank, -3~4ms/step)
```

---

```text
decode attention: fuse KV postprocess and cache write
```

---

```text
decode mmvq: stage Q2 inputs in shared memory
```

---

```text
decode moe: specialize shared Q2 gate/up warps
```

---

```text
decode attention: share Q2 KV scan by effective length
```

---

```text
archive optional Q8 routed-MoE experiment
```

---

```text
fix MTP KV cache overallocation
```

---

```text
add model execution and strategy layer APIs
```

---

```text
strategy: add chained cold KV cache reuse
```

---

```text
strategy: record cache and prefill timing metrics
```

---

```text
fix: release prefill cache before decode graph capture
```

---

```text
feat: add typed inference events and cache metrics
```

---

```text
perf: reduce prefill thread dispatch overhead
```

---

```text
chore: organize experiments and project artifacts
```

---

```text
feat: introduce identity-paged KV cache ownership
```

---

```text
feat: support shuffled paged prefill
```

---

```text
feat: complete identity-paged decode wiring
```

---

```text
feat: add batched multi-sequence prefill
```

---

```text
feat: add batched MTP Q2 decode path
```

---

```text
feat: capture stable batched Q2 decode graph
```

---

```text
perf: make batched Q2 dense FFN graph-safe
```

---

```text
extend small-token MoE kernels to T127
```

---

```text
Optimize shared Q4 gate-up kernel
```

---

```text
fix: restore MTP base hidden contract
```

---

```text
refactor cold KV cache to pinned-memory backend
```

---

```text
optimize cold KV restore over NVLink
```

---

```text
fix: bound long-prefill attention and special MoE memory
```

---

```text
feat(prefill): real paged prefill MLA kernel (v3) wired into paged append path; drop gather+22GB workspace; per-device smem opt-in
```

---

```text
docs+probe: mark paged_prefill_mla as the sole prefill attention path; add offline probe kernel
```

---

```text
paged_prefill_mla: even-division KV load (8 thr/row x 9 uint4), 1.53x on 32K append

Load loop rewritten so each row of the KV tile is covered by exactly 8 threads
x 9 uint4 with no remainder, removing the uneven 2.25-pass/32-lane pattern and
its tail divergence. Tail-block zeroing, per-device attribute setup and the
EMBED guard are unchanged.

Verified offline at the true shape (H=8, per-rank 8 heads):
  bare8k     10.662ms -> 9.954ms
  append32k  111.230ms -> 72.699ms  (1.53x)
Bit-identical to previous kernel (max abs diff = 0) and still matching a torch
reference (6.1e-5 / 9.5e-7).
```

---

```text
paged_prefill_mla: cp.async KV tile load (global->shared, no register round trip)

append32k 72.70 -> 65.92 ms on the H=8 bench (1.687x cumulative vs the
111.23 ms baseline); bare 8K 9.95 -> 9.45 ms. Bit-identical to the
previous kernel (vsPROD max=0) and within 9.5e-7 of the torch reference.
```

---

```text
paged_prefill_mla v9: split K tile cp.async into 2 groups, overlap tail DMA with first-half S matmul (65.92ms -> 62.97ms, bit-identical output)
```

---

```text
paged_prefill_mla v10: keep softmax scores in registers, drop SMEM round-trip (62.97ms -> 57.19ms, bit-identical)
```

---

```text
perf(paged_prefill_mla): install v43 split-K (NSPLIT=2) as default kernel

append32k 50.159ms -> 49.041ms (-1.12ms, -2.2%) on A100.
Root cause of the gain: 1024 blocks / 108 SMs = 9.48 -> 10 waves (94.8% eff);
splitting KV in 2 gives 2048 blocks = 18.96 -> 19 waves (99.8% eff).
Adds a second-pass reduction kernel with log-sum-exp rescaling.

NOTE: the file previously shipped here was an older WARPS=8 build with no HPB,
i.e. production was NOT running the best measured kernel. Fixed.
Workspace: launcher allocates NSPLIT*Q*H*L floats (~1.07GB transient at 32k).
Validated: TORCHmax=9.537e-07 mean=3.04e-08, vsPROD max=9.537e-07.
Backup of previous file: paged_prefill_mla.cu.bak_pre_v43
Full experiment log (11 falsified axes): PPM_OPT_FINDINGS.md
```

---

```text
docs: 64k/72k regression, workspace correction, bank-conflict criterion fix; KV layout front closed
```

---

```text
ppm: v49/v50/v51 BQ-shrink experiment (disproved); findings round14
```

---

```text
paged_prefill_mla: v56 - HPB=1/BQ=64 single-head-per-block + fix divergent __syncthreads in S stage

- S stage: move c/sact decl out of branch, put cp.async.wait_group 0 + __syncthreads()
  on the all-warp path. Previously the barrier sat inside if(w<MT*(BK/16)) which
  deadlocked/zeroed output for any BK != 64.
- Retune tile: HPB 2->1, BQ 32->64 (one head per block, 16 warps on one head).
- append32k 49.04ms -> 47.61ms (-2.9%), bare8k 7.83ms. Max abs diff vs prev prod 9.5e-07.
- Searched reuse axis (BQ*HPB = 64/80/96): 47.61 / 47.72 / 64.73 -> BQ64/BK64 optimal.
```

---

```text
docs: round15 findings - reverse BQ axis, divergent barrier bug, v56 shipped
```

---

```text
chore: archive 54 dead ppm variants + stale prod baks; keep only v56 as the single live paged MLA kernel
```

---

```text
prefill: route tc_mla fast path only on fresh prefill (K0==0)

The zero-copy tc_mla fast path is now gated on K0==0. End-to-end A/B on the
real model (TP=8, 2 interleaved rounds, fresh tokens each run) shows the
fast path only wins when there is no existing KV:

  fresh 32k    : tc 1988 vs paged 1817 tps  (+9.4%)  -> take tc
  append 32k+8k: tc  720 vs paged  926 tps  (-22%)   -> keep paged

On append tc_mla densely materializes the full [H,BQ,K0+T] score matrix and
computes the causally-dead prefix blocks that paged_prefill_mla skips (0.56x
of the work is wasted at 32k+8k). Earlier microbenchmarks claimed a uniform
1.47x win because they had no causal block skipping.

Default config (no env) now measures 1967 tps fresh / 923-939 tps append.
LJQ_PAGED_TC=0 still force-disables the fast path for A/B.
```

---

```text
prefill: enable tc_mla fast path for identity append
```

---

```text
docs: add project memory and correct prefill routing notes
```

---

```text
docs: forbid flash MLA on production prefill path
```

---

```text
docs: correct paged prefill ownership description
```

---

```text
docs: record existing batched prefill routing
```

---

```text
runtime: forbid environment-selected operator variants
```

---

```text
runtime: centralize fixed kernel bindings
```

---

```text
runtime: restore dedicated decode attention binding
```

---

```text
runtime: remove stale prebuilt attention binary
```

---

```text
prefill: keep KV append on paged attention
```

---

```text
docs: consolidate project memory
```

---

```text
dev: add transactional kernel hot swap
```

---

```text
dev: preserve production devd capacity
```

---

```text
docs: record live kernel swap qualification
```

---

```text
perf: make cold KV blocks layer contiguous
```

---

```text
docs: record cold KV restore optimization
```

---

```text
fix: synchronize prefill results before return
```

---

```text
docs: define synchronous prefill boundary
```

---

```text
chore: freeze and document prefill state
```

---

```text
docs: record unequal cached prefill result
```

---

```text
feat: graph batched Q2 decode
```

---

```text
milestone: stable batched decode and MTP stage
```

---

```text
feat: preserve FlashMLA Nova-B2 candidate
```

---

```text
Optimize peer all-reduce for decode
```

---

```text
route long B2 decode through FlashMLA Nova
```

---

```text
optimize B2Q2 special MoE with pointer batched GEMM
```

---

```text
decode: improve Q6 LM-head warp geometry
```

---

```text
decode: fuse exact B2Q2 residual add-copy
```

---

```text
docs: record residual add-copy cold acceptance
```

---

```text
decode: unify B1 and B2 on Nova attention
```

---

```text
promote Nova kernels and prepare batch scheduler
```

---

```text
clarify model-driven batch scheduler limits
```

---

```text
strategy: batch concurrent requests with cancellation
```

---

```text
fix scheduler boarding and bound CUDA residency
```

---

```text
keep decode graphs resident and cap total tokens
```

---

```text
chunk prefill and pin CUDA graph Q8 weights
```

---

```text
support partial radix cold KV caching
```

---

```text
report cold cache hits by tokens only
```

---

```text
wire chat template thinking controls through the service layer
```

---

```text
Split inference engine from API service
```

---

```text
fix q6 lookup above 65535 tokens
```

---

```text
Optimize bit-exact decode MoE routing
```

---

```text
add verified TP8 service launcher
```

---

```text
GOLDEN TP8: verified wcache startup and production E2E

Recovery point: if a future optimization or deployment fails, roll back to this commit.
```

---

```text
add reusable server black-box acceptance suite
```

---

```text
restore TP8 peer metadata broadcast operator
```

---

```text
decode: enable TP8 fused metadata-broadcast leaf in production graphs
```

---

```text
decode: revert to local-route unfused MoE path (fastest measured)

Long steady-state strategy A/B (B2, 8192 fake input ids, 2048 forced
output, post-prefill decode wall time, warmup + 2 formal rounds each):
  OLD local-route unfused      : 101.98 / 102.67 s  (40.03 tok/s agg)  <- this
  bcast + fused leaf (45c09f0) : 105.54 / 102.97 s  (-1.8%), tail 116.27 s
  V3 local-route + fused leaf  : 108.38 / 112.44 s  (-7.9%)
Conclusions: rank0-route broadcast adds a per-layer cross-rank spin
barrier for compute that was free in parallel; the fused leaf kernel
itself is slower than the separated production kernels. Operator
sources stay in-repo for reference; production graphs use the old path.
Black-box acceptance suite: 6/6 PASS after revert.
```

---

```text
refactor: 4-layer layout (ops/model/strategy/server) + tests; import-path-only rewrites; prune non-deps (archive/docs/benchmarks/tools-dev/md). E2E 6/6 pass
```

---

```text
harden: constants-only (no argparse/env), gitignore cutlass, prune dead code in decode_path (1553->1235); E2E 6/6 PASS
```

---

```text
refactor: split model orchestration into focused modules
```

---

```text
decode: prebuild and freeze static graph residency
```

---

```text
model: retire legacy KV API and add stress coverage
```

---

```text
refactor(model): remove legacy generation entrypoints
```

---

```text
refactor(model): collapse single-row decode wrapper
```

---

```text
refactor(model): name batch backends explicitly
```

---

```text
refactor(model): remove dead batch FFN path
```

---

```text
refactor(model): tighten execution facade
```

---

```text
refactor(model): unify prepared batch validation
```

---

```text
refactor(model): centralize input state cleanup
```

---

```text
refactor(model): remove unused rope mirror
```

---

```text
refactor(model): remove unused batch lm head
```

---

```text
fix(model): protect resident decode graphs
```

---

```text
refactor(model): remove obsolete batch prefill
```

---

```text
refactor(model): remove obsolete batch prefill blocks
```

---

```text
refactor(model): remove dead decode arguments
```

---

```text
refactor(model): trim unused helper inputs
```

---

```text
refactor(model): make rank-zero helpers explicit
```

---

```text
refactor(model): simplify rank-zero final stage
```

---

```text
refactor(model): flatten MONO capture path
```

---

```text
refactor(model): remove dead decode graph marker
```

---

```text
refactor(model): inline pack capture helper
```

---

```text
refactor(model): collapse eager rank shells
```

---

```text
refactor(model): centralize reset page cleanup
```

---

```text
refactor(model): reuse request page cleanup
```

---

```text
refactor(model): remove redundant load page reset
```

---

```text
refactor(model): remove unused tensor membership hook
```

---

```text
refactor(model): remove unused special layer constant
```

---

```text
Revert "refactor(model): remove unused special layer constant"

This reverts commit 102e6ca349d7211c64ca450a7f6741835bdf5afc.
```

---

```text
refactor(model): remove unused execution properties
```

---

```text
refactor(model): inline ffn layer selection
```

---

```text
refactor(model): remove unused batch offsets property
```

---

```text
refactor(model): trim unused KV load result field
```

---

```text
refactor(model): inline execution device access
```

---

```text
refactor(model): remove unused model path plumbing
```

---

```text
refactor(decode): mark graph lifecycle sections
```

---

```text
refactor(decode): remove self-import indirection
```

---

```text
refactor(decode): unify MTP execution entry
```

---

```text
refactor(decode): unify base decode entry
```

---

```text
refactor(decode): unify graph rollback release
```

---

```text
refactor(decode): simplify rollback cache cleanup
```

---

```text
refactor(decode): remove redundant graph mode
```

---

```text
refactor(decode): remove obsolete buffer fallback
```

---

```text
refactor(decode): reuse lm head phase helpers
```

---

```text
refactor(decode): reuse _rank_ffn in static decode body
```

---

```text
refactor(decode): reuse rank embed lookup helper
```

---

```text
refactor(decode): reuse rank attention helper
```

---

```text
refactor(model): unify batched decode orchestration
```

---

```text
refactor(model): remove superseded decode modules
```

---

```text
refactor(model): move batch graph backend out of orchestration
```

---

```text
refactor(model): streamline q2 decode flow
```

---

```text
refactor(model): unify q2 graph dispatch contract
```

---

```text
refactor(model): use one eager rmsnorm
```

---

```text
refactor(model): remove dead state and buffers
```

---

```text
refactor(model): separate KV pool and workspace semantics
```

---

```text
fix(decode): bind Nova attention in production graphs
```

---

```text
refactor(ops): make selected operator build and load deterministic
```

---

```text
perf(decode): fuse residual add with FFN norm
```

---

```text
perf(decode): pack IQ dequant loads
```

---

```text
perf(decode): batch fused RMS Q8 projections
```

---

```text
optimize shared expert with unified TN kernels
```

---

```text
optimize B512 attention fragment residency
```

---

```text
Revert "optimize B512 attention fragment residency"

This reverts commit e6dac55dcd11a74dbed595a356089c58ecb9c261.
```

---

```text
Revert "optimize shared expert with unified TN kernels"

This reverts commit b5346c3594abfc31b2bcc335aa967e317369a1f8.
```

---

```text
Revert "perf(decode): batch fused RMS Q8 projections"

This reverts commit a9cf0670db7720ed6c959359216e6599c3c5dec9.
```

---

```text
Revert "perf(decode): pack IQ dequant loads"

This reverts commit 5347b5e0713b92b1f6be4d970c3d1b454d2fa4c1.
```

---

```text
Revert "perf(decode): fuse residual add with FFN norm"

This reverts commit cc7290709a330206934121f83288038d128bb00e.
```

---

```text
fix(ops): preserve global prefill provider visibility
```

---

```text
test(server): require exact short-sentence output
```

---

```text
Revert "Revert "perf(decode): pack IQ dequant loads""

This reverts commit 71ee1a3c75b54d47e3a757fe4e45d55e1c667d4a.
```

---

```text
Revert "Revert "perf(decode): batch fused RMS Q8 projections""

This reverts commit 24e7795dde2251deb691ee753a0e6344ce10aa74.
```

---

```text
Revert "Revert "optimize shared expert with unified TN kernels""

This reverts commit 894d95eb375cf9a17d4ad574915e957dc309c28e.
```

---

```text
Revert "Revert "optimize B512 attention fragment residency""

This reverts commit 91da03ba41d5d9b8d6297fc1a47555584762667c.
```

---

```text
fix(decode): restore B2 verifier correctness
```

---

```text
fix(decode): restore correct B2 batch attention ABI
```

---

```text
refactor(decode): freeze stable batched Q2 attention ABI
```

---

```text
refactor(decode): unify batch-agnostic attention ABI
```

---

```text
fix(ops): drop RTLD_GLOBAL promotion of prefill_attn to stop symbol interposition

prefill_attn and decode_attn .so both export identical T symbols
(q8_mmvq_dual_t2_kernel / dual_forward_out / rms helpers). Promoting the
prefill module with RTLD_GLOBAL let the dynamic linker interpose decode's
calls onto the stale prefill copies, causing the cold-start bistable
kv_a-zero bug (7/12 cold runs bad). decode_attn loads standalone with
RTLD_NOW|RTLD_LOCAL (all undefined symbols resolved by libtorch), so the
promotion was never needed.

Verified: cold diag x12 all pass; test_decode_attn_abi 4 passed;
test_native_bn_attn_random_ab AB_PASS 1000/1000 failures=0.
```

---

```text
test(contract): symbol-isolation gate for ops artifacts

Three machine-enforced rules preventing recurrence of the RTLD_GLOBAL
interposition bug (6dacc9f masked it, 83c2477 fixed it):
1. RTLD_GLOBAL banned in ops/ python code.
2. Every selected artifact must load standalone (RTLD_NOW|RTLD_LOCAL).
3. No NEW duplicate exported T symbols across selected .so beyond the
   recorded baseline (tests/symbol_overlap_baseline.json).

Both runs: 3 passed (first run seeds baseline, second enforces it).
```

---

```text
decode_attn: unify B=1/Bn dispatch into single native Bn path (B<=16, Tmax=32)

- nova_decode_attn.cpp: Tmax 4->32; paged_batch_k0 accepts B in [1,16]
- decode_attn.py: drop B==1 fork, always dispatch _decode_attn_bn_impl
  (forward_rank_cached_inplace_tc_k0 kept as A/B reference entry)
- ABI freeze re-baselined (signature unchanged, body-only change)
Verified: 7 pytest green; random A/B vs tc_k0 B=1/2/4/8 (900 cases,
edge k0, real tp8 weights) max_out=6e-5 max_kv=0; bench B=1 parity
(184.5us bn vs 184.6us tc_k0).
```

---

```text
ops: remove legacy B1/B2 decode attention paths

Delete dead forked decode entries now that the unified Bn operator serves all batch sizes:
- nova_decode_attn.cpp: drop forward_rank_cached_inplace_tc_k0 (legacy B=1) and forward_rank_paged_batch_q2 (legacy fixed-B=2) plus their m.def bindings
- decode_attn.py: drop unused _decode_attn_b1_impl below the permanent ABI
- operator_selection.py: drop nova_b512/nova_b512_tn recipes; staticfrag ABI narrows to forward_rank_paged_batch_k0
- tests: ABI gate now locks the Bn-only surface and forbids deleted symbols; random AB gate compares fused B=2 against two B=1 calls of the same Bn operator (batch invariance)

Gates after deletion: build_selected OK, abi+symbol-isolation 8 passed, random AB 1000 cases failures=0 (max_out=1.2e-4, max_kv=0), server e2e 6/6 PASS.
```

---

```text
cleanup: remove dead fork code from 7 op files (e2e 6/6)

Removed unused legacy entry points (forward_rank/_tc/_cached_tc/_cached_inplace_tc_k0,
batch_q2 paths, paged_kv_gather pybind, etc.) across decode_attn_orch, nova_decode_attn,
decode_special_moe, decode_moe_route_q13, peer_ar_os, q8_cublas, q8_linear.

Note: initial automated removal wrongly deleted the paged_kv_scatter_cuda call inside
production forward_rank_paged_inplace_tc (prefill KV never written -> garbage output);
restored the call and its forward decl. Verified: rebuild + restart + server e2e 6/6.
```

---

```text
tp8: quantize special MoE and fix KV pool at 200K
```

---

```text
perf(prefill): make MoE scatter deterministic
```

---

```text
fix(prefill): release provider-private q8 weight cache
```

---

```text
refactor(decode): parameterize batch orchestration by query width
```

---

```text
refactor(decode): fail fast when mono capture fails
```

---

```text
refactor(decode): remove base phase graph fallback
```

---

```text
refactor(decode): share base and batch graph shell leaves
```

---

```text
refactor(decode): share base and batch attention ffn leaves
```

---

```text
refactor(decode): make batch decode shape explicit
```

---

```text
refactor(decode): unify graph shape contract
```

---

```text
test(decode): track shared attention leaf boundary
```

---

```text
feat(decode): expand MTP verification to B1Q4
```

---

```text
perf(decode): keep recursive MTP drafts on device
```

---

```text
fix decode MoE explicit workspace ABI
```

---

```text
feat: add isolated decode operator laboratory
```

---

```text
lab: evaluate MoE grouped decode candidates
```

---

```text
operator-lab: add fused dense gate-up candidate
```

---

```text
operator-lab: retain reports and remove experiment debris
```

---

```text
operator-lab: record T24 decode Amdahl analysis
```

---

```text
perf(attn): share Q4 paged KV scans
```

---

```text
operator-lab: retain rejected RMS and router reports

Validation recorded in reports:
- attention RMS + dual Q8 candidate: bitwise correct on 8 real TP8 layers and graph-safe, but 0.264815 ms vs 0.218949 ms (20.95% slower); rejected
- warp top-k router: bitwise across T=1..24 and graph-safe, but only 0.935 us/layer at T=4 (<0.12% decode ceiling); rejected
- no production operator selection or runtime path changed by this commit
```

---

```text
perf(attn): shard replicated MLA projections across TP8

Split replicated q_a/kv_a output rows across TP ranks, gather the projected latents with a graph-safe peer collective, then run the remaining attention path.

Validation:
- B1Q4 cold Engine startup and CUDA graph capture: PASS
- base B1Q4 plus MTP Q1..Q4 graph residency: PASS
- 128-token Q4 MTP generation: PASS (27.50 emitted tok/s in the measured run)
- fixed real-chat semantic suite: 8/8 PASS in English and Chinese
- repeated 96-token generation within candidate: bitwise token-identical
- full-vs-row-shard q_a/kv_a projection comparison: same result on candidate and HEAD, max abs diff <= 9.77e-4
- same-host cold-process base graph A/B: 50.370 ms -> 47.814 ms (-5.07%)
- same-host public advance A/B: 51.804 ms -> 49.249 ms (-4.93%)
- Python compile and git diff --check: PASS
- post-test GPU memory: 0 MiB on all 8 GPUs

Boundary:
- candidate and HEAD can diverge after near-tied logits because Q8 FP16 tile-order noise is amplified autoregressively; each tree is deterministic and both produced semantically valid text
- validated production shape is B1Q4; B1Q6 and batched Q6 are not covered by this commit
```

---

```text
decode: bring up fixed-width B1Q6 verify

Extend the decode-only attention, grouped Q8 MMVQ, fused RMS/RoPE, KV append, graph residency, and MTP prime contracts from Q4 to Q6. Q5/Q6 grouped Q8 uses the established Q4 leaf plus Q1/Q2 tails. Lock the rebuilt decode-attention artifact (recipe bbe34264bcbb6b3f, sha256 d216839740eb6fa4...).

Validation:
- Real TP8 leaf gate: Q6 vs 6xQ1 KV max abs 0; output max abs 1.22e-4 across six weight/length boundary cases.
- Cold startup: resident MONO B1Q6 and MTP Q1..Q6 graphs all captured and replayed; 64-token B1Q6 smoke 3.918 s (16.33 tok/s).
- Runtime gate: exact 256 tokens, 80 steps, widths 1..6, 181/400 accepted (45.25%), both accept/reject paths, 8.14 s (31.45 tok/s), MTP tail invariant base+4.
- Semantic cold-start gate: 8/8 answers passed (Paris, 4, East, Beijing, 8, Tokyo, 150 km, 9).
- Same-code Q4 control: 256 tokens, 82 steps, 176/246 accepted (71.54%), 6.678 s (38.34 tok/s). This stage establishes Q6 correctness; acceptance/performance optimization remains follow-up work.
- pytest: 8 passed; py_compile and lock/artifact hash checks passed.
```

---

```text
perf(moe): cache selected routed Down experts for B1Q6

Materialize frozen K24 routed-expert Down rows before CUDA graph capture
and use a graph-safe fused IQ gate/up plus cached/packed Down decode leaf.
Keep special MoE and MTP paths unchanged; the new operator role owns both
selected-row dequantization and decode, while the existing IQ lock record
remains byte-for-byte unchanged.

Validation:
- explicit builder published down_cache e5db05e3931ad707 (ABI:
  dequant_selected, decode); all 13 lock artifacts passed SHA checks
- cold no-injection startup built 568 cache/map pairs (71 layers x 8 ranks),
  4.992 GiB/rank; captured Base B1Q6 and unchanged MTP Q1..Q6 graphs
- 128-token Q6 E2E passed: 47 Base steps, median 63.610880 ms and mean
  64.102667 ms vs frozen baseline median 64.201729 ms; median improvement
  0.590849 ms (0.9203%)
- semantic suite passed 8/8 consecutive requests; final free memory was
  3.666870 GiB on rank0 and 4.994995 GiB on ranks1-7 (>=3 GiB gate)
- operator/symbol isolation suite: 13 passed; py_compile, frozen-table,
  CUDA provenance, lock/ABI/hash, diff-check, and GPU-clean gates passed
```

---

```text
perf(decode): overlap shared and routed MoE branches

Base B1Q6 dual-stream CUDA graph: run shared expert on a persistent per-rank auxiliary stream while route+routed expert remain on the rank stream, then join before the original combine order. Dense, special, and MTP FFN paths remain unchanged.

Tests: compileall and diff-check pass; original decode_ffn_rank and _mtp_ffn AST-exact. TP8 Base-Q6 DOT graphs non-empty (6961 nodes/rank). Fresh 128-token A/B control median 63.623680 ms/step; candidate repeats 61.160448 and 61.175297 ms/step (3.87%/3.85% reduction), with identical token head/tail. Semantic suite 8/8 passed; allocated memory stable across prompts.
```

---

```text
perf(mtp): fuse replay communication and reuse quantized Down

Reuse per-rank MTP Down dequantization across the two FFN projections. Add graph-capturable TP8 bf16 gather and fp16 root broadcast primitives, and capture embedding, peer exchange, MTP body, and both peer allreduces as one CUDA graph per rank. Keep the LM-head NCCL reduction and output-store graph unchanged.

Validation:
- Explicit builder published special_quant sha256 7ae5c79fc41ef06e... and peer_ar sha256 20cae57b56c3f8d1...; runtime loaded the locked artifacts and all 14 peer ABI symbols.
- Cold TP8 startup captured fused resident MTP Q1..Q6 graphs; formal output token head/tail and 82 draft accepts matched the prior baseline.
- Steady draft-chain median improved 8.32618 -> 7.12098 ms (-14.47%); versus the pre-Down baseline 8.56087 ms, total improvement is 16.82%. Q1 replay improved 1.36192 -> 1.10592 ms and Q6 2.02547 -> 1.70803 ms.
- 256-token runtime gate passed widths 1..6 with 149 accepts and 386 rejects, exact token count, and both paths exercised.
- Semantic suite passed 8/8 prompts; 15 pytest contract/isolation tests passed; py_compile, lock/hash, ABI, and diff-check gates passed.

Limit: optimized path is the resident TP8 MTP Q1..Q6 decode family; LM-head reduction remains outside the fused per-rank graphs.
```

---

```text
perf(mtp): overlap special MoE and parallelize exact Q3 gate-up

Overlap MTP shared and routed expert branches on per-rank CUDA streams, and expose row/branch parallelism in Q3 Gate/Up while preserving the legacy 24-K-block dot reduction order bit-for-bit.

Correctness: real-weight A/B is bitwise exact and deterministic for T=1,2,4,6,12,24; graph replay remains input-sensitive. Cold Q6 E2E reproduced the same 128-token head/tail and restored 82 accepted drafts. Semantic suite passed 8/8. Contract/identity/symbol/MTP tests: 9 passed.

Performance: Q3 Gate/Up speedups at T=1/2/4/6/12/24 were 3.25x/2.77x/1.87x/1.43x/1.27x/1.40x. Q6 draft-chain median improved from about 6.85 ms to 5.63 ms. B1 latency remains covered; flattened B2Q6/B4Q6 non-attention shapes were validated at T=12/24.
```

---

```text
docs(moe): reject static hot Gate-Up cache

Tests: real layer-32 Q6 weights; K12 hot/cold correctness finite with max_abs 7.73e-5; paired median at 50% hit 0.377856 ms vs production 0.359424 ms (5.1% slower); all-hit 0.264192 ms vs 0.327680 ms. Sequential and side-stream paths tested. Production dispatch unchanged; experimental sources/builds removed.
```

---

```text
docs(q6): reject native attention and dense TC layout

Acceptance summary:
- Native T6 Q8 was bit-exact but 12.47% slower at the leaf gate.
- Pair-shared Flash was numerically bounded (max 2.4414e-4) but Base-Q6 E2E improved only 61.4851 to 61.2751 ms (1.00343x), so rejected.
- Real Q6 routing used 48 distinct experts in all 71 captured MoE layers.
- Optimistic one-launch IQ4 WMMA proxy measured median direct/TC 0.26501x across real-weight layers 3/20/40/60, far below the 1.3x gate; IQ3 layout work rejected.
- Runtime selection, operator artifacts, lock files, and model ABI remain unchanged; ignored candidate trees and temporary scripts were removed.
```

---

```text
docs(decode): reject Q6 LM-head multi-T and map Bx graph gap

Acceptance summary:
- Verified startup freezes only base[B1Q6] and mtp[Q1..Q6]; current batch orchestration serially reuses B1 graphs and has no resident B2/B4 Base graph.
- Graph-safe Base-Q6 replay measured 60.1302 ms, with the mono layer body at 59.2637 ms (98.56%).
- The complete LM suffix measured 2.1759 ms (3.62%); Q6 LM matvec itself measured 0.9673 ms (1.61%).
- Both isolated multi-T Q6 candidates were bit-exact to the selected row loop.
- Register-resident T6 achieved only 0.392x; shared-weight T6 achieved 0.841x, violating the B1 hard gate.
- Shared-weight T12 was 0.987x and T24 only 1.080x, too small to matter end to end.
- Reject LM-head integration; MTP logic, runtime dispatch, ABI, selected artifacts, and lock remain unchanged.
- Temporary scripts, logs, and candidate build caches were removed.
```

---

```text
operator-lab: reject residual RMSNorm fusion

Validation: real-weight CUDA-graph leaf gate passed (T6 31.744us -> 19.456us, 1.632x; max abs 3.05e-5; x bit-exact; alloc/peak delta 0). Independent cold TP8 fixed-input Base-Q6 graphs, 300 replays: 61.278721ms -> 60.958208ms, only 0.526%/0.320513ms. Real 192-token output diverged at token index 23 and fixed logits hashes differed. Rejected; no production dispatch/artifact/lock change. B2Q6/B4Q6 not run because B1Q6 structural gate failed.
```

---

```text
b2q6: batch-2 MTP Q6 decode + prefill pagetable identity gate fix

- Fix: ops/decode_attn_orch.cpp prefill fastpath treated any local page
  table as identity; now gated on pt.size(0)==pool.size(0). B2 per-request
  local page tables no longer write/read wrong KV pages. B1 keeps full-pool
  identity table -> TC fastpath unchanged.
- model/{batch_decode,decode_backend,model,model_api}.py: batch-2 Q6 path.
- tests/test_decode_attn_abi.py: contract updated (public decode_attn ABI
  now only at MTP leaf; base decode uses projected private ABI by design).

Test conclusions (2026-08-14, node09 8xGPU tp8):
- pytest: 21/21 passed.
- single-step B2 vs B1: KV cache and logits bitwise equal.
- cold E2E 2 prompts x 64 tokens: B2 matches serial B1 bitwise for first
  16/23 tokens, then diverges only at near-tie logits (thinking/looking);
  both outputs coherent and semantically equivalent; hashes reproducible
  across runs (deterministic). Benign batch-shape reduction-order effect,
  not a KV bug.
- duplicate-row B2 probe: both rows bitwise identical for all 64 tokens.
- semantic8: 8/8 pass (en/zh factual + word problems).
- TPS (64-tok short run): B1 decode 38.6 tok/s, B2 aggregate 41.3 tok/s.
```

---

```text
ops/mmvq: batch rms flat-launch for B2Q6; keep per-b loop for t/grouped/dual

E2E-profiled decision (B2Q6, ms/round aggregate, prof window):
- rms_tn_matrix: flat <12,4> wins 49.9 -> 44.8, KEPT
- t (o_proj):    flat t<12> regresses 72.0 -> 85.6, reverted to per-b 4+2 loop
- grouped:       flat <12> regresses 38.0 -> 42.6, reverted to per-b loop
- dual: Q<=1 in production, never triggers; parity in micro-bench, reverted
Net B2 mmvq family: 165.6 -> 160.5; B1 unchanged (88.3).
Micro-benchmarks favored flat everywhere but E2E reversed for t/grouped;
gating decided strictly on E2E profile.

Pitfall fixed during dev: disabling the outer (B>1&&Q>2) gate dropped B2Q6
into the legacy grid.z t2 path which only computes 2 rows per request
(silently wrong). Gates now disable only the inner flat launch, keeping
the per-b loop.

Tests:
- Operator equality (standalone build, mmvq_bind): ALL torch.equal vs
  legacy loop reference, incl. B2Q6 forward/dual/rms/grouped paths.
- E2E margin probe b2: argmax 4583 == historical b2_next; b1 regression:
  identical top5/tie values to history.
- Kernel profile b1/b2: B1 kernel set byte-identical to pre-change.
```

---

```text
metrics: surface MTP decode_steps/accepted_tokens end-to-end

Wire batch_decode stats through ModelExecution.generate_batch ->
Strategy end event -> RemoteStrategy -> ServiceLayer._stats so API
responses and server logs expose:
  decode_steps, accepted_tokens, mtp_accepted_per_step, decode_tokens_per_step

Files:
  model/model_api.py          pass optional stats dict into generate_mtp_batch
  strategy/strategy.py        capture per-row accepts on end event
  strategy/remote_strategy.py copy end-event decode fields onto handle
  server/service.py           merge decode_stats + derived rates into metrics

Test conclusions (node09 8xA100, B1/B2 Q6 greedy MTP, post-restart):
  - B1 short probe: steps=39, accepted_tokens=61, mtp_acc/step=1.564,
    decode_tokens_per_step=2.538, decode_tps≈38.0; text coherent
  - B2 concurrent probe: row0 mtp_acc/step=1.303 tok/step=2.273 tps≈20.0;
    row1 mtp_acc/step=1.455 tok/step=2.333 tps≈20.5; both fields present
  - Live client traffic (ctx 3k-29k): mtp_acc/step mostly 1.7-2.3,
    decode_tps 20-31; cache hit path OK; fields appear in [ljqinfer] metrics
  - No behavior change to decode path; compile clean; CRLF of model_api preserved

Not pushed.
```

---

```text
feat(moe): promote dual-row down_cache decode (~0.91x SHORT step)

Replace ops/moe_q6_down_cache_v1.cu hot path with dual-row GU/Down
kernels (warp computes 2 rows + float4 x). Leaf T=6 graph ~0.81x;
live SHORT step ~0.907x vs pre-change baseline. Keep same decode ABI.
Candidate lab + REPORT under ops/operator_lab/candidates/moe_down_cache_fast_v1.
```

---

```text
feat(attn): promote Q6 triple-pair flash MLA W16/S4

Leaf long-KV (T=6, maxabs=0): 8k/16k/32k ratio 0.79/0.72/0.72; 4k 0.88.
E2E B1Q6 step_ms: short ~0.985x, mid ~0.960x vs stock; no crash.
Lock decode_attn recipe bd8a85978d6e0473.
```

---

```text
fix(decode): rollback flash Q6 W16/S4; keep MoE dual-row

e2e same-script B1Q6 (real prefill, decode_candidates ms/verify):
- full stock: 12k=105.3ms, 30k=137.0ms
- flash=stock + MoE fast: 12k=100.5ms (0.955x), 30k=131.9ms (0.963x)
- both fast (pre-rollback): 12k=144.1ms (1.37x), 30k=261.3ms (1.91x)

Conclusion: MoE dual-row is a real ~4-5% win; flash W16/S4 caused severe
long-ctx e2e regression despite leaf 8k+ wins. Revert decode_attn/source to
stock; leave down_cache on dual-row. Evidence under
ops/operator_lab/candidates/{ab_*,quick_e2e_*,flash_rollback_*}.
```

---

```text
docs(ops): strict flash leaf autopsy - k0=0 gate was empty-scan

Old bench_leaf used k0=0 so kend=1 regardless of nk; 8k+ wins were
overhead A/B not long-KV. Strict leaf (real k0, cap=204800, B1/B2)
still shows leaf win at 12k/30k, but e2e both-fast regressed hard
(12k 1.37x, 30k 1.91x). Keep flash stock; MoE dual-row remains.
Gate next flash attempts on real k0 + EXECUTION_LEN + long e2e.
```

---

```text
feat(attn): promote flash MLA v2a int4-vectorized mq kernel (strict leaf v2)

Strict leaf v2 (compact capacity + real k0 + L2-bust): 12k 0.761x / 30k 0.918x / 65k 0.923x, max_rel~8e-4.
E2E quick A/B (decode_candidates ms/verify): 12k 98.68 vs stock 100.14 (+1.5%), 30k 129.43 vs 131.86 (+1.8%). No regression.
Change: splitk_k0_mq_kernel main loop int4 vectorized ql/ck loads + half2 rope; smem layout unchanged.
```

---

```text
peer_ar_os: 8-block ar_oneshot (per-lane flags), leaf AR 0.055->0.021ms exact; e2e 12k 98.68->94.55ms, 30k 129.43->124.91ms
```

---

```text
lock: publish peer_ar fast; add 12k Amdahl reports
```

---

```text
flash_mla TC BQ48 rewrite: mma.sp-free HMMA shared-K kernel, NOVA_TC48 switch; e2e verify 30k 131.7->85.8ms, 4k 97->88.8ms
```

---

```text
moe gu: dp4a int8 kernel (NOVA_MOE_DP4A=1) in all three mw copies; 12k verify p50 88.8->77.4ms
```

---

```text
moe gu: remove env switch and legacy mw/grouped kernels; dp4a is the only path
```

---

```text
ops: remove 28 dead __global__ kernels across 4 MoE cu files; verified e2e perf and generation quality
```

---

```text
mtp: fused chain-draft graph batching (B1/B2), MAXH 32, parallel warm to avoid launch-queue deadlock; B2 37.6 tps verified
```

---

```text
decode attn: TC48 v2 batched split-K flash MLA in production path

- flash_mla_tc_bq48.cu: chunk split over device-side kend_max instead of
  pool capacity (CUDA-graph pool made most splits idle; 12k was 347us,
  scaling with pool pages 99/121/180/308us -> constant 94us)
- nova_flash_mla.cu: TC48 batched out_k0 entry, n_split=108/B
- nova_decode_attn.cpp: per-seq loop -> single batched call
- e2e 12k MTP verify: B1 74.6->54.8ms, B2 136.5->93.3ms (steady mean)
- numerics bit-identical vs leaf reference; sentinel accept unchanged
```

---

```text
fix(moe): gu launcher grid.y mismatch left rows 128-255 of routed hidden unwritten

Since adb67bb the single-row gu_iq3_dp4a kernel (row0=blockIdx.y*NWARPS) was
launched with the old dual-row grid (L+NW*2-1)/(NW*2), so only the lower half
of the FFN hidden rows was ever computed in the down_cache decode path (all 75
MoE layers). Fix grid.y to (L+NW-1)/NW in both launchers.

Verified: sentinel test shows all 256 rows written; 12k e2e B1 steady 58.2ms
(was 54.8ms with the half-work bug), B2 100.6ms, generation normal.
```

---

```text
moe down_cache v8: dp4a int8 down proj + gu smem staging (chain -20%, e2e B1 58.2->53.4ms, B2 100.6->90.6ms)
```

---

```text
moe down_cache v9: prmt-LUT iq4nl dequant in dp4a down kernel, down 127to117us, e2e B1 52.9ms B2 90.0ms, bitwise-identical to v8
```

---

```text
decode: finalize TP8 optimizations and clean experiments
```

---

```text
optimize paged prefill MLA for long prefixes
```

---

```text
optimize special MoE prefill projections
```

---

```text
decode: batch grouped Q8 across B1-B4
```

---

```text
decode: batch Q6 lm-head matvec rows
```

---

```text
decode: batch KV cache postprocess rows
```

---

```text
decode: batch plain q8 for larger request groups
```

---

```text
test: establish canonical decode-step benchmark
```

---

```text
decode: batch output projection with cuBLAS GEMM

Use persistent Q8 dequantized weights for the Q=6 output projection and share mutually-exclusive base graph pools across batch shapes to keep B4 resident. Record token trajectory hashes as diagnostics while numerical equivalence remains the correctness gate.
```

---

```text
decode: batch LM head with resident FP16 GEMM

Materialize every TP LM-head shard into persistent FP16 storage before CUDA graph capture, then run both base and MTP LM projection through caller-owned fixed-output cuBLAS GEMM buffers. Fail startup if residency construction is incomplete; do not retain a decode fallback to the Q6 matvec path.
```

---

```text
checkpoint: mark acceptable important milestone

Service-level validation passed for the current optimized decode stack. This is an acceptable and important checkpoint: cold startup, resident CUDA graphs, real HTTP tool use, concurrent batching, cache reuse, and recent long-request prefill/decode behavior were all verified.
```

---

```text
decode: sample MTP verify at temperature one

Use Base-model categorical sampling for the initial token and every MTP verify row. Keep the original argmax path only behind LJQ_MTP_GREEDY=1 for deterministic optimization benchmarks.
```

---

```text
test: cover strategy batches through B4

Extend long-context strategy acceptance through B3 and B4 for cold, exact-hit, and partial-hit KV paths. Update MTP path accounting for Q1-Q6 event widths and count actual accepted tokens.
```

---

```text
peer_ar: NaN-safe argmax (sentinel -inf + isfinite guard on fp16 round-trip)
```

---

```text
Revert "decode: batch LM head with resident FP16 GEMM"

This reverts commit 0c7154084f4fe5bf80f225381294711d41907439.
```

---

```text
Revert "Revert "decode: batch LM head with resident FP16 GEMM""

This reverts commit 4ff7a4dfb53e52606f736dc5f36776d32fcfc1f3.
```

---

```text
fix: capture MTP chain graph in private pool, not default allocator pool

Root cause of the drifting verify-corruption heisenbug after 0c71540:
capture_mtp_chain_graph used bare torch.cuda.graph() without a pool arg,
so its graph-private memory came from a pool interleaved with the default
caching allocator. Runtime eager allocations (which grew after the LM-head
FP16 GEMM residency change) could land in blocks whose addresses the
captured chain graph had baked in, corrupting verify inputs at
process-layout-dependent request lengths. Base graphs already used
_base_graph_pool; chain graphs now share the same private-pool scheme.
```

---

```text
optimize decode MoE route and tail combine
```

---

```text
optimize routed Down cache with exact Q8 DP4A
```

---

```text
checkpoint: full MTP decode below 45 ms

Canonical synchronized decode benchmark (prefix=12288, Q=6, 25 rounds, first 5 discarded): B1 p50 43.170 ms, B2 p50 65.984 ms, B4 p50 110.115 ms. Timing covers Base Q6 verify, sampling, MTP draft chain, commit, and all-TP GPU synchronization; one-time MTP prime is excluded.

Production HTTP E2E validation at 2.4K-3.7K input tokens: seven real streaming requests measured 41.238-42.656 ms per complete decode step; median 41.763 ms. The longest request produced 1311 tokens in 708 steps at 42.656 ms/step and 43.410 tok/s. All seven requests stayed below 45 ms/step.

Q8-DP4A routed Down cache remains active at 2.808 GiB/rank. Server E2E passed 6/6 and runtime logs contained no CUDA, NCCL, OOM, or RuntimeError failures.
```

---

```text
increase KV pool to 210K tokens

Raise EXECUTION_LEN from 200*1024 to 210*1024, giving 105 pages and a 215,040-token TP8 KV capacity. Cold-start deployment completed successfully with Base B1/B2/B4 and MTP Q1-Q6 CUDA graph residency captured; Engine and API health checks passed.
```

---

```text
feat: support safe mid-decode boarding

Add active-row compaction and low-frequency synchronous boarding across resident B1-B4 MTP graphs. Preserve epoch KV leases with append-only pages, report per-row decode steps and true active batch size, and drain cancelled FIFO heads without blocking later work.\n\nValidated on the live TP8 service with cumulative six-request boarding, B4 shrink/expand transitions, exact token closure, and multi-cancel queue draining; CPU contract suite: 14 passed.
```

---

```text
feat: distinguish semantic stop and log full decode step latency

Add a dedicated semantic-eos stop route while retaining legacy cancel compatibility. Track per-row wall-clock time around the complete MTP step and emit structured end reasons, acceptance, tokens/step, latency, and throughput metrics. Runtime validation: five warm tool calls measured complete-step median 47.200 ms (44.085-49.356 ms), accepted 10 tokens over 3 steps, 4.000 output tokens/step.
```

---

```text
feat(server): add OpenAI chat completions API
```
