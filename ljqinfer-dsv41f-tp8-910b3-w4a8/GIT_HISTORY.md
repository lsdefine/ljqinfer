# Development history

按开发顺序排列的完整提交消息。

```text
Bootstrap 910B resource engine and verified EP8 TP8 weight cache
```

---

```text
Add verified shm loading and A2-3 local-affinity startup acceptance
```

---

```text
Add isolated BF16 SwiGLU candidates with explicit backend audit limits
```

---

```text
Add isolated BF16 RMS candidate with recorded rounding limits
```

---

```text
Add bit-exact isolated FP4 scalar decoder and reproducible audits
```

---

```text
Add isolated K32 FP8 activation candidate with bitwise audits
```

---

```text
Add 910B packed prefill GEMM port with FP64-truth ULP audit

E4M3 RNE subnormal step fixed (clamp 121 -> 118 = 2**-9); verified against
native CPU float8 on 8193 samples (0 mismatches, was 6648).
Audit loads the released baseline by file path to avoid self-comparison and
scores device output in BF16 ULPs against unrounded FP64 truth.
```

---

```text
feat(prefill): bind 910B EP8 TP8 eager prefill and cold replay with real-weight evidence
```

---

```text
test(prefill): verify TP8 bounded replay and slot lifecycle at 129 tokens
```

---

```text
perf(prefill): cache FP4 byte lookup and validate short TP8 continuation
```

---

```text
Migrate routed experts to TP8 packed ABI and validate eight-device MoE
```

---

```text
Record all-rank TP cache build and independent byte audit
```

---

```text
Record TP8 snapshot residency, prefill replay and failed startup timing
```

---

```text
Fix vector-tree RMS count mode and integrate restricted prefill dispatch
```

---

```text
Stabilize Engram gate reduction across token-row shapes
```

---

```text
Record fixed-tree regression without deterministic algorithms
```

---

```text
Reuse identical activation quantization for TP8 expert gate and up
```

---

```text
Batch sparse attention movement and softmax while preserving single-query GEMMs
```

---

```text
perf(tp8): batch sparse attention and adapt FP4 MoE to grouped GEMM
```

---

```text
perf(tp8): preserve KV scales and fuse quantization; validate 5.44s 8K
```

---

```text
perf(tp8): fuse routed SwiGLU QDQ and record 15-pair validation
```

---

```text
perf: validate TP8 FP4 unpack block1024 with 15 paired trials
```

---

```text
perf: validate TP8 FP4 bit decode with 15 paired trials
```

---

```text
perf: archive validated TP8 FP4 bit decode block2048
```

---

```text
perf: validate TP8 FP8 bit decode with15 paired trials
```

---

```text
perf: archive validated TP8 grouped FP8 bit decode
```

---

```text
perf: preserve validated TP8 single-token encoder fused GEMV

Direct FP8 decode/reduce without whole-weight BF16 cache; explicit single-token encoder only, retained decoder tail ordinary.
Two independent 15-pair max-rank synchronized forward+finish runs: 1.076647 -> 0.934378 s; 1.060720 -> 0.917701 s; 30/30 wins. Prepared prefix excluded; not whole response latency.
English/Chinese semantics checked, rank IDs agree; English continuation differs. Baseline self-repeat logits drift; no HF equivalence claim.
Source, raw reports/logits, identical-input leaf audit, hash manifest and reproduction README archived. Default runtime unchanged; production integration and baseline nondeterminism remain open.
```

---

```text
perf: integrate instance-scoped encoder GEMV with TP8 evidence
```

---

```text
perf: fuse default activation FP8 QDQ with TP8 validation
```

---

```text
perf: fuse default KV FP8 rounding with TP8 evidence
```

---

```text
perf: integrate validated TP8 MoE FP4 unpack with 60.78% paired reduction

Archive paired and unpatched-native evidence plus FP4 QDQ validation. Native incremental median 1.729283s; not TTFT or subsecond acceptance. Preserve raw source.diff byte-for-byte, including two patch context blank lines flagged by whitespace checking.
```

---

```text
test: retain all hashed TP8 unpack raw logs and archived bytecode

Preserve original evidence bytes; fill 25 manifest entries excluded by generic ignore rules. No runtime changes.
```

---

```text
prefill: fuse NPU FP8 decode and E8M0 scale application

Retain FP32 decode and official wo_a BF16 rounding. CPU and explicit
lookup-table paths unchanged. 31 targeted tests pass.
Real-tokenizer bilingual baseline/integrated generations identical on
all eight ranks: Paris and chu chu wen ti niao.
Evidence: /data/logs/tp8_fp8_real_prompts_20260913_144125.

12288-token empty-Past forward+finish, six alternating warm pairs:
max-rank median 19.89047 -> 19.34735 s (2.73% observed reduction).
Each integrated case hits 3056 fused decodes, baseline zero.
Chunk medians 15.82152 -> 15.58293 s; CED 3.96757 -> 3.73971 s.
Not a sub-second result or controlled sentence speedup.
Full-chunk logits not bitwise identical; baseline A/A also varies.
Earlier short-step 15-pair A/B was exact across 32 cases.
Initial 12k semantic harness had mislabeled hardcoded tokens;
real-tokenizer run supersedes that semantic result.
Timing evidence: /data/logs/tp8_chunk12k_fp8_interleaved_20260913_143316.
```

---

```text
prefill: batch sixteen experts per grouped MoE GEMM

Bound FP4 dequantization to sixteen rather than eight active experts.
Preserve activation QDQ, weighted SwiGLU and TP reduction order.
12288-token empty-Past forward plus CED, 15 alternating pairs:
max-rank median 22.328466 -> 21.599065 s (-3.2667%).
All 15 pairs faster; paired median delta -0.738005 s.
Real-tokenizer bilingual tokens identical across eight ranks/variants.
Candidate hits 40 routed calls per case.
Paired evidence: /data/logs/tp8_group16_candidate_20260913_150909

Unpatched native: all eight ranks complete, two genuine generations,
one warmup plus three measured 12k cases. Per-case peak reset.
Native max-rank median 21.072980 s.
40 routed calls per case, GMM/group counters verified.
Native evidence: /data/logs/tp8_group16_native_20260913_152918
Source SHA256: b1dcdd1fa4351ba86e9352b8c8c43b627df34643ac0dd475367522d012c4c8f8
Wall includes CED, not server TTFT. No subsecond or independent HF
equivalence claim. Previous paired peak was cumulative.
```

---

```text
prefill: fuse four-copy NPU residual expansion with row broadcasts

Bound NPU specialization to BF16 activations, FP32 coefficients, H4/D5120 and 1..12288 rows; retain generic fallback and disable FMA.

Leaf four lengths bitwise equal; 15/15 pairs faster, 12.92148ms -> 5.25993ms. TP8 12k A/B: 12/15 faster, max-rank medians 21.222720s -> 20.939301s, paired median -0.481390s. Peak allocated 61056111104 -> 59261863936 bytes.

Native verification: 8 ranks complete, matching bilingual tokens, 80 expand and 40 routed calls per case; 1 warmup + 3 measurements, median max-rank 21.087811s. Includes CED, not service TTFT; 1s goal not achieved.

Evidence: /data/logs/tp8_expand_model_20260913_161316/decision.json; /data/logs/tp8_expand_native_20260913_162904/native_audit.json; leaf /data/logs/tp8_expand_rowbroadcast_20260913_161111.
```

---

```text
prefill: accelerate bounded NPU selection ID sorting without score perturbation

Use exact float32 ID keys only for caller-proven nonnegative IDs <= 2^24; preserve integer fallback and stable descending-score/ascending-ID ABI.

Validation: CPU/NPU 120 boundary cases; 8-rank 1416-call real12k AA/AB bitwise gate; bilingual generation; native no-monkeypatch run. Clean 15 alternating pairs all faster: maxrank medians 20.721216 -> 16.546623 s (-20.15%), paired delta median -4.189584 s. Scope: 12288-token empty-Past forward plus CED, not TTFT or independent HF equivalence.

Evidence: /data/logs/tp8_select_accept_20260913_172712/audit.json; native tp8_select_native_20260913_171253; boundary tp8_select_bounded_20260913_171150. Earlier isolated 12.291 s is not the accepted native result.
```

---

```text
prefill: fuse sparse KV gather, masking and FP32 materialization into one Ascend kernel

Real 12288-token TP8 run tp8_sparse_fusedgather_model_20260913_185556: 40 bitwise
AB sparse calls per rank on all 8 ranks, bilingual generation unchanged, 3 alternating
pairs all faster, max-rank median 12.004s -> 10.611s. Leaf operator 31.8% faster.
Duplicated SWA/global rows, -1 invalid masking and sink denominator order preserved.
```

---

```text
prefill: fuse Ascend four-copy residual collapse without FP32 intermediates

Sequential FP32 sums preserve eager BF16 rounding; 82 real calls per rank pass bitwise AB on eight ranks. Three alternating max-rank pairs: 10.9443/10.7027, 10.7862/10.7064, 10.8817/10.8057 seconds (native/fused). Short <=128 rows retain eager path. Default-entry bitwise smoke covers 128/129/12288. Forward+CED, not TTFT.
```

---

```text
prefill: keep Engram sign handling on NPU with exact negative-zero semantics

Remove copysign CPU fallback using FP32 integer sign bits. NPU signbit(-0) is false, so it is not an equivalent substitute. Eight ranks pass both real Engram calls bitwise and bilingual generation. Leaf 49152 values 1.461ms -> 0.073ms. Three whole-model pairs included; no stable end-to-end speedup claim.
```

---

```text
prefill: replace grouped MoE advanced indexing with exact index_select

2392 production-input comparisons bitwise equal across 8 ranks. Two-pair ABBA timing is observational; native repeated logits drift remains unresolved, not a whole-model correctness acceptance.
```

---

```text
prefill: batch NPU index selection by 128 queries and use index_select

Three warm max-rank pairs: median 10.754s to 8.471s (-21.23%). Fixed-rank reduction gives exact IDs across 64 real selection calls. Production HCCL unchanged; existing end-to-end logits drift remains unresolved. Baseline and candidate share three CPU sparse test failures.
```

---

```text
docs: pin selection benchmark replay to baseline 14b74b8
```

---

```text
prefill: enlarge NPU sparse and projection tiles; record drift limits
```

---

```text
prefill: reuse V4 tiled fused window attention with explicit reference fallback
```

---

```text
prefill(moe): histc counts, 64-expert dequant groups, runtime-length FP4 unpack

8-rank hot median 5.629s -> 5.291s on the 12288-token prefill. Isolated ABBA
AB shows 5.454 -> 5.323 (2.40%). The unpack kernel specialized on element
count, so each tail group size forced a fresh bisheng compile; N is now a
runtime argument. histc counts verified equal to bincount per geometry.
Quantization, QDQ and TP slice sums are untouched; top1 agreement on one
prompt is not HF numerical acceptance.
```

---

```text
perf(prefill): vectorize H4 Sinkhorn normalization with bounded fallback
```

---

```text
perf(prefill): share NPU index-key banks across 512-query tiles
```

---

```text
perf(prefill): vectorize FP4 MoE unpack with AscendC (-5.85% TP8 latency)

Exact decoded tensors on 1328 real-weight calls; whole-model logits drift remains unqualified. Add asset verification, dispatch contracts and evidence.
```

---

```text
perf(prefill): schedule sparse attention as contiguous 2048-query tiles

The production caller guarantees contiguous query positions and unique
selected IDs, so the sparse path no longer needs the generic adapter's
per-tile re-derivation: a single mask over [window slice | global bank]
feeds one fused attention call per 2048 queries, and the sink fold is
applied from the returned LSE.

Numerics: 8 ranks x 18 layers x 8 chunks = 144 same-input comparisons
against the previous path are bitwise identical (max abs diff 0.0).
Speed (clean 8-rank run, per round max over ranks, median over 6 rounds):
4.157 s -> 3.915 s, -5.8%. Generic inputs (duplicate or unordered IDs,
non-NPU, other head/dim shapes) fall back to try_sparse unchanged.

tests/test_prefill_ops.py + test_prefill_math.py: 8 passed, 3 failed;
the same 3 fail identically at HEAD (Triton CPU-pointer env issue).
```

---

```text
perf(prefill): bypass redundant full-coverage block scoring

12k TP8 isolated AB: slowest-rank median 3.86648s to 3.63036s. Three NPU selection cases matched indices exactly. Full-model outputs finite; existing baseline drift prevents bitwise end-to-end claim. Keep candidate score query geometry and FP4/FP8 formats unchanged.
```

---

```text
perf(prefill): port V4 BF16 full-bank index path with TP callback
```

---

```text
perf(prefill): widen activation QDQ tiles within A2 UB capacity

4096 replaces 512 elements per tile; unchanged FP32 output and arithmetic.
8192 exceeds UB. Leaf 12Kx5120 approx 4.10 -> 1.33ms, 10 cases passed:
/data/logs/tp8_qdq_ubfit_20260914_214132.
Full logits bitwise identical under deterministic mode, 8 ranks x 12 cases,
512/12288 tokens: /data/logs/tp8_qdq_det_20260914_215609 exit0.
Default paired trials: /data/logs/tp8_qdq_full_fix_20260914_214542,
12K approx 3.481 -> 3.204s; three distributed rounds, not 24 independent
samples; first rounds include warm-up. Baseline default logits drift;
deterministic mode removes drift, exact operator cause not isolated.
Regression: /data/apps/ascend-cann9/bin/python -m pytest -q
 tests/test_activation_qdq_dispatch.py : 7 passed.
Reproduce model AB from diagnostic.py in evidence directories under
CANN9/ATB, torchrun --nproc_per_node=8, fresh PREFILL_AUDIT_DIR,
qdq_candidate.py beside harness; old baseline is parent commit.
Unverified: independent HF oracle, other prompts/hardware, service latency.
1.5s target not met. No deterministic production mode or dtype option.
```

---

```text
perf(prefill): broadcast FP8 scales per 32 weights
```

---

```text
prefill: fused Engram gate kernel (Triton) behind shape guard

Replaces the eager fp32 Engram statistics+apply path with a fused Triton
kernel for the canonical prefill contract ([T,4,5120] bf16, T<=12288).
Guard falls back to the eager path for any other dtype/shape/eps/device or
when V41_ENGRAM_FUSED=0.

Verified: single-NPU bitwise equality fused vs eager for T=1/7/4096/12288
with and without mask (max_abs 0, 0 differing of 251658240 elements at
T=12288); 5 guard-rejection cases fall back correctly.
Whole-model 8-rank interleaved A/B: slowest-rank median 2.971632 -> 2.892009 s
(-2.679%), 16/16 pairs faster; bitwise-identical logits/hidden states under
deterministic algorithms.
Known: default-mode run-to-run drift comes from nondeterministic HCCL
all_reduce, not from this kernel; 1.5 s prefill goal still unmet.
```

---

```text
prefill: fuse hyper-connection projection into one bf16 pass (hi/lo split for fp32 weights); 2.954s -> 2.827s
```

---

```text
prefill: expand reads the four residual copies once per token (bitwise-identical); 2.856s -> 2.730s
```

---

```text
prefill: collapse launches one program per token instead of 40 serial walkers (bitwise-identical); 2.707s -> 2.665s
```

---

```text
perf: fuse MoE expert gather into FP4 decode
```

---

```text
feat: add optional native compact routed MoE dataflow
```

---

```text
Add optional native short-token router dataflow and model connector
```

---

```text
Implement packed shared-expert Cube path in existing prefill engine

Decode FP8 tiles on chip, join gate/up activation preparation, and fuse post-TP rounding with routed merge. Local MoE smoke and 16 exact merge cases pass. Eight-rank live-input diagnostic covered 320 MoE segments; candidate max relative L2 0.000171587.

Development checkpoint, NOT full-model numerical acceptance: 64-token warm A/B logits max difference 2.381; reference cross-request drift 0.781 remains unexplained. No long-prompt speed or 1.5s claim. Routed grouped path and other Torch operations remain.
```

---

```text
Revert "Implement packed shared-expert Cube path in existing prefill engine"

This reverts commit 6d8a3d2a086757611784435ac092b21d1b48fca7.
```

---

```text
Optimize layer-0 initial 12K SWA with bounded masked native attention
```

---

```text
Share initial pure-SWA fast path across compatible layers
```

---

```text
Improve initial SWA precision and remove flattened mask division overhead
```

---

```text
Overlap shared expert compute and reduction with routed prefill
```

---

```text
Fuse token-owned MoE combine and widen indexed FP4 decode tiles
```

---

```text
Store routed activation QDQ directly in BF16
```

---

```text
Preserve installed-kernel smoke validation logs
```

---

```text
Eliminate flat halo-span division from initial SWA packing
```

---

```text
Batch prefill RMS normalization within UB budget
```

---

```text
Reuse activation preparation within paired prefill projections
```

---

```text
Overlap Engram row loading and enable fused gate for released epsilon
```

---

```text
Shard Engram gate by tokens after FP32 reduce-scatter
```

---

```text
Use bounded BF16 operand GEMM for measured Engram projection
```

---

```text
Add independent native BnQ6 decode arena and window HC ring primitives
```

---

```text
Fix native decode HC collapse to consume previous sublayer PRE
```

---

```text
Add independent Q6 sliding attention and native arena entry
```

---

```text
Add independent BnQ6 activation QDQ with bit-exact smoke evidence
```

---

```text
Connect native BnQ6 FP8 QDQ unpack and projection plans

CPU smoke: QDQ/unpack exact, BF16 output rel L2 <=2.041e-6; full projection 136-436us on NPU7. Reproduce with runtime/decode/build.sh and packed_linear_smoke. Full model and TP8 untested.
```

---

```text
Add independent BnQ6 RMS and trailing-64 RoPE kernels

Explicit launch block count prevents repeated tail work. RMS eight cases and RoPE twelve cases match CPU oracles. Unified build verified; full attention integration still pending.
```

---

```text
Connect native BnQ6 KV normalization and RoPE to SWA core

Nine CPU-oracle cases pass across B1/B2/B4 and window boundaries. Shadow exact, ring unchanged, repeated output stable. Unified build verified. Partial attention core only: projections, TP communication, MoE and full decode remain unintegrated.
```

---

```text
Connect complete native BnQ6 local SWA attention projections

Fixed-buffer CANN plans independent from prefill and ATen. Six full-dimension structured-weight CPU smoke cases pass, including exact KV shadow and unchanged rings. Includes dense BF16 output projection. Local attention only; TP reduction and full decode integration remain pending.
```

---

```text
Connect native BnQ6 SWA attention to same-stream TP8 reduction
```

---

```text
Connect native HC pre and post to TP8 decode attention sublayer
```

---

```text
Preserve incoming PRE and advance fixed-address decode HC workspace
```

---

```text
Add independent BnQ6 native decode router with FP32 logits
```

---

```text
Verify same-stream decode HC to router handoff across TP8
```

---

```text
Add independent bounded BnQ6 FP4 decode expert chain
```

---

```text
Validate decode experts at canonical TP8 slice geometry
```

---

```text
Connect native decode router and bounded experts with TP8 reduction
```

---

```text
Add independent decode shared expert FP8 plan and smoke evidence
```

---

```text
Wire decode shared side stream and ordered TP8 MoE reductions
```

---

```text
Connect native decode HC pre and post to TP8 MoE sublayer
```

---

```text
Exercise native decode attention and MoE layer handoff on TP8
```

---

```text
feat(decode): bind native BnQ6 layer execution to Engine

Real-weight TP8 layer0 B1/B2 repeat/finite/unpublished checks pass in engine_native_clean_20260918_190107. Deterministic HCCL enabled in startup acceptance. Independent oracle and full-model compressed/Engram/MTP/head/acceptance remain incomplete.
```

---

```text
feat(decode): wire native Engram into BnQ6 Engine blocks

Validated real-weight layers 0/1 on TP8 for B1/B2, four repeats each, exact hashes/rows and unpublished canonical KV. Removed temporary intermediate taps. This is a wiring/determinism checkpoint, not independent layer truth or full-model acceptance.
```

---

```text
decode: integrate native source compression stage with TP8 evidence
```

---

```text
decode: wire native index query into source stage with TP8 numerical evidence
```

---

```text
Integrate native decode index scoring and deterministic CANN plans
```

---

```text
decode: integrate native TP-reduced Top-k selection with padded logical IDs
```

---

```text
decode: integrate native joint SWA and compressed attention into source blocks
```

---

```text
decode: integrate CSA reuse layers with native source storage
```

---

```text
decode: integrate Engram source ordering and validate native 24-layer prefix
```

---

```text
Wire native decode reindex stages and verify forty-layer zero-start execution
```

---

```text
Validate native decode reindex with synthetic long history
```

---

```text
Integrate native BnQ6 vocabulary head with FP64 capture validation
```

---

```text
Measure native BnQ6 full-chain eager latency without intermediate captures
```

---

```text
Load TP8 weights from tmpfs into HBM in 8.7s worst rank

Weight residency is decoupled from compute and made a memory property: the
unit cache, rank manifests, contiguous rank blobs and shared host Engram
tables all live on tmpfs, and any root that resolves onto a disk filesystem is
rejected instead of silently degrading a hot start into a disk crawl.

What changed
- model/shm_layout.py (new): canonical layout under V41_SHM_BASE
  (default /dev/shm/ljqinfer_dsv41f_tp8): wcache/ unit safetensors +
  wcache/rank{r}.json build manifests, rank{r}.bin + rank{r}.json snapshots,
  host/ shared Engram tables. require_memory() resolves /proc/mounts and
  admits only tmpfs/ramfs/hugetlbfs; V41_ALLOW_DISK_CACHE=1 is the unit-test
  escape hatch only.
- model/wcache.py, model/shm_loader.py: WeightCache roots and both the
  snapshot root and the cache root of a hot start are asserted memory-backed.
- manage.py: --cache/--snapshot/--host default to the canonical tmpfs layout;
  new `loadtest` subcommand is the acceptance harness (8 spawned processes,
  barrier-synchronised start so all cards contend simultaneously, timed
  weights-only path, device-side usability probe, NPU memory before/after,
  JSON evidence).
- model/bulk_h2d.py: new `staged` H2D transfer, now the default
  (V41_H2D_MODE=pageable|four_stage|staged). Bounded pinned ring of two
  512 MiB buffers, 16 threads memmove from the tmpfs mmap into the pinned
  buffer, async copy on a side stream overlapped with the next fill. Bounded
  staging keeps host memory flat (1 GiB per rank) instead of the 37.3 GiB per
  rank that full four-stage pinning would need.

Measured on A2-3, 8x 910B (65.46 GB HBM each), CANN9 python3.12, all eight
ranks loading concurrently from tmpfs. Per rank: 935 cache units,
1789 tensors, 40.03 GB device weights (37.3 GiB rank{r}.bin), 202.76 GB
host Engram tables shared read-only.

  weights-in-HBM seconds per rank (rank0..rank7), harness wall clock
  pageable (old default): 19.42 17.43 13.88 30.88 9.68 11.60 18.85 16.83
                          -> worst 30.88s, wall 49.7s, ~9.7 GB/s aggregate
  staged (new default):    8.31  8.10  8.13  7.91 8.43  8.75  7.80  7.86
                          -> worst 8.75s, wall 31.3s, ~34 GB/s aggregate
  staged, re-run as default: 7.73 7.55 7.72 8.69 7.63 7.34 7.46 7.22
                          -> worst 8.69s, usable probe 7.94s on rank0

Requirement was weights usable in HBM within 20s: met with 2.3x margin
(8.69s worst rank), and the old pageable path is kept only as a mode.
After load: allocated 40.03 GB, reserved 40.05 GB, free 25.02 GB per card.
Usability is proven inside the timed process by reading device elements of
sampled tensors (aligner.w1.bias, layers.30.hc_ffn_fn,
vision.patch_embed.proj.weight) after torch.npu.synchronize().

Cache provenance: the 688 existing unit files were hard-linked (zero copy)
into wcache/, and the eight rank manifests were rebuilt from cache hits alone
(135s per rank, no checkpoint read, 935 units each). All eight rebuilt
manifest digests equal source_manifest_sha256 recorded in the existing
rank{r}.json snapshots, so the 299 GB of resident blobs are provably the
blobs those manifests describe.

Reproduce (weights already resident):
  python3 manage.py loadtest --output artifacts/shm_residency.json
Regression: pytest tests/test_wcache.py tests/test_shm_snapshot.py
tests/test_weights.py -> 31 passed.
Note: artifacts/ is gitignored, so the numbers above are the record.
```

---

```text
tests: drop off-engine tests, run prefill ops on NPU

Deleted: audits targeting ops/cann kernels that are never built or loaded
(fp4_unpack, swiglu_weighted, activation_fp8 first generation), the gemm
audit that diffed against a foreign workspace, and the three *_tp8 audits
pinned to the removed /data/wcache root. The shm end-to-end continuation
now lives in smoke/gen_shm_tp8.py: torchrun 8x, weights only from /dev/shm,
hasher rebuilt from released config plus local tokenizer, greedy output
"The capital of France is" -> " Paris. The Eiffel" identical on all ranks,
peak 54.0 GiB per card.

tests/test_prefill_ops.py now pins the default device to npu; on CPU the
Triton kernels refused the pointers, so those cases tested nothing.
```

---

```text
docs: prefill zero-branch rebuild plan (1a/1b/2+3); drop stale audit artifacts

Plan: docs/PREFILL_REBUILD_PLAN.md (contract). Inventory: docs/prefill_op_inventory.md (24 ops).
CED confirmed from tech report: encoder L0-19 + decoder L20-39, decoder global KV projected from H20;
prefill = encode(12288 rows) + project(L20) + bounded replay(128 rows, finish_prefill).
Shape domains are exactly 12288 and 128; all branching moves to build time.
```

---

```text
ops/prefill: rms as single implementation (plan 1a, op 1/16)

From ops/_legacy_prefill/rms_batch.py. Selection ifs removed: 4
  - rows=1 if n==5120 else 4      -> _ROWS table, frozen by build(width)
  - if WEIGHT (kernel constexpr)  -> weight is mandatory, one code path
  - weight is not None (x2, launch+guard) -> gone with the above
Kept as raise-only guards: shape/contiguity/dtype/device. No allocation:
caller owns out=. Tiles come from the width table (kernel tiling, not a
path choice; plan R2). Widths 512/1280/5120 per model/v41_config.json
(dim 5120, q_lora_rank 1280, head_dim 512). Math follows node09
ops/prefill/residual.py rms: FP32 reduce, FP32 gain, single rounding.
```

---

```text
ops/prefill: hc_mix as single implementation (plan 1a, op 2/16)

Sources: ops/_legacy_prefill/hc_project.py + sinkhorn_vector.py + residual.mixes().
Selection ifs removed: 9
  - project(): 5 'return None' off-contract exits that silently fell back to
    the eager F.linear+rsqrt path -> guards that raise (R6, no implicit fallback)
  - mixes(): 'if z is None' eager fallback -> gone with it
  - mixes(): the 11-term 'is this exactly h=4,24,12288,fp32,iters=20' predicate
    that chose fused sinkhorn vs eager -> constants frozen by build(), one path
  - _weight_panels()/_const_f32() data_ptr caches -> panels built once at assembly
  - kernel masks on H -> build() requires power-of-two copies, mask dropped
Also removed one temporary: the rsqrt scaling is folded into _project_kernel,
so the raw [T,32] FP32 logits buffer of the legacy path no longer exists.
Contract: build(fn, scale, base, copies, dim, norm_eps, hc_eps, iters) ->
(project(x, *, out), gates(z, *, pre, post, comb)); caller owns all buffers.
```

---

```text
ops/prefill: residual collapse+expand as single implementations (plan 1a, op 3/16)

Sources: ops/_legacy_prefill/residual.collapse()/expand(), collapse_fused.py,
expand_broadcast.py.
Selection ifs removed: 2 compound predicates (7 and 11 clauses) that chose
between a fused NPU kernel and an eager torch path, plus the 2 lazy imports
they guarded. Off-contract input now raises instead of silently taking the
slow path (R6). Rows are no longer split at 128: one kernel covers 1..12288,
the row count only sets the grid (R5 shape domains stay in the control flow).
expand keeps comb[source,destination] orientation and FP32 accumulation.
Contract: build(copies, dim) -> (collapse(x, pre, *, out),
expand(x, residual, post, comb, *, out)); caller owns all buffers.
Note: node09 fuses collapse with the following RMS; kept separate here, that
fusion belongs to phase 4 (single-op optimisation), not to path compression.
```

---

```text
tests: single-op prefill timing harness; batch hc_mix.gates rows

tests/bench_prefill_ops.py times each new op at the real prefill shapes
(12288 encoder rows, 128 replay rows) and prints effective HBM traffic.
It is a slowness sentinel, not a correctness test.

Measured on one 910B (ms / GB-s), 12288 rows:
  rms[5120] 0.639 / 394   rms[1280] 0.163 / 386   rms[512] 0.163 / 155
  hc_mix.project 0.781 / 645   hc_mix.gates 0.904   collapse 1.235 / 509
  residual.expand 2.931 / 386
128 rows: every op lands at 0.15-0.20 ms, i.e. the kernel launch floor on
this stack is ~0.15 ms; the bounded replay tail will be launch bound, which
is the quantitative case for capturing it into a graph later.

hc_mix.gates went 2.238 -> 0.904 ms by solving R=64 tokens per program
instead of one (the 4x4 sinkhorn left the vector unit idle at num_warps=1).
No selection was introduced: R is frozen by build().
```

---

```text
ops/prefill: activation QDQ as single implementations (plan 1a, op 4/16)

Sources: ops/_legacy_prefill/activation_qdq.py, fp4_qdq.py, quant.py.
Selection ifs removed: 6
  - quant.fp4_roundtrip(): a 5-clause predicate on device/dtype/ndim/block
    chose between the Triton kernel and a pure-torch reference; both are gone,
    one kernel remains and off-contract input raises.
  - if e4m3_scale / else exp2-ceil scale, twice (reference and kernel path)
    -> one tl.constexpr frozen by build(), constant-folded at compile time.
  - output_dtype in (fp32, bf16) and 'if x.numel()' guards -> out= buffer.

Also removed, and this is the substantive part: legacy fp4 did amax/scale/
clamp in torch before the kernel, allocating x.float() (4 bytes per element),
the unflattened view, amax, the scale and the clamped copy - five temporaries
over the full tensor, then a kernel that only re-quantized. The new kernel
computes amax, scale, E2M1 rounding and rescale in one pass with zero
intermediate tensors, which is the no-scratch-space rule from the plan.
E4M3 rounding is shared by both ops as one device function (_e4m3).
```

---

```text
ops/prefill: dense FP8 linear as a single implementation (plan 1a, op 5/16)

Source: ops/_legacy_prefill/gemm.packed_linear() (326-line module).
Selection ifs removed: 11
  - output_tile None -> '1024 if npu else 256' device sniff, twice
  - activation_prepare None -> internal activation_fp8, an optional-callable
    seam that let two call sites quantize differently
  - table None -> per-device table cache dict, plus an 'elif fp4 and shape'
    repair branch that silently rebuilt a caller's table
  - fp4 / fused / else three-way tile loop, where 'fused' depended on a lazy
    'from ops.prefill import fp8_decode' that may or may not import
  - _fused_decode_available() device+module probe (whole function deleted)
FP4 does not live here: routed experts get their own module (op 9).

Contract changes, both deliberate:
  - activation arrives already quantized (ops.prefill.quant.act_fp8). Legacy
    re-quantized the same hidden state once per projection; a layer with q,
    kv and gate paid three times.
  - the dequantized tile lands in an assembler-owned workspace, so nothing is
    allocated at runtime. Legacy allocated a FP32 gather, a scale broadcast
    and a BF16 product per tile per call.
E4M3 and E8M0 are decoded by exponent-bit construction, never exp2.
```

---

```text
ops/prefill: grouped linear as a single implementation (plan 1a, op 6/16)

Source: ops/_legacy_prefill/gemm.grouped_linear() and
grouped_fp8_weight_linear().
Selection ifs removed: 9 (geometry/dtype/device predicates that each had a
torch fallback behind them, the second output_tile device sniff, and the
_fused_decode_available() probe inside the group loop).
The group loop is control flow over a build-time constant, not a selection:
every group takes the identical path. Dequantization is shared with op 5
(gemm._dequant_kernel), so FP8 decode exists once in the tree.
```

---

```text
ops/prefill: rotary embedding as a single in-place implementation (plan 1a, op 7/16)

Sources: ops/_legacy_prefill/attention.rope(), model/prefill_config.rotary_frequencies().

Selection ifs removed: 4
  - inverse=False/True flag inside the op, evaluated per call per layer
    (freqs.conj() or freqs): forward and inverse are now two builds and the
    conjugate is a sign frozen into the sine table.
  - 'while f.ndim < z.ndim: f = f.unsqueeze(-2)' rank-chasing loop, which
    silently accepted any rank and hid layout mistakes.
  - compressed-vs-SWA theta pick stays, but only inside table(), which runs
    once per layer at assembly time, never per call.

Contract change: the rotation is applied in place. Legacy did x.clone() then
overwrote the tail, allocating a full copy of q, of kv and of the attention
output on every layer and copying the untouched prefix for nothing. Every
call site in model/prefill_layer.py rebinds the name it passed in.

Numerics: pairs are interleaved (2j, 2j+1) exactly as view_as_complex read
them; math in FP32, stored back in the tensor dtype, same as legacy.
Positions arrive as an int32 tensor, so replay tails and encoder chunks use
one kernel with different data, not different code.

Not benchmarked yet; it is a pure elementwise pass over the rotated tail.
```

---

```text
ops/prefill: expert routing as a single implementation (plan 1a, op 8/16)

Source: ops/_legacy_prefill/residual.route().

Selection ifs removed: 5
  - score in {softmax, sigmoid, sqrtsoftplus, else raise}: the released
    config says sqrtsoftplus, so that is the only score compiled. A config
    asking for another one fails to build instead of taking a second path.
  - normalize flag and the topk>1 test: both are assembly-time constants.
  - route_weight-style optional bias handling inside the gather.

Two facts checked against model/v41_config.json rather than assumed:
  - the temperature key is gate_temp (1.0), not route_temperature;
  - n_routed_experts is 384, which is NOT a power of two, so the expert axis
    is padded to 512 inside the kernels and the tail is fenced in the top-k
    pass. n_activated_experts=6, route_scale=1.5.

Allocation removed: legacy ran F.linear(x.float(), w.float()), which
materialises a FP32 copy of the entire 12288x5120 activation (250 MB) plus a
FP32 gate weight, then four more torch ops over the logits. The gate matmul
now accumulates in FP32 inside the kernel and applies softplus-sqrt before
the scores reach memory; the only buffer is the assembler's [rows, 512]
score scratch. FP32 accumulation is preserved, so routing decisions are not
downgraded to BF16 logits.

Top-k is six masked argmax passes over 512 lanes, lowest index winning ties
to match torch.topk's stable order.

Not benchmarked yet; no model wiring in this commit.
```

---

```text
ops/prefill: SwiGLU as a single implementation (plan 1a, op 9/16)

Source: ops/_legacy_prefill/residual.swiglu().

Selection ifs removed: 2
  - limit>0 test: V4.1 clamps at 10.0 always, so the clamp is a constexpr.
  - route_weight is None test: a routed expert always carries a weight and a
    shared expert never does, so weighted/unweighted are two builds.

Geometry checked, not assumed: moe_inter_dim=2304, so the rank-local expert
width under TP8 is 288, which no power-of-two block divides. The K loop masks
its tail; a divisibility guard would have rejected the real model.

Allocation removed: legacy did gate.float() and up.float(), two full FP32
copies of the expert tile, before the elementwise work. Promotion is now per
element inside the kernel, and the routing weight multiply is fused in.

Not benchmarked yet; no model wiring in this commit.
```

---

```text
ops/prefill: expert dispatch as a single implementation (plan 1a, op 10/16)

Source: ops/_legacy_prefill/npu_routed.PackedRouted.__call__ (the
argsort/bincount half) and grouped_routed.py.

Selection ifs removed: 4
  - count==0 skip per expert, plus the 'is this rank's slice empty' test
  - the sorted()/argsort stable-vs-fast choice
  - the host-list-vs-device-tensor branch on the counts

Determinism, on purpose: ordering comes from one program per expert scanning
the flat id array with tl.cumsum, not from atomic_add cursors. Atomics would
permute the rows of an expert run between runs, and because expert outputs
are summed in FP32 the layer output would stop being reproducible. The extra
pass over 73728 ids buys bit-identical reruns.

Allocation removed: argsort + bincount + div/mod allocated four tensors per
layer. offsets/token_index/slot_index are now assembler-owned and reused by
all 20 layers.

Debt recorded, not hidden: dispatch() still does one 385-integer
device-to-host read per layer, because the expert loop is host control flow.
A padded fixed capacity per expert is what removes it, and that decision
belongs to the graph stage, not here.

Not benchmarked yet; no model wiring in this commit.
```

---

```text
ops/prefill: routed expert FFN in FP4 as a single implementation (plan 1a, op 11/16)

Sources: ops/_legacy_prefill/npu_routed.py (expert loop), moe_fp4_unpack.py
(nibble decode), model/weights.py (w13/w2 layout).

Selection ifs removed: 6
  - PackedRouted vs DenseRouted vs GroupedRouted class choice per layer
  - the lazy 'import PackedRouted only if fp4' inside the block
  - per-expert 'if count == 0: continue' and the empty-slice test
  - unpack()'s four device/dtype/geometry tests on every call
  - the 'shared expert or routed expert' weight test, now a build argument

Facts pinned from the repo, not assumed:
  w13[E, 2, width, dim/2] uint8, gate=0 up=1, two FP4 nibbles per byte along
  the contraction; w2[E, dim, width/2]; one E8M0 scale per 32 values.
  width = moe_inter_dim/TP = 2304/8 = 288, dim = 5120, E = 384 per rank bank.

The nibble decode keeps the reference bit construction (mantissa 0..7 ->
0,.5,1,1.5,2,3,4,6 by assembling the exponent field, sign from bit 3), so
there is no exp2 and no NPU rounding drift versus the released numbers.

SwiGLU is injected at build time: composition is part of the assembly, so the
op has no branch and no optional argument. All three matmuls write into
assembler-owned scratch; one weight slab is dequantised at a time into a
single shared buffer.

Known debt, deliberately not fixed here: each expert run dequantises three
weight slabs, so a token that routes to 6 experts pays the decode 18 times per
layer. The fix is a resident BF16 expert bank or a real FP4 matmul, and that
belongs to phase 2/3 where the memory budget is decided, not to this op.
```

---

```text
ops/prefill: expert combine as a single implementation (plan 1a, op 12/16)

Source: ops/_legacy_prefill/npu_routed.py (out.index_add_ half).

Selection ifs removed: 3
  - the FP32 accumulator choice (out.float() vs out) inside the expert loop
  - the 'scale by probability here or in swiglu' duplication: the weight is
    folded in by swiglu, so combine takes no scale argument
  - index_add_'s internal deterministic/atomic path choice

Determinism: inside one expert run a token appears at most once, because a
token cannot select the same expert twice. There is no collision, so the
write is load-add-store with no atomic, and the cross-expert accumulation
order is the assembled expert loop order. Same input, same bits, every run.

No allocation: the caller owns the run buffer and the token-major output.
```

---

```text
ops/prefill: TP reduce + add as a single implementation (plan 1a, op 13/16)

Source: model/prefill_build.Parallel.sum() and its call sites in
model/prefill_block.py.

Selection ifs removed: 4
  - world_size>1 test before every collective
  - 'shared expert output exists' test before the add
  - dtype promotion inside x += y (a torch-level implicit temporary)
  - reduce_shared / reduce-here-or-later flag threaded through the block

build() resolves (collective, add) once and returns one of four closures, so
the hot path has no branch at all; the remaining ifs raise on a shape that
does not match the frozen one.

Deliberate: group=None means a single-rank assembly. It is not a fallback -
changing topology requires a rebuild, which is the point.
```

---

```text
ops/prefill: indexer scoring + top-512 as a single implementation (plan 1a, op 14/16)

Sources: ops/_legacy_prefill/v4_index.py, selection.py, candidates.py.

Selection ifs removed: 14
  - _INDEX_TOPK_MODE: five spellings (ref, ref_reduce, ref_topk, kernel,
    kernel_bf16, kernel_unsorted) each giving a different numeric result
  - start_pos==0 vs !=0, repeated in three functions; prefill always starts
    at zero so the causal limit is unconditional
  - select()'s six-way eligibility test that returned None into a streaming
    fallback, plus the positions==arange test that did the same
  - candidates.build's full-coverage shortcut returning a tensor with an
    ad-hoc _fullcoverage_key_count attribute
  - selection.topk's id_upper_bound branch choosing float vs int argsort

Shapes frozen at build: q [tokens, heads, 128], bank [entries, 128], k=512,
ratio and id offset are constants. Score buffer, top-k value and index
buffers are allocated once; the call allocates nothing.

Honest gap, stated rather than hidden: legacy guaranteed 'descending score,
ascending id on exact ties' via a double sort. This version takes torch.topk's
order. FP32 sums of 32 head terms make exact ties rare but not impossible, so
tie order is currently unpinned; pinning costs a second sort and returns when
the indexer becomes one fused kernel.

Score buffer is [tokens, entries] FP32 = 302 MB at 12288 x 6144, the same
footprint legacy allocated per call, now allocated once.

Not run: no test, no numeric check (per instruction). Syntax checked only.
```

---

```text
ops/prefill: sliding-window attention as a single implementation (plan 1a, op 15/16)

Sources: ops/_legacy_prefill/swa_initial.py, window_attention.py,
window_dense_attn.py -- three files for one operation.

Selection ifs removed: 11
  - ENABLED module flag and supported(), a ten-clause predicate whose failure
    silently rerouted the caller to another attention implementation
  - the hard 12288-row shape test: the row count is now a build argument, so
    the 128-row CED/replay domain assembles the same code differently
  - MAIN_SPAN_LIMIT with its 'return None and let the caller gather' exit
  - duplicate-id detection and its second fallback
  - kvc-is-None and multi-key concat branches
  - FP16-vs-BF16 intermediate choice

Kept because it is numerics, not selection: queries and the halo bank are
FP16 (measured normalised Q/KV fit FP16 and keep more mantissa than BF16),
and the sink is folded through the log-sum-exp as
out * exp(-max(s-l,0)) / (1+exp(-|s-l|)), i.e. sigmoid(l-s) written so
neither exponent overflows.

Allocation: bank, boolean mask, FP16 query copy allocated once in build().
The mask depends only on frozen geometry, so it is filled during build and
never recomputed. Per call the only allocations are the two tensors returned
by npu_fused_infer_attention_score, which the CANN op allocates itself.

Not tested: no numerical or timing run in this commit.
```

---

```text
ops/prefill: sparse core attention as a single implementation (plan 1a, op 16/16)

Sources: ops/_legacy_prefill/attention.py (sparse half), sparse_gather.py,
window_dense_attn.py.

Selection ifs removed: 13
  - attention.select's NPU/CUDA/reference three-way, the v4 fast path plus its
    'if fast is not None' second exit, and index_merge.select_direct behind a
    nine-clause admission test
  - _fullcoverage_key_count attribute sniffing on the candidate tensor
  - query_tile chosen by device type and by whether candidates exist
  - id_upper_bound computed only on NPU and only for banks under 2**24
  - the 'if not len(keys): return' empty-bank exit
Selection no longer lives here at all: ops/prefill/indexer.py produces ids and
this file attends to them. A row whose id is -1 is masked, which is how a short
history is expressed -- not by a branch.

Contract: q [T,H,D] bf16, kv pool [N,D] bf16 (MLA single KV head, D=512),
ids [T,512] int32 with -1 padding, sink [H] fp32, out [T,H,D] bf16.

Allocation: the gathered bank is one query tile wide (tile*512*512 halves =
134 MB at tile 256). A full [T,512,512] gather at T=12288 would be 6.4 GB,
so the tile is a build argument, never a runtime decision. Bank, mask and the
FP16 query copy are allocated once in build(); per call only the two tensors
returned by npu_fused_infer_attention_score are allocated, by CANN itself.

Sink folded through the log-sum-exp as in swa_attend:
out = o * exp(-max(s-l,0)) / (1+exp(-|s-l|)).

Not tested: no numerical or timing run in this commit.
```

---

```text
docs: prefill rebuild progress table after step 1a

16 operators, 16 commits, 100 selection ifs removed. Records the debts
(linear_fp8 6x off peak, per-run FP4 weight decode, one D2H per layer in
dispatch, topk tie order) so step 1b does not have to rediscover them.
No operator has been executed yet.
```

---

```text
tests: operator-level check for the first five prefill ops, plus two real fixes

tests/prefill/test_ops_a.py builds each op with released constants, runs it on
one NPU and compares against plain torch. Per-case process, so a compiler
crash isolates.

Results (npu:0, this commit):
  rms[512/1280/5120]   rel 2.1e-3 / 3.0e-3 / 2.6e-3
  residual.collapse    rel 2.2e-3
  residual.expand      rel 3.5e-3
  quant.act_fp8        median rel 1.2e-2 (block-32 e4m3 round trip)
  quant.fp4            median rel 3.6e-2 (e2m1 round trip, as expected)
  swiglu[plain/weighted] rel 2.5e-3
  rope                 rel 1.8e-3

Two fixes the test forced, both physical, neither a behaviour choice:

1. rope.py: the kernel loaded the real part at 2j and the imaginary part at
   2j+1 as two strided loads. triton-ascend aborts on that pattern
   (InterleaveOptimization.cpp:136 assertion, process dumps core). Rewritten
   to one contiguous 2*D2 load, reshape to (R, D2, 2), tl.split / tl.join,
   one contiguous store. Same math, same in-place contract.

2. UB overflow, 192KB per core, not negotiable:
     quant.build   tile 4096 -> 2048  (needed 1990656 bits, 1572864 available)
     swiglu.build  rows 64 kb 512 -> rows 16 kb 256
   Both are build-time defaults, so no runtime selection is introduced.

Three test-side errors found and corrected while doing this, worth recording
because they were all my reference being wrong, not the kernel:
  - swiglu: wrote the gpt-oss form g*sigmoid(1.702g)*(u+1); V4.1 is silu(g)*u
    with the one-sided clamp on gate and the two-sided clamp on up.
  - residual.expand: transposed the combine matrix. Legacy stores
    y[dst] = p[dst]*x + sum_src r[src]*c[src][dst].
  - rope: assumed split halves. Legacy uses view_as_complex on (d/2, 2),
    i.e. interleaved pairs.

Six ops still untested: hc_mix, gemm, grouped, route, dispatch, combine,
add_reduce, moe_gemm_fp4, indexer, swa_attend, core_attend.
```

---

```text
tests: operator-level check for the router, the dispatcher and the projections

tests/prefill/test_ops_b.py, same shape as batch A: released constants, one
NPU, plain torch as the reference. The FP8 case feeds weights that E4M3 holds
exactly (small integers, unit E8M0 scale), so a mismatch there can only be a
layout or decode error.

Results (npu:0, this commit):
  gemm.linear_fp8        rel 1.6e-3
  grouped.bf16           rel 1.7e-3
  route.ids              exact match against torch topk
  route.probs            rel 2.9e-7
  dispatch.order         offsets/token_index/slot_index exact
  dispatch.gather        exact
  combine.scatter_add    exact
  add_reduce.add_only    rel 1.8e-3
  add_reduce.identity    bitwise
  hc_mix.project         rel 3.9e-7
  hc_mix.gates           finite, gate shapes as contracted

Two real defects the tests found, both in ops/prefill:

route: _gate_kernel accumulated into a [R, N] tile but loaded the gate weight
as [PAD, KB], so it never compiled: 384 against 512. With the accumulator
fixed to PAD the tile then asked 2621440 bits of UB against 1572864
available, so the expert axis is now walked in NBLK=128 blocks, each block
stored as it is finished, with kb=128 and tile=32. Fully resident scores were
never possible on this part; the blocked form is the only shape that fits.

dispatch: the histogram pass compared a [NB, PAD] tile, 1024 x 512 int32 =
1639168 bits, again past the 1572864-bit UB ceiling. nb=256 now.

Neither defect is visible by reading: one needs the compiler, the other needs
the real expert count.

Next: moe_gemm_fp4, indexer, swa_attend, core_attend.
```

---

```text
tests: operator-level check for the FP4 expert FFN and the three attentions

tests/prefill/test_ops_c.py, one NPU, plain torch as the reference.

  moe_gemm_fp4.expert   rel 3.20e-03  FP4 weights drawn from the eight E2M1
                                      magnitudes with a unit E8M0 scale, so
                                      only decode order and GEMM layout are
                                      under test
  indexer.ids                         top-k set matches the torch scores
  indexer.causal                      live ids per row == min(topk, (t+1)//ratio)
  swa_attend.out        rel 1.68e-03  window 128, sink through the lse
  core_attend.out       rel 1.67e-03  gathered top-k, -1 masked, sink idem

One real fix, found by the first run:

  moe_gemm_fp4.expert called the swiglu closure without the probs argument
  the closure requires, so every routed FFN run would have raised TypeError.
  expert now takes probs and hands the live slice to the swiglu the assembler
  froze; the stale docstring claiming probs lived in a weight buffer is gone.

Batches A, B and C together: 26 operator cases, 16 of 16 prefill operators
covered, no skips.

Next: 1b, the prefill assembly in model/.
```

---

```text
docs: record the 1a operator test results and the four defects they found
```

---

```text
prefill ops: engram gate and head close stage 1a (plan 4)

The 1a operator list in docs/PREFILL_REBUILD_PLAN.md section 4 names engram_gate
and final_rms+lm_head; neither had been written. What was written instead were
four stage-1b past-facing ops (swa_attend, indexer, core_attend, add_reduce), so
the "16/16" count in the progress note was right in number and wrong in content.
This commit writes the two that were missing; the 1b list shrinks by four.

ops/prefill/engram_gate.py  stats kernel ([T,4,3] FP32 reductions) + fold kernel.
  Numerics follow ops/_legacy_prefill/residual.engram_gate exactly, including the
  sign-bit read that replaces copysign (NPU signbit drops -0). Scratch (stats,
  gate) is caller-owned, so the op allocates nothing. No mask argument: prefill
  never passed one, the decode masked variant is not this op.
ops/prefill/head.py  final RMS into a caller-owned workspace, then the vocab
  shard in BF16 tiles. head.weight is never packed (model/weights.py), so there
  is no FP8 path here and no dequant workspace.

tests/prefill/test_ops_d.py, one NPU, against plain torch:
  engram_gate  T=128 x[128,4,5120] kv[128,25600]  rel 3.281e-07
  head         T=128 d=5120 vocab=16384 tile=4096  rel 2.972e-03 (BF16 matmul)
Both green. Run: python3 tests/prefill/test_ops_d.py [case]
```

---

```text
prefill ops: compress and embed, the last two the assembly needs (plan 6.1)

Walking the released layer stack call by call against ops/prefill showed exactly
two calls with no new-side owner: attention.compress (the CED compressor on the
source layers) and the embedding lookup that model/prefill_build.py kept inline.
candidates.build and selection.topk needed nothing: indexer.py had already
absorbed them. project_global needs nothing: it is gemm plus rope.

compress.py folds each KV pair into one row under a per-element softmax over the
pair and keeps the two-row ring carry the caller owns. The pair may straddle the
chunk edge, so the carry rows are read as group -1 and written by a second
kernel; nothing is allocated and nothing branches at call time.
  Selection ifs removed: 4 (ratio-1 pass-through, the carry-is-None pair, the
  'values came in as BF16' promotion). A ratio-1 view is not built at all.

embed.py writes one rank's vocabulary shard contribution, zero on the rows it
does not own, so the caller's existing add_reduce reproduces the full table.
  Selection ifs removed: 2 (tp==1 shortcut, dtype promotion).

Test: tests/prefill/test_ops_d.py, one NPU, 12/12 green.
  compress start=0/1/7 rel 1.0e-07 / 1.2e-07 / 9.7e-08 against the released
    torch statement, carry values and scores bit-exact
  embed bit-exact against F.embedding over 8 shards of a 4096-row table
  engram_gate 3.3e-07, head 3.0e-03 (unchanged)
Operator count for the prefill path is now 20; the 1a list is closed.
```

---

```text
prefill assembly: freeze the per-layer role table in prefill_config (plan 6.1)

Stage 1b starts by removing the first runtime choice: which kind of layer this
is. The released constants already decide it, so layer_roles() states it once.
Checked against model/v41_config.json on the box: layers 0-1 are swa (ratio 0),
2/8/14 are sources (ratio 2, also index sources), the other fifteen reuse their
source. No prefill layer reindexes and candidate_source_layer is 20, so neither
path exists below layer 20 -- both are guards that raise, not branches.

rotary_frequencies stays: tests/test_prefill_released.py and
tests/released_random.py state the released YaRN rule through it. The assembly
itself uses ops/prefill/rope.table.
```

---

```text
core_attend: attend the union of local window and selected rows

Found while writing the 1b assembly: the 1a core_attend only did the
selected-rows half of ops/_legacy_prefill/attention.py::_sparse_reference,
which softmaxes over window(128 causal rows) + selected(512) + sink together.
Attending to the selected half alone is a different model.

Change: bank width is window+topk; a second gather kernel fills the window
columns from geometric ids (p-(W-1-k), dead below local_start); the global
columns now also mask ix>=(p+1)//ratio and ix>=len(global_kv), which legacy
checked per row. No new op file (ops/ must stay small). Zero runtime if:
window/topk/ratio are build constants, validity is a mask.

Signature: core_attend(q, local_kv, local_start, global_kv, ids, positions,
sink, *, out). Test tests/prefill/test_ops_c.py core rewritten against the
legacy union reference: PASS rel=1.675e-03 tol=3e-2 (256 tokens, W=32,
topk=16, ratio=2, 8 heads, d=128, real NPU).
```

---

```text
prefill_attention: three build-time roles, no runtime branch

Rewrites model/prefill_attention.py against the 1a ops. The old file chose
between swa/sparse/dspark, replay/fresh and ratio/no-ratio inside one
attend(); those ifs are now build-time specialisation: build_window (layers
0-1), build_source (2/8/14: compress+index+union attend) and build_reuse
(the other 15: attend the source layer's store with the ids it chose).
The per-chunk selection is a plain dict passed down by the block, so a
reuse layer cannot silently re-index.

start != 0 raises: the 1a swa_attend/core_attend take no history, and one
prefill chunk is the whole prompt. Chunked prefill is a later step and will
be a different builder, not an if.

Not yet wired into prefill_layer/prefill.py (next commit); syntax and op
signatures checked on the A2, 72 lines.
```

---

```text
prefill_layer: MLA projections with no runtime choice left

Rewrites model/prefill_layer.py on the 1a ops. build_attention serves the
17 layers without a store, build_attention_source the three with one; the
shared tail (rope inverse, grouped wo_a, wo_b, rank reduce) is build_output.
What used to be decided per call is now decided by the builder: which GEMM a
weight needs (lin is a dict of closures, so a packed weight can never be
silently dequantised), whether this rank reduces (add_reduce variant), which
attend the layer owns, and where every intermediate lands (buf, allocated
once; the names are listed at the end of the file).

Not yet run: model/prefill_build.py, which allocates buf and fills lin/ops,
is the next file. Syntax-checked only, 133 lines against 127 before, with
the compressor and indexer paths that the old file guarded by three ifs.
```

---

```text
prefill_block: residual and MoE with the loop over experts only

Rewrites model/prefill_block.py on the 1a ops. build_moe is route ->
dispatch -> per-expert FP4 GEMM -> combine plus the shared experts, with the
routing weight of each entry gathered once into a preallocated buffer
instead of per-expert indexing; the only skip left is an expert that got no
rows, which is absent work, not another path. build_block is gates,
collapse, norm, sublayer, expand, twice; the two expands alternate between
the caller's two residual buffers so the block hands back the buffer it was
given and nothing is copied to keep the hyper-connection copies alive.

Gone with the old file: DenseRouted (a synthetic-model fallback that a real
FP4 deployment must never take), bind_shared_input probing by hasattr, the
expand_workspace present/absent pair, the shared-expert side stream and the
token-count guards. 95 lines against 157, and the dtype rounding points are
the released ones: shared experts rounded to BF16 before the FP32 add, one
rounding of the accumulator at the end.

Not yet run: prefill_build.py must supply lin/ops/buf first.
```

---

```text
prefill_block: reduce the routed half, the experts are TP not EP

weights.py pins the ABI as routed-tp: every rank keeps all 384 experts
and owns one intermediate slice, so the combined rows are a partial sum.
The block now reduces the shared half, rounds it to BF16 the way the
released model does, and hands it to the fused reduce-then-add that
finishes the routed half. Loop bound is the full expert count.
```

---

```text
prefill_buffers: one allocation for the whole chunk, shapes read off the weights

Replaces the per-call torch.empty scattered through the old layer/block:
every activation, scratch, routing index and accumulator prefill touches is
allocated once here. Geometry (heads, q-lora, kv, latent, indexer, expert
width) is taken from layer 0's rank-local weight shapes, so a wrong TP slice
raises here instead of corrupting a kernel; only counts the weights cannot
state (chunk tokens, experts, top-k, hyper-connection copies) come from the
config. GEMM dequant workspaces stay with the builder, keyed by K.
```

---

```text
prefill_linears: one build-time table of the layer's GEMMs

The layer asked lin('name', x) and the old PrefillLinear decided per call
whether the weight was packed FP8 or replicated BF16. That test is now a
build-time question to the checkpoint: a projection that shipped a .scale
gets ops.prefill.gemm bound to a shared [tile, K] dequant workspace, one
per distinct K; everything else gets a pre-transposed BF16 matmul. Short
names map to ABI keys in NAMES, so a missing weight is a KeyError.
```

---

```text
prefill 1b: ops namespaces bind every op at build time

model/prefill_ops.py (116 lines): layer_ops() freezes the attention path
(four RMS widths, five rotations plus the inverse one, fp8/fp4 activation
quantizers at the released block sizes 32/16, the compressor, the grouped
wo_a GEMM, one all-reduce) and block_ops() freezes the two hyper-connection
mixers, residual collapse/expand, router, dispatch/gather, the FP4 routed
expert with its own weighted SwiGLU, combine, the shared SwiGLU and the two
reductions. Geometry is read off the buffers, never recomputed.

Fixes found while binding: wo_a is stored packed [G*n, k], so the group
count is heads*dh//k (not shape[0]); buffers gained ow (wo_a dequant
workspace), the routed expert scratch egate/eup/eact/espace at the rank
slice width from tp_experts.w13, and the router's power-of-two logits row.
prefill_layer.py now names its norms rms_q/rms_kv/rms_c/rms_k and its two
compressed rotations ck_rope/cl_rope, because width and row stride are
frozen into each op.

py_compile clean on all five 1b files.
```

---

```text
engram_gate: take the residual in its [orders, tokens, width] layout

The prefill residual buffer is [copies, tokens, dim] and the n-gram order
axis is the same four copies, so the gate now indexes x/out as
order*M*D + token*D instead of assuming token-major rows. No permute and
no extra copy on the assembly path.

Tested on NPU: tests/prefill/test_ops_d.py 12/12, engram_gate rel=4.0e-08
against the plain-torch statement (tol 2e-2).
```

---

```text
engram_gate: take the order stride from the residual buffer

The gate runs on a token window of the [orders, tokens, width] residual, so
the order axis stride belongs to the whole buffer, not to the window. The
closure now reads x.stride(0) and passes it to both kernels; the guard checks
strides instead of demanding a contiguous tensor.

Tested on NPU (tests/prefill/test_ops_d.py, 14/14 ok):
  engram_gate            rel=4.015e-08
  engram_gate[window]    rel=4.015e-08   window of a 3x larger buffer
  engram_gate[outside]   rel=0            neighbouring tokens untouched
```

---

```text
1b: engram gate + encoder pass assembled (zero runtime branch)

model/prefill_engram.py  build per layer: layers 1/14 get the real gate,
  the other 18 get `passthrough`, so the loop never tests a layer id.
  Geometry read off the weights, not the config: wkv [25600, 768] is a
  contraction-TP slice (the full input is 6144 = 24 hash rows x 256, this
  rank owns 3 rows), so the projection is summed over ranks in FP32 before
  the gate folds it; q/k_weight [4, 5120] give orders and width.
  Runs one 1024-token tile at a time: a full [12288, 25600] projection
  would cost >0.5 GiB of scratch for a per-token op.

model/prefill_buffers.py  + gkv/gkvf/gkvw/gstats/ggate (one tile each).
model/prefill_build.py    each layer is now an (engram gate, block) pair.
model/prefill_forward.py  new: embed (vocab-sharded, summed over ranks --
  exact, every other rank contributes a zero row) then the 20 encoder
  layers. `selections` is a dict the 3 source layers fill and the 15 reuse
  layers read.

Verified: py_compile on all four; engram_gate itself is tested green
(tests/prefill/test_ops_d.py, 14 cases, rel 4e-08).
Not covered yet: layer-20 global KV projection, finish_prefill and replay
depend on past-facing ops (1b op list) that are not written; no 8-card
assembly smoke run yet.
```

---

```text
prefill: fix 3 parity root causes (layer2 final .838->.999, 20-layer tail cos .954)

Root causes found by layer-2 tensor-level parity against /data/ljqinfer_base:

1. rope table dispatch (ops/prefill + model/prefill_build): the compressed-row
   table was built with length tokens//ratio, so compressed rows read past its
   end; length must be `tokens`, and layer_ops must pick the table by ratio.
...[Truncated]...
   16289.66 / 16261.41 / 16225.44). Parity compares run0, so the numbers above
   are unaffected, but this state leak is real and still open.
```

---

```text
prefill: fuse engram gate tail into Triton, pin deterministic HCCL, drop probe

Three changes, each measured.

1. ops/prefill/engram_gate.py: the gate tail (rsqrt x2, dot, abs/clamp_min/sqrt,
   IEEE-sign where, sigmoid) ran as ~7 torch ops allocating ~7 temporaries per
   call on the hot path.  Folded into _gate_kernel, reading the preallocated
   stats scratch and writing the preallocated gate buffer.  Zero runtime
   allocation, 7 fewer kernel launches per engram layer.
   Unit check vs the old torch chain (rows=777, orders=4, width=5120, signed
   dots in [-10,10]): max abs diff 1.192e-07 (1 ulp, from rsqrt/sigmoid impl).
   Boundary that the old code warned about is preserved: dot==0 -> gate
   0.500249981880188 == sigmoid(sqrt(1e-6)), i.e. -0 still reads as negative
   via int32 bitcast, not copysign.

2. smoke/encoder_parity_tp8.py: pin HCCL_DETERMINISTIC=true before init.
   Root cause of the self-reproducibility failure reported in b1b0906 ('same':
   False, run l2 16289/16261/16225): HCCL all_reduce is not order-stable.
   Isolated proof, 8 ranks, same [12288,640] bf16 input all_reduced 6 times:
   default -> 3 distinct hashes; HCCL_DETERMINISTIC=true -> 6 identical.
   It is not state leakage: deep-zeroing every ckv_pool/index_pool/window
   tensor between runs still drifted (16231/16251/16287/16268).  Per-layer
   norms showed divergence already at layer 0 (1324.547/.549/.546/.549,
   rel 1.5e-6 = fp32 add-order), amplifying to 0.3% by layer 19.
   After the pin, all 4 runs are bitwise equal: l2 16222.3955078125,
   absmax 28.375, 'same': True.

3. model/prefill_forward.py: removed the PROBE per-layer tap; it allocated an
   fp32 copy (h.float().norm()) on the hot path every layer.

Parity vs base, 20 layers, 12288 tokens, TAIL=64, unchanged within noise:
cosine 0.9545610547065735 (b1b0906 measured 0.95376), diff max 4.375,
mean 0.1784, want_absmax 11.0625, got_absmax 10.625.

Remaining, known and measured:
- attention block cos 0.9347 is top-512 near-tie flipping (sorted score values
  match at cos 0.9998); needs _score_kernel bf16 accumulation to align.
- runtime allocations left: past.rows() builds arange%ring per read/write,
  core_attend's npu_fused_infer_attention_score has no out=.
- dispatch.py offsets.tolist() is a runtime D2H sync, must go before capture.
- tests/test_prefill_ops.py and 9 others fail collection: they import
  ops.prefill.attention/residual/candidates which do not exist.
```

---

```text
past: make WindowPast.rows() allocation-free via a prebuilt ring table

rows(t0,t1) ran torch.arange(t0,t1,device=npu) % ring on every read and every
write -- two device allocations plus two kernels per attention layer per chunk,
on the hot path, for an index vector that is fully known at build time.

Now __init__ builds ring_index = arange(2*ring) % ring once (to_() moves it),
and rows() returns the view ring_index[t0%ring : t0%ring + (t1-t0)].  Valid
because rows() already asserts 0 <= t1-t0 <= ring, so the slice never runs past
2*ring, and (t0+i)%ring == ((t0%ring)+i)%ring.

Checks: exhaustive CPU equivalence over t0 in [0,400) x n in {0,1,7,63,127,128}
-- 2400 cases, 0 mismatches vs the old arange expression; returned tensor is a
view into ring_index (no allocation).  8-rank encoder parity is bit-identical
to the previous commit: cosine 0.9545610547065735, diff max 4.375, mean
0.17835702002048492, same=True.  No numerical effect whatsoever, this is purely
an allocation removal.
```

---

```text
prefill: restart from the node09 model layer as the working base

WHY: the from-scratch prefill chain (88d1218) reached only the 20 encoder
layers and had diverged in interface (closure `encode` entry, past bound at
build time, no PrefillScratch). Per the new rule -- everything except the
low-level operator math must match node09 -- the model layer is taken back
from node09 verbatim, and our operators will be moved in one at a time.

WHAT:
  model/{prefill_build,prefill_layer,prefill_attention,prefill_block}.py
      <- node09 verbatim (our closure assembly stays in git at 88d1218)
  ops/prefill/*            <- node09 verbatim (27 files)
  ops/prefill_ours/        <- our 20 zero-branch operators, kept for the
                              one-at-a-time move-in; not imported yet

EXPERIMENT (proves this tree runs the full chain end to end):
  GEN_AUDIT_DIR=/data/genaudit_ours torchrun --nproc_per_node=8 \
      smoke/gen_shm_tp8.py
  -> "The Eiffel Tower stands on the Champ de Mars in Paris. Engineers
      debated its height, its iron lattice, and the wind loads it must
      survive across a long century."
  -> peak 54772796928 B (54.77 GB), identical to the node09 tree

  This also validates our past.py rows ring-table change against the full
  40-layer chain: node09's model layer reads it unmodified.

NEXT: move ops/prefill_ours operators in one by one, re-running the
generation above after each move; a move that changes the text is reverted.
```

---

```text
ops/prefill: route every RMSNorm through the single Triton kernel

WHAT: residual.rms no longer chooses an implementation. It used to ask
eleven runtime questions (V41_CANN_RMS env, device, dtype, contiguity, a
width whitelist, epsilon type, four weight checks, a token-count range) and
could land in one of three bodies: ops/cann/rms, ops/prefill/rms_batch, or a
plain torch expression. It now looks up a closure frozen by (width, eps) --
a dict lookup, not a branch -- and calls ops/prefill/rms.py, which does the
FP32 reduction with the weight in FP32 and rounds once on store.

The kernel also gained an empty-tile early return: the generation loop
hands prefill a zero-row batch, and a Triton launch with grid 0 aborts the
device (EE1003 coreDim=0). The torch fallback used to absorb that silently.

EXPERIMENT (8 x Atlas 800T A2, TP8, shm weights):
  GEN_AUDIT_DIR=/data/genaudit_rms torchrun --nproc_per_node=8 \
      smoke/gen_shm_tp8.py
  continuation unchanged: "The Eiffel Tower stands on the Champ de Mars in
  Paris. Engineers debated its height, its iron lattice, and the wind loads
  it must survive across a long century."
  peak 54.753 GB/rank, against 54.773 GB on the node09 rms path -- the
  20 MB is the CANN workspace that is no longer allocated.

TO REVERT: restore ops/prefill/residual.py from af64bf3. Nothing else
depends on this change; ops/prefill/rms.py stays as the operator library.
```

---

```text
ops/prefill: route the hyper-connection mix, collapse and expand through our kernels

WHAT: residual.mixes, residual.collapse and residual.expand no longer pick an
implementation at runtime.

  mixes    asked hc_project for a projection and silently transcribed the
           whole algebra in torch when it answered None, then asked nine more
           questions (device, ndim, copies==4, width==24, row range, three
           dtypes, two shapes, iters==20, hc_eps==1e-6) before it was allowed
           to use the Sinkhorn kernel. Now one frozen (project, gates) pair
           per layer does the projection with a high/low BF16 panel pair and
           keeps the whole Sinkhorn solve in registers.
  collapse fused only when 128 < T <= 12288 -- every shorter CED row used a
           different reduction order in torch. Now one kernel for all T.
  expand   had an eleven-clause gate in front of expand_broadcast and a torch
           twin behind it. Now one kernel.

Guards raise inside the frozen closures; nothing falls back. Empty tiles
return their (already allocated) output without a launch, because a Triton
grid of zero aborts on NPU while the old torch twin tolerated it.

TEST: 8-rank torchrun smoke/gen_shm_tp8.py on shm weights, greedy continuation
of the Eiffel Tower prompt. Text is unchanged from the node09 baseline --
"The Eiffel Tower stands on the Champ de Mars in Paris. Engineers debated its
height, its iron lattice, and the wind loads it must survive across a long
century." (16 occurrences across the audited steps, no exception on any rank).
Peak memory 54.7499 GB vs 54.7528 GB after the RMS move and 54.7728 GB on the
untouched node09 base: the torch twins and their temporaries are gone.

NOTE: ops/prefill/hc_residual.py holds our collapse/expand Triton kernels.
They cannot live in ops/prefill/residual.py because that file is the node09
interface module itself; four of our files (gemm.py, quant.py, residual.py,
__init__.py) collide with node09 names and must be renamed when moved in.
```

---

```text
ops/prefill: route SwiGLU and the engram gate through our kernels

WHAT: residual.swiglu and residual.engram_gate no longer decide anything per
call, and the fused-engram probe is deleted.

  swiglu      clamped in torch and asked, on every call, whether a clamp and
              a routing weight existed. Both are properties of the call site
              (the shared expert is unweighted, the routed experts are
              weighted), so they are now frozen into a closure keyed by
              (width, weighted, limit) and the kernel does silu*up, the clamp
              and the routing multiply in one pass.

  engram_gate called _fused_engram_gate first, which read the V41_ENGRAM_FUSED
              environment variable and then probed seven contract facts
              (device, dtypes, shapes, contiguity, mask). If any probe failed
              it returned None and the whole gate was transcribed again in
              torch: two _fixed_tree_sum reductions, an NPU-specific signbit
              workaround via int32 reinterpretation, a sigmoid and a masked
              fill. That second algebra is gone. One kernel now computes the
              statistics, the signed magnitude and the gated residual. The
              mask argument is rejected instead of silently supported: no
              prefill call site passes one.

  _fixed_tree_sum and _fused_engram_gate are removed with their only callers.

EXPERIMENT (8x910B, TP8, shm weights, smoke/gen_shm_tp8.py):
  continuation unchanged, "the Eiffel Tower ... on the Champ de Mars in
  Paris", matched on all 8 ranks (16 log hits, 0 errors).
  peak_bytes 54752476160 (was 54749941248 before this commit): +2.5MB, the
  engram gate's FP32 statistics and gate scratch, allocated per call for now.
  That temporary is a known debt, to be handed to the caller when prefill
  moves to preallocated buffers.

  residual.py runtime branches: 22 -> 16, and all sixteen that remain are the
  router (score selection) plus the closure-cache lookups.

REVERT: undoing this restores the torch engram transcription and the env
switch; nothing else depends on it.
```

---

```text
ops/prefill: route the MoE router through our kernel

WHAT: residual.route no longer selects a score function per call.

  route  computed the gate in torch and asked, on every call, which of three
         score functions to apply (softmax / sigmoid / sqrtsoftplus) and
         whether to normalise the top-k. The released bank is sqrtsoftplus
         with normalised top-8; that is a property of the checkpoint, not of
         the tokens. The score, the top-k width, the temperature and the
         route scale are now frozen into a closure built once per router
         (keyed on the gate weight and bias pointers), and a request for any
         other score, or for an unnormalised top-k, is refused loudly rather
         than quietly recomputed in torch.

  The kernel is ops/prefill/route.py (gate GEMM + top-k), already unit
  tested; this commit only makes the assembler use it. It is the default
  router of PrefillMoE (prefill_build passes router=None), so this is a live
  path, not a dead one.

TEST: smoke/gen_shm_tp8.py, 8 ranks on the shm snapshot, ports 29665.
  The Eiffel Tower prompt still continues with "... on the Champ de Mars in
  Paris" on all 16 sampled ranks/turns, byte-identical to the text produced
  before this change and to the base tree. No errors.
  peak_bytes 54801354240 (54.80 GiB), up 48 MB from 54.75 GiB: the router
  now owns a [12288, 256] FP32 logits scratch shared by every layer, plus
  per-call ids/probs. That temporary is a known debt, to be handed to the
  caller when prefill moves to preallocated buffers.

CONTRACT: our kernel returns (ids, probs) and emits int32 ids, while the
  assembler's contract is (probs, ids) with int64 ids for npu_routed; the
  wrapper swaps the order and widens the ids. Verified against the checks in
  ops/prefill/npu_routed.py, which reject anything else.

REVERT: undoing this restores the three-way torch score; nothing else
  depends on it.
```

---

```text
chore: drop commit-message scratch files that were staged by mistake
```

---

```text
prefill/gemm: split packed_linear into named FP8/FP4 ops, delete dead paths

Probe-driven cleanup. A probe appended to gemm.py recorded every branch
taken during one full 8-rank released generation (1756 distinct call
signatures, rank0). Facts:

  - FP8 calls took the fused Triton path 100% of the time; the torch
    gather fallback behind _fused_decode_available() never ran.
  - The staged fast path (name == 'layers.1.engram.wkv' with shape
    (12288, 768)) ran 0 times: real chunks are 3151/421/5 rows, so that
    hardcoded geometry was unreachable dead code.
  - The `table=` keyword had no caller anywhere in the tree.

Changes:
  - packed_linear -> packed_linear_fp8 / packed_linear_fp4, selected by
    weight dtype through a dict (a static weight-format property, not a
    runtime probe). Same A8 QDQ, same BF16 output rounding.
  - _fused_decode_available() -> _fp8_decode_module(): the NPU decode
    kernel is now a requirement that raises, not a silent slow path.
    Applied to grouped_fp8_weight_linear's copy of the probe as well.
  - Dropped the staged engram fast path, self.engram_bf16 and `table=`.
  - Kept fp8_table/_e4m3_values (used by fp8_decode.py's Triton kernel)
    and fp4_byte_table (used by grouped_routed.py) -- these are weight
    decode tables, not fallbacks.

if/elif in gemm.py: 42 -> 35. Remaining ones are closure caches, empty
batch early-outs and contract rejections, plus the encoder_gemv and
bind_shared_input row-count splits, which are live control flow.

Test: 8-rank smoke/gen_shm_tp8.py, 16/16 prompts still continue with
"Champ de Mars in Paris"; peak_bytes 54801581568 unchanged (54.80 GB).
```

---

```text
tests: make hc_mix gates a real oracle; drop duplicated CPU-float32 cases

case_hc_mix only asserted finiteness for gates, so the Sinkhorn/pre/post formula
had no NPU reference. Port the float32 oracle into the BF16 case (rel<=6.7e-8).
The two CPU cases it replaces cannot run at all now: hc_mix.project requires
BF16+contiguous and k=copies*dim tiling k_block, so they were asserting on a
geometry the engine never sees. Keep the pure-torch rope round trip.
Wrap the script-style runners of test_ops_a/d in __main__ so pytest can collect
the package without KeyError on sys.argv.
```

---

```text
gemm: activation_fp8 dispatches to the NPU kernel unconditionally

The entry guarded its fused call with five terms (device type, dtype in a
pair, ndim, last-dim non-zero, k % 32) and otherwise fell back to a torch
reference. In the engine the fused kernel took every call, so the reference
existed only for host unit tests that asserted on the probe itself, via fake
SimpleNamespace 'tensors'. Those tests measured the branch, not the engine.

Now: k % 32 stays as a raising contract, the kernel is called directly, and
_activation_fp8_reference plus its only helper _pow2_ceil_ratio are gone.
test_activation_qdq_dispatch.py keeps just the K contract; numerical parity
is tests/prefill/test_ops_a.py::case_quant on real NPU tensors.

Experiment: torchrun 8-rank smoke/gen_shm_tp8.py -> 16/16 ranks emit the
expected 'Champ de Mars in Paris' continuation, peak_bytes 54801354240,
byte-identical to the pre-change baseline. pytest tests/ -> 229 passed,
18 skipped, 0 failed. if-count in ops/prefill/gemm.py 40 -> 39 lines.
```

---

```text
gemm: output_tile is a signature default, not a device probe

packed_linear and grouped_fp8_weight_linear both computed their tile as
1024 if x.device.type == 'npu' else 256. The 256 leg was for host runs that
this engine no longer has, so the tile was always 1024. It is now a keyword
default; the positive-and-32-aligned contract check stays. No caller in
model/, ops/prefill/ or smoke/ passes output_tile, so behaviour is identical.

Experiment: torchrun 8-rank smoke/gen_shm_tp8.py -> 16/16 ranks emit the
expected continuation, peak_bytes 54801354240 (unchanged). pytest -> 229
passed, 18 skipped. if/elif lines in gemm.py 36 -> 32.
```

---

```text
attention: drop the CUDA select_direct leg and the unused _sparse_hybrid

select() guarded a nine-term branch on q.is_cuda that imported
.index_merge.select_direct. ops/prefill/index_merge.py does not exist in this
tree, and this engine runs on Ascend NPUs, so the branch was dead twice over:
the predicate can never be true, and taking it would raise ImportError.

_sparse_hybrid was a second full attention body (31 lines, per-row einsum
concatenation) with no caller anywhere outside its own def; sparse() reaches
only try_scheduled, try_sparse and _sparse_reference.

attention.py 254 -> 216 lines. Verified: pytest tests/ and the 8-rank shm
generation smoke (16/16 'Champ de Mars in Paris' hits) after the change.
```

---

```text
attention: fp8_roundtrip dispatches to the kv_qdq kernel unconditionally

The entry probed device type, dtype, ndim, last-dim and block alignment before
calling the fused kv_qdq kernel, else ran a torch reference. The engine always
took the kernel; the reference served only host tests of the probe itself, and
the kernel had no numeric test at all.

Added tests/prefill/test_ops_a.py::case_kv_roundtrip: NPU BF16, [96,512] and
[256,1536], fused vs plain torch (pow2 block scale from amax/448, e4m3 round,
clamp +-448) -> rel = 0.000e+00 on both, i.e. bit-identical. Only then was the
reference removed, along with the now-unused e4m3_round import.
tests/test_kv_qdq_dispatch.py kept the one real assertion (misaligned block
raises) and dropped the fallback-dispatch cases (fake SimpleNamespace 'npu'
tensors, fp16-goes-to-reference), which asserted on the probe, not the math.

attention.py 214 -> 209 lines, 'if ' 36 -> 35. pytest: 229 passed, 18 skipped.
smoke/gen_shm_tp8.py 8-rank: 16/16 ranks emit 'Champ de Mars in Paris'.
```

---

```text
attention: query tiles are constants, key gather is index_select only

Three tile defaults read 'X if q.device.type == npu else Y' (128/32,
512-or-128/32, 256/64) and the key gather had a third leg using fancy
indexing for host tensors. The host legs are unreachable in this engine, so
the tiles are now plain constants and the gather is index_select for the
candidate case, whole-bank matmul for the dense case. query_tile is None is
kept: it is the signal that the caller allows the scheduled/v4 fast paths,
not a device probe, so it cannot move into the signature.

Verified: pytest 221 passed/18 skipped; gen_shm_tp8 8-rank 16/16 'Champ de
Mars in Paris'. device.type probes in this file: 5 -> 2.
```

---

```text
attention: no device probes left in the file

Four conditions still asked q.device.type == 'npu' before entering the v4
index select, the candidate id-bound read, the shared-bank scoring and
try_sparse. This engine only ever runs on NPU tensors, so the clause was a
constant true that hid the real conditions. Those are kept verbatim:
full-coverage candidate banks, len(keys) <= 2**24+1, the SHARED_SCORE_LIMIT
traffic rule and query_tile is None as the caller's permission for a fast
path. String 'device.type' now appears 0 times in the file.

Verified: pytest 221 passed/18 skipped; gen_shm_tp8 8-rank 16/16 'Champ de
Mars in Paris'.
```

---

```text
v4_index: one kernel path, no mode switch, no _mod shim

_INDEX_TOPK_MODE was hardcoded 'kernel'; its five other spellings (ref,
ref_reduce, ref_topk, kernel_bf16, kernel_unsorted) were unreachable and are
gone with the SimpleNamespace _mod() shim. start_pos and offset were always
0 at the only call site, so both adapter branches collapsed.

Measured (probe build, smoke/gen_shm_tp8.py, 8 cards, 16/16 hits): 1106
select calls total. The device/size guard returned None 0 times -> now a
contract raise. The positions!=arange guard returned None 416 times (decode
steps and continuation chunks) -> kept, it is the real streaming signal.

91 -> 62 lines, if count 13 -> 5 (4 contract raises + 1 streaming signal).
pytest 221 passed/18 skipped, gen_shm_tp8 16/16 hits.
```

---

```text
attention.select: one TP contract check, then the empty-history boundary

v4_index.select raises on degenerate requests since c7e368b, so the
len(keys)==0 early return moved above the call. Two consequences the test
suite pinned down:
  - the empty bank must still yield all -1 ids (legal first token), and
  - the TP contract (total_heads != local heads needs a sum reduction)
    must be validated before that short-circuit, not after it.
So q.shape is unpacked once at the top, the TP check happens once at the
top, and the two duplicated copies of it inside the function are gone.
tests/test_prefill_ops.py::test_empty_history_and_tp_contract covers both.

Found because the earlier runner let `pytest | tail` swallow the exit code
and committed c7e368b with a failing test; the runner now gates on it.
pytest 221 passed/18 skipped/0 failed, gen_shm_tp8 16/16 hits.
```

---

```text
model: drop the dead CUDA workspace scaffolding from the prefill path

cuda_workspaces = parallel is not None and device.type == 'cuda' is
structurally false on this NPU build, and ops/prefill/ contains no
projection_workspace.py, swa_workspace.py, sparse_workspace.py or
expand.py at all, so taking any of those branches would have raised
ImportError. Instrumented smoke/gen_shm_tp8.py (8 cards, 4000+ attend
calls): swa_workspace and sparse_workspace were None every single time.

Removed: the cuda_workspaces flag and its four build sites, the swa/
sparse workspace attributes on PrefillAttention, the two attend
parameters with their three branches, and PrefillBlock.expand_workspace
with its _expand fallback. attend now has a single local/sparse path,
and the swa_initial gate loses its always-true q.device.type check.
model/: prefill_attention 99->84, layer 127->125, block 157->151,
build 138->119 lines.

Verified: pytest 221 passed, 0 failed (exit code checked).
smoke/gen_shm_tp8.py: 16/16 ranks still continue the prompt with
'Champ de Mars in Paris' (816 phrase occurrences, byte-identical
output volume vs the pre-change run; one earlier count of 15 was an
interleaved-stdout line split, not a numerical change).
```

---

```text
model: bind one attend variant per layer role at build time

attend() used to test view.mode at run time for every chunk of every
layer, although model.past.default_layer_views() is a pure function of
the layer number and model/prefill.py asserts past.views equals it.
The role is therefore known while the model is built.

prefill_attention.py now exposes four variants -- attend_swa,
attend_source, attend_reindex, attend_reuse -- sharing _stage (slot and
chunk-position contract), _select and _finish (read the ring before
writing it). None of them looks at view.mode. bind_attend() resolves the
variant from the static view once; PrefillAttention.__init__ stores it.
Non-prefill roles (dspark) get a sentinel that raises when called.

Behaviour preserved exactly, including the replay guards: a replay chunk
must not carry compressed rows, and only the KV source layer appends
newly complete global rows.

Tested: pytest tests -q -> 221 passed, 18 skipped (rc=0).
Tested: 8-rank real generation, smoke/gen_shm_tp8.py -> 16/16 correct
continuations (816 phrase occurrences), identical to cc8096d.
```

---

```text
model: resolve the shared-input projection at build time

The layer used to run hasattr(lin, 'bind_shared_input') for every chunk of
every layer and silently fall back to two separate projections. The GEMM
binding is chosen when the model is built and never changes afterwards,
so _bind_shared_projection() now resolves the strategy once in
PrefillAttention.__init__ and the hot path just calls self.project_pair.
Same two strategies, same call order, no run-time capability probe.

Tested: pytest tests -q -> 221 passed, 18 skipped (rc=0).
Tested: 8-rank real generation smoke/gen_shm_tp8.py -> 816 phrase
occurrences, byte-identical continuations to 0d7fa5d.
```

---

```text
model: bind the global-projection and index stages per layer role

The layer body used to re-decide, for every chunk of every layer, whether
it owns a compressed source (view.mode == 'full'), whether it projects
index queries (view.mode in ('full','reindex')) and whether it builds or
consumes candidate blocks (self.layer vs config['candidate_source_layer']).
All four questions are answered by the layer number alone, so they are now
resolved once in __init__ via _bind_global_projection/_bind_index_stage.

Five run-time mode/layer-id branches removed from the hot path; the only
conditions left in the stages are replay semantics and the contract checks.
Arithmetic, call order and dtypes are unchanged.

Tested: pytest tests -q -> 221 passed, 18 skipped (rc=0).
Tested: 8-rank real generation smoke/gen_shm_tp8.py -> 816 phrase
occurrences, byte-identical continuations to 1183be9.
```

---

```text
model: share one build-time shared-input projection binding

PrefillMoE._shared() probed hasattr(lin, 'bind_shared_input') on every
chunk of every MoE layer, the same implicit fallback that the attention
layer had. bind_shared_projection() now lives in model/prefill_linears.py
next to the other GEMM bindings and both call sites resolve it once in
__init__ (self.project_pair).

Tested: pytest tests -q -> 221 passed, 18 skipped (rc=0).
Tested: 8-rank real generation smoke/gen_shm_tp8.py -> 816 phrase
occurrences, byte-identical continuations to 08d4b83.
```

---

```text
model: decide the shared-expert stream overlap at build time

PrefillMoE.__call__ re-derived the overlap decision for every chunk of
every MoE layer: a function-local import of PackedRouted, an isinstance
check, a device-type test, a reduce_shared test and a lazily created
side stream. All of those are fixed when the block is built.

_bind_moe() now resolves __call__ to _moe_serial or _moe_overlapped once,
from the routed implementation, the weight device and the TP wiring, and
allocates the side stream there instead of on the first chunk. The only
test left on the hot path is the measured 12288-row chunk shape, which is
a genuine per-call shape gate, not a capability probe.

Tested: pytest tests -q -> 221 passed, 18 skipped (rc=0).
Tested: 8-rank real generation smoke/gen_shm_tp8.py -> 816 phrase
occurrences, byte-identical continuations to 87cd714.
```

---

```text
model: bind the Engram presence and TP reduction at build time

PrefillBlock.prepare() tested 'self.engram is None' on every chunk and
rebuilt a kwargs dict to forward two optional arguments whose signature
it already knows. PrefillEngram.__call__ carried a six-term condition
mixing build facts (token_shard flag, parallel wiring, world size,
reduce_sum presence, device type) with the chunk geometry.

Both are now resolved once: prepare_impl is _no_engram or _run_engram,
and gate_impl is _engram_plain / _engram_reduced / _engram_sharded.
Only the genuine shape gate survives, inside the sharded implementation,
which falls back to the all-reduce path for non-12k chunks. The dead
token_shard attribute (hardcoded True, no reader) is gone.

Tested: pytest tests -q -> 221 passed, 18 skipped.
Tested: torchrun --nproc_per_node=8 smoke/gen_shm_tp8.py -> the expected
phrase still occurs 816 times across the 8 ranks and 2 scenarios, so the
sharded Engram path is numerically unchanged.
```

---

```text
model: remove the dead prefill timing hooks

PrefillModel.__init__ set self.timing = None and nothing in the tree
ever assigned it: grep '\.timing *=' matches only that one line. The
eight 'if self.timing is not None' gates in the encoder, replay and
CED passes were therefore unreachable instrumentation sitting in the
per-layer loop. Benchmarks use tests/bench_prefill_ops.py instead.

Tested: pytest tests -q -> 221 passed, 18 skipped.
Tested: torchrun --nproc_per_node=8 smoke/gen_shm_tp8.py -> expected
phrase still occurs 816 times across 8 ranks and 2 scenarios.
```

---

```text
model: precompute the dspark target and replay retain plan

Three hot loops asked 'block.layer in self.targets' once per layer per
chunk, and replay additionally asked 'block.layer == 20' to find the
point where the encoder tail is retained. Both are properties of the
block list, which __init__ already validates.

__init__ now builds encoder_pass, ced_pass and replay_pass: tuples that
pair each block with the collector it needs (_collect or _skip) and, for
replay, with the retain action (_retain_tail or _no_retain). The loops
call what they were handed. The encoder/CED split constant is named
ENCODER_LAYERS instead of a bare 20 in slices.

Tested: pytest tests -q -> 221 passed, 18 skipped; eight-card
smoke/gen_shm_tp8.py still emits the reference continuation 816 times.
```

---

```text
model: schedule Engram prefetch once per request

EngramPrefetch.prepare() re-derived its whole schedule on every layer:
an enabled flag, a successor bound check, the successor's Engram
presence, a lazy pool/stream creation and a 'was anything submitted for
this layer' test. All of it follows from the block list handed to the
constructor.

_plan() now pairs each layer with one of four steps (plain, submit,
consume, submit+consume) via functools.partial, and the pool and side
stream are created once when the request actually has rows to move.
__exit__ calls a bound close action instead of re-testing the pool.
The dead 'enabled' switch is gone: PrefillModel hardcoded it to True
and nothing else ever set it; the NPU check stays at construction.

Tested: pytest tests -q -> 221 passed, 18 skipped; eight-card
smoke/gen_shm_tp8.py still emits the reference continuation 816 times.
```

---

```text
ops/prefill: the routed MoE runs one path

PackedRouted no longer probes x.device and keeps a serial CPU expert loop
behind a lazy import; every call enters the bounded CANN grouped GEMM in
grouped_routed. Inside it the FP4 decode, the activation QDQ and the
token-owned combine are each one kernel: the indexed_unpack capability
probe, the fused-QDQ device probe with its two lazy imports and the
len(x)==12288 combine gate are gone, replaced by contract guards that
raise. The combine ABI already accepts any row count up to 12288, so the
128-row replay domain uses the same kernel as the 12288-token chunk.

tests/test_moe_unpack_dispatch.py dropped the four cases that asserted the
removed dispatch (CPU byte-table reference, sys.modules injection, error
propagation) and derives the wrapper path from the repository root.

pytest: 217 passed, 18 skipped. Eight-card smoke/gen_shm_tp8.py on the
tmpfs residency reproduces the recorded continuations unchanged:
'The capital of France is' -> ' Paris. The Eiffel Tower is', the Chinese
prompt -> ' 处处闻啼鸟。 夜', and the 12288-token repeated prompt -> EOS.
```

---

```text
ops/prefill: delete the six unreferenced op modules

sinkhorn_vector, rms_batch, expand_broadcast, engram_projection,
engram_gate_fused and collapse_fused (279 lines) have no importer left in
the repository: their callers were rewritten when the RMSNorm, the
hyper-connection collapse/expand and the Engram gate were each routed
through one kernel. Keeping them would leave a second implementation of
four live ops available to future edits, which is the fork the rebuild is
removing.

Nothing imports them, so the compute path is untouched; pytest still
reports 217 passed, 18 skipped.
```

---

```text
docs: record how the working base is being routed onto the rebuilt ops

The progress note stopped at the phase-1a op tests, so a reader could not
tell which ops the live path already routes, what still holds a runtime
choice, or how each routing commit is accepted. States all three, including
that the twelve-thousand-token prefill and the three recorded continuations
are the gate, and that speed is measured only after the last fork is gone.
```

---

```text
docs: the smoke's long prompts are 3151 tokens, not twelve thousand

The acceptance list claimed the padded smoke prompt was the real 12k
prefill. It is 3151 tokens and has no recorded continuation. Replaced by
the two fingerprints that do have one, plus a separate measured 12276-token
question-answering run: eight ranks complete, ' Paris, on the Champ de
Mars.', prefill 2.57s warm (15.96s cold), 0.95s per decode step.
```

---

```text
compress: the residual carry ring is two ratios wide, as past holds it
```

---

```text
attention: the compressor carry is addressed on the two-ratio ring
```

---

```text
prefill: the eight-card path reads a twelve-thousand-token prompt end to end

The eleven past-facing operators now share one ABI, so the model path is
joined instead of stubbed. A 12272-token prompt answers ' Paris' with a
top-1 logit of 25.0 on all eight ranks, prefill 12.93s, peak 57.6GB.
The recorded English continuation still matches token for token.
```

---

```text
perf(prefill/moe): keep expert histogram on device, build ei/gl via nonzero+index_select

Removes 12 small H2D copies per routed() call (2 per expert group x 6 groups).
routed() host time 33.20 -> 29.16 ms/call at N=12272 (-12%), output bitwise identical.
```

---

```text
perf(prefill/swiglu): read gate and up through their own row stride

The routed expert GEMM writes gate and up into one [tokens, 2*width]
buffer, and the call site used to hand the kernel two .contiguous()
copies of its halves. Those copies moved ~678MB per routed call and
carried no arithmetic: at width=288 a row is 576 bytes, so the slice
copy ran at ~85GB/s and cost more than the SwiGLU kernel itself.

The kernel now takes the row stride of each half as a constexpr, so the
halves are consumed where the GEMM left them. Contiguous callers pass
stride(0) == width and compile exactly the kernel they compiled before;
the output is bitwise identical.
```

---

```text
prefill sparse attention: cache the data-independent window mask

try_scheduled rebuilt the [2048, <=4096] bool window/causal mask on every
tile of every layer -- 120 times per forward -- although the mask depends
only on (pos - key) offsets and is bitwise identical across tiles and
layers. It is now built once per distinct geometry and reused.

Measured on one 910B3 die at the real prefill shape (t=12288, h=8, d=512,
window=2049, ring=12352): 10.05 -> 6.92 ms per layer, 1.45x, saving 0.063s
over 20 layers and 0.125s over 40. Output is bitwise identical to the
previous implementation across interior and shifted-ring cases.

The mask construction alone was 28% of this kernel's time, against 38%
spent in the fused attention score itself.
```

---

```text
prefill sparse attention: skip compressed rows no query can reach

A query at position p may only select global rows below (p+1)//ratio, so an
early tile never reaches the tail of the compressed history. Those columns were
still materialised in the key tensor and then blanked by the mask. Each tile now
keeps only the rows its own maximum position can reach, which is read from the
positions themselves rather than assuming they are contiguous.

12.65 -> 10.44 ms per layer. An fp32 reference gives the same result either way
(5.96e-07), and the error against fp32 truth is unchanged at 3.196e-03.
```

---

```text
timing: per-layer marks with host/device split; docs: prefill flow notes
```

---

```text
prefill: drop graph-capture blockers, fix timing artefact

Measurement first: PrefillTiming(per_layer=True) syncs the device on every
mark, so 20 per-layer marks serialise host and device and inflate the first
20 layers to 1.85s. Segment-level timing gives the true steady state:
forward 0.395s (target 0.96s), CED finish 0.558s, total 0.953s, output OK.

CED graph capture now succeeds, but warmed replay is 0.627s against 0.658s
eager -- the bounded pass is device-bound, so capture buys ~5%. The static
6-group MoE that capture required costs 6x redundant expert GEMMs and is a
net loss; it stays behind LJQ_MOE_STATIC_ROWS, now defaulting to off.

Kept because they remove per-layer host work regardless of capture:
  - v4_index: pass zero_based instead of a torch.equal device probe
  - npu_routed: validate expert ids once, not every layer
  - moe_fp4_vector/moe_indexed: replace three out-of-graph event waits with
    a one-off synchronize at table build time
  - moe_indexed.unpack: accept a caller-owned out buffer so the decode
    scratch is reused instead of reallocated per expert group
```

---

```text
prefill: remove CED graph probe scaffolding
```

---

```text
bench_prefill_ops: fix stale residual import (hc_residual)
```

---

```text
tests: MoE bank decode vs grouped GEMM sentinel (real TP8 shapes)
```

---

```text
packed_linear: run the matmul in bfloat16, not fp32

activation_fp8 hands back fp32 and unpack8 decodes to fp32, so every
main-path projection was running a fp32 cube matmul and casting the
result down to bfloat16 anyway. The fp32 operands buy nothing: E4M3
carries three mantissa bits and the block scales are powers of two, so
bfloat16 represents the decoded weight exactly.

packed_fp8 N=5120 11.27ms -> 4.80ms (57 -> 134 TFLOPs).
Checked against real layer-0 weights: relative error 1e-8, every
element within two bfloat16 ulps of the old fp32 result.
```

---

```text
window_attention: derive the causal reach on the host

Every caller builds positions as arange(query_start, query_start+t), which
the function already documents as a precondition. The tile maximum was
still read back off the device, one sync per layer, for a number the host
already knew. Eighteen syncs per 20-layer chunk.

Worth 8ms in an 18-layer pipeline bench, which is not the point: a D2H
scalar read in the capture stream is what makes graph capture fail, so
this is groundwork for replay. LJQ_REACH_SYNC=1 restores the old path.
```

---

```text
prune: delete the unreferenced prefill variants, tests and smoke trees

Nothing outside the deleted files imported them; the reachability closure
from engine.py / manage.py / startup_910b.py / server/* has 71 python files
and none of these appeared in it.

  ops/prefill_ours/            -> ops/prefill/ (the path the engine calls)
  ops/_legacy_prefill/         -> ops/prefill/
  model/prefill_{ops,buffers,engram,forward}.py
                               -> model/prefill_build.py + model/prefill.py
  ops/prefill/{add_reduce,combine,compress,core_attend,dispatch,embed,
    grouped,head,indexer,moe_fp4_unpack,moe_gemm_fp4,rope,swa_attend}.py
                               -> the graph-capture rewrite, never wired in
  ops/cann/{activation_fp8,fp4_unpack,swiglu,swiglu_weighted,build_*}
                               -> ops/cann/{moe_indexed,moe_combine,rms}
  ops/cann/native_dataflow/    -> nothing, the native prefill experiment
  tests/, smoke/               -> dropped on request; scripts/gen_tp8.py keeps
                                  the eight-card generation gate

Gate after the deletion (scripts/gen_tp8.py, 8 ranks, real weights): all four
cases produce byte-identical token ids against the pre-deletion run, case0
still matches its recorded continuation. Long-prompt logits differ by up to
2.7 between two runs of the same commit, so the comparison is on token ids.
```

---

```text
prune: drop the rebuild-era docs and the ignore/pytest entries that pointed at deleted paths

docs/PREFILL_REBUILD_PLAN.md and docs/PREFILL_REBUILD_PROGRESS.md tracked the
second prefill implementation that commit 7d9fb63 removed, and docs/OP_ABI_AUDIT.md
audited that port against node09. Nothing they describe is in the tree anymore.
docs/prefill_flow.md, docs/prefill_cost_breakdown.md and docs/prefill_op_inventory.md
stay: they describe the CED architecture and the path that actually runs.

pyproject.toml lost [tool.pytest.ini_options] because testpaths pointed at the
deleted tests/ tree. .gitignore lost the fp4_unpack / activation_fp8 asset
entries for the same reason.
```

---

```text
model: collapse the nine weight/residency modules into checkpoint_weights, wcache and snapshot

The S-side reference keeps one checkpoint store, one unit cache and nothing
else; the 910B port had split residency across layout, snapshot, loader, hold
and a three-variant H2D bank. Keep the contiguous rank blob, which the eight
rank cold start does need, drop the unused pageable/four_stage variants and the
mode switch, and move the engine to model/model.py. Eight ranks load 40.0 GiB
each from tmpfs in 7-9 s.
```

---

```text
strategy: move the engine shell to strategy/decode_worker.py with an S-side build()

The generation worker lives where the A100 reference keeps it, so the
operator port only has to fill compute in, not relocate the boundary.
Eight ranks still load 40.0 GiB each from tmpfs (EXIT=0, no compute).
```

---

```text
clear the deck for the CANN rewrite: drop runtime, ops sources, stale docs and entry scripts

Everything removed here belongs to the abandoned port attempt. What stays is
the outer shell the operator work will build on: model weights and residency,
strategy worker, server. Operator sources are kept outside the tree for now.
```

---

```text
server: resync with S trunk 5b2379d (effort 50/75/100, stream stats, MODEL_NAME const, length stop, prefill chunk metrics)
```

---

```text
ops: state what layer 0 asks of the operator layer

Every entry is an existing call site in model/prefill_*, with the released
shapes, the CUDA and PyTorch references to read, and the three rules an
operator has to meet before it counts as done.
```

---

```text
model: decide at build time what three branches were deciding per chunk

None of the three exist in the CUDA reference; they grew during the 910B
port. The fused first-chunk probe is gone, the Engram gate is picked from
the chunk width the build already knows, and the MoE overlap no longer
reconsiders a width that cannot change. What remains raises.
```

---

```text
ops: separate the encoder regime from the CED regime in the contract

Prefill only runs layers 0..19 over a variable chunk; layers 20..39 run
later over a fixed 128 rows and are the ones that must capture. Same
operators, two acceptances. Also records that the HF reference file ships
toy dataclass defaults -- the released geometry is its config.json.
```

---

```text
ops: turn packed fp4 weights into bf16 the vector core can use
```

---

```text
ops: land the layer-0 residual and attention kernels, bit-checked on NPU
```

---

```text
prefill: FP8 GEMM, and storage arithmetic that the device can actually run

The E4M3/E8M0/E2M1 primitives move into ops/prefill/quant.py so the GEMM and
the attention path quantise by the same code. Reading an exponent with a left
shift costs 900 ms per 62M elements on this device; masking the exponent field
and keeping the result in FP32 costs nothing, which takes activation
quantisation from 1928 ms to 13.8 ms with bit-identical results.
```

---

```text
prefill: the layer-0 sliding window as the model calls it

attention.sparse takes the ring read plus the chunk and measures the band from
the end both sequences share, so history lengthens the band instead of shifting
it; the offset is still checked so a short ring read cannot silently attend to
the wrong positions. Matches the float reference to 3.7e-3 (BF16 rounding) and
runs 12288 queries in 3.15 ms.
```

---

```text
prefill: routed FP4 experts, one dequantised grouped GEMM per layer
```

---

```text
layer0 runs end to end on real weights, single path only

Brought up layer 0 on eight ranks with the released FP4/FP8 weights and
removed every branch the bring-up exposed: the MoE side-stream/serial pair
and its side_stream_safe probe, the three Engram gate variants, the optional
embed/head/router/routed injection points, the optional out= on the attention
ops, the per-chunk profiling clock and the reduce_sum None check. Named FP4/FP8
helpers replace the parameterised roundtrip. sparse() now views the ring rows
as one KV head so the fused NPU kernel gets TND.

Measured on the 8-rank run, 128 tokens: no host reads on the layer path,
15 repeats bit-identical, forward peak 1580 MiB over a 40.9 GiB resident
weight set (3.3 GiB assembly).
```

---

```text
prefill: real rope table, copy-mix without the [T,H,H,D] transient, probe_layer timing hook
```

---

```text
moe: dequantise FP4 experts with the device kernel

The packed experts were unpacked by a chain of small torch ops, which spent
137 ms a layer on integer work the vector units do badly. The AscendC kernel
already sitting in ops/kernels does the same gather in one pass: 13.3 ms for
w13 and 7.0 ms for w2, bit for bit identical to what the chain produced.

The kernel reads a row at a time, so w2 is handed to it sixteen rows at a
time: its rows are 288 columns wide, and a 144 byte row leaves the scale
offset off the alignment the gather needs, while sixteen of them are one
4608 column row over the same contiguous bytes.

The bank now holds each expert the way the kernel writes it and the grouped
GEMM takes a transposed view of it, so no copy is made to change the shape
and the GEMM itself is faster for reading the weights that way.

Layer 0 at 12288 tokens: 326.8 ms -> 208.1 ms, with the resident and peak
memory unchanged at 49.24 and 53.82 GiB.
```

---

```text
fuse hyper-connection expand/collapse into one AscendC kernel

expand and collapse each materialised a full FP32 copy of the residual
stream ([T,4,5120] -> 1 GiB) only to round it straight back to BF16.
One vector kernel per operation now reads the BF16 copies once, keeps the
accumulator in FP32 inside UB and rounds once at the store.

Measured on 8 cards, T=12288, layer 0:
  expand    37.37 -> 8.94 ms/run
  collapse  12.67 -> 4.34 ms/run
  WALL     207.5  -> 184.8 ms/run

collapse is bit-exact against the torch reference; expand differs on
9994 of 251658240 elements by one BF16 ulp, because the kernel adds the
post term before the source terms rather than after.
```

---

```text
hc: fuse 19-iter sinkhorn into AscendC kernel (mixes 24.05->9.41ms)
```

---

```text
Quantise prefill activations on device, not in torch

Every packed projection re-quantised its input with a chain of elementwise
torch ops that read and wrote 12288x5120 in FP32 several times over: 13.6 ms
a call, 108 ms a layer, 57 percent of the wall. The arithmetic is a per-32
amax, a power-of-two scale, and an E4M3 round trip, and all of it fits in
one pass over UB.

pre_fp8_act is that pass, reusing the gather table fp4_dequant already
builds. Its result is bit-identical to quant.fp8_roundtrip over 63M values
at K=5120 and 19M at K=1536, so the model sees the numbers it saw before.
The quantised copy lands in a run of the workspace dict, under a key no
projection can have, sized once at build time; nothing is allocated on the
call path.

13.57 ms to 2.36 ms a call; probed layer wall 190 ms to 158 ms.
```

---

```text
weights: load the W4A8 release into shm wcache

The published checkpoint stores every quantized tensor per expert, with
float32 companions and zero points beside each weight. Zero points are all
zero, so quant_group verifies and drops them; placement now reports them as
dropped instead of pretending they are shardable. Companions follow the
placement of their own weight and stay whole on row-parallel groups, which is
what prepare_rank already did -- accounting and sharding now read the same
rule instead of disagreeing.

Engram tables live in engram_int8/ and the top-level copies are [1,1] stubs,
so the index decides which file owns a tensor and stubs are skipped.

rank0 builds all 937 units in 492s, 41.405 GiB on disk, matching accounting.
```

---

```text
engram: drop the loader that contradicted the mapped tables

HostEngram.open read whole tables with get_tensor, which would have made a
98 GB table resident. Nothing called it: HostWeights maps the host cache
MAP_PRIVATE and constructs HostEngram from those views, so a gather pages in
only the rows it touches. Removing the secon
...[Truncated]...
ches what the code does.
```

---

```text
weights: name cached companions after the kernels, not the checkpoint

The released W4A8 checkpoint calls its per-channel float32 companion
weight_scale; every consumer in this engine asks for .scale. Cache the
tensor under the name the kernels already use so one vocabulary spans
the cache and the ops. Loading a rank also stopped re-hashing 41 GiB it
had just proven: that cost 43 of 52 seconds and re-derived what the unit
identity already guarantees.
```

---

```text
prefill layer0: W4A8 int4 expert path end-to-end

- dequant.cpp replaces fp4_dequant.cpp (w8/w4 launch, bit-exact vs torch ref)
- gemm/linears/npu_routed switched to int8 per-channel + packed int4 experts
- drop simulated FP4/FP8 activation roundtrips from prefill_layer (base model changed)
- routed expert prefix follows checkpoint layout (layers.N.ffn.*)
```

---

```text
dequant: fuse narrow rows into one pass and scope the vector barriers
```

---

```text
dequant: overlap fetch, expand and store
```

---

```text
Run the shared experts on a side stream beside the routed bank

The shared experts and the routed bank read the same activation and touch disjoint buffers, so the shared projections and their TP sum now run on a per-device side stream while the routed GEMMs proceed on the main stream. Layer0 prefill of 12288 tokens drops from 67.84 ms to 64.06 ms with a bit-identical output.

A projection captured the launch stream when it was built, so its weight expansion always went to the stream that was current at build time while the matmul followed the calling stream. On one stream that was invisible; from a side stream the matmul read a half-expanded workspace. The stream is now read at call time.
```

---

```text
Fuse routed SwiGLU into a preallocated AscendC kernel

Clamp, SiLU, up and routing probability in one FP32 pass, BF16 output.
Bind width288/limit10 at build; no hot-path dispatch/D2H/allocation.
Reuse gated buffer and drop unused hidden. No new production files.
Leaf 73728x288: 1.443354 -> 0.186324 ms (7.75x).
Rows1/6/17/128/768/73728 exact; each graph100 stable, alloc/peak delta0.
Real production: all8 ranks x4 calls bitwise equal to original Torch.
Layer0 TP8 T12288 alternating A/B 4x50, final3 rounds/max rank p50:
64.112986 -> 62.972152 ms (1.779% reduction).
Existing layer A-A has nondeterminism; no full-layer bit-exact claim.
Repro: python /tmp/ljq_swiglu_opt/leaf_production.py;
python -m torch.distributed.run --standalone --nproc_per_node=8
/tmp/ljq_swiglu_opt/stable_ab.py. Full model/decode not tested.
```

---

```text
Fuse six-route MoE combine into preallocated FP32 output

Avoid the full BF16-to-FP32 bank copy and index_add workspace.
Keep collective FP32 and six sequential adds in sorted-bank order.

Leaf 12288 tokens: 3.668 -> 1.552 ms; graph100 and dynamic routing pass.
8 ranks / 32 real calls bitwise match independent sequential reference.
Original-call Layer0 ABAB four pairs: 63.02757 -> 60.69021 ms (-3.71%).

Validation boundary: not bitwise equal to atomic index_add. Initial
Layer0 absolute RMS 1e-6 gate failed; AA also exceeds it. This commit
does not claim that gate or full-model E2E validation passed.
Evidence: /data/ljqinfer_combine_repro.tar.gz
```

---

```text
Enable layer0+1 prefill probe and reference-aligned Engram gate

Functional baseline only. Reuse projection output and bound 1024-row scratch. Eight-rank reference gate checks and fixed-input serial/overlap boundary checks pass. Zero temporary allocation remains unmet; whole-prefix bitwise repeatability also remains unresolved. Experimental fused gate is not integrated. Evidence: /tmp/ljq_layer01/functional_baseline.json
```

---

```text
Retain GEMM storage and enable deterministic prefill reductions

CPU ownership lifetime check passed; existing 12288-token layer0+1 TP8 regression passes all 8 ranks over 16 overlapped/serial runs. Repeatability only, not gold parity or full-prefill acceptance.
```

---

```text
Bind FP32 pooled-compressor projections with persistent input scratch

Official Compressor promotes weights/input above ratio 1 despite BF16 checkpoint storage. CPU/NPU leaf checks pass for lengths 1/17/3. 128x5120 ->512 NPU graph replays 100 times bitwise identically with zero allocated/peak growth; ratio1 stays BF16. Full prefill not yet validated.
```

---

```text
Restore all four compressor carry slots during cold replay

Replay retains final min(T,4) projected rows at absolute position modulo 4, matching SourcePast storage. Removed redundant FP32 input temporaries; bound projections own staging. Extracted actual replay function passed 56 CPU cases: start 0..7; lengths 1,2,3,4,5,17,128; other slot unchanged. NPU and end-to-end cold replay not yet tested.
```

---

```text
Implement build-bound prefill attention leaves and DMA page gather
```

---

```text
Fix Engram host rows to use decoded FP32 group scales

HostEngram ABI already carries FP32 multipliers, not E8M0 bytes.
Remove duplicate exponent decoding that shrank real values to ~1e-37.
Real cached rows: 2 layers x 8 rank slices match independent direct
multiply exactly; chunk/history equivalence and rank concatenation exact.
Corrected max abs 2.5-3.5. Repro: OMP_NUM_THREADS=4 python
/tmp/test_real_engram_scale_0925.py on atlas-a2c.
CPU host dequant only; no TP communication, NPU or whole-model claim.
```

---

```text
Fix W4/W8 AIV stride to cover both vector subcores

Use 2*GetBlockNum for dav-c220 vector lane stride; remove overlapping writes. Repo-built W4/W8 CPU exact, guards and graph100/peak0 pass (four geometries); /tmp/dq_repo_gate.json. Previously integrated TP8 full12k warm 3.281s includes this fix plus attention changes; no isolated full-model speed claim. Full-model independent oracle and three-chunk validation remain pending.
```

---

```text
Use Cube scoring and stable full-bank selection in CED; integrate prefill kernels

Take over existing fixed-build prefill work and integrate verified vector RoPE, FP8/FP4, candidate scoring and sparse correction; remove replaced candidate/correction kernels. CED full-bank candidates now reuse bound Cube scores; full selection uses stable descending sort.

TP8 real 12288-token warm max-rank CED 1.076s -> 0.668s; total 3.281s -> 2.882-2.915s. Layer0 467ms -> 42-44ms; reindex layers remain ~100ms. Reproduce: python /tmp/prefill_ced_repo_0926_run1/launch.py (run.py uses repository libraries, 8 rank results, exit0).

Leaf: 256 random/ties candidates and top512 exact vs old; 12288 top512 exact vs CPU random/ties, Cube score error <=5.97e-7. Candidate set identical; near-tie block ordering differs in 32 expanded slots, not bitexact old reduction order. Prior vector leaf/guard/graph100 gates passed. Syntax and whitespace checks pass.

This is a runnable performance checkpoint, NOT completed whole-model numerical/semantic or three-chunk acceptance. Those gates and CED<0.2s remain pending.
```

---

```text
Replace CED candidate reindex scalar scores and heap with Cube and masked stable sort

Dense logical-row mask preserves candidate membership and score-desc/ID-asc ties; UB staging and DMA avoid scalar GM write loss. Leaf T16 G12288 C16384: random/ties/changed/empty top512 matches CPU; warm selection 10.63ms -> 0.602ms. Real TP8 12288-token repository run: warm CED 0.668s -> 0.344-0.345s, total 2.57-2.60s. Repro /tmp/prefill_ced_repo_0926_run2/launch.py; exit0, all 8 ranks complete six iterations. Full independent model oracle, graph replay and multi-chunk validation remain pending; targets not yet met.
```

---

```text
Pipeline W4 tile prefetch to overlap DMA with vector decode

Six leaf shapes pass CPU exact, guard, graph100 and zero replay allocation. Narrow expert bank 3.045 -> 2.655 ms; wide bank unchanged. TP8 real 12288-token eager warm max-rank CED 0.3453-0.3456 -> 0.3373-0.3374 s; total 2.579-2.602 -> 2.557-2.559 s. Reproduce: /tmp/prefill_ced_repo_0926_run3/launch.py; leaf /tmp/prefill_dq_pipeline_0926_run1/leaf.py. All eight ranks complete. Full-model independent oracle, three-chunk validation and target <2 s remain pending.
```

---

```text
Use fused Sinkhorn and HC expansion in CED while retaining input staging

128-row leaf: mix 1.151->0.360 ms (max FP32 difference 2.38e-7), expand 0.237->0.043 ms bit-exact. Real cached-weight TP8 eager encoder20+CED20, 12288 tokens: warm rank-max CED 0.337->0.295-0.296 s; total 2.5116-2.5130 s. All 8 ranks complete six iterations; /tmp/prefill_ced_repo_0926_run4/launch.py. Timing diagnostic only; no whole-model semantic oracle or three-chunk validation.
```

---

```text
Replace CED candidate heap with block maxima and stable sort

CPU exact leaf checks: random, ties, changing counts, empty and 12k/12416/32k banks. TP8 run5 completed all ranks; hot max CED 0.28372/0.28406s vs run4 0.29632/0.29490s. Total 2.50756/2.51410s: no clear end-to-end gain. Semantic oracle and three-chunk validation remain outstanding.
```

---

```text
Skip zero-count CED experts in device-driven W4 dequantization

Share full/active template kernel; preserve encoder full-bank behavior and runtime stream binding. Leaf dynamic/all/empty tests exact, guarded, 100 repeat allocation-free. TP8 12288 repeated-token run6 exit0 all ranks: hot CED .2093/.1783s vs .2837/.2841s; total 2.517/2.405s, still above 2s. Four alternating active/full model passes: logits and hidden bitwise exact on all eight ranks (prefill_dq_active_parity_0926_run1). Active ratio21.5% is workload dependent, not a general guarantee. Independent semantic oracle and three-chunk validation not covered.
```

---

```text
Batch CPU token shard staging for prefill

Replace per-token tensor assignments with bulk int64 staging and vectorized shard masks. Eight-shard random and boundary comparison exact; CPU leaf median 216.32 to 2.03 ms. TP8 run7 eight ranks complete and exit0: encoder hot 2.104/2.086 s vs prior approximately 2.207 s; total 2.303/2.345 s. CED varied 0.199/0.259 s, so neither stable sub-0.2 CED nor sub-2 total claimed. Reproduce /tmp/prefill_ced_repo_0926_run7/launch.py. Independent semantic oracle and multichunk validation outstanding.
```

---

```text
Limit score reduction masking to the invalid tail

Six leaf cases pass old/new exact equality, CPU oracle and output guards. TP8 run8: all ranks complete six timing iterations; uninstrumented hot total maxima 2.22909/2.22836 s, CED 0.18728/0.18186 s. Timing cases are diagnostic, not independent whole-model semantic validation.
```

---

```text
Use vector barriers between score reduction vector operations

Six leaf cases pass exact old/new, CPU oracle and guards. Full tile 0.15056 to 0.14569 ms; boundary 0.09667 to 0.08507 ms. TP8 run9 all ranks complete six diagnostic timings. Hot total maxima 2.26254/2.21892 s, CED 0.21595/0.18264 s: end-to-end benefit is within variation, not a stable sub-0.2 s claim. Independent whole-model semantics not validated here.
```

---

```text
Use vector-only barriers for RMS and HC statistics

Replace 15 vector-to-vector full barriers; retain scalar and DMA synchronization and reduction order.
Clean rebuilt baseline leaf: 12 boundary cases bitwise equal, guards/finite/repeats pass. 12k RMS 0.4240->0.4111ms; stats 1.5418->1.5041ms. Reproduce: /tmp/prefill_encoder_hc_cleanbase_0926_run1/launch.py.
TP8 12k full run all ranks complete: encoder hot 2.0467/2.0364->2.0319/2.0131s versus run9. CED varies 0.2065/0.2210s; total 2.2382/2.2338s, no stable end-to-end win or sub-2s claim. Run: /tmp/prefill_encoder_hc_barrier_full_0926_run1/launch.py.
Independent full-model oracle, full-model AB output parity and three-chunk validation not performed for this change.
```

---

```text
Batch joint attention packing through UB DMA

Replace scalar GM index accesses with buffered DMA; copy BF16 KV without FP32 round-trip. Same predicates, ordering and ABI.

Leaf: six independent CPU-oracle cases, guards and 16 repeats pass; 12k pack 5.498 -> 4.460 ms. TP8 full 12288-token prefill: all eight ranks exact logits and hidden versus baseline, four repeats exact, both exits 0. Three hot rank-max encoder means 2.044 -> 1.999 s; full means 2.271 -> 2.215 s (CED noisy). Full target <2 s not yet met.

Reproduce: /tmp/prefill_joint_pack_dma_0926_run1/launch.py and /tmp/prefill_joint_pack_full_0926_run1/launch.py on atlas-a2c; artifacts retained there. No independent full-model oracle or three-chunk validation in this change.
```

---

```text
Batch sorted attention selection using its valid prefix

Find the valid sorted score prefix by binary search, initialize -1 padding in bulk, and copy selected int64 indices in aligned UB blocks. Preserve stable Sort ordering and invalid suffix semantics; no new runtime flags or files.

TP8 12288-token single chunk, three warm uninstrumented rank-max means: encoder 2.048534 -> 1.983845 s; total 2.276081 -> 2.182928 s. CED 0.227617 -> 0.199235 s is variable and not attributed to this kernel. All 8 ranks logits/hidden bit-exact vs baseline and repeated runs exact. Six leaf oracle/guard/repeat cases passed.

Reproduce on atlas-a2c: python /tmp/prefill_sorted_prefix_0926_run1/launch.py; python /tmp/prefill_sorted_prefix_full_0926_run1/launch.py (baseline from d19e397). Full-model independent oracle and multi-chunk validation not performed in this change. Total <2s target remains unmet.
```

---

```text
Hoist sparse pack bounds and split local/global index loops

Keep metadata and per-query causal bound out of 512 selected-index checks. No ABI, storage, precision or file-count change.

Six leaf cases: CPU oracle, guards, exact repeat, 100 graph replays, zero peak growth. 12k pack 4.486 -> 3.163 ms. TP8 12288-token uninstrumented full test, three warm runs using per-run maximum across ranks: encoder 2.038952 -> 2.012073 s; CED 0.203273 -> 0.202969 s; total 2.242061 -> 2.214804 s. All eight ranks exact baseline logits/hidden and repeat-exact, exit 0. Timing is one AB series, not a statistical confidence claim.

Reproduce: /tmp/prefill_joint_pack_bounds_0926_run1/{leaf.py,graph_leaf.py}; /tmp/prefill_joint_pack_bounds_full_0926_run1/launch.py. Full independent model oracle, multi-chunk validation and <2s target remain pending.
```

---

```text
Batch four prefill score GEMMs without changing TP selection order

Group consecutive full-bank query tiles into one Cube call, keeping score reduction, HCCL and top-k geometry unchanged. Single-tile CED and tiled-bank paths keep the same computation.

TP8 12288-token encoder20+CED20 eager: mean max-rank hot encoder 1.978315 -> 1.963575 s; total 2.173625 -> 2.162942 s (3 hot repeats; noise-sensitive). Full-bank selection median 93.260 -> 85.369 ms. Eight ranks exact logits/hidden and repeat-exact.

Integrated TP8 321/544 and 12288/6144 shapes: exact scores/IDs, repeat exact, zero peak growth. Four head/tail cases pass 100 graph replays with zero peak growth. Reproduce via /tmp/prefill_score_grouped_mm_integrated_0926_run1/launch.py; full-model evidence /tmp/prefill_score_grouped_mm_full_0926_run1.

Not independent model oracle or semantic validation; multi-chunk and sub-2s goal remain open.
```

---

```text
Pair HC expand outputs to reduce vector barriers without changing sums

Keep DT2560 and sequential FP32 accumulation/BF16 rounding. Reuse consumed
BF16 input UB for output and process two independent output groups together.
12k leaf median 2.804390 -> 2.577532 ms. TP8 12k three warm rank-max
means: encoder 1.984066 -> 1.949636 s; total 2.198597 -> 2.190731 s.
CED variation is substantial; total <2s remains unmet.

Eight ranks repeat-exact and baseline/candidate logits+hidden bit-exact.
Rows 1/33/12288: guard/repeat, sampled CPU oracle, 100 graph replays,
dynamic inputs, poisoned-output restoration and zero memory growth pass.
Graph harness fixed to launch on current capture stream, not cached stream.
Reproduction: /tmp/prefill_hc_expand_pair_0926_run1/{leaf,graph_v2,integrated_test}.py;
full TP8 AB: /tmp/prefill_hc_expand_pair_full_0926_run1/launch.py.
Independent full-model oracle and multi-chunk not covered.
```

---

```text
Overlap HC cast writeback with statistics reduction

Keep per-tile sum order and final completion barrier; replace earlier full
barriers by producer-consumer events. No new source files or switches.

12k stats leaf 1.503752 -> 1.362968 ms. Alternating 12-round complete
mix median 2.913417 -> 2.753445 ms, all outputs exact. Six-shape guards,
100 graph replays, dynamic input and zero allocation growth pass.
TP8 12k logits/hidden bit-exact on all ranks. Three warm rank-max means:
encoder 1.946242 -> 1.947202 s (no established full-encoder gain);
total 2.168275 -> 2.153629 s, dominated by CED variance.
Evidence: /tmp/prefill_hc_stats_overlap_0926_run1/acceptance_checkpoint.json
and /tmp/prefill_hc_stats_overlap_full_0926_run1.
Not independent model oracle; multi-chunk and sub-2s target not proven.
```

---

```text
Use directional fences for score reduction head loads

Replace per-head Vec.load full sync with V_MTE2/MTE2_V dependencies; preserve arithmetic order and other kernels.
TP8 12288 tokens, three warm rank-max means: encoder 1.947664 -> 1.917673 s; total 2.151348 -> 2.108765 s. CED variation is not encoder gain.
Eight ranks exact logits/hidden and repeat. Eight leaf geometries CPU oracle/masks/guards pass; graph100, changed inputs, output sentinel and zero allocation growth pass.
Reproduce: /tmp/prefill_score_fences_0926_run1/{launch.py,graph.py}, /tmp/prefill_score_fences_full_0926_run1/launch.py.
Under 2 seconds, independent full-model oracle, multi-chunk remain unverified/not achieved.
```

---

```text
Use directional fences in joint attention correction
```

---

```text
Overlap encoder shared-expert HCCL with routed compute

Build one side stream and per-layer ready/done events. Keep ACL compute
on the main stream to preserve shared workspace safety; join before add.
No new files or runtime selection branch; CED binding remains serial.

TP8 12288-token same-model ABBA (18 rounds, two warmups), all 8 ranks
exact hidden/logits. Initial experiment total 2.092475 -> 2.022458 s.
Integrated repeat: {"baseline": {"encoder_s": 1.9012782152276486, "ced_s": 0.211177445249632, "total_s": 2.1122391507960856}, "candidate": {"encoder_s": 1.8339088610373437, "ced_s": 0.19704320607706904, "total_s": 2.030804035253823}}

Repro: python /tmp/prefill_shared_reduce_integrated_0926_run1/launch.py
(use a fresh output directory). No independent full-model oracle or
graph-capture acceptance claimed; 2-second target not yet established.
```

---

```text
Batch source-attention score reduction by contiguous query row

Preserve FP32 four-head summation order and mask semantics while loading a full row into UB. Fixed H=4/KT<=8192 specialization, unchanged ABI.

TP8 12288-token native encoder+CED, two process-order pairs, all 8 ranks hidden/logits exact: 2.026401->2.019259s and 2.044239->2.006534s (hot rank-max means). Leaf eight-rank exact including padded/invalid queries. Evidence and repro: /tmp/prefill_source_rowreduce_0926/full_launch.py, leaf.py, summary.json. Whole-model graph and independent semantic oracle not claimed.
```

---

```text
Fuse Engram gate elementwise stages around unchanged reductions

Two vector kernels reduce GM traffic; preserve rotation, FP32 reduction order and BF16 boundaries. Eight-rank leaf exact over five shapes/types; native TP8 12288-token AB 10 rounds all hidden/logits exact. Hot rank-max encoder 1.843447->1.810853s, encoder+CED 2.049668->2.024215s.

Repro/evidence: /tmp/engram_opt_0926/model_ab.py and evidence/model_ab3, evidence/leaf1. AB harness scratch/OOM failures fixed using shared baseline arena. Whole-model graph and independent semantic oracle not claimed.
```

---

```text
Overlap Engram CPU row lookup with encoder computation

Build-owned pinned rows and persistent CPU-only worker; caller dispatches H2D before consumption. Avoid concurrent worker NPU dispatch observed blocked in rtKernelLaunch. Synchronous stage dispatches all rows for capture interface. Replaces unused old helper in two existing files.

TP8 cached real weights: small_v2b eight ranks x12 chunk cases exact (two slots, 2x128 chunks, eager and stage/body). long_v2 12 alternating rounds exact, four hot rounds each: rank-max single-wall base 2.029409s -> candidate 1.947403s. Repro /tmp/engram_load_0926/{small_v2b,long_v2}/launch.py. Independent semantic oracle and whole-model graph not claimed; combined production revalidation follows.
```

---

```text
Optimize prefill Engram with token-sharded gate and reduce-scatter

Preserve BF16 projection -> FP32 reduction -> BF16 gate boundaries.
TP8 12288 same-process alternating max-rank median 1.89243 -> 1.85256 s.
8 ranks: 14 full-model rounds hidden/logits exact, 12 slot/append cases,
5 leaf sizes, changed-input graph replay exact at 33 and 12288.
Repro/evidence atlas-a2c:/tmp/engram_rs_0926: model_ab.py/model1,
small.py/small1, leaf_diag.py/leaf4 (exit 0).
Intermediate projection differs by up to 1.1920929e-7 in observed cases;
final tested outputs exact. No independent full-model oracle added.
1.5s and <=100ms/layer goals remain unmet. Combined integration pending.
```

---

```text
Optimize prefill HC expansion buffering and fuse collapse with RMS

Keep BF16 collapse rounding and reduction order; stage coefficients and
use double-buffered expansion output. No runtime experiment switches.
Isolated TP8 alternating AB: 8 ranks x14 exact rounds, hot maxrank median
1.915184s -> 1.879509s (35.675ms). Leaf boundary/graph evidence:
/tmp/hc_pipeline_0926/norm_edge_result.json and paired_v2/results.
Combined with c6a175e Engram: 8 ranks x10 exact hidden/logits rounds,
baseline_exact=true, exit0; hot plain maxrank median1.860165s.
Reproduce: python /tmp/prefill_engram_hc_c6a175e_v2/launch.py
Frozen build/source: /tmp/prefill_engram_hc_c6a175e/repo.
Layer2/8/14 still ~131/131/152ms; 1.5s/100ms goals not met.
Combined test is eager baseline equivalence, not independent HF truth or
full-model graph validation. No whole-suite regression claim.
```

---

```text
Optimize prefill source selection with grouped score and sort tiles

Keep TP reduction geometry and stable score/ID ordering unchanged.
Verified frozen a8baffa A/B: 8 ranks x 10 rounds hidden/logits exact.
Source selection medians L2/L8/L14: 54.99/54.97/55.01 ->
49.28/49.17/49.08 ms. Plain warm max-rank means 1.918 -> 1.862s;
wall samples are noisy, not a fixed E2E speedup claim.
Leaf score/IDs and real three-source IDs passed prior isolated tests.
No full graph/HF regression claim; 1.5s and 100ms/layer remain unmet.
```

---

```text
Vectorize prefill joint-pack IDs and DMA missing-count output

Replace scalar selected-ID loop with vector predicates and write count by
DataCopyPad, avoiding intermittent scalar missing-count publication.
Validation: six wide-ID cases; six original graph cases plus three seeded
repeats, each with graph100, dynamic inputs, sentinels and zero peak growth.
TP8 12288: eight ranks x twelve ABBA rounds, 38 pack calls each, hidden and
logits exact against frozen baseline outputs. First four rounds excluded.
Max-rank wall median 1.807380 -> 1.754417s; mean 1.808041 -> 1.766397s.
Evidence: /tmp/prefill_pack_f82c6c9 and /tmp/prefill_pack_count_dma_v3.
No new independent HF oracle or full-suite regression claim.
```

---

```text
Pipeline prefill attention selection with build-owned replay graphs

Overlap Cube, ordered TP sums and stable sort with two score slots. Share serialized scratch and capture one graph per scoring geometry outside requests; reset owned graphs before transport.

TP8 12288-token leaf: 29.928 -> 21.110ms; four cases, exact scores/IDs and dynamic restore. Explicit model AB warm max-rank medians 1.792107 -> 1.684761s. Integrated production median 1.697949s, 8 warm after 4 cold rounds; 8 ranks x12 exact saved logits/hidden; double close and exit 0.
Reproduce: python /tmp/select_pipeline_production_027f9a4/launch.py
AB evidence: /tmp/select_pipeline_model_explicit_027f9a4

Saved baseline is not an independent oracle. Multi-chunk and integrated per-layer timing pending. 1.5s target not met.
```

---

```text
prefill: fuse encoder stable top512 sorting and ID output in UB

Replace pipeline Sort+finish with AscendC vector indices, stable UB sort and
gather-packed int64 IDs; remove full sorted GM values/IDs from pipeline.
Short banks pad to 544; specialize K512, aligned banks <=6144 at build.
Generic/CED selection remains unchanged. No new production files.

Leaf: 36 shape/distribution cases exact vs CPU stable oracle and old kernel.
192x6144: ~0.118 to ~0.079 ms. TP8 12k selection ~21.08 to ~20.66 ms.
Whole 12k encoder A/B max-rank warm mean 1.703740->1.693811 s; reverse
B/A 1.716806->1.702649 s. Final production 1.69381754 s, 8 ranks x12
rounds bitwise vs original baseline; first four cold rounds excluded.
TP8 five cases: 100 individually checked graph replays and zero peak growth.
Whole-model parity is baseline parity, not independent model oracle.
1.5s and every-layer<100ms goals remain unmet; no new layer timings.

Reproduce on atlas-a2c: python /tmp/prefill_topk_bounds_07fa613/launch.py;
python /tmp/select_topk_production_graph100_07fa613/graph_launch.py;
python /tmp/select_topk_production_final_07fa613/launch.py.
Evidence: /tmp/select_topk_model_{ab,ba_bounds}_07fa613.
Build attention.cpp with bisheng -x cce --cce-aicore-arch=dav-c220
-O2 -std=c++17 -shared -fPIC and CANN tikcfw include paths; .so ignored.
```

---

```text
perf: replay fixed-address CED body with an owned NPU graph

Capture on first staged call, restore input rows before replay, and reset the outer graph before owned selection graphs. No new files.
TP8 12288-token existing production harness, 36 rounds with 4 cold excluded and old/new/new/old pairing: cross-rank max mean 1.698206s -> 1.670981s; 288 saved-reference logits/hidden comparisons exact; all ranks exit 0 and close twice. No independent oracle or multi-slot validation in this change. First call includes warmup/capture cost.
```

---

```text
perf: fuse shared expert SwiGLU and remove FP32 intermediates

Replace seven ACL calls with one AscendC kernel; BF16 inputs, FP32 activation/product, BF16 store. Remove three shared activation scratch buffers.
Leaf 12288x288: 175us ->53us, 3.5M elements exact. TP8 production 8 rounds exact against saved logits/hidden; paired encoder ABBA 16 rounds, first4 excluded, cross-rank max mean old 1.667074520s -> new 1.666947646s; all128 rank-rounds exact and all8 exits0. CED uses fused implementation on both sides.
Reproduce baseline builder from parent residual.py with Workspace and shape-shared FP32 scratch, alternate encoder closures, run existing /tmp/select_topk_production_final_07fa613/run.py in memory. Logs launcher.log SHARED_AB3. No new files. Independent HF oracle and dynamic slot matrix not run.
```

---

```text
perf: avoid NPU position H2D barrier and identity gate scaling
```

---

```text
perf: vectorize complete-window indices in joint attention packing
```

---

```text
perf: tile joint attention KV copies instead of fencing every row
```

---

```text
perf: overlap attention correction DMA and narrow loop synchronization
```

---

```text
perf: narrow joint-pack vector and DMA synchronization
```

---

```text
perf: scan SWA correction positions once per row
```

---

```text
perf: overlap SWA correction DMA and use directed synchronization
```

---

```text
perf: tile contiguous current KV in window packing
```

---

```text
perf: parallelize position generation by exclusive cache lines
```

---

```text
perf: narrow frequency gather buffer-reuse synchronization
```

---

```text
perf: batch consecutive rotary frequency rows with static tile dispatch
```

---

```text
perf: batch paged KV writes within physical page boundaries
```

---

```text
perf: batch paged reads within page and valid-prefix boundaries
```

---

```text
perf: narrow compressor load and arithmetic synchronization
```

---

```text
perf: batch compressor pairs with strided DMA
```

---

```text
fix(npu): transfer tensor descriptor ownership to ACL tensor list
```

---

```text
Fix bounded cold prefill replay and own NPU capture lane
```

---

```text
Prebind embedding IndexSelect scratch outside prefill compute
```

---

```text
Remove strided prefill seed fill temporary with build-owned broadcast
```

---

```text
Share prefill scratch and dispatch startup-bound phase/length plans

Reuse dequant scratch, ACL workspace buckets, rotary tables, FP32 head and reduction stream within one serialized Past lane. Engine owns one transport and dispatches prebound exact shapes without runtime rebuild; release plans before transport. No new repository files.

TP8 shared/independent/Engine small controls: each 8 ranks x12 cases, exit0; Engine output comparison metrics equal independent control, five measured append bodies per rank have zero stage/encoder/retain allocation deltas. Shared replay128 costs 557756928 bytes. Repro scripts: /tmp/cold_replay_9f097fd_20260927_085123/{shared,independent,engine}_plans_small.py via torch.distributed.run --standalone --nproc_per_node=8 SCRIPT OUTPUT_DIR.

Boundaries: metric equality is not direct tensor bitwise proof or independent gold. Existing cold truncated-replay error remains. Mixed lengths, long resident plans, arbitrary-T coverage and throughput not validated by these small diagnostic runs.
```

---

```text
Reuse serialized Engram storage to fit resident long-prefill plans

Reuse Engram send storage for the later same-stream gate key; receive buffers remain independent. Omit staged projection input allocations. TP8 engine_plans_long_alias (under /tmp/cold_replay_9f097fd_20260927_085123) exit0, all 8 ranks x12 cases, 3x12288 chunks, cold store/restore, replay128 and suffix128 with three startup-bound resident plans, one weights/Past owner. Long build delta 17931683328 -> 16925049344 bytes versus staged-input trim; measured append stage/body/retain peak deltas zero. Output comparison metrics equal long_cache_source_trim on all ranks; not bitwise tensor gold or a throughput benchmark. Arbitrary short tails remain unsupported.
```

---

```text
Support valid-prefix prefill and cold replay in startup-bound plans

Stage real row counts in fixed buffers; limit compression, carry and window publication to valid prefixes; retain the real encoder tail and reuse the Past-owned Engram stream. Dispatch to the smallest startup-bound plan without building during execution.

Verified TP8: 36K (3x12288) cold store/restore/replay/suffix, plus 1/17/127/128-token cold replay with exact same-bucket outputs. Default attention library deployed and verified with 8x24 cases. No new repository files.

Known numerical limitation: forced 128-vs-256 bucket recomputation is not invariant (1-token KL 2.027 and top1 differs). Replay distribution checks cover the recorded prompts only; this commit does not claim general numerical acceptance or new performance results.
```

---

```text
Document graph-captured B1-B4 Q6 decode development plan
```

---

```text
Document evidence-backed commits and independently deliverable tasks

Change: require commits after verified advances, reproducible evidence,
explicit contracts, resource ownership and independent task acceptance.
Validation: documentation-only update from d3ea266. Exact UTF-8 readback
and five required-clause assertions passed. git diff --cached --check
passed; staged file list contains only AGENT.md.
Reproduce: git show --check HEAD; inspect AGENT.md commit/task sections.
Result: rules persisted. No runtime code changes or device experiments;
this commit makes no decode correctness or performance claim.
```

---

```text
Clarify rollback checkpoints and staged decode latency goals

Version state: documentation only. Runtime code, configuration and deployed
libraries unchanged from 920643e. NOT a new working decode checkpoint.
Prefill baseline remains 8cdca784; complete graph-captured decode and its
latency targets are not established by these documentation commits.

Plan: first deliver complete B1Q6 rounds below 40ms; ordinary target 34ms,
hard target 25ms are later goals. Commit messages directly record usable
scope, measured results, limitations, prior usable hashes and rollback
compatibility so recovery does not require rerunning historical experiments.

Checks: exact UTF-8 readback and staged diff --check passed; AGENT.md only.
No device experiment or runtime/performance change.
Rollback: revert this commit to restore prior plan text only. No rebuild,
library replacement, configuration migration or restart required.
```

---

```text
decode: add graph-safe window primitives with 80 CPU-reference cases

Rollback: standalone leaf checkpoint only. Default Engine/prefill unchanged
from 4f53222; missing native runtime still prevents full-model decode.
NOT a <40ms full-round checkpoint. No deployed library replaced.

910B3/CANN9.0.1 Q6 B=1/2/3/4/8, 16 changed-input graph cases each pass:
accept/sharded BF16 embedding/HC PRE exact vs CPU, rotary abs error <0.002.
Covers inactive rows, bad slot, changing positions, acceptance and shard misses.
200 replays/width: torch allocated and peak unchanged. Native kernels allocate
nothing. Leaf-only B1/B4/B8 wall averages .0266/.0373/.0449ms.
TP, real weights, KV commit and full-engine allocator behavior untested.

Build: bash ops/decode/build.sh ops/decode/window.cpp /tmp/decode_window_20260927/libwindow.so
Test: TASK_QUEUE_ENABLE=0 python /tmp/decode_window_20260927/test.py
Evidence: /tmp/decode_window_20260927/result.json complete=true, five widths.
Source and build script versioned together; no runtime compatibility change.
```

---

```text
paging: eliminate per-ensure device allocation with persistent host staging

Rollback state: default PageTable now copies preallocated CPU page IDs into
existing device table. Prefill/decode contracts unchanged; no library upgrade.
CPU nine tests pass including 7200 randomized operations against baseline and
oracle; storage gate covers 1680 torch ops. NPU 910B3/CANN9.0.1: 200 updates
across 4 slots, release/reuse, 1..32768 positions pass; allocated/peak/after
all 1024 bytes (0-byte transient increase in torch allocator vs old 512).
NPU row IDs exactly equal owned IDs. No full-model regression in this commit.
Blocking H2D guarantees host staging reuse safety, may add page-boundary sync;
no latency improvement claimed. Host staging adds 8*max_pages bytes/table.

Reproduce: python /tmp/test_paging_ensure_20260927.py (CPU env)
python /tmp/test_paging_npu_20260927.py (CANN env)
Logs: matching .log files in /tmp. Single NPU, no TP8 concurrency.
```

---

```text
decode: require explicit build inputs instead of missing legacy defaults

Rollback: source/ABI/runtime unchanged; build CLI requires source and output.
CANN9.0.1 explicit window build succeeds; missing args rejected before compile.
No kernel change or fresh device performance claim.
```

---

```text
decode: bind window leaves to persistent caller storage and current stream

Rollback state: standalone Window binding usable with 6cdea46 window ABI;
Engine/prefill not rerouted, full model decode remains incomplete. No library
deployment required beyond building window.cpp for this caller.
910B3/CANN9.0.1 non-default stream graph tests pass B1/2/3/4/8 Q6 x16
changing-input CPU-reference cases, alias rejection and five resident graphs
reverse replay. 200 graph replays per B: no allocator high-water growth.
Leaf-only latencies ~0.027/0.031/0.035/0.037/0.045ms, NOT full round.
Reproduce: python /tmp/decode_window_20260927/test_binding.py (exit0).
Evidence: binding_result.json and test_binding.log in same directory.
No real weights, TP8, KV commit or 40ms full-model acceptance claimed.
```

---

```text
decode: add independent RMSNorm and trailing RoPE leaves

Rollback: leaf-only advance from 551bf4e. Default Engine/prefill unchanged;
full-model decode still unavailable, no full-round latency claim.
910B3/CANN9.0.1 B1..4 Q6: 384 device cases pass on default and non-default
streams, repeated calls, in-place/out-of-place, output canaries. RMS dims
128/512/1280/5120 weighted/unweighted; RoPE heads1/4/16/128 dim128/512,
forward/inverse. Independent numpy oracle atol=.002 rtol=.01.
CPU oracle selfchecks additionally compare FP64/complex128.
Graph/concurrent-stream execution, B8 device, integration NOT yet tested.
Build ops/decode/norm.cpp with ops/decode/build.sh; exports dec_rms_norm and
dec_rope, MAX_BATCH default4; B8 requires rebuild. No installed lib replaced.
Reproduce TASK_QUEUE_ENABLE=0 python /tmp/decode_norm_leaf/device_test.py --run-npu.
Graph owners must retain buffers; caller supplies stream and preallocated outputs.
```

---

```text
decode: bind explicit preallocated GEMM plans for Q6

Leaf checkpoint over b6f27f2, not a working full decode checkpoint.
Engine/prefill unchanged. BF16 input/physical NK weights; caller selects
BF16 or FP32 output, supplies workspace and owns graph/plan lifetime.
No hidden prefill path or allocation in launch. Uses CANN libopapi directly.
910B3 CANN9.0.1 B1/2/3/4/8, four NK shapes 4x5120,1280x5120,
1024x4096,16160x5120 x two output dtypes: 40 captured plans, three changed
inputs each vs FP64; 100 replays each with zero allocator peak increment.
Gates: FP32 abs<=2e-4; BF16 error vs real FP64<=half output ULP+2e-4.
Original fixed .02 BF16 gate failed at B3 vocab (.03125 BF16-vs-BF16);
diagnosed true value 4.79687403038 at rounding midpoint. Old failure retained.
These rounding tests do not establish full-model token correctness.
Leaf event times 0.0196..0.1378 ms, not end-to-end speed.
No custom binary changed; reload Python to rollback. Previous leaf b6f27f2.
Reproduce python /tmp/decode_gemm_20260927/test_rounding.py;
real weights/TP communication/workspace>0/concurrent plans not tested.
```

---

```text
decode: retain canonical FP32 RMS weights without lossy conversion

Previous leaf checkpoint aa25f26 used BF16 RMS weights, incompatible with
canonical rank0 q_norm/kv_norm FP32 cache storage. This version takes FP32
weights directly; BF16 activations/results and RoPE remain unchanged.
ABI intentionally renamed to dec_rms_norm_f32 so stale BF16 callers fail
at symbol lookup rather than silently reinterpret memory. Rebuild norm.cpp
and rebind/recapture before use; previous libdecode_norm.so is incompatible.

910B3/CANN9.0.1: 384 RMS/RoPE cases passed, two sequential streams,
in/out of place, repeat execution and canaries; B1..4, RMS widths
128/512/1280/5120. Fresh FP32 random weight CPU oracle, no truncation.
Repro: /tmp/decode_norm_fp32_20260927/device_test.py --run-npu.
No graph/concurrent-stream or full-model performance claim. This remains
a leaf checkpoint, not a working full decode or <40ms complete round.
Default Engine/prefill unaffected; revert source AND leaf library together.
```

---

```text
decode: expand canonical INT8 weights in fixed-storage projection graphs

Rollback: independent decode leaves only; default Engine/prefill unchanged.
Pairs with bba65b6 FP32 RMS ABI and aa25f26 Matmul. No existing deployed
library replaced. Build gemm.cpp with ops/decode/build.sh to caller path.

910B3/CANN9.0.1 rank0 layer0 real cache weights (manifest SHA256 checked):
B1/2/3/4 Q6 x wq_a/wkv x four changing inputs =32 cases.
INT8 row expansion matches CPU BF16 exactly; GEMM checked against FP64
half-ULP+2e-4 bound; RMS uses canonical FP32 weights and independent FP64
reference, max observed difference 0.001953125. 100 captured replays per
chain have no torch allocator peak increase. B1 wq_a 0.07933ms and
wkv 0.05558ms, each includes expansion+GEMM+RMS (NOT a full round).
Reproduce /tmp/decode_projection_20260927/test_chain.py; result.json complete.

Limits: K 32-aligned <=5120, caller-owned aligned non-overlapping buffers.
Only rank0 layer0 Q/KV chain tested; no attention/Past/MoE/TP8/target-draft
integration or end-to-end speed claim. Complete B1Q6 <40ms still unfinished.
```

---

```text
decode: add read-only Q6 SWA over canonical ring and pending rows

Rollback checkpoint: independent attention leaf; default Engine/prefill unchanged.
Supports BF16 B1..B4 Q6 with dynamic int64 slot/start/active and FP32 sink;
512-d latent keys, 128 causal positions, explicit modulus/pad, current stream.
Validated canonical modulus144/pad16 and alternate160/32: each 32 graph cases
against FP64 CPU attention with BF16 half-ULP+2e-4, ring bitwise unchanged.
100 replays per B had zero torch allocator peak growth; all four graphs retained
and replayed with exact output repeatability. Scripts/results:
/tmp/decode_attention_20260927/test{,_canonical}.py and *result.json.
B1 leaf ~0.4ms, B4 ~1ms: correctness baseline, NOT optimized full-round latency.
No QDQ/projection/global sparse/TP8/full model or <40ms claim.
Caller validates disjoint storage ranges and device metadata lifetimes.
New dec_swa_q6 ABI; rebuild with ops/decode/build.sh; no legacy binary compatibility.
```

---

```text
decode: replace per-key SWA global barriers with dependency events

Rollback choice: builds on aaf0d02 read-only SWA correctness baseline.
MTE2_V/V_MTE2 protect BF16 staging reuse; PIPE_V orders vector ops;
V_S/S_V protect ReduceSum/Exp scalar access. Entry/output retain full sync.
Same canonical144+16 ring graph test, B1..4 each 8 dynamic input cases,
FP64 oracle BF16 half-ULP+2e-4 passes, ring unchanged, four retained graphs
replay exactly, 100 timed replays zero torch allocator peak growth.
Leaf ms before -> after B1 .404951 -> .106432, B2 .627561 -> .147744,
B3 .626420 -> .148071, B4 .974718 -> .226106. Max error statistics unchanged.
Reproduce /tmp/decode_attention_sync_20260927/test.py; result canonical_result.json.
No full-model/TP8/CSA or full-round latency validation. ABI unchanged; rebuild
libattention.so with ops/decode/build.sh. Prefill/default Engine unaffected.
```

---

```text
decode: add fixed-buffer local KV FP8 round trip

Previous usable leaf checkpoint: fb48097. Adds dec_kv_qdq in attention.cpp;
finite BF16[B,6,512], block32 power-of-two scale and E4M3FN RNE, with
exact in-place or disjoint output. Partial overlap rejected.
NPU B1..4 x 2 alias modes x 7 inputs x 3 repetitions (168 checks) passed
bitwise against independent CPU torch float8 reference, including signed
zero, subnormals and midpoint ties. Captured replay of each variant passed;
100 replay timing 0.01856-0.02051 ms, torch allocator peak delta zero.
This is a KV leaf checkpoint, NOT a full decode or <40ms engine result.
No TP8, attention-chain integration, nonfinite inputs or concurrent streams
validated. Default Engine/prefill unchanged. Rebuild attention.cpp using
ops/decode/build.sh and recreate graphs before using new symbol; rollback
to fb48097 removes this ABI addition. Evidence: /tmp/decode_kv_qdq_20260927.
```

---

```text
decode: publish accepted Q6 pending KV into canonical Past rings

Rollback selection: adds dec_pending_ring only; existing leaf ABI unchanged.
CPU kernel shim/oracle 1212 cases, invalid-host 15, prefill-tail 2 and
cold-gate 2 pass (CPU does not validate DMA). NPU0 B1/2/3/4 across
ring/pad 144/16,160/16,145/32: 1968 eager/captured/accept->publish and
accumulated-history cases pass bitwise including full-history/guard checks.
Dynamic rejection/error/inactive/duplicate slots do not publish. Pending
and result are unchanged. 32 graph replays have zero torch allocator growth.
Evidence: /tmp/decode_pending_ring_20260927/main_npu_result.json and test_npu.py.

Caller owns disjoint fixed buffers and supplies current stream. Only ring
KV publication: no position/carry/source-page transaction, no replay flag
updates. Cold slots must complete replay before activation. No whole-model
integration, whole-round correctness or latency claim. Rebuild window .so
to use new symbol; no weight ABI change.
```

---

```text
decode: bind explicit W8 projections to caller-owned shared scratch

Rollback index: previous c4c6648 retains verified pending-ring publication; this adds only W8Matmul in existing ops/decode/gemm.py, no new files or native ABI changes. Caller supplies library, BF16 expansion storage and GEMM workspace. Sharing requires serialized launches; no runtime tensor allocation or cached expanded weights.

Validated rank0 real layer0 attention math chain, B1Q6..B4Q6, four synthetic input/phase/history cases each, stagewise CPU references and graph replay. Mean device graph ms over 100 replays: B1=0.376572, B2=0.434597, B3=0.490422, B4=0.542448. Borrowed expansion buffer 13,107,200 bytes (12.5 MiB); ACLNN requested zero workspace for these shapes; measured torch peak delta zero. Not evidence for nonzero-workspace sharing or total native allocator use.

Limits: no production NativeDecode integration, TP reduction, global sparse/HC/MoE/draft, independent end-to-end HF oracle or complete-round latency. Full B1Q6 <40ms remains unproven. Same dec_w8_expand library ABI; restart callers after Python rollback/update, no C++ rebuild for this change. Repro: /tmp/decode_swa_binding_20260927/test.py and result.json.
```

---

```text
decode: bind local Q6 attention stage with caller-owned buffers

Rollback selection: parent 52686e5 retains W8 shared-scratch projection binding; this adds bind_swa_stage to existing model/decode.py only. Explicit partial stage, not an alternate complete decoder. Five projection plans, tensors and libraries supplied by caller; output remains pre-TP FP32; canonical Past is read-only. No native ABI change.

Validation: rank0 actual layer0 weights, B1Q6 through B4Q6, four synthetic input/phase/history cases per batch, stagewise CPU references and captured graph replay. Device graph averages ms: B1=0.374760, B2=0.429358, B3=0.485448, B4=0.533325. Torch peak allocation delta zero; shared expansion 12.5 MiB. Repro /tmp/decode_swa_stage_20260927/test.py and result.json.

Limits: not independent end-to-end HF validation, no production Engine hookup, TP reduction, HC/MoE/CSA/draft or complete-round latency. Old NativeDecode weight/ring ABI still incompatible and is not silently selected by this function. B1Q6 full-round <40ms unproven. Rollback Python source and rebuild graph callers; no native rebuild needed.
```

---

```text
decode: fuse Q6 HC collapse and RMS with mandatory BF16 rounding

Rollback point: standalone dec_hc_norm added to existing norm.cpp; previous RMS/RoPE entrypoints unchanged. Rebuild decode norm library to use new symbol. Caller supplies disjoint residual BF16[B,6,4,5120], FP32 pre/weight and BF16 output; B1..4, current stream, no workspace or allocation. Collapse performs ordered FP32 accumulation and BF16 rounding before RMS.

Validation: atlas-a2c single NPU0, B1..4 x eps(1e-6,1e-20) x8 changed inputs =64 graph cases, CPU sequential FP32 collapse + FP64 norm oracle, guarded output unchanged. 100 replays/group show torch allocator peak delta0; leaf device means 0.01716..0.02114ms. Includes zeros, cancellation, small magnitude, negative gates/weights.

Original test failed 42 values at BF16 power-of-two boundaries: symmetric neighboring-spacing average underestimated the wider rounding interval. Diagnostic entire failed tensor equals both rounded CPU FP32 and FP64 references bitwise. Kernel unchanged; corrected test uses directional half-ULP +3e-6*abs(ref)+2e-5. Original test/log/failure.pt preserved, not claimed to pass.

Evidence: /tmp/decode_hc_norm_20260927/{test.py,test_corrected.py,diagnose.log,corrected_result.json}; library SHA256 9ff79db8677b065d12719219b20fe5ce5766946c39a8c5b5b544ae9d970cc5f1. Limits: isolated HC collapse/RMS only, synthetic inputs; no HC mixing/projection, TP, full layer or complete decode performance claim.
```

---

```text
decode: fuse HC gates and Sinkhorn into a fixed-buffer Q6 leaf

Rollback point: add dec_hc_gates to existing norm.cpp, previous entrypoints unchanged. Rebuild norm library for new symbol. Caller supplies FP32 projection z already scaled by inverse RMS, scale[3], base[24], and disjoint pre/post/comb outputs. B1..4, current stream, no allocation or workspace. Computes sigmoid pre+eps, 2*sigmoid post, row softmax+eps, initial column normalization then iters-1 row/column pairs. Does not perform HC projection or inverse RMS.

Validation: atlas-a2c NPU0, B1..4 x iters(1,2,20) x6 changed inputs=72 graph cases; independent CPU FP32 formula, atol2e-6/rtol2e-5. Includes zero, small/random/large logits and zero scales/base. Output guards unchanged, 100 replays per group torch allocator peak delta0. Max absolute errors pre/post/comb: [1.1920928955078125e-07, 2.384185791015625e-07, 1.4901161193847656e-06]. iters20 B1/B4 leaf means 0.02284/0.02600ms; not complete-layer timings.

Evidence: /tmp/decode_hc_gates_20260927/{test.py,test.log,result.json}; library SHA256 be2d60379ec4de2d12e16964396f092e808ab41755971a24f661cb560faccd82. Limits: synthetic pre-normalized projections only; no real-weight HC integration, full layer, TP or complete decode performance. ABI accepts iters1..100 and positive finite eps; device tests cover iters1/2/20 and eps1e-6 only. Previous rollback 06c97e8 contains tested HC collapse+RMS without this gate leaf.
```

---

```text
decode: support explicit FP32 operands in fixed-address GEMM

Rollback point: retain existing BF16 and W8 interfaces; accept same-dtype FP32 inputs only with FP32 output. No casts, new files, fallback, or changed native ABI. Existing descriptor dtype mapping already supports FP32.

NPU0 validation: B1..4 Q6, HC-shaped [6B,20480] x [24,20480], six synthetic cases each, CPU FP64 reference atol2e-5/rtol2e-5, 16 bit-identical repeats per case; 100 captured replays per batch, torch allocator peak delta zero. Workspace 22446592 bytes per plan, caller owned: budget max workspace for serialized plans, not sum per layer. FP32 graph means [0.03836119890213013, 0.03682820081710816, 0.037164599895477296, 0.037226600646972655] ms.

Regression: real rank0 layer0 BF16/W8 attention chain B1..4 x4 cases passes, zero replay allocator peak delta. Evidence /tmp/decode_hc_fp32_20260927/{test.py,result.json} and /tmp/decode_swa_fp32_regression_20260927/{test.py,result.json}. Limits: synthetic HC projection only, not real-weight HC integration or complete layer; no claim of full round <40ms. Previous rollback 829dfa6 has HC gates but BF16-only Matmul.
```

---

```text
decode: prepare FP32 HC residual statistics and normalize projections

Adds dec_hc_prepare (BF16 residual -> FP32 cast plus inverse RMS) and
explicit in-place dec_hc_scale. Caller-owned aligned disjoint buffers,
current stream, Q6 B1..4. No allocation or hidden fallback.

Evidence: /tmp/decode_hc_integrated_20260927/test.py and result.json.
Real rank0 layer0 attention/FFN HC weights, cache SHA256 verified;
B1..4 x both kinds x 4 synthetic scales = 32 cases complete.
CPU FP64 oracle: stats atol/rtol 2e-6; projection/gates 2e-5.
Sixteen identical replays per case; allocator peak delta zero.
B1 prepare+projection+scale+gates: attention 0.0737284ms, FFN 0.0716196ms.
GEMM workspace 22446592 bytes; no collapse or residual expansion timed.

Rollback/ABI: additive exports; rebuild ops/decode/norm.cpp. Existing
exports unchanged. FP32 GEMM requires 7abe844. Tested binary SHA256:
43f6c55440801f715059e5d674c0ff0737b2ffe2d42f1f01c885d9be4ae4f1d3
No production model binding yet; synthetic residuals, not HF full-model
validation. TP/MoE/CSA/DSpark and <40ms full-round remain unproven.
```

---

```text
decode: bind fixed-buffer HC mixing in existing model module

bind_hc_mix_stage validates shapes/dtypes/device, disjoint storage including
workspace, and bound FP32 projection identity. Retains buffers/library and
uses current stream. No device allocation or implicit fallback.

Evidence /tmp/decode_hc_binding_20260927/{test.py,result.json,test.log}:
real rank0 layer0 attention/FFN weights with synthetic residuals;
B1..4 x 2 kinds x 4 scales = 32 cases, CPU FP64 oracle passed.
16 identical replays per case; 100-replay allocator peak delta zero.
B1 attention 0.0800012ms, FFN 0.0775138ms. Workspace 22446592 bytes.
These are HC prepare/projection/scale/gates timings only, not full decode.

Rollback: additive Python binding; requires 19aa3e1 norm exports and
7abe844 FP32 GEMM. Existing NativeDecode path not changed or made compatible.
No collapse/expand or Engine integration; TP/MoE/CSA/DSpark and full B1Q6
<40ms remain unfinished. No new repository files.
```

---

```text
decode: add fixed-buffer Q6 HC residual expansion

Adds dec_hc_expand: out[target] = post[target]*x + sequential sum of
residual[source]*comb[source,target], FP32 arithmetic then BF16 rounding.
Q6 B1..4, caller-owned disjoint output, current stream, no allocations.

Evidence /tmp/decode_hc_expand_20260927/{test.py,result.json,test.log}:
4 batches x 10 synthetic cases, bit-exact versus CPU sequential FP32/BF16.
Includes asymmetric permutation (orientation), cancellation, signed gates,
zero and 0.001/100 amplitudes. 16 replays/case deterministic; guards and
inputs unchanged. Rejects alias, misalignment and invalid batches.
100-replay allocator peak delta zero. B1 0.018632ms, B4 0.022731ms.
Large max_rounding_abs is BF16 quantization at amplified inputs, not an
oracle mismatch: all 40 cases have zero BF16 bit mismatches.

Rollback: additive norm library export; rebuild ops/decode/norm.cpp.
No new repository files or prefill changes. Synthetic leaf test only;
not full HC/attention/layer or Engine integration. Existing NativeDecode
ABI remains incompatible; MoE/CSA/TP/DSpark and B1Q6 <40ms unfinished.
```

---

```text
decode: bind HC collapse-normalization and residual expansion

Adds bind_hc_residual_stage returning two explicit fixed-buffer callables.
Caller inserts its actual attention/FFN sublayer between them. Validates
shape, dtype, alignment, device and disjoint storage; retains buffers and
library; uses current stream. No allocation, synchronization or fallback.

Evidence /tmp/decode_hc_residual_binding_20260927/{test.py,result.json}:
B1..4 x layer0 attention/FFN HC weights x four input scales = 32 cases.
HC gates checked against CPU FP64. Collapse/RMS and expansion checked by
local-rounding CPU oracles consuming separately verified device gates;
expansion bit-exact, normalization directional-half-ULP tolerance passed.
16 deterministic replays per case, input preservation, overlap rejection,
100-replay allocator peak delta zero. B1 combined HC-only chain:
attention weights 0.093763ms, FFN weights 0.090995ms; workspace 22446592 B.

Rollback: additive binding requires 1f1068a norm exports and existing HC
mix binding. No new files or prefill changes. Inputs, norm weights and
sublayer outputs are synthetic; this is not a real attention/FFN layer.
No Engine integration or full-round speed claim. NativeDecode ABI still
incompatible; MoE/CSA/TP/DSpark and B1Q6 <40ms remain unfinished.
```

---

```text
decode: bind explicit FP32 TP sum and BF16 HC output boundary

Adds dec_tp_cast and bind_tp_output using the caller-owned TP8 communicator.
No new communicator or repository files; fixed disjoint buffers, current
stream, no hot-path allocation. Documents that collapse consumes PRE from
the previous sublayer rather than the current HC mix.

Rollback checkpoint: TP8 layer0 real-weight HC -> collapse/RMS -> SWA ->
FP32 SUM -> BF16 -> residual expansion passes B1..4, four changing cases
per batch on every rank. /tmp/decode_hc_tp_chain_20260927/test.py and
result_rank0..7.json; all ranks complete and processes exited.
Independent gathered local totals agree with SUM within max abs 6.17e-7;
BF16 cast and HC expansion bit-exact, incoming PRE/ring unchanged,
wrong-PRE negative control and deterministic replay pass; allocator peak
increase zero. B1 chain 0.547634..0.547662ms; B4 0.784065..0.784127ms.

Scope: real layer0 weights but synthetic residual/PRE/positions/history.
Not full-engine integration, FFN/CSA/DSpark or a <40ms full-round result.
Revert this commit to a0fc6d6 if TP output boundary breaks; that checkpoint
has validated HC residual binding but no integrated TP output boundary.
```

---

```text
decode: assemble fixed-buffer HC SWA attention with explicit TP boundary

Adds bind_hc_swa_attention in existing model/decode.py. Retains caller-owned
plans/storage, rejects PRE input/output alias and mismatched HC/projection
edges. Executes mix -> incoming-PRE collapse -> SWA -> FP32 TP sum -> BF16
cast -> residual expansion. No new repository files or hidden fallback.

Rollback scope: assembled real layer0 attention weights on TP8, B1Q6..B4Q6,
4 dynamic input/history cases per batch per rank. All 8 ranks completed;
wrong PRE/post/projection bindings rejected; peak allocator delta zero.
Observed max rank graph times (ms): B1 0.543948, B2 0.648657, B3 0.716072,
B4 0.778314. FP32 reduction vs gathered reference max abs <=6.166e-7.
Evidence /tmp/decode_hc_swa_bound_20260927/test.py, test.log,
result_rank{0..7}.json. Numerical assertions inherited from TP chain test.

Boundary: synthetic residual/incoming PRE/positions/history; not an
independent full-model oracle. No FFN, CSA, DSpark or Worker replacement;
this is a bound attention sublayer, not a complete Engine decode round.
No claim that B1Q6 full-round <40ms has been achieved.
```

---

```text
decode: add fixed-buffer SwiGLU and double-rounded MoE finish

Adds decode-only dec_swiglu and dec_moe_finish in existing norm.cpp.
SwiGLU clamps gate only above and up symmetrically, evaluates in FP32,
then rounds to BF16. Finish requires already TP-summed FP32 inputs and
rounds shared through BF16 before adding routed and rounding the result.
Caller-owned disjoint output, current stream, no allocations or fallback.

Rollback scope: independent FFN leaves only, not shared/routed expert
projection integration, Worker integration, or a complete decode engine.
Evidence /tmp/decode_ffn_leaves_20260927/{test.py,result.json,test.log}:
72 activation cases (B1..4 width288 plus B1 widths32/4096; limit0/10),
24 finish cases B1..4; observed CPU BF16 bit mismatches zero in both.
Activation test tolerance allowed rtol .008 / atol 1e-7 for exp backend
variation; zero observed mismatches is not a universal bit-exact claim.
Finish asserted exact bits and detected omitted intermediate rounding.
Input preservation, output guards, alias/geometry rejection, 8 repeated
replays/case and peak allocation delta zero passed. Local leaf replay
~.019-.021ms; no full FFN/engine latency claim.
```

---

```text
decode: bind caller-owned shared expert projections and TP reduction

Rollback boundary: shared expert binding only; no routed experts or Engine
integration. Three fixed-buffer W8 plans execute w1/w3 -> clipped SwiGLU ->
w2 -> FP32 TP8 sum, leaving explicit MoE double rounding to the next stage.
No new repository files, allocations or communicator in the replay path.

/tmp/decode_shared_expert_20260927: all 8 ranks completed B1..4Q6 x4
synthetic hidden-input cases with real layer0 weights. Stage-local FP64 CPU
projection references, gathered local-sum reference, input/guard checks,
negative limit/alias checks and repeated deterministic replay passed.
Peak allocation delta zero; max TP error 2.98023224e-07.
Max-rank graph ms B1..4: [0.2976717948913574, 0.30734899520874026, 0.31464080810546874, 0.3379764175415039].
Test SHA256 2ded4b7702dde5550be4c45025a301d0c34a995297c8c28d6ca21aded69a6774.
Not a complete FFN or full-round timing; <40ms remains unverified.
```

---

```text
decode: add target 384-to-6 routing leaf with corrected softplus

Adds dec_route in existing norm.cpp: FP32 logits and correction bias to
INT64 IDs/FP32 probabilities. Bias affects selection only; sqrtsoftplus
scores are normalized then scaled. Fixed buffers/current caller stream,
no allocations. Compensated log(1+t) preserves negative-logit scores.

Rollback checkpoint (leaf only): /tmp/decode_route_20260927, B1..4Q6 x8
synthetic cases passed; IDs matched CPU on non-tie cases, all-equal ties
observed [0,1,2,3,4,5], repeated replay identical. Probability max absolute
error 8.941e-8. Guards/input preserved, invalid host params/alias rejected.
30 replay event means B1..4: .024637/.020935/.021992/.022509 ms; peak delta 0.
Tests use temperature=1, scale=1.5; other valid values not yet tested.
Not router projection, expert GEMMs, full MoE or full engine. Draft 128/3
not supported by this target ABI. Full B1Q6 <40ms remains unverified.
Previous usable shared-expert point 269f16b; rebuild norm.cpp library and
recreate bindings/graphs after source changes or rollback. No new repo files.
```

---

```text
decode: bind FP32 target router projection and explicit hidden widening

Adds dec_gate_cast and bind_target_route in existing norm.cpp/model/decode.py.
BF16 hidden is explicitly widened into caller-owned FP32 storage before
FP32 GEMM and target 384/6 routing; caller chooses text/vision bias. No
allocation, implicit weight cast, new communicator or repository file.

Rollback checkpoint: /tmp/decode_target_route_20260927 completed B1..4Q6
x text/vision biases x (temperature,scale)=(1,1.5),(.5,.75),(2,2):24 cases.
Real layer0 replicated router weights on rank0, synthetic BF16 hidden.
CPU FP64 projection oracle: max logit abs error 2.146e-6; selected IDs
match and max probability abs error 1.193e-7. Widening bit-exact; guards,
input preservation, repeat replay, invalid temperature/alias checks pass.
30-replay event means default B1..4: .038223/.042078/.100257/.103287 ms.
Allocator peak delta zero. Not expert GEMMs/full MoE/full Engine, no
full-round <40ms claim. Draft gate unsupported. Previous route-only
checkpoint dd998e5; rebuild norm library and recreate bindings/graphs
after rollback. Test library SHA256: 61b33dc4432fb14bd4419593f02294f22eef858368f368950a5f5a7ed069257b
Test script SHA256: c4a82a9f60cdd5d80449e3605c0198573d6db13d4874daca93d4b2b4789e3227
```

---

```text
decode: add stable fixed-buffer expert dispatch with single-owner metadata DMA

Rollback scope: independent dispatch leaf only, not routed GEMMs or complete FFN. B1..4Q6, six choices, IDs must be in [0,384), stable token/choice order within each expert. Caller owns all storage; cumulative INT64 ends for grouped GEMM.

/tmp/decode_dispatch_20260927: all four batches x6 cases bit-exact for rows, probabilities, inverse and ends; guard and graph checks passed, replay peak delta zero. Graph ms .056858/.100444/.193701/.323583. Library SHA256 d2506c7094aa3c6ffb78aaba9d9825b45998f1ad1264635edaf546622c115ddb.

Initial cross-core GM scalar stores lost probability/inverse updates while row copies and ends passed. Single-owner UB metadata with exact-length DMA fixes tested cases. Scalar quadratic ranking remains a performance limitation, especially B4; optimize next. No full-engine latency claim. Requires rebuilt norm library exporting dec_dispatch; existing entry points retained.
```

---

```text
decode: replace quadratic dispatch ranking with stable counting sort

Rollback scope: same dec_dispatch ABI and caller buffers as 7e3f990; histogram/cursors replicated in per-core UB, linear O(384+pairs) metadata work, core0 alone DMA-writes shared metadata. No allocations or cross-core scalar stores. Existing prefill unchanged.

/tmp/decode_dispatch_linear_20260927: B1..4Q6 x6 synthetic cases all outputs bit-exact, guards pass, graph replay peak delta0. Graph ms .043197/.036596/.039555/.042481 versus .056858/.100444/.193701/.323583 in prior leaf test (1.32/2.74/4.90/7.62x). Separate-run leaf timings, not full-engine speedup. Library SHA256 7735bb4b2d40a0474bcd2d67ffcad8b088999e8c0f37c3f7a31c11e336a233d4.

IDs must be [0,384); fixed six choices and 5120 hidden. Stable dispatch only; expert dequant/GMM/FFN integration and complete B1Q6 latency still unfinished. Rebuild norm library; ABI unchanged.
```

---

```text
decode: add probability-weighted routed activation and ordered combine

Rollback boundary: additive dec_routed_act/dec_routed_combine in norm.cpp;
router/dispatch ABI unchanged. Fixed caller buffers; B1..4 Q6, six choices,
288 intermediate, 5120 hidden. Clip gate at +10 and up at +/-10; multiply
probability BEFORE final BF16 round. Combine BF16 down outputs in FP32
ascending grouped row order, not original choice order. TP is external.

Evidence: /tmp/decode_routed_leaves_20260927/{test.py,result.json}, rank0
B1..4 x6 synthetic cases PASS. Activation CPU BF16 mismatches=0; combine
bit-exact. Guards/inputs unchanged; repeated replay deterministic; invalid
batch/alias/alignment rejected; allocator peak delta=0. Premature-round
negative control changes 92198 elements; choice-order control 336021.
Activation max error vs UNROUNDED FP32=.499588 within directional half
BF16 ULP + FP32 tolerance (zero mismatch vs rounded BF16 oracle).
Two-leaf graph ms B1..4: .028609/.029265/.031475/.034793.
Library SHA256 1b2b0cad92701d4df9c52157501e5b8c2da2513067a210d714de03a466f513d8.

Limits: standalone synthetic leaves; valid device indices required.
No routed dequant/GEMM binding, TP8 FFN or full-round <40ms evidence.
Revert to c4a5fc0 to remove these leaves while retaining stable dispatch.
```

---

```text
decode: add explicit active signed INT4 bank expansion

Rollback boundary: additive dec_w4_expand in existing gemm.cpp; no GEMM or engine binding change. Caller-owned BF16 bank, signed nibble LUTs, FP32 channel scales, device cumulative ends; inactive experts untouched. Fixed E384, w13 N576 K5120 / w2 N5120 K288. Narrow-row packing, AIV guard, explicit MTE2_S before scalar reads.

Evidence: /tmp/decode_w4_20260927/result.json; actual layer0 rank0 manifest weight SHA verified; 2 banks x6 cases (empty, boundary experts, scattered,144 active), bit-exact CPU expansion, guards and inactive sentinels PASS; capture replay peak delta 0. Active 6/36/144 leaf ms: w13 .150223/.814768/3.182573, w2 .095849/.449625/1.908247. Not expert GEMM/full FFN or complete decode timing; expansion traffic remains a performance risk for <40ms full round. Library SHA b8805a0c24558554d8377f0f26a2f63fb6b44f5de3714b5d8761eaf0b4d69dc1.
```

---

```text
decode: bind fixed-address grouped expert GEMM with explicit workspace

Rollback boundary: additive GroupedMatmul in existing ops/decode/gemm.py; Matmul/W8Matmul and model binding unchanged. BF16 bank transposed by descriptor, INT64 device cumulative ends, repeatable executor, caller workspace; no route readback or implicit expansion. TensorList owns member descriptor; close destroys executor first and is idempotent.

Evidence /tmp/decode_grouped_20260927/result.json: two real-size geometries x B1..4Q6 x3 synthetic exact-binary cases (all expert0, all expert383, scattered0/1/191/383), CPU BF16 bit-exact; guard/replay/peak0. Workspace each 16777728 bytes, serial plans may borrow same storage. Four-active-expert leaf graph ms: w13 .074241/.068396/.079254/.069816; w2 .058729/.061168/.060666/.055478. Not real INT4 routed chain or full FFN/engine validation. Source SHA eda2c381f38ec748cb914c4466f59a737918ebf6da3feb5b38ba54dcf73d916e.
```

---

```text
decode: bind explicit rank-local routed expert chain

Rollback boundary: additive bind_routed_expert in model/decode.py;
existing Engine execution unchanged. Fixed caller storage and shared
serial workspace; dispatch -> active W13 unpack -> grouped GEMM ->
probability-before-BF16 activation -> active W2 unpack -> grouped GEMM
-> ascending grouped-row FP32 combine. No hidden route read/allocation.

Evidence: /tmp/decode_routed_chain_20260927; real layer0 rank0 packed
weights with synthetic inputs/IDs, B1..4 x3 cases (6/spread/1 experts).
CPU intermediate oracle relL2 <=.02; observed output max 0.000479735842.
Sixteen same-input graph replays per case bit-exact; guards intact;
graph replay allocator peak delta zero. Single-expert timings only;
not representative routing performance. No gate, TP, shared-expert,
complete FFN/Engine or B1Q6 <40ms claim. Revert this commit to remove
the chain binding while retaining validated individual decode leaves.
Source SHA256 69b69047c2681fb25097eca66c364413096789591f6a5b58f7082ecffe2167f7
Weight SHA256 2496a210ad76d66e598d269c616ff502f60c2ba926c524b32a651f83cc24b808
```

---

```text
decode: bind target MoE gate and TP8 expert branches

Validated real layer0 TP8 weights with synthetic normalized hidden inputs: B1..4 x text/vision bias x8 ranks; CPU stage oracles, exact self-replays, guards, zero replay allocation. Worst output relative L2 0.000600693. Evidence /tmp/decode_target_moe_20260927_r2. Does not cover HC, full layer/engine or latency target.
```

---

```text
decode: compose HC target FFN with distinct incoming and outgoing PRE

Rollback scope: HC mix/collapse-normalize -> target MoE TP8 -> HC residual; caller-owned fixed buffers, shared serial workspace. No attention/KV/Worker/full-round claim.
Evidence: /tmp/decode_hc_ffn_20260927; real layer0 TP8 weights, synthetic residual and incoming PRE, B1..4 x text/vision (64 rank-cases). Independent CPU stage oracles; MoE oracle consumes observed normalized input, not an independent end-to-end HF run. Eight exact graph repeats, guards intact, graph peak allocation delta zero.
Worst relative L2: {"routed": 0.00027215410955250263, "shared": 0.0002293151628691703, "output": 0.0006103182677179575, "hc_pre": 4.078151505382266e-08, "hc_post": 5.006651804251305e-08, "hc_comb": 3.4905486501202176e-08, "hc_norm": 0.00012647060793824494, "ffn": 0.0001435561862308532, "expand": 0.0}
Max-rank graph ms by batch [text,vision]: [[2.121735954284668, 1.735849952697754], [3.3044979095458986, 2.6848560333251954], [4.584252166748047, 3.521292877197266], [5.347911071777344, 4.310041046142578]]; isolated FFN only, <40ms whole round unproven.
Validated model/decode.py SHA256 114fc63d14e2a4905b27ab9a23cbb975774ea77d50ceb02021e3957fe5a8077a
```

---

```text
decode: add paged CSA Q6 joint attention alongside SWA

Rollback scope: attention leaf only; preserves dec_swa_q6 ABI.
Adds dec_csa_q6 one softmax over ring, pending local/compressed rows,
selected paged compressed bank and sink. No persistent Past writes.

Evidence /tmp/decode_csa_20260927/test.py and result.json: single NPU,
B1..4 x ratios0/1/2 x3 metadata cases =36 passes, H8 D512.
Synthetic BF16 inputs vs independent FP64 CPU oracle; worst relL2
8.027678268263116e-05, max abs .00048828125. Shuffled pages/slots,
ring wrap, odd/even starts, future/padded IDs, missing pages, inactive
slots; 8 exact replays, guards/input immutability pass; replay peak0.
Source and library hashes verified. B4 ratio1 longest leaf ~1.076ms.
Not yet Python-stage or NativeDecode integrated; no TP8/full-layer,
DSpark or end-to-end B1Q6 <40ms claim.
```

---

```text
decode: bind fixed-resource target attention and FFN layer

Rollback scope: TargetLayer with shared serialized scratch, explicit CSA binding and projection edge validation.
Evidence: /tmp/decode_target_layer_20260927/attempt5/rank{0..7}.json; all eight ranks complete for B1Q6 layer0 real weights, synthetic input; eager/captured exact-repeat smoke and reported peak_delta=0. Source SHA256 6c279b4e89e1d8f103ce295f0a74fba455887a82400857b1a993cf83735ce41f.
Limitations: not full-model accuracy, no Engram or compressed-source integration in this test; NativeDecode old entry still not integrated; no end-to-end latency or <40ms claim. Head/source changes excluded.
```

---

```text
decode: add INT8 Engram row dequantization leaf

Recovery boundary: standalone dec_engram_rows_i8 implements INT8[18B,256] with FP32 block32 scales to BF16[6B,768]. Existing decode leaves unchanged; full Engram and complete decode are NOT integrated.

Device test: /tmp/decode_engram_20260927/test_rows.py; rows_result.json complete=true, batches 1,2,3,4 all exact=true and peak_delta=0. Built libnorm_rows.so; source SHA256 9adddfe00a8e9fdf0ecffb8f97369ddf28cd09b4062895b7107bffb9f89c2b37. No full-model correctness or latency claim.
```

---

```text
decode: add fixed-buffer Engram gate and BF16 widening

Rollback landmark: independent synthetic CPU formula vs NPU gate B1/B2/B3/B4, max absolute BF16 output differences 0.00048828125/0.000030517578125/0.0001220703125/0.00390625; mismatch fractions below 0.0001. Eager/capture exact, changed-input replay exact, five replays peak allocated delta zero. Evidence /tmp/decode_engram_20260927/gate_sync_result.json and test_gate_sync.py.

Fixed V-to-MTE2 dependency before reusing BF16 staging buffer: original gate had up to 4.19921875 error; explicit event removes corruption. Includes zero-norm cases. Rotation/projection/TP integration and full decode latency NOT established; caller must preserve rounding and supply rotated FP32 query.
```

---

```text
decode: wire native CSA source between HC collapse and attention

Rollback scope: replaces obsolete source FP8/arena ABI with explicit W8/BF16 SourcePlan, native compression/QDQ/scoring/selection, and a build-time TargetLayer source hook. No prefill fallback or canonical Past mutation.

Checks: source delivery reports native compilation and 59 CPU shim cases; main agent checked delivered attention.cpp SHA256 and executed 14 actual-AST CPU hook cases covering ordering and pointer/metadata rejection. Python AST and git diff --check pass.

NOT a device-validated or complete decode milestone: source NPU numerics, DMA ordering, TP8 capture and full-engine latency remain unverified. Previous 74b08bf retains device-tested Engram leaves; b97b5bc is the prior single-layer TP8 checkpoint. Engram integration, acceptance commit and DSpark runtime still pending.
```

---

```text
decode: bind fixed embedding entry with compact initial HC coefficients

Rollback boundary: source hook from 1d8935b retained. Adds shard embedding and BF16 TP SUM binding; legacy window PRE stays separate from compact constant initial PRE. 36 CPU contract cases passed: B1..4 x rank0..7, replay call order, layout/overlap/device and native-error rejection. AST and diff-check passed. No NPU numeric, capture or latency validation; NativeDecode arena runtime remains unconverted and full decode is not complete.
```

---

```text
decode: add accepted-prefix native Past transaction

Commit component passed 451 single-device NPU cases against independent full-storage host reference. CPU binding checks also passed. Excludes existing greedy edits; graph replay, TP8 integration and end-to-end timing remain unverified.
```

---

```text
decode: bind fixed-resource Engram with validated host staging contract

Module checkpoint: hash/raw INT8+FP32 gather and explicit resource/lifetime ABI. Main-agent rerun of deployed source: 9 CPU contract tests PASS (B1-B4 byte gathers/canaries, all-rank hash/history oracle, invalid resource/alias/workspace/device guards, failure cleanup and stream lifecycle). Native GEMM/ACL/HCCL are mocked: this is NOT device numerical, capture, TP8 or latency acceptance. Source SHA256 b1041c044070a22c6bf7bbf8bf5e9d38a0f528150eeee1d6fc92a9f355921672. Evidence: local engram_fixed_20260927/main_recheck.log and test_contract.py. ABI replaces retired owner/arena constructor; fixed-resource runtime migration is a separate pending commit. Do not treat this checkpoint as standalone full-engine enablement. Revert this commit together with any subsequent fixed-resource caller migration to restore the previous ABI. No prefill files changed.
```

---

```text
decode: deliver fixed-resource runtime lifecycle CPU milestone

Local milestone, NOT a usable full-decode rollback point. Fixed-resource B1-B4 runtime wiring plus device argument normalization, side capture stream dependencies, cleanup references, shared-transport close ordering. Main reran 6 CPU contract tests and 3 independent lifecycle tests; corrected pre-fix control fails all 3. No NPU runtime capture/numerics/performance or full-model build claimed. BLOCKER: canonical compressor weights FP32 versus SourcePlan BF16 contract. Depends on d9ee487 Engram ABI and pending NativeHead/DSpark/window changes; deploy as coordinated set and restart, matching native libraries required. Previous related HEAD d9ee487 (Engram host-only milestone), transaction core 8f7fc60; prefill known baseline 8cdca784. No <40ms claim. Evidence runtime_module_audit/REPORT.md and main_six_tests.log outside repo.
```

---

```text
decode: deliver greedy and accept single-device module gates

Current-source build and 66 device checks passed for B1/B4, sequentially emulating rank candidates. Does not validate HCCL, graphs, or complete model. Test: /tmp/test_greedy_module_20260928.py /tmp/greedy_module_current.so.
```

---

```text
decode: deliver bounded DSpark component acceptance

Fix grouped CANN V3 ABI and bind fixed-resource draft components. Device 0: five plan executions and 18 leaf checks. CPU: 1500 binding calls and 131584 FP8 values. Evidence: /tmp/dspark_device_accept_20260928/ACCEPTANCE.md. Not full NativeDSpark, TP8, graph capture, or production enablement.
```

---

```text
decode: bind SourcePlan to canonical FP32 weight storage

Fix startup contract rejecting canonical FP32 norm/compressor tensors as BF16. Borrow validated same-address FP32 storage; ratio1 BF16 projection and native runtime path unchanged. No extra device allocations or repository files.

CPU module gate: 5 tests PASS, independently rerun against stable source SHA256 76f3c40dbc346c06a1513072ee802015ea675111eef94e3811e0ccd04d104eb3. Includes 32 manifest-backed full/reindex builds, 36 fail-closed cases, lifetime/alias checks, and CPU arithmetic with 9 SHA-verified real weight units. Reproduce locally: cd source_contract_fix && python -B test_contract.py; evidence main_contract_recheck.log outside repo. CPU math is not device kernel validation.

Checkpoint scope: fixes SourcePlan construction/type-binding blocker only. NPU SourcePlan execution, HCCL, full decode graphs and latency NOT tested by this commit. Reverting restores the old incompatible BF16 gate.
```

---

```text
decode: wire fixed-resource DSpark lifecycle into target runtime

Adds preallocated draft resources, verify/seed/propose capture groups, explicit prefill-tail seed, slot-keyed receipts for rebatching, and accepted-target feature seeding. Worker no longer accepts externally supplied draft tokens.

Module gate: 12 CPU tests independently rerun on a stable SHA-verified deployed snapshot. Covers capture order, seed chunk alignment, receipt rejection, commit/seed/publication order, rebatching, and lifecycle. CPU streams/graphs/native calls are mocks; this does NOT validate NPU graph execution or full decode numerics.

Rollback boundary: parent 462243e retains accepted component gates but requires external drafts. This commit establishes runtime wiring only. Target feature extraction still uses torch.mean(out=); replacing it with a native leaf is pending. WindowPast currently uses a borrowed-field adapter. Full strategy scheduling, end-to-end TP8, and B1Q6 latency <40ms are not accepted here. No repository files added.
```

---

```text
decode: use native target feature extraction and canonical Past rings

Replaces target-layer 37/38/39 torch.mean binding with fixed-pointer dec_ds_features; removes the synthetic modulus adapter and consumes WindowPast.ring directly. No repository files added.

Module gates: device0 leaf 72 exact numerical cases, 13 invalid-input checks, 128 graph replays across B1-B4/two streams with stable addresses and allocated bytes. Final source SHA256 2faf199c17acfca81559c5c69748b5280458649594eca9a80a6dcb14334361e4; library SHA256 1a6f1ffb48b3e8c88bd588e7ee754b59fb950f887318b980eedb831baf054ad2. Main independently inspected raw results/source match; no duplicate device run. Integrated CPU wiring 12 tests pass; real WindowPast binding 1500 mocked leaf calls pass; feature closure pointers and indices checked.

Rollback boundary: parent 44bc807 has accepted CPU lifecycle wiring but uses high-level mean and a window adapter. Revert this commit as a unit for native feature/ring regressions. Rebuild norm library from this source or explicitly supply /tmp/decode_features_20260928/libdecode_norm.so via existing libraries mapping; default repository .so is absent. No full NativeDSpark/TP8 generation, scheduler or latency acceptance claimed.
```

---

```text
server: reject duplicate generation IDs before enqueue

Serialize duplicate check, CPU enqueue and handle registration to prevent orphan decode jobs. Missing request fields now return HTTP 400 before enqueue.

Component gate: 6 CPU tests passed (duplicate, 8-thread duplicate race, invalid fields, rejected query retry, completed stream release, disconnect cancellation). Real generate/events code with fake queue backend; no model/NPU/TP8 execution.

Rollback scope: this commit only changes server/engine_server.py request admission. Decode lifecycle and bootstrap remain separate; no full-decode or latency claim.
```

---

```text
decode: connect row lifecycle and B1-B4 batch scheduler

Single Engine/Past/NativeDecode; chunked prefill handoff, Q6 accepted output reconstruction, FIFO boarding, EOS/length/cancellation and synchronized slot release. Failed commits quarantine slots and settle queued requests.

Validation: 12 CPU component tests independently passed (0.533s), real Past and scheduler with mocked compute/transport. Covers B1-B4 rebatching, acceptance 1-6, cleanup, cancellation and failure boundaries. No NPU/TP8/end-to-end correctness or latency acceptance.

Rollback: revert this worker-only commit to remove scheduling/service attachment while retaining original Engine compute APIs and server duplicate-ID protection. Restart workers before rollback; never reuse quarantined live slots. Bootstrap and deployment libraries remain a separate pending task.
```

---

```text
server: default generation to supported greedy decoding

Omitted temperature now selects 0 instead of unsupported 1. Explicit nonzero temperature remains unchanged and is rejected by NativeDecode admission; no sampling fallback. Eight CPU service tests passed, including omission, explicit rejection, duplicate concurrency and disconnect cleanup.

Rollback boundary: this one-line default only; revert restores required explicit temperature=0. Keeps duplicate-ID protection and row scheduler. No NPU or full-engine acceptance.
```

---

```text
decode: assemble single-engine TP8 startup and shutdown

Append bootstrap/main to existing worker. Shared weights/Past/prefill transport; B1-B4 capture before scheduler readiness; CPU Gloo control. Decode tensor budget is lowest-rank measured free HBM after prefill minus a policy 4 GiB reserve, not a measured graph peak guarantee.

Component acceptance: 12 CPU mock startup/failure-cleanup tests plus 12 lifecycle tests rerun on merged worker passed. No NPU model startup, full decode, numerical integration or latency acceptance. Default ops/decode libraries must be built before launch; tested temporary libraries are not silently substituted.

Rollback boundary: revert this append-only startup commit to restore 42e5d13 while retaining 6ddc3c4 row scheduler and HTTP fixes. Stop workers before reverting; no second engine or eager fallback.
```

---

```text
decode: build four native libraries into runtime default paths

Build-only delivery: no-argument build.sh compiled norm/attention/gemm/window successfully on A2C. All four default-path shared libraries load and literal dec_* symbols resolve. CPU shell syntax and git diff checks pass. Ignore generated decode .so files; no new tracked files. This is not device numerical, TP8 startup, full decode or latency acceptance. Parent cceacb7 is a component-only bootstrap milestone. Rebuild all four libraries after source rollback and restart workers; binaries are not versioned.
```

---

```text
Reuse serial routed expert expansion and dispatch buffers

bank2 aliases the consumed bank13; down projection overwrites dead dispatch rows. NPU synthetic TP8 dimensions D5120/I288/K6/E8/T16: encoder and CED each 3 eager and 3 captured replays exact against 4aeb81f. Saves 1.055 GiB bank plus 480 MiB rows at T8192. Full TP8 startup/generation still unverified; prior boot OOM before decode.
```

---

```text
Set 8K prefill deployment and omit source-only query scratch

Keep 2097152-token physical KV pool, 1M logical sequence ceiling and B1-4. source20 ratio1 publication never uses attention banks, query output or selection workspace. Syntax/diff checks pass; TP8 startup advances past this source binding but still OOMs later in prefill. Full decode capture, generation and runtime allocation invariance remain unverified.
```

---

```text
correctness: establish TP8 semantic-correctness baseline and reference outputs

Correctness version / 正确性版本: preserve real chat and thinking-tool outputs
as a regression reference in correctness_reference.json. This is observed
semantic correctness, NOT independent tensor/logit equivalence or speed acceptance.

Configuration: TP8, 512K context capacity, 8K prefill chunks, 2M KV pool,
B1..4 prebuilt target/seed/draft graphs; startup preallocation retained.
Share serial prefill/decode expert and scoring scratch and immutable prepared
weights; compact scoring and prebound active extents reduce capacity cost.
Fix scheduler host token buffer wiring. Set policy reserve to 3GiB after measured
bootstrap: decode tensor ledger 459635088 bytes; minimum post-capture free ~3.892GiB.

Evidence: v31 four official-chat-format requests (fact, arithmetic, translation,
explanation) correct, all EOS, 8/8 ranks complete, exit 0, source hashes unchanged.
v32 emits nonempty thinking then valid multiply(a=137,b=24); actual local tool
returns 3288; continuation answers '137 * 24 = 3288' and EOS. Original v32 exit 1:
harness wrongly asserted nonempty thinking again after tool return. Saved-output
semantic review passes; do NOT interpret this as an all-rank clean-exit pass.

Known issues: MTP acceptance ineffective; short-input prefill ~5.6s and decode
~2.5 tokens/s in v30 are not performance targets. Independent logits truth,
B4 live generation and restored ~12K prefill <2s remain unverified.
Next tracks: prefill <2s at full 512K capacity without paying for unused context;
diagnose/fix MTP while preserving this semantic reference.

Previous source commit: 2394ab3; historical prefill performance baseline: 8cdca784.
Native attention kernel changed: rebuild matching attention library and restart
when using/rolling back this source; do not mix compiled library generations.
Evidence: /tmp/prefill_space_ab/boot_v28, boot_v31, boot_v32.
Checks at commit: source hashes match captured runs; modified Python AST parses;
git diff --check clean. No claim of newly rerun full test suite.
```

---

```text
perf(attention): avoid repeated sparse IDs for padded queries

Use legal distinct IDs only for discarded queries and overwrite their
corrected output with zero, including NaN intermediates.

Validation: six NPU exact-output cases (0/1/64/127/4096/8192 valid rows),
valid IDs and missing counts unchanged; NaN poison test passed.
Production-path TP8 boot_v41: all eight ranks complete, six generated
sequences identical to v40. With pending CED optimization, short prefill
1.164s and 12K 2.400/2.442s; not a claim of 12K <2s or repaired MTP.
Evidence: /tmp/prefill_space_ab/{joint_pad_v3,boot_v41}.
```

---

```text
fix(mtp): restore accepted drafts with signed INT4 and independent vocabulary

Previous correctness reference: af71258. Fix mtpq signed INT4 lookup and
unrotated mtp.0.embed / mtp.2.head binding, feeding BF16 head_norm directly.
No C++/mask changes, no new files or runtime buffers.

Production-source TP8 boot_mtp_fixed_v1: all 8 ranks and 11 cases complete.
512K configured context / 8K chunk / 2M pool, B1-B4 graphs built; generation B1.
Six existing cases match baseline token IDs exactly; translation repeat exact.
Six multi-token cases end with EOS; math, knowledge, reasoning and code correct.
Translation 19 rounds/0 accepted -> 7 rounds/12 accepted, reported decode TPS
1.938 (INT4-only control) -> 4.757 and 5.214 (two requests in one fixed boot).
Reasoning: 9 accepted/3 rounds/8.623 TPS; code: 14/5/7.172 TPS.
CPU all 256 packed bytes pass, E2M1 negative control rejected.
All ranks allocated/reserved delta zero, allocator peak +512 bytes after
bootstrap; device free at end 3.72-3.74 GiB. Not proof of zero tiny allocations.

Limits: full B1 rounds still 0.50-0.60s, not <40ms; 12K prefill 2.38-2.41s,
not <2s. Actual 512K prompts, concurrent B2-B4 generation and exhaustive
independent end-to-end numerical equivalence not tested in this change.
Measurements include existing uncommitted prefill.py/prefill_build.py changes,
which are deliberately excluded; this commit alone is not the full measured tree.
Python-only ABI-compatible change; restart to rebuild graph plans, no library
rebuild. Rollback also requires restart. Existing norm library unchanged.
Evidence: /tmp/prefill_space_ab/boot_mtp_fixed_v1 and test_mtp_int4_lookup_cpu.py.
```

---

```text
correctness: freeze verified TP8 generation and compact CED baseline

Include the exact prefill working tree used by MTP correctness and code-generation runs. Capture compact 16K and full-capacity CED paths at startup, reusing arenas and choosing from host-known extent.

Reference: boot_mtp_fixed_v1, all 8 ranks and 11 cases passed; six control token sequences unchanged. boot_mtp_code_v1 generated executable interval-merge code: 338 tokens, 63 decode steps, 274 accepted drafts, six asserts plus two extra checks passed.

512K configured capacity, 8K chunk, 2M pool. This is a correctness/performance reference, not a claim of exhaustive numerical equivalence or 512K input validation. Prefill 12K input still above 2 seconds; optimize from this baseline.
```

---

```text
perf(decode): skip scalar scoring of unused context and bulk-fill tails

Incremental optimization on correctness baseline b37aea5443e38dbab3643e6d6e941626f82e9616.
Keep 8K prefill chunks, 512K context capacity and 2M KV pool. Only
ops/decode/attention.cpp changes; no new production files or fallback paths.
Score active 32-element tiles only, then fill the required -inf tail in
2048-FP32 vector/DMA blocks. Arithmetic and full-width TP reduction unchanged.
Fixed per-core UB output buffer grows from 128 to 8192 bytes; no runtime
large tensor/workspace allocation is introduced.

Validation: six B1/B4 ratio1/2 width8..524288 microcases pass bit-exact,
output guards, and captured graph replay with changing lengths/slots/active.
Short-history leaf timing: width524288 5.350->0.234ms;
width262144 3.146->0.225ms. These are NOT whole-engine speedups.
TP8 same-process shared-weight dual-graph code generation ABBA:
base777.135 vs candidate748.047ms/full step (-3.74%); reverse BAAB:
base818.032 vs candidate775.793ms/full step (-5.16%). All 8 generations
match the baseline 338 output IDs, 63 steps and 274 accepted draft tokens.
Production-path rebuild (no dual-graph/library switch) passes all 8 ranks
and two more exact generations: 740.965/726.240ms per full step,
7.241/7.387 output tokens/s; prompt prefill1.537/1.257s (NOT an 8K test).
Independent-process baseline/control timings vary substantially; do not
compare absolute latency across processes as causal optimization evidence.

Limits: short merge_intervals prompt, greedy B1Q6 only end-to-end;
not an HF-wide correctness certificate, long-context/B2-B4 end-to-end
performance untested here. Full-width tail writes/HCCL still charge unused
capacity. The <40ms complete-step target remains far from met.
Logs/scripts: /tmp/prefill_space_ab/decode_tail_{sameprocess,reverse,production}_v1
and score_micro_v1.py. Diagnostic harnesses remain outside the repository.
ABI/config unchanged. Rebuild ops/decode/attention.cpp using build.sh and
restart on deploy/rollback; reverting source alone does not revert the .so.
Previous library backup: /tmp/prefill_space_ab/libdecode_attention_before_tail.so.
```

---

```text
perf(decode): integrate fixed-shape Cube wo_b and queued W8/W4 expansion

Keep production whole-graph scoring and MTP. Allocate tilings at startup;
add reproducible Cube and host tiling builds to existing build.sh.

TP8 B1 code generation twice: 338 token IDs exactly match reference,
63 steps and 274 accepted draft tokens. Full step 170.7056/160.6368 ms
versus previous W48-only 518.55/546.56 ms (cross-process comparison).
60ms target NOT met. B1-4 capture passed; long-context and B2-4 generation
not covered. Preserve 512K max_seq, 2M pool, 8K chunk. No score segmentation.

Rebuild reproduces both libraries and four tilings byte for byte.
Production import and asset hashes verified. Shared libraries untracked.
Evidence: /tmp/prefill_space_ab/decode_cube_only_w48_v1/engine_run_v2
and installation_t173.json. No redundant post-install device run.
```

---

```text
perf(decode): reuse fixed-shape Cube wo_b in MTP draft layers

Reuse existing startup-owned tiling and WoB plans for the three draft layers. No new kernels, runtime allocations or capacity changes.

TP8 B1 code generation: two exact 338-token matches to 1864827, 63 steps and 274 accepted tokens each; all 8 ranks and library/module fingerprints verified. Whole-step wall time 113.46/114.60ms versus prior 170.71/160.64ms. B1-4 graph capture passed. 512K capacity, 2M pool, 8K chunk and whole-graph MTP retained. Not a B2-4 generation or long-context validation; 60ms target remains unmet.
```

---

```text
perf(decode): tile target attention QK softmax and PV

Batch target SWA/CSA into 16-key AIV-local QK, joint online softmax and tree PV. Coalesce adjacent KV DMA; retain masks, pending precedence, sink denominator and ABI. No GM workspace or capacity/runtime allocation changes.

Validation: 12 synthetic device cases stable against independent FP64 reference; error comparable to prior kernel. TP8 whole-graph B1 code generation twice, B1-4 capture, all ranks and loaded artifacts verified. 512K max_seq,2M pool,8K prefill chunk unchanged.

Observed 109.0437/110.4488 ms per step versus prior 113.4579/114.6048. Greedy output changed at token185 in generated assert data:332 tokens,62 steps,269 accepted rather than338/63/274. Function AST identical;6 generated asserts plus5 external cases passed and inputs unchanged. Not bitwise equivalent or a strictly matched-workload speedup; no long-context/B2-4 generation claim.60ms target not reached.
```

---

```text
perf(decode): avoid replicated dispatch metadata construction

Build cumulative ends, stable inverse and probabilities once on worker 0.
Other workers compute stable row rank from local IDs without cross-core
metadata dependencies. Preserve target/draft ABI, ordering and buffers.

Validation: target/draft B1/B4 CPU exact, guards and changed-route graph
replay passed. TP8 whole-graph code generation twice: identical 332 output
tokens, 62 steps, 269 accepted draft tokens versus 8ed4e8d; all ranks passed.
Observed full-step latency 106.424/105.336 ms versus prior 109.044/110.449 ms;
leaf dispatch savings 2-4 us/call. Two runs are not a broad speed guarantee.
Preserve 512K max_seq, 2M KV pool, 8K chunk and startup-owned buffers.
The <=60 ms target remains unmet. No FP4/full-AIV/overlap candidates included.
Tested norm library SHA256: 5d36f9f3a97846d3c6f54f93603d5f83d84bce0c7bb22078517124ce5cfd8369
```

---

```text
perf(decode): use fixed-shape Cube for draft vocabulary projection

Replace the draft-only BF16 [6*B,5120] x [16160,5120]^T FP32
projection with 16 Cube column partitions, including the 800-column tail.
Reuse WoB plan validation/lifecycle. Charge four 224-byte immutable tiling
buffers to startup allocation; no new hot-path allocation or runtime switch.
Preserve 512K max_seq, 2M KV pool, 8K prefill chunk and whole-graph MTP.

Validation on atlas-a2c / CANN 9.0.1 / TP8:
- B1..4 CPU FP32 oracle, tail guards, changing-input graph replay and repeats.
- Native B1 code generation twice: 332 identical baseline tokens, 62 steps,
  269 accepted draft tokens; B1..4 startup graph capture passed on all ranks.
- Native complete wall step: 89.0496 / 88.8951 ms; decode 60.13 / 60.24 tok/s.
- Prior production observations: 106.4237 / 105.3358 ms, but a subsequent
  plain run was 170.10 ms. Not a controlled speedup ratio; <=60ms not met.
- Eight-rank actual library fingerprints verified; source unchanged in run.

Reproduction artifacts (outside engine tree):
/tmp/prefill_space_ab/decode_draft_vocab_cube_v1/numeric_worker.py
/tmp/prefill_space_ab/decode_draft_vocab_native_v1/engine_worker.py
and engine_coordinator.py, engine_run/{summary,generation}.json, manifests.
Coordinator guards its initial worktree and existing output directory;
use a fresh output root/status snapshot for reruns.
Build: bash ops/decode/build.sh ops/decode/draft_vocab_cube.cpp ops/decode/libdecode_draft_vocab_cube.so
       bash ops/decode/build.sh --draft-vocab-tiling ops/decode

Limits: real generation measured at B1 on one code prompt; B2..4 numerical
and capture coverage only. No new full-capacity long-context throughput
claim, no strict same-session A/B speedup claim, no <=60ms claim.
```

---

```text
refactor(decode): schedule active W4 expert tiles continuously

Compact active expert IDs in the existing counter UB and distribute tiles
across expert boundaries. Preserve 40-worker guard, double buffering,
arithmetic, output layout and inactive-output-untouched contract.
No new global workspace; one source file, no framework changes.

Validation: 10 eager numerical/sentinel cases and changing-route graph
active-output/inactive-preservation checks passed. Isolated and native TP8
runs passed all eight ranks with actual library fingerprints verified.
Both native code-generation repeats match all 332 baseline output tokens,
62 decode steps and 269 accepted draft tokens. B1 generation, B1-4 capture;
512K max_seq, 2M KV pool, 8K chunk. Not broad workload or long-context proof.

Performance: isolated 87.9726/87.8102 ms per complete step; native
89.4665/88.7848 ms vs prior 03aaf56 baseline 89.0496/88.8951 ms.
No demonstrated end-to-end speedup beyond run-to-run noise; do not count
this as closing the 60 ms target gap. Keep this bounded scheduling change
as a correctness-verified implementation change, not a performance claim.

Evidence: /tmp/prefill_space_ab/decode_w4_flat_tiles_v1 and
/tmp/prefill_space_ab/decode_w4_flat_native_v1. Native GEMM SHA256:
a79f403d47351fc5a9bdf1b3e16ecb4d76dd237d401b4dba3ea6cbaea70552bb
```

---

```text
perf(decode): overlap shared and routed expert computation after routing

Keep routed/shared TP reductions ordered on main stream; partition startup workspace and fork/join a batch-owned side stream. TP8 B1-4 capture and two real code generations pass, token-exact to 8fd13a8 (269 accepted / 62 steps). Unprofiled step times 84.25/83.01 ms vs prior 88.52 ms; decode tensor budget +32 bytes. No hot-path large allocation. B1 generation tested; B2-4 capture only.
```

---

```text
perf(decode): cache selected W8 projections within startup memory budget

Cache 710 MiB/rank once at startup, shared by B1-B4; remove 96/136 non-MoE W8 expansions per target step. Keep native expansion arithmetic and unchanged libraries.

Validated 8 ranks, 704 real-weight and 72 batch projection checks bitwise, zero hot-replay allocator growth. Full TP8 engine B1 code generation twice: exact reference tokens, 269 accepted/62 steps, 81.369/81.105 ms per step (dual-stream 84.254/83.011; original 88.520). B1-B4 graph capture; 512K max_seq, 2M pool, 8K chunk unchanged. Not a full-length 512K correctness or B2-B4 end-to-end throughput claim. Evidence: /tmp/prefill_space_ab/decode_w8_integrate_v1 and decode_nonmoe_w8_opt_v1.
```

---

```text
perf(decode): compute rank-local Engram hashes and reuse host views

CPU synthetic metadata: 960 exact cases, 47.84 to 17.29 us/call. TP8 B1 two generations token-equal to reference: 269 accepted / 62 steps, 81.118/81.233 ms versus immediate W8 baseline 81.369/81.105 ms; no significant whole-engine speedup established. B1-B4 captured, not generation-tested beyond B1. 512K capacity, 2M KV pool, 8K chunk unchanged. Reproduction and evidence: /tmp/prefill_space_ab/decode_host_opt_v1/{engine_coordinator.py,verified_result.json}.
```

---

```text
perf(decode): fuse HC scale and gates with grouped barriers

Keep legacy leaf ABI and FP32 reduction order; no extra persistent memory.
23 leaf numeric cases bitwise pass; TP8 two B1 generations match baseline tokens, 62 steps and 269 accepted draft tokens. B1-4 graph capture passes.
Historical parent: 81.118/81.233 ms per step; HC: 81.300/80.494 ms. Mean delta ~0.34%, within run variation; no claim of robust full-engine gain. Leaf graph cost reduced. 512K context, 2M KV pool, 8K prefill chunk unchanged.
Evidence: /tmp/prefill_space_ab/decode_hc_integrate_v1 and /data/tmp/decode_hc_opt_v1.
```

---

```text
perf(decode): reuse completed owner stream for Engram drain

Preserve owner synchronization and fallback synchronization for other streams and standalone cleanup. Validated 36 CPU state/failure cases and three tiny single-NPU transfer cases. No full-model benchmark; end-to-end speedup unmeasured.
```

---

```text
perf(decode): use fixed Cube plan for draft Markov projection

Isolated B1 graph replay 35.85us -> 10.01us; B1-B4 CPU parity and guards passed with random and real weights. Full-engine benefit and acceptance remain unmeasured.
```

---

```text
perf(decode): capture compact 8K score communication graph

Reuse score buffers and retain full-capacity fallback. TP8 score/reduce/topK CPU reference and alternating graph guards pass. Full engine startup memory and end-to-end speed are not yet measured.
```

---

```text
Use canonical NZ shm weights and separate CANN prefill/decode MoE

Prepare INT4 NZ and HP correction bias offline; reject old routed cache ABI. Replace runtime BF16 expansion with explicit DynamicQuant and GroupedMatmulV5 plans, caller-owned scratch and independent prefill/decode modules. Move draft MoE construction into decode operator module. Preserve attention arena borrower capacity.

Validation: 13 production modules import; single-card E384 prefill T16 and decode batch1 routed chains finite, repeat-identical and graph-identical; E128 projection matches official wrapper; isolated cache mapping and 12 mocked draft construction cases pass. Evidence under /tmp/prefill_space_ab/{prefill_nz_integration_v2,decode_routed_integration_v1,modular_acl_probe_v1,modular_cache_probe_v1}. Full model cache, TP8, full engine and integrated latency NOT tested; no speedup claimed. No server restart or cache deletion.
```

---

```text
fix(weights): omit unused routed scale_bias from NZ cache

Bump cache ABI and exact placement accounting. Removes 5,195,268,096 bytes per rank. Validated all eight placement budgets and rank0 full H2D bytewise roundtrip; TP8 generation verification remains pending.
```

---

```text
fix(prefill): reuse caller-owned phase arena for routed scratch

Validated TP8 full-geometry bootstrap/capture and 2x128-token generation; full-step medians 56.83/57.05 ms with compact NZ cache.
```

---

```text
fix(runtime): activate verified compact NZ weight cache

Default-path TP8 bootstrap/capture and 2x128-token real generation passed with unchanged full geometry; full-step medians 57.04/56.76 ms.
```

---

```text
decode: use fixed-address CANN W8A16 projections without per-step expansion

Replace decode W8 expansion+Matmul with WeightQuantBatchMatmulV2 and explicit BF16-to-FP32 Cast where needed. Caller owns persistent BF16 scales, projected output and workspace; remove obsolete expansion scratch. Prefill unchanged.

TP8 B1 Q6 full draft/target/commit/finish host-wall median: baseline 56.78465ms; integrated runs 51.50333/51.33005ms (24 post-warmup samples each). 8/8 ranks completed, two 128-token merge_intervals generation smokes, coherent function body; output capped, not broad quality evaluation. Single-card official Python op comparison matched fixed C API. Reference target 34ms not reached.

Repro: python -B /tmp/prefill_space_ab/w8_official_v1/coordinator_integrated.py (choose fresh output dir); evidence engine_integrated/{summary,generation}.json. Existing unrelated gemm.cpp work excluded.
```

---

```text
decode: scope standard HCCL reduction to startup graph capture

Save and restore process-wide reduction policy around decode capture; prefill unchanged. TP8 two-request integrated smoke completed; output IDs match deterministic baseline. Isolated independent runs reduce full decode step from ~51.5ms to 48.36-48.85ms. Evidence: /tmp/prefill_space_ab/hccl_capture_integrated_v4. Broad quality and long-context validation not claimed.
```

---

```text
decode: pack routed/shared MoE TP reductions into one collective

Explicit caller-owned FP32 [2,6B,5120] views preserve independent sums and finish order; no extra storage/copy or prefill change.

TP8 integrated 2x128 smoke: full step 46.494866/46.530814ms vs earlier 47.779059/47.556377ms (not paired statistics). Eight ranks exit0; both outputs match latest unchanged-source run. Reproduce with /tmp/prefill_space_ab/moe_packed_reduce_integrated_v1/coordinator.py (fresh output path required). B4 performance and broad quality untested; target34ms not reached.
```

---

```text
decode: compile vector-only norm leaves for dav-c220-vec

Explicit AIV compile target, no arithmetic/runtime allocation/prefill changes.
MoE same-resource bench dual 343.474 -> 313.758 us; two seeds exact.
TP8 full-step two 128-token smoke medians 46.495/46.531 -> 41.337/41.370 ms; all ranks exit0; output IDs exact vs packed-reduce baseline. Not paired multi-run statistics or broad quality/B4 validation. Target34ms remains unmet.
Evidence /tmp/decode_norm_vec_v1/results.json and /tmp/prefill_space_ab/decode_norm_vec_integrated_v1/engine_run/summary.json. Reproduce coordinator.py with fresh output path.
Only norm compile-target diff staged; preexisting HC/build/attention/gemm changes preserved.
```

---

```text
decode: compile attention leaves as AIV-only with explicit task geometry

Preserve original logical stride by launching twice the logical block count. No arithmetic, allocations or prefill changes.
Leaf SWA/CSA ratios 1,2 at B1/B4: six bit-exact finite comparisons. TP8 2x128-token smoke matches norm-AIV reference outputs exactly; eight ranks exit0 and source hashes unchanged. Full step 41.337/41.370 -> 40.947/40.849 ms (short test, not paired statistics).
Evidence /tmp/decode_attention_vec_v1/results.json and /tmp/prefill_space_ab/decode_attention_vec_integrated_v1/engine_run/summary.json. Reproduce coordinator.py with fresh output path. Only incremental attention/build changes staged; preexisting dirty edits preserved. Broad quality and B4 engine performance untested.
```

---

```text
decode: parallelize sparse source scoring across row teams

Only task scheduling changed; no new GM storage, ABI or prefill change. Ten leaf checks exact. Installed working-tree TP8 smoke: 40.806/40.380 -> 38.609/38.667ms; eight ranks successful, source unchanged. Outputs in previously observed baseline set, not paired identical. Short-run evidence only.

Evidence: /tmp/decode_source_rowteams_v1 and /tmp/prefill_space_ab/decode_source_rowteams_integrated_v1. Preexisting query cache and HC/model/gemm/build edits remain unstaged; timing applies to tested working tree, not isolated clean HEAD.
```

---

```text
decode: fuse W4A8 input restoration and group-count preparation

Keep official DynamicQuant/GMM and caller-owned buffers. Eight leaf cases and two-seed single-layer comparisons exact. Single-layer dual-stream 309.550 to 304.694us. TP8 two 128-token smoke runs: 38.267/38.033ms full-step medians versus row-team 38.609/38.667ms; eight ranks complete, source hashes unchanged. Outputs belong to observed baseline set, not pairwise identical. Not broad quality validation or proof of stable performance.

Evidence: /tmp/decode_w4prep_v1 and /tmp/prefill_space_ab/decode_w4prep_integrated_v1. Timing applies to tested working tree including preexisting edits; only preparation increment staged.
```

---

```text
perf(decode): restrict draft greedy selection to current proposal row

Keep target greedy API and padded chain layout. Local B1/B2/B4 five-step equality and graph replay writeback verified; TP8 B1 two short generations completed. Existing output nondeterminism remains unresolved; no deterministic or causal whole-engine speedup claim.
```

---

```text
perf(decode): use Cube for draft shared-expert down projection

Keep BF16 inputs and FP32 output with caller-owned buffers and startup tiling. Local deterministic comparison and dynamic replay passed; TP8 short generations completed on all ranks. Local kernel speedup is verified; no full-step gain is established for this change alone. Prefill unchanged.
```

---

```text
perf(decode): fuse HC projection with early weight DMA

Keep FP32 weights and caller-owned storage; overlap paired-column weight DMA with input conversion and RMS reduction. Dynamic input/weight sentinel replay B1-B4 matches the previous fused kernel bitwise. TP8 short generations pass on all ranks with matching tokens. Observed full step 36.95/37.03ms versus 37.36/37.42ms in one prior run; stable causal speedup not established. Prefill unchanged.
```

---

```text
perf(decode): vectorise source_scores paged key scoring

Stage each 64-key tile with one DMA and score all four heads with
broadcast Mul plus two BlockReduceSum levels, removing the per-key
scalar read-back that dominated long-context decode. Tiles touching
the pending rows keep the original path.

128k context: 202.28 -> 74.35 ms/step, short context unchanged at
32.80 ms; both generations byte-identical to the previous build.
```

---

```text
decode/attention: vectorize source_select top-512 with two-pass lower-bound + GatherMask compaction

Pass1: two-level BlockReduceMax -> 64-elem segment maxima -> scalar heap gives t0,
a provable lower bound of the true top-512 threshold.
Pass2: per-64-elem Compare(GE t0) + GatherMask compacts surviving indices to the
front of each segment with a -1 sentinel; scalar loop only touches survivors
(~65711 -> ~1600 GetValue per row).

Note: Compare's dst mask occupies a 32B-aligned slot per repeat (only 8B valid),
so Compare/GatherMask must be issued per 64 elements.

micro-bench (len=131422): 3.203ms -> 0.652ms (4.91x), output bit-identical.
engine ctx128k full step: 74.35ms -> 49.76ms; short_ref unchanged; generated text identical.
```

---

```text
decode sel: replace scalar heaps with A2 proposal hw sort (Sort/MrgSort) - 0.597->0.260ms per call, bit-identical; 128k step 49.76->46.54ms
```

---

```text
decode/attention sel_p2: remove SELCHUNK upper bound on hardware-Sort fast path

Root cause: nsegv = ceil(min(ceil((st+6)/ratio), width)/2048)*32. At ctx=131422
(>131072 = 64*2048 cliff) the ratio=1 indexer layers get nsegv=2080 > SELCHUNK(2048),
falling back to the scalar-heap slow path (~307us vs ~10us for ratio=2 layers).

Fix: keep the fast path for any nsegv>=512 by Sorting in overlapping windows
(win=min(nsegv,2048), last window aligned to nsegv-win) and taking t0 = min of each
window's 512th largest. That min is <= the global 512th largest, so it stays a valid
lower bound and sel_p3 still returns the exact top-512 -> bit-exact output.

Measured (128k, TP8 NZ, rank0 kernel_details, 3 profiled steps):
  sel_p2      4759.6us -> 358.7us (-92.5%)
  median step 45.99ms  -> 44.06ms (-1.93ms)
  generation.json md5 identical for both cases (bit-exact)
```

---

```text
decode: cube-based indexer logits (v13) + fix scores bulk V_MTE2 race

- new source_cube kernel: batch indexer logits via Cube MatMul (n1024/n2048 tiling)
- dec_source_scores consumes precomputed logits; bulk path
- BUGFIX: bulk loop reused UB buffer sA across heads with only PipeBarrier<PIPE_V>;
  missing fence<HardEvent::V_MTE2>() let next DataCopy overwrite sA before vector reads
  finished -> wrong scores -> candidate blowup -> sel_p3 0.745->5.073ms
- results (128k / short median full step, TP8 NZ):
  v11 44.05 / ~34.0 ; v13 pre-fix 46.66 / 33.71 ; v13 fixed 42.478 / 33.829
  source_scores 5.504->3.561, +source_cube 0.288, sel_p3 5.073->0.737
- numerics: micro bench vs torch fp32 golden diff 0.0; ctx128k generation bit-identical to v11
- repro: /tmp/prefill_space_ab/decode_ctx128k_prof_v13fix (coordinator.py), micro_cube.py
- also includes Engram prefetch + gemm work present in the tree
```

---

```text
decode/attention swa+csa v18: head-group KV sharing (G=2) + 32-row tiles

- HEADGROUP=2: two heads share one loaded KV tile, halves DMA+cast work
- TILEROWS=16->32: halves per-tile scalar sync overhead
- fix: WholeReduceSum repeatTimes is uint8; n*8 reached 256 at K=32 and
  overflowed, corrupting attention output (engine reported decode commit
  receipt err=105). Now chunked to <=16 rows (repeat<=128); identical at K=16.

full-engine A/B (2 runs each): short_clean 33.87 -> 32.73/32.75 ms,
ctx128k_profile 42.88 -> 40.71/40.92 ms, 0 receipt failures
```

---

```text
repo hygiene: drop dead attention_v11.cpp and untrack build-generated tiling blobs
```

---

```text
decode: tile draft attention and remove duplicate prepare
```

---

```text
Compact draft Markov projection and remove unused decode implementations
```

---

```text
refactor(decode): remove retired source stub and unused bank alias

NativeSource tombstone -> live SourcePlan; unused routed_bank2 view -> existing shared score bank. Unroll singleton bank initialization. AST normalized-equivalence checks pass. TP8 eight-rank startup/capture/generation passes, 2x192 tokens match observed baseline outputs; baseline is not bit-reproducible. Full-step medians 32.701/32.856 ms; no speedup claim. Device ABI unchanged. Update integration status and rollback boundaries.
```

---

```text
refactor(decode): allocate source logits scratch once at maximum size

Replace grow-on-bind retained banks with one owner-serialized scratch allocation; release owner reference on close. TP8 eight ranks complete two 192-token generations. Owned bytes 2273821360 -> 2173158064 (-96 MiB/rank). Short full-step medians 32.642/32.523ms are smoke measurements, not isolated speedup or bit-equivalence proof. Evidence: /tmp/decode_followup/cube_logits_once/full.
```

---

```text
fix(decode): implement paper HSI candidate-restricted reindexing

L20 builds persistent 16384-position pools after TP sum; L24/28/32/36 rescore pool only with paged and pending KV support. Preserve Full scanning, Reuse, short/full graphs and Past budget.

Validate independent CPU formulas: 12 pool/select and 6 score graph cases; TP8 B1-4 capture and 80/8000-token generation smoke passed. Reproducers in extra-info/hsi. No old-decode oracle, speedup, long-context full-model replay or end-to-end numerical equivalence claimed.
```

---

```text
perf(decode): bound score graph prefixes and vectorize HSI memory access
```

---

```text
perf(decode): vector-filter HSI selection candidates with stable ties

Preserve exact heap ordering; filter by a conservative GE threshold using Compare/GatherMask. Isolated 12-case semantic/capture fixture passed. At the 131K fixture, selector timing decreased from 1.406 ms to 0.888 ms; no end-to-end speedup claimed.
```

---

```text
perf(decode): exclude masked scores before HSI heap fills
```

---

```text
decode: vectorize hsi_scores candidate scoring

Replace the scalar inner loop with a pure-AIV vector chain. The old code
did 32 GetValue scalar read-backs and ~10 sync() full-pipe barriers per 8
candidates; arithmetic was never the limit (50M MAC total).

ReduceSum results now land in 32B-strided slots and are packed by
BlockReduceSum; ReLU, head weighting, 4-head folding and scaling are all
vector ops. Invalid candidates use an arithmetic mask instead of
Compare/Select. Zero scalar read-backs remain in the loop.

128K median_full_step 51.536979 -> 47.472169 ms (-4.064810, -7.89%),
8 ranks complete, exit 0, source_unchanged, output readable.
```

---

```text
perf(hsi_pool): GE+GatherMask prefilter on block-max scan (-0.22ms/step)

Vectorized the 16384-block scan: load CH=2048 scores per DataCopy, compare
against the current heap root with GE and compress survivors with GatherMask,
so only real contenders reach the scalar hsi_offer path.

hsi_pool 2.950 -> 2.726 ms/step (rank0 kernel_details, 3 captured steps).
hsi_scores (3.45) and hsi_select (2.24) unchanged -- verified as controls.

NOTE for future work: the final pop loop looks like dead weight but MUST stay.
Dropping it makes hsi_pool 1.911ms (-35%) yet costs hsi_select +2.75ms
(2.24 -> 4.99), a net loss. hsi_select tie-breaks on id VALUE so pool order is
numerically irrelevant, but its threshold/GatherMask early-out needs
score-DESC input to raise the heap watermark fast. Order is numerically free,
performance-critical.
```

---

```text
perf(hsi_pool): hardware proposal sort replaces scalar heap

Batched top-k merge: per 2048 blocks do BlockReduceMax -> CreateVecIndex
-> 4x512 Sort -> MrgSort(4-way) -> MrgSort(2-way) against the running
accumulator, then Extract the top slots. Removes the 2048-iteration
scalar sift-down heap while keeping the exact descending order the
downstream hsi_select early-exit depends on.

single-op bench (width=131072, deepest step, 50 calls): 2.814 -> 0.093 ms
full-model 128K smoke: 47.472 -> 43.841 ms/step
correctness: extra-info/hsi/check_hsi.py 12/12 (width 64/32768/131072),
pool+select+replay all match the reference.
```

---

```text
docs: clarify captured decode reduction nondeterminism
```

---

```text
perf(hsi): prefilter select with parallel group-max threshold
```

---

```text
perf(hsi): halve hsi_scores by folding 32 small reductions into 2 vector ops

Replace the per-head "Duplicate(red) + 8x ReduceSum + BlockReduceSum"
reduction with a single Add fold (128 -> 64 lanes) followed by one
WholeReduceSum that emits the 8 packed sums for that head, mirroring the
existing consume() pattern in this file.

hsi_scores is V-bound, not MTE2-bound: an earlier double-buffered
prefetch attempt made it slower (854.54 -> 900.14 / 896.40 us), while
cutting vector instruction count halves it.

kernel: hsi_scores 854.54 -> 391.14 us (-54.2%, n=12)
full step (context_128k, 24 tok, 8 samples, 8/8 ranks):
  41.0427 -> 39.3650 ms (-1.678 ms)
cumulative since baseline: 51.5370 -> 39.3650 ms (-23.6%)

Also fixes a correctness defect: the old path left stale values in the
padded tail (idx 16368/16369), where two distinct ids yielded bit-identical
scores; those large positive leftovers could be picked by top-k. The new
path masks them correctly.

Note: not bit-exact, floating point summation order changes.
```

---

```text
perf(decode): collapse 7 length-bucketed verify graphs into one

Under candidate_scoring (HSI) capture_score_width() returns immediately,
so all 7 length buckets captured byte-identical programs: the reduce
count stays batch*6*16384 and the score/select scalar args are unchanged.
The buckets were pure redundancy -- 6 extra NPUGraph captures plus their
device memory -- with no mechanism for a per-length speed difference.

Measured (8-rank TP8, 24 tok, median full step ms, 2 runs each):
  128K 39.01/38.66 -> 39.39/39.33
  32K  39.61/39.14 -> 40.75/40.45
  2K   39.05/38.14 -> 39.93/40.09
Spread is within the ~1ms run-to-run noise seen on identical code; since
the captured programs are provably identical this is not a regression.
Untested: non-candidate_scoring path (capture_score_width still honors
width there, but that path is not used by this model config).

Repro: python3 /tmp/decode_followup/context128k/smoke_coordinator.py OUT
```

---

```text
refactor(decode): drop verify_graphs bucket map and dead capture_score_width

After the single-graph collapse the bucket map held exactly one entry, so
verify() still ran a next(...) scan to pick the only graph and close() had
an always-false `graph is not self.graph` guard. verify() now replays
p.graph directly and raises explicitly when the window exceeds capacity.
capture_score_width() had zero call sites left and is removed; it was a
no-op under candidate_scoring anyway.

Pure structural change, no kernel or arg touched. 8-rank TP8 24 tok smoke
(median full step ms): 128K 39.20, 32K 40.14, 2K 39.72 -- within noise of
the single-graph commit (39.39/40.75/39.93).

Repro: python3 /tmp/decode_followup/context128k/smoke_coordinator.py OUT
```

---

```text
decode swiglu: replace 5 PIPE_ALL with precise pipe events, dual-buffer gate/up

Arithmetic sequence unchanged (bit-identical). Gate/up now use separate
UB buffers so both DataCopy issue back-to-back; PIPE_ALL barriers replaced
by fence<MTE2_V>/<V_MTE3>/<MTE3_MTE2> (WAR guard), reusing the existing
fence template style from attention.cpp.

Measured: 128k 38.83 / 32k 40.17 / 2k 39.23 ms (complete=true, 8 ranks).
Within run-to-run noise vs 39.20/40.14/39.72 baseline -- kept for
correctness/clarity, NOT claimed as a speedup.
```

---

```text
perf(decode): overlap HC coefficient prediction with sublayer compute
```

---

```text
perf(decode): reuse source query latent in attention

Alias the normalized source query latent and skip duplicate wq_a and RMSNorm in the eight source layers. Validated 512 projection comparisons across 8 ranks, B1-4 and two input scales; short and long semantic smoke checks passed. Full-step medians: baseline 38.167 ms, reuse 37.770 ms, reverse baseline 37.888 ms; significant end-to-end speedup not established. Excludes experimental HSI score kernels.
```

---

```text
feat(decode): bind vendor fused lightning indexer (aclnnLightningIndexer)

One aclnn dispatch replaces our score -> all-rank reduce -> top-k chain: the
op scores all 32 index heads against paged keys and emits the selected key
positions directly. torch_npu has no binding for it, so the two-stage aclnn
contract is driven through ctypes.

Measured on A2 (910B), B=1 S=6 N=32 D=128 sparse_count=2048, block_size=128:
  kv=4K   per-dispatch 153.9us  amortised  69.1us
  kv=32K  per-dispatch 169.1us  amortised  83.9us
  kv=128K per-dispatch 264.3us  amortised 176.8us
Our current chain costs about 900us per index layer at 128K, so the ceiling
is roughly 5ms of the 38.9ms step across the eight index layers.

Correctness: at kv=1024 sparse_count=64 the selected set matches a CPU
reference of our own rule sum_h relu(q_h . k) * w_h exactly, 64/64 on all six
query rows, with causal tails aligned via sparse_mode=3 plus actual lengths.

Not yet proven, deliberately kept out of the production path for now:
end-to-end gain, capturability of an aclnn executor inside NPUGraph, and the
cost of giving every rank all 32 heads (the op fuses scoring with top-k, so
per-rank head sharding cannot be preserved).

Op contract notes, none of which are documented, all recovered from the
tiling checks: block_table is mandatory int32 (keys are always paged),
layout must be BSND/PA_BSND, pre_tokens and next_tokens must be INT64_MAX,
return_values must be false under PA_BSND, sparse_indices must be 4-D, and
the executor is single-shot unless marked repeatable.
```

---

```text
test(decode): gate the vendor indexer on page geometry and graph capture
```

---

```text
test(decode): prove the captured indexer honours a changed sequence length
```

---

```text
perf(decode): replace the index select chain with the vendor fused indexer

Decode's index source scored every cached row, reduced the scores across all
eight ranks and ran a host-side top-k. The CANN build on this box ships
aclnnLightningIndexer, which does the same selection inside one kernel, so the
chain is now one dispatch over the paged bank.

Making it graph-capturable needed the paged geometry to stay in range for the
warmup batch: inactive slots carry -1, unallocated block-table columns hold
junk and a sequence ending on a page boundary points one block past its last
page. Those rows are never read back, but the gathers and the vendor kernel
still tile over them, so each index is clamped into the bank.

Eight-rank smoke (131072 input tokens, 24 output tokens, full step wall time,
repeated-token fixture - performance only, not model quality):
  128k 38.906ms -> 34.323ms, 32k -> 32.756ms, 2k -> 33.132ms
```

---

```text
test(decode): record the probes behind the fused indexer's row order

vendor_select_check.py replays the paged select against the old scoring chain
on one rank; the two probes pin the row order and the per-row layout the
vendor kernel expects, which is what the clamped geometry relies on.
```

---

```text
feat(serve): two-stage serving stack script + fix tokenizer path and greedy temperature default

scripts/serve.sh start|stop|status brings up the TP8 engine (must run as -m
strategy.decode_worker, else strategy/ shadows the package) and the protocol
front-end.  service.py pointed at a tokenizer path that no longer exists and
defaulted temperature to 1.0, which the greedy-only engine rejects with 400,
so every stock Anthropic/OpenAI client failed; requests now run greedy and the
original value is kept as temperature_requested.
```

---

```text
feat(decode): on-device temperature sampling end to end

Sampling now happens inside the captured decode graph: window.cpp gained a
Gumbel-max path with a per-row fp32 temperature vector and an in-graph seed
counter, so temperature costs one extra kernel argument instead of a host
round trip. Temperature rides OPEN in milli-units, is replayed from Row on
every STEP, and the front end no longer clamps requests to zero.

Verified on 8 cards: T=0 reproduces greedy exactly (3/3 identical), T=1
varies with sane text, and a mixed-temperature batch of four shows no
cross-row leakage.
```

---

```text
feat(server): image input end to end on NPU

Accept Anthropic-style image blocks, run the native DeepSeek V4.1 vision tower
tensor-parallel across the eight ranks, and replace the placeholder rows of the
prefill embedding with the aligned features.

The aligner and the start/newline/end delimiters come from the quantised pack's
own vision shard: its aligner.w2 and delimiters differ from the bf16 release,
and mixing the two makes the model describe an empty image.

Verified: the official carrots/corn examples answer 'Carrots'/'Corn', a
synthetic shape image is described correctly including its embedded text, and
text-only requests are unchanged.
```

---

```text
feat(serve): rank-local cold KV prefix cache in the worker prefill path

Engine now binds one ColdCache per rank (identical SPMD admission order
keeps lookups and evictions in lockstep, so no TP broadcast is needed).
open_row restores the longest cached prefix, replays the sliding window
that mark_cold cleared, and publishes each committed chunk. Cold spans
stay on source-ratio multiples because export/import address rows as
t//ratio; image rows are excluded since placeholder ids collide.

Verified on 8x A2: 9815-token prompt 3.83s cold, 2.56s with 9686 tokens
read from cache, answers unchanged for same and sibling questions.
```

---

```text
feat(serve): publish cold-cache timings and exact stored-token counts

The worker now times the three cold-cache stages separately (lookup,
restore, store) and reports how many tokens each request committed, so
model_prefill_seconds again measures only chunk compute.  service copies
the new cache_stored_tokens field and server prefers it over the
blocks * block_size estimate, which silently degraded to "blocks" when
no block size was published -- Anthropic usage reported 1-2 creation
tokens for a 9.8k prompt.

Verified on the live stack: cold miss reports creation=9814/read=0,
a repeat reports read=9686 with creation=128 for the fresh tail, and the
per-request log line carries cache_hit_rate 0.9869, lookup 0.007s,
load 0.230s, store 0.599s.
```

---

```text
fix(api): reject temperature other than 0/1 and surface over-length as 400

- service.intake: temperature must be exactly 0 (greedy) or 1 (sampling)
- server: MAX_TOTAL_TOKENS 1M -> 524282, matching engine row_cap (max_seq 524288 minus Q=6)
- remote_strategy: engine 4xx re-raised as ValueError instead of requests.HTTPError;
  service._submit maps it to ServiceError so clients get 400 invalid_request_error, not 500

Verified on the live stack: temp 0/1/default -> 200; 0.5/0.7/2/-1/'abc' -> 400;
600k-word prompt -> 400 with explicit token accounting; 200k-word prompt -> 200.
```

---

```text
perf(prefill): drop residual 1-token span from cold-align chunking

_prefill_spans aligned chunk ends to cold_align=2, so an odd-length prompt
produced a trailing 1-token span. Each span runs a full 61-layer forward
whose cost is dominated by host launch, not by token count, so odd-length
prompts paid for an entire extra forward pass.

Measured on 439 vs 440 token prompts, same engine instance, n=4:
  before: odd prefill 1.366s / TTFT 2.133s | even 0.547s / 1.291s
  after : odd prefill 0.622s / TTFT 1.523s | even 0.593s / 1.603s
The 0.82s odd/even gap is gone. Cross-restart jitter measures +-0.3s, so
only the within-instance paired comparison is treated as meaningful.

Cold-cache store still clips to the alignment boundary. Verified a
440-token prompt re-sent hits 310 cached tokens with identical output.
```

---

```text
perf(metrics): attribute prefill queue drain away from cache store

store_chunk's first read of the slot blocks until the prefill kernels
retire, so that wait landed in cache_store_seconds and was subtracted
from model_prefill_seconds (which is computed as lane wall minus cold).
Cold-store therefore looked like 0.8s when the real copy is ~30ms.

Drain the prefill queue explicitly before timing the store, matching the
reference engine's on_drain accounting, and report the wait separately as
cache_store_drain_seconds. Measurement only; no behaviour change.

Measured (439/440 tokens, medians, n=4 each):
  before: model_prefill 0.593 / cache_store 0.822
  after:  model_prefill 1.157 / cache_store 0.034 / drain 0.643
  TTFT unchanged (1.31/1.34); cache hit path verified (310 tok, same output)
```

---

```text
perf(prefill): cut CHUNK 8192->2048, TTFT 1.30s -> 0.67s

The prefill compute plan is bound to a fixed CHUNK-row geometry, so a
379-token prompt was executing 8192 rows -- a 21x padding waste. Cost
tracked span count, not real tokens (379/1819/4819 tokens all drained
~0.59s; 10219 tokens crossed into a 2nd span and drained 1.23s).

Smaller spans win across every length measured; the per-span cost falls
faster than the span count rises, so even long prompts get faster:

  tokens  drain 8192 -> 2048   ttft 8192 -> 2048
     379   0.588 -> 0.115       1.31 -> 0.90
    1819   0.625 -> 0.264       1.28 -> 0.75
    4819   0.640 -> 0.530       1.35 -> 1.42
   10219   1.233 -> 0.810       2.20 -> 2.06

Same-harness medians (t_m.py, 440 tokens):
  time_to_first_token 1.282/1.314 -> 0.669/0.691  (-48%)
  model_prefill       1.132/1.140 -> 0.513/0.519  (-55%)
  cache_store_drain   0.587/0.588 -> 0.217/0.226  (-62%)

Also frees ~3GB: decode_tensor_limit 2.29GB -> 5.34GB, which lifts the
previous 110MB headroom that made extra plan bindings impossible.

Not validated: output correctness beyond smoke-level metric runs, and
concurrency/batch behaviour under the new span count.
```

---

```text
Revert "perf(prefill): cut CHUNK 8192->2048, TTFT 1.30s -> 0.67s"

This reverts commit db9f8374c02998ad1f55ddd60b26af45abdf79bf.
```

---

```text
perf(prefill): use official aclnnRmsNorm and pool norm workspaces

The hand-rolled eight-op RMS chain is replaced by a single aclnnRmsNorm
call, measured at 1.206ms -> 0.171ms on 8192x5120 (7.06x) with identical
numerics.

The official kernel asks for a 16MB workspace, and the norm binders in
prefill_layer allocated one private arena per plan instead of using the
pool, so the larger demand grew into about 2.9GB of extra device memory
and left the decode budget negative. Both binders now accept the
build-owned Workspace, whose buckets are shared by capacity: free memory
after prefill is 5.38GB against 5.41GB before the change, and the decode
tensor limit returns to 2.26GB.
```

---

```text
prefill: stop computing dead rows when the chunk is not full

The encoder always ran the bound 8192-row chunk through all 61 layers even
when the request only filled part of it, because every plan froze its row
count at bind time and the real token count was dropped before the layer
loop.

Plans now accept a row count: _HCPlan rewrites the row slot it already keeps
as a Python int, _CastStatsPlan rewrites its ctypes field, and Workspace.bind
registers whichever plans expose set_rows so the encoder can broadcast the
live row count in one place. Only plans whose capacity matches the chunk are
retargeted, which leaves the ratio-2 compressor and the fixed CED stage
untouched.

Projections keep the fast path: a full chunk still runs the bound plan, while
a partial chunk projects the live rows eagerly into the same buffers and
returns the full view so downstream stages keep their fixed shapes.

Measured: 309 tok 1.03s, 1509 tok 1.08s, 3738 tok 1.16s (was a flat ~1.21s
regardless of length). 26009 tok over 4 chunks is 4.94s, matching the old
full-chunk cost, so long prompts do not regress.
```

---

```text
prefill: post every hand-written launch to the torch_npu task queue

The engine had to run with TASK_QUEUE_ENABLE=0 because prefill reaches the
device by hand: custom kernels launched by address, and repeatable ACL
executors driven straight to the stream. Both bypass the torch_npu task queue,
so with the queue on they overtook the torch ops feeding them and attention
read storage that was not written yet (ACL 361001, MTE out of range).

Launches now go through ops/queued.py, which posts the frozen argument list to
the same queue the torch ops use; call sites keep their ctypes signatures. The
queue is enabled by default: a cold 3617-token prefill goes 1.366s -> 1.254s on
identical code, and the engine no longer depends on an unusual start variable.
```

---

```text
prefill: early-stop MoE grouped matmul on live rows

Route padding rows of a partially filled chunk into a sentinel expert
group so the W4A8 grouped matmul skips them.

- widen counts/ends by experts+4 and hand W4A8 only ends[:experts]
- add PackedRouted.set_rows() wired through workspace.register
- fill ids[live:] with the sentinel expert id when live < length

Verified on 8k chunk: output stays correct; model_prefill_seconds
improves ~5% (1683 tok 1.029 -> 0.962, 11403 tok 2.139 -> 2.030).
The gain is uniform across prompt lengths, which shows MoE is not the
dominant cost of an underfilled chunk -- attention still runs the full
chunk width and is the next target.
```

---

```text
prefill: re-orchestrate as plain operator composition

Rewrite the prefill path as pure functions that take their row count from
the live tensors, following the reference engine eager style:

- residual.py: collapse/expand/mixes call the released HC kernels
  (pre_hc_collapse/expand/cast_stats/sinkhorn) directly. The plan wrappers
  and their capacity bookkeeping are gone; every launch takes the caller
  row count as its first uint32.
- prefill.py / prefill_block.py / prefill_build.py / prefill_layer.py /
  moe.py: drop the build-time binding layer in favour of direct calls.

Net 255 fewer lines. Kernel math is untouched.
```

---

```text
chore: ignore CANN crash dumps under extra-info/
```

---

```text
prefill: drop the bound-plan path from projections\n\nA projection is y = f(x). The ACL plan froze x's address and the full\nchunk shape (M=8192) at build time, so _chunked kept a parallel eager\nbranch to handle short prompts -- two code paths for one operation.\n\nKeep only the eager one: rows come from the caller's tensor.\n  gemm.py        -- linear() is the whole contract now; bind()/linear.bind\n                    are gone. bind_matmul stays, prefill attention's QK^T\n                    schedule is its only remaining user.\n  prefill_linears -- _chunked loses bind/set_rows/live[]/register.\n\nMeasured on a 1269-token prompt: 1.72 / 1.50 / 1.45 s against a\n1.644 / 1.449 / 1.451 s baseline -- unchanged, within noise. Freeing\nmatmul from the padded M buys nothing, which is itself the finding:\nthe fixed cost does not live in how many rows we multiply.\nOutputs verified identical.
```

---

```text
prefill: drop the workspace argument nobody reads

kernel() already takes its dequant space from space[k][:n]. Once the
bound-plan path went away, the workspace passed down through linears(),
projection(), _chunked() and _float_projection() was never touched by
any of them -- four signatures carrying a value for no reader.

Outputs verified identical, timings unchanged.
```

---

```text
prefill: derive rows from inputs instead of freezing capacity at build time
```

---

```text
prefill: carry live row counts through every stage instead of full-chunk buffers
```

---

```text
prefill: drop the row-plan registry now that live rows flow through call arguments
```

---

```text
engram: share the live rows across ranks so the collective shrinks with the chunk
```

---

```text
residual: read the IEEE sign bit instead of copysign (NPU CPU fallback); prefill chunk 8192 -> 4096
```

---

```text
prefill: chunk 4096 -> 3072 (2k 0.76-0.87s, 6k 0.79-0.88s)
```

---

```text
prefill: chunk 3072 -> 8192; small chunks deadlock the device on long prompts

Clean (cache-free) benchmarks show the 3072 win was a prefix-cache artefact:
2k 1.02-1.20s, 6k 1.36-1.48s, and 12k/24k hang outright (plog: watchdog
timeout, moduleId 7, 300s). At 8192 every length completes: 2k 1.27-1.51,
6k 1.35-1.48, 12k 1.61-1.66, 24k 2.83-3.87. The multi-chunk path is broken;
until it is fixed the chunk must stay large enough to avoid it.
```

---

```text
Refactor prefill/decode orchestration around existing optimized kernels

Remove plan/bind shells; preserve 8K chunk, 512K context and 2M pool.
Keep native computation unchanged; queue infrastructure and host B4 metrics updated.
Validated: 12 CPU contracts, structural audit, 11 requests and 16 real B4 steps.
r8 TP8, 128 output tokens, two measured trials per length (6 cached input tokens):
2K TTFT 3.273-3.312s, model prefill 2.304-2.344s, decode 76.398-80.851 tok/s.
12K TTFT 5.841-6.227s, model prefill 4.856-4.881s, decode 71.088-77.072 tok/s.
Limits: repeated free generation drifts; independent full-model numerical
acceptance and maximum-capacity runs remain unverified. Production route not switched.
```

---

```text
refactor(prefill): register optimized kernels behind Tensor operators

Local operator/composition milestone; NOT full-engine validated. Preserve optimized device kernels, replace Python raw-pointer bridges and pending events with PrivateUse1 registrations, current-stream dispatch and recordStream. No eager address-bound plan or reference fallback. GMM scale INT64[E,1,N].

NPU final suite T=1/7/100/129 passed bitwise comparisons: CANN matmul/dynamic quant including async input lifetime; four GMM cases; eight SWA/sparse attention compositions; joint pack/corrections/paged read. Earlier basic/selection tests passed. Python syntax and diff checks passed; ops/prefill bridge audit empty. Probes outside repo.

Not measured: full-engine/TP8/replay128 integration, latency or peak memory. No speed claim. Running R19 unchanged. Previous source f0a0ef4; not a new known-good full-engine rollback point. Host extension builds outside repo with matching torch/torch_npu+CANN9.0.1 and existing libhc/libattention/libdq. Device ABI unchanged. Activation/rollback requires fresh process with matching sources; no service restart performed.

Evidence: /tmp/torch_boundary_3xv5jo_x/probe_final_fixed.log ends FINAL_TENSOR_COMPOSITIONS_PASS.
```

---

```text
perf(prefill): pre-encode constant GMM scales and simplify expert ordering

Move immutable npu_trans_quant_param out of live-T projection; stable FP32 expert sort (exact IDs) and scatter permutation inverse. Keep Tensor operators and optimized kernels, no address/shape plans.

TP8 real Engine.open_row, cold cache disabled, encoder 8192+4096 rows: max-rank 1.965/1.957/1.931s vs ~3.64s. Scale-only 2.295s; prior encoding 80 calls cost 1.321s. Warm 100/2048 encoder 0.243/0.577s.

All 8 ranks: routed old/new bitwise at T=1/100/129/2048/8192 with local reduction disabled; all encoded weight scales identical. Full engine 7 cases completed. Not an independent whole-model numerical oracle or HTTP latency test. Reproduce: /tmp/encoder_optimized_db57e3qc/probe.py with saved bootstrap env; results rank*.json/*.correct/*.pass.
```

---

```text
perf(ced): reuse index scores and call-local rotary rows

Share call-local rotary rows across layers with identical compressed/uncompressed configuration. Reuse full-index scores to construct candidates and select IDs in one pass; preserve candidate masking, stable sorting, tensor interfaces and optimized kernels. No plan or CED graph added. 19 insertions / 14 deletions.

TP8 actual Engine.open_row, prefix cache disabled, 12288 encoder rows (8192+4096), CED128 finish_prefill including head/logits: seven warm runs max-rank 0.2035-0.2253s versus prior ~0.233-0.244s. Cold first 0.2690s. Separate 100/128/2048/8192/12288/12288 boundary run all 8 ranks PASS, CED 0.2287/0.2070/0.2012/0.1939/0.2260/0.2138s.

Real-input local candidate/selected IDs old/new bitwise equal on 8 ranks with reduction isolated. That instrumented run later hit CANN OOM; two clean uninstrumented engine runs passed. This is not a whole-model independent numerical oracle, HTTP deployment, or long-context validation.

Evidence/reproduce: /tmp/ced_verify_9p4et1qm and /tmp/ced_edges_9fmffql5 probe.py with restore.json bootstrap environment, rank*.json and rank*.pass. Local equivalence: /tmp/ced_verify_5isoflbf/rank*.correct.
```

---

```text
Align cold KV with pooled rank-zero storage and TP restore; forward decode metrics
```

---

```text
checkpoint: 当前是个检查点；冷KV上线已验收，开始论文式命中边界与CED优化前基线
```

---

```text
perf(prefill): merge bounded cold replay with uncached suffix

Deepest aligned hit; replay window and suffix in one eager encoder call.
Only new suffix updates compressed KV/carry. No new plan or production files.
4 files +26/-23. Rollback checkpoint: 27e1d3b.

TP8 instrumented worker TTFT: 3433 old .918s -> .512/.525s;
12289 old .904s -> .564/.574s. Eight-rank semantic tests passed
Paris/323/10/ORCHID up to 12377 tokens. Same-hit first-logit KL
1.6e-7..9.7e-6 on these cases, not general equivalence proof.

Deployed RPC: 10 correct streaming completions with EOS.
Repeated-hit TTFT: 26 tokens .487-.527s;3151 .544-.547s;
11424 .598-.613s. First 3151 nearly-miss 1.053s; first 11424
partial hit3122/suffix8302 two chunks 4.664s. No universal speed claim.
CED remains .217-.240s, NOT .1s. W8 BF16 residency rejected:
760MB/rank for only 10-20ms CED savings, no stable TTFT benefit.
No new B2-B4, vision or exhaustive quality regression.
Evidence: /tmp/paper_semantic_v08gyqq1, /tmp/paper_w8_9g2t5knb,
/tmp/paper_deploy_yg3k_le2 (RPC results and byte backups).
```

---

```text
Deploy paged CED128 NPU graphs and 1M context limits

Keep encoder eager and Past-owned history; register paged index as a Tensor op. Split 2048-row index pages into 1024-row views without KV copies; gather only sparse selected KV. Remove 16k graph dispatch. Align engine/frontend limits at 1048576 (Q6 headroom).

TP8 short full-model tests: all ranks passed, math and Paris generation/EOS match eager. Warm CED 134-138ms (14 tokens), 84.2-84.4ms (2241 tokens), below prior 243ms target. Paris eager/graph logits maxdiff 1.2337; not bitwise equivalence.

1M paged-index and pack/attention device tests passed (same graph dynamic lengths); index 13.823ms, not full CED latency. Full 1M end-to-end generation not tested. Production startup with max_seq=1048576 and one public API request passed HTTP200, answer323/EOS. First cold API TTFT2.096s includes lazy capture; not warm timing. Leave new engine/frontend running.
```

---

```text
fix: bound CED graph memory and preallocate serialized prefill workspace

Keep 1M max sequence and 8192 chunk; no plan layer. Bounded RoPE, shared per-layer graph pool, eager FP32 head, direct paged pack. Bind 1GiB CANN scratch before capture; drain/reset owners before unbind. Warm 128/8192/8192 before ready.

TP8 integration: 12 sequential requests through 64k and back to 8k. Eight ranks passed including owner close; no OOM, closed allocated bytes constant 55935988736, slots/tails restored. CED 82-87ms; 64k encoder 28-35s; final 8k encoder 2.04s. Allocator retries persist (up to 23); 1M end-to-end OOM freedom NOT established.

Deployed forward only. Native build and eight-rank startup passed. Authenticated temperature 0/1 chat HTTP200; sensible output, server TTFT 0.542/0.426s. No long-context numerical equivalence claim.
```

---

```text
perf(prefill): select final sparse indices with native NPU topk

Replace full stable score sort with torch.topk and retain sorted selected output for sorted_ids invalid-tail handling. Keep candidate-block ranking unchanged. No plan layer or context reduction.

Leaf tests: 16x65536 topk about 0.072ms vs sort 0.139ms; probe peak allocated 8.6MB vs 67MB. 16x524288 topk 0.445ms, score-multiset checks pass. Boundary tests widths 32 through 524288 pass including masked rows and ties.

Deployed TP8 service: 2k model prefill 0.728s; 8k 2.140/2.041s; 64k 19.690/18.016s. Five sequential requests completed without OOM; 8k/64k hit four prefix tokens. Not controlled A/B against prior isolated probe; no 1M end-to-end validation. Authenticated chat smoke returns 42. Remaining encoder performance gap not resolved.
```

---

```text
perf(prefill): skip redundant aligned W4A8 quantized copy

Six aligned/padded GMM leaf cases bitwise equal. Service 8K hot 1.403-1.553s; 32K 7.112-8.410s, jitter not resolved. 1M configuration unchanged, not retested.
```

---

```text
perf(prefill): quantize before expert dispatch and expose encoder/CED spans

Leaf quantization and full routed MoE bitwise equal for 1/128/512/8192 rows; 8k leaf ~8.19 to ~7.49ms. TP8 hot 8k model 1.349/1.476/1.407s; 32k 7.701/7.623/6.993s. Rank0 stream event spans report encoder and CED without per-layer sync; hot CED 82-84ms. Official chat template returns 42. No 1M validation or claim of resolved jitter.
```

---

```text
perf(prefill): prefetch Engram uploads on a dedicated stream

Cross-stream values, partial drains, exception reuse and repeated 8k/32k requests pass; chat returns 42. Full-model speedup not established: encoder 8k 1.717/1.277/1.277s, 32k 7.927/6.526/7.314s. CED 82-84ms; 1M not retested.
```

---

```text
perf(prefill): budget index query tiles and reuse aligned key bank

Old/new selection exact across 8191..524288 keys, including candidate paths. Single-device 8192-query/16384-key select 261ms to 107ms (no TP communication). TP8 zero-hit hot 8k total 1.347/1.356/1.349s; 32k 8.329/5.251/5.335s, CED 82-84ms. Chat 42 passes. Residual jitter remains; 1M full model not tested.
```

---

```text
diag(prefill): opt-in chunk/index spans and per-rank allocation retries

Five 32k zero-hit requests, four 8192-token chunks each: all four slow chunks coincide with a per-rank allocation retry outside select. Last chunk 1.384 -> 2.228s while select 0.461 -> 0.487s. Correlation only; triggering allocation not located. No per-span synchronization. Disabled 8k generation passed, no trace files added. No 1M end-to-end validation.
```

---

```text
perf(prefill): preallocate Engram upload and shared communication buffers

Allocate 990MiB shared communication storage plus 24MiB device upload slots before decode budgeting. Alias send/gather storage on the serialized stream; fixed pinned sources and consumption events protect cross-chunk reuse.

Leaf: exact old/new output with mocked collectives for ranks 0/7, full/tail reuse; pointer, early-drain and host-error reuse checks PASS. Real TP8: five zero-hit 32k requests 5.316/5.227/5.188/5.159/5.190s model prefill, CED 83.7-85.2ms; zero chunk allocation retries across all ranks in this sample. Chat generation 42 PASS. 1M/8192 configuration unchanged, not a 1M validation or full-path zero-allocation claim.
```

---

```text
perf(prefill): remove context-dependent index tile cliffs with shared scratch

Use one 32-query tile and explicit matmul/score/topk outputs. Alias only inactive Engram send tail, preserving live gathered residual; capacity checked at startup for 1M. Wire encoder and CED state ownership.

Leaf old/new exact indices and independent FP32 oracle pass, including tail shapes and 524288 keys; live-residual sentinel passes. Threshold 40960/45056-key leaf: old 221/442ms, new 88/94ms (8192 query rows, single rank).

TP8 zero-hit 131072-token model prefill: baseline 46.872s -> 24.303s and traced 22.410s; encoder 22.231s, CED 90.9ms. Chunk10/11 2.119/3.318s -> 1.344/1.303s. Alloc retries across ranks 38 -> 1, no OOM; remaining chunk14 stall unresolved. 32k 5.60s vs prior 5.22s mean: short-context tradeoff, not universal speedup. Formal-template chat returns 42/eos.

Reproduce leaf /tmp/index_uniform_327ef1tb/probe.py and traced/bench.py (enable traces/enabled). 1M capacity retained, not an end-to-end 1M or semantic long-context validation.
```

---

```text
perf(prefill): group TopK in shared arena while preserving TP reduction order

Unify existing Engram buffers into one unchanged 990 MiB arena; preserve live residual prefix. Keep 32-row matmul and HCCL reductions; group selection into fixed 128 rows without context-dependent dispatch. Eight-rank leaf checks pass exact ordered IDs, independent CPU oracle, residual sentinel, real scatter/gather, and 1M index extent. Full-model zero-cache 128k: 21.609748s / 6065 TPS and 21.555274s / 6081 TPS; encoder 21.464286/21.408926s, CED 90.526/90.345ms. Request wall 22.104302/22.399656s remains below 6k TPS. 32k model 5.335581s. Chat smoke 42, health OK. 1M configuration retained; no 1M full-model or long-duration validation claimed.
```

---

```text
fix(scheduler): defer >128k uncached admission; cap API at 512k

Keep FIFO pending jobs and reuse aligned cache lookup before DMA; engine 1M unchanged. CPU threshold/FIFO/cancel/lease tests pass. Live 131073-token admission waited 5.728s; active B1 178 steps at 32.474ms wall. API 524293 total tokens rejected with HTTP400, normal generation returns 42. No 1M full-model retest.
```

---

```text
config(server): rotate API authentication key
```

---

```text
config(server): name model deepseek-v41-flash
```

---

```text
Increase host prefix cache default to 50 GiB; verify generation and cache reuse
```

---

```text
fix(api): accept OpenAI user image_url blocks; verify blocking and SSE vision
```
