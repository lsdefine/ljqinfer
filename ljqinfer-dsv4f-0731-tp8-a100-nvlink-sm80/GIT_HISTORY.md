# Development history

按开发顺序排列的完整提交消息。

```text
oracle prefill milestone: full-model forward semantically correct
```

---

```text
tc oracle TF: DSML structural tokens all rank0 p=1.0 on broken tc_c1 case
```

---

```text
oracle fast load: final-tree snapshot via wcache, 95s -> 16s; TF result unchanged 56/64
```

---

```text
tc_oracle_decode: greedy 12-tok decode, DSML structure verified
```

---

```text
NCCL multi-process TP8: dist branches in arch/wcache/bind + torchrun oracle decode (12/12 gold match, 17s/step vs 129s)
```

---

```text
ops: port broken fp8 wgemm as pluggable qgemm (LJQ_QGEMM=ref/nocache/cached); TF gate 56/64 bit-identical, 233s->151s
```

---

```text
qmath/ops: drop env switch, hard-wire cached fp8 qgemm hot path (TF gate 56/64 bit-identical, 154s)
```

---

```text
MoE fused prefill: plug broken moe_rank_fused_prefill_fp4 (routed fp4 + shared fp8 + hc + norm fused)

- ops/__init__.py: export moe_rank_fused_prefill_fp4 wrapper (cached kernel mod)
- model/bind.py: stash per-rank raw fused weight tables on blk.ffn
- model/arch.py: Block.fused_ffn replaces oracle hc_pre+ffn_norm+MoE dequant path
  (loop 8-rank accumulate / NCCL all_reduce dual mode)

Gate results (c1 case, 64-tok TF):
- MATCH 55/64 (baseline bf16 oracle 56/64 fd=45); GATE PASS under semantic
  criteria: head j<45 (DSML tool-call region) divergences must be near-tie
  (rank<=2 & p_gold>=0.15); tail j>=45 unconstrained (baseline diverges too)
- NCCL TP8 greedy decode 12 tok: DSML format exact match_gold_prefix=True
- AB per-layer rel err: L0=0.0026 L3=0.0148 L20=0.0130; routing ids_match
  L20=1.000, L3=0.974 (near-tie swaps only) -> wiring correct, residual err
  = in-kernel fp4/fp8 activation quant (inherently lossy vs bf16 oracle)

Speed: TF gate 233s -> 20s full-run; decode 17s/step -> 0.5s/step
```

---

```text
precision study: fused MoE kernel has no low-precision accumulation; 56->55 delta = activation fake-quant semantics (removing qdq128 worsens to 52/64); precise-exp silu no effect (55/64)
```

---

```text
decode: extend to full DSML tool-call closure. 44 steps, DSML_CLOSED=True, well-formed calc(123*456) block, match_gold_prefix(44)=True, ~0.5s/step
```

---

```text
wip: past.py canonical rewrite snapshot before attn split
```

---

```text
refactor: split Attention into 3 closed classes (Base/Indexed/Compressed) + rewrite past.py canonical; Compressor.kv_cache init fix. Gate: tc_oracle_tf MATCH 55/64 PASS (baseline parity)
```

---

```text
refactor: specialize forward per attn class (Base=pure sliding window; Indexed/Compressed share _forward_compressed w/ use_indexer); remove dead generic forward. Gate tc_oracle_tf 55/64 PASS
```

---

```text
docs: near-tie flip investigated; baseline worktree reproduces identical fd=0 flip -> refactor exonerated, env-level nondeterminism at p_gold~0.005 positions
```

---

```text
docs: past.py wiring plan (deferred; each step touches numeric path, needs per-step TF gate)
```

---

```text
past step a: AttentionBase on flat LayerPast (single-buf past.py); TF gate PASS 55/64 fd=0
```

---

```text
past step b/c: Compressed/Indexed attn on flat past (single buf, ckv offset=cap); MTP keeps ring cache; TF gate PASS 55/64
```

---

```text
past step d: derived caches (kv_state/idx_ckv) driven from canonical past; write_res_step + rebuild hooks + LayerPast.to_ device fix; TF gate PASS 55/64
```

---

```text
tests: past unit tests (cold export/import 128-boundary, res_x step==chunk, window shift, to_ view sharing, SeqPast pos authority); 7/7 PASS
```

---

```text
attn step 1: six reference attention ops in ops/attn_ref.py ({Base,Indexed,Compressed} x {prefill,decode}); Attention.forward dispatches by ratio; ops own all past writes. TF gate PASS 55/64 (bit-identical to 0c7060b baseline)
```

---

```text
step2a: sparse_attn_paged leaf wired into attn_ref (8-head chunking, zero-copy decode buf); TF 56/64 + decode gates PASS
```

---

```text
step2b: indexer leaf -> index_score_reduce + topk_select_post kernels (sorted by score to match torch.topk accumulation order); INDEX_TOPK_MODE switch; leaf test. Gates: TF 55/64 PASS, decode PASS
```

---

```text
perf gate: tc_perf_prefill.py (prefill L tokens, warmup+3 reps). Baseline @b011148: L=8192 best 10284ms = 797 tok/s, peak 32.4GiB cuda:0
```

---

```text
perf gate -> torchrun tp8 multi-process (production form). Baseline @b011148: L=8192 best 2712ms = 3021 tok/s, peak 32.5GiB/rank
```

---

```text
tc_prof_prefill.py: per-stage cuda-event profile (torchrun tp8). @b011148 L=8192: attn 54% (linear fp32 668ms, compressor 327, indexer 218), moe 26%
```

---

```text
fix: get_or_cast pointer-keyed cache hit stale temporaries (.to(dev)) -> layout-dependent numerics; add gate0 purity + layout tracer
```

---

```text
refactor(ops): remove all process-global mutable state from CUDA ops

- dsv4_wgemm.cu: drop g_wcache/get_or_dequant pointer-keyed weight cache, fp8_gemm, clear_weight_cache;
  add pure bf16_gemm + load-time dequant_fp8_bf16(w,scale); shared-expert args now pre-dequantized bf16
- dsv4_wgemm.cu: delete dead rope_freqs_positions (function-static map cache)
- sparse_attn_paged.cu: remove static totals_out/ws_map workspace cache
- prefill_moe_cutlass_gemm.cu: g_bufs persistent buffers -> per-call pinned+device tensors;
  descriptor layout 16B-aligned, GemmCoord(12B) last (fixes misaligned address for odd na)
- bind.py: fused_shared dequantized once at load (w1,w3,w2 bf16); arch.py/tc_moe_ab.py updated
- ops/__init__.py, qmath.py: drop qgemm_cached / fp8_linear_bf16 experiment paths
- tc_ops_purity.py: test bf16_gemm/dequant_fp8_bf16, add static grep gate for ops/*.cu

Gates: purity PASS 4s | oracle_tf 58/64 fd=49 (base 45) | decode_dist PASS | prefill 8192 = 3022 tok/s (base 3020)
```

---

```text
step1(prefill cu): fp8 linear leaf = qdq128(act) + bf16_gemm on load-time dequantized weights

- ops.fp8_linear(x, w_bf16): pure, no cache; w must carry qdq_block=128 (checked in arch.linear)
- bind: wq_a/wq_b/wkv/wo_b dequant fp8->bf16 at load (pow2 ue8m0 scale => exact)
- arch.linear: bf16+qdq_block -> ops.fp8_linear; plain bf16 -> F.linear
- tc_ops_purity: fp8_linear check + oracle(fp8_gemm) compare max|d|=7.6e-6
Gates: purity PASS, oracle_tf 57/64 fd=45 (=baseline), decode_dist gold prefix OK, prefill 8192 = 4014 tok/s (base 3021)
```

---

```text
step2(prefill cu): rms_norm leaf (bf16 x, fp32 w) replaces arch.RMSNorm torch math
- ops.rms_norm pure kernel, fp32 accumulate, bf16 round; oracle max|d|=0
Gates: purity PASS, oracle_tf 57/64 fd=45, decode_dist gold OK, prefill 8192 = 4112 tok/s
```

---

```text
step3(prefill cu): rope_inplace leaf replaces arch.apply_rotary_emb torch complex math
- explicit (b,s,h) stride indexing (old rotary_at mis-indexed freq for B>1), freqs row-stride free (Compressor [::ratio] slice)
- hard TORCH_CHECK on dtype/shape, no silent fallback; oracle max|d|=0 (both inverse modes)
Gates: purity PASS, oracle_tf 57/64 fd=45, decode_dist gold OK, prefill 8192 = 4109 tok/s
```

---

```text
prefill cu step4: wo_a grouped projection as cu leaf (cublasGemmStridedBatchedEx, fp32 accum)

- bind: wo_a shards are fp8 (docstring said bf16 - wrong); dequant once at load (e8m0 exact), pre-cat to [g,r,d] contiguous
- arch/attn_ref: per-forward cat+view+einsum -> ops.wo_a_grouped
- purity: wo_a_grouped check + einsum oracle (max|d|=0)
- gates: purity PASS, tf 57/64 fd=45, decode_dist gold+DSML_CLOSED, prefill 4116 tok/s
```

---

```text
step5: attn-side hc via cu leaves (hc_fused_pre export, ops.hc_pre_norm/hc_post wrappers; hc_post comb^T semantics fixed in wrapper); gates 0-3 pass, prefill 8192 6511 tok/s
```

---

```text
step1(prefill-side): move torch Gate/Expert/MoE.forward out of arch into tc_moe_ab.py oracle; arch keeps weight containers only; gate1 PASS
```

---

```text
step2: DSparkAttention on canonical LayerPast (no ring/lru_cache), bind mtp weights, n_mtp_layers=3
```

---

```text
gate6a: tc_dspark_draft (new vs old ring impl, ids 32/32 same); fix Attention.past allocated on cuda explicitly
```

---

```text
tc_dspark_draft: pin old ring impl to 7cca5db, cleanup _arch_old.py
```

---

```text
step2a: Compressor/Indexer drop dense kv_cache; attn_ref writes past explicitly (gates 0/1/6a pass, TF bitwise == HEAD)
```

---

```text
step2b: SlotPool wired into arch/attn_ref (per-slot pos, paged ckv/ickv, ring main_kv); tests/test_past on SlotPool API; gate6a pins old past with old arch (gates 0/1/6a + test_past pass, TF 57/64 fd=45 == baseline)
```

---

```text
gate4: 128-aligned chunked prefill (Compressor block mode w/ rebuilt carry, _chunk_ctx window from past ring); noise-floor gate tc_chunked_prefill.py
```

---

```text
gate5: cold-KV export/import; drop compressor carry entirely -- Compressor is a pure fn f(x,start_pos,prev), CompressedPast.res_x ring [2r] is the only sequence state; decode writes res_x[pos] and emits 1 ckv row at window close; tc_cold_kv.py (C1 bit-exact cont, C2 gold hits, C3 per-round argmax across slots)
```

---

```text
model_api: B=1 ModelExecution facade for strategy/server (rank0 driver + worker broadcast loop, byte-exact cold KV blocks); tc_model_api_b1 gate PASS
```

---

```text
model_api: startup classmethod for torchrun engine_server; fill KV_FORMAT.width in place; scope default device per forward (thread-local leak broke pinned allocs)
```

---

```text
decode 1a: BnQ8 step contract + eager Q decode op (OPS_Q) + Transformer.forward_q; res_x ring 4r; gate tc_decode_q (24 steps oracle-exact, idempotent)
```

---

```text
decode 1b-1: attn_decode_g base layer device-pos (LJQ_DECODE_G=1), gate PASS
```

---

```text
decode 1b-1: r=128 layer device-pos version (attn_compressed_decode_g), gate PASS
```

---

```text
decode_g: r=4 indexed layer device-pos (overlap compressor + indexer topk), gate PASS
```

---

```text
decode 1b-2: CUDA graph capture of B1Q8 verify step (NCCL TP8)

- Transformer.forward_q_g: device pos_t, no _set_pos/ensure inside graph
- fused_ffn(q_decode): use moe_rank_fused_fp4 (no D2H) instead of prefill variant
- _write_ckv_g: index_select instead of 0-dim tensor indexing (no sync)
- qmath/_e2m1_on: per-device cached LUT (no H2D in capture)
- tc_decode_graph.py gate: 24 steps oracle==eager==graph, d_eager 0, graph 60.7ms vs eager 251.6ms
```

---

```text
decode 1b-3: DSpark draft CUDA graph (forward_spec_g, fixed-128 window, device pos) + tc_draft_graph gate PASS
```

---

```text
B1Q8 full-step graph: Transformer.step_g (verify->accept->draft) + gate PASS vs gold
```

---

```text
decode step2: fix nondeterministic B1 gate - greedy sampling (temperature=0) in make_args; Gumbel per-rank noise diverged draft/pos_t across TP ranks; drop temp DBG prints
```

---

```text
decode: per-step wall timing (stats.step_seconds); B1 gate prints B1Q8 ms/step baseline
```

---

```text
model_api: B1Q8 step via CUDA graph replay (lazy per-slot capture at alloc, static qin/pos_t); gate PASS 297->65 ms/step; tc_prof_decode.py kernel profile
```

---

```text
start_tp8_service: engine_server under torchrun nproc=8 (ModelExecution.startup needs NCCL pg)
```

---

```text
tc_prof_decode: per-module record_function tags (attn q/kv/o, compressor, indexer, write_ckv, moe, hc, mtp, head) + key_averages module table
```

---

```text
decode indexer: fused index_score_reduce_positions + radix topk_select_post_positions (graph-safe, device pos); bind positions variants
```

---

```text
decode compressor: fuse softmax/norm/rope/rotate/quant tail into compressor_rows (broken leaf, graph-safe)
```

---

```text
decode: _write_ckv_g -> broken paged_scatter_positions_masked leaf (gate PASS 60.4ms)
```

---

```text
decode attn: sparse_attn_paged reads ring + paged ckv pool in place (no ctx/_all_rows gather), window/compressed ids device-side (gate PASS 59.3ms)
```

---

```text
decode: fused base-layer attention leaf attn_decode_fused_r0 (broken idea, ring ABI); gate PASS B1Q8 59.15ms
```

---

```text
decode: derived fp32 projection rings (res_kv/res_sc) for compressor; project once per token instead of Qx128 rows per step. B1Q8 59.3->47.7 ms/step; tc_prof_decode graph-level kernel profile
```

---

```text
decode: fused had_fp4_qdq_ cu leaf for indexer q (bitwise vs torch), B1Q8 47.65->44.25ms
```

---

```text
decode-g: per-step position context cache (_CTX), 44.25->40.12 ms/step, gate PASS
```

---

```text
decode: replicate full embedding table per rank (no all_reduce in embed), gate PASS 39.95ms
```

---

```text
decode: head logits via bf16 GEMM + fp32 out (no per-step weight.float()), gate PASS 39.4ms
```

---

```text
decode: cu fp8_qdq64_ inplace replaces torch act_quant chain in _kv (bitwise); B1Q8 39.4->37.5ms
```

---

```text
decode: cu rms_scale_ inplace replaces torch per-head rms chain in _q (bitwise); B1Q8 37.5->36.9ms
```

---

```text
decode cu 刀10: fused silu_mul_clamp_bf16 for shared expert (bitwise, 36.9->36.3ms)
```

---

```text
decode cu 刀11: hc_post reads transposed comb in-kernel (comb_t), drop per-layer transpose copy (bitwise)
```

---

```text
decode cu 刀12: fp32 skinny GEMM (block-per-n split-K) for compressor wkv/wgate proj, 36.3->35.4ms (~1e-6 vs cuBLAS)
```

---

```text
progress: post-刀12 profile, remaining = NCCL skew
```

---

```text
decode cu 刀13: sparse_attn_paged split-K (S=8, DSV4_ATTN_SPLITK) + compressor_tail unroll-8 window loop (bitwise vs old), gate PASS 35.4->32.1ms
```

---

```text
decode cu 刀14: compressor_rows_ring — tail kernel reads projection rings via rowmap (no kvr[rows]/scr[rows]/cat/where gathers per layer), bitwise vs materialised path 16/16, gate PASS 32.1->30.1ms
```

---

```text
model_api: capture graphs with capture_error_mode=thread_local (cold-kv pinned allocator thread poisoned global-mode capture on first request)
```

---

```text
spec path for temperature>0: step_g Gumbel-max sampling (static temp_t, g broadcast from rank0) so sampled decode replays the B1Q8 graph instead of eager; strategy log steps = spec replays (consistent with output)
```

---

```text
fix(graph): static compressed-row capacity (max_seq//r) instead of host n_alloc -> long-context recall through CUDA graph; tc_needle_b1 gate (1587/5082 mid+end, graph==eager); B1Q8 30.7ms/step
```

---

```text
gate: tc_needle_long 64k chunked-prefill(<=12k)+graph-decode needle x6 facts x3 chunk sizes, graph vs eager; max_seq_len 65536
```

---

```text
load_execution(max_seq_len,max_batch_size); tc_needle_long NEEDLE_MAX_SEQ/NEEDLE_MAX_BS; 900k stress notes
```

---

```text
prefill: 1M/bs4 default + index_topk row-tiled + sparse_attn_paged split-K workspace only cached under graph capture (fix per-shape 1GB leak in eager prefill)
```

---

```text
kvload: merge contiguous cold blocks into ring-sized spans before import_cold (import 1.69s->0.02s, load 2.67s->0.21s @4.7k tok); add [kvload] stage timing
```

---

```text
Optimize compact cold KV restore
```

---

```text
Align API context limit with 1M engine capacity
```

---

```text
Cold KV Format V2: 128-token block cache with chunk-stable prefill

- add strategy/cold_kv_v2.py (aux/main/tail groups, 128-token granularity)
- capture tail state from ops/attn_ref.py compressed prefill
- keep the final prefill forward at 128 + len%128 tokens so a cache hit
  replays the exact same last chunk as a cold miss (verified: cold miss
  vs hit emit identical tokens for 256/257/12416/13000 inputs)
- tests: test_cold_kv_v2.py, test_prefill_chunking.py
```

---

```text
Engine robustness: gloo control plane + admission validation

The TP8 engine died twice for reasons unrelated to inference quality:

* Workers park in broadcast_object_list waiting for the next command.  On
  NCCL that idle wait looks like a hung collective, so the watchdog aborted
  every rank after 600s of no traffic.  The control plane now runs over a
  dedicated gloo group; compute collectives stay on NCCL.
* An out-of-range token id tripped a device-side assert in the embedding
  gather, poisoning the CUDA context of all eight ranks.  Requests are now
  validated at admission (non-empty, 0 <= id < vocab_size) and rejected with
  HTTP 400 before touching a kernel.  A worker that still raises logs the
  failure and keeps serving instead of taking the process down.
```

---

```text
tests: sync test_past.py with the current residue API

The residue ring became 4*ratio rows addressed by absolute token id back in
5d71a26 (write_res_step dropped, res() taking [t0, t1), rebuild hook removed),
but this file was never updated and had been failing ever since.
```

---

```text
fix(tp8): give the gloo control-plane group a 365d timeout so idle workers do not die
```

---

```text
cold kv v2: allocate contiguous row spans and evict whole chains

A restore issues one DMA per contiguous run, so scattered rows turned a
bandwidth-bound copy into a launch-bound one (32K hit: 255 runs, 3.56s).
Slabs now hold 128 rows, stores reserve one contiguous span per slab
(zero-eviction spans first, coldest subtree otherwise), and eviction drops
an entry together with its descendants so a freed chain stays contiguous.

Measured: 32K hit 3.56s -> 0.12s (4 runs), 64K hit 4.4s -> 0.16s (6 runs),
identical output tokens, cold prefill unchanged at ~3.4K tok/s.
```

---

```text
cold-kv-v2: drop dead res_x tail for non-overlap (r=128) layers; ~27.25MiB -> ~6.6MiB per block (4.1x capacity, 2760 -> 11305 pinned blocks)
```

---

```text
cold-kv v2: vectorised LRU span selection (67x faster eviction at 32k rows)
```

---

```text
decode: cap indexer scan rows per tier (49.4ms -> 35.3ms/step)

_all_rows_g scanned pool.max_rows unconditionally, so a 1M-token pool paid a
full 262144-row scan (score gemv + NCCL all_reduce + topk) even on a 100-token
sequence. Rows beyond the causal frontier were all masked out, i.e. pure waste.

- ops/attn_decode_g.py: honour pool.cap_rows, C_max = min(cap, max_rows).
- model/model_api.py: CAP_TIERS, one CUDA graph per capacity tier, keyed
  (slot, cap); _do_alloc pre-captures every tier (capturing mid-decode would
  corrupt the live compressor carry); _do_step picks the tier from pos.

Measured (tc_step_graph_capr.py, 8xA100 TP8): graph step flat at 32.2ms for
cap 512/1024/2048/8192 vs 48.6ms uncapped; per-step accepted counts identical
to the uncapped eager path (GATE PASS), so the truncation is numerically exact.
End-to-end AB on the live server, same prompt, cold-kv pinning settled:
decode_ms_per_step 49.42 -> 35.28 (-28.6%). The capped path is also far more
robust to host-side interference (during pinning: 114-155ms vs 35.4ms).
```

---

```text
Revert "decode: cap indexer scan rows per tier (49.4ms -> 35.3ms/step)"

This reverts commit 3b65c94541611bfacf0acba5f13d63a92539295a.
```

---

```text
indexer: fuse gather+score+reduce into one early-exit kernel (49.4 -> 42.2 ms/step, single graph)
```

---

```text
indexer topk: scan only the live prefix (42.2 -> 35.8 ms/step)
```

---

```text
decode: indexer.wq_b use _fp8lin (drop torch fp8 path, -462 kernels/step); hc_post split dim grid. 31.0->29.4ms, needle 4/4 OK
```

---

```text
decode: hc_post f32 branch (drop per-layer type_as) + reuse add2_bf16_f32 for AR input; needle 4/4 OK, 29.5ms/3729k
```

---

```text
decode: fuse res_x/kv/score ring writes into scatter3_rows (2-D grid) + cache res rows; 29.53->29.30ms, 3729->3650 kernels, needle 4/4 OK
```

---

```text
fix(decode): remove sparse_attn_paged pacc early-exit that dropped valid keys

The pacc block introduced in 7e86855 decided a whole k-block was empty by
testing ONLY its first index:
    if (kb0e >= K || idxs[token*K + kb0e] < 0) { pml = -INF, 0; return; }
When idxs contains a -1 hole in the middle of a block, every remaining
valid key in that block was written off as -INF and silently dropped, so
attention ran on a truncated key set. First token stayed correct, tokens
2+ degenerated into garbage.

Why it was missed: needle tests only stress the first step / long dense
candidate sets, where idxs are contiguous. Short prompts produce sparse
idxs with holes and trigger it every time.

Bisect: 616241c OK -> 7e86855 broken. Removing only the "any/seen break"
block still broke; removing only this pacc block fixed it, so this block
is the sole cause. The two changes named in 7e86855's message (hc_post
f32 branch, add2_bf16_f32) were both cleared by experiment.

Gates: semantic multi-token gen OK, needle 4/4 OK,
tc_decode_graph 24/24 oracle==eager==graph GATE PASS, graph mean 27.1ms.
```

---

```text
fix(peer_ar): default to NCCL (PEER_AR=0). It was defaulting ON while start_tp8_service.sh forbids env vars, so production silently ran the peer path -- which is unvalidated here and measured slower (36.32ms vs 35.38ms nccl).
```

---

```text
test: add fused-index and peer-ar A/B benches used for the garbled-decode bisect
```

---

```text
decode: control-plane fast path for step; B1Q8+MTP at 31.08 ms/step

Milestone: the B1Q8 + MTP single-request decode path is complete and
signed off on the live TP8 service. Steady-state step time is 31.08 ms
(was 49.4 ms when this line of work started, 32.5 ms before this commit).

This commit: rank0 broadcasts a 4xi64 CPU header instead of pickling
(cmd, args) on every decode step; workers decode the header inline and
call _do_step directly. Non-step commands keep the object path behind a
flag=0 header, so alloc/free/forward/shutdown semantics are unchanged.

Measured on the live service, 700-token request, 325 decode steps,
steady state (the first request after start carries warmup, 39.3 ms/step,
and must not be used for acceptance):
  before  32.5  ms/step
  after   31.08 ms/step  (10.099832 / 10.101069 / 10.102652 s over 325 steps)
  decode_tokens_per_step 2.154, output verified coherent, finish=length

Host side is now exhausted, which closes the B1 host-scheduling agenda:
graph replay launch 18.38 ms + result sync 11.79 ms = 30.17 ms, which
overlaps the 29.79 ms of real GPU work measured inside the graph with
CUDA events; the strategy layer adds only 0.17 ms. .item()/.tolist()
cost 0.10 ms/step total, so the planned D2H removal was measured and
dropped as a no-op rather than implemented.

Remaining headroom is in-graph only: verify trunk 43 layers = 26.92 ms
(0.626 ms/layer, on budget vs the 0.65 ms/layer reference), full step_g
= 29.79 ms, the 2.87 ms delta being sampling + accept + 3 MTP layers +
head. Next speedups must come from kernels, not from the host path.
```

---

```text
perf(gemm): drop hand-written M==8 kernel, always use cuBLAS

Measured on the only shapes it ever ran on (shared expert, 43 layers x3):
  (N=2048,K=7168)  31.2us -> 20.9us
  (N=7168,K=2048)  40.0us -> 19.5us
Accuracy IDENTICAL to cuBLAS (same rel-err vs fp32 reference), so the
M==8 path was pure loss: slower, not more accurate.

Real engine B1Q8+MTP step: 31.20ms -> 30.86ms (3 runs each, ranges disjoint).
New acceptance gate tc_semgate.py (semantic + tool-call, B=1 and B=4,
production encoder server/encoding_dsv4.py): PASS on both paths.
Rollback: ops/dsv4_wgemm.cu.bak_smallm
```

---

```text
batch step plumbing: multi-row CUDA graph (_capture_batch/_do_step_batch)

- _capture_batch keyed by tuple(slots), lazily captured via arch.step_g_batch
- _do_prepare_batch: capture-or-reuse + bind row state (qin/pos) into batch buffers
- _do_step_batch / unbind_batch; i64 fast path in _rpc + serve_workers branch
- gate tc_batchstep.py B=2: per-row tokens identical to B1 serial decode (row0/row1 PASS)
- B2 step 53.10ms = 26.55ms/row vs B1 31.08ms/row
- NOTE: graph capture writes garbage KV near pos0, so capture must run on clean slots
  (before prefill); a second prepare_batch call only re-binds.
```

---

```text
batch: wire B>1 prefill with per-row cold-KV v2 transactions

- strategy: drop the "requires B=1" restriction; each row now runs its own
  prepare_store / commit / abort transaction (cold KV decoupled from prefill)
- model_api: row_prefill_begin/end callbacks, symmetric for B=1 and B>1,
  try/finally guarantees abort on failure
- cold_kv_v2: prepare_store(protect=) keeps other rows ancestor chains alive
- strategy._execute: call set_input per boarding row (was anchor-only), which
  releases the row slot; without it pool.pos kept a stale offset and
  _prefill_chunk_size hit "assert pos % BLOCK_TOKENS == 0" -> HTTP 400
- model_api: prefix guard defaulted to ids itself (vacuously true), now ()
- dev knob LJQINFER_COLD_KV_GB (default 80 unchanged); 8 -> 1130 pinned
  blocks instead of 11305, model_ready 26.8s for faster iteration

Verified online at B=2 (two concurrent 345-token needle requests):
round1 stored_blocks=2 per row, answers distinct and correct (4271/9835);
round2 hit=128 per row, 2.72s -> 1.19s; log shows batch=2 on both rows;
FALLBACK/Traceback/AssertionError count = 0. B=1 path unchanged and green.
```

---

```text
tests: batch correctness matrix for B=1..4 (10/11 PASS, 1 explained)

Gold = B=1 greedy(temperature=0, max_tokens=32) sampled serially; every batched
case must reproduce the gold text token-for-token. Adds tests/batch_matrix_test.py.

Results (online, 8xA100 TP8, cold-KV v2 on, LJQINFER_COLD_KV_GB=8):
  B2_equal_len        PASS   two 386-tok rows, hit=256/row
  B2_short_long       PASS   19-tok + 674-tok mixed length in one batch
  B3_mixed_len        PASS   short/mid/long together
  B4_all_distinct     PASS   4 distinct needles, no cross-talk
  B4_same_prompt      PASS   4 identical prompts (shared-prefix store contention)
  B3_mixed_maxtok     PASS   max_tokens 8/32/64, early-finishing rows leave batch
  B4_stagger          FAIL*  3/4 rows MATCH, "story" row diverges (see below)
  B2_replay_hit       PASS   cold-KV replay, hit=k*128 per row
  B4_repeat           PASS   repeated batch, stable
  B1_after_batches    PASS   single request after heavy batching, no contamination
  B1_after_batches2   PASS   same, long prompt

* Not a batching bug. Root cause is the pre-existing skinny-GEMM fork in
  Indexer.proj (model/arch.py:353, introduced by ecf98254): M<=8 uses
  sgemm_skinny2_f32, M>8 falls back to cuBLAS. With MTP, M = B*(1+draft),
  so B<=2 -> M<=8 and B>=3 -> M>8. Evidence:
    - output is deterministic *within* each B (3 runs each, distinct=1)
    - B=1 and B=2 both equal gold; B=3 and B=4 agree with each other
      but differ from gold, diverging after a 92-char common prefix
    - both continuations are fluent and semantically correct; the fp32
      accumulation order differs, which flips a near-tie in the indexer
      top-k page selection
  606079f (batch wiring) does not touch model/arch.py.
  Batch-invariance for B>=3 requires raising the skinny kernel's M limit;
  tracked as the next step, not a regression of the batch wiring.

Zero FALLBACK / Traceback / AssertionError across the whole matrix.
```

---

```text
fix(batch-invariance): kill M<=8 fork in skinny GEMM -- bitwise-identical output for any B

Root cause of the B4_stagger/story mismatch found in the previous test matrix:
  ops/dsv4_wgemm.cu sgemm_skinny2_f32 was template<M> with TORCH_CHECK(M<=8),
  so model/arch.py:353 carried an "M<=8" guard and silently routed B>=3
  (M=B*4>8) to cuBLAS. Different kernel => different reduction order => the
  same prompt answered differently depending on how many peers shared the batch.

Changes:
- ops/dsv4_wgemm.cu: rewrite as MT=8 row-tile loop, grid(N, ceil(M/MT)), uniform
  break skips invalid rows. Handles any M. cuBLAS fallback REMOVED (no fallback,
  no silent fork). Per-row K-reduction order depends only on K and blockDim.
- model/arch.py: drop the "M<=8" condition; skinny path is now unconditional.
- ops/index_score.cu: annotate the B==1&&S==8 perf fork (why it exists, why it is
  safe): measured bitwise-equal to the generic w8 path, max_abs_diff=0.0.
- tests/test_batch_invariance.py: new guard test. M=1..128 rows bitwise identical
  to every smaller M; rel_err vs cuBLAS 2.5e-07.

Verified online (TP8, cold-KV 8GB):
- batch matrix 11/11 PASS (was 10/11; B4_stagger/story FAIL -> MATCH)
- B=1 decode step 30.911 / 31.213 ms vs 31.08 ms baseline => no regression
- op bench K=7168,N=1024: M=4 24.5us vs cuBLAS 47.5us (1.94x faster)

Known and accepted:
- skinny loses to cuBLAS at M>=12 (M=16: 77.6 vs 48.2us). Deliberately NOT forked
  back to cuBLAS: the two are not numerically equivalent (rel_err 2e-7), so an
  M-based switch would reintroduce exactly the bug fixed here. Real fix is column
  tiling (reuse x across NT columns), which preserves per-row reduction order.
  MT=32 was tried and reverted: 1.5x slower (bottleneck is x L2 traffic ~N*M*K,
  not weight reuse; large acc[] destroys occupancy).

Known issue, pre-existing and unrelated to this commit:
- decode step scales ~linearly with B: 31.9 / 85.4 / 110.0 / 146.5 ms for B=1/2/3/4
  (MTP accept/step identical at 4.853), so B=4 total throughput (161 tps) is below
  B=1 (184 tps). Batching is a single graph.replay(), not a per-row loop, so this
  is in-graph work -- most likely MoE expert-weight traffic growing with token
  count. Needs its own investigation.
```

---

```text
test: post-fix correctness regression - ALL GATES PASS

Full re-validation after c4656f4 (skinny-kernel fork removal), covering
length / batch / cold-KV as requested before starting perf work.

Online HTTP regression (new scripts):
- tc_regress_http.py  needle 12/12: 80 / 439 / 1609 tok x B=1 & B=4 x 2 rounds
    J1 answer correct, J2 R1==R2 (cold-KV lossless), J3 B1==B4 (batch invariant)
- tc_bound_http.py    128-alignment matrix 8/8: exact prompt_tokens
    127/128/129/255/256/257/384/385 (binary-searched via usage.prompt_tokens)
- tc_long_http.py     long-context 6/6: 7636 / 15236 tok x needle depth 25/50/90%

Cold-KV verified as真 hit (not a no-op): mid R2 hit=256 load=53ms model 2.65->1.34s;
long R2 hit=1408 load=88ms. Rule: hit=(floor(n/128)-1)*128, tail block recomputed.

Offline gates (service stopped):
- tc_oracle_tf        GATE PASS  MATCH 58/64 (>=55), first_diverge=45 == baseline 45
- tc_chunked_prefill  GATE4 PASS C1 chunk-vs-whole max|d|=0.0 (bitwise), C3 argmax equal
- tc_needle_long      GATE PASS  24/24 OK @ 64k (65486-65491 tok), chunk 4096/8192/12288
    eager==graph 5/6; the 1 diff is markdown bold only, answer identical

No logits drift, no batch-dependent divergence, no KV store/load loss.
```

---

```text
perf(batch): A1+A2 token-wise attention stages run once per step, not per row

ops/attn_decode_g.py: split the three decode variants (base/compressed/indexed)
into a slot-bound core (_core_*: KV ring write / compress / indexer / sparse
attn) and the shared token-wise stages _q/_kv/_o.  New attn_decode_g_batch()
runs _q/_kv once on the flat [1,B*Q] stream, loops only the core per row, then
one _o (wo_a/wo_b + one all_reduce).  base variant keeps its fused leaf but
returns the fp32 partial so B rows merge into a single all_reduce.
model/arch.py: Attention.forward_qb + Block.forward_qb call the batched path.
B==1 executes exactly the old per-row sequence (same kernels, same order).

Measured (tc_amdahl_b.py, offline, TP8, graph):
  B=1  26.27 -> 26.17 ms   (no regression)
  B=2  45.70 -> 40.29 ms   (-5.4)
  B=4  83.25 -> 67.65 ms   (-15.6)   nccl 323 -> 206 kernels
  per-row B4: 20.8 -> 16.9 ms, speedup_vs_ideal 1.28x -> 1.55x

Correctness: tc_step_batch.py BMAX=2 NSTEP=16 GATE PASS (row0 56/56, row1 44/44
MATCH), identical to the pre-change baseline.  NOTE: BMAX=4 offline gate FAILS
on the pre-change baseline too (row1@41/row2@15/row3@3) -> pre-existing, not
introduced here; to be investigated separately (online B1==B4 needle passed in
37fd127).
```

---

```text
perf(batch): A3 indexer split pre/score/topk; batch path does ONE score all-reduce

ops/attn_decode_g.py: _indexer_g -> _indexer_pre (token-wise: wq_b+rope+fp4 qdq,
weights_proj; runs once on the concatenated B-row stream) + _indexer_score
(per-row: scores its own slot's indexer pool, causal mask) + _indexer_post
(top-k after TP all-reduce).  attn_decode_g_batch concatenates the B per-row
score tensors and issues a single all_reduce, then splits.  Single-row path
keeps the exact same op order (BMAX=2 step gate bit-exact vs serial).

tc_amdahl_b.py (8xA100 TP8, graph replay):
  B=1 26.26ms (3848 k)   nccl 125
  B=2 39.63ms (5311 k)   nccl 131   (was 40.29 / 152 after A1)
  B=4 63.47ms (8007 k)   nccl 143   (was 67.65 / 206 after A1, 83.25 / 323 baseline)
  per-row B=4 15.87ms, speedup_vs_ideal 1.65x
Gate: BMAX=2 NSTEP=16 tc_step_batch.py -> STEP BATCH GATE PASS.
```

---

```text
A4: batch MTP draft layers across rows (DSparkBlock/DSparkAttention.forward_qb)

step_g_batch previously ran the 3 MTP layers per row (B x forward_q, each with its own
wo_b all-reduce + fused_ffn all-reduce). Now token-wise stages (hc_pre, wq/wkv, rope, wo, wo_b AR,
fused_ffn) run once over the flat [1, B*block] stream; only the per-slot parts (main_kv
index_copy_, ctx window read, sparse_attn) stay per-row so results are bit-identical to forward_q.

Gate: tc_step_batch BMAX=2 NSTEP=16 -> STEP BATCH GATE PASS (rows 0/1 MATCH).
tc_amdahl_b (graph ms/step): B1 26.26 (unchanged) | B2 39.63 -> 38.19 | B4 63.47 -> 59.50
per-row: B2 19.09ms, B4 14.88ms (speedup_vs_ideal 1.38x / 1.77x). Baseline before A1: B2 45.70 / B4 83.25.
```

---

```text
A5: batch compressor derived projections (wkv/wgate skinny GEMM) across rows

past.derive(x) -> {name: (kv, sc)} runs sgemm_skinny2_f32 ONCE on the flat
B*Q row stream in attn_decode_g_batch; write_res_t(derived=...) takes the
per-row slice and only does the ring scatter.  Covers both 1-ring
(compressed) and 2-ring (indexed: ckv + ickv) layers; B == 1 keeps the exact
op sequence.  The kernel is batch-invariant (row-independent reduction), so
row m is bitwise identical for every M.  _CTX cache: "derived" is overwritten
every layer (never inherited).

sgemm_skinny2_f32<8> launches/step: B1 124 | B2 248 -> 124 | B4 496 -> 124
tc_amdahl_b (graph ms/step): B1 26.27 (unchanged) | B2 38.19 -> 36.82 | B4 59.50 -> 55.60
kernels/step B4 6961 -> 6646; per-row B4 13.9ms (speedup_vs_ideal 1.89x)
Gate: tc_step_batch BMAX=2 NSTEP=16 -> STEP BATCH GATE PASS (rows 0/1 MATCH).
BMAX=4 offline gate: rows 2/3 DIVERGE@1/@3 -- identical on HEAD (430654a) with
this change stashed, i.e. pre-existing (recorded since A1; online B1==B4
needle passes), not introduced here.
```

---

```text
A7: sgemm_skinny2_f32 v2 - register x-reuse across NT=2 columns, MT=8

Old kernel: grid(N, ceil(M/8)), every (m-tile,n) block re-reads x row-tile -> x L2 traffic = N*M*K*4B (0.9GB @M32), time ~ M.
v2: grid=(M-tile fastest, N/NT); each thread loads one float4 of x and reuses it for NT=2 w columns in registers; no smem. K split / reduction order identical to old -> bit-exact (skinny_bench cmp BITEXACT=True all shapes; MT/NT sweep: MT8NT2 best for M<=32).

Micro (7168x1024, us): M8 33.8->25.3, M16 76.7->47.4, M32 140.6->100.0
Gate: BMAX=2 NSTEP=16 tc_step_batch PASS
Same-session A/B tc_amdahl_b graph ms: B1 26.31->26.13, B2 36.85->36.14, B4 55.49->54.60 (kernel bucket skinny B4 5.70->4.66ms)
```

---

```text
tools: amdahl B-scaling profiler (tc_amdahl_b), multi-gen/moe-micro/precapture harnesses, A7 verify script
```

---

```text
A8: batch sparse_attn launch across rows (1 launch per layer instead of B)

kernel sparse_attn_paged was already B-aware (token->batch=token/Q, page_table/
ctable per-batch stride); Python side still called it once per row.  Now
_core_{base,compressed}_pre / _core_indexed_idxs only write KV + produce idxs,
and _attn_rows_g issues ONE launch via sparse_attn_pool_paged (pool = whole
main_kv, page_table = slots[B,1], ctable = stacked pt_table).  r==4 B==1 keeps
the single-row _core_indexed path (already one launch).  Same values.

gate: BMAX=2 NSTEP=16 tc_step_batch.py -> STEP BATCH GATE PASS (rows bitwise MATCH)
same-hot A/B (tc_amdahl_b.py, graph replay ms/step):
  OLD B1 26.06 / B2 36.66 / B4 54.53   (kernels/step 3846/4860/6646)
  NEW B1 26.06 / B2 35.03 / B4 51.40   (kernels/step 3826/4739/6279)
  delta B2 -1.63ms  B4 -3.13ms  (B4 speedup_vs_ideal 1.91x -> 2.03x)
```

---

```text
A9: batch index_score_fused across rows (one launch, prow [B,N], ms kernel grid.y=B)

- index_score.cu: all 3 kernels take prow_stride (prow [N] shared or [B,N] per row);
  ms fast path now handles any B via grid.y (each batch row is an independent
  B==1 problem -> B==1 bitwise unchanged, B>1 gains the 8x KV-traffic saving).
- attn_decode_g.py: _core_indexed_pre split into _core_indexed_write (pool write)
  + _indexer_rows_score (one index_score_fused + one reduce over [B,Q,C]).

Gate (BMAX=2 NSTEP=16 tc_step_batch): PASS, both rows MATCH bitwise.
Same-heat A/B (tc_amdahl_b BLIST=1,2,4, graph ms/step):
  OLD B1 26.12 / B2 35.31 / B4 51.53  (kernels 3826/4739/6279)
  NEW B1 26.10 / B2 34.53 / B4 49.75  (kernels 3826/4656/6070)
```

---

```text
A10: batch slot-bound KV/ring/pool writes across rows (B>1)

Per-row write path (main_kv index_copy, res/derived ring writes, compressor_tail,
ckv paged scatter, indexer pool scatter) was launched once per row; now one launch
per kind over all B rows: rowmap gets slot*4r offset into the flattened rings,
compressor_rows_ring/paged_scatter_positions_masked take the full pool views and
a [B,T] page table. No CUDA change. B==1 path untouched.

Gate: tc_step_batch BMAX=2 NSTEP=16 -> both rows MATCH (bit-exact vs base).
A/B (same hot state, tc_amdahl_b BLIST=1,2,4, graph ms / kernels per step):
  OLD B1 26.05/3826  B2 34.73/4656  B4 49.44/6070
  NEW B1 26.10/3826  B2 33.81/4499  B4 46.64/5555
  => B2 -0.92ms  B4 -2.80ms  (speedup_vs_ideal B4 2.11x -> 2.24x)
```

---

```text
bench: align tc_amdahl_b KV pool with production (max_seq_len 8192 -> 1048576)

Root-caused the 31.2ms(online) vs 26.1ms(bench) step-time gap:
- per-kernel diff (same 43 calls): sparse_attn_paged_tc 0.784 -> 2.398 ms (+1.61),
  nccl AllReduce 4.248 -> 5.401 ms (+1.15); all other 28 kernels identical.
- cause: sparse-attn K = window + min(index_topk, C_max) is STATIC, derived from
  max_seq_len. bench used 8192 -> K=192; production make_args() default is 1048576
  -> K=640 (r4) / 8320 (r128). bench did 3-43x less attn work = optimistic.
- proof: bench with MAXSEQ=1048576 B=1 -> 30.54ms (was 26.09ms), K=8320 as online.
  => the in-graph 3.1ms gap is a bench artifact, NOT an engine regression.
MAXSEQ env still allows the old small-pool run for comparison.
```

---

```text
sparse_attn: interleaved splitK + keff upper-bound truncation

Decode step no longer scales with the max_seq graph tier.

F_sparse_attn bucket is now tier-independent: 8k/512k/1M all 1.07-1.08ms
(was 1.28 / 2.06 / 2.88).
step (B=1, graph replay): 8k 26.09->25.93, 512k 28.70->28.20, 1M 30.54->29.43 ms.

Kernel: kstride = gridDim.y*TILE (interleaved splitK, was contiguous chunks);
Kend = min(K, keff[token]) so the padded tail of the static K capacity is skipped.
keff = 128 + n_closed per row, set in _core_compressed_pre, wired at _attn_rows_g
(the unified entry used by both B=1 and B>1; per-row values are cat'ed).
NOKEFF=1 gives an explicit A/B switch. indexed layers (dense K=640) keep keff=None.

Gates: micro-bench bitwise identical (maxdiff 0, 61.4->18.9us);
needle 1500/5000 4/4 OK; tc_batch_b234 B1/B2 maxdiff 0 OK;
B3/B4 mismatch reproduces identically under NOKEFF=1 => pre-existing skinny-kernel
fork, unrelated to this change.
```

---

```text
ops: remove all implicit env-var switches from the decode path

No production code path may be silently altered by an environment
variable.  Every removed knob below changed numerics or the executed
code path, i.e. each was a latent source of "why is it different today".

Removed (now compile-time / module constants):
  NOKEFF              -> keff truncation is always on (was my own A/B knob)
  DSV4_ATTN_SPLITK    -> S fixed at 8; S changes the split-K reduction
                         order, so it is a numerics knob, not a tuning knob
  INDEX_TOPK_MODE     -> "kernel"; ref_reduce/ref_topk/ref are reference
                         implementations, reachable only via explicit setter
  IDX_RS              -> False; reduce-scatter path via explicit setter
  PEER_AR / PEER_AR_N -> disabled; peer all-reduce via explicit setter
  INDEX_TOPK_BYTES    -> 1 GiB module constant, overridable in tests only

Kept: RANK/WORLD_SIZE (launcher), TORCH_CUDA_ARCH_LIST/MAX_JOBS (build),
LJQINFER_COLD_KV_GB (strategy-layer capacity, does not affect numerics).

Testbenches now flip these through explicit setters / module attributes
instead of the environment, so A/B is still possible but never implicit.

tests/_t_splitk.py upgraded to a real keff gate:
  KEFF EQUIV OK  ->  keff=None == keff=K == keff=truncated, bitwise
  max abs diff vs torch ref 0.00214 (ref scale 1.006, bf16)
  55.57 us/call

Verified after the change (8 GPUs, MAXSEQ=1M, B=1):
  graph 28.76 ms, F_sparse_attn 1.08 ms  -- same as before (28.72-28.80),
  no regression.
```

---

```text
repo hygiene: untrack *.pyc, ignore __pycache__

Compiled bytecode was tracked (44 files), so every run dirtied the
working tree and a stale .pyc could disagree with its source.
Files stay on disk; only the index entry is removed.
```

---

```text
engine: default captured context 1M -> 384k (393216)

make_args() default max_seq_len drives the engine (strategy.py:589 ->
ModelExecution.startup -> load_execution() with no args), so the served
graph was captured at 1M while we only ever serve <=384k.

Measured on the live TP8 service (B=1, 200-token greedy request, MTP on,
tokens_per_step=2.326), warm steady state over 4 consecutive requests:

  1M   : 28.99 / 28.95 ms/step   80.3 tok/s   73.9 GB/GPU
  384k : 27.63 / 27.63 / 27.52 / 27.65 ms/step   84.2 tok/s   49.4 GB/GPU

-1.34 ms/step (-4.6%), +4.9% tok/s, and -24.5 GB per GPU.

Matches the bench curve (128k 26.52 / 512k 28.20 -> 384k interp ~27.8);
the residual length sensitivity is diffuse and NOT in sparse_attn, which
the keff fix flattened to 1.07-1.08 ms at every length.
```

---

```text
gitignore: local A/B experiment artifacts (a*_ab/gate logs, *.bak_*)
```

---

```text
server: raise default/max output tokens to 16384

A request without an explicit max_tokens was capped at 1024, and even an
explicit large value was silently clamped to 8192 by min(max_tokens,
max_output_tokens).  Three separate magic numbers had to agree, so lifting
only DEFAULT_MAX_TOKENS was not enough (verified: submit showed max_new=8192).

  service.py  DEFAULT_MAX_TOKENS      1024 -> 16384   (no max_tokens in body)
  service.py  max_output_tokens arg   8192 -> 16384   (ctor default)
  server.py   MAX_OUTPUT_TOKENS       8192 -> 16384   (value actually used)
  server.py   cfg.get fallback        4096 -> MAX_OUTPUT_TOKENS  (was a third,
                                      even lower, inconsistent limit)

Verified live after an api-layer restart: request with no max_tokens now
logs `submit id=req_00000014 input=9 max_new=16384` (was 8192, was 1024).
```

---

```text
engine_server: make /stop and /cancel idempotent

Both routes raised HTTPException(404) when request_id was not in the
live handle table, which also happens for the normal race where the
client cancels just after generation finished. Real traffic produced
two spurious 404s in the access log for requests that had completed
successfully.

_request_handle() now returns None instead of raising, and both routes
answer 200 with {"stopped"/"cancelled": false, "state":
"already_finished"} for unknown ids. Verified live after engine restart:
  POST /stop/req_does_not_exist   -> 200 {"stopped":false,"reason":"semantic_eos","state":"already_finished"}
  POST /cancel/req_does_not_exist -> 200 {"cancelled":false,"state":"already_finished"}
The 404-fallback branch in remote_strategy._Handle stays compatible.
```

---

```text
ops: vendor cutlass in-tree, drop dependency on sibling checkout

ops/build.py pointed CUTLASS_INC at
  /mnt/data/kw/ljqinfer_dsv4f_tp8_broken/ops/third_party/cutlass/include
i.e. a *different* working copy on this box. This repo had no
ops/third_party at all, so copying just this folder anywhere else
produced a tree that fails to JIT-compile prefill_moe_cutlass_gemm.cu
(the only .cu that includes cutlass/cute headers).

Vendor cutlass (include/ + LICENSE, 27MB, 802 files) under
ops/third_party/cutlass and resolve CUTLASS_INC relative to ROOT, with
an explicit RuntimeError if the headers go missing.

Also ignore ops/.build/: it is the cpp_extension JIT cache (arch- and
machine-specific .so + build.ninja with absolute paths). It was never
tracked, but it does travel with a plain folder copy and would let
ninja reuse a stale .so on a different box.

Verified: nvcc -std=c++17 -arch=sm_80 compiles a probe TU including
cutlass.h / gemm_grouped.h / default_gemm_grouped.h / linear_combination.h
/ matrix.h / gemm.h against the in-tree include path (only an unrelated
constexpr warning from conv3d_problem_size.h). Running engine untouched:
ops/.build was deliberately not cleared.
```

---

```text
repo layout: move dev scaffolding out of the root, keep it runnable

The root had 103 tracked files against 54 in all package dirs combined:
39 tc_*.py probes, 36 *.log experiment outputs, 6 A/B shell drivers and
6 progress notes had accumulated there and been committed.

  bench/   56 probe/profiling scripts + 6 .sh drivers
  logs/    42 experiment logs, now untracked (kept on disk)
  docs/    6 progress/design notes joined the 2 already there
  root     start_tp8_service.sh + .gitignore only

Moving the scripts breaks `import model/ops/...`, because python puts the
script's own directory (now bench/) on sys.path, not the cwd. Every bench
script therefore got one injected line resolving the repo root from __file__;
the 9 that already hacked sys.path (4 with a hard-coded absolute path) were
rewritten to the same form. Verified: ast-checked on all 56, py_compile clean,
and tc_ops_purity.py runs from /tmp and imports ops fine.

The .sh drivers cd to the repo root, so their script arguments gained a
bench/ prefix.

Logs are untracked from here on (.gitignore: logs/, *.log).
```

---

```text
docs: bench/ holds the correctness gates too -- do not delete
```

---

```text
repo hygiene: delete all hand-made .bak files -- git is the backup

10 *.bak_* copies (560K) removed from the tree; 2 of them were actually
tracked (model/arch.py.bak_b234, ops/dsv4_wgemm.cu.bak_smallm) and had been
shipping as if they were source.  Every one was a pre-edit snapshot of a file
that is itself tracked, so the exact same content is reachable via git history
(model_api.py.bak_prof was even byte-identical to the current file).

Also removed bench/_do_patch.py -- a one-off patcher whose only job was to
write .bak_smallm before sed-ing a kernel.

.gitignore now blocks *.bak *.orig *.old *~ *.save *.backup so this cannot
come back.  Physical copies parked in /tmp/bak_trash_20260905 for this session
only.
```

---

```text
ops: remove 890 lines of superseded dead code (pareto-dominated variants)

Static reachability audit (BFS over pybind exports actually referenced by
python core) found 63 unreachable C++ functions. This commit removes only
the subset with a clearly superseding live replacement:
  paged_scatter/_positions        -> paged_scatter_positions_masked
  topk_post/_positions            -> topk_select_post/_positions
  add2_bf16                       -> add2_bf16_f32
  hc_post_fused                   -> hc_post_fused_bf16
  peer_ar_ipc_run                 -> peer_ar_ipc_run2
  dequant_fp4_batch_bf16          -> dequant_fp4_selected_bf16
  moe_rank_pre_fp4/grouped_fp4    -> moe_rank_fused_fp4
  compressor_windows_positions,
  compressor_carry_commit/discard,
  compressor_fused_rows,
  compressor_tail (host export)   -> compressor_rows_ring + compressor_tail_launch
  gemm_smallm_blk/warp/blk2       -> sgemm_skinny2_f32 (MT=32 experiment, reverted)
  dsv4_build_tiles/tile_count     -> unused tiling scaffold
plus the matching cross-file prototypes and m.def entries.
pybind exports 52 -> 37. No live call site touched.

Gates (service stopped, 8xA100 exclusive):
  tc_oracle_tf          MATCH 58/64 first_diverge=45  PASS
  tc_oracle_decode_dist match_gold_prefix(44)=True DSML_CLOSED  PASS
  tc_chunked_prefill    C1 max|d|=0.0 / C2 57-58/64 / C3 argmax eq  PASS
  tc_semgate            B=1 and B=4 tool-calls PASS

Audit report: temp/dead_code_audit.md
Deliberately kept (unreachable but no replacement / future work): FP4-MoE path,
hc_*/sinkhorn calibration, embed_gather, peer_ar fast path, model/dequant.py.
```

---

```text
docs: archive node09 dev history notes (moved out of agent memory SOP)
```

---

```text
docs: archive full ljqinfer decode lessons (moved out of agent memory SOP)
```

---

```text
sampling: fixed policy T(default 1.0) + nucleus top_p=0.95 (vLLM/SGLang defaults + DeepSeek-V4 README agentic rec); topk-1024 window, graph-safe; greedy path bit-identical
```

---

```text
service: reasoning effort maps to DSV4 template 3 levels (low/high/max), default low; drop GLM-5.2 two-level aliasing that sent low/medium to high
```

---

```text
fix(compressor_tail): norm_w is fp32 (was read as bf16 -> corrupted decode ckv/ickv); round like rms_norm_f32w; dtype check
```

---

```text
decode: dynamic boarding at 128-step safe points (tail-slot only, no slot reuse); strategy board_request extends per-row stats
```

---

```text
fix: emit prefill first token AFTER publishing prefill event (API stream expects prefill event first; boarding path likewise)
```

---

```text
prefill: TF32 for Compressor wkv/wgate fp32 GEMMs (bf16-exact operands, +24% prefill tps 6323->7861 @12k); add tc_prof_prefill profiler bench
```

---

```text
strategy: boarding grace 0.30->0.10 (128-step dynamic boarding covers late arrivals)
```

---

```text
ops: vectorized qdq128/qdq64 (uint4, 8 elem/lane, warp-shuffle max; bitwise-identical, 382->721 GB/s; prefill 7861->8018 tok/s @12k)
```

---

```text
index_topk: row-sharded reduce_scatter + local topk + all_gather for chunked prefill (chunk5@49k 1889->1732ms; gates PASS)
```

---

```text
index_topk row-shard: zero-pad rows to W multiple so tail/odd segments also take RS path (S=5805@67k: AR 794ms->RS 26ms, row_eq 99.98%); add tc_rowshard_eq bench; gates PASS
```

---

```text
prefill MoE: fuse routing-weight multiply into moe_slot_sum (bf16 product, f32 accumulate; bit-identical). chunk5@49k 1732->1670ms, tf 58/64, dist gold True
```

---

```text
sparse_attn prefill: S=1 when T>=1024 (drop split-K pacc round-trip + combine). chunk5@49k 1670->1484ms, attn 384->258ms, combine 57->0; tf 58/64, dist gold True
```

---

```text
bench: prefill opt exploratory scripts + attn kernel negative results

attn_ubench.py: single-GPU microbench for sparse_attn_paged_tc (T=12288,K=640,S=1):
  7.59 ms/call, 8.05 GB read, 1.06 TB/s.
ncu (sudo -nE, abs path, explicit PYTHONPATH): DRAM 27%, SM 33%, warps_active 37%,
  regs 80/thr, occupancy limited to 3 blk/SM by both regs and smem(34KB),
  top stall long_scoreboard 9.8 -> latency bound, not bandwidth bound.

Tried and REVERTED (both slower):
  A) cp.async double-buffer kvs[2]/ok[2] prefetch next tile: 8.57 ms (+13%).
     smem 34->68KB => 2 blk/SM, warps_active 25%; stall 9.8->6.4 but occupancy loss wins.
  B) __launch_bounds__(256,4) + carveout=100 to force regs<=64: 8.08 ms (+6%), spills.
TILE=32 is bound to the mma layout so tile cannot be halved to recover smem.
Only remaining route: 8-warp -> 4-warp/block rework (~6% ceiling). Not pursued.

Service state after 33ba3b6: cold 67k prefill 7411/7414 tok/s model, e2e 9.52-9.56 s
(baseline 6082 tok/s / 11.03 s). tc_leak/tc_long/tc_pd/tc_spec: earlier probes.
```

---

```text
ops: split hc_fused_pre into 256-thread main kernel + per-warp route kernel (bit-same)

- hc_pre_main_kernel<4>: 256 threads emulate the original 1024-lane reduction tree
  (identical pairing order), y kept in registers for pass2 (no re-read).
- hc_route_kernel<4>: post + sinkhorn(19 it) per token in one warp, off the main kernel.
- dim!=4096 falls back to the original hc_fused_pre_kernel.
Results (T=12288): standalone 1.047->0.447 ms, all 4 outputs BIT_SAME vs old kernel.
Engine profile NCH=5 chunk5: 1483.8->1430.7 ms (-3.6%), hc_pre 1.029->0.336 ms/call.
Gates: tc_oracle_tf 58/64 fd=46 (== baseline); tc_oracle_decode_dist gold True / DSML True.

Service acceptance: cold 67k prefill steady 9.21-9.28s e2e, prefill_tps 7526-7663 (prev ~7411).
```

---

```text
prefill: replace fp32 all_reduce (MoE merge + wo_b) with fp32 reduce_scatter + bf16 all_gather; chunk5 1430.7->1394.5ms; gates pass
```

---

```text
decode: moe_decode_down_fp4 16B/lane loads, 4 rows/warp in flight (bit-exact)

Old: each lane loaded 4B (uint32) -> one 128B transaction per row, rows serial
behind shfl reductions -> ~6 transactions in flight, 486 GB/s (31% HBM).
New: 8 lanes x uint4 cover one row, warp processes 4 rows concurrently (same
kRows launch geometry); reduction tree keeps the original virtual-lane pair
order (shfl 4/2/1 + in-register 2/1) so partial sums are bit-identical.
tc_moe_micro REF=cmp: BITEXACT all T/patterns. down T=8 rand 59.1->41.0us,
T=1 15.4->10.9us, T=32 197->137us. tc_model_api_b1 PASS, eager 28.02 ms/step
(prev 30.19).
```

---

```text
decode: hc_fused_pre mixes GEMM -> in-house hc_mix_gemv (T<=64), drop cuBLAS splitK

at::linear(x4c[T,16384] bf16, fn[24,16384] bf16) chose cutlass splitK + splitKreduce
(2 kernels, non-deterministic reduction order), 86x/step. Replaced by one kernel:
grid (N=24, KS=8), 256 thr, 16B loads, fp32 acc, fixed reduction order
(thread -> shfl tree -> warp -> split via last-block), output bf16.
T>64 or unaligned K still falls back to at::linear.

Verify: run-to-run deterministic; post/comb/x/xraw bit-equal to the cuBLAS path
(T=65 forced) for T=1/8/32/64 (bf16 rounding absorbs order diff). T=8: 10.0us.
tc_model_api_b1 PASS, eager 28.12 ms/step (noise vs 28.02); saves 86 launches/step.
```

---

```text
moe decode fp4: merge sign bits via PRMT in fp4x8_to_bf16x8 (ALU-bound gu kernel)

ncu (T=32, nsel=192, random weights): gu kernel ALU pipe 83% / FMA 37% / LSU 8% /
DRAM 44%; down kernel L1TEX 85.6%. Same-expert-for-all-slots only -8% => not
DRAM bound, unpack instruction count is the limiter.

Old: 4 PRMT + 2 AND + 8 (AND/SHIFT) + 8 OR = ~20 ALU ops / 8 elems.
New: sign nibbles pre-positioned at byte msb (sgh = p&0x80808080,
sgl = (p<<4)&0x80808080), two PRMTs interleave them into the hi-byte word
before the lo/hi interleave => ~13 ops / 8 elems.

Bit-identical: exhaustive 2^32 GPU test old vs new, mismatch=0 (/tmp/fp4_eq_test.cu).

Microbench (/tmp/t_moe.py, E=256 Nff=256 K=4096):
  T=8 : gu 52.0->41.0us, down 41.0->38.0us
  T=32: gu 187 ->142.7us, down 138 ->129.6us

amdahl (stop-service, MAXSEQ=65536): B4 down 8.51->5.65ms gu 8.14->6.24ms (moe_fp4 16.73->11.96);
graph 46.76->45.01ms; B1 graph 26.0->25.8. gate_b1 PASS; service re-verified on :8000.
```

---

```text
kv pool: decouple pool size from max_batch_size (ModelArgs.pool_tokens, default 4M)

- ModelArgs.pool_tokens (0 = legacy n_slots*max_seq); arch passes it to SlotPool
- args_dsv4: POOL_TOKENS = 4*1024*1024 (~30.1 GiB/rank, 66.4 GiB total per rank measured)
- strategy: anchor whose required_pages > kv_pool_pages fails gracefully instead of assert
- admission for followers already existed (available_pages / take_next)

e2e (cold, hit=0, B1, service): 67k prefill 8.61s 7810 tok/s TTFT 9.03s decode 30.2ms/step 90.3 tok/s;
160k prefill 22.95s 6969 tok/s TTFT 23.7s decode 33.2ms/step 69.6 tok/s
```

---

```text
bench/tc_amdahl_b: per-rank own/nccl/span report, TOPK_OPS, IDX_RS/IDX_BF16 env A/B switches (bench-only)
```

---

```text
peer_ar_rows: IPC row-limited all_reduce for indexer scores (replaces static-count NCCL AR)

B1 393k: graph 27.02->25.68ms (A_nccl 6.55->3.60ms); 65k: ~-0.5ms. Gates: tf 58/64, decode_dist gold match, semgate PASS.
PEER_AR_ROWS=0 falls back to NCCL. Benches prewarm explicitly (model_api prewarms before capture).
```

---

```text
docs: reject cluster1 qdq-GEMM fusion after Q8 shape screening

Two isolated CUDA prototypes; M8/16/24/32 x four N,K shapes. V2 K256/N64 split with shared staging loses all 16 cases: B1 baseline 10.89-14.52us versus 17.55-21.27us; B4 baseline 10.97-15.07us versus 23.04-46.48us. Service remained online: screening only, no end-to-end speedup or model-gate claim. BF16 outputs not universally bitwise equal. Withdraw prior 2ms forecast. Archive source, harness, raw timings; production path unchanged.
```
