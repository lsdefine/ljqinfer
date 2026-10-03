# ljqinfer_dsv41f_eptp8

Fresh repository for DeepSeek-V4.1-Flash on node09. No inherited V4 Git history, model runtime, weights, or generated binaries. HTTP code is migrated from the original server directory.

Target: routed experts EP8; attention/shared-expert projections TP8 on the same eight ranks. Small router/norm parameters may be replicated; this is not EP8 x TP8 = 64 devices. A synchronous EP8/TP8 prefill baseline is implemented.

Implemented: CPU/GPU Past, cold serialization, separate CED prefill/finish/replay, real Engram hashing and a canonical BF16/FP8/FP4 computation baseline. Packed GEMMs decode bounded weight tiles; this is not an optimized fused kernel runtime. GPU graphs and asynchronous offload are not implemented. Past reference tensors use unpacked storage.

Run: `python -m pytest -q`.

## Execution boundary

- `model/model_api.py`: `ModelExecution` reserves pages, calls injected
  `forward(tokens, *, past, slot, start, history_tokens)`, then commits position. Compute writes
  model state only: no slot allocation, position advancement or cold storage.
- `strategy/strategy.py`: `Strategy.open` restores a matching prefix and replays its hot state. Session
  `step` computes at most 12,288 tokens and cold-stores the completed chunk;
  `run` finishes input; `extend` appends input. Use sessions as context managers
  to release pages/leases. Failures discard dirty slots, not prior checkpoints.
- Sessions may be interleaved on one serialized execution lane/CUDA stream.
  No worker queue, continuous batching or decode loop yet.
- Full hits restore/replay Past with no new forward output. `extend(next_chunk)`
  then `run()` continues prefill without cached logits.

```python
from model.model_api import ModelExecution
from strategy.strategy import Strategy
execution = ModelExecution(compute, past, prefill_chunk_tokens=12288)
strategy = Strategy(execution, cold_kv, namespace=model_namespace)
with strategy.open(input_ids) as session:
    output = session.run()
```

Compute-only test mocks use these production entrypoints, not bench loops.
Full-capacity regression:
`V41_FULL_GPU=1 python -m pytest tests/test_engine.py::test_full_capacity_engine -q`.

## Cold-cache state reference

- `model/paging.py`: independent page ownership and row mapping.
- `model/past.py`: four shared sources, layer rings, compressor carry and verify-prefix repair.
- `strategy/cold_kv.py`: token-exact trie, variable-length field-major segments,
  byte budgeting, leased readers, leaf-LRU eviction and complete-field publication.
- `model/cold.py`: arbitrary-token `store_prefix` and empty-slot `restore_prefix`.

## Global-only cold cache / bounded replay

Cold persists only the four source KV/index histories and token prefix metadata,
not SWA rings or compressor carry. Any matching token prefix can be restored,
including an interior point never saved as an endpoint; leases clip backing
segments and pin their dependencies. Complete compression rows for [a,b) are
floor(b/r)-floor(a/r); physical page alignment does not restrict token hits.

Cold import marks Past as replay-pending. `Strategy.open` calls the model's
`replay(tokens, *, past, slot, start, history_tokens)` on the last up to 128
matched tokens before allowing more prefill. Global KV/index rows are read-only;
compute reconstructs SWA and incomplete compressor carry. Up to three preceding
token IDs are supplied for Engram initialization. Causal masking must use each
replay query's position, not expose all loaded history to earlier queries.
This follows report section 3.2.2's approximate bounded replay, not exact
full-forward equivalence. No replay is needed between live prefill chunks.

`PrefillModel.forward` evaluates encoder blocks 0..19 and layer 20 global
projections, retaining the last 128 encoder outputs in hot Past only. It does
not run decoder MoE or produce logits for every input chunk.
`session.run(); output = session.finish_prefill()` explicitly prepares generation:
blocks 20..39 (attention AND MoE) consume those encoder outputs with bounded SWA,
returning last-token logits and DSpark target hiddens. Encoder/global KV and
committed position remain unchanged. Pure cache-prefill need not call finish.
Cold `replay` remains a separate 40-block bounded reconstruction, not an alias
of forward. Its encoder tail can also feed finish. Paper section 3.2.2 explicitly
allows approximate replay state, not full-history numerical equivalence.
Generation/DSpark decode workers are not integrated.

`model.prefill_build.build_prefill` binds unsharded or canonical rank-local
weights to the text backbone. Callers own the process group, host tables,
weight lifetime and token history; `PrefillParallel` requires eight ranks.
Engram hashing follows the official tokenizer normalization and n-gram scheme.

Tests retain released dimensions with deterministic lazy random weight/table
values. BF16/FP8/FP4 single-rank and EP8/TP8 paths exercise prefill, decoder
finish, cold restore, bounded replay and continuation. The FP32 diagnostic
fixture is retained separately. Distributed smoke/diagnostic command:
`PYTHONPATH=. OMP_NUM_THREADS=1 torchrun --standalone --nproc-per-node=8 tests/run_parallel_prefill.py --length 129 --output /tmp/v41-new-run`.
Use a new output directory; `--trace` records intermediate differences.

Independent attention checks share surrounding projections/blocks, not a fully
independent model oracle. Free-propagation reference parity still has an ordinary
failing test (no xfail or relaxed threshold). Single-vs-eight-rank cosine/top1
and finite outputs are diagnostics, NOT numerical acceptance. Removing SWA FP8
rounding in diagnostic tests is likewise not acceptance. Trained-checkpoint
accuracy, full-model long-context performance and DSpark generation remain
unvalidated; detailed run results are recorded in the implementation commit.

Cache namespaces must identify weights, layout, precision and rank ownership;
callers are responsible for the namespace and actual executed token history.
The tensor budget includes pending plus published tensor payload, not Python
trie metadata or allocator overhead. Variable tensors are grouped by field;
this is not yet a large preallocated slab allocator. Abort does not undo earlier
residency eviction. CPU copies complete before publication.

Engine cold writes use `store_chunk`: append 1..12,288 new tokens from a live
parent lease; close the old lease after receiving the new one. Consecutive
chunks do not rescan the full prefix.
Timings distinguish end-to-end cache operations from preallocated pinned DMA;
results and validation counts belong in commit messages, not this document.

Chunk ring writes retain only the final window; intra-chunk attention must use
forward-local KV before committing the ring. This tests state mechanics, NOT
model logits, packed index semantics, attention workspaces or model throughput.
Large preallocated cold slabs and async offload remain unimplemented; the
full-capacity multi-rank cache workload remains unvalidated.

This is an attention-state reference, not a complete resumable model session:
Engram history is explicitly supplied, but DSpark hidden/continuation sidecars
are not integrated with a generation worker. Production pending-state kernels must replace the
reference ring snapshot/restore path; no decode latency claim is made.

## Grouped prefill MoE

Canonical EP8 prefill has one routed-MoE implementation: SM80 CUTLASS
BF16 grouped GEMM with FP32 accumulation. No backend selector or fallback;
the slow comparison oracle lives only in `tests/routed_reference.py`.
FP4 E2M1/E8M0 banks expand for at most eight local experts at a time.
V4.1 FP8/K32 activation rounding and pre-w2 routing weights are preserved.
CUDA toolkit is needed for the first JIT build; CUTLASS headers/license are vendored.

`tests/test_grouped_moe.py` checks packed decoding, ragged execution and
released-dimension projections against FP64 independently of the old backend.
Free-propagation parity is reported separately: the old backend is not an
exact oracle, and even FP64 accumulation changes the rounding trajectory.
This validates arithmetic, not trained-model quality. FP32-output experiments
are not another production path; the BF16 rounding contract remains unchanged.

## Weight layout and restart cache

`model/weights.py` defines byte-preserving canonical EP8/TP8 layouts:
- Routed experts: 48/rank for backbone, 16/rank for DSpark. FP4 banks are
  `w13[E,2,I,K/2]` (gate, up), `w2[E,K,I/2]`, with matching E8M0 scales.
- Explicit TP axes for attention/shared projections; norms, routers and small
  shared projections replicate. Native image serving uses symmetric TP8 vision
  via `model/vision_runtime.py`; aligned embeddings replicate across ranks.
- Engram tables remain host-only: FP8 `[rows,256]`, E8M0 bytes `[rows,8]`.
  `HostEngram.open` maps complete files; `gather(ids, rank=...)` selects three
  of 24 hash rows/rank, paired with input-sharded Engram projections.

`WeightCache(root).get_or_build(key, builder)` stores complete bounded units
AFTER preparation. `identity(source=immutable_revision, unit=name, rank=rank)`
keys the cache; a hit maps CPU tensors without calling the builder. Choose a
shm or disk root explicitly. H2D remains necessary. No old cache is deleted.
The builder can call `prepare_rank(load_complete_unit(), rank)`; it must supply
complete weight/scale pairs and local expert banks, not incomplete downloads.
Future kernel prepacking belongs inside the builder with a new `pack_abi`.
Locking and atomic publication protect builds; corrupt files fail explicitly.

These are canonical layouts, not architecture-specific GEMM swizzles. Whole-
checkpoint streaming load and Engram async prefetch remain unimplemented.
Device/parallel compute binding is provided by `build_prefill`; host gathering
is a CPU reference, not an optimized transfer path. No whole-engine startup claim.

## HTTP server

Migrated from the original `server/` layout: HTTP, service, OpenAI adapter and
localhost engine RPC remain separate. Only model-specific tokenization, EOS,
DSML and reasoning-effort handling change to V4.1. The official encoder is
vendored from revision `2bc89ac599031fa673cab993f1df02fc4a98c673`.

The deployed backend is `strategy.decode_worker` under TP8 torchrun, serving
localhost RPC 62001; `python -m server.server` exposes HTTP 8000. Preserve the
existing model/rank/environment configuration when restarting; do not run a
second worker on the occupied GPUs. Replace development key `devkey` before
exposing the service. Tokenizer path is in `server/service.py`.

Native image input is available through the OpenAI chat endpoint, alongside
plain text, streaming and cancellation. See [IMAGE_SUPPORT.md](IMAGE_SUPPORT.md)
for an executable base64 image example, limits, validation evidence and explicit
unverified boundaries. `/health` is readiness only; actual text and image SSE
requests were verified on the restored TP8 deployment.
