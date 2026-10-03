# GLM53 resident dynamic batching

## Resident service update

This section supersedes the historical fixed-cohort lifecycle below.
The service now uses `model/glm53_resident.py` for B1-B4: commit-point
boarding, fixed logical graph rows, page-map compaction without KV copies,
epoch page leases, and resident Verify/JointDraft/Append graphs. B1-B4 at
2K are warmed before readiness; larger Verify context buckets are captured
lazily and cached. Prefill remains eager. No compute kernels were changed.

At most four rows are active. Finished leases are reclaimed at epoch end;
an epoch with insufficient pages drains before further admission. New-request
prefill can pause existing streams: this is not chunked-prefill scheduling.
Singleton requests also use the resident path, not the old prefix-cache path.

### API measurements (output tokens/s)

Same audit script and prompt, max_tokens=256, thinking off, two repetitions.
Old baseline: 22a46d2. These are historical comparisons, not interleaved A/B.
Decode uses the client first-delta-to-last-delta estimator, excluding TTFT.
E2E aggregate includes TTFT and queueing and is a separate metric.

| Concurrency | Old mean per-request decode | Resident mean decode | Old E2E aggregate | Resident E2E aggregate |
| --- | ---: | ---: | ---: | ---: |
| 1 | 137.68 | 109.40 | 48.03 | 93.14 |
| 2 | 36.58 | 87.05 | 60.70 | 132.93 |
| 4 | 19.75 | 54.11 | 65.87 | 157.45 |
| 8 | 18.19 | 47.40 | 62.37 | 159.95 |

Both eight-request runs admitted all requests in one epoch with zero new
Verify captures, with at most four active rows. Singleton decode regressed;
this is not a uniform speedup. Outputs/acceptance are not constrained equal
to the old path, so these are service measurements, not fixed-work timings.

### Verified scope and limits

- Final eight-rank model and queue regressions passed: graph/eager equality
  under identical schedules, repeat epochs, dynamic 2-to-4 boarding, shrinking
  batches, mixed limits, cancellation, initially cancelled requests with zero
  output, and post-cancel reuse.
- HTTP audit passed chat, streaming, multi-turn, eight reasoning questions,
  tool roundtrips, tool_choice=none, concurrency 1/2/4/8 and final health.
- RPC audit passed late boarding, active cancellation, a one-token peer,
  subsequent requests and 4K-to-cached-2K bucket switching.
- Production 144K allocation/startup passed. This does not establish full
  144K generation safety. Maximum-context and many-bucket stress, independent
  HF quality parity and the singleton decode regression remain open.
- Original prefix-cache state is invalidated per epoch.

Evidence: `/mnt/data2/kw/glm53_int4_tp8/service_audit/resident_boarding/`
including `final_integration.log`, `final_queue.log`, `results.json`,
`perf.json`, `rpc_audit.json` and service logs.

Reproduce with BATCH_AUDIT_DIR set to a writable directory:

```sh
torchrun --standalone --nproc_per_node=8 tests/test_glm53_resident.py
torchrun --standalone --nproc_per_node=8 tests/test_glm53_batch_service.py
```

## Historical fixed-cohort implementation

The material below preserves earlier compute and numerical evidence. Its
singleton, graph-lifetime, no-boarding and HTTP-status statements describe
the previous implementation, not the current resident service.


The strategy admits up to four requests over 20ms, subject to aggregate KV
capacity. Singleton cohorts retain the original B1 generator. Independent
request page tables, lengths, acceptance, EOS and cancellation remain intact.

## Joint computation

- Attention Q/KV/index projections, output projection and output TP reduction
  operate on B*8 rows. Full index layers pack all requests into two TP gathers.
  Sparse score/top-k and MLA KV reads remain request-local.
- Router, shared-expert and output-head GEMMs no longer split back to Q8.
- DFlash embedding, projections, SDPA, FFN, output head and TP reductions execute
  jointly. Grouped convolution explicitly stops at each request's Q8 boundary.
- Draft tokens transfer to CPU once per batch, rather than once per request.
- Routed experts retain the existing per-route GEMV implementation; there is no
  expert-grouped GEMM in this change.

## Verified results (node09, A100 TP8)

| Batch | Complete model round | Strategy round |
| --- | ---: | ---: |
| B1 | 43.73ms | not separately measured |
| B2 | 62.27ms | 61.66ms |
| B3 | 78.30ms | 77.00ms |
| B4 | 96.00ms | 93.75ms |

Every sample is the maximum across eight ranks, full-active-batch only. Model
samples exclude the first six rounds (30/30/32/34 samples); strategy samples
exclude the first round (11/9/9). Timings include draft, verify, commit and token
callbacks, but exclude prefill, capture, shape-transition preparation, queueing
and cancellation polling. They are not end-to-end latency or output-token tps.
B4/B1 model round ratio is 2.195, not four. The previous revision's model B4 was
146.09ms and strategy B4 142.82ms; these are historical runs, not an interleaved A/B.

Controlled same-input CUDA-graph microbench, rank 0 (not full-round timing):
B4 single attention layer 1.218 -> 0.565ms; draft 12.939 -> 3.667ms;
full verify 125.070 -> 85.223ms. B1 verify in that probe regressed
36.693 -> 38.923ms; singleton service still uses the original generator.

## Correctness evidence and numerical limits

Eight model cases passed on all eight ranks: B1/B2/B3/B4, mixed exits, row
permutation, cancellation and EOS at anchor. B1-4 were independently repeated
with identical same-batch token outputs. B1 after teardown matched its baseline.
Strategy B2/3/4 outputs matched direct generation with identical cohorts and
limits; cancellation passed and client errors were empty. No HTTP test was run.

Dedicated TP tests at capacities >2048 checked packed sparse-index IDs against
the original per-request implementation, including mutation of one request:
other requests remained unchanged. Batched DFlash convolution matched the old
per-request kernel exactly. All checks passed on eight ranks for B2/3/4.

B1 same-input logits matched exactly in the controlled microbench. Larger
GEMM shapes are NOT bit-identical to B1: full greedy sequences diverge. A real
prompt first-block B4 ablation had 100% top1 agreement and mean KL 0.00365;
subsequent same-input B4 shadow blocks had top1 agreement 96.875-100% and mean
KL 0.000849-0.018644. Synthetic-prefix tests had lower top1 agreement
(81.25-90.625%, KL 0.0108-0.0164). These measurements are not a broad model-quality
certification. B1-sequence equality is reported diagnostically, not asserted.

Evidence directory:
`/mnt/data2/kw/glm53_int4_tp8/service_audit/joint_batch_6f477f3/`
Includes `summary.json`, `installed/`, `bench.rank*.json`, `ablate.rank*.json`,
`integration/` (same-input shadow checks), and `boundaries.rank*.json`.

## Remaining boundaries

- Fixed cohorts: no mid-cohort admission; not continuous batching.
- Prefill, sparse KV kernels, per-request commit/append remain request-local.
- Tested engine capacity is 8192 tokens. Production 144K startup and full-model
  long-context capacity stress are untested; long-index synthetic tests do not
  establish those properties.
- Graphs/scratch are per-cohort, not reused across cohorts. Original per-slot
  draft capture still prepares append graphs; its startup overhead is not part
  of the steady-state measurements.
- Batch entry invalidates B1 prefix cache. HTTP was not started by this change.

## Reproduce installed functional tests

Set `BATCH_AUDIT_DIR` to a writable directory, then from repository root:

```sh
torchrun --standalone --nproc_per_node=8 tests/test_glm53_batch_boundaries.py
torchrun --standalone --nproc_per_node=8 tests/test_glm53_batch.py
torchrun --standalone --nproc_per_node=8 tests/test_glm53_batch_service.py
```
