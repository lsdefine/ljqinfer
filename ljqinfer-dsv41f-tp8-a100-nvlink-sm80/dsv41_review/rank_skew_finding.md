# Decode step: where the time goes (corrected 2026-09-16)

## 1. The "rank 0 skew" was a profiling artefact -- retracted
An earlier version of this note claimed rank 0 spent 11.5 ms/step inside
`oneshot_ar` against 2.3 ms on rank 3/5/6, i.e. ~9 ms/step of arrival skew.
That number came from dividing a *cumulative* AR total by the step count.

Aligning the eight per-rank traces on absolute timestamps disproves it.
For any given all-reduce the eight ranks enter within 1-5 us of each other
and run for the same 17-21 us:

| AR idx | r0 start/dur | r3 start/dur | r7 start/dur |
|--------|--------------|--------------|--------------|
| 881    | 49629.4/17.8 | 49628.3/18.7 | 49628.2/18.8 |
| 882    | 49794.3/19.7 | 49793.0/20.3 | 49792.2/21.1 |
| 883    | 49986.7/17.8 | 49987.0/17.4 | 49986.4/17.9 |

Across 890 calls the median is 21-22 us on *every* rank.  Rank 0's 132 ms
total is 108 ms of AR index 0 alone -- the one-off cost of the first
collective after profiler start, when the other ranks have not yet been
scheduled in.  Drop that single sample and rank 0 totals 23.9 ms, matching
ranks 2-7 (22.7-25.1 ms).  There is no per-step skew to remove.

## 2. Clean decode kernel ledger (trace_r3, steady state)
9.9 decode steps, 22.79 ms/step wall, GPU busy 90.3%, idle 2.22 ms/step.

| kernel | ms/step | calls/step | us/call |
|--------|---------|-----------|---------|
| oneshot_ar | 2.402 | 89 | 27.0 |
| moe_decode_down_fp4 | 1.871 | 40 | 46.7 |
| moe_decode_gu_fp4 | 1.736 | 40 | 43.3 |
| cutlass 64x64 bf16 gemm | 1.335 | 180 | 7.4 |
| ampere 64x64 gemm (3 variants) | 1.652 | 134 | ~12 |
| _route_gemv | 0.906 | 43 | 21.0 |
| sparse_attn_decode | 0.749 | 38 | 19.7 |
| _hc_gemv | 0.663 | 86 | 7.7 |
| ncclDevKernel_AllGather_RING_LL | 0.663 | 24.5 | 27.1 |
| direct_copy | 0.627 | 55 | 11.4 |
| splitKreduce | 0.556 | 235 | 2.4 |
| _hc_gates | 0.533 | 86 | 6.2 |
| rms_norm / rope / memcpy32 / topk / sort | ~2.0 | 500+ | 2-9 |

Two facts follow.  AR is 11.7% of the step, not the 40% the artefact
suggested, and at 27 us for 61 KB over NVLink it is close to done.  The
step issues 1600+ kernels; roughly 700 of them are 2-3 us, together 5-6 ms.

## 3. What helped: CPU affinity (committed)
128 cores / 2 NUMA nodes; `nvidia-smi topo -m` maps GPU0-3 to CPUs 0-63 and
GPU4-7 to CPUs 64-127.  All eight workers had ran affinity 0-127, so launch
threads migrated across NUMA nodes.  `strategy/decode_worker.py` now pins
rank r to a private 16-core slice of its own GPU's NUMA node.

Measured (t32.py, 900 generated tokens, repeated):
- short prompt: 9.48 s -> 9.01-9.31 s; step 21.0 -> 20.60 ms
- 32K prompt:  13.50 s -> 13.23-13.39 s; step 23.96 -> 23.53 ms
Four consecutive A/B runs favoured the pinned build.  The mechanism is less
launch-thread jitter, not the removal of skew (see section 1).

## 4. Disproved (do not retry)
- Decode-private thin GEMM.  Decode dense projections go through
  `PrefillLinear` -> `F.linear`, and cuBLAS answers M=6 with a 64x64 tile
  plus split-K (hence 235 `splitKreduce` launches).  A single-pass triton
  kernel that streams the weight once with no split-K measured 21.02/24.49
  ms against a 20.57/23.56 baseline.  Split-K is not waste here: without
  splitting K only N/64 blocks exist and the SMs starve, so the extra
  partial reduction buys back more bandwidth than it costs.
- `FastAllReduce.BLOCKS`: 16 (current) beats 32 (9.47/13.46) and
  48 (9.59/13.58).  The in-code comment "measured optimum" is correct.
- Earlier rounds: IPC all-reduce variants, hc_mix_gemv, fusing wq_a+wkv,
  FP8 dense GEMV, dropping the compressor `x.float()`, route_gate retuning
  and its tl.dot rewrite (110 us vs 38.8 us), direct_copy (already
  0.016 ms/step).

## 5. Remaining headroom
The large kernels sit near their bandwidth ceilings (dense 1.45 TB/s, MoE
gu 1.63 TB/s on a ~1.55-1.9 TB/s part) and together account for ~9 ms,
close to the 8.4 ms floor computed earlier.  With only 2.22 ms of idle per
step there is no waiting left to reclaim, so the remaining ~5-6 ms lives in
the count of small kernels, not in any single operator.  That is an
orchestration change -- fold rms_norm, rope and the cache write into the
attention prologue, fold the routing prologue into the MoE kernels -- worth
about ten launches per layer, 400 per step.
