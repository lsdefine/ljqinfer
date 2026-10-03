# Decode kernel ledger (v4.1 FP8, TP8, A100x8)

Measured with `SPEC_PROF=200` on the live 384K server, 10 decode steps, all
eight ranks dumped as `PROFJSON <rank> ...` in `worker.log`.  Everything below
is kernel self-time, so the columns add up to the whole GPU-busy step; there is
no unexplained remainder this time.

## rank0, ms per step (total 23.25, 2534 launches/step)

| kernel | ms | calls |
| --- | --- | --- |
| `oneshot_ar` | 6.41 | 89 |
| `moe_down` | 1.87 | |
| `moe_gu` | 1.72 | |
| `cutlass_s16816gemm` (5 variants) | 1.35 + 0.67 + 0.52 + 0.48 + 0.47 + 0.34 | 180 |
| `ncclDevKernel_AllGather_RING_LL` | 0.94 | 25 |
| sparse attention | 0.84 + 0.30 | |
| `_hc_gemv` | 0.66 | 86 |
| `splitKreduce_kernel` | 0.56 | 235 |
| `_hc_gates` | 0.54 | |
| `fp4_gemv_dec3` | 0.46 | |
| top-k | 0.41 | |
| `rms_norm` | 0.40 | 102 |
| `_collapse_norm` | 0.32 | |
| rope | 0.29 | 150 |
| `memcpy32_post` | 0.29 | 162 |
| `_expand` | 0.21 | |

## The decisive cross-rank comparison

Every kernel except the all-reduce lands within 0.02 ms of itself on all eight
ranks: the tensor-parallel split is balanced, there is no straggler doing extra
work.  `oneshot_ar` is the only asymmetric one:

| rank | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `oneshot_ar` ms/step | 6.41 | 2.83 | 2.86 | 2.76 | 3.32 | 7.46 | 6.20 | 7.74 |
| total busy ms/step | 23.25 | 19.64 | 19.63 | 19.48 | 20.11 | 24.05 | 22.77 | 24.53 |

So the 6.41 ms is not reduction work.  The floor is ~31 us per call (the rank
that waits least: 2.76 ms / 89), which is the same order as NCCL's ~35 us on
this box - that is the eight-GPU small-message latency floor.  The rest,
~2.2 ms per step on average, is spin-wait on peers that have not arrived yet.

## Landed

* `5401b00` - rotate the peer read order inside `oneshot_ar`.  Every rank used
  to read rank0's buffer first, making it an eight-way NVLink hot spot.  Short
  request 10.31 s -> 10.10 s, 32K request 15.07 s -> 14.89 s for 900 tokens
  (~0.3 ms/step).

## Rejected by measurement (do not retry blind)

* **Fusing `wq_a` + `wkv` into one taller-N GEMM.**  Implemented with a
  concatenated BF16 cache and one `torch.mm` plus a split; short 10.10 -> 10.68 s,
  32K 14.89 -> 15.17 s.  Reason: the dense projections are bandwidth-bound on
  the weights, and concatenation does not reduce the bytes read - it only saves
  a launch, while the output split adds copies.  Reverted.
* One-shot IPC all-reduce as an *algorithm* (it already beats NCCL here).
* V4's `hc_mix_gemv`, a hand-written dense GEMV, and `V41_BF16_DENSE=0`
  (A100 has no FP8 tensor cores).

## Where the remaining headroom actually is

1. `oneshot_ar` 6.41 ms - of which ~2.8 ms is an unavoidable latency floor and
   ~2.2 ms is skew absorption.  Cutting the *count* (89/step: 40 layers x 2,
   plus draft) is structural: RMSNorm between the attention and MoE reductions
   is non-linear, so neither reduction can be deferred.  Lowering the skew means
   attacking whatever desynchronizes the ranks between reductions, not the
   kernel.
2. Dense GEMM ~3.8 ms + 0.56 ms of split-K reduction, against a ~1.56 ms
   bandwidth floor - the M=8 shape keeps cuBLAS at roughly half of peak.
3. MoE 3.59 ms is already near its bandwidth limit.
