# Decode glue findings (32K, TP8, rank2 unless noted)

Measured with `/tmp/serve_384k` restarts and a fixed temperature-0 pair of
requests (900-token short, 900-token 32K).  Numbers are `decode_ms_per_step`
from the worker log, with the accepted-token count kept as a correctness
guard: any change that moves the step count changed the output.

## Verified baselines

| build | short | 32K |
| --- | --- | --- |
| 219007d (before this round) | 433 steps, 21.54 ms | 429 steps, 24.58 ms |
| head weight built once | 430 steps, 21.19 ms | 429 steps, 24.14 ms |
| + rotation rows selected once | 427 steps, 20.93 ms | 429 steps, 23.85 ms |

## What the captured graph actually runs

A `with_stack` profile of the eager warm-up pass (`SPEC_WARMPROF=1`) shows the
per-step ATen traffic behind the graph: 310 `mm`, 173 `copy_`, 104
`remainder`, 98 `add`, 59 `index_select`, 44 `bmm`.  Most of these are per
layer repeats of the same index arithmetic, and they survive into the graph as
hundreds of small kernels: 219 `splitKreduce` (from `mm`/`bmm`), 141+122
elementwise kernels, 110 device-to-device copies.

## Rule learned: a graph replay never runs Python

Caching a tensor in a Python dict that lives across passes is wrong even when
the key looks stable.  `scratch.pos_t` is a graph-stable buffer: its identity
never changes while its contents change every step, so a dict keyed on it
serves the rows selected during the first capture forever.  The draft and
verify graphs share one scratch, so they also served each other's rows.  Both
mistakes cost roughly 10% of the accepted tokens (433 -> 474..516 steps) while
looking like a speedup on the step timer.

The fix that works: clear the per-pass cache at the top of `forward`.  Each
capture then records exactly one `index_select`, the remaining layers read the
tensor it produced, and replay re-selects from the live positions.

## Rejected by measurement

* Dropping the `float()` before the compressor projections: the weights are
  FP8, so the activation dtype changes the quantisation and the output.
* Caching rotation rows keyed on the table alone, or on the table and the
  position buffer, without clearing per pass (see above).

## Still open

* `AllGather_RING_LL`, 0.69 ms over 25 launches per step, separate from the
  one-shot all-reduce.
* 219 `splitKreduce` launches: the decode GEMMs are M=8 and cuBLAS still picks
  split-K, so every projection pays a second kernel.
* 110 device-to-device copies and ~340 elementwise launches per step, mostly
  index arithmetic that could be hoisted out of the layer loop the same way
  the rotation rows were.
