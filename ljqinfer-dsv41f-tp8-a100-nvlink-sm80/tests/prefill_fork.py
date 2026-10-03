"""Does one prefill, repeated on one unchanging row, land in one place?

    torchrun --standalone --nproc-per-node=8 -m tests.prefill_fork

fork_hunt puts the first failure to repeat in the hidden state prefill hands
back, before a single decode step has run, and the failures come in a few
exact shapes rather than a smear -- the same count of moved entries, the same
worst gap, again and again.  That is the shape of a few fixed numeric paths,
not of noise.  So run prefill alone, many times, and sort the results: how
many distinct answers, how often each, and where in the tensor they differ.
A tail-heavy difference means the last, partly filled tile is reading what the
previous run left behind.
"""
import sys
from collections import Counter

import torch
import torch.distributed as dist

sys.path.insert(0, '/mnt/data/kw/ljqinfer_dsv41f_tp8')

from strategy.decode_worker import bootstrap        # noqa: E402
from tests.batch_probe import _prefill              # noqa: E402
from tests.tie_probe import served_row              # noqa: E402
from tests.ulp_probe import _ulps                   # noqa: E402

REPEATS = 24


def main():
    toks, _ = served_row()
    with torch.inference_mode():
        eng = bootstrap()
        say = dist.get_rank() == 0
        order, keep, tally = [], {}, Counter()
        for _ in range(REPEATS):
            past = eng.past
            slot = past.alloc()
            state, got = _prefill(eng, slot, toks)
            h = state.hidden.detach().to('cpu', copy=True)
            past.release(slot)
            key = hash(h.contiguous().view(torch.uint8).numpy().tobytes())
            if key not in keep:
                keep[key] = h
                order.append(key)
            tally[key] += 1
        if not say:
            return
        first = keep[order[0]]
        print('PF hidden shape=%s dtype=%s' % (tuple(first.shape), first.dtype),
              flush=True)
        print('PF distinct=%d of %d  counts=%s'
              % (len(order), REPEATS, [tally[k] for k in order]), flush=True)
        for i, key in enumerate(order[1:], 1):
            h = keep[key]
            bad = (first != h)
            d = _ulps(first, h)
            print('PF variant=%d n=%d/%d max=%.1f ulp worst_abs=%.3e'
                  % (i, int(bad.sum()), bad.numel(), float(d.max()),
                     float((first.float() - h.float()).abs().max())), flush=True)
            if bad.dim() == 2:
                per_row = bad.sum(1)
                print('PF variant=%d rows_touched=%s of %d  per_row=%s'
                      % (i, int((per_row > 0).sum()), bad.shape[0],
                         per_row.tolist()), flush=True)
            else:
                nz = bad.nonzero().flatten()
                print('PF variant=%d idx_first=%s idx_last=%s'
                      % (i, nz[:8].tolist(), nz[-8:].tolist()), flush=True)


if __name__ == '__main__':
    main()
