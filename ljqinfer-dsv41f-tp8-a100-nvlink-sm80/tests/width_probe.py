"""At which prompt length does a row start caring how many ride beside it?

    torchrun --standalone --nproc-per-node=8 -m tests.width_probe

A 39-token prompt answered four abreast walked away from its solo self at
token 36; a 384-token prompt never did.  The sliding window holds 128
tokens, so the suspicion is that a window not yet full is read differently
when the batch is wide.  This walks the prompt length across that edge.
"""
import torch

from strategy.decode_worker import MAX_BATCH, bootstrap
from tests.batch_probe import _prefill

LENS = (39, 64, 100, 127, 130, 200, 384)
NEW = 96


def toks_of(n):
    return tuple((i * 7919 + 13) % 60000 + 1 for i in range(n))


def ride(eng, toks, n, width):
    past = eng.past
    slots = [past.alloc() for _ in range(width)]
    states, got = [], []
    for s in slots:
        st, out = _prefill(eng, s, toks)
        states.append(st); got.append(out)
    while len(got[0]) < n:
        states, ems = eng.spec.step_gb(states, past=past, slots=tuple(slots))
        for r, e in enumerate(ems):
            got[r].extend(int(t) for t in e)
    for s in slots:
        past.release(s)
    return [g[:n] for g in got]


def main():
    with torch.inference_mode():
        eng = bootstrap()
        for n in LENS:
            toks = toks_of(n)
            ref = ride(eng, toks, NEW, 1)[0]
            for width in (2, MAX_BATCH):
                rows = ride(eng, toks, NEW, width)
                diffs = []
                for g in rows:
                    diffs.append(next((j for j in range(NEW) if g[j] != ref[j]), -1))
                print('WIDTH prompt=%3d width=%d rows_agree=%s first_diff=%s'
                      % (n, width, len({tuple(g) for g in rows}) == 1, diffs),
                      flush=True)
        print('WIDTH done', flush=True)


if __name__ == '__main__':
    main()
