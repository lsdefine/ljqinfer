"""Does the same question, asked twice of the same engine, answer twice alike?

    torchrun --standalone --nproc-per-node=8 -m tests.determinism_probe

The served engine was caught answering one unchanged prompt two different
ways on consecutive, strictly serial requests.  A slot handed back should
carry nothing forward, so this asks the engine directly: run the same row
eight times, borrowing and returning a slot each time, at a served prompt
length and at a long one, then once more four abreast.
"""
import torch

from strategy.decode_worker import MAX_BATCH, bootstrap
from tests.batch_probe import PROMPT, _prefill

LONG = 128         # tokens collected per run
REPEATS = 8
SHORT = tuple((i * 7919 + 13) % 60000 + 1 for i in range(39))


def solo(eng, toks, n):
    past = eng.past
    slot = past.alloc()
    state, got = _prefill(eng, slot, toks)
    while len(got) < n:
        ss, ems = eng.spec.step_gb([state], past=past, slots=(slot,))
        state = ss[0]
        got.extend(int(t) for t in ems[0])
    past.release(slot)
    return got[:n]


def abreast(eng, toks, n, width):
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


def report(tag, runs, ref):
    for i, got in enumerate(runs):
        bad = next((j for j in range(min(len(got), len(ref))) if got[j] != ref[j]), -1)
        print('DETERMINISM %-14s run=%d len=%3d matches_first=%s first_diff=%d'
              % (tag, i, len(got), got == ref, bad), flush=True)


def main():
    with torch.inference_mode():
        eng = bootstrap()
        for tag, toks in (('short39', SHORT), ('long384', PROMPT)):
            runs = [solo(eng, toks, LONG) for _ in range(REPEATS)]
            report(tag, runs, runs[0])
            print('DETERMINISM %-14s distinct=%d'
                  % (tag, len({tuple(r) for r in runs})), flush=True)
        ref = solo(eng, SHORT, LONG)
        rows = abreast(eng, SHORT, LONG, MAX_BATCH)
        report('short39_b4', rows, ref)
        # The served prompt is real text, where the margin between the top
        # two candidates is wide; a random token string sits near a tie and
        # flips on noise, so only this comparison decides anything.
        pool = {tuple(solo(eng, PROMPT, LONG)) for _ in range(4)}
        rows = abreast(eng, PROMPT, LONG, MAX_BATCH)
        for i, row in enumerate(rows):
            print('DETERMINISM real_b4 row=%d in_solo_set=%s' %
                  (i, tuple(row) in pool), flush=True)
        print('DETERMINISM real solo distinct=%d' % len(pool), flush=True)


if __name__ == '__main__':
    main()
