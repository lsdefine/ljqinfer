"""Debug harness for the decode path the worker serves.

    torchrun --standalone --nproc-per-node=8 -m tests.batch_probe

Two questions, asked of the real engine rather than of a bench rig: does a
batched round still produce what width one produces (including across a row
leaving mid-flight, which reindexes every survivor inside the graph), and
what does a round cost at each width.  Debug tooling, so the knobs live here
as constants -- edit the file, not the command line.
"""
import time

import torch

from strategy.decode_worker import MAX_BATCH, bootstrap

# Synthetic prompt with a short period, so the n-gram drafter really proposes
# and rounds land on accepted > 1 instead of degenerating to one token a round.
PROMPT = tuple((i * 13 + 7) % 60000 + 1 for i in range(384))
LONG = 64          # tokens per row in the agreement check
ROUNDS = 32        # timed rounds per width
RESERVE = 2048     # headroom booked on a slot before it decodes


def _prefill(eng, slot, toks):
    """Bring one slot to the point where a decode round can take it."""
    eng.past.ensure(slot, len(toks) + RESERVE + LONG)
    eng.engine.prefill_chunk(slot, toks, history_tokens=())
    out = eng.engine.finish_prefill(slot)
    first = out.logits[-1].argmax().reshape(1)
    state = eng.spec.open(out.main_hidden[-1:], first, past=eng.past,
                          slot=slot, history=toks)
    return state, [int(first)]


def agreement(eng, toks):
    """Width `MAX_BATCH` over one prompt must equal width 1 over it.

    The prompt repeats so the n-gram drafter actually proposes and rounds
    land on accepted>1; the last row gets off halfway, so the survivors are
    judged across a reindex as well.
    """
    past = eng.past
    single = past.alloc()
    state, ref = _prefill(eng, single, toks)
    while len(ref) < LONG:
        # Width 1 is the same captured path as width MAX_BATCH, one row wide.
        ss, ems = eng.spec.step_gb([state], past=past, slots=(single,))
        state, emitted = ss[0], ems[0]
        ref.extend(int(t) for t in emitted)
    past.release(single)

    slots = [past.alloc() for _ in range(MAX_BATCH)]
    states, got = [], []
    for s in slots:
        st, first = _prefill(eng, s, toks)
        states.append(st)
        got.append(first)
    live, dropped = list(range(MAX_BATCH)), -1
    while max(len(got[r]) for r in live) < LONG:
        ss, ems = eng.spec.step_gb([states[r] for r in live], past=past,
                                   slots=tuple(slots[r] for r in live))
        for i, r in enumerate(live):
            states[r] = ss[i]
            got[r].extend(int(t) for t in ems[i])
        if len(live) == MAX_BATCH and min(len(got[r]) for r in live) >= LONG // 2:
            dropped = live.pop()
            past.release(slots[dropped])
    for r in live:
        past.release(slots[r])
    if eng.rank:
        return
    for r in range(MAX_BATCH):
        n = min(len(got[r]), len(ref))
        print('BATCH_PROBE agree r=%d dropped=%s len=%d matches_single=%s '
              'first_diff=%d'
              % (r, r == dropped, len(got[r]), got[r][:n] == ref[:n],
                 next((i for i in range(n) if got[r][i] != ref[i]), -1)),
              flush=True)


def timing(eng, toks):
    """Round time at every width, plus what the allocator handed out.

    A served round is arithmetic over space taken at startup, so a non-zero
    allocation delta inside the timed window is a round still shopping.
    """
    past = eng.past
    for width in range(1, MAX_BATCH + 1):
        slots = [past.alloc() for _ in range(width)]
        states = [_prefill(eng, s, toks)[0] for s in slots]
        for _ in range(4):                       # warm: no capture inside the window
            states, _e = eng.spec.step_gb(states, past=past, slots=tuple(slots))
        torch.cuda.synchronize()
        before = torch.cuda.memory_stats()
        start, ntok = time.perf_counter(), 0
        for _ in range(ROUNDS):
            states, ems = eng.spec.step_gb(states, past=past, slots=tuple(slots))
            ntok += sum(len(e) for e in ems)
        torch.cuda.synchronize()
        dt = time.perf_counter() - start
        after = torch.cuda.memory_stats()
        for s in slots:
            past.release(s)
        if eng.rank:
            continue
        print('BATCH_PROBE time b=%d round_ms=%.2f tokens=%d tok_per_s=%.1f '
              'ms_per_row=%.2f'
              % (width, 1000 * dt / ROUNDS, ntok, ntok / dt,
                 1000 * dt / ROUNDS / width), flush=True)
        print('BATCH_PROBE alloc b=%d allocs=%d segments=%d '
              'reserved_delta_MiB=%.1f rounds=%d'
              % (width,
                 after['allocation.all.allocated'] - before['allocation.all.allocated'],
                 after['segment.all.allocated'] - before['segment.all.allocated'],
                 (after['reserved_bytes.all.current']
                  - before['reserved_bytes.all.current']) / 1048576.0,
                 ROUNDS), flush=True)


def main():
    with torch.inference_mode():
        eng = bootstrap()
        agreement(eng, PROMPT)
        timing(eng, PROMPT)


if __name__ == '__main__':
    main()
