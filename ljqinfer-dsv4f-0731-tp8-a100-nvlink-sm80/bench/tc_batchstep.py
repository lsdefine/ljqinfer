# -*- coding: utf-8 -*-
"""B=2 batched-step gate (torchrun 8 ranks).
A: each row decoded alone through the B1 graph (_rpc step).
B: both rows decoded together through the batched graph (_rpc step_batch).
PASS if per-row token sequences are identical.
torchrun --nproc_per_node=8 tc_batchstep.py
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # repo root on sys.path (this file lives in bench/)
import json, os, sys, time
import torch
from model.model_api import load_execution, _prefill_chunk_size

N_GEN = int(os.environ.get("N_GEN", "24"))
RANK = int(os.environ.get("RANK", "0"))


def log(*a):
    if RANK == 0:
        print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def prep(ex, row, ids, temp=0.0, fresh=True):
    """prefill + spec_init; returns (slot, pos, first_token)."""
    if fresh:
        ex.set_input(ids, row=row)
    slot = ex._slot(row)
    pos = ex.pool.pos[slot]
    logits = None
    while pos < len(ids):
        n = _prefill_chunk_size(pos, len(ids), ex.prefill_chunk)
        logits = ex._forward(slot, pos, tuple(ids[pos:pos + n]))
        pos += n
    tok = ex._pick(logits, temp)
    ex._rpc("spec_init", slot, tok, pos, temp)
    return slot, pos, tok


def solo(ex, row, ids, nsteps):
    slot, pos, tok = prep(ex, row, ids)
    out = [tok]
    for _ in range(nsteps):
        new = ex._rpc("step", slot, pos)
        out.extend(new)
        pos += len(new)
    ex._rpc("rewind", slot, pos)
    return out


def batched(ex, rows_ids, nsteps):
    # capture MUST happen on clean slots: graph warmup writes garbage KV around
    # pos0, which would corrupt an already-prefilled row (observed: diff@48).
    slots = []
    for row, ids in enumerate(rows_ids):
        ex.set_input(ids, row=row)          # release + alloc a fresh slot
        slots.append(ex._slot(row))
    slots = tuple(slots)
    t0 = time.perf_counter()
    ex._rpc("prepare_batch", slots)         # capture on clean slots
    log(f"prepare_batch{slots} captured in {time.perf_counter()-t0:.2f}s")
    poss, outs = [], []
    for row, ids in enumerate(rows_ids):
        s, p, t = prep(ex, row, ids, fresh=False)
        assert s == slots[row], (s, slots)
        poss.append(p); outs.append([t])
    ex._rpc("prepare_batch", slots)         # graph exists -> bind only
    step_ms = []
    for _ in range(nsteps):
        ts = time.perf_counter()
        new = ex._rpc("step_batch", slots, tuple(poss))
        step_ms.append((time.perf_counter() - ts) * 1e3)
        for b in range(len(slots)):
            outs[b].extend(new[b])
            poss[b] += len(new[b])
    ex._rpc("unbind_batch", slots)
    for s, p in zip(slots, poss):
        ex._rpc("rewind", s, p)
    st = sorted(step_ms[1:])
    if st:
        log(f"B{len(slots)} steps={len(st)} ms/step mean={sum(st)/len(st):.2f} "
            f"median={st[len(st)//2]:.2f} min={st[0]:.2f}")
    return outs


def main():
    ex = load_execution()
    if RANK != 0:
        ex.serve_workers()
        return
    ids0 = json.load(open("/tmp/c1_ids.json"))
    ids1 = ids0[:len(ids0) // 2] + [ids0[0]]     # different length AND content
    log(f"prompts: {len(ids0)} / {len(ids1)} caps={ex.capabilities}")

    a0 = solo(ex, 0, ids0, N_GEN); ex.reset()
    a1 = solo(ex, 0, ids1, N_GEN); ex.reset()
    log("A0", a0)
    log("A1", a1)

    b = batched(ex, [ids0, ids1], N_GEN)
    log("B0", b[0])
    log("B1", b[1])

    ok0, ok1 = a0 == b[0], a1 == b[1]
    def fd(x, y):
        return next((i for i, (p, q) in enumerate(zip(x, y)) if p != q), "len")
    log(f"row0 {'PASS' if ok0 else 'FAIL first_diff=' + str(fd(a0, b[0]))}")
    log(f"row1 {'PASS' if ok1 else 'FAIL first_diff=' + str(fd(a1, b[1]))}")
    log("PASS" if (ok0 and ok1) else "FAIL")
    ex.reset()
    ex.shutdown()
    sys.exit(0 if (ok0 and ok1) else 1)


if __name__ == "__main__":
    main()
