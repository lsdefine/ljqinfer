# -*- coding: utf-8 -*-
"""generate_batch B>1 end-to-end gate (torchrun 8 ranks).

A: each row generated alone through the B=1 path.
B: both rows through _generate_multi, with different max_new_tokens so row1
   leaves the batch early -> exercises unbind + fallback to the B1 graph.
PASS if per-row token sequences are identical.

torchrun --nproc_per_node=8 tc_genmulti.py
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # repo root on sys.path (this file lives in bench/)
import json, os, sys, threading, time

from model.model_api import load_execution

RANK = int(os.environ.get("RANK", "0"))
N0 = int(os.environ.get("N0", "24"))
N1 = int(os.environ.get("N1", "10"))


def log(*a):
    if RANK == 0:
        print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def gen(ex, rows_ids, limits):
    B = len(rows_ids)
    for r, ids in enumerate(rows_ids):
        ex.set_input(list(ids), row=r)
    out = [[] for _ in range(B)]

    def emit(row, toks):
        out[row].extend(list(toks))

    evs = [threading.Event() for _ in range(B)]
    st = {}
    ex.generate_batch([list(x) for x in rows_ids], list(limits), evs, emit,
                      temperatures=[0.0] * B, eos_token_id=None, stats=st)
    return out, st


def main():
    ex = load_execution()
    if RANK != 0:
        ex.serve_workers()
        return
    ids0 = json.load(open("/tmp/c1_ids.json"))
    if os.environ.get("SAME") == "1":
        ids1 = list(ids0)
    elif os.environ.get("NEAR") == "1":
        ids1 = ids0[:len(ids0) - 5]
    else:
        ids1 = ids0[:len(ids0) // 2] + [ids0[0]]
    log(f"len ids0={len(ids0)} ids1={len(ids1)}")
    if os.environ.get("SWAP") == "1":
        ids0, ids1 = ids1, ids0
    log(f"caps={ex.capabilities}")
    t = ex.warmup_batch_graphs()
    log(f"warmup_batch_graphs {t:.2f}s")

    a0, _ = gen(ex, [ids0], [N0]); ex.reset()
    a1, _ = gen(ex, [ids1], [N1]); ex.reset()
    b, st = gen(ex, [ids0, ids1], [N0, N1]); ex.reset()

    log("A0", a0[0]); log("B0", b[0])
    log("A1", a1[0]); log("B1", b[1])
    ok0, ok1 = a0[0] == b[0], a1[0] == b[1]

    def fd(x, y):
        return next((i for i, (p, q) in enumerate(zip(x, y)) if p != q), "len")

    ms = st.get("step_seconds") or []
    log(f"batch steps={st.get('steps')} row_steps={st.get('row_steps')} "
        f"mean_step={1e3 * sum(ms) / max(len(ms), 1):.2f}ms")
    log(f"row0 {'PASS' if ok0 else 'FAIL first_diff=' + str(fd(a0[0], b[0]))} "
        f"(len {len(a0[0])} vs {len(b[0])})")
    log(f"row1 {'PASS' if ok1 else 'FAIL first_diff=' + str(fd(a1[0], b[1]))} "
        f"(len {len(a1[0])} vs {len(b[1])})")
    if os.environ.get("SAME") == "1":
        n = min(len(b[0]), len(b[1]))
        same = b[0][:n] == b[1][:n]
        log(f"SELFCHECK identical-input rows equal={same} first_diff={fd(b[0][:n], b[1][:n])} len={len(b[0])},{len(b[1])}")
    log("PASS" if (ok0 and ok1) else "FAIL")
    ex.shutdown()
    sys.exit(0 if (ok0 and ok1) else 1)


if __name__ == "__main__":
    main()
