# -*- coding: utf-8 -*-
"""Pre-capture cost probe (torchrun 8 ranks).

Answers three design questions:
  1. VRAM cost of holding every batch graph (all slot subsets with |S|>=2, 11 of them)
  2. wall time to capture them at startup
  3. graph replay time vs B (B1 _graph vs B2/B3/B4 batch graphs), same pos, empty qin

torchrun --nproc_per_node=8 tc_precapture.py
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # repo root on sys.path (this file lives in bench/)
import itertools, os, sys, time
import torch
from model.model_api import load_execution

RANK = int(os.environ.get("RANK", "0"))
POS = 256
REPS = 20


def log(*a):
    if RANK == 0:
        print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def free_gb(dev):
    torch.cuda.synchronize(dev)
    return torch.cuda.mem_get_info(dev)[0] / (1 << 30)


def bench(fn, reps=REPS):
    fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t) * 1e3)
    ts.sort()
    return ts[len(ts) // 2], ts[0]


def main():
    ex = load_execution()
    if RANK != 0:
        ex.serve_workers()
        return
    dev = ex.dev
    slots = [ex._rpc("alloc") for _ in range(4)]
    log(f"slots={slots} free={free_gb(dev):.2f}GB")

    subsets = [c for n in (1, 2, 3, 4) for c in itertools.combinations(slots, n)]
    log(f"{len(subsets)} subsets to capture")
    base = free_gb(dev)
    t_all = time.perf_counter()
    rows = []
    for sub in subsets:
        f0 = free_gb(dev)
        t0 = time.perf_counter()
        ex._rpc("prepare_batch", tuple(sub))
        dt = time.perf_counter() - t0
        f1 = free_gb(dev)
        rows.append((sub, dt, f0 - f1))
        log(f"  capture {sub} {dt:.2f}s  vram -{f0-f1:.3f}GB  free={f1:.2f}GB")
    total_t = time.perf_counter() - t_all
    total_v = base - free_gb(dev)
    log(f"TOTAL capture {total_t:.1f}s  vram {total_v:.2f}GB  free_left={free_gb(dev):.2f}GB")

    # replay time vs B (empty qin, same pos) -- pure graph cost curve
    sub1 = (slots[0],)
    ex._rpc("prepare_batch", sub1)
    med1, min1 = bench(lambda: ex._rpc("step_batch", sub1, (POS,)))
    log(f"B1 replay median={med1:.2f}ms min={min1:.2f}ms  -> {med1:.2f}ms/row")
    for n in (2, 3, 4):
        sub = tuple(slots[:n])
        ex._rpc("prepare_batch", sub)
        pos = tuple([POS] * n)
        med, mn = bench(lambda: ex._rpc("step_batch", sub, pos))
        log(f"B{n} replay median={med:.2f}ms min={mn:.2f}ms  -> {med/n:.2f}ms/row"
            f"  speedup_vs_B1={med1*n/med:.2f}x")
    ex.reset()
    ex.shutdown()


if __name__ == "__main__":
    main()
