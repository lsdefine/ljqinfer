"""Bench + correctness for the one-shot peer all-reduce.  torchrun --nproc_per_node=8 tc_peer_ar.py"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # repo root on sys.path (this file lives in bench/)
import os
import time

import torch
import torch.distributed as dist

from ops import peer_ar

R = int(os.environ["RANK"])
W = int(os.environ["WORLD_SIZE"])
torch.cuda.set_device(R)
dist.init_process_group("nccl")

N = 8 * 7168                      # the hidden all-reduce that dominates the decode step
assert peer_ar.init(N), "peer_ar init failed"

from ops.build import load_wgemm
_ext = load_wgemm(False)


def ar2(y):
    _ext.peer_ar_ipc_run2(y)


def log(*a):
    if R == 0:
        print(*a, flush=True)


# --- correctness: same result as NCCL, repeated to exercise the seq double buffer
torch.manual_seed(1234 + R)
bad = 0
for it in range(50):
    x = torch.randn(N, device="cuda", dtype=torch.float32)
    ref = x.clone()
    dist.all_reduce(ref)
    y = x.clone()
    peer_ar.all_reduce(y)
    y2 = x.clone()
    ar2(y2)
    d = max((y - ref).abs().max().item(), (y2 - ref).abs().max().item())
    if d > 1e-3:
        bad += 1
        log(f"  iter {it}: max|diff| = {d:.3e}")
log(f"correctness over 50 iters: {'FAIL ' + str(bad) if bad else 'OK'}")


def bench(fn, iters=300):
    for _ in range(30):
        fn()
    torch.cuda.synchronize()
    dist.barrier()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e6


buf = torch.randn(N, device="cuda", dtype=torch.float32)
log(f"nccl    all_reduce: {bench(lambda: dist.all_reduce(buf)):8.2f} us")
log(f"peer-AR all_reduce: {bench(lambda: peer_ar.all_reduce(buf)):8.2f} us")
log(f"peer-2S all_reduce: {bench(lambda: ar2(buf)):8.2f} us")

# --- the number that matters: 113 back-to-back calls inside one replayed graph
for name, fn in (("nccl ", lambda: dist.all_reduce(buf)),
                 ("peer ", lambda: peer_ar.all_reduce(buf)),
                 ("peer2", lambda: ar2(buf))):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(5):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    dist.barrier()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(113):
            fn()
    torch.cuda.synchronize()
    dist.barrier()
    for _ in range(3):
        g.replay()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(20):
        g.replay()
    torch.cuda.synchronize()
    log(f"graph x113 {name}: {(time.perf_counter() - t0) / 20 * 1e3:7.3f} ms/replay")

dist.barrier()
dist.destroy_process_group()
