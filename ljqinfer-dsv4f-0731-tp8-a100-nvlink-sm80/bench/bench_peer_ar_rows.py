"""torchrun --nproc_per_node 8 bench/bench_peer_ar_rows.py
Correctness + latency of peer_ar_rows vs NCCL for indexer score [Q, C]."""
import os, sys, time
import torch, torch.distributed as dist
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ops import peer_ar_rows as par

dist.init_process_group("nccl")
r, w = dist.get_rank(), dist.get_world_size()
torch.cuda.set_device(r)
dev = torch.device("cuda", r)
Q, RATIO = 8, 4
C_MAX = int(os.environ.get("C_MAX", 98304))
assert par.init(Q * C_MAX), "init failed"

def mk(seed, pos):
    g = torch.Generator(device=dev).manual_seed(seed * 100 + r)
    x = torch.randn(Q, C_MAX, generator=g, device=dev)
    lim = ((pos + 1) // RATIO)
    x[torch.arange(C_MAX, device=dev)[None, :] >= lim[:, None]] = float("-inf")
    return x

ok = True
for it, L in enumerate([1, 4000, 45000, 200000, C_MAX * RATIO - 1, 7]):
    pos = torch.full((Q,), L, dtype=torch.int64, device=dev) + torch.arange(Q, device=dev)
    x = mk(it, pos)
    ref_nccl = x.clone(); dist.all_reduce(ref_nccl)
    parts = [torch.empty_like(x) for _ in range(w)]; dist.all_gather(parts, x)
    ref_seq = parts[0].clone()
    for j in range(1, w): ref_seq += parts[j]              # fixed order == kernel
    y = x.clone(); par.all_reduce_rows_(y, pos, RATIO); torch.cuda.synchronize()
    fin = torch.isfinite(ref_seq)
    biteq = torch.equal(y, ref_seq)
    maxd = (y[fin] - ref_nccl[fin]).abs().max().item()
    tk_ok = torch.equal(y.topk(min(2048, int(fin[0].sum())), dim=-1).indices,
                        ref_nccl.topk(min(2048, int(fin[0].sum())), dim=-1).indices)
    tail_ok = torch.equal(torch.isinf(y), torch.isinf(ref_nccl))
    ok &= biteq and tail_ok
    if r == 0:
        print(f"L={L:7d} biteq_seq={biteq} tail_ok={tail_ok} |y-nccl|max={maxd:.3e} topk_eq={tk_ok}", flush=True)

def timeit(fn, n=50):
    for _ in range(5): fn()
    torch.cuda.synchronize(); dist.barrier()
    t = time.perf_counter()
    for _ in range(n): fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / n * 1e3

# reference: pure sync (no live data) and the legacy peer_ar (hidden AR)
pos0 = torch.full((Q,), -1, dtype=torch.int64, device=dev); x0 = torch.zeros(Q, C_MAX, device=dev)
g0 = torch.cuda.CUDAGraph(); s0 = torch.cuda.Stream(); s0.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s0):
    for _ in range(2): par.all_reduce_rows_(x0, pos0, RATIO)
torch.cuda.current_stream().wait_stream(s0)
with torch.cuda.graph(g0):
    for _ in range(21): par.all_reduce_rows_(x0, pos0, RATIO)
t0 = timeit(lambda: g0.replay())
par._mod().set_oneshot_max(-1); dist.barrier()
g0b = torch.cuda.CUDAGraph()
with torch.cuda.graph(g0b):
    for _ in range(21): par.all_reduce_rows_(x0, pos0, RATIO)
t0b = timeit(lambda: g0b.replay())
par._mod().set_oneshot_max(-1); dist.barrier()
if r == 0: print(f"SYNC floor 21x: oneshot={t0:.3f} twoshot={t0b:.3f}", flush=True)
try:
    from ops import peer_ar as legacy
    h = torch.randn(Q, 7168, device=dev, dtype=torch.bfloat16)
    legacy.prewarm(h.numel()) if 'numel' in legacy.prewarm.__code__.co_varnames else legacy.prewarm()
    hh = h.clone()
    for _ in range(3): legacy.all_reduce(hh)
    t_leg = timeit(lambda: [legacy.all_reduce(hh) for _ in range(21)])
    t_nl = timeit(lambda: [dist.all_reduce(hh) for _ in range(21)])
except Exception as e:
    t_leg = float('nan'); t_nl = float('nan'); print('legacy err', repr(e)[:200])
if r == 0: print(f"REF 21x: rows_sync_only={t0:.3f}ms legacy_peer_ar(hidden bf16)={t_leg:.3f}ms nccl(hidden)={t_nl:.3f}ms", flush=True)
for L in [4000, 45000, 200000, C_MAX * RATIO - 1]:
    pos = torch.full((Q,), L, dtype=torch.int64, device=dev)
    x = mk(9, pos)
    xn = x.clone(); xp = x.clone()
    t_n = timeit(lambda: [dist.all_reduce(xn) for _ in range(21)])
    t_p = timeit(lambda: [par.all_reduce_rows_(xp, pos, RATIO) for _ in range(21)])
    # graph replay of the peer path (the real deployment mode)
    g = torch.cuda.CUDAGraph(); s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(2): par.all_reduce_rows_(xp, pos, RATIO)
    torch.cuda.current_stream().wait_stream(s)
    with torch.cuda.graph(g):
        for _ in range(21): par.all_reduce_rows_(xp, pos, RATIO)
    t_g = timeit(lambda: g.replay())
    par._mod().set_oneshot_max(-1); dist.barrier()
    t_two = timeit(lambda: [par.all_reduce_rows_(xp, pos, RATIO) for _ in range(21)])
    par._mod().set_oneshot_max(1 << 30); dist.barrier()
    t_one = timeit(lambda: [par.all_reduce_rows_(xp, pos, RATIO) for _ in range(21)])
    par._mod().set_oneshot_max(-1); dist.barrier()
    par._mod().set_oneshot_max(-1); dd = []
    for m in (1, 2, 4, 3, 6, 5, 7):
        par._mod().set_dbg(m); dist.barrier()
        dd.append(f"skip{m}={timeit(lambda: [par.all_reduce_rows_(xp, pos, RATIO) for _ in range(21)]):.3f}")
    par._mod().set_dbg(0); par._mod().set_oneshot_max(-1); dist.barrier()
    if r == 0: print(f"  L={L} forced: oneshot={t_one:.3f} twoshot={t_two:.3f} " + " ".join(dd), flush=True)
    lane_res = []
    par._mod().set_lanes(32)
    for pl in (2048, 4096, 8192, 16384, 1 << 30):
        par._mod().set_per_lane(pl); dist.barrier()
        g2 = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g2):
            for _ in range(21): par.all_reduce_rows_(xp, pos, RATIO)
        lane_res.append((pl, timeit(lambda: g2.replay())))
    par._mod().set_per_lane(8192); dist.barrier()
    if r == 0:
        print(f"C_MAX={C_MAX} L={L:7d} 21x: nccl={t_n:.3f}ms peer={t_p:.3f}ms peer_graph={t_g:.3f}ms perlane:" + " ".join(f"{a}={b:.3f}" for a,b in lane_res), flush=True)
if r == 0: print("ALL_OK" if ok else "FAIL", flush=True)
dist.barrier(); dist.destroy_process_group()
