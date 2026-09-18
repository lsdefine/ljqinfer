import torch, time, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import ops
m = ops._mod()
torch.manual_seed(0)
Q, r, d, rd = 8, int(os.environ.get('R','4')), 512, 64
def bench(overlap, rotate, dt=torch.float32, noq=False):
    S = 2 * r if overlap else r; C = 2 * d if overlap else d
    kv = torch.randn(Q, S, C, device='cuda', dtype=dt)
    sc = torch.randn(Q, S, C, device='cuda', dtype=dt)
    if overlap: sc[:, :r, :] = float('-inf')
    ape = torch.randn(r, C, device='cuda')
    w = torch.ones(d, device='cuda', dtype=torch.float32)
    fr = torch.randn(Q, rd // 2, 2, device='cuda')
    valid = torch.ones(Q, device='cuda', dtype=torch.int32)
    f = lambda: m.compressor_rows(kv, sc, ape, w, fr, valid, r, d, rd, 1e-6, overlap, rotate, noq)
    gp = f'/tmp/ctail_gold_{r}_{int(overlap)}_{int(rotate)}_{int(noq)}_{str(dt)[-4:]}.pt'
    y = f().cpu()
    if os.environ.get('GOLD'): torch.save(y, gp)
    elif os.path.exists(gp):
        g = torch.load(gp); print('  exact vs gold:', torch.equal(y, g), 'maxdiff', (y.float() - g.float()).abs().max().item())
    for _ in range(5): f()
    torch.cuda.synchronize(); t = time.time()
    for _ in range(100): f()
    torch.cuda.synchronize(); tot = (time.time() - t) / 100 * 1e6
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as p:
        f(); torch.cuda.synchronize()
    ks = [(e.key[:40], round(e.device_time, 1)) for e in p.key_averages() if e.device_time > 0]
    print(f'overlap={overlap} rotate={rotate} noq={noq} dt={dt}: wall {tot:.1f}us', ks)
bench(False, False); bench(False, True); bench(True, True); bench(True, False); bench(True, True, noq=True); bench(False, False, torch.bfloat16)

# ring variant vs materialised windows (must be bitwise equal)
def ring_check(overlap, rotate, dt=torch.float32):
    torch.manual_seed(1)
    C = 2 * d if overlap else d; R = 2 * r if overlap else r
    kvr = torch.randn(4 * r, C, device='cuda', dtype=dt); scr = torch.randn(4 * r, C, device='cuda', dtype=dt)
    ape = torch.randn(r, C, device='cuda'); w = torch.randn(d, device='cuda', dtype=torch.float32)
    fr = torch.polar(torch.ones(Q, rd // 2, device='cuda'), torch.randn(Q, rd // 2, device='cuda'))
    w0 = torch.randint(0, 4 * r, (Q,), device='cuda'); w0[0] = 0  # has_prev=0 case
    valid = torch.randint(0, 2, (Q,), device='cuda', dtype=torch.int32)
    ar = torch.arange(r, device='cuda')
    rows = (w0[:, None] + ar) % (4 * r)
    kv = kvr[rows]; sc = scr[rows]
    if overlap:
        prow = (w0[:, None] - r + ar) % (4 * r); hp = (w0 >= r)
        rows = torch.cat([prow, rows], 1)
        sc = torch.cat([torch.where(hp[:, None, None], scr[prow], float('-inf')), sc], 1)
        kv = torch.cat([kvr[prow], kv], 1)
    ref = m.compressor_rows(kv, sc, ape, w, fr, valid, r, d, rd, 1e-6, overlap, rotate, False)
    out = m.compressor_rows_ring(kvr, scr, rows.contiguous(), (w0 >= r).int(), ape, w, fr, valid,
                                 r, d, rd, 1e-6, overlap, rotate, False)
    print(f'RING overlap={overlap} rotate={rotate} dt={dt}: equal={torch.equal(ref, out)} maxdiff={(ref.float()-out.float()).abs().max().item()}')
for ov in (False, True):
    for ro in (False, True):
        ring_check(ov, ro); ring_check(ov, ro, torch.bfloat16)
