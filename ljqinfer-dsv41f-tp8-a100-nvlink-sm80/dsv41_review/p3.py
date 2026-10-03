import sys, torch, triton, triton.language as tl

@triton.jit
def _k(X, W, O, M, N, K, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, SK: tl.constexpr):
    pn = tl.program_id(0); pk = tl.program_id(1)
    om = tl.arange(0, BM); on = pn * BN + tl.arange(0, BN)
    kpp = tl.cdiv(tl.cdiv(K, SK), BK) * BK
    k0 = pk * kpp; kend = tl.minimum(k0 + kpp, K)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    while k0 < kend:
        ok = k0 + tl.arange(0, BK)
        x = tl.load(X + om[:, None] * K + ok[None, :], mask=(om[:, None] < M) & (ok[None, :] < K), other=0.0)
        w = tl.load(W + on[:, None] * K + ok[None, :], mask=(on[:, None] < N) & (ok[None, :] < K), other=0.0)
        acc += tl.dot(x, tl.trans(w))
        k0 += BK
    p = O + om[:, None] * N + on[None, :]
    m = (om[:, None] < M) & (on[None, :] < N)
    if SK == 1:
        tl.store(p, acc, mask=m)
    else:
        tl.atomic_add(p, acc, mask=m)


def bench(fn, n=40):
    for _ in range(5): fn()
    torch.cuda.synchronize(); g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): fn()
    for _ in range(3): g.replay()
    torch.cuda.synchronize()
    a = torch.cuda.Event(True); b = torch.cuda.Event(True)
    a.record()
    for _ in range(n): g.replay()
    b.record(); torch.cuda.synchronize()
    return a.elapsed_time(b) / n * 1e3


SH = [(5120, 5120), (1280, 4608), (5120, 1280), (5120, 640), (4096, 1024), (5120, 128), (1024, 640)]


def main():
    M = int(sys.argv[1]); torch.cuda.set_device(0)
    BM = max(16, triton.next_power_of_2(M))
    print('M=%d  shape       cuBLAS   triton+splitK   GB/s(bf16)  cfg' % M)
    tc = tt = 0.0
    for K, N in SH:
        x = torch.randn(M, K, device='cuda', dtype=torch.bfloat16)
        w = torch.randn(N, K, device='cuda', dtype=torch.bfloat16)
        o = torch.zeros(M, N, device='cuda', dtype=torch.float32)
        ob = torch.empty(M, N, device='cuda', dtype=torch.bfloat16)
        ref = (x.float() @ w.float().t())
        t_cu = bench(lambda: torch.mm(x, w.t(), out=ob))
        best = (1e9,)
        for BN in (32, 64, 128):
            for BK in (64, 128, 256):
                for SK in (1, 2, 4, 8, 16):
                    if triton.cdiv(N, BN) * SK > 1024 or K // SK < BK: continue
                    for nw in (4, 8):
                        g = (triton.cdiv(N, BN), SK)
                        def run(BN=BN, BK=BK, SK=SK, nw=nw, g=g):
                            if SK > 1: o.zero_()
                            _k[g](x, w, o, M, N, K, BM, BN, BK, SK, num_warps=nw, num_stages=3)
                        try:
                            run(); torch.cuda.synchronize()
                            if (o - ref).abs().max().item() / ref.abs().max().item() > 2e-2: continue
                            t = bench(run)
                            if t < best[0]: best = (t, BN, BK, SK, nw)
                        except Exception: pass
        tc += t_cu; tt += best[0]
        print('(%5d,%5d) %8.2fus %8.2fus  cu %5.0f -> tri %5.0f GB/s  BN=%d BK=%d SK=%d nw=%d'
              % (K, N, t_cu, best[0], 2*N*K/t_cu/1e3, 2*N*K/best[0]/1e3, *best[1:]))
    print('TOTAL cuBLAS %.1fus  triton %.1fus  save %.1fus/layer -> %.2fms/step(x40)'
          % (tc, tt, tc - tt, (tc - tt) * 40 / 1e3))


main()
