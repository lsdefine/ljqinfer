import sys, torch, triton, triton.language as tl

@triton.jit
def _read(W, O, N, K, BN: tl.constexpr, BK: tl.constexpr):
    pid = tl.program_id(0)
    on = pid * BN + tl.arange(0, BN)
    a = tl.zeros((BN, BK), dtype=tl.float32)
    for k0 in range(0, K, BK):
        ok = k0 + tl.arange(0, BK)
        w = tl.load(W + on[:, None] * K + ok[None, :], mask=on[:, None] < N, other=0)
        a += w.to(tl.float32)
    tl.store(O + on, tl.sum(a, 1), mask=on < N)

@triton.jit
def _bdot(X, W, O, M, N, K, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid = tl.program_id(0)
    on = pid * BN + tl.arange(0, BN)
    om = tl.arange(0, BM)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, K, BK):
        ok = k0 + tl.arange(0, BK)
        x = tl.load(X + om[:, None] * K + ok[None, :], mask=om[:, None] < M, other=0.)
        w = tl.load(W + on[:, None] * K + ok[None, :], mask=on[:, None] < N, other=0.)
        acc += tl.dot(x, tl.trans(w))
    tl.store(O + om[:, None] * N + on[None, :], acc, mask=(om[:, None] < M) & (on[None, :] < N))

def bench(fn, it=30):
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(5): fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): fn()
    for _ in range(3): g.replay()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(True), torch.cuda.Event(True)
    a.record()
    for _ in range(it): g.replay()
    b.record(); torch.cuda.synchronize()
    return a.elapsed_time(b) / it * 1e3

SH = [(5120,5120),(1280,4608),(5120,1280),(5120,640),(4096,1024),(5120,128),(1024,640)]

def main():
    M = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    BM = max(16, triton.next_power_of_2(M))
    torch.cuda.set_device(0)
    print('M=%d  shape(K,N)      cuBLAS |  P1 pure-read fp8   |  P2 triton bf16 dot' % M)
    for K, N in SH:
        x = torch.randn(M, K, device='cuda', dtype=torch.bfloat16)
        w8 = torch.randint(0, 255, (N, K), device='cuda', dtype=torch.uint8)
        wb = torch.randn(N, K, device='cuda', dtype=torch.bfloat16)
        o1 = torch.empty(N, device='cuda', dtype=torch.float32)
        ob = torch.empty(M, N, device='cuda', dtype=torch.bfloat16)
        o2 = torch.empty(M, N, device='cuda', dtype=torch.float32)
        tcu = bench(lambda: torch.mm(x, wb.t(), out=ob))
        p1 = 1e9; p2 = 1e9
        for BN in (64, 128, 256):
            for BK in (64, 128, 256):
                for nw in (4, 8):
                    g = (triton.cdiv(N, BN),)
                    try:
                        p1 = min(p1, bench(lambda: _read[g](w8, o1, N, K, BN, BK, num_warps=nw, num_stages=3)))
                        p2 = min(p2, bench(lambda: _bdot[g](x, wb, o2, M, N, K, BM, BN, BK, num_warps=nw, num_stages=3)))
                    except Exception: pass
        print('(%5d,%5d) %7.2fus | %7.2fus %6.0fGB/s | %7.2fus %6.0fGB/s'
              % (K, N, tcu, p1, N*K/p1/1e3, p2, 2*N*K/p2/1e3))

main()
