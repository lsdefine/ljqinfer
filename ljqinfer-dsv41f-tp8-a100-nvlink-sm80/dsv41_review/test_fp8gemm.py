"""FP8 weight-direct GEMM on tensor cores (Triton) vs cuBLAS BF16 mm.

The dense path keeps a dequantized BF16 copy of every FP8 weight and calls
cuBLAS: tensor cores at 1.45 TB/s, but two bytes per weight element.  The two
earlier attempts at reading the FP8 bytes directly were hand-written SIMT
GEMVs, which give up the tensor cores and lose.  This is the third road: keep
tl.dot, unpack E4M3 to BF16 with integer ops in registers, so the DRAM traffic
halves while the math stays on the tensor cores.

E4M3 (bias 7) -> BF16 (bias 127) is a bit move: sign to bit 15, the seven E+M
bits to bits 4..10, then one multiply by 2^(e_scale-7) folds both the bias
difference and the E8M0 block scale.  The multiply is a power of two, so it is
exact in BF16 and subnormal E4M3 lands back in BF16's normal range.

Decode GEMMs are tall-and-skinny (M<=48), so N alone does not fill the device
on the narrow shapes; the kernel splits K across programs and reduces with
atomics, which is exactly why cuBLAS reaches for splitKreduce here.
"""
import sys, torch, triton
import triton.language as tl


@triton.jit
def _fp8_gemm(X, W, S, O, M, N, K, s_sn, s_sk,
              BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
              SK: tl.constexpr):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_m = tl.arange(0, BM)
    nmask = offs_n < N
    sn = offs_n // 32
    kpp = tl.cdiv(tl.cdiv(K, BK), SK) * BK          # K per program, BK-aligned
    kstart = pid_k * kpp
    kstop = tl.minimum(kstart + kpp, K)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(kstart, kstop, BK):
        offs_k = k0 + tl.arange(0, BK)
        kmask = offs_k < K
        x = tl.load(X + offs_m[:, None] * K + offs_k[None, :],
                    mask=(offs_m[:, None] < M) & kmask[None, :], other=0.0)
        w8 = tl.load(W + offs_n[:, None] * K + offs_k[None, :],
                     mask=nmask[:, None] & kmask[None, :], other=0)
        u = w8.to(tl.uint16)
        bits = ((u & 0x7F) << 4) | ((u & 0x80) << 8)
        wv = bits.to(tl.uint16).to(tl.bfloat16, bitcast=True)
        e = tl.load(S + sn[:, None] * s_sn + (offs_k // 32)[None, :] * s_sk,
                    mask=nmask[:, None] & kmask[None, :], other=0).to(tl.int32)
        # 2^(e-127) * 2^120 == 2^(e-7); a power of two, so BF16 keeps it exact
        sc = ((e - 7 + 127) << 23).to(tl.float32, bitcast=True).to(tl.bfloat16)
        acc += tl.dot(x, tl.trans(wv * sc))
    p = O + offs_m[:, None] * N + offs_n[None, :]
    m = (offs_m[:, None] < M) & nmask[None, :]
    if SK == 1:
        tl.store(p, acc, mask=m)
    else:
        tl.atomic_add(p, acc, mask=m)


def fp8_gemm(x, w8, s, out, BN=128, BK=128, SK=1, nw=4, ns=3):
    M, K = x.shape
    N = w8.shape[0]
    _fp8_gemm[(triton.cdiv(N, BN), SK)](x, w8, s, out, M, N, K, s.stride(0), s.stride(1),
                                        triton.next_power_of_2(max(M, 16)), BN, BK, SK,
                                        num_warps=nw, num_stages=ns)
    return out


def dequant(w8, s):
    n, k = w8.shape
    sf = torch.exp2(s.to(torch.float32) - 127.0)
    sf = sf.repeat_interleave(32, 0).repeat_interleave(32, 1)[:n, :k]
    return (w8.to(torch.float32) * sf).to(torch.bfloat16)


def graph_bench(fn, iters=50):
    st = torch.cuda.Stream(); st.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(st):
        for _ in range(3): fn()
    torch.cuda.current_stream().wait_stream(st)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(iters): fn()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(True), torch.cuda.Event(True)
    a.record(); g.replay(); b.record(); torch.cuda.synchronize()
    return a.elapsed_time(b) * 1000.0 / iters


SHAPES = [(5120, 5120), (1280, 4608), (5120, 1280), (5120, 640),
          (4096, 1024), (5120, 128), (1024, 640)]   # (K, N) real decode shapes


def main():
    M = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    dev = 'cuda'
    torch.manual_seed(0)
    tot_cublas = tot_fp8 = 0.0
    print('M=%d  (K,N)          cuBLAS    triton-fp8   GB/s   maxrel   cfg' % M)
    for K, N in SHAPES:
        x = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
        w8 = (torch.randn(N, K, device=dev) * 0.3).to(torch.float8_e4m3fn)
        s = torch.randint(120, 131, ((N + 31) // 32, (K + 31) // 32),
                          device=dev, dtype=torch.uint8)
        wb = dequant(w8, s)
        ref = torch.mm(x, wb.t())
        wu = w8.view(torch.uint8)
        of = torch.empty(M, N, device=dev, dtype=torch.float32)
        ob = torch.empty(M, N, device=dev, dtype=torch.bfloat16)

        def run(BN, BK, SK, nw, ns):
            if SK > 1:
                of.zero_()
            fp8_gemm(x, wu, s, of, BN, BK, SK, nw, ns)

        best = None
        for BN in (64, 128):
            for BK in (128,):
                for SK in (1, 2, 4, 8):
                    if triton.cdiv(N, BN) * SK > 512: continue
                    for nw in (4, 8):
                        for ns in (3,):
                            try:
                                t = graph_bench(lambda: run(BN, BK, SK, nw, ns), 30)
                            except Exception:
                                continue
                            if best is None or t < best[0]: best = (t, BN, BK, SK, nw, ns)
        run(*best[1:]); torch.cuda.synchronize()
        rel = (of - ref.float()).abs().max().item() / ref.float().abs().max().item()
        t_cu = graph_bench(lambda: torch.mm(x, wb.t(), out=ob))
        tot_cublas += t_cu; tot_fp8 += best[0]
        print('  (%5d,%5d) %8.2fus %8.2fus %6.0f %8.1e  BN=%d BK=%d SK=%d nw=%d ns=%d'
              % (K, N, t_cu, best[0], N * K / best[0] / 1e3, rel, *best[1:]))
    print('TOTAL  cuBLAS %.1fus   triton-fp8 %.1fus   x%.2f' %
          (tot_cublas, tot_fp8, tot_cublas / tot_fp8))


main()
