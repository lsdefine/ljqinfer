"""Bench/verify fused indexer score vs the current gather+einsum+reduce chain (single GPU)."""
import os, sys, time, torch
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # repo root on sys.path (this file lives in bench/)
from ops import _mod

torch.manual_seed(0)
dev = "cuda"
B, S, H, D = 1, 8, 8, 128
N = 262144          # C_max = max_seq // ratio
ratio = 4
R = N               # pool rows (flat)

q = (torch.randn(B, S, H, D, device=dev) * 0.5).to(torch.bfloat16)
pool = (torch.randn(R, D, device=dev) * 0.5).to(torch.bfloat16)
prow = torch.randperm(R, device=dev).to(torch.int64).contiguous()
w = (torch.randn(B, S, H, device=dev) * 0.5).to(torch.bfloat16)

m = _mod()


def ref(pos):
    kv = pool.index_select(0, prow).unsqueeze(0)              # [1, N, D]
    sc = torch.einsum("bshd,btd->bsht", q, kv)                # bf16 [1,S,H,N]
    return m.index_score_reduce_positions(sc, w.float(), ratio, pos)


def fused(pos):
    return m.index_score_fused(q, pool, prow, w, pos, ratio)


def timeit(fn, pos, n=10):
    for _ in range(3):
        fn(pos)
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(n):
        fn(pos)
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / n * 1e3


ok_all = True
for base in (4096, 65536, 262143, 1048575):
    pos = torch.tensor([base + i for i in range(S)], device=dev, dtype=torch.int64)
    a, b = ref(pos), fused(pos)
    lim = ((pos + 1) // ratio).tolist()
    bad = 0
    md = 0.0
    for s in range(S):
        L = min(lim[s], N)
        x, y = a[0, s, :L], b[0, s, :L]
        md = max(md, (x - y).abs().max().item() if L else 0.0)
        bad += int((x != y).sum().item())
        tail_ok = bool(torch.isinf(a[0, s, L:]).all() and torch.isinf(b[0, s, L:]).all())
        if not tail_ok:
            print(f"  [tail] row {s} mask mismatch")
            ok_all = False
    frac = bad / max(sum(min(l, N) for l in lim), 1)
    print(f"pos~{base:>8}  lim={lim[0]:>7}  max|diff|={md:.6g}  bitwise-mismatch={frac*100:.4f}%")
    if md > 2e-2:
        ok_all = False
    tr, tf = timeit(ref, pos), timeit(fused, pos)
    print(f"           ref={tr:.3f} ms   fused={tf:.3f} ms   speedup={tr/tf:.2f}x")

print("RESULT:", "PASS" if ok_all else "FAIL")
