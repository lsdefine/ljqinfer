"""Batch-invariance guard for fp32 skinny GEMM (ops.sgemm_skinny2_f32).

WHY THIS EXISTS
---------------
2026-09-05: sgemm_skinny2_f32 was templated for M<=8 only, and Indexer.proj silently
fell back to cuBLAS for M>8. With MTP, M = B*(1+draft), so B<=2 used the skinny kernel
and B>=3 used cuBLAS -> different fp32 reduction order -> indexer top-k picked different
pages -> the model produced a *different sentence* at B>=3. Deterministic, silent, and
it cost hours to find. The kernel now covers any M with an identical per-row reduction
order, and there is NO fallback path. This test exists so that never comes back:
any future B (8, 16, 32...) must stay bitwise identical to B=1.

Run: python tests/test_batch_invariance.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from ops import _mod

M_LIST = [1, 2, 3, 4, 5, 7, 8, 9, 12, 16, 17, 32, 33, 64, 128]
FAIL = []

def check(cond, msg):
    print(("  PASS  " if cond else "  FAIL  ") + msg)
    if not cond:
        FAIL.append(msg)

def main():
    m = _mod()
    torch.manual_seed(0)
    for (K, N) in [(7168, 1024), (2048, 512), (512, 256)]:
        x = torch.randn(max(M_LIST), K, device="cuda", dtype=torch.float32)
        w = torch.randn(N, K, device="cuda", dtype=torch.float32)
        print(f"[K={K} N={N}]")
        ref = None
        for M in M_LIST:
            y = m.sgemm_skinny2_f32(x[:M].contiguous(), w)
            check(y.shape == (M, N), f"M={M} shape {tuple(y.shape)}")
            if ref is None:
                ref = y.clone()
            else:
                # every row shared with the smaller run must be BITWISE identical
                k = min(M, ref.shape[0])
                check(torch.equal(y[:k], ref[:k]), f"M={M} rows[:{k}] bitwise == M={M_LIST[0]}..prev")
                if M > ref.shape[0]:
                    ref = y.clone()
            # sanity: still numerically a correct GEMM (vs cuBLAS, not bitwise)
            c = x[:M] @ w.t()
            rel = ((y - c).abs().max() / c.abs().max()).item()
            check(rel < 1e-5, f"M={M} rel_err_vs_cublas={rel:.2e} < 1e-5")
    print("\nRESULT:", "ALL PASS" if not FAIL else f"{len(FAIL)} FAILURES")
    return 1 if FAIL else 0

if __name__ == "__main__":
    sys.exit(main())
