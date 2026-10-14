"""Small CUDA finish kernel; full-vocabulary sort/softmax/cumsum stay in PyTorch."""
import torch
import triton as tr
import triton.language as tl


@tr.jit
def _finish(C, I, U, O, V: tl.constexpr, DROP: tl.constexpr, R: tl.constexpr,
            BLOCK: tl.constexpr):
    row = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    live = row < R
    base = row * V
    total = tl.load(C + base + V - 1, live, other=1.)
    valid = (total > 0.) & (total < float('inf'))
    # upper_bound(DROP): retain the boundary token and always at least one.
    lo = tl.full((BLOCK,), 0, tl.int32)
    hi = tl.full((BLOCK,), V, tl.int32)
    for _ in range((V + 1).bit_length()):
        mid = (lo + hi) // 2
        val = tl.load(C + base + mid, live & (mid < V), other=float('inf'))
        right = (lo < hi) & (val <= DROP)
        hi = tl.where((lo < hi) & ~right, mid, hi)
        lo = tl.where(right, mid + 1, lo)
    first = tl.minimum(lo, V - 1)
    tail = tl.load(C + base + first - 1, live & (first > 0), other=0.)
    u = tl.load(U + row, live, other=0.)
    target = tail + (total - tail) * u
    # CDF is positive FP32: predecessor(total) avoids target rounding up to total.
    below = (total.to(tl.int32, bitcast=True) - 1).to(tl.float32, bitcast=True)
    target = tl.minimum(target, below)
    lo = first
    hi = tl.full((BLOCK,), V, tl.int32)
    for _ in range((V + 1).bit_length()):
        mid = (lo + hi) // 2
        val = tl.load(C + base + mid, live & (mid < V), other=float('inf'))
        right = (lo < hi) & (val <= target)
        hi = tl.where((lo < hi) & ~right, mid, hi)
        lo = tl.where(right, mid + 1, lo)
    pick = tl.minimum(lo, V - 1)
    token = tl.load(I + base + pick, live, other=0)
    tl.store(O + row, tl.where(valid, token, -1), live)


def inverse_cdf_cuda(cdf, ids, uniforms, top_p):
    rows, vocab = cdf.shape
    out = torch.empty(rows, device=cdf.device, dtype=torch.long)
    _finish[(tr.cdiv(rows, 32),)](cdf, ids, uniforms, out, vocab, 1 - top_p,
                                rows, 32, num_warps=1, enable_fp_fusion=False)
    # No CPU round trip. Invalid distributions fail explicitly rather than emit a
    # plausible token; a CUDA assert is fatal to this worker, as in multinomial.
    torch._assert_async((out >= 0).all(), 'non-finite sampling distribution')
    return out
