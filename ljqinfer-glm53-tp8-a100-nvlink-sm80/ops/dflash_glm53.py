"""DFlash2 grouped convolution, ported from GPU140 TP4 reference."""
import torch
import triton as tr
import triton.language as tl
@tr.jit
def _grouped(H, D, B, O, C: tl.constexpr, T: tl.constexpr,
             DS0: tl.constexpr, DS1: tl.constexpr, DS2: tl.constexpr,
             SIDE: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    t, c = i // C, i % C
    ok = i < T * C
    h0 = tl.load(H + i, ok, 0).to(tl.float32)
    h1 = tl.load(H + i - C, ok & (t > 0), 0).to(tl.float32)
    b0 = tl.load(B + SIDE * 2 * C + c, ok, 0).to(tl.float32)
    b1 = tl.load(B + (SIDE * 2 + 1) * C + c, ok, 0).to(tl.float32)
    d0 = tl.load(D + t * DS0 + c // 16 * DS2, ok, 0).to(tl.float32)
    d1 = tl.load(D + t * DS0 + DS1 + c // 16 * DS2, ok, 0).to(tl.float32)
    w0 = (b0 + d0).to(H.dtype.element_ty).to(tl.float32)
    w1 = (b1 + d1).to(H.dtype.element_ty).to(tl.float32)
    p0 = (h0 * w0).to(H.dtype.element_ty).to(tl.float32)
    p1 = (h1 * w1).to(H.dtype.element_ty).to(tl.float32)
    tl.store(O + i, p0 + p1, ok)

def grouped(hidden, delta, base, side, out=None):
    if out is None:
        out = torch.empty_like(hidden)
    _grouped[(tr.cdiv(hidden.numel(), 256),)](hidden, delta, base, out,
          hidden.shape[1], hidden.shape[0], *delta.stride(), side, 256, enable_fp_fusion=False)
    return out