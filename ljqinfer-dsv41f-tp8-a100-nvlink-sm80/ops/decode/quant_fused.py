"""Decode-only fused activation quantization. One launch per call.

Bit-exact against ops.prefill.quant / ops.prefill.attention references.

sm_80 has no hardware FP8 cast in triton (`tl.float8e4nv` fails to compile),
so both the E4M3 scale and the E2M1 payload are emulated with plain fp32
arithmetic.  Round-to-nearest-even comes from the classic magic-number trick
(x + 2^23*1.5 - 2^23*1.5), which is exactly RNE for |x| < 2^22 -- the reference
resolves midpoint ties to the even code, so this matches by construction.
"""
import torch
import triton
import triton.language as tl

_MAGIC = 12582912.0  # 2**23 * 1.5


@triton.jit
def _rne(x):
    """Round to nearest, ties to even, without libdevice.

    The magic-number trick ((x + 2**23*1.5) - 2**23*1.5) is algebraically
    simplified away by triton, degrading to round-half-up, so do it explicitly.
    """
    q = tl.floor(x + 0.5)
    tie = (x + 0.5) == q
    odd = (q - 2.0 * tl.floor(q * 0.5)) != 0.0
    return tl.where(tie & odd, q - 1.0, q)


@triton.jit
def _e4m3(v):
    """Quantize a positive fp32 to E4M3 (finite, max 448), emulated."""
    # exact floor(log2(v)) from the fp32 exponent field; tl.log2 rounds and
    # lands one binade off right below powers of two.
    b = v.to(tl.float32).to(tl.int32, bitcast=True)
    e = tl.maximum(((b >> 23) & 0xFF).to(tl.float32) - 127.0, -6.0)
    step = tl.exp2(e - 3.0)                          # 3 mantissa bits
    return tl.minimum(_rne(v / step) * step, 448.0)


@triton.jit
def _fp4_rt(X, Y, E4M3: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + offs).to(tl.float32)
    amax = tl.max(tl.abs(x), 0)
    if E4M3:
        s = _e4m3(tl.maximum(amax, 6.0 * 1.953125e-3) / 6.0)   # 6*2**-9
    else:
        s = tl.exp2(tl.ceil(tl.log2(tl.maximum(amax, 6.0 * 1.1754944e-38) / 6.0)))
    u = tl.minimum(tl.maximum(tl.math.div_rn(x, s), -6.0), 6.0)
    # E2M1 grid: 0, .5, 1, 1.5, 2, 3, 4, 6  ->  step is 0.5 below 2, then 2**(e-1)
    step = tl.maximum(tl.exp2(tl.floor(tl.log2(tl.abs(u))) - 1.0), 0.5)
    q = tl.minimum(tl.maximum(_rne(tl.math.div_rn(u, step)) * step, -6.0), 6.0)
    tl.store(Y + offs, (q * s).to(Y.dtype.element_ty))


def fp4_roundtrip(x, *, block, e4m3_scale):
    """Drop-in for ops.prefill.quant.fp4_roundtrip, one kernel launch."""
    if x.shape[-1] % block:
        raise ValueError('FP4 block alignment')
    xc = x.contiguous()
    y = torch.empty_like(xc)
    n = xc.numel() // block
    if n:
        _fp4_rt[(n,)](xc.view(-1), y.view(-1), E4M3=bool(e4m3_scale),
                      BLOCK=block, num_warps=1)
    return y


@triton.jit
def _fp8_rt(X, Y, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + offs).to(tl.float32)
    amax = tl.maximum(tl.max(tl.abs(x), 0), 1e-4)
    scale = tl.exp2(tl.ceil(tl.log2(amax / 448.0)))
    u = tl.minimum(tl.maximum(x / scale, -448.0), 448.0)
    q = tl.where(u >= 0, _e4m3(tl.abs(u)), -_e4m3(tl.abs(u)))
    tl.store(Y + offs, (q * scale).to(Y.dtype.element_ty))


def fp8_roundtrip(x, block=32):
    if x.shape[-1] % block:
        raise ValueError('FP8 block alignment')
    xc = x.contiguous()
    y = torch.empty_like(xc)
    n = xc.numel() // block
    if n:
        _fp8_rt[(n,)](xc.view(-1), y.view(-1), BLOCK=block, num_warps=1)
    return y


@triton.jit
def _swiglu(G, U, W, Y, n_cols, limit, HAS_W: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    for off in range(0, n_cols, BLOCK):
        idx = off + tl.arange(0, BLOCK)
        m = idx < n_cols
        g = tl.load(G + row * n_cols + idx, mask=m, other=0.).to(tl.float32)
        u = tl.load(U + row * n_cols + idx, mask=m, other=0.).to(tl.float32)
        if limit > 0:
            g = tl.minimum(g, limit)
            u = tl.minimum(tl.maximum(u, -limit), limit)
        y = (g / (1.0 + tl.exp(-g))) * u
        if HAS_W:
            y = y * tl.load(W + row).to(tl.float32)
        tl.store(Y + row * n_cols + idx, y.to(Y.dtype.element_ty), mask=m)


def swiglu(gate, up, limit, route_weight=None):
    g = gate.contiguous()
    u = up.contiguous()
    n = g.shape[-1]
    rows = g.numel() // n
    y = torch.empty_like(g)
    w = None
    if route_weight is not None:
        w = route_weight.contiguous().view(-1)
        if w.numel() != rows:
            return None
    blk = min(triton.next_power_of_2(n), 1024)
    _swiglu[(rows,)](g.view(-1), u.view(-1), w if w is not None else g.view(-1),
                     y.view(-1), n, float(limit), w is not None, BLOCK=blk, num_warps=4)
    return y


@triton.jit
def _rms_split2(X, W0, W1, Y0, Y1, n0, n1, stride, eps, BLOCK: tl.constexpr):
    """One program per (row, half) of a packed [T, n0+n1] projection."""
    row = tl.program_id(0)
    idx = tl.arange(0, BLOCK)
    if tl.program_id(1) == 0:
        n, src, W, dst = n0, X + row * stride, W0, Y0 + row * n0
    else:
        n, src, W, dst = n1, X + row * stride + n0, W1, Y1 + row * n1
    m = idx < n
    x = tl.load(src + idx, mask=m, other=0.0).to(tl.float32)
    inv = tl.rsqrt(tl.sum(x * x, 0) / n.to(tl.float32) + eps)
    w = tl.load(W + idx, mask=m, other=0.0).to(tl.float32)
    tl.store(dst + idx, (x * inv * w).to(dst.dtype.element_ty), mask=m)


def rms_split2(x, w0, w1, eps):
    """RMS-norm both halves of a packed projection in a single launch.

    The packed GEMM that produces x leaves each half strided, which the C++
    rms_norm refuses; normalizing here also replaces two launches with one.
    Arithmetic matches rms_norm_f32w_kernel: fp32 accumulate, fp32 weight.
    """
    n0, n1 = w0.numel(), w1.numel()
    if (x.dim() != 2 or x.stride(1) != 1 or x.shape[1] != n0 + n1
            or x.dtype != torch.bfloat16 or w0.dtype != torch.float32
            or w1.dtype != torch.float32):
        return None
    rows = x.shape[0]
    y0 = torch.empty((rows, n0), device=x.device, dtype=x.dtype)
    y1 = torch.empty((rows, n1), device=x.device, dtype=x.dtype)
    blk = triton.next_power_of_2(max(n0, n1))
    _rms_split2[(rows, 2)](x, w0, w1, y0, y1, n0, n1, x.stride(0), float(eps),
                           BLOCK=blk, num_warps=8)
    return y0, y1


@triton.jit
def _swiglu_packed(GU, Y, n_cols, stride, limit, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    for off in range(0, n_cols, BLOCK):
        idx = off + tl.arange(0, BLOCK)
        m = idx < n_cols
        g = tl.load(GU + row * stride + idx, mask=m, other=0.0).to(tl.float32)
        u = tl.load(GU + row * stride + n_cols + idx, mask=m, other=0.0).to(tl.float32)
        if limit > 0:
            g = tl.minimum(g, limit)
            u = tl.minimum(tl.maximum(u, -limit), limit)
        y = (g / (1.0 + tl.exp(-g))) * u
        tl.store(Y + row * n_cols + idx, y.to(Y.dtype.element_ty), mask=m)


def swiglu_packed(gu, limit):
    """SwiGLU over a packed [T, 2n] gate/up GEMM, without splitting it first."""
    if gu.dim() != 2 or gu.stride(1) != 1 or gu.shape[1] % 2:
        return None
    n = gu.shape[1] // 2
    y = torch.empty((gu.shape[0], n), device=gu.device, dtype=gu.dtype)
    blk = min(triton.next_power_of_2(n), 1024)
    _swiglu_packed[(gu.shape[0],)](gu, y, n, gu.stride(0), float(limit),
                                   BLOCK=blk, num_warps=4)
    return y
