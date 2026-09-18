"""CUDA leaves for the frozen Qwen/DFlash ABI; no NPU or runtime selector.

Explicit BF16 casts preserve the original operator rounding boundaries.
Recurrent carry rounds to the pending snapshot dtype after each token output.
"""
import torch
import triton as tr
import triton.language as tl
from triton.language.extra.cuda import libdevice


@tr.jit
def _pack_verify_rows(X, O, M: tl.constexpr, D: tl.constexpr,
                      S: tl.constexpr, B: tl.constexpr):
    i = tl.program_id(0) * B + tl.arange(0, B)
    row, col = i // D, i % D
    x = tl.load(X + row * S + col, (row < M) & (i < 32 * D), 0)
    tl.store(O + i, x, i < 32 * D)


def pack_verify_rows(x):
    """Fuse padding and strided row copy without changing fixed-M32 GEMM."""
    if (x.ndim != 2 or x.stride(1) != 1 or
            not 0 < x.shape[0] <= 32 or x.shape[1] <= 0):
        raise ValueError('verify packing requires 1..32 rows and contiguous columns')
    out = torch.empty((32, x.shape[1]), dtype=x.dtype, device=x.device)
    _pack_verify_rows[(tr.cdiv(out.numel(), 512),)](
        x, out, x.shape[0], x.shape[1], x.stride(0), 512)
    return out


@tr.jit
def _norm(X, W, R, Z, O, S, D: tl.constexpr, H: tl.constexpr,
          XS0: tl.constexpr, XS1: tl.constexpr, ZS0: tl.constexpr, ZS1: tl.constexpr,
          EPS: tl.constexpr, MODE: tl.constexpr, ADD: tl.constexpr,
          BLOCK: tl.constexpr):
    row = tl.program_id(0)
    j = tl.arange(0, BLOCK)
    x = tl.load(X + (row // H) * XS0 + (row % H) * XS1 + j, j < D, 0).to(tl.float32)
    if ADD:
        r = tl.load(R + row * D + j, j < D, 0).to(tl.float32)
        x = (x + r).to(X.dtype.element_ty).to(tl.float32)
        tl.store(S + row * D + j, x, j < D)
    ss = tl.sum(x * x, 0)
    if MODE == 1 or MODE == 3:
        inv = tl.rsqrt(tl.maximum(ss, EPS))
        if MODE == 1:
            inv = inv.to(X.dtype.element_ty).to(tl.float32)
        y = x * inv
    else:
        inv = tl.rsqrt(ss / D + EPS)
        w = tl.load(W + j, j < D, 0).to(tl.float32)
        if MODE == 0:
            w = (w + 1.).to(W.dtype.element_ty).to(tl.float32)
        y = (x * inv * w).to(X.dtype.element_ty).to(tl.float32)
        if MODE == 2:
            z = tl.load(Z + (row // H) * ZS0 + (row % H) * ZS1 + j, j < D, 0).to(tl.float32)
            y = y * (z / (1. + tl.exp(-z)))
    tl.store(O + row * D + j, y, j < D)


def norm(x, weight=None, eps=1e-6, mode=0, residual=None, z=None):
    shape = x.shape
    if x.stride(-1) != 1:
        x = x.contiguous()
    if weight is not None:
        weight = weight.contiguous()
    if z is not None and z.stride(-1) != 1:
        z = z.contiguous()
    d = x.shape[-1]
    # Flatten leading token dimensions, preserving the packed head stride.
    h = x.shape[-2] if x.ndim >= 3 else 1
    rows = x.numel() // d
    if x.ndim > 3:
        x = x.reshape(-1, h, d)
    xs0 = x.stride(-3) if x.ndim == 3 else x.stride(0) if x.ndim == 2 else d
    xs1 = x.stride(-2) if x.ndim == 3 else d
    if residual is not None:
        residual = residual.contiguous()
    if z is not None and z.ndim > 3:
        z = z.reshape(-1, h, d)
    zs0 = z.stride(-3) if z is not None and z.ndim == 3 else z.stride(0) if z is not None and z.ndim == 2 else d
    zs1 = z.stride(-2) if z is not None and z.ndim == 3 else d
    out = torch.empty(x.shape, device=x.device, dtype=x.dtype)
    summed = torch.empty_like(out) if residual is not None else out
    _norm[(rows,)](x, weight, residual, z, out, summed, d, h, xs0, xs1, zs0, zs1,
                  eps, mode, residual is not None, tr.next_power_of_2(d), enable_fp_fusion=False)
    return (out.reshape(shape), summed.reshape(shape)) if residual is not None else out.reshape(shape)


@tr.jit
def _conv(X, BASE, W, O, P, C: tl.constexpr, T: tl.constexpr, K: tl.constexpr,
          XB: tl.constexpr, XT: tl.constexpr, WC: tl.constexpr, WK: tl.constexpr,
          PREFILL: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    b = tl.program_id(1)
    t, c = i // C, i % C
    ok = i < T * C
    acc = tl.full((BLOCK,), 0, tl.float32)
    for tap in range(K):
        s = t + tap - K + 1
        a = tl.load(BASE + b * (K - 1) * C + (s + K - 1) * C + c, ok & (s < 0), 0).to(tl.float32)
        a += tl.load(X + b * XB + s * XT + c, ok & (s >= 0), 0).to(tl.float32)
        w = tl.load(W + c * WC + tap * WK, ok, 0).to(tl.float32)
        prod = (a * w).to(X.dtype.element_ty).to(tl.float32)
        acc = acc + prod
        if PREFILL:
            acc = acc.to(X.dtype.element_ty).to(tl.float32)
    tl.store(O + b * T * C + i, acc, ok)
    if not PREFILL:
        a = tl.load(X + b * XB + t * XT + c, ok, 0)
        tl.store(P + b * T * C + i, a, ok)


def conv(x, base, weight, pending=None):
    b, t, c = x.shape
    prefill = pending is None
    if prefill:
        weight = weight.reshape(c, -1)
        k, wc, wk = weight.shape[1], weight.stride(0), weight.stride(1)
    else:
        k, wc, wk = weight.shape[0], weight.stride(1), weight.stride(0)
    out = torch.empty((b, t, c), device=x.device, dtype=x.dtype)
    _conv[(tr.cdiv(t * c, 256), b)](x, base, weight, out, pending, c, t, k,
            x.stride(0), x.stride(1), wc, wk, prefill, 256, enable_fp_fusion=False)
    if prefill:
        # Separate launch: no cross-CTA race between base readers and writers.
        if t >= k - 1:
            base.copy_(x[:, -(k - 1):])
        else:
            base.copy_(torch.cat((base[:, t:], x), dim=1))
    return out


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


@tr.jit
def _recurrent(Q, K, V, G, BETA, BASE, STATES, IDX, O,
               HK: tl.constexpr, HV: tl.constexpr, DK: tl.constexpr, DV: tl.constexpr,
               QS: tl.constexpr, KS: tl.constexpr, VS: tl.constexpr,
               GS: tl.constexpr, BS: tl.constexpr, BV: tl.constexpr, BK: tl.constexpr):
    vi = tl.program_id(0) * BV + tl.arange(0, BV)
    bh = tl.program_id(1)
    batch, vh = bh // HV, bh % HV
    kh = vh // (HV // HK)
    ki = tl.arange(0, BK)
    mask = (vi[:, None] < DV) & (ki[None, :] < DK)
    s = tl.load(BASE + ((batch * HV + vh) * DV + vi[:, None]) * DK + ki[None, :], mask, 0).to(tl.float32)
    for t in range(8):
        row = batch * 8 + t
        q = tl.load(Q + row * QS + kh * DK + ki, ki < DK, 0).to(tl.float32) * (DK ** -0.5)
        k = tl.load(K + row * KS + kh * DK + ki, ki < DK, 0).to(tl.float32)
        v = tl.load(V + row * VS + vh * DV + vi, vi < DV, 0).to(tl.float32)
        g = tl.exp(tl.load(G + row * GS + vh).to(tl.float32))
        beta = tl.sigmoid(tl.load(BETA + row * BS + vh).to(tl.float32)).to(BETA.dtype.element_ty).to(tl.float32)
        s = s * g
        pred = tl.sum(s * k[None, :], 1)
        delta = (v - pred) * beta
        s = s + delta[:, None] * k[None, :]
        y = tl.sum(s * q[None, :], 1)
        slot = tl.load(IDX + row)
        tl.store(STATES + ((slot * HV + vh) * DV + vi[:, None]) * DK + ki[None, :], s, mask)
        tl.store(O + (row * HV + vh) * DV + vi, y, vi < DV)
        s = s.to(STATES.dtype.element_ty).to(tl.float32)


def recurrent(q, k, v, g, beta, base, state, indices):
    rows, hk, dk = q.shape
    hv, dv = v.shape[1:]
    out = torch.empty(v.shape, dtype=v.dtype, device=v.device)
    _recurrent[(tr.cdiv(dv, 16), rows // 8 * hv)](q, k, v, g, beta, base, state, indices, out,
        hk, hv, dk, dv, q.stride(0), k.stride(0), v.stride(0), g.stride(0), beta.stride(0),
        16, tr.next_power_of_2(dk), num_warps=4, enable_fp_fusion=False)
    return out


@tr.jit
def _swiglu(G, U, O, N: tl.constexpr, OUT_N: tl.constexpr, D: tl.constexpr, GS: tl.constexpr, US: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row, col = i // D, i % D
    g = tl.load(G + row * GS + col, i < N, 0).to(tl.float32)
    u = tl.load(U + row * US + col, i < N, 0).to(tl.float32)
    a = (g * tl.sigmoid(g)).to(G.dtype.element_ty).to(tl.float32)
    tl.store(O + i, a * u, i < OUT_N)


def swiglu(gate, up):
    shape = gate.shape
    g, u = gate.reshape(-1, shape[-1]), up.reshape(-1, shape[-1])
    out = torch.empty(shape, device=g.device, dtype=g.dtype)
    _swiglu[(tr.cdiv(g.numel(), 256),)](g, u, out, g.numel(), out.numel(), g.shape[-1], g.stride(0), u.stride(0), 256)
    return out


@tr.jit
def _norm_rope_row(X, W, COS, SIN, O, ROW, H: tl.constexpr, D: tl.constexpr,
                   S0: tl.constexpr, S1: tl.constexpr, FS: tl.constexpr,
                   RD: tl.constexpr, EPS: tl.constexpr, BLOCK: tl.constexpr):
    j = tl.arange(0, BLOCK)
    token, head = ROW // H, ROW % H
    offset = token * S0 + head * S1
    x = tl.load(X + offset + j, j < D, 0).to(tl.float32)
    inv = tl.rsqrt(tl.sum(x*x, 0)/D + EPS)
    w = tl.load(W+j, j<D, 0)
    w = (w.to(tl.float32)+1.).to(W.dtype.element_ty).to(tl.float32)
    y = (x*inv*w).to(X.dtype.element_ty).to(tl.float32)
    if RD > 0:
        rj = tl.where(j < RD//2, j+RD//2, j-RD//2)
        xr = tl.load(X + offset+rj, j<RD, 0).to(tl.float32)
        wr = tl.load(W+rj, j<RD, 0)
        wr = (wr.to(tl.float32)+1.).to(W.dtype.element_ty).to(tl.float32)
        yr = (xr*inv*wr).to(X.dtype.element_ty).to(tl.float32)
        yr = tl.where(j<RD//2, -yr, yr)
        # Frozen ABI keeps a half-width table; both halves share one entry.
        jh = tl.where(j < RD//2, j, j - RD//2)
        co = tl.load(COS+token*FS+jh, j<RD, 0).to(tl.float32)
        si = tl.load(SIN+token*FS+jh, j<RD, 0).to(tl.float32)
        y = tl.where(j<RD, y*co+yr*si, y)
    tl.store(O+ROW*D+j,y,j<D)


@tr.jit
def _qk_norm_rope(Q,K,QW,KW,COS,SIN,OQ,OK,
                  T: tl.constexpr, HQ: tl.constexpr, HK: tl.constexpr, D: tl.constexpr,
                  QS0: tl.constexpr, QS1: tl.constexpr, KS0: tl.constexpr, KS1: tl.constexpr,
                  FS: tl.constexpr, RD: tl.constexpr, EPS: tl.constexpr, BLOCK: tl.constexpr):
    row=tl.program_id(0)
    if row<T*HQ:
        _norm_rope_row(Q,QW,COS,SIN,OQ,row,HQ,D,QS0,QS1,FS,RD,EPS,BLOCK)
    else:
        _norm_rope_row(K,KW,COS,SIN,OK,row-T*HQ,HK,D,KS0,KS1,FS,RD,EPS,BLOCK)


def qk_norm_rope(q,k,qw,kw,frequencies,rotary_dim,eps):
    t,hq,d=q.shape
    hk=k.shape[1]
    cos,sin=frequencies if rotary_dim>0 else (q,q)
    oq=torch.empty(q.shape,device=q.device,dtype=q.dtype)
    ok=torch.empty(k.shape,device=k.device,dtype=k.dtype)
    _qk_norm_rope[(t*(hq+hk),)](q,k,qw,kw,cos,sin,oq,ok,t,hq,hk,d,
        q.stride(0),q.stride(1),k.stride(0),k.stride(1),cos.stride(0),
        rotary_dim,eps,tr.next_power_of_2(d),enable_fp_fusion=False)
    return oq,ok


@tr.jit
def _norm_pack(X, W, R, Z, O, S, D: tl.constexpr, H: tl.constexpr,
          XS0: tl.constexpr, XS1: tl.constexpr, ZS0: tl.constexpr, ZS1: tl.constexpr,
          EPS: tl.constexpr, MODE: tl.constexpr, ADD: tl.constexpr,
          BLOCK: tl.constexpr, ROWS: tl.constexpr):
    row = tl.program_id(0)
    j = tl.arange(0, BLOCK)
    x = tl.load(X + (row // H) * XS0 + (row % H) * XS1 + j, (j < D) & (row < ROWS), 0).to(tl.float32)
    x = x.to(O.dtype.element_ty).to(tl.float32)
    if ADD:
        r = tl.load(R + row * D + j, (j < D) & (row < ROWS), 0).to(tl.float32)
        x = (x + r).to(O.dtype.element_ty).to(tl.float32)
        tl.store(S + row * D + j, x, (j < D) & (row < ROWS))
    ss = tl.sum(x * x, 0)
    if MODE == 1 or MODE == 3:
        inv = tl.rsqrt(tl.maximum(ss, EPS))
        if MODE == 1:
            inv = inv.to(O.dtype.element_ty).to(tl.float32)
        y = x * inv
    else:
        inv = tl.rsqrt(ss / D + EPS)
        w = tl.load(W + j, (j < D) & (row < ROWS), 0).to(tl.float32)
        if MODE == 0:
            w = (w + 1.).to(W.dtype.element_ty).to(tl.float32)
        y = (x * inv * w).to(O.dtype.element_ty).to(tl.float32)
        if MODE == 2:
            z = tl.load(Z + (row // H) * ZS0 + (row % H) * ZS1 + j, (j < D) & (row < ROWS), 0).to(tl.float32)
            y = y * (z / (1. + tl.exp(-z)))
    tl.store(O + row * D + j, y, j < D)


def norm_verify_packed(x, weight, eps=1e-6, residual=None):
    """BF16 RMSNorm to M32; FP32 reduced input rounds before residual addition."""
    tensors = (weight,) if residual is None else (weight, residual)
    if (x.device.type != 'cuda' or
            x.dtype not in ((torch.bfloat16,) if residual is None else
                            (torch.bfloat16, torch.float32))):
        raise ValueError('verify norm requires BF16 input or FP32 reduced input with residual')
    if any(t.device.type != 'cuda' or t.device != x.device or
           t.dtype != torch.bfloat16 for t in tensors):
        raise ValueError('verify norm requires same-device CUDA BF16 payloads')
    if (x.ndim != 2 or not 0 < x.shape[0] <= 32 or
            x.shape[1] <= 0 or x.stride(1) != 1):
        raise ValueError('verify norm requires 1..32 rows and contiguous columns')
    if weight.shape != (x.shape[1],) or not weight.is_contiguous():
        raise ValueError('verify norm requires a contiguous delta weight vector')
    if residual is not None and (residual.shape != x.shape or not residual.is_contiguous()):
        raise ValueError('verify norm residual must be contiguous with real input shape')
    if not eps > 0:
        raise ValueError('verify norm epsilon must be positive')
    m, d = x.shape
    y = torch.empty((32, d), device=x.device, dtype=torch.bfloat16)
    summed = torch.empty((m, d), device=x.device, dtype=torch.bfloat16) if residual is not None else y
    _norm_pack[(32,)](x, weight, residual, None, y, summed, d, 1,
                     x.stride(0), d, d, d, eps, 0, residual is not None,
                     tr.next_power_of_2(d), m, enable_fp_fusion=False)
    return (y, summed) if residual is not None else y


def swiglu_verify_packed(gate, up):
    g = gate.reshape(-1, gate.shape[-1])
    u = up.reshape_as(g)
    assert 0 < g.shape[0] <= 32 and g.stride(1) == u.stride(1) == 1
    out = torch.empty((32, g.shape[1]), device=g.device, dtype=g.dtype)
    _swiglu[tr.cdiv(out.numel(), 256),](g, u, out, g.numel(), out.numel(), g.shape[1], g.stride(0), u.stride(0), 256)
    return out


def gated_norm_verify_packed(x, z, weight, eps=1e-06):
    m, h, d = x.shape
    assert 0 < m <= 32 and z.shape == x.shape and (x.stride(-1) == z.stride(-1) == 1)
    out = torch.empty((32, h * d), device=x.device, dtype=x.dtype)
    _norm_pack[32 * h,](x, weight, None, z, out, out, d, h, x.stride(0), x.stride(1), z.stride(0), z.stride(1), eps, 2, False, tr.next_power_of_2(d), m * h, enable_fp_fusion=False)
    return out


@tr.jit
def _logical_kv_gather(K, V, S, KO, VO, L: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    i = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    token, dim = i // 256, i % 256
    start = tl.load(S + row)
    # Masked tail deliberately matches the old clamped gather as well.
    src = tl.minimum(tl.where(token < start, token, L - 8 + token - start), L - 1)
    offset = row * L * 256 + src * 256 + dim
    k = tl.load(K + offset, i < L * 256, 0)
    v = tl.load(V + offset, i < L * 256, 0)
    tl.store(KO + row * L * 256 + i, k, i < L * 256)
    tl.store(VO + row * L * 256 + i, v, i < L * 256)






@tr.jit
def _logical_mask_pack(S,O,L:tl.constexpr,SS:tl.constexpr,BLOCK:tl.constexpr):
 r=tl.program_id(0);i=tl.program_id(1)*BLOCK+tl.arange(0,BLOCK)
 s=tl.load(S+r*SS);bits=tl.full((BLOCK,),0,tl.int32)
 for bit in tl.static_range(8):
  idx=i*8+bit
  allowed=(idx%L)<(s+idx//L+1)
  bits=bits|(allowed.to(tl.int32)<<bit)
 tl.store(O+r*L+i,bits.to(tl.uint8),i<L)
def logical_mask_packed(start,length):
 o=torch.empty((start.numel(),length),device=start.device,dtype=torch.uint8)
 _logical_mask_pack[(start.numel(),tr.cdiv(length,256))](start,o,length,start.stride(0),256)
 return o

def logical_kv_gather(key, value, start):
    b, length = key.shape[:2]
    ordered_k, ordered_v = torch.empty_like(key), torch.empty_like(value)
    _logical_kv_gather[(b, tr.cdiv(length * 256, 1024))](
        key, value, start, ordered_k, ordered_v, length, 1024)
    return ordered_k, ordered_v


@tr.jit
def _gdn_conv_norm(X,B,W,P,Q,K,V,XB:tl.constexpr,XT:tl.constexpr):
 row=tl.program_id(0);head=tl.program_id(1)
 b=row//8;t=row%8;j=tl.arange(0,128);c=head*128+j
 acc=tl.full((128,),0,tl.float32)
 for tap in range(4):
  s=t+tap-3
  a=tl.load(B+b*3*2560+(s+3)*2560+c,s<0,0).to(tl.float32)
  a+=tl.load(X+b*XB+s*XT+c,s>=0,0).to(tl.float32)
  w=tl.load(W+tap*2560+c).to(tl.float32)
  acc=acc+(a*w).to(tl.bfloat16).to(tl.float32)
 z=acc.to(tl.bfloat16).to(tl.float32)
 z=(z/(1.+libdevice.exp(-z))).to(tl.bfloat16).to(tl.float32)
 raw=tl.load(X+b*XB+t*XT+c)
 tl.store(P+row*2560+c,raw)
 if head<8:
  inv=tl.rsqrt(tl.maximum(tl.sum(z*z,0),1e-6)).to(tl.bfloat16).to(tl.float32)
  z=z*inv
  if head<4:tl.store(Q+row*512+head*128+j,z)
  else:tl.store(K+row*512+(head-4)*128+j,z)
 else:tl.store(V+row*1536+(head-8)*128+j,z)
def gdn_conv_norm(x,base,w,pending):
 b=x.shape[0]
 q=torch.empty((b*8,4,128),device=x.device,dtype=x.dtype);k=torch.empty_like(q)
 v=torch.empty((b*8,12,128),device=x.device,dtype=x.dtype)
 _gdn_conv_norm[(b*8,20)](x,base,w,pending,q,k,v,x.stride(0),x.stride(1),num_warps=4,enable_fp_fusion=False)
 return q,k,v


@tr.jit
def _gdn_gate_prepare(AB,D,T,G,B,N:tl.constexpr,BLOCK:tl.constexpr):
 i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK);h=i%12;r=i//12
 a=tl.load(AB+r*24+h,i<N,0).to(tl.float32)
 bias=tl.load(T+h).to(tl.float32);decay=tl.load(D+h).to(tl.float32)
 x=a+bias
 soft=tl.where(x>20.,x,libdevice.log1p(libdevice.exp(x)))
 tl.store(G+i,decay*soft,i<N)
 beta=tl.load(AB+r*24+12+h,i<N,0)
 tl.store(B+i,beta,i<N)
def gdn_gate_prepare(ab,decay,bias):
 g=torch.empty(ab.shape[:-1]+(12,),device=ab.device,dtype=torch.float32)
 beta=torch.empty_like(g,dtype=ab.dtype)
 _gdn_gate_prepare[(tr.cdiv(g.numel(),128),)](ab,decay,bias,g,beta,g.numel(),128,enable_fp_fusion=False)
 return g,beta


# DFlash uses effective RMS weights and separately rounded RoPE products.
@tr.jit
def _dflash_norm_rope_row(X, W, COS, SIN, O, ROW, H: tl.constexpr, D: tl.constexpr,
                   S0: tl.constexpr, S1: tl.constexpr, FS: tl.constexpr,
                   RD: tl.constexpr, EPS: tl.constexpr, BLOCK: tl.constexpr):
    j = tl.arange(0, BLOCK)
    token, head = ROW // H, ROW % H
    offset = token * S0 + head * S1
    x = tl.load(X + offset + j, j < D, 0).to(tl.float32)
    inv = tl.rsqrt(tl.sum(x*x, 0)/D + EPS)
    w = tl.load(W+j, j<D, 0)
    w = w.to(tl.float32)
    y = (x*inv*w).to(X.dtype.element_ty).to(tl.float32)
    if RD > 0:
        rj = tl.where(j < RD//2, j+RD//2, j-RD//2)
        xr = tl.load(X + offset+rj, j<RD, 0).to(tl.float32)
        wr = tl.load(W+rj, j<RD, 0)
        wr = wr.to(tl.float32)
        yr = (xr*inv*wr).to(X.dtype.element_ty).to(tl.float32)
        yr = tl.where(j<RD//2, -yr, yr)
        # Frozen ABI keeps a half-width table; both halves share one entry.
        jh = tl.where(j < RD//2, j, j - RD//2)
        co = tl.load(COS+token*FS+jh, j<RD, 0).to(tl.float32)
        si = tl.load(SIN+token*FS+jh, j<RD, 0).to(tl.float32)
        a = (y*co).to(X.dtype.element_ty).to(tl.float32)
        b = (yr*si).to(X.dtype.element_ty).to(tl.float32)
        y = tl.where(j<RD, a+b, y)
    tl.store(O+ROW*D+j,y,j<D)
@tr.jit
def _dflash_qk_norm_rope(Q,K,QW,KW,COS,SIN,OQ,OK,
                  T: tl.constexpr, HQ: tl.constexpr, HK: tl.constexpr, D: tl.constexpr,
                  QS0: tl.constexpr, QS1: tl.constexpr, KS0: tl.constexpr, KS1: tl.constexpr,
                  FS: tl.constexpr, RD: tl.constexpr, EPS: tl.constexpr, BLOCK: tl.constexpr):
    row=tl.program_id(0)
    if row<T*HQ:
        _dflash_norm_rope_row(Q,QW,COS,SIN,OQ,row,HQ,D,QS0,QS1,FS,RD,EPS,BLOCK)
    else:
        _dflash_norm_rope_row(K,KW,COS,SIN,OK,row-T*HQ,HK,D,KS0,KS1,FS,RD,EPS,BLOCK)
def dflash_qk_norm_rope(q,k,qw,kw,frequencies,rotary_dim,eps):
    t,hq,d=q.shape
    hk=k.shape[1]
    cos,sin=frequencies if rotary_dim>0 else (q,q)
    oq=torch.empty(q.shape,device=q.device,dtype=q.dtype)
    ok=torch.empty(k.shape,device=k.device,dtype=k.dtype)
    _dflash_qk_norm_rope[(t*(hq+hk),)](q,k,qw,kw,cos,sin,oq,ok,t,hq,hk,d,
        q.stride(0),q.stride(1),k.stride(0),k.stride(1),cos.stride(0),
        rotary_dim,eps,tr.next_power_of_2(d),enable_fp_fusion=False)
    return oq,ok

@tr.jit
def _attention_gate_pack(Y,G,O,M:tl.constexpr,H:tl.constexpr,D:tl.constexpr,Y0:tl.constexpr,Y1:tl.constexpr,Y2:tl.constexpr,Y3:tl.constexpr,G0:tl.constexpr,G1:tl.constexpr,G2:tl.constexpr,G3:tl.constexpr,B:tl.constexpr):
 i=tl.program_id(0)*B+tl.arange(0,B)
 row=i//(H*D); h=i//D%H; d=i%D
 b=row//8; q=row%8
 y=tl.load(Y+b*Y0+q*Y1+h*Y2+d*Y3,(row<M)&(i<32*H*D),0).to(tl.float32)
 g=tl.load(G+b*G0+q*G1+h*G2+d*G3,(row<M)&(i<32*H*D),0).to(tl.float32)
 sig=(1./(1.+tl.exp(-g))).to(tl.bfloat16).to(tl.float32)
 tl.store(O+i,y*sig,i<32*H*D)
def attention_gate_pack(y,g):
 assert y.shape==g.shape and y.ndim==4 and y.shape[1]==8 and y.shape[0] in (1,2,3,4)
 assert y.dtype==g.dtype==torch.bfloat16
 b,q,h,d=y.shape
 o=torch.empty((32,h*d),device=y.device,dtype=y.dtype)
 _attention_gate_pack[(tr.cdiv(o.numel(),512),)](y,g,o,b*q,h,d,*y.stride(),*g.stride(),512,enable_fp_fusion=False)
 return o


@tr.jit
def _dflash_path_select(S,C,O,S0:tl.constexpr,S1:tl.constexpr,S2:tl.constexpr,S3:tl.constexpr,C0:tl.constexpr,C1:tl.constexpr,C2:tl.constexpr):
    b=tl.program_id(0)
    j=tl.arange(0,16)
    prev=tl.full((),0,tl.int32)
    for step in range(7):
        row=tl.load(S+b*S0+step*S1+prev*S2+j*S3).to(tl.float32)
        highest=tl.max(row,0)
        prev=tl.min(tl.where(row==highest,j,2147483647),0)
        token=tl.load(C+b*C0+step*C1+prev*C2)
        tl.store(O+b*7+step,token)

def dflash_path_select(scores,candidate):
    assert scores.shape==(candidate.shape[0],7,16,16)
    assert candidate.shape[1:]==(7,16)
    out=torch.empty((candidate.shape[0],7),device=candidate.device,dtype=candidate.dtype)
    _dflash_path_select[(candidate.shape[0],)](scores,candidate,out,*scores.stride(),*candidate.stride(),num_warps=4)
    return out
