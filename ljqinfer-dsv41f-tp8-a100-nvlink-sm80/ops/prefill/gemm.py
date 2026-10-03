"""Explicit prefill packed-GEMM baseline, not a decode dispatcher.

Canonical ABI: FP8 E4M3 weights with 32x32 E8M0 scales; FP4 E2M1
low-nibble-first int8 bytes with per-row K/32 E8M0 scales. Both quantize
activations to FP8/K32. Decode only a bounded output-row tile, never keep a
whole-model dequantized copy. Replace this operator with a fused packed GEMM
without changing weight/state contracts. This baseline is NOT optimized.
"""
import torch
import torch.nn.functional as F


def activation_fp8(x):
    if x.shape[-1] % 32:
        raise ValueError('activation K must be a multiple of 32')
    z = x.float().unflatten(-1, (-1, 32))
    s = (z.abs().amax(-1, keepdim=True).clamp_min(1e-4) / 448).log2().ceil().exp2()
    # Keep the dequantized value in FP32: no extra BF16 round between quant
    # and GEMM. Weight and activation power-of-two scales remain exact.
    return ((z / s).clamp(-448, 448).to(torch.float8_e4m3fn).float() * s).flatten(-2)


def packed_linear(x, weight, scale, *, output_tile=256, activation_prepare=None):
    if x.dtype != torch.bfloat16:
        raise TypeError('released quantized linear requires BF16 activation')
    if x.ndim < 2 or weight.ndim != 2 or scale.ndim != 2:
        raise ValueError('matrix geometry')
    if output_tile <= 0 or output_tile % 32:
        raise ValueError('output tile must be positive and 32-aligned')
    if weight.dtype not in (torch.int8, torch.float8_e4m3fn):
        raise TypeError('expected canonical packed FP4 or FP8')
    if scale.dtype != torch.uint8:
        raise TypeError('scales must be canonical E8M0 bytes')
    n, stored_k = weight.shape
    k = stored_k * (2 if weight.dtype == torch.int8 else 1)
    bn = 1 if weight.dtype == torch.int8 else 32
    if k % 32 or x.shape[-1] != k or scale.shape != ((n+bn-1)//bn,k//32):
        raise ValueError('weight/activation/scale geometry')
    if x.device != weight.device or x.device != scale.device:
        raise ValueError('rank-local tensors must be on the same device')
    a = activation_fp8(x) if activation_prepare is None else activation_prepare(x)
    out = torch.empty((*x.shape[:-1],n),device=x.device,dtype=torch.bfloat16)
    table = torch.tensor([0.,.5,1.,1.5,2.,3.,4.,6.],device=x.device)
    for start in range(0,n,output_tile):
        end = min(n,start+output_tile)
        if bn == 1:
            b = weight[start:end].view(torch.uint8)
            code = torch.stack((b & 15, b >> 4),-1).flatten(-2).long()
            v = table[code & 7] * torch.where(code & 8 != 0,-1.,1.)
            s = (scale[start:end].float()-127).exp2()
        else:
            v = weight[start:end].float()
            s = (scale[start//32:(end+31)//32].float()-127).exp2()
            s = s.repeat_interleave(32,0)[:end-start]
        w = (v.unflatten(-1,(-1,32))*s[...,None]).flatten(-2)
        out[...,start:end] = F.linear(a,w).bfloat16()
    return out


def grouped_linear(x, weight):
    """[T,G,K] @ [G,N,K], FP32 accumulation and one output rounding.

    Process one group at a time so TP group count cannot select a different
    batched BF16 reduction path. Temporary FP32 weights are bounded to a group.
    This explicit baseline is a replacement boundary for fused grouped GEMM.
    """
    if x.ndim != 3 or weight.ndim != 3:
        raise ValueError('grouped GEMM requires rank-three tensors')
    if x.shape[1] != weight.shape[0] or x.shape[2] != weight.shape[2]:
        raise ValueError('grouped GEMM geometry')
    if x.dtype != weight.dtype or x.dtype not in (torch.bfloat16, torch.float32):
        raise TypeError('grouped GEMM requires matching BF16 or FP32 tensors')
    if x.device != weight.device:
        raise ValueError('grouped GEMM device mismatch')
    out = x.new_empty((x.shape[0], x.shape[1], weight.shape[1]))
    for group in range(x.shape[1]):
        out[:, group] = F.linear(x[:, group].float().contiguous(),
                                weight[group].float().contiguous()).to(x.dtype)
    return out


class PrefillLinear:
    """Bind canonical rank-local tensors; no implicit FP32 fallback for bytes."""
    def __init__(self, weights):
        self.weights = weights
        self.activation_preparers = {}
        self.projection_workspace = None

    def fused(self, names, x):
        """Concatenated projection: one GEMV instead of len(names) launches.

        Falls back to a cat of the individual projections when the workspace
        cannot take the packed path, so the result is defined for any weights.
        """
        ws = self.projection_workspace
        if ws is not None:
            parts = []
            for n in names:
                w = self.weights[n+'.weight']
                if w.dtype != torch.float8_e4m3fn:
                    parts = None
                    break
                parts.append((w, self.weights[n+'.scale']))
            out = ws.fused(x, parts) if parts else None
            if out is not None:
                return out
        return torch.cat([self(n, x) for n in names], -1)

    def __call__(self, name, x):
        w = self.weights[name+'.weight']
        if w.dtype in (torch.int8, torch.float8_e4m3fn):
            if self.projection_workspace is not None:
                return self.projection_workspace(x, w, self.weights[name+'.scale'])
            return packed_linear(x,w,self.weights[name+'.scale'],
                                 activation_prepare=self.activation_preparers.get(name))
        if w.dtype not in (torch.float32, torch.bfloat16):
            raise TypeError('unsupported released linear weight dtype')
        return F.linear(x.to(w.dtype),w)
