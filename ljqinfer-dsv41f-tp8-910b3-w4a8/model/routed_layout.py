"""CPU preparation of routed signed INT4 for decode CANN HP GMM.
Physical storage: [E, Kpad/64, N, 32 packed bytes]. No device work.
The returned int32 shape is the descriptor used by CANN, not a dense matrix.
"""
import torch


def pack_projection(weight, scale):
    if weight.device.type != 'cpu' or scale.device.type != 'cpu':
        raise ValueError('offline preparation requires CPU tensors')
    if weight.dtype != torch.int8 or not weight.is_contiguous():
        raise ValueError('expected contiguous checkpoint INT4 byte storage')
    e, k = weight.shape[0], weight.shape[-1]
    n = weight.numel() // (e*k) * 2
    if n % 64 or k % 32 or tuple(scale.shape[:1]) != (e,) or scale.numel() != e*n:
        raise ValueError('invalid expert projection geometry')
    kp = (k+63)//64*64
    ws = scale.float().reshape(e,n)
    if not torch.isfinite(ws).all():
        raise ValueError('nonfinite scale')
    src = weight.view(torch.uint8).reshape(e,n//2,k)
    nz = torch.zeros((e,kp//64,n,32),dtype=torch.uint8)
    bias = torch.empty((e,n),dtype=torch.float32)
    # Bound temporary memory to one expert, not a dense whole-model expansion.
    for expert in range(e):
        x = src[expert]
        z = torch.zeros((n,kp//2),dtype=torch.uint8)
        z[0::2,:k//2] = (x[:,0::2]&15) | ((x[:,1::2]&15)<<4)
        z[1::2,:k//2] = (x[:,0::2]>>4) | (x[:,1::2]&240)
        nz[expert].copy_(z.reshape(n,kp//64,32).permute(1,0,2))
        low=(x&15).to(torch.int32); high=(x>>4).to(torch.int32)
        low=torch.where(low>=8,low-16,low)
        high=torch.where(high>=8,high-16,high)
        sums=torch.empty(n,dtype=torch.int64)
        sums[0::2]=low.sum(1);sums[1::2]=high.sum(1)
        bias[expert]=sums.float()*8*ws[expert]
    return nz.view(torch.int32).reshape(e,kp,n//8), ws.contiguous(), bias
