"""Routed MoE ABI, SM80, BF16 activations, FP32 accumulation/output.

GU: uint8 [E,2*I,H/2], scales FP16 [E,2*I,H/G]; gate then up.
Down INT4: uint8 [E,H,I/2], scales FP16 [E,H,I/G].
Down FP8: raw E4M3FN bytes [E,H,I], scales F32 [E,ceil(H/128),ceil(I/128)].
INT4 signed two's-complement, even K low nibble. G=64 or 128.
IDs int32 [T,topk], weights FP32 same shape, already routed/scaled.
Invalid IDs are ignored. Duplicates are additive. Output is rank-local,
pre all-reduce, excludes shared expert. Caller owns all workspace/output.
Create/validate outside capture; warm all shape variants before capture.
"""
from dataclasses import dataclass
import torch
import triton
import triton.language as tl


# Shared allocation/dispatch cutoff; decode workspaces stay compact.
DENSE_PREFILL_MIN_TOKENS = 128


@triton.jit
def int4_values(P, S, e, n, k, N:tl.constexpr, K:tl.constexpr, G:tl.constexpr, mask):
    b=tl.load(P+e*N*(K//2)+n*(K//2)+k//2,mask,0).to(tl.int32)
    q=(b>>((k%2)*4))&15
    q=tl.where(q>=8,q-16,q)
    s=tl.load(S+e*N*(K//G)+n*(K//G)+k//G,mask,0).to(tl.float32)
    return (q.to(tl.float32)*s).to(tl.bfloat16)


@triton.jit
def fp8_values(P, S, e, n, k, N:tl.constexpr, K:tl.constexpr, mask):
    # Manual E4M3FN decode avoids requiring native SM89 FP8 instructions.
    b=tl.load(P+e*N*K+n*K+k,mask,0).to(tl.int32)
    exp=(b>>3)&15
    mant=b&7
    val=tl.where(exp==0,mant*0.001953125,(1.+mant*0.125)*tl.exp2(exp.to(tl.float32)-7.))
    val=tl.where((b&128)!=0,-val,val)
    val=tl.where((exp==15)&(mant==7),float('nan'),val)
    s=tl.load(S+e*tl.cdiv(N,128)*tl.cdiv(K,128)+(n//128)*tl.cdiv(K,128)+k//128,mask,0)
    return (val*s).to(tl.bfloat16)


@triton.jit(do_not_specialize=['T'], do_not_specialize_on_alignment=['T'])
def combine(P,Ids,W,Y,T,H:tl.constexpr,TOP:tl.constexpr,E:tl.constexpr,B:tl.constexpr,KT:tl.constexpr,Shared=None,FUSED:tl.constexpr=False):
    t=tl.program_id(0); n=tl.program_id(1)*B+tl.arange(0,B)
    a=tl.arange(0,KT)
    e=tl.load(Ids+t*TOP+a,a<TOP,-1)
    valid=(a<TOP)&(e>=0)&(e<E)
    w=tl.load(W+t*TOP+a,valid,0)
    z=tl.load(P+(t*TOP+a[:,None])*H+n[None,:],valid[:,None]&(n[None,:]<H),0)
    y=tl.sum(z*w[:,None],axis=0)
    if FUSED:
        y=y+tl.load(Shared+t*H+n,n<H,0)
    tl.store(Y+t*H+n,y,n<H)


@dataclass
class Workspace:
    hidden: torch.Tensor
    partial: torch.Tensor
    slots: torch.Tensor
    counts: torch.Tensor
    # Layer-shared scratch, populated afresh on each large-prefill call.
    weight: torch.Tensor | None = None

    @classmethod
    def create(cls,tokens,topk,experts,hidden,intermediate,device):
        r=tokens*topk
        return cls(torch.empty((r,intermediate),device=device,dtype=torch.bfloat16),
                   torch.empty((r,hidden),device=device,dtype=torch.float32),
                   torch.empty((experts,r),device=device,dtype=torch.int32),
                   torch.empty(experts,device=device,dtype=torch.int32),
                   torch.empty(experts*2*intermediate*hidden,device=device,dtype=torch.bfloat16)
                   if tokens>=DENSE_PREFILL_MIN_TOKENS else None)


def validate(x,ids,rw,gu,gs,down,ds,out,ws,group,fp8):
    """Metadata-only checks, no host reads of device tensors. No aliasing allowed."""
    t,h=x.shape; e,n,h2=gu.shape; i=n//2; top=ids.shape[1]; r=t*top
    if group not in (64,128) or h%group or i%group or h2*2!=h or n%2:
        raise ValueError('invalid group/alignment/GU shape')
    if min(t,h,e,i,top)<=0 or top>32: raise ValueError('invalid dimensions')
    specs=[(x,(t,h),torch.bfloat16),(ids,(t,top),torch.int32),(rw,(t,top),torch.float32),
           (gu,(e,2*i,h//2),torch.uint8),(gs,(e,2*i,h//group),torch.float16),
           (down,(e,h,i if fp8 else i//2),torch.uint8),
           (ds,(e,triton.cdiv(h,128),triton.cdiv(i,128)) if fp8 else (e,h,i//group),torch.float32 if fp8 else torch.float16),
           (out,(t,h),torch.float32),(ws.hidden,(r,i),torch.bfloat16),
           (ws.partial,(r,h),torch.float32),(ws.slots,(e,r),torch.int32),(ws.counts,(e,),torch.int32)]
    if ws.weight is not None:
        specs.append((ws.weight,(e*2*i*h,),torch.bfloat16))
    for a,shape,dtype in specs:
        if a.shape!=shape or a.dtype!=dtype or a.device!=x.device or not a.is_cuda or not a.is_contiguous():
            raise ValueError(f'expected {shape} {dtype} contiguous on {x.device}, got {a.shape} {a.dtype}')
    bases=[a.untyped_storage().data_ptr() for a,_,_ in specs]
    if len(set(bases))!=len(bases): raise ValueError('overlapping storage is unsupported')
    return t,h,e,i,top
