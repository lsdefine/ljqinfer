"""DFlash2 fixed-128 RoPE and greedy path selection.

Keep the two BF16 product roundings before the RoPE sum; do not fuse
RMS reductions here. Frequencies are shared across six layers per call.
Selection preserves torch.argmax's first-index tie break for finite scores.
"""
import torch
import triton as tr
import triton.language as tl
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

@tr.jit
def _rope_only(X,C,S,O,H:tl.constexpr,D:tl.constexpr,S0:tl.constexpr,S1:tl.constexpr,FS:tl.constexpr):
 row=tl.program_id(0);j=tl.arange(0,128);t=row//H;h=row%H
 rj=tl.where(j<64,j+64,j-64);jh=j%64
 x=tl.load(X+t*S0+h*S1+j).to(tl.float32)
 y=tl.load(X+t*S0+h*S1+rj).to(tl.float32)
 y=tl.where(j<64,-y,y)
 c=tl.load(C+t*FS+jh).to(tl.float32);s=tl.load(S+t*FS+jh).to(tl.float32)
 a=(x*c).to(X.dtype.element_ty).to(tl.float32)
 b=(y*s).to(X.dtype.element_ty).to(tl.float32)
 tl.store(O+row*D+j,a+b)
def rope_only(x,freq):
 t,h,d=x.shape;c,s=freq
 assert d==128 and x.stride(-1)==1
 o=torch.empty(x.shape,device=x.device,dtype=x.dtype)
 _rope_only[(t*h,)](x,c,s,o,h,d,x.stride(0),x.stride(1),c.stride(0),enable_fp_fusion=False)
 return o


def frequencies(positions,dtype):
 inv=1000000.**(-torch.arange(0,128,2,device=positions.device,dtype=torch.float32)/128)
 angles=positions.float()[:,None]*inv[None,:]
 return angles.cos().to(dtype),angles.sin().to(dtype)
