"""Four-stream prefill residual expansion into caller-owned storage.

Keep sequential FP32 additions and disable FMA to preserve reference rounding.
The caller must provide non-overlapping contiguous output storage.
"""
import torch
import triton as tr
import triton.language as tl

@tr.jit
def _expand(X,R,P,C,O,D:tl.constexpr,B:tl.constexpr):
 t=tl.program_id(0); d=tl.program_id(1)*B+tl.arange(0,B);mask=d<D
 x=tl.load(X+t*D+d,mask,0).to(tl.float32)
 r0=tl.load(R+(t*4+0)*D+d,mask,0).to(tl.float32)
 r1=tl.load(R+(t*4+1)*D+d,mask,0).to(tl.float32)
 r2=tl.load(R+(t*4+2)*D+d,mask,0).to(tl.float32)
 r3=tl.load(R+(t*4+3)*D+d,mask,0).to(tl.float32)
 for h in tl.static_range(4):
  a=r0*tl.load(C+t*16+h)
  b=r1*tl.load(C+t*16+4+h)
  c=r2*tl.load(C+t*16+8+h)
  e=r3*tl.load(C+t*16+12+h)
  z=((a+b)+c)+e
  y=tl.load(P+t*4+h)*x+z
  tl.store(O+(t*4+h)*D+d,y,mask)

def expand(x,residual,post,comb,*,out):
 t,d=x.shape
 if (residual.shape!=(t,4,d) or out.shape!=residual.shape
     or x.dtype!=torch.bfloat16 or residual.dtype!=x.dtype or out.dtype!=x.dtype
     or post.dtype!=torch.float32 or comb.dtype!=torch.float32
     or post.shape!=(t,4) or comb.shape!=(t,4,4)
     or any(not z.is_cuda or not z.is_contiguous() for z in (x,residual,post,comb,out))):
  raise ValueError('requires contiguous CUDA BF16 x[T,D], residual/out[T,4,D], FP32 post/comb')
 if any(z.device != x.device for z in (residual,post,comb,out)):
  raise ValueError('residual operands must share one CUDA device')
 if any(out.untyped_storage().data_ptr() == z.untyped_storage().data_ptr() for z in (x,residual,post,comb)):
  raise ValueError('output must not alias residual inputs')
 if not t or not d:
  return out
 _expand[(t,tr.cdiv(d,256))](x,residual,post,comb,out,d,256,num_warps=4,enable_fp_fusion=False)
 return out
