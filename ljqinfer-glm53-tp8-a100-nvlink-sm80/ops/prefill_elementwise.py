import torch
from contextlib import contextmanager
import triton
import triton.language as tl

@triton.jit(do_not_specialize=['N', 'S'], do_not_specialize_on_alignment=['N', 'S'])
def square_fp32(X,Y,H:tl.constexpr,S,N,B:tl.constexpr):
 j=tl.program_id(0)*B+tl.arange(0,B)
 x=tl.load(X+(j//H)*S+j%H,j<N,0).to(tl.float32)
 tl.store(Y+j,x*x,j<N)

@triton.jit(do_not_specialize=['N', 'S0', 'S1', 'S2'], do_not_specialize_on_alignment=['N', 'S0', 'S1', 'S2'])
def rope_apply(X,C,S,Y,N,HEADS:tl.constexpr,D:tl.constexpr,S0,S1,S2,B:tl.constexpr):
 j=tl.program_id(0)*B+tl.arange(0,B)
 row=j//(HEADS*D);head=(j//D)%HEADS;d=j%D
 base=row*S0+head*S1+(d//2)*2*S2
 e=tl.load(X+base,j<N,0).to(tl.float32)
 o=tl.load(X+base+S2,j<N,0).to(tl.float32)
 c=tl.load(C+row*32+d//2,j<N,0)
 s=tl.load(S+row*32+d//2,j<N,0)
 even=e*c-o*s
 odd=e*s+o*c
 tl.store(Y+j,tl.where(d%2==0,even,odd),j<N)

@triton.jit(do_not_specialize=['N', 'L0', 'L1', 'R0', 'R1'], do_not_specialize_on_alignment=['N', 'L0', 'L1', 'R0', 'R1'])
def pack_rope(L,R,C,S,O,TABLE,POS,
              N,H:tl.constexpr,D:tl.constexpr,PLAIN:tl.constexpr,
              L0,L1,R0,R1,
              FIRST:tl.constexpr,PAGED:tl.constexpr,PAGE:tl.constexpr,
              NP:tl.constexpr,PHYSICAL:tl.constexpr,B:tl.constexpr):
 j=tl.program_id(0)*B+tl.arange(0,B)
 row=j//(H*D);head=(j//D)%H;d=j%D
 if FIRST:
  rot=d<64
  pd=d-64
  rd=d
 else:
  rot=d>=PLAIN
  pd=d
  rd=d-PLAIN
 plain=tl.load(L+row*L0+head*L1+pd,(j<N)&(~rot),0)
 rb=row*R0+head*R1+(rd//2)*2
 e=tl.load(R+rb,(j<N)&rot,0).to(tl.float32)
 o=tl.load(R+rb+1,(j<N)&rot,0).to(tl.float32)
 c=tl.load(C+row*32+rd//2,(j<N)&rot,0)
 s=tl.load(S+row*32+rd//2,(j<N)&rot,0)
 even=e*c-o*s
 odd=e*s+o*c
 val=tl.where(rot,tl.where(rd%2==0,even,odd),plain.to(tl.float32))
 if PAGED:
  pos=tl.load(POS+row,j<N,0)
  valid=(j<N)&(pos>=0)&(pos<NP*PAGE)
  page=tl.load(TABLE+pos//PAGE,valid,0)
  off=(page*PAGE+pos%PAGE)*D+d
  tl.store(O+off,val,valid&(page>=0)&(page<PHYSICAL))
 else:
  tl.store(O+j,val,j<N)

class ElementwiseFusion:
 """Phase-owned scratch; sequential layers only. Explicit chunk scope invalidates RoPE."""
 def __init__(self,rms_apply,tokens,device):
  self.rms_apply=rms_apply
  self.tokens=tokens
  self.qout=torch.empty((tokens,8,576),device=device,dtype=torch.float16)
  self.iqout=torch.empty((tokens,4,128),device=device,dtype=torch.float16)
  self.means={h:torch.empty((tokens,1),device=device,dtype=torch.float32) for h in (512,2048,6144)}
  self.outputs={h:torch.empty((tokens,h),device=device,dtype=torch.float16) for h in self.means}
  self.ready=False
  self.in_chunk=False
  self.first_chunk=False
  if tokens>=16:
   from .prefill_rms_full import extension
   self.rms_full=extension().forward
 def prepare(self,positions,inv_freq):
  if not 0<positions.numel()<=self.tokens:raise ValueError('RoPE exceeds workspace')
  angle=positions.double()[:,None]*inv_freq[None,:]
  self.c=angle.cos().float();self.s=angle.sin().float()
  self.ready=True
 def chunk(self,positions,inv_freq,*,first_chunk=False):
  return self._chunk(positions,inv_freq,first_chunk=first_chunk)
 @contextmanager
 def _chunk(self,positions,inv_freq,*,first_chunk):
  if self.in_chunk:raise RuntimeError('prefill workspace is not reentrant')
  self.prepare(positions,inv_freq)
  self.in_chunk=True
  self.first_chunk=first_chunk
  try:yield self
  finally:
   self.in_chunk=False
   self.first_chunk=False
   self.ready=False
 def begin(self,positions,inv_freq):
  # Standalone binding calls must refresh even at the same tensor address.
  if not self.in_chunk:self.prepare(positions,inv_freq)
 def rms(self,x,w):
  assert x.ndim==2 and x.stride(1)==1 and x.dtype==torch.float16
  n,h=x.shape
  if h not in self.outputs or not 0<n<=self.tokens:raise ValueError('RMS workspace shape')
  y=self.outputs[h][:n]
  if n>=16:
   self.rms_full(x,w,y)
   return y
  elif (x.stride(0)%4==0 and x.data_ptr()%8==0 and w.data_ptr()%8==0
        and w.is_contiguous() and w.dtype==x.dtype):
   from ops.decode_rms_small import extension
   extension().forward(x,w,y)
   return y
  else:
   # Unaligned views retain the original ATen mean path.
   squared=torch.empty(x.shape,device=x.device,dtype=torch.float32)
   square_fp32[(triton.cdiv(x.numel(),1024),)](x,squared,h,x.stride(0),x.numel(),1024,enable_fp_fusion=False)
   mean=squared.mean(-1,keepdim=True)
  a=torch.rsqrt(mean+1e-5)
  self.rms_apply[(triton.cdiv(x.numel(),1024),)](x,w,a,y,h,x.stride(0),x.numel(),1024,enable_fp_fusion=False)
  return y
 def rope(self,x,positions,inv_freq):
  assert self.ready
  heads=x.shape[1] if x.ndim==3 else 1
  y=torch.empty(x.shape,dtype=torch.float16,device=x.device)
  rope_apply[(triton.cdiv(x.numel(),1024),)](x,self.c,self.s,y,x.numel(),heads,64,x.stride(0),x.stride(1) if x.ndim==3 else 0,x.stride(-1),1024,enable_fp_fusion=False)
  return y
 def pack(self,plain,rot,out,*,first=False,table=None,positions=None):
  assert self.ready and plain.stride(-1)==rot.stride(-1)==1
  h=plain.shape[1] if plain.ndim==3 else 1
  d=plain.shape[-1]+64;n=plain.shape[0]*h*d
  paged=table is not None
  assert not paged or h==1
  pack_rope[(triton.cdiv(n,1024),)](plain,rot,self.c,self.s,out,
   table if paged else self.c,positions if paged else self.c,
   n,h,d,plain.shape[-1],plain.stride(0),plain.stride(1) if plain.ndim==3 else 0,
   rot.stride(0),rot.stride(1) if rot.ndim==3 else 0,first,paged,
   out.shape[1] if paged else 1,table.numel() if paged else 1,out.shape[0] if paged else 1,
   1024,enable_fp_fusion=False)
  return out
 def query(self,ql,qr):return self.pack(ql,qr,self.qout[:ql.shape[0]])
 def cache(self,norm,rot,pool,table,pos):
  self.pack(norm,rot,pool,table=table,positions=pos)
 def index_query(self,iq):return self.pack(iq[...,64:],iq[...,:64],self.iqout[:iq.shape[0]],first=True)
 def index_cache(self,ik,pool,table,pos):
  self.pack(ik[:,64:],ik[:,:64],pool,first=True,table=table,positions=pos)
