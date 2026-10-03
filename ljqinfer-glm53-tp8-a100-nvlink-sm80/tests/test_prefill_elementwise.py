"""Production prefill fusion regressions; run directly on one CUDA device."""
import torch
from model.glm53_block import _rms_apply,_rms_half,_rope_interleaved,_write_cache
from ops.prefill_elementwise import ElementwiseFusion

@torch.inference_mode()
def main():
 torch.manual_seed(431)
 device='cuda';f=ElementwiseFusion(_rms_apply,129,device)
 inv=8000000.**(-torch.arange(0,64,2,device=device,dtype=torch.float64)/64)
 count=0
 for n in (*range(1,16),16,17,31,127,129):
  for h in (512,2048,6144):
   # Large-row kernel retains its existing four-element stride contract.
   for pad in ((0,1,64) if n<16 else (0,64)):
    x=torch.randn((n,h+pad),device=device,dtype=torch.float16)[:,:h]
    w=torch.randn(h,device=device,dtype=torch.float16)
    assert torch.equal(f.rms(x,w),_rms_half(x,w)),('rms',n,h,pad)
    count+=1
 # Unaligned storage offsets must use the original reduction path.
 for h in (512,2048,6144):
  x=torch.randn((8,h+4),device=device,dtype=torch.float16)[:,1:1+h]
  w=torch.randn(h+1,device=device,dtype=torch.float16)[1:]
  assert torch.equal(f.rms(x,w),_rms_half(x,w))
 # Replayed graphs must consume new values, not captured host scalars.
 x=torch.randn((8,6144),device=device,dtype=torch.float16)
 w=torch.randn(6144,device=device,dtype=torch.float16)
 f.rms(x,w);torch.cuda.synchronize()
 graph=torch.cuda.CUDAGraph()
 with torch.cuda.graph(graph):y=f.rms(x,w)
 for factor in (0.,.7,-1.3):
  x.normal_().mul_(factor);w.normal_();graph.replay()
  assert torch.equal(y,_rms_half(x,w))
 graph.reset()
 pos=torch.arange(129,device=device,dtype=torch.int64)
 ql=torch.randn((129,8,512),device=device,dtype=torch.float16)
 qr=torch.randn((129,8,256),device=device,dtype=torch.float16)[...,192:]
 iq=torch.randn((129,4,128),device=device,dtype=torch.float16)
 norm=torch.randn((129,512),device=device,dtype=torch.float16)
 rot=torch.randn((129,576),device=device,dtype=torch.float16)[:,512:]
 ik=torch.randn((129,128),device=device,dtype=torch.float16)
 for offset in (0,8192,135168):
  pos.copy_(torch.arange(129,device=device)+offset)
  with f.chunk(pos,inv):
   assert torch.equal(f.query(ql,qr),torch.cat((ql,_rope_interleaved(qr,pos,inv)),-1))
   assert torch.equal(f.index_query(iq),torch.cat((_rope_interleaved(iq[...,:64],pos,inv),iq[...,64:]),-1))
   table=torch.arange((offset+129+63)//64,device=device,dtype=torch.int64).flip(0)
   for width,left,right,first in ((576,norm,rot,False),(128,ik[:,64:],ik[:,:64],True)):
    pool=torch.full((table.numel(),64,width),-3.,device=device,dtype=torch.float16);ref=pool.clone()
    if first:
     f.index_cache(ik,pool,table,pos)
     rows=torch.cat((_rope_interleaved(right,pos,inv),left),-1)
    else:
     f.cache(left,right,pool,table,pos)
     rows=torch.cat((left,_rope_interleaved(right,pos,inv)),-1)
    _write_cache(rows,ref,table,pos)
    assert torch.equal(pool,ref),('cache',offset,width)
  assert not f.in_chunk and not f.ready
 # A short view must never expose stale rows from the maximum workspace.
 with f.chunk(pos[:7],inv):assert f.query(ql[:7],qr[:7]).shape==(7,8,576)
 try:
  with f.chunk(pos,inv,first_chunk=True):raise RuntimeError('injected')
 except RuntimeError:pass
 assert not f.ready and not f.in_chunk and not f.first_chunk
 f.begin(pos,inv);old=f.c.clone();pos.add_(1);f.begin(pos,inv)
 assert not torch.equal(old,f.c)
 torch.cuda.synchronize()
 print({'ALL_PASS':True,'rms_cases':count,'positions':[0,8192,135168]},flush=True)

if __name__=='__main__':main()
