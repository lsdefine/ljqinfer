"""Draft direct RMS weights and BF16 RoPE rounding; strided Q8 B1..4."""
import unittest
import torch
from ops.kernels import K



def qk_norm_rope(q,k,qw,kw,freq,rd,eps):
 return K.dflash_qk_norm_rope(q,k,qw,kw,freq,eps)

@unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
class DraftQKNormRopeTest(unittest.TestCase):
 def test_draft_qknorm_rope(self):

  import flashinfer
  torch.manual_seed(2718)
  report={'complete':False,'checks':[],'timing':[],'not_e2e':True}
  def metric(a,b):
   return {'exact':torch.equal(a,b),'max_abs':(a.float()-b.float()).abs().max().item(),'rrmse':((a.float()-b.float()).square().mean().sqrt()/a.float().square().mean().sqrt().clamp_min(1e-20)).item()}
  for batch in (1,2,3,4):
   scale=1.
   packed=torch.randn(batch*8,12,128,device='cuda',dtype=torch.bfloat16)*scale
   q,k=packed[:,:8],packed[:,8:10]
   qw=(torch.randn(128,device='cuda')*.1+1).bfloat16();kw=(torch.randn(128,device='cuda')*.1+1).bfloat16()
   for start in (0,12000):
    pos=torch.arange(start,start+batch*8,device='cuda');freq=K.rope_frequencies(pos,128,10000000.,q.dtype)
    def old():
     nq=flashinfer.norm.rmsnorm(q.contiguous(),qw,eps=1e-6);nk=flashinfer.norm.rmsnorm(k.contiguous(),kw,eps=1e-6)
     return K.apply_rope(nq,freq,128),K.apply_rope(nk,freq,128)
    def new():return qk_norm_rope(q,k,qw,kw,freq,128,1e-6)
    a,b=old(),new();report['checks'].append({'scale':scale,'start':start,'q':metric(a[0],b[0]),'k':metric(a[1],b[1])})
  assert all(c[x]['rrmse'] < 0.0005 for c in report['checks'] for x in ('q','k'))
  graph = torch.cuda.CUDAGraph()
  with torch.cuda.graph(graph): captured = new()
  packed.mul_(0.75)
  graph.replay()
  torch.cuda.synchronize()
  for expected, actual in zip(old(), captured):
   assert metric(expected, actual)['rrmse'] < 0.0005

if __name__ == '__main__':
 unittest.main()
