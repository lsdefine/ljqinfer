"""GPU regression: bounded HC token tiles equal independent short batches."""
import torch,time,json
from ops.decode.hc_pre import hc_pre
torch.set_num_threads(2)
torch.manual_seed(17)
h,d=4,4096
fn=torch.randn(2*h+h*h,h*d,device='cuda',dtype=torch.bfloat16)*0.01
scale=torch.ones(3,device='cuda');base=torch.randn(2*h+h*h,device='cuda')
kw=dict(norm_eps=1e-6,hc_eps=1e-6,iters=20)
for rows in [1,6,33,530,2048]:
 x=torch.randn(rows,h,d,device='cuda',dtype=torch.bfloat16)
 t=time.perf_counter();got=hc_pre(x,fn,scale,base,**kw);torch.cuda.synchronize()
 elapsed=time.perf_counter()-t
 chunks=[hc_pre(z,fn,scale,base,**kw) for z in x.split(6)]
 ref=[torch.cat([z[i] for z in chunks]) for i in range(3)]
 for a,b in zip(got,ref):
  assert torch.isfinite(a).all()
  torch.testing.assert_close(a,b,rtol=0,atol=0)
 print('PASS',rows,'first_call_seconds',elapsed,flush=True)
