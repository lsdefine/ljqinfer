"""NPU causal/count/uniqueness regression for official eager Top512.

Run with the same CANN Python/environment as the engine, on an idle NPU.
Numerical equality to the retired scorer is intentionally not the acceptance gate.
"""
import sys,json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch,torch_npu
from ops.prefill.attention import select
torch.npu.set_device(0);torch.manual_seed(602)
class Parallel:
 def logits(self,x,out):out.copy_(x[None])
report=[]
def run(t,start):
 k=(start+t)//2
 q=torch.randn(t,32,128,device='npu',dtype=torch.bfloat16)
 w=torch.rand(t,32,device='npu',dtype=torch.float32)
 key=torch.randn(max(1,k),128,device='npu',dtype=torch.bfloat16)
 pos=torch.arange(start,start+t,device='npu');valid=torch.tensor([k],device='npu')
 ids=select(q,w,key,pos,valid,ratio=2,total_heads=32,parallel=Parallel(),workspace=None).cpu()
 limits=((torch.arange(start,start+t)+1)//2).clamp(max=k)
 assert ids.shape==(t,512) and ids.dtype==torch.int64
 assert ((ids==-1)|((ids>=0)&(ids<limits[:,None]))).all()
 assert torch.equal((ids>=0).sum(1),limits.clamp(max=512))
 ordered=ids.sort(1).values
 assert not ((ordered[:,1:]==ordered[:,:-1])&(ordered[:,1:]>=0)).any()
 result={'rows':t,'start':start,'keys':k,'pass':True}
 report.append(result);print(result,flush=True)
for t,start in [(1,0),(1,1),(2,0),(127,0),(129,127),(1024,0),(1025,0),(1026,0),(1027,1),(1,1025),(2,1025),(127,8191),(129,8192),(8192,0),(4096,32768),(8192,122880)]:run(t,start)
print('PASS',len(report),flush=True)
