"""Q8 decode split-MLA: independent FP32 oracle and live-metadata graph replay."""
import sys,json,statistics
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from ops.sparse_ops import sparse_mla,make_mla_workspace
from ops.sparse_mla_cuda import forward

torch.cuda.set_device(0);torch.manual_seed(731)
torch.backends.cuda.matmul.allow_tf32=False
N=131072;T=8;K=2048
pool=torch.randn(N//64,64,576,device='cuda',dtype=torch.float16)*.3
table=torch.randperm(N//64,device='cuda')
q=torch.randn(T,8,576,device='cuda',dtype=torch.float16)*.5
ids=torch.full((T,K),-1,device='cuda',dtype=torch.int32)
ids64=ids.long();pos=torch.arange(T,device='cuda');ctx=torch.tensor([T],device='cuda')
out=torch.empty((T,8,512),device='cuda',dtype=torch.float16);old=torch.empty_like(out)
ws=make_mla_workspace(q,splits=32)
def candidate():return sparse_mla(q,pool,table,ids,pos,ctx,out,ws,splits=32)
def baseline():return forward(q,pool,table,ids64,pos,ctx,old)
def oracle():
 result=[]
 for row in range(T):
  tok=ids[row].long();valid=(tok>=0)&(tok<ctx[0])&(tok<=pos[row])
  safe=tok.clamp_min(0);kv=pool[table[safe//64],safe%64].float()
  score=q[row].float()@kv.T/16
  if not valid.any():result.append(torch.zeros_like(out[row],dtype=torch.float32));continue
  score[:,~valid]=-float('inf');result.append(score.softmax(-1)@kv[:,:512])
 return torch.stack(result)
def measure(fn):
 for _ in range(3):fn()
 torch.cuda.synchronize();g=torch.cuda.CUDAGraph()
 with torch.cuda.graph(g):
  for _ in range(20):fn()
 g.replay();torch.cuda.synchronize();times=[]
 for _ in range(5):
  a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)
  a.record();g.replay();b.record();b.synchronize();times.append(a.elapsed_time(b)/20)
 return statistics.median(times)
candidate();torch.cuda.synchronize();graph=torch.cuda.CUDAGraph()
with torch.cuda.graph(graph):candidate()
records=[]
for n in [0,54,2048,32768,131072,54]:
 ctx.fill_(n);pos.copy_(torch.arange(max(0,n-T),max(0,n-T)+T,device='cuda'))
 ids.fill_(-1)
 for row in range(T):
  if n:
   selected=torch.randperm(n,device='cuda')[:min(K,n)];ids[row,:selected.numel()].copy_(selected)
 # Keep one entirely empty row at all lengths; holes and causal tail are intentional.
 ids[-1].fill_(-1);ids[0,::17]=-1;ids64.copy_(ids)
 graph.replay();torch.cuda.synchronize();ref=oracle();baseline()
 rel=float((out.float()-ref).norm()/ref.norm().clamp_min(1e-20))
 old_rel=float((old.float()-ref).norm()/ref.norm().clamp_min(1e-20))
 assert torch.isfinite(out).all() and torch.equal(out[-1],torch.zeros_like(out[-1]))
 assert rel<.002,(n,rel)
 records.append(dict(context=n,relative_error=rel,baseline_relative_error=old_rel,candidate_ms=measure(candidate),baseline_ms=measure(baseline)))
 print(json.dumps(records[-1]),flush=True)
print('PASS split MLA: oracle, holes, empty rows, shuffled pages, dynamic graph replay',flush=True)
