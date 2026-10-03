import os,sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
import torch.distributed as dist
from ops.sparse_index_tp import TPIndexParallel
from ops.sparse_index_query import select_prefill
from ops.sparse_index_decode import select_decode
from ops import sparse_topk_v2
from model.glm53_workspace import IndexBuffers
rank=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(rank)
dist.init_process_group('nccl',device_id=torch.device('cuda',rank))
p=TPIndexParallel();records=[]
for phase,fn in [('prefill',select_prefill),('decode',select_decode)]:
 owner=IndexBuffers(65,8192,phase,'cuda')
 for t,n in [(1,2048),(8,8192),(7,4096),(65,4096),(8,2048)]:
  torch.manual_seed(120+t)
  pool=torch.randn(n//64,64,128,device='cuda',dtype=torch.float16)
  table=torch.randperm(n//64,device='cuda')
  q=torch.randn(t,4,128,device='cuda',dtype=torch.float16)
  weights=torch.randn(t,4,device='cuda');pos=torch.arange(n-t,n,device='cuda');ctx=torch.tensor([n],device='cuda')
  result=torch.empty(t,2048,device='cuda',dtype=torch.int64)
  kw=dict(parallel=p,logical_capacity=n)
  if phase=='decode':kw['topk_workspace']=sparse_topk_v2.workspace((t+7)//8,n,device='cuda')
  fn(q,weights,pool,table,pos,ctx,result,**kw);ref=result.clone()
  scratch=owner.view(t,n)
  assert all(v.is_contiguous() and v.data_ptr()==owner.buffers[k].data_ptr() for k,v in scratch.items())
  fn(q,weights,pool,table,pos,ctx,result,scratch=scratch,**kw)
  assert torch.equal(ref,result),(phase,t,n)
  # No wrapper-owned torch.empty/full calls in the supplied-scratch path.
  from unittest.mock import patch
  def forbidden(*a,**k):raise AssertionError('unexpected scratch allocation')
  with patch.object(torch,'empty',forbidden),patch.object(torch,'full',forbidden):
   fn(q,weights,pool,table,pos,ctx,result,scratch=scratch,**kw)
  assert torch.equal(ref,result)
  records.append(dict(phase=phase,tokens=t,capacity=n,equal=True))
  if t==8 and n==2048:
   graph=torch.cuda.CUDAGraph()
   torch.cuda.synchronize()
   with torch.cuda.graph(graph):fn(q,weights,pool,table,pos,ctx,result,scratch=scratch,**kw)
   graph.replay();torch.cuda.synchronize();assert torch.equal(ref,result)
   graph.reset()
print(json.dumps({'rank':rank,'cases':records}),flush=True)
print('INDEX_BUFFERS_PASS',rank,flush=True)
dist.barrier();dist.destroy_process_group()
