"""Independent sparse TP gather and request-boundary convolution tests."""
import os,sys,json
from pathlib import Path
from types import SimpleNamespace
from datetime import timedelta
P=Path(__file__).resolve().parents[1];R=Path(os.environ['BATCH_AUDIT_DIR']);R.mkdir(exist_ok=True,parents=True)
sys.path[:0]=[str(R),str(P)]
import torch,torch.distributed as dist
from model.glm53_joint_attention import select_joint
from model.glm53_joint_draft import grouped_batch
from ops.dflash_glm53 import grouped
from ops.sparse_index_decode import select_decode
from ops.sparse_topk_v2 import workspace
rank=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(rank);dist.init_process_group('nccl',timeout=timedelta(minutes=5))
class TP:
 world=8
 def __init__(self):self.rank=rank
 def gather_rows(self,x,y):dist.all_gather_into_tensor(y,x)
tp=TP();torch.manual_seed(414+rank);results=[]
with torch.inference_mode():
 for batch in [2,3,4]:
  h=torch.randn(batch*8,6144,device=rank,dtype=torch.bfloat16)
  delta=torch.randn(batch*8,2,384,device=rank,dtype=h.dtype);base=torch.randn(2,2,6144,device=rank,dtype=h.dtype)
  for side in [0,1]:
   ref=torch.cat([grouped(h[i*8:(i+1)*8],delta[i*8:(i+1)*8],base,side) for i in range(batch)])
   actual=grouped_batch(h,delta,base,side);assert torch.equal(ref,actual),(batch,side,'conv')
  bindings=[];engines=[];qs=[];ws=[];expected=[]
  for row in range(batch):
   capacity=4096;length=[1031,2067,3097,4001][row]
   table=torch.randperm(64,device=rank);dist.broadcast(table,src=0)
   pool=torch.randn(64,64,128,device=rank,dtype=torch.float16)
   q=torch.randn(8,4,128,device=rank,dtype=torch.float16);w=torch.rand(8,4,device=rank)
   pos=torch.arange(length,length+8,device=rank);ctx=torch.tensor(length+8,device=rank)
   binding=SimpleNamespace(parallel=tp,topk=2048,capacity=capacity,index_pool=pool,index_table=table,topk_workspace=workspace(1,capacity,device=rank),ids=torch.empty(8,2048,device=rank,dtype=torch.int64))
   select_decode(q,w,pool,table,pos,ctx,binding.ids,parallel=tp,logical_capacity=capacity,topk_workspace=binding.topk_workspace)
   expected.append(binding.ids.clone());bindings.append(binding);engines.append(SimpleNamespace(positions=pos,context=ctx));qs.append(q);ws.append(w)
  select_joint(bindings,torch.cat(qs),torch.cat(ws),engines)
  same=[torch.equal(a,b.ids) for a,b in zip(expected,bindings)];assert all(same),(batch,same)
  # Perturb one request; every other request must stay bitwise unchanged.
  changed=torch.cat(qs);changed[:8].neg_();select_joint(bindings,changed,torch.cat(ws),engines)
  assert all(torch.equal(expected[i],bindings[i].ids) for i in range(1,batch))
  results.append(dict(batch=batch,conv_exact=True,index_exact=same,isolated=True))
(R/f'boundaries.rank{rank}.json').write_text(json.dumps(results));print('BOUNDARIES_PASS',rank,flush=True)
dist.barrier(device_ids=[rank]);dist.destroy_process_group()
