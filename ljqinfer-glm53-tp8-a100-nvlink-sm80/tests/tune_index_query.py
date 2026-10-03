import triton as tr
import os,sys,json,math,time,argparse,statistics
from pathlib import Path
from datetime import timedelta
import torch
import torch.distributed as dist
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from ops.sparse_index_tp import TPIndexParallel
from ops.sparse_index_query import select_prefill as select_paged
from ops.sparse_index_query import extension
p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--bench',action='store_true');a=p.parse_args()
rank=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(rank)
dist.init_process_group('nccl',timeout=timedelta(minutes=12),device_id=torch.device('cuda',rank));par=TPIndexParallel()
# One process builds; peers wait rather than racing the extension lock.
if rank==0:extension()
dist.barrier();extension();torch.backends.cuda.matmul.allow_tf32=True
records=[]
def note(obj):
 records.append(obj);Path(a.output+f'.rank{rank}.json').write_text(json.dumps(records,indent=2))
 if rank==0:print(json.dumps(obj),flush=True)

def correctness(t,n,k,tile,empty=False,ties=False,dtype=torch.float16):
 torch.manual_seed(413+t)
 page=64;pages=math.ceil(n/page);table=torch.randperm(pages,device='cuda',dtype=torch.int64)
 pool=(torch.randint(-8,9,(pages,page,128),device='cuda').float()/8).to(dtype)
 qa=(torch.randint(-8,9,(t,32,128),device='cuda').float()/8).to(dtype)
 wa=torch.randint(-4,5,(t,32),device='cuda').float()/8
 if ties:wa.zero_()
 q=qa[:,rank*4:(rank+1)*4].contiguous();w=wa[:,rank*4:(rank+1)*4].contiguous()
 pos=torch.linspace(0,n+3,t,device='cuda').long();ctx=torch.tensor([0 if empty else n],device='cuda')
 ids=torch.empty((t,k),device='cuda',dtype=torch.int64)
 def run():return select_paged(q,w,pool,table,pos,ctx,ids,parallel=par,logical_capacity=n,query_tile=tile,key_tile=256)
 def check():
  keys=pool[table].reshape(-1,128)[:n].double();scores=torch.zeros((t,n),device='cuda',dtype=torch.float64)
  for h in range(32):scores.add_((qa[:,h].double()@keys.T).relu()*wa[:,h,None].double())
  mask=(torch.arange(n,device='cuda')[None,:]>pos[:,None])|(torch.arange(n,device='cuda')[None,:]>=ctx)
  scores.masked_fill_(mask,-torch.inf)
  expected=scores.argsort(dim=-1,descending=True,stable=True)[:,:k]
  expected.masked_fill_(~torch.isfinite(scores.gather(1,expected)),-1)
  valid=ids>=0
  assert torch.equal(valid.sum(1),(expected>=0).sum(1))
  selected=scores.gather(1,ids.clamp_min(0)).masked_fill(~valid,-torch.inf)
  target=scores.gather(1,expected.clamp_min(0)).masked_fill(expected<0,-torch.inf)
  assert torch.equal(selected.sort(dim=1,descending=True).values,target.sort(dim=1,descending=True).values),(rank,t,n)
  assert all(len(torch.unique(row[row>=0]))==int((row>=0).sum()) for row in ids)
  allids=[torch.empty_like(ids) for _ in range(8)];dist.all_gather(allids,ids)
  assert all(torch.equal(ids,x) for x in allids)
 run();check();torch.cuda.synchronize();dist.barrier()
 g=torch.cuda.CUDAGraph()
 with torch.cuda.graph(g):run()
 ctx.fill_(n//2);pos.sub_(2);g.replay();torch.cuda.synchronize();check();g.reset()
 note(dict(test='reference_and_dynamic_graph',t=t,n=n,k=k,empty=empty,ties=ties,dtype=str(dtype),passed=True))
for args in [(1,257,64,8),(6,1031,64,8),(8,1025,64,8),(17,1031,64,8),(8,65,64,8,True),(8,777,64,8,False,True)]:
 for dt in [torch.float16,torch.bfloat16]:correctness(*args,dtype=dt)

def timer(fn):
 for _ in range(2):fn()
 torch.cuda.synchronize();dist.barrier();g=torch.cuda.CUDAGraph()
 with torch.cuda.graph(g):fn()
 for _ in range(2):g.replay()
 torch.cuda.synchronize();samples=[]
 for _ in range(5):
  dist.barrier();s=torch.cuda.Event(enable_timing=True);e=torch.cuda.Event(enable_timing=True)
  s.record();g.replay();e.record();e.synchronize();samples.append(s.elapsed_time(e))
 local=torch.tensor(samples,device='cuda');dist.all_reduce(local,op=dist.ReduceOp.MAX)
 result=statistics.median(local.tolist());g.reset();return result


from ops.sparse_index_query import _score_query
t=12288;n=77824;page=64;local=t//8
torch.manual_seed(481);q=torch.randn(8*t,4,128,device='cuda',dtype=torch.float16);w=torch.randn(8*t,4,device='cuda');pool=torch.randn(n//page,page,128,device='cuda',dtype=q.dtype);table=torch.randperm(n//page,device='cuda');pos=torch.arange(65536,n,device='cuda');ctx=torch.tensor([n],device='cuda');score=torch.empty(local,n,device='cuda')
for bq in [1,2,4,8]:
 for bn in [32,64,128]:
  for nw in [4,8]:
   def run():_score_query[(tr.cdiv(local,bq),tr.cdiv(n,bn))](q,pool,w,table,pos,ctx,score,t,n,page,rank*local,local,bq,bn,num_warps=nw)
   note(dict(bq=bq,bn=bn,warps=nw,score_ms=timer(run)))
note(dict(ALL_PASS=True));dist.barrier();dist.destroy_process_group()
