import os,sys,json,math,time,argparse,statistics
from pathlib import Path
from datetime import timedelta
import torch
import torch.distributed as dist
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from ops.sparse_index_tp import TPIndexParallel
from ops.sparse_index_query import select_prefill as select_paged
from ops.sparse_index_query import extension
p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--bench',action='store_true');p.add_argument('--baseline-wrapper',required=True);a=p.parse_args()
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
 ctx.fill_(n//2);pos.sub_(2);g.replay();torch.cuda.synchronize();check()
 qa.neg_();wa.mul_(2);q.copy_(qa[:,rank*4:(rank+1)*4]);w.copy_(wa[:,rank*4:(rank+1)*4])
 pool.neg_();table.copy_(table.roll(1));ctx.fill_(n);pos.add_(3)
 g.replay();torch.cuda.synchronize();check();g.reset()
 note(dict(test='reference_and_dynamic_graph',t=t,n=n,k=k,empty=empty,ties=ties,dtype=str(dtype),passed=True))
for args in [(1,257,64,8),(6,1031,64,8),(8,1025,64,8),(17,1031,64,8),(65,1031,64,3),(25,257,257,2),(8,65,64,8,True),(8,777,64,8,False,True)]:
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

if a.bench:
 import importlib.util
 spec=importlib.util.spec_from_file_location('archived_baseline',a.baseline_wrapper)
 baseline=importlib.util.module_from_spec(spec);spec.loader.exec_module(baseline)
 if rank==0:baseline.extension()
 dist.barrier();baseline.extension()
 forward=baseline.forward
 from ops.sparse_mla_cuda import forward as pipe,extension as pipe_ext
 from ops.sparse_mla_pair import PairWorkspace
 if rank==0:pipe_ext()
 dist.barrier();pipe_ext()
 from ops.sparse_ops import sparse_mla,make_mla_workspace
 groups=[dist.new_group([i,i+1]) for i in range(0,8,2)]
 pair=groups[rank//2]
 for t in [12288]:
  n=65536+t;page=64;pages=math.ceil(n/page);k=2048
  torch.manual_seed(481)
  table=torch.randperm(pages,device='cuda',dtype=torch.int64)
  pool=torch.randn(pages,page,128,device='cuda',dtype=torch.float16)
  qa=torch.randn(t,32,128,device='cuda',dtype=torch.float16)
  wa=torch.randn(t,32,device='cuda')/math.sqrt(32)
  q=qa[:,rank*4:(rank+1)*4].contiguous();w=wa[:,rank*4:(rank+1)*4].contiguous()
  pos=torch.arange(65536,n,device='cuda',dtype=torch.int64);ctx=torch.tensor([n],device='cuda')
  ids=torch.empty(t,k,device='cuda',dtype=torch.int64)
  def select():return select_paged(q,w,pool,table,pos,ctx,ids,parallel=par,logical_capacity=n)
  select();torch.cuda.synchronize();note(dict(stage='start_bench',queries=t,final_kv=n))
  # Independent sampled full-head FP32 check, allowing only tiny boundary score differences.
  rows=torch.tensor([0,t//2,t-1],device='cuda');keys=pool[table].reshape(-1,128)[:n].float()
  scores=torch.zeros((3,n),device='cuda');torch.backends.cuda.matmul.allow_tf32=False
  for h in range(32):scores.add_((qa[rows,h].float()@keys.T).relu()*wa[rows,h,None])
  scores.masked_fill_(torch.arange(n,device='cuda')[None,:]>pos[rows,None],-torch.inf)
  vals,_=scores.topk(k,dim=1);got=scores.gather(1,ids[rows]);gap=(vals.min(1).values-got.min(1).values).clamp_min(0)
  assert float(gap.max())<1e-3
  assert all(len(torch.unique(row))==k for row in ids[rows])
  torch.backends.cuda.matmul.allow_tf32=True
  note(dict(test='large_sample_reference',queries=t,max_boundary_gap=float(gap.max())))
  for rep in range(2):note(dict(queries=t,round=rep,index_tp8_ms=timer(select)))
  torch.manual_seed(491);main_pool=torch.randn(pages,page,576,device='cuda',dtype=torch.float16)*.3
  torch.manual_seed(501+rank)
  main_q=torch.randn(t,8,576,device='cuda',dtype=torch.float16)*.3;out=torch.empty(t,8,512,device='cuda',dtype=torch.float16)
  if t>8:
   def attn():forward(main_q,main_pool,table,ids,pos,ctx,out)
  else:
   idx32=torch.empty_like(ids,dtype=torch.int32);ws=make_mla_workspace(main_q,splits=32)
   def attn():
    idx32.copy_(ids);sparse_mla(main_q,main_pool,table,idx32,pos,ctx,out,ws,splits=32,block_n=32)
  if t>8:
   attn();ref=out.clone()
   workspace=PairWorkspace(main_q,ids,pair)
   def candidate():workspace(main_q,main_pool,table,ids,pos,ctx,out)
   candidate();torch.cuda.synchronize()
   error=float((out.float()-ref.float()).norm()/ref.float().norm())
   assert error<.001,error
   def full():select();candidate()
   for rep in range(2):
    note(dict(queries=t,round=rep,baseline_ms=timer(attn),pair_ms=timer(candidate),full_ms=timer(full),rel=error))
  del main_pool,main_q,out
note(dict(ALL_PASS=True));dist.barrier();dist.destroy_process_group()
