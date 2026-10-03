import os,sys,math,json,argparse
from pathlib import Path
from datetime import timedelta
import torch
import torch.distributed as dist
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from ops.sparse_mla_cuda import extension
from ops.sparse_mla_pair import PairWorkspace,create_pair_group
parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);args=parser.parse_args()
rank=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(rank)
dist.init_process_group('nccl',timeout=timedelta(minutes=12),device_id=torch.device('cuda',rank))
if rank==0:extension()
dist.barrier();extension();group=create_pair_group()
torch.backends.cuda.matmul.allow_tf32=False
for t,n,k in [(1,63,32),(6,2053,2048),(8,1031,64),(17,4097,2048),(7,65,64),(5,129,33)]:
 torch.manual_seed(791+t);page=64;pages=math.ceil(n/page)
 table=torch.randperm(pages,device='cuda',dtype=torch.int64)
 pool=torch.randn(pages,page,576,device='cuda',dtype=torch.float16)*.3
 ids=torch.stack([torch.randperm(n,device='cuda')[:k] for _ in range(t)]).long()
 ids[:,::17]=-1
 if t>1:ids[0].fill_(-1);ids[1,0]=n+17
 pos=torch.arange(n-t,n,device='cuda',dtype=torch.int64);ctx=torch.tensor([n],device='cuda')
 torch.manual_seed(817+rank);q=torch.randn(t,8,576,device='cuda',dtype=torch.float16)*.3
 out=torch.empty(t,8,512,device='cuda',dtype=torch.float16);ws=PairWorkspace(q,ids,group)
 def run():ws(q,pool,table,ids,pos,ctx,out)
 def check(stage):
  cache=pool[table].flatten(0,1).float();ref=[]
  for row in range(t):
   ix=ids[row];ix=ix[(ix>=0)&(ix<int(ctx))&(ix<=pos[row])]
   ref.append((q[row].float()@cache[ix].T/16).softmax(-1)@cache[ix,:512] if ix.numel() else torch.zeros(8,512,device='cuda'))
  ref=torch.stack(ref);err=float((out.float()-ref).norm()/ref.norm().clamp_min(1e-20))
  assert torch.isfinite(out).all() and err<.001,(rank,t,stage,err)
  if rank==0:print(json.dumps(dict(t=t,stage=stage,rel=err)),flush=True)
 for _ in range(2):run()
 torch.cuda.synchronize();check('eager');dist.barrier()
 g=torch.cuda.CUDAGraph()
 with torch.cuda.graph(g):run()
 q.mul_(1.1);ctx.sub_(3);pos.sub_(2);g.replay();torch.cuda.synchronize();check('graph_metadata')
 pool.mul_(.9);table.copy_(table.roll(1));ids.copy_(ids.roll(1,1));q.neg_()
 g.replay();torch.cuda.synchronize();check('graph_data')
 ctx.zero_();g.replay();torch.cuda.synchronize();assert torch.equal(out,torch.zeros_like(out));check('empty')
 g.reset();dist.barrier()
Path(args.output+f'.rank{rank}.json').write_text(json.dumps(dict(rank=rank,ALL_PASS=True,cases=6)))
print('ALL_PASS rank',rank,flush=True)
dist.destroy_process_group()
