"""Standalone exact selection validation/Graph A-B benchmark; no score/TP comm."""
import argparse,json,os,sys,statistics
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from ops.sparse_topk_v2 import extension,workspace,select_out
from ops.sparse_index_decode import extension as old_extension
p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--bench',action='store_true');p.add_argument('--quick',action='store_true');a=p.parse_args()
rank=int(os.environ.get('LOCAL_RANK','0'));torch.cuda.set_device(rank)
world=int(os.environ.get('WORLD_SIZE','1'))
if world>1:
 import torch.distributed as dist
 from datetime import timedelta
 dist.init_process_group('nccl',timeout=timedelta(minutes=20),device_id=torch.device('cuda',rank))
 if rank==0:extension();old_extension()
 dist.barrier()
mod=extension();old=old_extension();records=[]
def note(x):
 records.append(x);Path(a.output+f'.rank{rank}.json').write_text(json.dumps(records,indent=2))
 if rank==0:print(json.dumps(x),flush=True)
def oracle(s,pos,k):
 # Integer ordered IEEE keys, stable low-ID tie break; independent full sort.
 bits=s.view(torch.int32).long()&0xffffffff
 keys=torch.where((bits&0x80000000)!=0,(~bits)&0xffffffff,bits|0x80000000)
 valid=torch.arange(s.shape[1],device=s.device)[None,:]<=pos[:,None]
 keys=keys.masked_fill(~valid,-1)
 idx=keys.argsort(dim=1,descending=True,stable=True)[:,:k]
 return idx.masked_fill(~valid.gather(1,idx),-1).int().sort(dim=1).values

def check(s,pos,out,k,tag):
 exp=oracle(s,pos,k);got=out.sort(dim=1).values
 assert torch.equal(got,exp),(tag,rank,s.shape,((got!=exp).sum().item()),got[0,:16].tolist(),exp[0,:16].tolist())

def values(r,n,kind):
 s=torch.randn((r,n),device='cuda')
 if kind=='ties':s=torch.randint(-3,4,(r,n),device='cuda').float()
 elif kind=='flat':s.fill_(1)
 elif kind=='narrow':s=1+s*1e-6
 elif kind=='overflow':s=s*1e8
 elif kind=='tiny':s=s*1e-38
 elif kind=='infinity':s[:,::5]=float('inf');s[:,1::7]=-float('inf')
 elif kind=='zeros':s.zero_();s[:,::2]=-0.0
 elif kind=='ramp':s.copy_(torch.linspace(-10,10,n,device='cuda').expand(r,-1))
 elif kind=='positive':s=s.abs()*4+10
 return s

torch.manual_seed(891+rank)
cases=[(6,257,64),(6,8192,2048),(6,65542,2048),(1,1048576,2048),(3,65,65)]
kinds=['normal','ties','flat','narrow','overflow','tiny','infinity','zeros','ramp','positive']
if a.quick:cases=cases[:3];kinds=kinds[:4]
for r,n,k in cases:
 for kind in kinds:
  s=values(r,n,kind);pos=torch.full((r,),n-1,device='cuda',dtype=torch.int64)
  if r>1:pos[:min(4,r)]=torch.tensor([-1,0,k-1,k][:min(4,r)],device='cuda')
  out=torch.empty((r,k),device='cuda',dtype=torch.int32);w=workspace(r,n,device=s.device)
  select_out(s,pos,out,w);check(s,pos,out,k,kind)
  torch.cuda.synchronize();g=torch.cuda.CUDAGraph()
  with torch.cuda.graph(g):select_out(s,pos,out,w)
  for frac in [0.5,1.0]:
   s.copy_(values(r,n,kind));pos.fill_(int(n*frac)-1);out.fill_(-777)
   g.replay();torch.cuda.synchronize();check(s,pos,out,k,'graph_'+kind)
  g.reset()
  note(dict(test='oracle_dynamic_graph',rows=r,width=n,k=k,kind=kind,passed=True))
if a.bench:
 def capture(fn):
  for _ in range(3):fn()
  torch.cuda.synchronize();g=torch.cuda.CUDAGraph()
  with torch.cuda.graph(g):
   for _ in range(20):fn()
  for _ in range(3):g.replay()
  torch.cuda.synchronize();return g
 def measure(g):
  if world>1:dist.barrier()
  st=torch.cuda.Event(enable_timing=True);en=torch.cuda.Event(enable_timing=True)
  st.record()
  for _ in range(10):g.replay()
  en.record();en.synchronize();us=st.elapsed_time(en)*1000/200
  if world>1:
   z=torch.tensor(us,device='cuda');dist.all_reduce(z,op=dist.ReduceOp.MAX);us=z.item()
  return us
 for n in [8192,65536,81920,1048576]:
  for r in [1,6,8]:
   k=2048;s=values(r,n,'normal');pos=torch.full((r,),n-1,device='cuda',dtype=torch.int64)
   out=torch.empty((r,k),device='cuda',dtype=torch.int32);prev=torch.empty_like(out);w=workspace(r,n,device=s.device)
   old.select_out(s,pos,prev);select_out(s,pos,out,w);check(s,pos,out,k,'bench')
   assert torch.equal(out.sort(1).values,prev.sort(1).values)
   ga=capture(lambda:old.select_out(s,pos,prev));gb=capture(lambda:select_out(s,pos,out,w))
   out.fill_(-777);gb.replay();torch.cuda.synchronize();check(s,pos,out,k,'bench_replay')
   aa=[];bb=[]
   for rep in range(3):
    for name,g in [('old',ga),('v2',gb),('v2',gb),('old',ga)]:
     v=measure(g);(aa if name=='old' else bb).append(v)
   note(dict(test='benchmark',rows=r,width=n,k=k,old_us=statistics.median(aa),v2_us=statistics.median(bb),speedup=statistics.median(aa)/statistics.median(bb),old_samples_us=aa,v2_samples_us=bb,workspace_bytes=sum(v.numel()*v.element_size() for v in w),boundary_candidates=w[1][:,2].tolist(),timing='Graph device events; max ranks before sample median; ABBA; excludes scoring and communications'))
   ga.reset();gb.reset()
note(dict(ALL_PASS=True,world=world,gpu=torch.cuda.get_device_name()))
if world>1:dist.barrier();dist.destroy_process_group()
