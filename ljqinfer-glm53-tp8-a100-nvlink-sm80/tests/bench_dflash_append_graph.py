import os,sys,json,time,statistics
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
import torch.distributed as dist
from model.glm53_dflash import DFlash2
from model.glm53_mtp_pool import MTPKVPool
from model.glm53_dflash import AppendGraphs
rank=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(rank);dist.init_process_group('nccl')
device=torch.device('cuda',rank);pool=MTPKVPool(max_tokens=4096,max_sequence_tokens=4096,device=device)
pool.reserve(0,4096)
pool.host_page_table[0].reverse();pool.page_table[0].copy_(torch.tensor(pool.host_page_table[0],device=device))
d=DFlash2(SimpleNamespace(device=device,rank=rank,mtp_kv=pool));c=AppendGraphs(d)
for n in range(1,9):c.build(n)
rows=[]
with torch.inference_mode():
 for n in range(1,9):
  torch.manual_seed(100+n)
  features=tuple(torch.randn((n,6144),device=device) for _ in range(6))
  for start in [0,63,2040,4088]:
   result=SimpleNamespace(start=start,features=features)
   pool.k.zero_();pool.v.zero_();pool.lengths[0]=start;d.append(result)
   refk=pool.k.clone();refv=pool.v.clone()
   pool.k.zero_();pool.v.zero_();pool.lengths[0]=start;c(result)
   assert torch.equal(refk,pool.k) and torch.equal(refv,pool.v),(rank,n,start)
  result=SimpleNamespace(start=63,features=features);times=[]
  for fn in [d.append,c]:
   samples=[]
   for _ in range(12):
    pool.lengths[0]=63;torch.cuda.synchronize();t=time.perf_counter();fn(result);torch.cuda.synchronize();samples.append((time.perf_counter()-t)*1000)
   times.append(statistics.median(samples[2:]))
  vals=torch.tensor(times,device=device);dist.all_reduce(vals,op=dist.ReduceOp.MAX)
  row=dict(n=n,ms=vals.tolist(),exact=True);rows.append(row)
  if rank==0:print(row,flush=True)
(Path(os.environ.get('DECODE_AUDIT_DIR','/mnt/data2/kw/glm53_int4_tp8/service_audit/decode50'))/f'append_production.rank{rank}.json').write_text(json.dumps(rows))
c.close();d.close();dist.barrier();dist.destroy_process_group()
