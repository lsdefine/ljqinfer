import os,sys,json,time,statistics
from pathlib import Path
from datetime import timedelta
sys.path.insert(0,'/mnt/data/kw/ljqinfer_glm53_tp8')
import torch
import torch.distributed as dist
from tokenizers import Tokenizer
from model.glm53_engine import Engine
R=Path('/mnt/data2/kw/glm53_int4_tp8/service_audit')
label=sys.argv[1];rank=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(rank)
def note(**x):
 with (R/f'engine_history_{label}.rank{rank}.jsonl').open('a') as f:f.write(json.dumps(x)+'\n')
 if rank==0:print(x,flush=True)
def check(tag,z):
 vals=[z.logits,*z.features]
 path=R/f'engine_history_baseline.{tag}.rank{rank}.pt'
 if label=='baseline':torch.save([v.cpu() for v in vals],path)
 else:
  ref=torch.load(path,weights_only=True,map_location='cpu')
  eq=[torch.equal(a,b.cpu()) for a,b in zip(ref,vals)]
  diff=[(a.float()-b.cpu().float()).abs().max().item() for a,b in zip(ref,vals)]
  note(stage='correctness',tag=tag,equal=eq,maxdiff=diff)
  assert all(eq),(tag,eq,diff)
@torch.inference_mode()
def run():
 dist.init_process_group('nccl',timeout=timedelta(minutes=20))
 e=Engine.load(capacity=((144*1024+8+63)//64)*64,prefill_chunk_tokens=12288)
 tok=Tokenizer.from_file('/mnt/data2/kw/GLM-5.3-UNCENSORED-FP8/tokenizer.json')
 body=json.loads((R/'fix_request.json').read_text());text='\n'.join(m['content'] for m in body['messages'])
 tokens=tok.encode(text).ids
 ids=torch.tensor((tokens*((12288+len(tokens)-1)//len(tokens)))[:12288],device='cuda',dtype=torch.int64)
 note(stage='loaded')
 def forward():
  e.reset();return e.prefill(ids,last_logits_only=True)
 for i in range(2):
  z=forward();torch.cuda.synchronize();del z;note(stage='warm',iteration=i)
 samples=[]
 for i in range(5):
  dist.barrier();torch.cuda.synchronize()
  a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)
  start=time.perf_counter();a.record();z=forward();b.record();b.synchronize()
  v=torch.tensor([time.perf_counter()-start,a.elapsed_time(b)/1000],device='cuda',dtype=torch.float64)
  dist.all_reduce(v,op=dist.ReduceOp.MAX);samples.append(v.tolist());note(stage='sample',iteration=i,rankmax=v.tolist())
  if i==4:check('12k',z)
  del z
 note(stage='timing',median_wall=statistics.median(x[0] for x in samples),median_gpu=statistics.median(x[1] for x in samples),samples=samples,prefill_tps=12288/statistics.median(x[0] for x in samples))
 z=e.prefill(ids[:33],last_logits_only=True);check('tail33',z);del z
 z=e.verify(ids[:8]);check('verify8',z);del z;e.commit(0)
 e.reset();z=e.prefill(ids[:17],last_logits_only=True);check('first17',z);del z
 note(stage='DONE',ALL_PASS=True,allocated=torch.cuda.memory_allocated(),reserved=torch.cuda.memory_reserved(),scope='Engine.prefill 12288 history0; subsequent33,verify8,first17 correctness; no HTTP')
 dist.barrier();dist.destroy_process_group()
if __name__=='__main__':run()
