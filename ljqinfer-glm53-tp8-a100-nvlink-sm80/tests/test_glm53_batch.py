"""TP8 integration regression; run torchrun --standalone --nproc_per_node=8."""
import os,sys,json,time,hashlib
from pathlib import Path
from datetime import timedelta
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
import torch.distributed as dist
from tokenizers import Tokenizer
from jinja2 import Environment
from model.glm53_generate import Generator
from model.glm53_batch import BatchSession
R=Path(os.environ['BATCH_AUDIT_DIR']);R.mkdir(exist_ok=True,parents=True)
rank=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(rank)
dist.init_process_group('nccl',timeout=timedelta(minutes=15))
g=Generator.load(capacity=8192,prefill_chunk_tokens=12288)
tok=Tokenizer.from_file('/mnt/data2/kw/GLM-5.3-UNCENSORED-FP8/tokenizer.json')
template=Environment().from_string((Path(__file__).resolve().parents[1]/'server/chat_template.jinja').read_text())
texts=['Write a thread-safe Python LRU cache with tests.',
       'Explain ACID and database isolation levels with examples.',
       'Explain DNS resolution, HTTPS and TLS 1.3 with examples.',
       'Write a TypeScript mapLimit function preserving order with tests.']
inputs=[tok.encode(template.render(messages=[{'role':'user','content':t}],tools=[],add_generation_prompt=True,enable_thinking=False,reasoning_effort='high',clear_thinking=True),add_special_tokens=False).ids for t in texts]
records=[];baseline=[]
for prompt in inputs:
 g.reset();baseline.append(g.generate(prompt,max_new_tokens=128,eos_token_ids=[],use_graph=True).token_ids)
print('BASELINE_READY',rank,flush=True)

def run(name,indices,limits,**kwargs):
 prompts=[inputs[i] for i in indices]
 s=BatchSession(g,[len(p)+n+8 for p,n in zip(prompts,limits)])
 t=time.perf_counter();finished=[]
 results=s.generate(prompts,limits,on_done=lambda i,r:finished.append(i),**kwargs)
 torch.cuda.synchronize()
 record=dict(name=name,elapsed=time.perf_counter()-t,rounds=s.last_rounds,
             done_order=finished,rows=[dict(tokens=x.token_ids,steps=x.steps,reason=x.finish_reason,
               baseline_prefix_exact=x.token_ids==baseline[idx][:len(x.token_ids)])
               for idx,x in zip(indices,results)])
 records.append(record)
 (R/f'integration.rank{rank}.json').write_text(json.dumps(records,indent=2))
 print('CASE',name,rank,[x['baseline_prefix_exact'] for x in record['rows']],flush=True)
 # GEMM shape changes can alter greedy trajectories; B1 equivalence is diagnostic.
 assert all(x.token_ids and len(x.token_ids)<=limit for x,limit in zip(results,limits))
 assert all(x.finish_reason in ('length','eos','cancelled') for x in results)
 if name.startswith('B'):
  original=[x.token_ids for x in results]
  s.close();s=BatchSession(g,[len(p)+n+8 for p,n in zip(prompts,limits)])
  repeated=s.generate(prompts,limits,**kwargs)
  assert [x.token_ids for x in repeated]==original, (name,'same-batch nondeterminism')
 assert len(set(finished))==len(indices)==len(finished)
 assert all(e.length==d.pool.lengths[0] and e.pending is None for e,d in s.slots)
 s.close();return record

for b in [1,2,3,4]:
 run(f'B{b}',list(range(b)),[128]*b,eos_token_ids=[])
run('mixed_finish',[3,1,0,2],[1,17,41,64],eos_token_ids=[])
run('permuted_B2',[1,0],[48,48],eos_token_ids=[])
calls=[0]
def cancel():
 calls[0]+=1
 return [calls[0]>=4,False,False]
r=run('cancel_row',[0,1,2],[64]*3,eos_token_ids=[],should_stop=cancel)
assert r['rows'][0]['reason']=='cancelled' and len(r['rows'][0]['tokens'])<64
assert all(len(x['tokens'])==64 for x in r['rows'][1:])
r=run('eos_at_anchor',[0,1],[32,32],eos_token_ids=[baseline[0][0]])
assert len(r['rows'][0]['tokens'])==1 and r['rows'][0]['reason']=='eos'
# Return to B1 after all pooled slots have been released.
g.reset();after=g.generate(inputs[0],max_new_tokens=32,eos_token_ids=[],use_graph=True)
assert after.token_ids==baseline[0][:32]
(R/f'done.rank{rank}.json').write_text(json.dumps(dict(done=True,cases=len(records),baseline=baseline)))
g.close();dist.barrier(device_ids=[rank]);dist.destroy_process_group()
print('DONE',rank,flush=True)
