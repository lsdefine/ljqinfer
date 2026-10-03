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
from model.glm53_resident import ResidentBatch
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

s=ResidentBatch(g);s.warm();assert s.capture_count==4
records=[]
def run(name,dynamic=False,eager=False):
    g.cache.clear()  # Compare graph/eager and repeat from identical cold state.
    prompts=inputs[:2] if dynamic else inputs
    limits=[96,24] if dynamic else [96,24,48,64]
    admitted=[];fin=[];trace=[]
    def board(remaining,bucket):
        # Mid-decode admissions, not initial coalescing.
        if len(trace)<2 or len(admitted)>=2:return None
        i=2+len(admitted);admitted.append(i)
        return inputs[i],[48,64][i-2]
    before=s.capture_count;t=time.perf_counter()
    out=s.generate(prompts,limits,eos_token_ids=[],use_graph=not eager,
        board_request=board if dynamic else None,
        on_round=lambda rec:trace.append(rec),on_done=lambda i,r:fin.append(i))
    rec=dict(name=name,seconds=time.perf_counter()-t,captures=s.capture_count-before,
        rounds=trace,done=fin,rows=[dict(tokens=r.token_ids,steps=r.steps,reason=r.finish_reason) for r in out])
    records.append(rec);(R/f'resident.rank{rank}.json').write_text(json.dumps(records,indent=2))
    assert len(out)==4 and sorted(fin)==list(range(4)),rec
    assert all(len(r.token_ids)==n for r,n in zip(out,[96,24,48,64]))
    assert s.used_pages==0 and not s.row_ids and not s.busy
    if not eager:assert rec['captures']==0,rec
    if dynamic:assert admitted==[2,3] and {r['batch'] for r in trace}>={2,4}
    print('PASS',name,rank,rec['seconds'],flush=True)
    return out
first=run('fixed_graph')
repeat=run('repeat_graph')
assert [r.token_ids for r in first]==[r.token_ids for r in repeat]
dyn=run('dynamic_graph',True)
dyn_eager=run('dynamic_eager',True,True)
assert [r.token_ids for r in dyn]==[r.token_ids for r in dyn_eager]
# Cancel an active row, compact, board, then reuse the executor across epochs.
checks=[0];ends=[]
def cancel():
 checks[0]+=1
 return [False,checks[0]>=3,False]
res=s.generate(inputs[:3],[40,40,1],eos_token_ids=[],should_stop=cancel,on_done=lambda i,r:ends.append(i))
assert res[1].finish_reason=='cancelled' and len(res[2].token_ids)==1
assert sorted(ends)==[0,1,2] and s.capture_count==4
run('after_cancel')
res=s.generate(inputs[:2],[32,32],eos_token_ids=[],should_stop=lambda:[True,True])
assert all(not x.token_ids and x.finish_reason=='cancelled' for x in res)
run('after_pre_cancel')
# Same B1 graph across context lengths and different physical leases.
assert s.frozen and len(s.graphs)==4 and all(d.graph is None for _,d in s.slots)
metrics=[]
long_prompt=inputs[0]*80
assert 2048<len(long_prompt)<7000,len(long_prompt)
before=s.capture_count
long1=s.generate([long_prompt],[8],eos_token_ids=[],on_prefill=lambda i,m:metrics.append(m))
long2=s.generate([long_prompt],[8],eos_token_ids=[],on_prefill=lambda i,m:metrics.append(m))
assert len(long1[0].token_ids)==len(long2[0].token_ids)==8
assert metrics[-1]['cache_hit_tokens']==len(long_prompt)-1 and metrics[-1]['prefill_tokens']==1,metrics
assert s.capture_count==before==4 and not s.last_rounds
# The same request also executes at a nonzero page lease.
moved=s.generate([inputs[1],long_prompt],[8,8],eos_token_ids=[],on_prefill=lambda i,m:metrics.append(m))
assert metrics[-1]['cache_hit_tokens']==len(long_prompt)-1 and metrics[-1]['prefill_tokens']==1,metrics
assert s.capture_count==4 and not s.busy
(R/f'cache.rank{rank}.json').write_text(json.dumps(metrics,indent=2))
s.close();g.close();dist.barrier();print('RESIDENT_PASS',rank,flush=True)
dist.destroy_process_group()
