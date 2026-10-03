"""Real GLM53 strategy queue test; no HTTP transport."""
import os,sys,json,time,threading,traceback
from pathlib import Path
from datetime import timedelta
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
import torch.distributed as dist
from model.glm53_generate import Generator
from model.glm53_resident import ResidentBatch
from strategy import glm53_strategy as strategy
R=Path(os.environ['BATCH_AUDIT_DIR']);rank=int(os.environ['LOCAL_RANK'])
torch.cuda.set_device(rank);dist.init_process_group('nccl',timeout=timedelta(minutes=15))
strategy._control=dist.new_group(backend='gloo',timeout=timedelta(minutes=15))
strategy._generator=Generator.load(capacity=8192,prefill_chunk_tokens=12288)
strategy._resident=ResidentBatch(strategy._generator);strategy._resident.warm()
from tokenizers import Tokenizer
from jinja2 import Environment
tok=Tokenizer.from_file('/mnt/data2/kw/GLM-5.3-UNCENSORED-FP8/tokenizer.json')
tpl=Environment().from_string((Path(__file__).resolve().parents[1]/'server/chat_template.jinja').read_text())
texts=['Write a thread-safe Python LRU cache with tests.',
       'Explain ACID and database isolation levels with examples.',
       'Explain DNS resolution, HTTPS and TLS 1.3 with examples.',
       'Write a TypeScript mapLimit function preserving order with tests.']
inputs=[tok.encode(tpl.render(messages=[{'role':'user','content':t}],tools=[],add_generation_prompt=True,enable_thinking=False,reasoning_effort='high',clear_thinking=True),add_special_tokens=False).ids for t in texts]
# Independent direct-generation reference with identical cohort and limits.
baseline={}
for b in (2,3,4):
 strategy._generator.cache.clear()
 baseline[b]=[x.token_ids for x in strategy._resident.generate(inputs[:b],[48]*b)]
strategy._generator.cache.clear()
cancel_reference=[x.token_ids for x in strategy._resident.generate(inputs[1:3],[32,32])]

original=strategy._resident.generate;batch_runs=[];clients=[];errors=[]
def traced(*args,**kwargs):
 rounds=[];kwargs['on_round']=lambda row:rounds.append(row)
 strategy._generator.cache.clear()  # Match the independent cold reference.
 result=original(*args,**kwargs)
 batch_runs.append(dict(batch=len(args[0]),rounds=rounds))
 return result
strategy._resident.generate=traced

def run_clients():
 try:
  for b in [2,3,4]:
   # Enqueue the whole cohort before the consumer is notified by the last put.
   with strategy._jobs.mutex:
    queues=[]
    for i in range(b):
     from queue import Queue
     q=Queue();q.cancel_handle=h=strategy.Handle();q.request_id=h.request_id
     strategy._jobs.queue.append((inputs[i],48,q,h));queues.append(q)
    strategy._jobs.not_empty.notify()
   row=[]
   for i,q in enumerate(queues):
    tokens=[];metrics=None
    while True:
     item=q.get(timeout=600)
     if item['type']=='prefill':metrics=item['metrics']
     if item['type']=='token':tokens+=item['token_ids']
     if item['type']=='end':break
    assert item['batch_size']==b,(b,item)
    assert metrics is not None and metrics['batch_size']==b,metrics
    # Earlier requests in this cohort publish the shared system prefix.
    assert 0 <= metrics['cache_hit_tokens'] < len(inputs[i]),metrics
    assert (metrics['cache_hit_tokens'] == 0) if i == 0 else (metrics['cache_hit_tokens'] > 0),metrics
    assert metrics['prefill_tokens']+metrics['cache_hit_tokens']==len(inputs[i]),metrics
    assert metrics['model_prefill_seconds']>=0,metrics
    assert tokens==baseline[b][i],(b,i,'tokens')
    row.append(dict(row=i,tokens=tokens,end=item))
   clients.append(dict(batch=b,rows=row));print('SERVICE_BATCH',b,flush=True)
  # Use the public query API and cancel one row before service; other rows continue.
  strategy.BATCH_WAIT_SECONDS=0.05
  qs=[strategy.query(inputs[i],32) for i in range(3)]
  qs[0].cancel_handle.cancel()
  row=[]
  for i,q in enumerate(qs):
   tokens=[]
   while True:
    item=q.get(timeout=600)
    if item['type']=='token':tokens+=item['token_ids']
    if item['type']=='end':break
   if i==0:assert item['cancelled'] and not tokens,item
   else:assert tokens==cancel_reference[i-1],(i,'cancel peer')
   row.append(dict(row=i,tokens=tokens,end=item))
  clients.append(dict(cancel=True,rows=row))
  # Semantic stop is not cancellation, including before initial admission.
  q=strategy.query(inputs[0],32)
  q.cancel_handle.stop_at_semantic_eos()
  while True:
   item=q.get(timeout=600)
   if item['type']=='end':break
  assert item['finish_reason']=='semantic_eos' and not item['cancelled'],item
  clients.append(dict(semantic_stop=True,end=item))

 except Exception:errors.append(traceback.format_exc())
 finally:strategy._jobs.put(None)

if rank==0:
 thread=threading.Thread(target=run_clients);thread.start()
strategy.worker_loop()
(R/f'service.rank{rank}.json').write_text(json.dumps(batch_runs,indent=2))
if rank==0:
 thread.join();(R/'service_clients.json').write_text(json.dumps(dict(clients=clients,errors=errors),indent=2));assert not errors,errors
print('SERVICE_DONE',rank,flush=True)
