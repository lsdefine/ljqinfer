"""Eight-rank real-weight integration gate; no draft model claims."""
import os,sys,json,time,traceback
from pathlib import Path
from datetime import timedelta
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
import torch.distributed as dist
from model.glm53_engine import Engine
rank=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(rank)
out=Path(os.environ.get('AUDIT_DIR','/mnt/data2/kw/glm53_int4_tp8/engine_q8_audit'));out.mkdir(exist_ok=True)
def note(**kw):
 with (out/f'rank{rank}.jsonl').open('a') as f:f.write(json.dumps(kw)+'\n')
 print(rank,kw,flush=True)
try:
 dist.init_process_group('nccl',timeout=timedelta(minutes=20))
 engine=Engine.load(capacity=2048,prefill_chunk_tokens=16)
 note(stage='loaded',allocated=torch.cuda.memory_allocated(),dense_dtype=str(engine.w.layers[0].ffn.gu.dtype))
 prompt=torch.tensor([154820,1001,2002,3003,4004,5005,6006,7007,8008,9009,1010,2011,3012,4013,5014,6015],device=engine.device)
 def save_result(r):return r.logits.clone(),tuple(x.clone() for x in r.features)
 def check(name,a,b,tol=.03):
  assert torch.isfinite(a).all() and torch.isfinite(b).all(),name+' nonfinite'
  rel=float((a.float()-b.float()).norm()/b.float().norm().clamp_min(1e-9))
  cosine=float(torch.nn.functional.cosine_similarity(a.float().flatten(),b.float().flatten(),dim=0))
  note(test=name,rel=rel,cosine=cosine)
  assert rel<tol,(name,rel,tol)
 r=engine.prefill(prompt);base,feat=save_result(r)
 assert len(feat)==6 and all(x.shape==(16,6144) for x in feat)
 assert torch.isfinite(base).all()
 note(stage='prefill',length=engine.length,top=base[-1].topk(5).indices.tolist())
 tokens=torch.arange(1234,1242,device=engine.device)
 v=engine.verify(tokens);vlog,vfeat=save_result(v)
 assert engine.length==16
 engine.commit(0)
 altered=tokens.clone();altered[3:]=(altered[3:]+101)%154880
 changed=engine.verify(altered)
 check('causal_prefix_logits',vlog[:3],changed.logits[:3],tol=1e-6)
 for j,(a,b) in enumerate(zip(vfeat,changed.features)):
  check('causal_prefix_features_'+str(j),a[:3],b[:3],tol=1e-6)
 engine.commit(3);assert engine.length==19
 second=torch.arange(2234,2242,device=engine.device)
 r=engine.verify(second);branch,branch_feat=save_result(r);engine.commit(0)
 assert engine.length==19
 engine.reset();assert engine.length==0 and engine.mtp_kv.lengths==[0]
 engine.prefill(prompt)
 pref=engine.prefill(tokens);plog,pfeat=save_result(pref)
 check('prefill_vs_verify_logits',vlog,plog)
 for i,(a,b) in enumerate(zip(vfeat,pfeat)):check('features_'+str(i),a,b)
 engine.reset();engine.prefill(prompt);engine.verify(tokens);engine.commit(3)
 rerun=engine.verify(second);check('reject_suffix_replay',branch,rerun.logits,tol=1e-6)
 engine.commit(0)
 engine.capture_verify();note(stage='captured',allocated=torch.cuda.memory_allocated())
 r=engine.verify(second);check('graph_vs_eager',r.logits,rerun.logits.clone())
 engine.commit(2)
 third=torch.arange(3234,3242,device=engine.device)
 r=engine.verify(third);graphlog=r.logits.clone();engine.commit(0)
 g=engine.graph;engine.graph=None
 r=engine.verify(third);check('graph_dynamic_positions_ids',graphlog,r.logits);engine.commit(0)
 engine.graph=g
 torch.cuda.synchronize();t=time.perf_counter()
 for _ in range(5):engine.verify(third);engine.commit(0)
 torch.cuda.synchronize();ms=(time.perf_counter()-t)*200
 note(stage='Q8_PASS',q=8,verify_wall_ms=ms,allocated=torch.cuda.memory_allocated())
 if os.environ.get('CHECK_CROSS_Q')=='1':
  engine.reset();engine.prefill(prompt);engine.prefill(tokens[:3])
  cross=engine.verify(second)
  check('cross_q3_prefix_recompute',branch,cross.logits)
  engine.commit(0)
 (out/f'rank{rank}.json').write_text(json.dumps(dict(status='PASS',q=8,verify_wall_ms=ms)))
 torch.cuda.synchronize();engine.graph=None;g.reset();del g
 dist.barrier(device_ids=[rank]);dist.destroy_process_group()
except Exception:
 note(stage='FAIL',traceback=traceback.format_exc());raise
