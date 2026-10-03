import os,sys,json,hashlib
from pathlib import Path
import torch,torch.distributed as dist
from transformers import AutoTokenizer
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from model.glm53_engine import Engine
rank=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(rank);dist.init_process_group('nccl')
e=Engine.load(capacity=2048,prefill_chunk_tokens=16)
tok=AutoTokenizer.from_pretrained('/mnt/data2/kw/GLM-5.3-UNCENSORED-FP8',local_files_only=True)
cases=['The capital of France is Paris. Paris is known for the Eiffel Tower. The city has many museums and parks. Visitors enjoy its history and architecture.', 'Water freezes at zero degrees Celsius. When heated to one hundred degrees Celsius at sea level, water boils and becomes steam. This is a physical change.', 'Python is a programming language. A function takes arguments and returns a result. Lists store multiple values in an ordered sequence and support indexing.']
records=[]
for text in cases:
 ids=tok.encode(text,add_special_tokens=False);assert len(ids)>=27
 prompt=torch.tensor(ids[:16],device=e.device);a=torch.tensor(ids[16:24],device=e.device);b=torch.tensor(ids[19:27],device=e.device)
 e.reset();e.prefill(prompt);r=e.verify(a);e.commit(3)
 r=e.verify(b);baseline=r.logits.clone();e.commit(0)
 hashes=[]
 for _ in range(16):
  r=e.verify(b)
  blob=b''.join(t.contiguous().view(torch.uint8).cpu().numpy().tobytes() for t in (r.logits,)+r.features)
  hashes.append(hashlib.sha256(blob).hexdigest());e.commit(0)
 assert len(set(hashes))==1,'non-deterministic identical input'
 e.reset();e.prefill(prompt);e.prefill(a[:3]);r=e.verify(b);cross=r.logits.clone();e.commit(0)
 assert torch.isfinite(cross).all() and torch.isfinite(baseline).all()
 lp=baseline.log_softmax(-1);lq=cross.log_softmax(-1);kl=(lp.exp()*(lp-lq)).sum(-1)
 rec=dict(text=text,distinct=len(set(hashes)),hash=hashes[0],top1_agreement=float((baseline.argmax(-1)==cross.argmax(-1)).float().mean()),kl_mean=float(kl.mean()),kl_max=float(kl.max()),rel=float((baseline-cross).norm()/baseline.norm()),base_tokens=tok.batch_decode(baseline.argmax(-1).tolist()),cross_tokens=tok.batch_decode(cross.argmax(-1).tolist()))
 records.append(rec)
 if rank==0:print(json.dumps(rec),flush=True)
r=Path('/mnt/data2/kw/glm53_int4_tp8/engine_q8_audit/text');r.mkdir(exist_ok=True)
(r/f'rank{rank}.json').write_text(json.dumps(dict(status='MEASURED',records=records),indent=2))
dist.barrier(device_ids=[rank]);dist.destroy_process_group()
