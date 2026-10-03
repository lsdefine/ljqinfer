from pathlib import Path
import sys,json
from types import SimpleNamespace
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from model.glm53_cache import PrefixState
from model.glm53_mtp_pool import MTPKVPool
torch.cuda.set_device(0)
pool=MTPKVPool(max_tokens=1024,max_sequence_tokens=1024,layers=2,device='cuda')
pool.reserve(0,1024)
e=SimpleNamespace(capacity=1024,kv=torch.empty((2,16,64,576),device='cuda',dtype=torch.float16),index=torch.empty((1,16,64,128),device='cuda',dtype=torch.float16),mtp_kv=pool,length=0,pending=None)
c=PrefixState(e)
pointers={k:v.data_ptr() for k,v in c.fields.items()}
def fill(start,end,seed):
 for j,(k,v) in enumerate(c.fields.items()):
  vals=(torch.arange((end-start)*v[0].numel(),device='cuda').reshape(end-start,*v.shape[1:])+seed+j)%1024
  v[start:end].copy_(vals)
def save(tokens):
 e.length=pool.lengths[0]=len(tokens)
 c.publish(tokens)
 return {k:v[:len(tokens)].clone() for k,v in c.fields.items()}
def restore(tokens,reference,expected,source):
 hit,src=c.restore(tokens)
 assert (hit,src)==(expected,source),(hit,src,expected,source)
 for k,v in c.fields.items():
  assert torch.equal(v[:hit].view(torch.uint8),reference[k][:hit].view(torch.uint8)),k
  assert v.data_ptr()==pointers[k]
 assert e.length==pool.lengths[0]==hit
 assert pool.host_page_table[0]==list(range(pool.logical_pages))
 torch.cuda.synchronize()
 print(json.dumps(dict(hit=hit,source=src,fields=len(c.fields),bit_exact=True)),flush=True)
a=list(range(70));fill(0,70,1);ra=save(a)
restore(a+[9000],ra,70,'hot')
# Destroy every resident row, forcing retrieval from host rather than recompute.
c.invalidate()
for v in c.fields.values():v.fill_(-7)
restore(a+[9000],ra,70,'cold')
b=a[:65]+list(range(1000,1021))
restore(b,ra,65,'hot');fill(65,len(b),21);rb=save(b)
c.invalidate()
for v in c.fields.values():v.fill_(-11)
restore(a+[9000],ra,70,'cold')
c.invalidate();restore(b+[9000],rb,len(b),'cold')
c.invalidate();restore(b[:33]+[9000],rb,33,'cold')
# Extend an existing leaf and verify segmented restore.
restore(b+[9000],rb,len(b),'cold')
d=b+list(range(2000,2080));fill(len(b),len(d),73);rd=save(d)
c.invalidate()
for v in c.fields.values():v.zero_()
restore(d+[9000],rd,len(d),'cold')
# A resident request maps target/index from shared pools and draft from its
# own page table. Restore the same cached prefix at a nonzero physical lease.
from copy import copy
for offset in (4, 8):
 request=copy(e);request.cache_page_span=(offset,4)
 request.mtp_kv=copy(pool)
 request.mtp_kv.host_page_table=[list(range(offset,offset+4))]
 request.mtp_kv.lengths=[0]
 bound=c.for_engine(request)
 before={k:v.clone() for k,v in c.fields.items()}
 hit,src=bound.restore(d+[9000])
 assert (hit,src)==(len(d),'cold'),(hit,src)
 torch.cuda.synchronize()
 for k,v in bound.fields.items():
  assert torch.equal(v[:hit].view(torch.uint8),rd[k].view(torch.uint8)),k
  whole=c.fields[k];lo=offset*64;hi=lo+hit
  assert torch.equal(whole[:lo],before[k][:lo]),('left overwrite',k)
  assert torch.equal(whole[hi:],before[k][hi:]),('right overwrite',k)
 assert request.length==request.mtp_kv.lengths[0]==len(d)
 print(json.dumps(dict(lease_offset=offset,hit=hit,bit_exact=True,isolated=True)),flush=True)
 # Extend this request and publish from its own physical location.
 extra=d+[3000+offset]
 for j,v in enumerate(bound.fields.values()):v[len(d)].fill_(offset+j)
 request.length=request.mtp_kv.lengths[0]=len(extra)
 bound.publish(extra)
 expected={k:v[:len(extra)].clone() for k,v in bound.fields.items()}
 c.invalidate();restored,_=c.restore(extra+[9999]);torch.cuda.synchronize()
 assert restored==len(extra)
 for k,v in c.fields.items():assert torch.equal(v[:restored],expected[k]),k
c.clear();hit,src=c.restore(a);assert (hit,src)==(0,'miss')
print('CACHE_GEOMETRY_PASS')
