import os,sys,json,time,statistics,runpy
from pathlib import Path
from datetime import timedelta
os.environ['TORCH_CUDA_ARCH_LIST']='8.0'
sys.path.insert(0,'/mnt/data/kw/ljqinfer_glm53_tp8')
import torch
import torch.distributed as dist
from model.glm53_block import TransformerBlock,SparseAttentionBinding
from model.glm53_layer_weights import load_attention,load_indexer
from ops.sparse_index_tp import TPIndexParallel
from ops.sparse_mla_pair import create_pair_group
import ops.sparse_mla_cuda as mla
r=Path('/mnt/data2/kw/glm53_int4_tp8/service_audit')
rank=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(rank);torch.set_num_threads(2)
dist.init_process_group('nccl',timeout=timedelta(minutes=10))
torch.backends.cuda.matmul.allow_tf32=False
from ops.sparse_mla_prefill import extension
old=new=extension()
mods=[old,new];records=[]
def note(**kw):
 records.append(kw);(r/f'v_projected_integrated.rank{rank}.json').write_text(json.dumps(records,indent=2))
 if rank==0:print(json.dumps(kw),flush=True)
root='/mnt/data2/kw/GLM-5.3-UNCENSORED-FP8'
cfg=json.loads((Path(root)/'config.json').read_text());counts={k:cfg['indexer_types'].count(k) for k in ['full','shared']};assert counts==dict(full=21,shared=57)
a=load_attention(root,10,rank,'cuda');b=load_attention(root,12,rank,'cuda');w=load_indexer(root,10,rank,'cuda')
parallel=TPIndexParallel();group=create_pair_group();t=n=12288;page=64
torch.manual_seed(932)
table=torch.randperm(n//page,device='cuda',dtype=torch.int64)
ip=torch.randn(n//page,page,128,device='cuda',dtype=torch.float16)*.2
kp=torch.randn(n//page,page,576,device='cuda',dtype=torch.float16)*.1;sp=kp.clone()
x=torch.randn(t,a['norm'].numel(),device='cuda',dtype=torch.float16)*.2
pos=torch.arange(t,device='cuda',dtype=torch.int64);ctx=torch.tensor([n],device='cuda',dtype=torch.int64)
f=SparseAttentionBinding(tokens=t,capacity=n,parallel=parallel,pair_group=group,index_weights=w,index_pool=ip,index_table=table)
s=SparseAttentionBinding(tokens=t,capacity=n,parallel=parallel,shared_from=f);s.pair=f.pair
f.latent=torch.empty((t,8,512),device='cuda',dtype=torch.float16);s.latent=f.latent
blocks=[TransformerBlock(aa,None,None,prefill_op=None,decode_op=None,all_reduce=dist.all_reduce,sparse=ss) for aa,ss in [(a,f),(b,s)]]
from model.glm53_workspace import IndexBuffers
index_owner=IndexBuffers(t,n,'prefill','cuda')
index_view=index_owner.view(t,n)
original_pair=f.pair

import copy,inspect,textwrap
import model.glm53_block as mb
from ops.prefill_elementwise import ElementwiseFusion
fusion=ElementwiseFusion(mb._rms_apply,t,'cuda')
f.prefill_elementwise=s.prefill_elementwise=fusion
f.index_scratch=s.index_scratch=index_view

from ops.sparse_mla_pair import ProjectedPair
cf=copy.copy(f);cs=copy.copy(s)
cs.shared_from=cf
cf.projected_pair=ProjectedPair(original_pair,a['v_b'])
cs.projected_pair=ProjectedPair(original_pair,b['v_b'],(cf.projected_pair.send,cf.projected_pair.recv))
bindings=[[f,s],[cf,cs]]

def run(kind,which):
 blocks[kind].sparse=bindings[which][kind]
 return blocks[kind].attention(x,pos,[kp,sp][kind],table,ctx,phase='prefill')

with fusion.chunk(pos,f.inv_freq,first_chunk=True):
 for kind,label in enumerate(['full','shared']):
  y0=run(kind,0).clone();idx=f.ids.clone();cache0=[kp,sp][kind].clone();ip0=ip.clone()
  y=run(kind,1)
  checks=dict(output=torch.equal(y0,y),ids=torch.equal(idx,f.ids),kv=torch.equal(cache0,[kp,sp][kind]),index_kv=torch.equal(ip0,ip))
  note(stage='correctness',kind=label,checks=checks,maxdiff=(y0.float()-y.float()).abs().max().item())
  assert all(checks.values()),checks
sequence=[0]*21+[1]*57

def chunk(which):
 with fusion.chunk(pos,f.inv_freq,first_chunk=True):
  for kind in sequence:run(kind,which)
for which in [0,1]:chunk(which)
torch.cuda.synchronize()
samples=[[],[]];walls=[[],[]]
for rep in range(3):
 for which in ([0,1] if rep%2==0 else [1,0]):
  dist.barrier();torch.cuda.synchronize()
  st=torch.cuda.Event(enable_timing=True);en=torch.cuda.Event(enable_timing=True)
  begin=time.perf_counter();st.record();chunk(which);en.record();en.synchronize()
  value=torch.tensor([st.elapsed_time(en),(time.perf_counter()-begin)*1000],device='cuda')
  dist.all_reduce(value,op=dist.ReduceOp.MAX)
  ms,wall=value.tolist();samples[which].append(ms);walls[which].append(wall)
  note(stage='sample',rep=rep,variant=which,ms=ms)
note(ALL_PASS=True,scope='78 independent attention calls, 21 full+57 shared, first 12288 chunk; not model forward',variants=['103c842','V-before-output-exchange'],medians_ms=[statistics.median(v) for v in samples],rankmax_samples=samples,rankmax_wall_samples=walls)
dist.barrier();dist.destroy_process_group()
