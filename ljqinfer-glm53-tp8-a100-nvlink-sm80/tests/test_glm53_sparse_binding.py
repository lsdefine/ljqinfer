"""TP8 real-weight sparse layer attention validation, not full-model logits."""
import os,sys,json
from pathlib import Path
from datetime import timedelta
import torch
import torch.distributed as dist
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from model.glm53_block import TransformerBlock,SparseAttentionBinding
from model.glm53_layer_weights import load_attention,load_indexer
from ops.sparse_index_tp import TPIndexParallel
from ops.sparse_mla_pair import create_pair_group
rank=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(rank)
dist.init_process_group('nccl',timeout=timedelta(minutes=10));torch.set_num_threads(2)
torch.backends.cuda.matmul.allow_tf32=False
root='/mnt/data2/kw/GLM-5.3-UNCENSORED-FP8'
report=Path(os.environ['REPORT']);records=[]
def note(**kw):
 records.append(kw);Path(str(report)+f'.rank{rank}.json').write_text(json.dumps(records,indent=2))
 if rank==0:print(json.dumps(kw),flush=True)
a=load_attention(root,10,rank,'cuda');b=load_attention(root,12,rank,'cuda')
w=load_indexer(root,10,rank,'cuda');assert load_indexer(root,12,rank,'cuda') is None
note(test='real_weights_loaded',passed=True)
# Independent torch reference: explicit complex RoPE and per-head attention.
def norm(x,weight):
 z=x.float();return (z/(z.square().mean(-1,keepdim=True)+1e-5).sqrt()*weight.float()).half()
def rope(z,pos):
 zc=torch.view_as_complex(z.float().reshape(*z.shape[:-1],32,2))
 angles=pos.double()[:,None]/(8000000.**(torch.arange(32,device='cuda',dtype=torch.float64)/32))[None,:]
 rot=torch.polar(torch.ones_like(angles),angles).to(torch.complex64)
 if z.ndim==3:rot=rot[:,None,:]
 return torch.view_as_real(zc*rot).flatten(-2).half()
def project(a,x,pos):
 xn=norm(x,a['norm']);qa=norm(xn@a['q_a'].T,a['q_a_norm'])
 qb=(qa@a['q_b'].T).reshape(-1,8,256)
 q=torch.stack([qb[:,h,:192]@a['k_b'][h].T for h in range(8)],1)
 q=torch.cat((q,rope(qb[:,:,192:],pos)),-1)
 raw=xn@a['kv_a'].T
 kv=torch.cat((norm(raw[:,:512],a['kv_a_norm']),rope(raw[:,512:],pos)),-1)
 return xn,qa,q,kv
def reference(a,x,pos,ctx,pool,table,ids):
 _,_,q,rows=project(a,x,pos)
 physical=table[pos//64]*64+pos%64
 torch.testing.assert_close(pool.view(-1,576)[physical],rows,rtol=1e-3,atol=1e-3)
 vals=[]
 for i in range(len(x)):
  ix=ids[i];ix=ix[(ix>=0)&(ix<=pos[i])&(ix<ctx)]
  kv=pool.view(-1,576)[table[ix//64]*64+ix%64].float()
  logits=q[i].float()@kv.T/16
  vals.append((logits.softmax(-1)@kv[:,:512]).half())
 latent=torch.stack(vals)
 heads=torch.stack([latent[:,h]@a['v_b'][h].T for h in range(8)],1)
 out=heads.reshape(len(x),2048)@a['o'].T;dist.all_reduce(out)
 return (x+out).half()
parallel=TPIndexParallel()
pair_group=create_pair_group() if os.environ.get("PAIR")=="1" else None
for t in [1,6,8]:
 n=int(os.environ.get("CAPACITY","4096"));page=64;torch.manual_seed(920+t)
 table=torch.randperm(n//page,device='cuda',dtype=torch.int64)
 ip=torch.randn(n//page,page,128,device='cuda',dtype=torch.float16)*.2
 kp=torch.randn(n//page,page,576,device='cuda',dtype=torch.float16)*.1;sp=kp.clone()
 x=torch.randn(t,a['norm'].numel(),device='cuda',dtype=torch.float16)*.2
 pos=torch.arange(n-t-5,n-5,device='cuda',dtype=torch.int64);ctx=torch.tensor([n-5],device='cuda',dtype=torch.int64)
 full=SparseAttentionBinding(tokens=t,capacity=n,parallel=parallel,pair_group=pair_group,index_weights=w,index_pool=ip,index_table=table)
 shared=SparseAttentionBinding(tokens=t,capacity=n,parallel=parallel,pair_group=pair_group,shared_from=full)
 def block(attn,binding):
  return TransformerBlock(attn,None,None,prefill_op=None,decode_op=None,all_reduce=dist.all_reduce,sparse=binding)
 f=block(a,full);s=block(b,shared)
 assert full.ids.data_ptr()==shared.ids.data_ptr()
 def run(phase):
  y=f.attention(x,pos,kp,table,ctx,phase=phase)
  z=s.attention(x,pos,sp,table,ctx,phase=phase)
  return y,z
 def check(phase,y,z):
  xn,qa,_,_=project(a,x,pos)
  iq=(qa@w['wq_b'].T).view(t,4,128);iq=torch.cat((rope(iq[:,:,:64],pos),iq[:,:,64:]),-1)
  ik=torch.nn.functional.layer_norm(xn@w['wk'].T,(128,),w['k_norm'],w['k_bias'],1e-6)
  ik=torch.cat((rope(ik[:,:64],pos),ik[:,64:]),-1)
  torch.testing.assert_close(ip.view(-1,128)[table[pos//64]*64+pos%64],ik,rtol=1e-3,atol=1e-3)
  iw=(xn.float()@w['weights_proj'].T)*(32**-.5)
  keys=ip[table].reshape(n,128).float()
  scores=torch.einsum('thd,nd->thn',iq.float(),keys).relu().mul(iw[:,:,None]).sum(1)*(128**-.5)
  dist.all_reduce(scores)
  mask=(torch.arange(n,device='cuda')[None,:]>pos[:,None])|(torch.arange(n,device='cuda')[None,:]>=ctx)
  scores.masked_fill_(mask,-torch.inf)
  threshold=scores.topk(2048,dim=-1).values[:,-1]
  selected=scores.gather(1,full.ids)
  delta=(threshold-selected.min(1).values).clamp_min(0)
  assert (delta<=scores.abs().masked_fill(mask,0).amax(1)*1e-4+1e-5).all(),delta
  assert (full.ids.sort().values[:,1:]!=full.ids.sort().values[:,:-1]).all()
  ry=reference(a,x,pos,ctx,kp,table,full.ids);rz=reference(b,x,pos,ctx,sp,table,full.ids)
  errors=[]
  for actual,expected in [(y,ry),(z,rz)]:
   err=(actual.float()-expected.float()).norm()/expected.float().norm().clamp_min(1e-10)
   assert torch.isfinite(actual).all() and err<.003,(phase,t,float(err))
   errors.append(float(err))
  note(test='layer_attention_reference',phase=phase,t=t,selection_margin_error=float(delta.max()),rel_l2=max(errors),full_shared_rel_l2=errors,passed=True)
 for phase in ['decode','prefill']:
  for _ in range(3):y,z=run(phase)
  check(phase,y,z)
 # Capture layer calls, then change device inputs and metadata.
 stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
 with torch.cuda.stream(stream):
  for _ in range(3):run('decode')
 torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize()
 g=torch.cuda.CUDAGraph()
 with torch.cuda.graph(g):gy,gz=run('decode')
 x.mul_(.75);pos.add_(2);ctx.add_(2);gy.fill_(float('nan'));gz.fill_(float('nan'));g.replay()
 check('graph_changed_inputs',gy,gz)
 note(test='layer_dynamic_graph',t=t,passed=True)
 g.reset()
note(ALL_PASS=True);dist.barrier();dist.destroy_process_group()
