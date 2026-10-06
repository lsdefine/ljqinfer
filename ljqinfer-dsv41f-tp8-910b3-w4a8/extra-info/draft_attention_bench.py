import ctypes as C,json,time,statistics
from pathlib import Path
import torch,torch_npu
P=C.c_void_p;U=C.c_uint32
root=Path('/tmp/draft_attention_opt_v1');torch.npu.set_device(0)
libs=[C.CDLL(str(root/('libnorm_'+x+'.so'))) for x in ('base','candidate')]
fns=[]
for lib in libs:
 f=lib.dec_ds_attention;f.argtypes=[P]*7+[U]*4;f.restype=C.c_int;fns.append(f)
def call(f,ts,b,slots,ring,pad):
 assert f(P(torch.npu.current_stream().npu_stream),*[P(t.data_ptr()) for t in ts],b,slots,ring,pad)==0

def oracle(q,kv,h,r,sink,ring,pad):
 out=torch.zeros_like(q)
 for b,row in enumerate(r.tolist()):
  slot,pos,acc,_,active,err,_,_=row
  valid=active and not err and 0<=slot<len(h) and 0<=pos<=2**63-17 and 1<=acc<=6
  valid=valid and not any(j!=b and rr[4] and rr[0]==slot for j,rr in enumerate(r.tolist()))
  if not valid:continue
  end=pos+acc+1
  keys=torch.cat([h[slot,torch.arange(max(0,end-128),end)%ring+pad],kv[b,:5]],0).float()
  query=q[b,:5].float();score=torch.einsum('qhd,kd->qhk',query,keys)*(512**-.5)
  weights=torch.softmax(torch.cat([score,sink[None,:,None].expand(5,-1,1)],-1),-1)[...,:len(keys)]
  out[b,:5]=torch.einsum('qhk,kd->qhd',weights,keys).bfloat16()
 return out

def capture(f,ts,b,ring,pad,reps=1):
 stream=torch.npu.Stream()
 with torch.npu.stream(stream):
  for _ in range(3):call(f,ts,b,4,ring,pad)
 stream.synchronize();g=torch.npu.NPUGraph()
 with torch.npu.graph(g,stream=stream):
  for _ in range(reps):call(f,ts,b,4,ring,pad)
 stream.synchronize()
 return g,stream

def timed(g,stream,reps):
 values=[]
 with torch.npu.stream(stream):
  for _ in range(3):g.replay()
  stream.synchronize()
  for _ in range(7):
   st=torch.npu.Event(enable_timing=True);en=torch.npu.Event(enable_timing=True)
   st.record();g.replay();en.record();en.synchronize();values.append(st.elapsed_time(en)*1000/reps)
 return statistics.median(values)

results=[];perf=[]
with torch.inference_mode():
 for b in (1,2,3,4):
  for case,(ring,pad,pos,mode,scale) in enumerate([(134,0,0,'live',1),(256,128,126,'live',1),(256,128,255,'live',1),(134,7,20000,'live',3),(256,128,999,'inactive',1),(256,128,34,'invalid',1),(256,128,34,'duplicate',1)]):
   torch.manual_seed(1100+b*11+case)
   q=(torch.randn(b,6,8,512)*scale).bfloat16();kv=torch.randn(b,6,512).bfloat16();h=torch.randn(4,ring+pad,512).bfloat16();sink=torch.linspace(-2,2,8)
   r=torch.zeros(b,8,dtype=torch.int64)
   for i in range(b):r[i]=torch.tensor([3-i,pos+i,i%6+1,9,1,0,0,0])
   if mode=='inactive':r[0,4]=0
   if mode=='invalid':r[0,2]=0;r[-1,5]=1
   if mode=='duplicate' and b>1:r[1,0]=r[0,0]
   cpu=[q,kv,h,r,sink];device=[t.npu() for t in cpu];outs=[torch.empty_like(device[0]) for _ in fns]
   ref=oracle(*cpu,ring,pad);graphs=[]
   for fn,out in zip(fns,outs):graphs.append(capture(fn,device+[out],b,ring,pad))
   for g,s in graphs:
    with torch.npu.stream(s):g.replay()
    s.synchronize()
   a,c=[o.cpu() for o in outs]
   err=(c.float()-a.float());rel=float(err.norm()/(a.float().norm()+1e-20))
   row=dict(batch=b,case=case,mode=mode,max_abs=float(err.abs().max()),rel_l2=rel,base_cpu_max=float((a.float()-ref.float()).abs().max()),candidate_cpu_max=float((c.float()-ref.float()).abs().max()))
   assert torch.isfinite(c).all() and rel<.003 and row['max_abs']<=.032,row
   assert row['candidate_cpu_max']<=.016,row
   assert float((c.float()-ref.float()).norm()/(ref.float().norm()+1e-20))<.003,row
   assert torch.equal(c[:,5],torch.zeros_like(c[:,5]))
   saved=c.clone()
   for _ in range(16):
    g,s=graphs[1]
    with torch.npu.stream(s):g.replay()
    s.synchronize();assert torch.equal(outs[1].cpu(),saved),'non-deterministic'
   # Change device receipt after capture; invalid->live and shifted ring positions.
   rr=r.clone();rr[:,1]+=137;rr[:,4]=1;rr[:,5]=0;rr[:,2]=6
   device[3].copy_(rr);torch.npu.synchronize()
   for g,s in graphs:
    with torch.npu.stream(s):g.replay()
    s.synchronize()
   aa,cc=[o.cpu() for o in outs];delta=cc.float()-aa.float()
   row['mutated_rel_l2']=float(delta.norm()/(aa.float().norm()+1e-20));assert row['mutated_rel_l2']<.003,row
   results.append(row)
   for g,s in graphs:g.reset()
   if case==2:
    gs=[capture(fn,device+[out],b,ring,pad,128) for fn,out in zip(fns,outs)]
    paired=[]
    for order in ((0,1),(1,0),(0,1)):
     pair={}
     for i in order:pair[str(i)]=timed(*gs[i],128)
     paired.append(pair)
    perf.append(dict(batch=b,samples_us=paired))
    for g,s in gs:g.reset()
  print('BATCH_DONE',b,flush=True)
result=dict(complete=True,cases=results,performance=perf)
(root/'micro_results.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
