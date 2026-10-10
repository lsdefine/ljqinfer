"""Independent B1..4 HC epilogue validation and graph A/B; no engine imports."""
import ctypes as C
import json, statistics, time
from pathlib import Path
import torch, torch_npu
P=C.c_void_p; U=C.c_uint32
repo=Path(__file__).resolve().parents[1]
root=repo/'ops/decode'
old=C.CDLL(str(root/'libdecode_norm.so'))
new=C.CDLL(str(root/'libdecode_hc_finish.so'))
for lib,name,n in [(old,'dec_tp_cast',3),(old,'dec_moe_finish',4),(old,'dec_hc_expand',6),(new,'dec_hc_finish',7)]:
 f=getattr(lib,name);f.argtypes=[P]*n+[U];f.restype=C.c_int

def call(lib,name,*args):
 s=torch.npu.current_stream().npu_stream
 rc=getattr(lib,name)(P(s),*[P(x.data_ptr()) if x is not None else P() for x in args[:-1]],U(args[-1]))
 assert rc==0,(name,rc)

def capture(fn,repeat=128):
 for _ in range(3):fn()
 torch.npu.synchronize()
 g=torch.npu.NPUGraph()
 with torch.npu.graph(g):
  for _ in range(repeat):fn()
 return g

def measure(g,repeat=128):
 a=torch.npu.Event(enable_timing=True);b=torch.npu.Event(enable_timing=True)
 a.record()
 for _ in range(10):g.replay()
 b.record();b.synchronize()
 return a.elapsed_time(b)*1000/(repeat*10)

torch.npu.set_device(0);torch.manual_seed(123)
results=[]
for batch in range(1,5):
 rows=batch*6; shape=(rows,5120)
 x=torch.randn(shape,device='npu',dtype=torch.float32)
 shared=torch.randn_like(x);res=torch.randn((rows,4,5120),device='npu',dtype=torch.bfloat16)
 post=torch.randn((rows,4),device='npu');comb=torch.randn((rows,4,4),device='npu')
 mid=torch.empty_like(x,dtype=torch.bfloat16);ref=torch.empty_like(res);out=torch.empty_like(res)
 for moe in [False,True]:
  def baseline():
   if moe:call(old,'dec_moe_finish',x,shared,mid,batch)
   else:call(old,'dec_tp_cast',x,mid,batch)
   call(old,'dec_hc_expand',mid,res,post,comb,ref,batch)
  def candidate():call(new,'dec_hc_finish',x,shared if moe else None,res,post,comb,out,batch)
  for scale in [0.001,1.,100.]:
   x.normal_().mul_(scale);shared.normal_().mul_(scale)
   baseline();candidate();torch.npu.synchronize()
   assert torch.equal(ref.cpu(),out.cpu()),(batch,moe,scale,(ref.float()-out.float()).abs().max().item())
   alias=res.clone();call(new,'dec_hc_finish',x,shared if moe else None,alias,post,comb,alias,batch)
   torch.npu.synchronize();assert torch.equal(ref.cpu(),alias.cpu()),('alias',batch,moe)
  ga,gb=capture(baseline),capture(candidate)
  for _ in range(3):ga.replay();gb.replay()
  torch.npu.synchronize()
  times=[[],[]]
  for j in range(8):
   for k in ([0,1] if j%2==0 else [1,0]):times[k].append(measure([ga,gb][k]))
  a,b=map(statistics.median,times)
  row=dict(batch=batch,moe=moe,bit_exact=True,baseline_us=a,candidate_us=b,saving_us=a-b,samples=times)
  results.append(row);print(json.dumps(row),flush=True)
  del ga,gb
print('RESULT_JSON',json.dumps(results))
print('PASS',len(results),flush=True)
