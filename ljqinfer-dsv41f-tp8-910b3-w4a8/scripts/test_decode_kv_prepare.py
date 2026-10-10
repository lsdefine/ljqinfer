"""Standalone KV preparation A/B against the original three leaf kernels."""
import ctypes as C
import json,statistics
from pathlib import Path
import torch,torch_npu
P=C.c_void_p;U=C.c_uint32;F=C.c_float
repo=Path(__file__).resolve().parents[1];root=repo/'ops/decode'
old=C.CDLL(str(root/'libdecode_norm.so'));att=C.CDLL(str(repo/'ops/decode/libdecode_attention.so'));new=C.CDLL(str(root/'libdecode_kv_prepare.so'))
old.dec_rms_norm_f32.argtypes=[P]*4+[U,U,F];old.dec_rope.argtypes=[P]*4+[U]*4
att.dec_kv_qdq.argtypes=[P]*3+[U];new.dec_kv_prepare.argtypes=[P]*5+[U,F]
for lib,name in [(old,'dec_rms_norm_f32'),(old,'dec_rope'),(att,'dec_kv_qdq'),(new,'dec_kv_prepare')]:getattr(lib,name).restype=C.c_int

def invoke(fn,*args):
 ret=fn(P(torch.npu.current_stream().npu_stream),*[P(x.data_ptr()) if isinstance(x,torch.Tensor) else x for x in args]);assert ret==0,ret

def capture(fn):
 for _ in range(3):fn()
 torch.npu.synchronize();g=torch.npu.NPUGraph()
 with torch.npu.graph(g):
  for _ in range(64):fn()
 return g

def measure(g):
 a=torch.npu.Event(enable_timing=True);b=torch.npu.Event(enable_timing=True)
 a.record()
 for _ in range(12):g.replay()
 b.record();b.synchronize();return a.elapsed_time(b)*1000/768

torch.npu.set_device(0);torch.npu.config.allow_internal_format=False
results=[]
for batch in range(1,5):
 rows=batch*6
 x=torch.empty((rows,512),device='npu',dtype=torch.bfloat16);w=torch.empty(512,device='npu');freq=torch.empty((rows,64),device='npu')
 n=torch.empty_like(x);r=torch.empty_like(x);y=torch.empty_like(x);z=torch.empty_like(x)
 def baseline():
  invoke(old.dec_rms_norm_f32,x,w,n,batch,512,F(1e-6));invoke(old.dec_rope,n,freq,r,batch,1,512,0);invoke(att.dec_kv_qdq,r,y,batch)
 def candidate():invoke(new.dec_kv_prepare,x,w,freq,z,batch,F(1e-6))
 for seed in range(5):
  torch.manual_seed(7100+seed)
  w.copy_(torch.randn_like(w)*0.2+1);a=torch.randn((rows,32),device='npu')*100;freq.copy_(torch.stack([a.cos(),a.sin()],dim=-1).reshape(rows,64))
  for scale in [0.,1e-5,0.2,3.,1000.]:
   x.copy_(torch.randn_like(x)*scale);baseline();candidate();torch.npu.synchronize()
   assert torch.equal(y.cpu().view(torch.int16),z.cpu().view(torch.int16)),(batch,seed,scale,(y.float()-z.float()).abs().max().item())
  backup=x.clone();invoke(new.dec_kv_prepare,x,w,freq,x,batch,F(1e-6));torch.npu.synchronize();assert torch.equal(y.cpu().view(torch.int16),x.cpu().view(torch.int16)),('inplace',batch);x.copy_(backup)
 ga=capture(baseline);gb=capture(candidate);times=[[],[]]
 for repeat in range(8):
  for k in ([0,1] if repeat%2==0 else [1,0]):times[k].append(measure([ga,gb][k]))
 a,b=map(statistics.median,times);row=dict(batch=batch,bit_exact=True,cases=25,baseline_us=a,candidate_us=b,saving_us=a-b,samples=times)
 results.append(row);print(json.dumps(row),flush=True);del ga,gb
print('RESULT_JSON',json.dumps(results));print('PASS',flush=True)
