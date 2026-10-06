"""Independent FP16 W2 combine regression; no engine or weights.
Run: PYTHONPATH=. python scripts/test_prefill_combine_half.py
Checks finite FP16 patterns, CPU fixed-order oracle, strided and side stream.
"""
import json
import torch,torch_npu
from ops.prefill.native import ops
candidate=ops.routed_combine_half

def main():
 torch.npu.set_device(0);torch.manual_seed(714)
 report=dict(complete=False,cases=[])
 def case(n,exhaustive=False,strided=False):
  x=torch.randn((n*6,10240 if strided else 5120),dtype=torch.float16)
  if exhaustive:
   bits=torch.arange(65536,dtype=torch.int32).to(torch.int16).view(torch.float16)
   bits=bits[torch.isfinite(bits)]
   x.flatten().copy_(bits.repeat((x.numel()+len(bits)-1)//len(bits))[:x.numel()])
  inv=torch.randperm(n*6,dtype=torch.int64)
  a=x.npu();i=inv.npu()
  if strided:a,x=a[:,::2],x[:,::2]
  ref=ops.routed_combine(a.to(torch.bfloat16),i);got=candidate(a,i)
  assert torch.equal(ref,got),(n,(ref-got).abs().max().item())
  if n and n<=127:
   rows=inv.reshape(n,6).sort(dim=1).values
   v=x.to(torch.bfloat16).float()[rows]
   oracle=v[:,0]+0.0
   for j in range(1,6):oracle=oracle+v[:,j]
   assert torch.equal(got.cpu(),oracle),('cpu oracle',n)
  report['cases'].append(dict(rows=n,exhaustive=exhaustive,strided=strided,exact=True))
 for n in [0,1,7,20,21,127,1024,8192]:case(n)
 case(127,True)
 for n in [1,127,8192]:case(n,strided=True)
 with torch.npu.stream(torch.npu.Stream()):
  case(127,strided=True)
  case(8192)
 a=torch.zeros((6,5120),device='npu',dtype=torch.float16);i=torch.arange(6,device='npu')
 rejected=0
 for x,j in [(a.float(),i),(a,i.int()),(a[:,:1],i),(a,i[:-1])]:
  try:candidate(x,j)
  except RuntimeError:rejected+=1
  else:raise AssertionError('invalid accepted')
 torch.npu.synchronize()
 report.update(complete=True,rejected=rejected)
 print(json.dumps(report),flush=True)

if __name__=='__main__':main()
