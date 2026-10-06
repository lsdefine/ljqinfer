"""Independent NPU MoE finish regression; no engine or model weights.

Run: PYTHONPATH=. python scripts/test_prefill_moe_finish.py
Covers empty/small/8192 rows, strided inputs, side stream, fallback and rejection.
Reference preserves shared BF16 rounding before FP32 addition.
"""
import json
import torch,torch_npu
from ops.prefill.residual import moe_add
from ops.prefill.native import ops as native

def main():
 torch.npu.set_device(0)
 torch.manual_seed(711)
 report=dict(complete=False,cases=[],rejected=0)
 def check(n,strided=False):
  width=10240 if strided else 5120
  a=torch.randn((n,width),device='npu',dtype=torch.float32)
  b=torch.randn_like(a)
  if strided:a,b=a[:,::2],b[:,::2]
  ref=(a.float()+b.to(torch.bfloat16).float()).to(torch.bfloat16)
  got=moe_add(a,b)
  torch.npu.synchronize()
  exact=torch.equal(got,ref)
  report['cases'].append(dict(rows=n,strided=strided,exact=exact))
  assert exact,(n,strided)
  assert got.dtype==torch.bfloat16 and got.shape==a.shape
 for n in (0,1,127,1024,8192):check(n)
 for n in (1,127,8192):check(n,True)
 with torch.npu.stream(torch.npu.Stream()):
  check(127,True)
  check(8192)
 for dtype in (torch.bfloat16,torch.float32):
  a=torch.randn((7,64),device='npu').to(dtype);b=torch.randn_like(a)
  expected=(a.float()+b.to(dtype).float()).to(dtype)
  assert torch.equal(moe_add(a,b,dtype=dtype),expected)
  report['cases'].append(dict(fallback=str(dtype),exact=True))
 a=torch.zeros((1,5120),device='npu')
 for b in (a.to(torch.bfloat16),torch.zeros((2,5120),device='npu')):
  try:native.moe_finish(a,b)
  except RuntimeError:report['rejected']+=1
  else:raise AssertionError('invalid input accepted')
 torch.npu.synchronize()
 report['complete']=True
 print(json.dumps(report),flush=True)
if __name__ == '__main__':
 main()
