"""Independent W13 half-input SwiGLU regression. Run: PYTHONPATH=. python scripts/test_prefill_swiglu_half.py"""
import json
import torch,torch_npu
from ops.prefill.native import ops
candidate=ops.routed_swiglu

def main():
 torch.npu.set_device(0);torch.manual_seed(719)
 report={'complete':False,'cases':[],'samples':[]}
 for rows in [0,1,7,16,17,127,762,24576,49152]:
  h=torch.randn(rows,576,device='npu',dtype=torch.float16)*5
  prob=torch.rand(rows,device='npu')
  ref=ops.routed_swiglu(h.to(torch.bfloat16),prob);got=candidate(h,prob)
  assert torch.equal(ref,got),(rows,'activation',(ref-got).abs().max().item())
  if rows:
   qa,sa=ops.dynamic_quant(ref);qb,sb=ops.dynamic_quant(got)
   assert torch.equal(qa,qb) and torch.equal(sa,sb),(rows,'quant')
  report['cases'].append(dict(rows=rows,exact=True))
 bits=torch.arange(65536,dtype=torch.int32).to(torch.int16).view(torch.float16)
 bits=bits[torch.isfinite(bits)];h=bits.repeat(9)[:576*900].reshape(900,576).npu();prob=torch.rand(900,device='npu')
 assert torch.equal(candidate(h,prob),ops.routed_swiglu(h.to(torch.bfloat16),prob))
 report['cases'].append(dict(exhaustive=True,exact=True))

 for dtype in [torch.float16,torch.bfloat16]:
  for strided in [False,True]:
   stream=torch.npu.Stream()
   with torch.npu.stream(stream):
    h=torch.randn(127,1152 if strided else 576,device='npu',dtype=dtype)
    if strided:h=h[:,::2]
    prob=torch.rand(254,device='npu')[::2]
    got=candidate(h,prob);ref=candidate(h.to(torch.bfloat16).contiguous(),prob.contiguous())
   stream.synchronize()
   assert torch.equal(got,ref),(dtype,strided,'side stream')
   report['cases'].append(dict(dtype=str(dtype),strided=strided,side_stream=True,exact=True))
 h=torch.randn(7,576,device='npu',dtype=torch.float16);prob=torch.rand(7,device='npu')
 for x,p in [(h.float(),prob),(h[:,:575],prob),(h,prob[:6]),(h,prob.half())]:
  try:candidate(x,p)
  except RuntimeError:pass
  else:raise AssertionError('invalid input accepted')
 report['invalid_rejected']=4
 report['complete']=True
 print(json.dumps(report),flush=True)
if __name__=='__main__':main()
