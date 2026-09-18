
import torch, sys, os
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # repo root on sys.path (this file lives in bench/)
import ops
mod = ops._mod(); torch.manual_seed(0); dev="cuda:0"
def bench(fn, it=200, wu=30):
    for _ in range(wu): fn()
    torch.cuda.synchronize()
    a,b = torch.cuda.Event(True), torch.cuda.Event(True); a.record()
    for _ in range(it): fn()
    b.record(); torch.cuda.synchronize(); return a.elapsed_time(b)/it*1000
print("LJQ_SMALLM8 =", os.environ.get("LJQ_SMALLM8", "(unset)"))
for (N,K,tag) in [(2048,7168,"sw1/sw3"),(7168,2048,"sw2")]:
    w=(torch.randn(N,K,device=dev,dtype=torch.bfloat16)*0.02).contiguous()
    x=(torch.randn(8,K,device=dev,dtype=torch.bfloat16)*0.05).contiguous()
    ref=x.float()@w.float().t()
    y=mod.bf16_gemm(x,w)
    rel=((y.float()-ref).abs().max()/ref.abs().max()).item()*100
    print("  %-8s M=8 -> %7.1fus  rel=%.4f%%" % (tag, bench(lambda: mod.bf16_gemm(x,w)), rel))
