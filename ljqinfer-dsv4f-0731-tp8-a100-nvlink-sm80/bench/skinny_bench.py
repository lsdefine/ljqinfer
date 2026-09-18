import sys,os,torch,time
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # repo root on sys.path (this file lives in bench/)
from ops import _mod
m=_mod(); torch.manual_seed(0)
mode=sys.argv[1]; P='/tmp/skinny_ref.pt'
ref=torch.load(P) if mode=='cmp' else {}
for (K,N) in [(7168,1024),(2048,512),(7168,512)]:
  x=torch.randn(128,K,device='cuda'); w=torch.randn(N,K,device='cuda')
  for M in [1,3,8,16,32,64,128]:
    xx=x[:M].contiguous()
    y=m.sgemm_skinny2_f32(xx,w); torch.cuda.synchronize()
    for _ in range(5): m.sgemm_skinny2_f32(xx,w)
    torch.cuda.synchronize(); t0=time.perf_counter()
    for _ in range(50): m.sgemm_skinny2_f32(xx,w)
    torch.cuda.synchronize(); us=(time.perf_counter()-t0)/50*1e6
    k=f'{K}x{N}x{M}'
    if mode=='cmp': print(f'{k:16s} {us:7.1f}us  BITEXACT={torch.equal(y.cpu(),ref[k])}')
    else: ref[k]=y.cpu(); print(f'{k:16s} {us:7.1f}us')
if mode!='cmp': torch.save(ref,P)
