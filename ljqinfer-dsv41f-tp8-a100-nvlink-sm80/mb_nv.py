import torch,sys
sys.path.insert(0,'.')
from ops.decode.fp4_moe_decode import extension
ext=extension(); dev='cuda:0'; torch.manual_seed(0)
E,K,Nout,TOPK=128,5120,5120,6
def tm(f,it=50):
    for _ in range(10): f()
    torch.cuda.synchronize(); import time; t=time.perf_counter()
    for _ in range(it): f()
    torch.cuda.synchronize(); return (time.perf_counter()-t)/it*1e6
for T in (6,24):
  for Nff in (256,288,320,352):
    x=torch.randn(T,K,device=dev,dtype=torch.bfloat16)
    ids=torch.randint(0,E,(T,TOPK),device=dev,dtype=torch.long)
    wts=torch.rand(T,TOPK,device=dev,dtype=torch.float32)
    w13=torch.randint(0,255,(E,2*Nff,K//2),device=dev,dtype=torch.uint8)
    s13=torch.randint(120,134,(E,2*Nff,K//32),device=dev,dtype=torch.uint8)
    w2=torch.randint(0,255,(E,Nout,Nff//2),device=dev,dtype=torch.uint8)
    s2=torch.randint(120,134,(E,Nout,Nff//32),device=dev,dtype=torch.uint8)
    us=tm(lambda: ext.moe_rank_decode_fp4(x,ids,wts,w13,s13,w2,s2))
    print('T=%2d Nff=%3d NV=%d  %8.1fus  us/Nff=%.4f'%(T,Nff,Nff//32,us,us/Nff))
    del w13,s13,w2,s2; torch.cuda.empty_cache()
