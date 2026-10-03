
import torch,sys
sys.path.insert(0,'.')
from ops.decode.fp4_moe_decode import extension
base=extension()
dev='cuda:0'; torch.manual_seed(0)
E,Nff,K,Nout,TOPK=384,288,5120,5120,6
w13w=torch.randint(0,255,(E,2*Nff,K//2),dtype=torch.uint8,device=dev)
w13s=torch.randint(100,140,(E,2*Nff,K//32),dtype=torch.uint8,device=dev)
w2w=torch.randint(0,255,(E,Nout,Nff//2),dtype=torch.uint8,device=dev)
w2s=torch.randint(100,140,(E,Nout,Nff//32),dtype=torch.uint8,device=dev)
per=(2*Nff*K//2+2*Nff*K//32+Nout*Nff//2+Nout*Nff//32)
def tm(f,rep=20):
    for _ in range(5): f()
    torch.cuda.synchronize()
    g=torch.cuda.CUDAGraph(); s=torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3): f()
    torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
    with torch.cuda.graph(g):
        for _ in range(rep): f()
    torch.cuda.synchronize(); e0,e1=torch.cuda.Event(True),torch.cuda.Event(True); ts=[]
    for _ in range(5):
        e0.record(); g.replay(); e1.record(); torch.cuda.synchronize(); ts.append(e0.elapsed_time(e1)*1000/rep)
    ts.sort(); return ts[2]
print('E=%d TOPK=%d  (weights %.0f MB)'%(E,TOPK,E*per/1e6))
for T,Us in ((6,(36,18,6,1)),(24,(144,120,96,72,36,6,1))):
    x=torch.randn(T,K,dtype=torch.bfloat16,device=dev)
    wts=torch.rand(T,TOPK,dtype=torch.float32,device=dev)
    print('--- T=%d  sel=%d'%(T,T*TOPK))
    for U in Us:
        pool=torch.randperm(E)[:U]
        ids=pool[torch.randint(0,U,(T*TOPK,))].view(T,TOPK).to(dev).long()
        f=lambda: base.moe_rank_decode_fp4(x,ids,wts,w13w,w13s,w2w,w2s)
        t=tm(f)
        print('  U=%3d uniq  %7.1fus   naive_bytes=%5.0fMB -> %4.0f GB/s | uniq_bytes=%5.0fMB -> %4.0f GB/s'%(
            U,t,T*TOPK*per/1e6,T*TOPK*per/(t*1e-6)/1e9,U*per/1e6,U*per/(t*1e-6)/1e9))
