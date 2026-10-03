import torch,sys,pathlib
sys.path.insert(0,'.')
from torch.utils.cpp_extension import load
from ops.decode.fp4_moe_decode import extension
base=extension()
V={n:load(name='v41_moe_'+n,sources=['tests/moe_%s.cu'%n],extra_cuda_cflags=['-O3','--use_fast_math'],verbose=False) for n in sys.argv[1:]}
dev='cuda:0'; torch.manual_seed(0)
E,Nff,K,Nout,TOPK=128,288,5120,5120,6
w13w=torch.randint(0,255,(E,2*Nff,K//2),dtype=torch.uint8,device=dev)
w13s=torch.randint(100,140,(E,2*Nff,K//32),dtype=torch.uint8,device=dev)
w2w=torch.randint(0,255,(E,Nout,Nff//2),dtype=torch.uint8,device=dev)
w2s=torch.randint(100,140,(E,Nout,Nff//32),dtype=torch.uint8,device=dev)
def tm(f,rep=30):
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
for T in (6,24):
    x=torch.randn(T,K,dtype=torch.bfloat16,device=dev)
    ids=torch.randint(0,E,(T,TOPK),dtype=torch.long,device=dev)
    wts=torch.rand(T,TOPK,dtype=torch.float32,device=dev)
    by=T*TOPK*(2*Nff*K//2+2*Nff*K//32+Nout*Nff//2+Nout*Nff//32)
    a=lambda: base.moe_rank_decode_fp4(x,ids,wts,w13w,w13s,w2w,w2s)
    ya=a(); ta=tm(a)
    print('T=%2d base %7.1fus (%4.0f GB/s)'%(T,ta,by/(ta*1e-6)/1e9))
    for n,m in V.items():
        f=lambda m=m: m.moe_rank_decode_fp4(x,ids,wts,w13w,w13s,w2w,w2s)
        yb=f(); torch.cuda.synchronize(); tb=tm(f)
        print('     %-4s %7.1fus (%4.0f GB/s)  %+.1f%%  bitexact=%s'%(n,tb,by/(tb*1e-6)/1e9,(ta/tb-1)*100,bool(torch.equal(ya,yb))))
