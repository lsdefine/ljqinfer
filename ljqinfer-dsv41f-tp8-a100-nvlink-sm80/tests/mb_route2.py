import torch, triton, sys
sys.path.insert(0,'.')
from ops.decode import route_gate as RG
torch.manual_seed(0); dev='cuda:0'
K,N,TOPK,REP=5120,384,6,50
w=torch.randn(N,K,device=dev,dtype=torch.bfloat16); b=torch.randn(N,device=dev,dtype=torch.float32)*0.01
def gbench(fn):
    s=torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(5): fn()
    torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
    g=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(REP): fn()
    torch.cuda.synchronize()
    for _ in range(3): g.replay()
    torch.cuda.synchronize()
    e0,e1=torch.cuda.Event(True),torch.cuda.Event(True); e0.record()
    for _ in range(10): g.replay()
    e1.record(); torch.cuda.synchronize()
    return e0.elapsed_time(e1)*1000/(10*REP)
for T in (6,24):
    x=torch.randn(T,K,device=dev,dtype=torch.bfloat16)
    tp=max(2,triton.next_power_of_2(min(T,16)))
    print('== T=%d'%T)
    for ks2,bn2,bk2,nw2 in [(10,32,64,4),(10,64,64,4),(10,128,64,4),(5,32,64,4),(20,32,64,4),(10,32,128,4),(10,64,128,8),(2,64,64,4),(1,64,64,4),(4,128,64,4)]:
        if K%(ks2*bk2) or T>16: 
            if T>16: pass
        if K%(ks2*bk2): continue
        z2=torch.empty((ks2,T,N),device=dev,dtype=torch.float32)
        xx=x[:16] if T>16 else x; TT=min(T,16)
        f=lambda: RG._route_gemv[(triton.cdiv(N,bn2),ks2)](xx,RG._const(w,x.dtype),z2,1.0,TT,K,N,tp,bn2,bk2,K//ks2,1,num_warps=nw2)
        print('   gemv ks=%-2d bn=%-3d bk=%-3d nw=%d -> %6.2f us'%(ks2,bn2,bk2,nw2,gbench(f)))
    ks=10; z=torch.empty((ks,min(T,16),N),device=dev,dtype=torch.float32)
    TT=min(T,16)
    prob=torch.empty((TT,TOPK),device=dev,dtype=torch.float32); ids=torch.empty((TT,TOPK),device=dev,dtype=torch.int64)
    for tw in (1,2,4,8):
        f=lambda tw=tw: RG._route_topk[(TT,)](z,RG._const(b,torch.float32),prob,ids,2.5,1.0,TT,N,triton.next_power_of_2(N),ks,1,TOPK,True,False,num_warps=tw)
        print('   topk nw=%d -> %6.2f us'%(tw,gbench(f)))
