import sys,json,math
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from ops.sparse_ops import *
from ops.sparse_mla_cuda import forward as sparse_mla
torch.manual_seed(219);torch.cuda.set_device(0)
torch.backends.cuda.matmul.allow_tf32=False

def err(a,b):
    d=a.float()-b.float()
    return dict(rel=float(d.norm()/b.float().norm().clamp_min(1e-20)),max=float(d.abs().max()))

def run(t,n,top,dtype):
    page=64;np=math.ceil(n/page)
    table=torch.randperm(np,device='cuda',dtype=torch.int64)
    pool=torch.randn(np,page,576,device='cuda',dtype=dtype)*.3
    kp=torch.randn(np,page,128,device='cuda',dtype=dtype)
    ctx=torch.tensor([n],device='cuda',dtype=torch.int64)
    pos=torch.arange(n-t,n,device='cuda',dtype=torch.int64)
    qi=torch.randn(t,32,128,device='cuda',dtype=dtype)
    w=torch.randn(t,32,device='cuda',dtype=torch.float32)/math.sqrt(32)
    sc=torch.empty(t,n,device='cuda'); vals=torch.empty(t,top,device='cuda')
    idx64=torch.empty(t,top,device='cuda',dtype=torch.int64)
    idx=torch.empty(t,top,device='cuda',dtype=torch.int32)
    index_scores(qi,kp,w,table,pos,ctx,sc)
    keys=kp[table].flatten(0,1)[:n].float()
    ref=((qi.float()@keys.T).relu()*w[:,:,None]).sum(1)/math.sqrt(128)
    ref.masked_fill_(torch.arange(n,device='cuda')[None,:]>pos[:,None],-float('inf'))
    mask=ref.isfinite();e=err(sc[mask],ref[mask]);assert e['rel']<2e-5,e
    topk_indices(sc,vals,idx64,idx)
    idx=idx.long()
    # Exact selected score multiset for the supplied scores; tied indices arbitrary.
    assert torch.equal(vals.sort(1).values,sc.topk(top,dim=1).values.sort(1).values)
    chosen=ref.gather(1,idx.long().clamp_min(0))
    bound=ref.topk(top,dim=1).values[:,-1:]
    assert bool(((chosen>=bound-0.0001)|(idx<0)).all())
    # Include holes, future positions, out-of-range positions, and an empty row.
    idx[:,::17]=-1
    idx[0,:]=-1
    if t>1:idx[1,0]=n+12
    q=torch.randn(t,8,576,device='cuda',dtype=dtype)*.3
    out=torch.empty(t,8,512,device='cuda',dtype=dtype)
    def reference():
        cache=pool[table].flatten(0,1)
        rows=[]
        for i in range(t):
            ids=idx[i].long();ids=ids[(ids>=0)&(ids<int(ctx))&(ids<=pos[i])]
            if ids.numel()==0:rows.append(torch.zeros(8,512,device='cuda'))
            else:
                k=cache[ids].float();s=q[i].float()@k.T/16
                rows.append(s.softmax(-1)@k[:,:512])
        return torch.stack(rows)
    truth=reference()
    for splits in [1]:
        ws=make_mla_workspace(q,splits=splits)
        sparse_mla(q,pool,table,idx,pos,ctx,out,ws,splits=splits)
        e2=err(out,truth);assert e2['rel']<(0.008 if dtype==torch.bfloat16 else .001),e2
        assert torch.equal(out[0],torch.zeros_like(out[0]))
        for _ in range(2):sparse_mla(q,pool,table,idx,pos,ctx,out,ws,splits=splits)
        torch.cuda.synchronize()
        g=torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):sparse_mla(q,pool,table,idx,pos,ctx,out,ws,splits=splits)
        q.mul_(1.1);ctx.sub_(3);g.replay();saved=out.clone()
        truth=reference();eg=err(saved,truth);assert eg['rel']<(0.008 if dtype==torch.bfloat16 else .001),eg
        pool.mul_(.9);table.copy_(table.roll(1));idx.copy_(idx.roll(1,1));pos.sub_(2)
        g.replay();eg2=err(out,reference());assert eg2['rel']<.001,eg2
        g.reset()
    # Scatter uses same page table; invalid write positions are ignored.
    x=torch.randn(t,128,device='cuda',dtype=dtype);original=kp.clone()
    scatter_keys(x,kp,table,pos)
    expected=original.clone()
    for i in range(t):expected[table[pos[i]//page],pos[i]%page]=x[i]
    assert torch.equal(expected,kp)
    print('PASS',t,n,top,str(dtype),e,e2,eg,flush=True)

for dt in [torch.float16]:
    for t,n,k in [(1,63,32),(8,2053,2048),(17,4097,2048),(7,65,64),(5,129,33),(3,257,257)]:run(t,n,k,dt)
print('ALL_PASS',flush=True)
