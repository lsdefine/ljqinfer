"""HSI semantic checks against independent mathematical CPU references."""
import ctypes as C
import json
from pathlib import Path
import torch
import torch_npu

torch.set_num_threads(4)
torch.npu.set_device(0)
lib=C.CDLL(str(Path(__file__).resolve().parents[2]/'ops/decode/libdecode_attention.so'))

def launch(name, tensors, scalars):
    fn=getattr(lib,name)
    fn.argtypes=[C.c_void_p]*(1+len(tensors))+[C.c_int]*len(scalars)
    fn.restype=C.c_int
    rc=fn(C.c_void_p(torch.npu.current_stream().npu_stream),
          *[C.c_void_p(x.data_ptr()) for x in tensors], *scalars)
    assert rc==0,(name,rc)

def pool_ref(scores, starts, active, slots):
    b,_,width=scores.shape
    out=torch.full((b,6,16384),-1,dtype=torch.int64)
    for i in range(b):
        if not active[i] or not 0<=slots[i]<b or starts[i]<0:continue
        for t in range(6):
            limit=min(int(starts[i])+t+1,width)
            newest=(limit-1)//8
            mx=scores[i,t,:newest*8].reshape(-1,8).amax(1)
            order=torch.argsort(mx,descending=True,stable=True)
            order=order[torch.isfinite(mx[order])][:2047]
            pos=(order[:,None]*8+torch.arange(8)).reshape(-1)
            out[i,t,:len(pos)]=pos
            pos=newest*8+torch.arange(8)
            out[i,t,-8:]=torch.where(pos<limit,pos,-1)
    return out

def select_ref(scores,pool):
    out=torch.full((*scores.shape[:2],512),-1,dtype=torch.int64)
    for b in range(scores.shape[0]):
        for t in range(6):
            valid=(pool[b,t]>=0)&torch.isfinite(scores[b,t])
            ids=pool[b,t,valid]; vals=scores[b,t,valid]
            byid=torch.argsort(ids,stable=True)
            order=byid[torch.argsort(vals[byid],descending=True,stable=True)][:512]
            out[b,t,:len(order)]=ids[order]
    return out

records=[]
gen=torch.Generator().manual_seed(451)
for batch,width in [(1,64),(4,32768),(1,131072)]:
    scores=torch.randint(-16,17,(batch,6,width),generator=gen).float()
    dev=scores.npu();starts=torch.zeros(batch,dtype=torch.int64).npu()
    active=torch.ones(batch,dtype=torch.int64).npu();slots=torch.arange(batch,dtype=torch.int64).npu()
    pool=torch.full((batch,6,16384),999,dtype=torch.int64,device='npu')
    selected=torch.full((batch,6,512),999,dtype=torch.int64,device='npu')
    rank_scores=torch.empty((batch,6,16384),device='npu')
    def run():
        launch('dec_hsi_pool',[dev,slots,starts,active,pool],[batch,width,batch])
        launch('dec_hsi_select',[rank_scores,pool,selected],[batch])
    rank_cpu=torch.randint(-5,6,rank_scores.shape,generator=gen).float()
    rank_scores.copy_(rank_cpu)
    run();torch.npu.synchronize()
    graph=torch.npu.NPUGraph()
    with torch.npu.graph(graph):run()
    for step,origin in enumerate([0,7,min(width-6,16382),width-6]):
        sc=torch.tensor([max(0,origin-i*3) for i in range(batch)])
        ac=torch.ones(batch,dtype=torch.int64)
        if batch==4 and step==3:ac[-1]=0
        starts.copy_(sc);active.copy_(ac)
        graph.replay();torch.npu.synchronize()
        ref=pool_ref(scores,sc.tolist(),ac.tolist(),list(range(batch)))
        got=pool.cpu()
        assert torch.equal(got,ref),('pool',batch,width,step,(got!=ref).nonzero()[:8])
        refsel=select_ref(rank_cpu,ref)
        assert torch.equal(selected.cpu(),refsel),('select',batch,width,step)
        records.append(dict(batch=batch,width=width,origin=origin,pool=True,select=True,replay=True))
print(json.dumps(dict(pool_select=records)),flush=True)
