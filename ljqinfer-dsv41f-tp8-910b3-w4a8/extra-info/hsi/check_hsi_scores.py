"""Paged HSI scores vs CPU formula, not old decode output."""
import ctypes as C
import json
from pathlib import Path
import torch
import torch_npu

torch.set_num_threads(4)
torch.npu.set_device(0)
lib=C.CDLL(str(Path(__file__).resolve().parents[2]/'ops/decode/libdecode_attention.so'))
fn=lib.dec_hsi_scores
fn.argtypes=[C.c_void_p]*11+[C.c_int]*5
fn.restype=C.c_int
g=torch.Generator().manual_seed(2501)
def sample(shape):
    return (torch.randint(-8,9,shape,generator=g).float()/16).bfloat16()
records=[]
for b in [1,4]:
    rpp=32; maxpages=8; pages=16
    q=sample((b,6,4,128)); weight=sample((b,6,4))
    bank=sample((pages,rpp,128)); pending=sample((b,6,128))
    table=torch.stack([torch.randperm(pages,generator=g)[:maxpages] for _ in range(b)])
    slots=torch.arange(b-1,-1,-1,dtype=torch.int64)
    starts=torch.tensor([63+i*32 for i in range(b)],dtype=torch.int64)
    active=torch.ones(b,dtype=torch.int64)
    pool=torch.full((b,6,16384),-1,dtype=torch.int64)
    for i in range(b):
        for t in range(6):pool[i,t,:256]=torch.randperm(256,generator=g)
    cpu=[q,weight,bank,table,pending,slots,starts,active,pool]
    dev=[x.npu() for x in cpu]
    out=torch.full((b,6,16384),123.,device='npu')
    def run():
        assert fn(C.c_void_p(torch.npu.current_stream().npu_stream),
                  *[C.c_void_p(x.data_ptr()) for x in dev+[out]],b,b,pages,maxpages,rpp)==0
    run();torch.npu.synchronize()
    graph=torch.npu.NPUGraph()
    with torch.npu.graph(graph):run()
    for step in range(3):
        if step==1:
            starts.add_(1);table[:,1]=-1;pending.copy_(sample(pending.shape))
        if step==2:
            active[-1]=0
            if b>1:starts[0]=-1;slots[1]=-1
        for d,c in zip(dev,cpu):d.copy_(c)
        graph.replay();torch.npu.synchronize()
        ref=torch.full(out.shape,float('-inf'))
        for i in range(b):
            origin=int(starts[i]);slot=int(slots[i])
            if not active[i] or origin<0 or not 0<=slot<b:continue
            for t in range(6):
                ids=pool[i,t,:256];valid=(ids>=0)&(ids<origin+t+1)
                key=torch.zeros((256,128))
                for j,id in enumerate(ids.tolist()):
                    if not valid[j]:continue
                    if id>=origin:key[j]=pending[i,id-origin].float()
                    else:
                        page=int(table[slot,id//rpp])
                        if page<0:valid[j]=False
                        else:key[j]=bank[page,id%rpp].float()
                score=((q[i,t].float()@key.T).relu()*weight[i,t].float()[:,None]).sum(0)/64
                ref[i,t,:256]=torch.where(valid,score,float('-inf'))
        got=out.cpu()
        assert torch.equal(torch.isfinite(got),torch.isfinite(ref)),('mask',b,step)
        mask=torch.isfinite(ref)
        torch.testing.assert_close(got[mask],ref[mask],rtol=1e-5,atol=1e-6)
        records.append(dict(batch=b,step=step,finite=int(mask.sum()),max_error=float((got[mask]-ref[mask]).abs().max()) if mask.any() else 0.))
print(json.dumps(dict(complete=True,paged_scores=records)),flush=True)
