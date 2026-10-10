"""NPU leaf: legacy equivalence, soft bias, per-row isolation and graph replay."""
import ctypes as C, sys
from pathlib import Path
import torch, torch_npu
P=C.c_void_p;U=C.c_uint32
lib=C.CDLL(sys.argv[1]);torch.npu.set_device(0)
size=lib.dec_greedy_workspace_bytes;size.argtypes=[U];size.restype=C.c_uint64
old=lib.dec_sample_local;old.argtypes=[P]*6+[U]*2;old.restype=C.c_int
new=lib.dec_sample_bias_local;new.argtypes=[P]*7+[U]*2;new.restype=C.c_int

def run(b,rank):
    rows=b*6
    torch.manual_seed(7)
    x=torch.randn(rows,16160,device='npu',dtype=torch.float32)
    partial=torch.empty(size(b),device='npu',dtype=torch.uint8)
    pair=torch.empty(rows,2,device='npu',dtype=torch.float32)
    temp=torch.zeros(b,device='npu');seed=torch.tensor([37],dtype=torch.int32,device='npu')
    bias=torch.zeros(rows,device='npu')
    def call(fn):
        tensors=[x,partial,pair,temp,seed]+([bias] if fn is new else [])
        rc=fn(P(torch.npu.current_stream().npu_stream),*[P(t.data_ptr()) for t in tensors],b,rank)
        assert rc==0,rc
    for temperature in (0.,1.):
        temp.fill_(temperature);call(old);torch.npu.synchronize();ref=pair.cpu().clone()
        call(new);torch.npu.synchronize();assert torch.equal(ref,pair.cpu()),(b,rank,temperature)
    temp.zero_();x.fill_(-100);x[:,7]=5
    own=128822-rank*16160
    if 0<=own<16160: x[:,own]=0
    call(new);torch.npu.synchronize();assert (pair[:,1].cpu()==rank*16160+7).all()
    bias[::2]=12
    call(new);torch.npu.synchronize()
    expected=torch.full((rows,),rank*16160+7,dtype=torch.float32)
    if 0<=own<16160:expected[::2]=128822
    assert torch.equal(pair[:,1].cpu(),expected),(b,rank,pair.cpu())
    # Finite bias is not a forced token: a 20-logit margin still wins.
    x[:,7]=20;call(new);torch.npu.synchronize();assert (pair[:,1].cpu()==rank*16160+7).all()
    x[:,7]=5
    g=torch.npu.NPUGraph()
    with torch.npu.graph(g):call(new)
    for val in (0.,12.,0.):
        bias.fill_(val);g.replay();torch.npu.synchronize()
        target=128822 if val and 0<=own<16160 else rank*16160+7
        assert (pair[:,1].cpu()==target).all(),(b,rank,val,pair.cpu())
    print('PASS B',b,'rank',rank,flush=True)

for b in (1,2,4):
    for rank in (0,7):run(b,rank)
print('ALL_PASS')
