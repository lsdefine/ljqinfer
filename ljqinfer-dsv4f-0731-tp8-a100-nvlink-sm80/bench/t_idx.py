
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # repo root on sys.path (this file lives in bench/)
import os, torch
from ops import index_topk
torch.manual_seed(0)
dev="cuda"
for start_pos, s, n in ((0, 4096, 1024), (8192, 4096, 3072), (12288, 3000, 3822)):
    ratio=4; h=8; d=128
    q=torch.randn(1,s,h,d,device=dev,dtype=torch.bfloat16)
    kv=torch.randn(1,n,d,device=dev,dtype=torch.bfloat16)
    w=torch.rand(1,s,h,device=dev,dtype=torch.bfloat16)
    import ops as _ops
    _ops._INDEX_TOPK_BYTES = 1 << 40; a=index_topk(q,kv,w,ratio,start_pos,7,512)
    _ops._INDEX_TOPK_BYTES = s*h*n*2//5; b=index_topk(q,kv,w,ratio,start_pos,7,512)
    _ops._INDEX_TOPK_BYTES = 1 << 30
    print(start_pos,s,n,"shape",tuple(a.shape),"equal",torch.equal(a,b), "sets_equal", all(set(a[0,i].tolist())==set(b[0,i].tolist()) for i in range(0,s,97)))
