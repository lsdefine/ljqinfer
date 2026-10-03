"""DSV41f streaming selection port. Caller owns TP group and prepared operands.
32 global index heads, 4/rank. Keys replicated; no model projections here.
Weights already include 1/sqrt(32). No hidden fallback. FP32 scores before TP sum.
"""
import torch
import torch.distributed as dist
import triton as tr
import triton.language as tl
from .sparse_index_merge import select_direct

@tr.jit
def _gather(K, PT, C, OUT, N:tl.constexpr, PAGE:tl.constexpr, B:tl.constexpr):
    x=tl.program_id(0)*B+tl.arange(0,B)
    row=x//128; d=x%128
    ctx=tl.load(C)
    page=tl.load(PT+row//PAGE,row<N,other=0)
    v=tl.load(K+(page*PAGE+row%PAGE)*128+d,(row<N)&(row<ctx),other=0)
    tl.store(OUT+x,v,row<N)

class TPIndexParallel:
    def __init__(self,group=None):
        self.group=group
        self.world=dist.get_world_size(group);self.rank=dist.get_rank(group)
        if self.world!=8:raise ValueError('TP8 required')
    def __call__(self,x):dist.all_reduce(x,group=self.group)
    def scatter_sum(self,x,out):dist.reduce_scatter_tensor(out,x,group=self.group)
    def gather_rows(self,x,out):dist.all_gather_into_tensor(out,x,group=self.group)

def select_paged(q,weights,pool,table,positions,context,result,*,parallel,logical_capacity,query_tile=3072,key_tile=6144):
    t,h,d=q.shape;n=logical_capacity;k=result.shape[1]
    assert h==4 and d==128 and parallel.world==8
    assert q.dtype in (torch.float16,torch.bfloat16,torch.float32)
    assert pool.ndim==3 and pool.shape[2]==128 and pool.is_contiguous()
    assert table.dtype==positions.dtype==context.dtype==result.dtype==torch.int64
    assert table.ndim==positions.ndim==1 and context.numel()==1
    assert weights.shape==(t,h) and positions.shape==(t,) and result.shape==(t,k)
    assert all(x.is_cuda and x.device==q.device and x.is_contiguous() for x in (q,weights,pool,table,positions,context,result))
    assert 0<k<=n<=table.numel()*pool.shape[1] and k+key_tile<=8192
    assert query_tile>0 and key_tile>0 and t>0
    keys=torch.empty((n,128),dtype=torch.float32,device=q.device)
    _gather[(tr.cdiv(n*128,1024),)](pool,table,context,keys,n,pool.shape[1],1024)
    # Context remains device data during replay. Invalid rows are masked before merge.
    pos=torch.minimum(positions,context.reshape(())-1).clamp_min(-1)
    return select_direct(q,weights,keys,pos,1,k,total_heads=32,reduce_scores=parallel,
                         query_tile=min(t,query_tile),key_tile=key_tile,result=result,weights_prescaled=True)
