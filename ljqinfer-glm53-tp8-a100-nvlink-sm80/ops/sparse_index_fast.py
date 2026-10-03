"""TP8 paged index selection: prefill local score + exact radix selection.
Scores are summed across all heads BEFORE selecting. Explicit query tile.
"""
import os
from functools import lru_cache
from pathlib import Path
import torch
import triton as tr
from .sparse_index_score_tp import _score_local

@lru_cache(None)
def extension():
    from torch.utils.cpp_extension import load
    os.environ.setdefault('TORCH_CUDA_ARCH_LIST','8.0')
    os.environ.setdefault('MAX_JOBS','2')
    return load(name='glm53_index_radix_prefill',sources=[str(Path(__file__).with_name('sparse_index_radix.cu'))],extra_cuda_cflags=['-O3'],verbose=False)

def select_paged(q,weights,pool,table,positions,context,result,*,parallel,logical_capacity,query_tile=1024,key_tile=None):
    t,h,d=q.shape;n=logical_capacity;k=result.shape[1]
    assert h==4 and d==128 and parallel.world==8
    assert pool.ndim==3 and pool.shape[2]==128
    assert q.dtype==pool.dtype and q.dtype in (torch.float16,torch.bfloat16)
    assert weights.dtype==torch.float32
    assert table.dtype==positions.dtype==context.dtype==result.dtype==torch.int64
    assert table.ndim==positions.ndim==1 and context.numel()==1
    assert weights.shape==(t,h) and positions.shape==(t,) and result.shape==(t,k)
    assert all(x.is_cuda and x.device==q.device and x.is_contiguous() for x in (q,weights,pool,table,positions,context,result))
    assert 0<k<=n<=table.numel()*pool.shape[1] and t>0 and query_tile>0
    mod=extension()
    for lo in range(0,t,query_tile):
        hi=min(t,lo+query_tile);rows=hi-lo
        # Pad query rows to world size, independent of device context length.
        padded=tr.cdiv(rows,8)*8;local=padded//8
        scores=torch.empty((padded,n),device=q.device,dtype=torch.float32)
        if padded>rows:scores[rows:].zero_()
        _score_local[(tr.cdiv(rows,8),tr.cdiv(n,64))](q[lo:hi],pool,weights[lo:hi],table,positions[lo:hi],context,scores,rows,n,pool.shape[1],8,64)
        owned=torch.empty((local,n),device=q.device,dtype=torch.float32)
        parallel.scatter_sum(scores,owned)
        pos=torch.full((padded,),-1,device=q.device,dtype=torch.int64)
        pos[:rows].copy_(torch.minimum(positions[lo:hi],context.reshape(())-1).clamp_min(-1))
        start=parallel.rank*local
        selected=torch.empty((local,k),device=q.device,dtype=torch.int32)
        mod.select_out(owned,pos[start:start+local],selected)
        gathered=torch.empty((padded,k),device=q.device,dtype=torch.int32)
        parallel.gather_rows(selected,gathered)
        result[lo:hi].copy_(gathered[:rows])
    return result
