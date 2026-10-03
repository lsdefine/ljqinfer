"""Exact all-head index scoring with query-row ownership, TP8.
Prepared 4-head queries are exchanged, never the T x N score matrix.
Keys/table/positions/context must be replicated. Weights include 1/sqrt(32).
"""
import torch
import triton as tr
import triton.language as tl
from .sparse_index_fast import extension

@tr.jit(do_not_specialize=['T', 'N', 'START', 'ROWS'], do_not_specialize_on_alignment=['T', 'N', 'START', 'ROWS'])
def _score_query(Q,K,W,TABLE,POS,CTX,S,T,N,
                 PAGE:tl.constexpr,START,ROWS,
                 BQ:tl.constexpr,BN:tl.constexpr):
    rr=tl.program_id(0)*BQ+tl.arange(0,BQ)
    hd=tl.arange(0,BQ*32)
    row=START+tl.program_id(0)*BQ+hd//32
    head=hd%32
    d=tl.arange(0,128)
    offset=((head//4)*T+row)*4+head%4
    q=tl.load(Q+offset[:,None]*128+d[None,:],(row[:,None]<T)&((tl.program_id(0)*BQ+hd[:,None]//32)<ROWS),other=0)
    n=tl.program_id(1)*BN+tl.arange(0,BN)
    ctx=tl.load(CTX)
    page=tl.load(TABLE+n//PAGE,(n<N)&(n<ctx),other=0)
    base=tl.multiple_of((page*PAGE+n%PAGE)*128,16)
    key=tl.trans(tl.load(K+base[:,None]+d[None,:],(n[:,None]<N)&(n[:,None]<ctx),other=0))
    dot=tl.dot(q,key).reshape((BQ,32,BN))
    weight=tl.load(W+offset,(row<T)&((tl.program_id(0)*BQ+hd//32)<ROWS),other=0).reshape((BQ,32))
    score=tl.sum(tl.maximum(dot,0.)*weight[:,:,None],1)*0.08838834764831845
    pos=tl.load(POS+START+rr,(START+rr<T)&(rr<ROWS),other=-1)
    score=tl.where((n[None,:]<=pos[:,None])&(n[None,:]<ctx),score,-float('inf'))
    tl.store(S+rr[:,None]*N+n[None,:],score,(rr[:,None]<ROWS)&(n[None,:]<N))

def _select_paged(q,weights,pool,table,positions,context,result,*,selector,parallel,logical_capacity,query_tile=256,key_tile=None,scratch=None):
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
    mod=selector
    queries=(scratch['queries'] if scratch is not None else torch.empty((8*t,4,128),device=q.device,dtype=q.dtype))
    ww=(scratch['weights'] if scratch is not None else torch.empty((8*t,4),device=q.device,dtype=weights.dtype))
    parallel.gather_rows(q,queries);parallel.gather_rows(weights,ww)
    local=tr.cdiv(t,8);start=parallel.rank*local
    selected=(scratch['selected'] if scratch is not None else torch.empty((local,k),device=q.device,dtype=torch.int32))
    pos=(scratch['pos'] if scratch is not None else torch.empty((local,),device=q.device,dtype=torch.int64))
    pos.fill_(-1)
    valid=max(0,min(local,t-start))
    if valid:pos[:valid].copy_(torch.minimum(positions[start:start+valid],context.reshape(())-1).clamp_min(-1))
    scores=(scratch['scores'] if scratch is not None else torch.empty((min(local,query_tile),n),device=q.device,dtype=torch.float32))
    for lo in range(0,local,query_tile):
        rows=min(query_tile,local-lo)
        _score_query[(tr.cdiv(rows,2),tr.cdiv(n,128))](queries,pool,ww,table,positions,context,scores,t,n,pool.shape[1],start+lo,rows,2,128,num_warps=4)
        mod.select_out(scores[:rows],pos[lo:lo+rows],selected[lo:lo+rows])
    gathered=(scratch['gathered'] if scratch is not None else torch.empty((local*8,k),device=q.device,dtype=torch.int32))
    parallel.gather_rows(selected,gathered)
    result.copy_(gathered[:t])
    return result


def select_prefill(q,weights,pool,table,positions,context,result,*,parallel,logical_capacity,query_tile=1536,key_tile=None,scratch=None):
    # Explicit prefill entry, single-CTA-per-row top-k for every shape.
    return _select_paged(q,weights,pool,table,positions,context,result,selector=extension(),parallel=parallel,logical_capacity=logical_capacity,query_tile=query_tile,key_tile=key_tile,scratch=scratch)
