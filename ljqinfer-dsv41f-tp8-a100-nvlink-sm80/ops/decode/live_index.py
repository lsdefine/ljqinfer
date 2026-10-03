"""Experimental single-graph paged FP32 index scores; no host length reads."""
import torch
import triton as tr
import triton.language as tl

@tr.jit
def _score(Q, W, Pool, PT, Pos, Out,
           H:tl.constexpr,D:tl.constexpr,N:tl.constexpr,RPP:tl.constexpr,RATIO:tl.constexpr,
           QS0:tl.constexpr,QS1:tl.constexpr,WS0:tl.constexpr,
           SCALE:tl.constexpr,BD:tl.constexpr,BK:tl.constexpr,NPROG:tl.constexpr,
           BH:tl.constexpr,QWIN:tl.constexpr,PTS:tl.constexpr):
    # Persistent grid: work is proportional to the live prefix, not to the
    # pool capacity, while the launch shape stays fixed for graph replay.
    # The per-head scores are one (BK,BD)x(BD,BH) mma so the scan runs on
    # tensor cores; the index query is already fp4-rounded upstream, so bf16
    # operands cost nothing that the block selection can see.
    pid=tl.program_id(0); t=tl.program_id(1)
    limit=(tl.load(Pos+t)+1)//RATIO
    nblk=(limit+BK-1)//BK
    dims=tl.arange(0,BD)
    hs=tl.arange(0,BH)
    hv=hs<H
    q=tl.load(Q+t*QS0+hs[:,None]*QS1+dims[None,:],hv[:,None]&(dims[None,:]<D),0)
    qt=tl.trans(q.to(tl.bfloat16))
    weight=tl.load(W+t*WS0+hs,hv,0).to(tl.float32)
    for block in range(pid,nblk,NPROG):
        rows=block*BK+tl.arange(0,BK)
        valid=(rows<limit)&(rows<N)
        # Batched rows: PT is the stacked table [B, pages]; token t belongs to
        # row t//QWIN.  PTS==0 keeps the single-slot table exactly as before.
        page=tl.maximum(tl.load(PT+(t//QWIN)*PTS+rows//RPP,valid,0),0)
        keys=tl.load(Pool+(page*RPP+rows%RPP)[:,None]*D+dims[None,:],valid[:,None]&(dims[None,:]<D),0)
        dot=tl.dot(keys.to(tl.bfloat16),qt)
        score=tl.sum(tl.maximum(dot,0)*weight[None,:],1)
        tl.store(Out+t*N+rows,score*SCALE,valid)

NPROG=128

_OUT={}


def _out_buf(t,max_rows,device):
    key=(t,max_rows,device)
    buf=_OUT.get(key)
    if buf is None:
        buf=torch.empty((t,max_rows),device=device,dtype=torch.float32)
        _OUT[key]=buf
    return buf


def scores(q,w,pool,table,pos,ratio,max_rows,total_heads=None):
    t,h,d=q.shape
    # A stacked page table [B, pages] means q carries B rows of t//B tokens each,
    # exactly like the paged attention kernel: B is read off the table, never
    # passed in, so one call shape serves single-send and batched decode.
    if table.dim()==2:
        b=table.shape[0]
        assert t%b==0, (t,b)
        qwin,pts=t//b,table.stride(0)
    else:
        qwin,pts=1,0
    # The radix leaf clamps its scan to NL=(pos+1)//ratio and never reads the
    # tail [NL,max_rows); _score writes every row below that bound itself.  So
    # the -inf fill was a capacity-sized write nobody observed.  A persistent
    # buffer also pins the address across graph replays.
    out=_out_buf(t,max_rows,q.device)
    BK=128
    nprog=min(tr.cdiv(max_rows,BK),NPROG)
    _score[(nprog,t)](q,w,pool,table,pos,out,h,d,max_rows,pool.shape[1],ratio,
        q.stride(0),q.stride(1),w.stride(0),d**-.5*(total_heads or h)**-.5,tr.next_power_of_2(d),BK,nprog,
        max(16,tr.next_power_of_2(h)),qwin,pts,num_warps=4,num_stages=2)
    return out

def paged_scores(q,w,index_pool,slot,pos,ratio,total_heads,reduce_scores=None):
    # TP: gather the tiny per-rank index query/weights (heads are sharded
    # contiguously across ranks) and score every head locally, instead of
    # all-reducing a capacity-sized FP32 score matrix.
    if reduce_scores is not None and getattr(reduce_scores,'world',1)>1:
        t,h,d=q.shape
        world=reduce_scores.world
        # One collective, not two: the per-head weight rides in an extra lane of
        # the query buffer (both are a few KB, so latency, not size, rules).
        packed=torch.empty((t,h,d+1),device=q.device,dtype=torch.float32)
        packed[...,:d]=q
        packed[...,d]=w
        pa=torch.empty((world,t,h,d+1),device=q.device,dtype=torch.float32)
        reduce_scores.gather_rows(packed,pa)
        q=pa[...,:d].permute(1,0,2,3).reshape(t,world*h,d)
        w=pa[...,d].permute(1,0,2).reshape(t,world*h)
        reduce_scores=None
    out=scores(q,w,index_pool.data,index_pool.pt.table[slot],pos,ratio,
               index_pool.pt.row_cap//ratio,total_heads)
    if reduce_scores is not None:
        reduce_scores(out)
    return out


def candidate_rows(score,pos,ratio,block_size,top_blocks):
    """Same block-max, forced newest block and stable logical-ID tie rule."""
    t,n=score.shape
    pad=(-n)%block_size
    if pad:
        score=torch.nn.functional.pad(score,(0,pad),value=-torch.inf)
    block_score=score.unflatten(-1,(-1,block_size)).amax(-1)
    lens=(pos+1)//ratio
    newest=(lens-1)//block_size
    ids=torch.arange(block_score.shape[1],device=pos.device)
    block_score=block_score.masked_fill(ids[None,:]>=newest[:,None],-torch.inf)
    take=min(top_blocks-1,block_score.shape[1])
    chosen=block_score.topk(take,dim=-1).indices
    chosen=chosen.masked_fill(~torch.isfinite(block_score.gather(1,chosen)),-1)
    if take<top_blocks-1:
        chosen=torch.nn.functional.pad(chosen,(0,top_blocks-1-take),value=-1)
    chosen=torch.cat((chosen,newest[:,None]),dim=-1)
    rows=(chosen[:,:,None]*block_size+torch.arange(block_size,device=pos.device)).flatten(1)
    return rows.masked_fill((rows<0)|(rows>=lens[:,None]),-1)
