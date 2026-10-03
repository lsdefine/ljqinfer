"""Sparse MLA leaves for A100. No weights, model scheduling, or implicit fallback.

ABI: one sequence; prepared/rotated queries. Indexer score = sum_h
weight[h] * relu(dot(q[h], k) / sqrt(128)); weights already include
1/sqrt(32). Main KV retains [physical_page, page_size, 576]. Indices
are logical token positions (unique valid entries, -1 padding). positions
and context are device tensors, so graph replay can change the prefix.
"""
import math
import torch
import triton as tr
import triton.language as tl


@tr.jit
def _scatter(X, POOL, TABLE, POS, Q:tl.constexpr, D:tl.constexpr,
             PAGE:tl.constexpr, CAP:tl.constexpr, B:tl.constexpr):
    row=tl.program_id(0); d=tl.arange(0,B)
    pos=tl.load(POS+row); valid=(pos>=0)&(pos<CAP)
    page=tl.load(TABLE+pos//PAGE, valid, other=0)
    v=tl.load(X+row*D+d,d<D,other=0)
    tl.store(POOL+(page*PAGE+pos%PAGE)*D+d,v,valid&(d<D))


@tr.jit
def _score(Q,K,W,TABLE,POS,CTX,S,T:tl.constexpr,N:tl.constexpr,
           PAGE:tl.constexpr,BQ:tl.constexpr,BN:tl.constexpr):
    rows=tl.program_id(0)*BQ+tl.arange(0,BQ)
    n=tl.program_id(1)*BN+tl.arange(0,BN)
    hd=tl.arange(0,BQ*32)
    d=tl.arange(0,128)
    q=tl.load(Q+(tl.program_id(0)*BQ*32+hd[:,None])*128+d[None,:],
              (tl.program_id(0)*BQ+hd[:,None]//32)<T,other=0)
    ctx=tl.load(CTX)
    page=tl.load(TABLE+n//PAGE,(n<N)&(n<ctx),other=0)
    base=tl.multiple_of((page*PAGE+n%PAGE)*128,16)
    k=tl.trans(tl.load(K+base[:,None]+d[None,:],
              (n[:,None]<N)&(n[:,None]<ctx),other=0))
    dot=tl.dot(q,k).reshape((BQ,32,BN))
    weights=tl.load(W+rows[:,None]*32+tl.arange(0,32)[None,:],rows[:,None]<T,other=0)
    scores=tl.sum(tl.maximum(dot,0.)*weights[:,:,None],axis=1)*0.08838834764831845
    pos=tl.load(POS+rows,rows<T,other=-1)
    scores=tl.where((n[None,:]<=pos[:,None])&(n[None,:]<ctx),scores,-float('inf'))
    tl.store(S+rows[:,None]*N+n[None,:],scores,(rows[:,None]<T)&(n[None,:]<N))


@tr.jit
def _clean_indices(I,V,OUT,T:tl.constexpr,K:tl.constexpr,B:tl.constexpr):
    x=tl.program_id(0)*B+tl.arange(0,B)
    idx=tl.load(I+x,x<T*K,other=0); val=tl.load(V+x,x<T*K,other=-float('inf'))
    tl.store(OUT+x,tl.where(val>-float('inf'),idx,-1),x<T*K)


@tr.jit
def _mla(Q,POOL,TABLE,IDX,POS,CTX,PART,LSE,OUT,H:tl.constexpr,
         PAGE:tl.constexpr,CAP:tl.constexpr,TOP:tl.constexpr,SPLITS:tl.constexpr,
         CHUNK:tl.constexpr,BN:tl.constexpr):
    row=tl.program_id(0); split=tl.program_id(1)
    h=tl.arange(0,16); d=tl.arange(0,512); r=tl.arange(0,64)
    q=tl.load(Q+row*H*576+h[:,None]*576+d[None,:],h[:,None]<H,other=0)
    qr=tl.load(Q+row*H*576+h[:,None]*576+512+r[None,:],h[:,None]<H,other=0)
    pos=tl.load(POS+row); ctx=tl.minimum(tl.load(CTX),CAP)
    acc=tl.full((16,512),0,tl.float32)
    m=tl.full((16,),-float('inf'),tl.float32); z=tl.full((16,),0,tl.float32)
    for start in range(split*CHUNK,(split+1)*CHUNK,BN):
        ix=start+tl.arange(0,BN)
        tok=tl.load(IDX+row*TOP+ix,ix<TOP,other=-1)
        valid=(ix<TOP)&(tok>=0)&(tok<ctx)&(tok<=pos)
        page=tl.load(TABLE+tok//PAGE,valid,other=0)
        base=tl.multiple_of((page*PAGE+tok%PAGE)*576, 16)
        kv=tl.trans(tl.load(POOL+base[:,None]+d[None,:],valid[:,None],other=0))
        kr=tl.trans(tl.load(POOL+base[:,None]+512+r[None,:],valid[:,None],other=0))
        score=(tl.dot(q,kv)+tl.dot(qr,kr))*0.0625
        score=tl.where(valid[None,:],score,-float('inf'))
        nm=tl.maximum(m,tl.max(score,1))
        safe=tl.where(nm == -float('inf'),0.,nm)
        alpha=tl.exp(m-safe)
        prob=tl.exp(score-safe[:,None])
        acc=acc*alpha[:,None]+tl.dot(prob.to(kv.dtype),tl.trans(kv))
        z=z*alpha+tl.sum(prob,1); m=nm
    result=acc/tl.where(z>0,z,1.)[:,None]
    if SPLITS == 1:
        tl.store(OUT+row*H*512+h[:,None]*512+d[None,:],result,h[:,None]<H)
    else:
        tl.store(PART+((row*SPLITS+split)*H+h[:,None])*512+d[None,:],result,h[:,None]<H)
        tl.store(LSE+(row*SPLITS+split)*H+h,tl.where(z>0,m+tl.log(z),-float('inf')),h<H)


@tr.jit
def _merge(PART,LSE,OUT,H:tl.constexpr,SPLITS:tl.constexpr,BS:tl.constexpr):
    row=tl.program_id(0); h=tl.program_id(1)
    s=tl.arange(0,BS);d=tl.arange(0,512)
    l=tl.load(LSE+(row*SPLITS+s)*H+h,s<SPLITS,other=-float('inf'))
    mx=tl.max(l,0);mx=tl.where(mx == -float('inf'),0.,mx)
    w=tl.exp(l-mx); denom=tl.sum(w,0)
    v=tl.load(PART+((row*SPLITS+s[:,None])*H+h)*512+d[None,:],s[:,None]<SPLITS,other=0)
    y=tl.sum(v*w[:,None],0)/tl.where(denom>0,denom,1.)
    tl.store(OUT+row*H*512+h*512+d,y)


def _tensors(*xs):
    assert all(x.is_cuda and x.is_contiguous() for x in xs)
    assert len({x.device for x in xs})==1


def scatter_keys(x,pool,table,positions):
    """Caller owns unique write positions and valid physical page table."""
    _tensors(x,pool,table,positions)
    assert x.ndim==2 and pool.ndim==3 and x.shape[1]==pool.shape[2]
    assert x.dtype==pool.dtype and table.dtype==torch.int64 and positions.dtype==torch.int64
    assert positions.numel()==x.shape[0] and table.numel()<=pool.shape[0]
    _scatter[(x.shape[0],)](x,pool,table,positions,x.shape[0],x.shape[1],pool.shape[1],table.numel()*pool.shape[1],tr.next_power_of_2(x.shape[1]))


def index_scores(q,kpool,weights,table,positions,context,out,*,block_q=4,block_n=128,warps=4):
    _tensors(q,kpool,weights,table,positions,context,out)
    t=q.shape[0];n=out.shape[1]
    assert q.shape==(t,32,128) and kpool.ndim==3 and kpool.shape[-1]==128
    assert q.dtype==kpool.dtype and q.dtype in (torch.float16,torch.bfloat16)
    assert weights.shape==(t,32) and weights.dtype==torch.float32
    assert out.shape==(t,n) and out.dtype==torch.float32
    assert positions.shape==(t,) and positions.dtype==table.dtype==context.dtype==torch.int64
    assert context.numel()==1 and n<=table.numel()*kpool.shape[1]
    _score[(tr.cdiv(t,block_q),tr.cdiv(n,block_n))](q,kpool,weights,table,positions,context,out,t,n,kpool.shape[1],block_q,block_n,num_warps=warps,num_stages=1)


def topk_indices(scores,values,indices64,out):
    """Exact torch CUDA top-k baseline; unordered ties unspecified. No D2H."""
    _tensors(scores,values,indices64,out)
    t,k=out.shape
    assert out.dtype==torch.int32 and indices64.dtype==torch.int64
    assert scores.dtype==values.dtype==torch.float32 and values.shape==indices64.shape==out.shape
    assert scores.shape[0]==t and 0<k<=scores.shape[1]
    torch.topk(scores,k,dim=-1,sorted=False,out=(values,indices64))
    _clean_indices[(tr.cdiv(t*k,256),)](indices64,values,out,t,k,256)


def make_mla_workspace(q,*,splits):
    assert splits>0
    t,h,_=q.shape
    return (torch.empty((t,splits,h,512),device=q.device,dtype=torch.float32),
            torch.empty((t,splits,h),device=q.device,dtype=torch.float32))


def sparse_mla(q,pool,table,indices,positions,context,out,workspace,*,splits=1,block_n=32,warps=4):
    """Preselected sparse latent attention, no projection/communication cost.

    Phase-specific split count is explicit, not chosen from runtime context.
    Duplicate valid indices are not allowed; -1 holes are allowed anywhere.
    Empty rows return zero. Tables must contain valid allocated physical pages.
    """
    part,lse=workspace
    _tensors(q,pool,table,indices,positions,context,out,part,lse)
    t,h,d=q.shape;top=indices.shape[1]
    assert d==576 and 0<h<=16 and pool.ndim==3 and pool.shape[-1]==576
    assert q.dtype==pool.dtype==out.dtype and q.dtype in (torch.float16,torch.bfloat16)
    assert indices.dtype==torch.int32 and indices.shape[0]==t and top>0
    assert positions.shape==(t,) and positions.dtype==context.dtype==table.dtype==torch.int64
    assert context.numel()==1 and out.shape==(t,h,512)
    assert part.shape==(t,splits,h,512) and lse.shape==(t,splits,h)
    assert part.dtype==lse.dtype==torch.float32
    chunk=tr.cdiv(top,splits*block_n)*block_n
    _mla[(t,splits)](q,pool,table,indices,positions,context,part,lse,out,h,pool.shape[1],table.numel()*pool.shape[1],top,splits,chunk,block_n,num_warps=warps,num_stages=1)
    if splits>1:
        _merge[(t,h)](part,lse,out,h,splits,tr.next_power_of_2(splits))
    return out
