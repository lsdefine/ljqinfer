import torch
import triton as tr
import triton.language as tl

@tr.jit
def _score_local(Q,K,W,TABLE,POS,CTX,S,T:tl.constexpr,N:tl.constexpr,
           PAGE:tl.constexpr,BQ:tl.constexpr,BN:tl.constexpr):
    rows=tl.program_id(0)*BQ+tl.arange(0,BQ)
    n=tl.program_id(1)*BN+tl.arange(0,BN)
    hd=tl.arange(0,BQ*4)
    d=tl.arange(0,128)
    q=tl.load(Q+(tl.program_id(0)*BQ*4+hd[:,None])*128+d[None,:],
              (tl.program_id(0)*BQ+hd[:,None]//4)<T,other=0)
    ctx=tl.load(CTX)
    page=tl.load(TABLE+n//PAGE,(n<N)&(n<ctx),other=0)
    base=tl.multiple_of((page*PAGE+n%PAGE)*128,16)
    k=tl.trans(tl.load(K+base[:,None]+d[None,:],
              (n[:,None]<N)&(n[:,None]<ctx),other=0))
    dot=tl.dot(q,k, input_precision="ieee").reshape((BQ,4,BN))
    weights=tl.load(W+rows[:,None]*4+tl.arange(0,4)[None,:],rows[:,None]<T,other=0)
    scores=tl.sum(tl.maximum(dot,0.)*weights[:,:,None],axis=1)*0.08838834764831845
    pos=tl.load(POS+rows,rows<T,other=-1)
    scores=tl.where((n[None,:]<=pos[:,None])&(n[None,:]<ctx),scores,-float('inf'))
    tl.store(S+rows[:,None]*N+n[None,:],scores,(rows[:,None]<T)&(n[None,:]<N))
