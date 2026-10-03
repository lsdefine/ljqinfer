"""Two prefill entry points, GPU routing lists + grouped BF16 Tensor Core GEMM.

Workspace retains E*(T*topk) routing capacity for arbitrary skew.
Large prefill maps a bounded tile grid through device-resident counts;
no launch grid proportional to E*T*topk. At >=128 actual tokens, dequantize each
matrix once into preallocated BF16 scratch and use pipelined grouped GEMM.
Below 128 actual tokens, reuse the decode expert implementation.
No hot-path allocations or CPU tensor reads. Explicit-out CUDA Graph safe.
"""
import torch
import triton
import triton.language as tl
from ops.moe_common import combine,validate,DENSE_PREFILL_MIN_TOKENS
from ops.moe_decode import _run as _run_decode


@triton.jit(do_not_specialize=['R'], do_not_specialize_on_alignment=['R'])
def _dispatch(Ids,Slots,Counts,R,E:tl.constexpr,B:tl.constexpr):
    r=tl.program_id(0)*B+tl.arange(0,B)
    e=tl.load(Ids+r,r<R,-1)
    valid=(r<R)&(e>=0)&(e<E)
    pos=tl.atomic_add(Counts+e,1,valid,sem='relaxed')
    tl.store(Slots+e*R+pos,r,valid)


@triton.jit
def unpack(P,S,W,E:tl.constexpr,N:tl.constexpr,K:tl.constexpr,G:tl.constexpr,FP8:tl.constexpr,B:tl.constexpr,PACK:tl.constexpr):
    e=tl.program_id(1)
    if FP8:
        q=tl.program_id(0)*B+tl.arange(0,B);n=q//K;k=q%K;mask=q<N*K
        b=tl.load(P+e*N*K+q,mask,0).to(tl.int32)
        # E4M3FN exponent rebias, preserving subnormals and NaNs on SM80.
        mag=b&127
        v=((mag<<20)+(120<<23)).to(tl.float32,bitcast=True)
        v=tl.where(mag<8,mag.to(tl.float32)*0.001953125,v)
        v=tl.where(mag==127,float('nan'),v)
        v=tl.where((b&128)!=0,-v,v)
        sc=tl.load(S+(e*tl.cdiv(N,128)+n//128)*tl.cdiv(K,128)+k//128,mask,0)
    else:
        # Load each packed word/scale once; byte loads support unaligned views.
        j=tl.program_id(0)*(B//PACK)+tl.arange(0,B//PACK)
        if PACK==8: ptr=P.to(tl.pointer_type(tl.uint32))
        else: ptr=P
        b=tl.load(ptr+e*(N*K//PACK)+j,j<N*K//PACK,0).to(tl.uint32)
        sc0=tl.load(S+e*(N*K//G)+j//(G//PACK),j<N*K//PACK,0).to(tl.float32)
        nib=((b[:,None]>>(4*tl.arange(0,PACK)[None,:]))&15).to(tl.int32)
        # Exact signed INT4 -> FP32 in the unit-ULP binade, without I2F.
        v=((nib^8)|0x4b000000).to(tl.float32,bitcast=True)-8388616.0
        v=tl.reshape(v,(B,));sc=tl.reshape(tl.broadcast_to(sc0[:,None],(B//PACK,PACK)),(B,))
        q=tl.program_id(0)*B+tl.arange(0,B);mask=q<N*K
    tl.store(W+e*N*K+q,(v*sc).to(tl.bfloat16),mask)

@triton.jit(do_not_specialize=['R'], do_not_specialize_on_alignment=['R'])
def dense_gu(X,Slots,Counts,W,Y,R,K:tl.constexpr,N:tl.constexpr,TOP:tl.constexpr,DOWN:tl.constexpr,E:tl.constexpr,BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr):
    es=tl.arange(0,triton.next_power_of_2(E))
    counts=tl.load(Counts+es,es<E,0)
    # Prefix tile counts map each CTA to one expert/M tile, including skew.
    ends=tl.cumsum(tl.cdiv(counts,BM),0)
    task=tl.program_id(1)
    e=tl.sum((task>=ends).to(tl.int32),0)
    start=tl.sum(tl.where(es==e-1,ends,0),0)
    cnt=tl.sum(tl.where(es==e,counts,0),0)
    n=tl.program_id(0)*BN+tl.arange(0,BN);kk=tl.arange(0,BK)
    m0=(task-start)*BM
    if e<E:
        m=m0+tl.arange(0,BM)
        r=tl.load(Slots+e*R+m,m<cnt,0)
        xr=r if DOWN else r//TOP
        acc=tl.zeros((BM,BN),tl.float32)
        au=tl.zeros((BM,BN),tl.float32)
        for st in range(tl.cdiv(K,BK)):
            k=st*BK+kk
            a=tl.load(X+xr[:,None]*K+k[None,:],(m[:,None]<cnt)&(k[None,:]<K),0)
            b=tl.load(W+e*N*K+n[:,None]*K+k[None,:],(n[:,None]<N//2)&(k[None,:]<K),0)
            u=tl.load(W+e*N*K+(n+N//2)[:,None]*K+k[None,:],(n[:,None]<N//2)&(k[None,:]<K),0)
            acc=tl.dot(a,tl.trans(b),acc)
            au=tl.dot(a,tl.trans(u),au)
        # Preserve both BF16 projection boundaries before fused activation.
        acc=acc.to(tl.bfloat16).to(tl.float32)
        au=au.to(tl.bfloat16).to(tl.float32)
        acc=acc*tl.sigmoid(acc)*au
        tl.store(Y+r[:,None]*(N//2)+n[None,:],acc,(m[:,None]<cnt)&(n[None,:]<N//2))

@triton.jit(do_not_specialize=['R'], do_not_specialize_on_alignment=['R'])
def dense(X,Slots,Counts,W,Y,R,K:tl.constexpr,N:tl.constexpr,TOP:tl.constexpr,DOWN:tl.constexpr,E:tl.constexpr,BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr):
    es=tl.arange(0,triton.next_power_of_2(E))
    counts=tl.load(Counts+es,es<E,0)
    # Prefix tile counts map each CTA to one expert/M tile, including skew.
    ends=tl.cumsum(tl.cdiv(counts,BM),0)
    task=tl.program_id(1)
    e=tl.sum((task>=ends).to(tl.int32),0)
    start=tl.sum(tl.where(es==e-1,ends,0),0)
    cnt=tl.sum(tl.where(es==e,counts,0),0)
    n=tl.program_id(0)*BN+tl.arange(0,BN);kk=tl.arange(0,BK)
    m0=(task-start)*BM
    if e<E:
        m=m0+tl.arange(0,BM)
        r=tl.load(Slots+e*R+m,m<cnt,0)
        xr=r if DOWN else r//TOP
        acc=tl.zeros((BM,BN),tl.float32)
        for st in range(tl.cdiv(K,BK)):
            k=st*BK+kk
            a=tl.load(X+xr[:,None]*K+k[None,:],(m[:,None]<cnt)&(k[None,:]<K),0)
            b=tl.load(W+e*N*K+n[:,None]*K+k[None,:],(n[:,None]<N)&(k[None,:]<K),0)
            acc=tl.dot(a,tl.trans(b),acc)
        if not DOWN:acc=acc.to(tl.bfloat16).to(tl.float32)
        tl.store(Y+r[:,None]*N+n[None,:],acc,(m[:,None]<cnt)&(n[None,:]<N))

@triton.jit(do_not_specialize=['R'], do_not_specialize_on_alignment=['R'])
def activate(P,A,R,I:tl.constexpr,B:tl.constexpr):
    j=tl.program_id(0)*B+tl.arange(0,B);r=j//I;n=j%I
    g=tl.load(P+r*2*I+n,j<R*I,0)
    u=tl.load(P+r*2*I+I+n,j<R*I,0)
    tl.store(A+j,g*tl.sigmoid(g)*u,j<R*I)

def _run(x,ids,rw,gu,gs,down,ds,out,ws,group=64,fp8=False,shared=None,done=None,preunpacked=False):
    t,h,e,i,top=validate(x,ids,rw,gu,gs,down,ds,out,ws,group,fp8)
    if t<DENSE_PREFILL_MIN_TOKENS:
        return _run_decode(x,ids,rw,gu,gs,down,ds,out,ws,group,fp8,shared,done)
    if ws.weight is None:
        raise ValueError("large prefill requires Workspace.create weight scratch")
    r=t*top;ws.counts.zero_()
    _dispatch[(triton.cdiv(r,128),)](ids,ws.slots,ws.counts,r,e,128)
    if not preunpacked:
        unpack[(triton.cdiv(2*i*h,2048),e)](gu,gs,ws.weight,e,2*i,h,group,False,2048,8 if gu.data_ptr()%4==0 else 2)
    dense_gu[(triton.cdiv(i,64),triton.cdiv(r,128)+e)](x,ws.slots,ws.counts,ws.weight,ws.hidden,r,h,2*i,top,False,e,128,64,64,num_warps=4,num_stages=3)
    unpack[(triton.cdiv(h*i,4096),e)](down,ds,ws.weight,e,h,i,group,fp8,4096,8 if down.data_ptr()%4==0 else 2)
    dense[(triton.cdiv(h,128),triton.cdiv(r,128)+e)](ws.hidden,ws.slots,ws.counts,ws.weight,ws.partial,r,i,h,top,True,e,128,128,64,num_warps=4,num_stages=3)
    if done is not None:
        torch.cuda.current_stream(x.device).wait_event(done)
    combine[(t,triton.cdiv(h,1024))](ws.partial,ids,rw,out,t,h,top,e,1024,triton.next_power_of_2(top),shared,FUSED=shared is not None)
    return out

def prefill_int4(x,ids,rw,gu,gs,down,ds,out,ws,group=64):
    return _run(x,ids,rw,gu,gs,down,ds,out,ws,group,False)


def prefill_int4_fp8(x,ids,rw,gu,gs,down,ds,out,ws,group=64):
    return _run(x,ids,rw,gu,gs,down,ds,out,ws,group,True)


def moe_prefill(x,weights,out,workspace,down_dtype="int4",*,group=64,groups=8,
                 group_topk=4,routed_scale=2.5,tp_size=1,all_reduce=None):
    """Complete prefill MoE layer; see ops.moe_layer for the ABI and TP contract."""
    from ops.moe_layer import run_layer
    return run_layer(_run,x,weights,out,workspace,down_dtype,group,groups,group_topk,
                     routed_scale,tp_size,all_reduce,prepare_gu=True)
