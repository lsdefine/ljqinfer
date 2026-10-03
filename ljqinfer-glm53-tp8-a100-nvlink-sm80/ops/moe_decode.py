"""Two decode entry points. Direct routed GEMV, no expert dispatch/sort.

No allocations/copies in the launch path. Warm JIT before CUDA Graph capture.
BF16 operands, FP32 reductions and output. Bounded K tiles avoid register
spills; output-channel-first CTA order reuses selected expert weights.
"""
import torch
import triton
import triton.language as tl
from ops.moe_common import int4_values,fp8_values,combine,validate


@triton.jit
def _gate_up(X,Ids,P,S,A,H:tl.constexpr,I:tl.constexpr,E:tl.constexpr,TOP:tl.constexpr,G:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr):
    r=tl.program_id(1); n=tl.program_id(0)*BN+tl.arange(0,BN)
    e=tl.load(Ids+r); valid=(e>=0)&(e<E)
    kk=tl.arange(0,BK)
    ag=tl.full((BN,),0,tl.float32); au=tl.full((BN,),0,tl.float32)
    for k0 in range(tl.cdiv(H,BK)):
        k=k0*BK+kk
        x=tl.load(X+(r//TOP)*H+k,k<H,0).to(tl.float32)
        mask=valid&(n[:,None]<I)&(k[None,:]<H)
        g=int4_values(P,S,e,n[:,None],k[None,:],2*I,H,G,mask).to(tl.float32)
        u=int4_values(P,S,e,(n+I)[:,None],k[None,:],2*I,H,G,mask).to(tl.float32)
        ag+=tl.sum(g*x[None,:],axis=1); au+=tl.sum(u*x[None,:],axis=1)
    # Match oracle's BF16 projection boundaries before SwiGLU.
    ag=ag.to(tl.bfloat16).to(tl.float32);au=au.to(tl.bfloat16).to(tl.float32)
    z=ag*tl.sigmoid(ag)*au
    tl.store(A+r*I+n,z,n<I)


@triton.jit
def _down(A,Ids,P,S,Y,H:tl.constexpr,I:tl.constexpr,E:tl.constexpr,G:tl.constexpr,FP8:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr):
    r=tl.program_id(1);n=tl.program_id(0)*BN+tl.arange(0,BN)
    e=tl.load(Ids+r);valid=(e>=0)&(e<E)
    acc=tl.full((BN,),0,tl.float32); kk=tl.arange(0,BK)
    for k0 in range(tl.cdiv(I,BK)):
        k=k0*BK+kk
        a=tl.load(A+r*I+k,k<I,0).to(tl.float32)
        mask=valid&(n[:,None]<H)&(k[None,:]<I)
        if FP8: w=fp8_values(P,S,e,n[:,None],k[None,:],H,I,mask)
        else: w=int4_values(P,S,e,n[:,None],k[None,:],H,I,G,mask)
        acc+=tl.sum(w.to(tl.float32)*a[None,:],axis=1)
    tl.store(Y+r*H+n,acc,n<H)


@triton.jit
def _round_bf16_f32(x):
    # Preserve BF16 round-to-nearest-even; widen by placing its bits in
    # the high half of FP32 instead of asking for another conversion.
    return tl.inline_asm_elementwise(
        "{ .reg .b16 b,z; mov.b16 z,0; cvt.rn.bf16.f32 b,$1; mov.b32 $0,{z,b}; }",
        constraints="=f,f", args=[x], dtype=tl.float32, is_pure=True, pack=1)


@triton.jit
def _round_bf16_pair(x):
    return tl.inline_asm_elementwise(
        "{ .reg .b32 b; .reg .b16 lo,hi,z; mov.b16 z,0; cvt.rn.bf16x2.f32 b,$3,$2; mov.b32 {lo,hi},b; mov.b32 $0,{z,lo}; mov.b32 $1,{z,hi}; }",
        constraints="=f,=f,f,f",args=[x],dtype=tl.float32,is_pure=True,pack=2)
@triton.jit
def _project_packed(X,Ids,P,S,Z,N_OUT:tl.constexpr,K_IN:tl.constexpr,E:tl.constexpr,TOP:tl.constexpr,G:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr,PACK:tl.constexpr,SPLIT:tl.constexpr,LOP:tl.constexpr=True,PAIR:tl.constexpr=True):
    r=tl.program_id(1);n=tl.program_id(0)*BN+tl.arange(0,BN)
    sk=tl.program_id(2) if SPLIT else 0;j=sk*(BK//PACK)+tl.arange(0,BK//PACK)
    e=tl.load(Ids+r);valid=(e>=0)&(e<E)
    mask=valid&(n[None,:]<N_OUT)&(j[:,None]<K_IN//PACK)
    if PACK==8: ptr=P.to(tl.pointer_type(tl.uint32))
    else: ptr=P.to(tl.pointer_type(tl.uint16))
    v=tl.load(ptr+(e*N_OUT+n[None,:])*(K_IN//PACK)+j[:,None],mask,0).to(tl.uint32)
    sc=tl.load(S+(e*N_OUT+n[None,:])*(K_IN//G)+j[:,None]//(G//PACK),mask,0).to(tl.float32)
    acc=tl.full((BK//PACK,BN),0,tl.float32)
    for idx in tl.static_range(PACK):
        if LOP:
            bits=tl.inline_asm_elementwise("lop3.b32 $0,$1,15,0x4b000008,0x6a;",constraints="=r,r",args=[v>>(4*idx)],dtype=tl.uint32,is_pure=True,pack=1)
            q=bits.to(tl.float32,bitcast=True)-8388616.0
        else:
            q=((v>>(4*idx))&15).to(tl.int32)
            q=((q^8)|0x4b000000).to(tl.float32,bitcast=True)-8388616.0
        if PAIR:
            w=_round_bf16_pair(q*sc)
        else:
            w=_round_bf16_f32(q*sc)
        x=tl.load(X+(r//TOP if SPLIT else r)*K_IN+PACK*j+idx,PACK*j+idx<K_IN,0).to(tl.float32)
        acc=acc+w*x[:,None]
    z=tl.sum(acc,0)
    tl.store(Z+((r*tl.cdiv(K_IN,BK)+sk) if SPLIT else r)*N_OUT+n,z,n<N_OUT)



@triton.jit
def _reduce_activate(Z,A,H:tl.constexpr,I:tl.constexpr,SPLITS:tl.constexpr,BS:tl.constexpr,BN:tl.constexpr):
    r=tl.program_id(0);n=tl.program_id(1)*BN+tl.arange(0,BN)
    sk=tl.arange(0,BS)
    g=tl.load(Z+(r*SPLITS+sk[:,None])*2*I+n[None,:],(sk[:,None]<SPLITS)&(n[None,:]<I),0)
    u=tl.load(Z+(r*SPLITS+sk[:,None])*2*I+I+n[None,:],(sk[:,None]<SPLITS)&(n[None,:]<I),0)
    g=tl.sum(g,axis=0).to(tl.bfloat16).to(tl.float32)
    u=tl.sum(u,axis=0).to(tl.bfloat16).to(tl.float32)
    tl.store(A+r*I+n,g*tl.sigmoid(g)*u,n<I)

@triton.jit
def _down_fp8(A,Ids,P,S,Y,H:tl.constexpr,I:tl.constexpr,E:tl.constexpr,G:tl.constexpr,BN:tl.constexpr):
    r=tl.program_id(1);n=tl.program_id(0)*BN+tl.arange(0,BN)
    e=tl.load(Ids+r);valid=(e>=0)&(e<E)
    k=tl.arange(0,triton.next_power_of_2(I))
    mask=valid&(n[:,None]<H)&(k[None,:]<I)
    b=tl.load(P+(e*H+n[:,None])*I+k[None,:],mask,0).to(tl.int32)
    mag=b&127
    # E4M3FN normal values via exponent rebias; preserve subnormals/NaNs.
    bits=(mag<<20)+(120<<23)
    v=bits.to(tl.float32,bitcast=True)
    v=tl.where(mag<8,mag.to(tl.float32)*0.001953125,v)
    v=tl.where(mag==127,float('nan'),v)
    v=tl.where((b&128)!=0,-v,v)
    s=tl.load(S+(e*tl.cdiv(H,128)+n[:,None]//128)*tl.cdiv(I,128)+k[None,:]//128,mask,0)
    w=_round_bf16_f32(v*s)
    a=tl.load(A+r*I+k,k<I,0).to(tl.float32)
    z=tl.sum(w*a[None,:],1)
    tl.store(Y+r*H+n,z,n<H)


def _run(x,ids,rw,gu,gs,down,ds,out,ws,group,fp8,shared=None,done=None):
    t,h,e,i,top=validate(x,ids,rw,gu,gs,down,ds,out,ws,group,fp8)
    # Contiguous uint8 views can still have an unaligned storage offset.
    tuned = (h == 6144 and i == 256 and t <= 128
             and gu.data_ptr() % 4 == 0 and (fp8 or down.data_ptr() % 2 == 0))
    if tuned:
        # Reuse partial until down overwrites it; split writes are disjoint.
        # Packed K-major reduction uses 6*2*I <= H scratch elements per route.
        splits = triton.cdiv(h,1024)
        _project_packed[(triton.cdiv(2*i,16),t*top,splits)](x,ids,gu,gs,ws.partial,2*i,h,e,top,group,16,1024,8,True,num_warps=4,enable_fp_fusion=True)
        _reduce_activate[(t*top,triton.cdiv(i,64))](ws.partial,ws.hidden,h,i,splits,triton.next_power_of_2(splits),64)
    else:
        _gate_up[(triton.cdiv(i,4),t*top)](x,ids,gu,gs,ws.hidden,h,i,e,top,group,4,512,num_warps=4,enable_fp_fusion=False)
    if tuned:
        if fp8:
            _down_fp8[(triton.cdiv(h,4),t*top)](ws.hidden,ids,down,ds,ws.partial,h,i,e,group,4,num_warps=1,enable_fp_fusion=True)
        else:
            _project_packed[(triton.cdiv(h,16),t*top)](ws.hidden,ids,down,ds,ws.partial,h,i,e,top,group,16,256,4,False,num_warps=1,enable_fp_fusion=True)
    else:
        _down[(triton.cdiv(h,4),t*top)](ws.hidden,ids,down,ds,ws.partial,h,i,e,group,fp8,4,triton.next_power_of_2(i),num_warps=4,enable_fp_fusion=False)
    if done is not None:
        torch.cuda.current_stream(x.device).wait_event(done)
    combine[(t,triton.cdiv(h,128))](ws.partial,ids,rw,out,t,h,top,e,128,triton.next_power_of_2(top),shared,FUSED=shared is not None)
    return out


def decode_int4(x,ids,rw,gu,gs,down,ds,out,ws,group=64):
    return _run(x,ids,rw,gu,gs,down,ds,out,ws,group,False)


def decode_int4_fp8(x,ids,rw,gu,gs,down,ds,out,ws,group=64):
    return _run(x,ids,rw,gu,gs,down,ds,out,ws,group,True)


def moe_decode(x,weights,out,workspace,down_dtype="int4",*,group=64,groups=8,
                 group_topk=4,routed_scale=2.5,tp_size=1,all_reduce=None):
    """Complete decode MoE layer; see ops.moe_layer for the ABI and TP contract."""
    from ops.moe_layer import run_layer
    return run_layer(_run,x,weights,out,workspace,down_dtype,group,groups,group_topk,
                     routed_scale,tp_size,all_reduce)
