"""Complete rank-local MoE ABI: router + shared side stream + routed experts.

Two entry points live in moe_prefill / moe_decode. Inputs are post-FFN-norm
BF16 [T,H]; output is FP32 [T,H]. Norm and residual belong to the block.
Only routed down changes with down_dtype ('int4' or 'fp8'). Router is FP32;
shared weights are BF16, prepared once outside capture (from official FP8
with prepare_shared_fp8). No new quantization of the shared expert.

Grouped routing: sigmoid; bias affects selection only; top-2 sum per group;
keep 4 of 8 groups and 8 experts; unbiased weights sum to 2.5. Ties follow
legacy decode: group score descending / group id ascending, then candidate
position within that group ordering. Nonfinite selection scores fail closed
for that token (IDs=-1, weights=0); shared output is not sanitized.

Workspace/output/weights must remain alive until GPU execution finishes.
Workspace is single-flight and may be reused by sequential layers. Create,
prepare and warm each shape/precision before CUDA Graph capture. Precision
is a host launch parameter, fixed within a captured graph. An optional
in-place all_reduce callback runs exactly once AFTER branch join/add; for
TP>1 it is mandatory and must enqueue on the current stream (graph-safe
when capturing). Without it, the result is explicitly rank-local.
"""
from dataclasses import dataclass
from typing import Callable
import math
import torch
import triton
import triton.language as tl
import triton.language.extra.cuda.libdevice as lib
from ops.moe_common import Workspace, validate, DENSE_PREFILL_MIN_TOKENS


@dataclass(frozen=True)
class MoELayerWeights:
    router: torch.Tensor              # FP32 [E,H], replicated
    bias: torch.Tensor                # FP32 [E], selection correction
    gu: torch.Tensor                  # routed INT4 [E,2*I,H/2]
    gu_scale: torch.Tensor
    down: torch.Tensor                # routed INT4 or raw FP8
    down_scale: torch.Tensor
    shared_gu: torch.Tensor           # BF16 [2*S,H], gate then up
    shared_down: torch.Tensor         # BF16 [H,S], rank-local TP shard


def prepare_shared_fp8(packed, scale):
    """Startup only: official 128x128 E4M3FN block scales -> BF16 matrix.

    This explicit materialization costs 6*H*S bytes per layer for gate/up/down
    combined (9 MiB for H=6144,S=256). Router preparation is likewise explicit:
    router.float().contiguous(). No hidden weight caches in the call path.
    """
    if packed.ndim != 2 or packed.dtype not in (torch.uint8, torch.float8_e4m3fn):
        raise ValueError('expected raw uint8 or E4M3FN matrix')
    n,k=packed.shape
    if scale.shape != (triton.cdiv(n,128),triton.cdiv(k,128)) or scale.dtype != torch.float32:
        raise ValueError('expected FP32 128x128 block scales')
    if not packed.is_cuda or scale.device != packed.device:
        raise ValueError('expected tensors on the same CUDA device')
    with torch.no_grad():
        f=packed.view(torch.float8_e4m3fn).float()
        return (f*scale.repeat_interleave(128,0).repeat_interleave(128,1)[:n,:k]).bfloat16().contiguous()


@triton.jit
def _route(L,B,Ids,W,E:tl.constexpr,NG:tl.constexpr,GS:tl.constexpr,KEEP:tl.constexpr,TOP:tl.constexpr,SCALE:tl.constexpr,PROB:tl.constexpr=False):
    t=tl.program_id(0)
    gg=tl.arange(0,NG);jj=tl.arange(0,GS)
    idx=gg[:,None]*GS+jj[None,:]
    v=tl.load(L+t*E+idx);b=tl.load(B+idx)
    # Match the FP32 torch sigmoid: no exp2 approximation or approximate divide.
    prob=v if PROB else lib.div_rn(1.,1.+lib.exp(-v))
    sel=prob+b
    bad=tl.sum(tl.sum(((sel!=sel)|(tl.abs(sel)==float('inf'))).to(tl.int32),1),0)>0
    sel=tl.where((sel==sel)&(tl.abs(sel)!=float('inf')),sel,-float('inf'))
    first=tl.max(sel,1)
    fi=tl.min(tl.where(sel==first[:,None],jj[None,:],GS),1)
    second=tl.max(tl.where(jj[None,:]!=fi[:,None],sel,-float('inf')),1)
    score=first+second
    rank=tl.sum(((score[None,:]>score[:,None])|((score[None,:]==score[:,None])&(gg[None,:]<gg[:,None]))).to(tl.int32),1)
    cand=tl.where(rank[:,None]<KEEP,sel,-float('inf'))
    order=rank[:,None]*GS+jj[None,:]
    # Positive/negative IEEE floats -> increasing uint32 keys; zero sign ignored.
    bits=tl.where(cand==0,0.,cand).to(tl.uint32,bitcast=True)
    ordered=tl.where((bits&0x80000000)!=0,~bits,bits^0x80000000)
    # Low key encodes group-ranked tie order, then expert id.
    key=(ordered.to(tl.uint64)<<32)|((E-1-order)*E+idx).to(tl.uint64)
    key=tl.sort(tl.reshape(key,(E,)),descending=True)
    expert=(key%E).to(tl.int32)
    lane=tl.arange(0,E)
    selected=lane<TOP
    pr=tl.gather(tl.reshape(prob,(E,)),expert,0)
    total=tl.full((),0,tl.float32)
    for q in range(TOP):total+=tl.sum(tl.where(lane==q,pr,0.),0)
    inv=tl.where((total>6.103515625e-5)&(~bad),SCALE/total,0.)
    tl.store(Ids+t*TOP+lane,tl.where(bad,-1,expert),selected)
    tl.store(W+t*TOP+lane,tl.where(bad,0.,pr*inv),selected)


@triton.jit(do_not_specialize=['N'], do_not_specialize_on_alignment=['N'])
def _router_cast(X,Y,N,B:tl.constexpr):
    q=tl.program_id(0)*B+tl.arange(0,B)
    tl.store(Y+q,tl.load(X+q,q<N,0).to(tl.float32),q<N)


@triton.jit(do_not_specialize=['T'], do_not_specialize_on_alignment=['T'])
def _shared_act(G,A,T,S:tl.constexpr,B:tl.constexpr):
    q=tl.program_id(0)*B+tl.arange(0,B); row=q//S; col=q%S
    # Match routed GU's BF16 projection boundary before SiLU.
    g=tl.load(G+row*2*S+col,q<T*S,0).to(tl.bfloat16).to(tl.float32)
    u=tl.load(G+row*2*S+S+col,q<T*S,0).to(tl.bfloat16).to(tl.float32)
    tl.store(A+q,g*tl.sigmoid(g)*u,q<T*S)


@dataclass
class MoELayerWorkspace:
    routed: Workspace
    logits: torch.Tensor
    router_x: torch.Tensor
    ids: torch.Tensor
    route_weights: torch.Tensor
    shared_gu: torch.Tensor
    shared_act: torch.Tensor
    shared_out: torch.Tensor
    aux_stream: torch.cuda.Stream
    fork: torch.cuda.Event
    done: torch.cuda.Event

    @classmethod
    def create(cls,tokens,experts,hidden,intermediate,shared_intermediate,device,topk=8):
        if min(tokens,experts,hidden,intermediate,shared_intermediate,topk)<=0:
            raise ValueError('dimensions must be positive')
        device=torch.device(device)
        with torch.cuda.device(device):
            def alloc(shape,dtype=torch.float32):return torch.empty(shape,device=device,dtype=dtype)
            stream=torch.cuda.Stream(device=device)
            fork=torch.cuda.Event();done=torch.cuda.Event()
            fork.record();done.record()  # Instantiate event handles before capture.
            return cls(Workspace.create(tokens,topk,experts,hidden,intermediate,device),
                       alloc((tokens,experts)),alloc((tokens,hidden)),
                       alloc((tokens,topk),torch.int32),alloc((tokens,topk)),
                       alloc((tokens,2*shared_intermediate)),
                       alloc((tokens,shared_intermediate),torch.bfloat16),
                       alloc((tokens,hidden)),stream,fork,done)


def _validate_layer(x,w,out,ws,group,down_dtype,groups,keep,scale,tp_size,all_reduce):
    if down_dtype not in ('int4','fp8'):
        raise ValueError("down_dtype must be 'int4' or 'fp8'")
    t,h,e,i,top=validate(x,ws.ids,ws.route_weights,w.gu,w.gu_scale,w.down,w.down_scale,
                       out,ws.routed,group,down_dtype=='fp8')
    if not isinstance(groups,int) or groups<=0 or groups&(groups-1) or e%groups:
        raise ValueError('groups must be a positive power of two dividing E')
    gs=e//groups
    if gs<2 or gs&(gs-1) or not 1<=keep<=groups or top>keep*gs:
        raise ValueError('invalid grouped top-k dimensions')
    if not math.isfinite(scale) or scale<=0 or not isinstance(tp_size,int) or tp_size<1:
        raise ValueError('invalid scale/TP size')
    if (all_reduce is not None and not callable(all_reduce)) or (tp_size>1 and all_reduce is None):
        raise ValueError('TP>1 requires an explicit in-place all_reduce callback')
    if w.shared_down.ndim!=2:raise ValueError('shared_down must be a matrix')
    s=w.shared_down.shape[1]
    if s<=0:raise ValueError('empty shared expert')
    specs=[(w.router,(e,h),torch.float32),(w.bias,(e,),torch.float32),
           (w.shared_gu,(2*s,h),torch.bfloat16),(w.shared_down,(h,s),torch.bfloat16),
           (ws.logits,(t,e),torch.float32),(ws.router_x,(t,h),torch.float32),
           (ws.shared_gu,(t,2*s),torch.float32),(ws.shared_act,(t,s),torch.bfloat16),
           (ws.shared_out,(t,h),torch.float32)]
    for a,shape,dtype in specs:
        if a.shape!=shape or a.dtype!=dtype or a.device!=x.device or not a.is_cuda or not a.is_contiguous():
            raise ValueError(f'expected {shape} {dtype} contiguous on {x.device}')
    buffers=[x,out,w.gu,w.gu_scale,w.down,w.down_scale,ws.ids,ws.route_weights,
             ws.routed.hidden,ws.routed.partial,ws.routed.slots,ws.routed.counts]
    if ws.routed.weight is not None:buffers.append(ws.routed.weight)
    buffers.extend(a for a,_,_ in specs)
    if len({a.untyped_storage().data_ptr() for a in buffers})!=len(buffers):
        raise ValueError('layer tensors may not share storage')
    if ws.aux_stream.device!=x.device:
        raise ValueError('side stream is on a different device')
    if torch.cuda.current_stream(x.device)==ws.aux_stream:
        raise ValueError('root and shared side stream must differ')
    # Routing must not silently inherit an application-wide TF32 precision loss.
    if torch.backends.cuda.matmul.allow_tf32:
        raise ValueError('set torch.backends.cuda.matmul.allow_tf32=False before warm/capture')
    return t,h,e,s,top


@torch.no_grad()
def run_layer(expert_fn,x,w,out,ws,down_dtype,group,groups,keep,scale,tp_size,all_reduce,prepare_gu=False):
    """Internal shared orchestration; public phase entry points are separate."""
    t,h,e,s,top=_validate_layer(x,w,out,ws,group,down_dtype,groups,keep,scale,tp_size,all_reduce)
    i=w.gu.shape[1]//2
    prepare_gu=prepare_gu and t>=DENSE_PREFILL_MIN_TOKENS
    if prepare_gu and ws.routed.weight is None:
        raise ValueError("large prefill requires weight scratch")
    with torch.cuda.device(x.device):
        # Preserve Q8 GEMM accumulation; routed kernels/TP stay batched.
        gemm_rows=getattr(ws,'gemm_rows',t)
        if gemm_rows<=0 or t%gemm_rows:raise ValueError('invalid GEMM row group')
        root=torch.cuda.current_stream(x.device)
        ws.fork.record(root);ws.aux_stream.wait_event(ws.fork)
        with torch.cuda.stream(ws.aux_stream):
            for q in range(0,t,gemm_rows):
                torch.mm(x[q:q+gemm_rows],w.shared_gu.t(),out=ws.shared_gu[q:q+gemm_rows],out_dtype=torch.float32)
            _shared_act[(triton.cdiv(t*s,1024),)](ws.shared_gu,ws.shared_act,t,s,1024)
            for q in range(0,t,gemm_rows):
                torch.mm(ws.shared_act[q:q+gemm_rows],w.shared_down.t(),out=ws.shared_out[q:q+gemm_rows],out_dtype=torch.float32)
            if prepare_gu:
                # Independent of routing. Rebuild every replay; no weight cache.
                from ops.moe_prefill import unpack
                unpack[(triton.cdiv(2*i*h,2048),e)](w.gu,w.gu_scale,ws.routed.weight,e,2*i,h,group,False,2048,8 if w.gu.data_ptr()%4==0 else 2)
            ws.done.record(ws.aux_stream)
        _router_cast[(triton.cdiv(t*h,1024),)](x,ws.router_x,t*h,1024,num_warps=4)
        # Keep sigmoid rounding identical to the independent PyTorch routing ABI.
        # Approximate exp2 sigmoid can swap almost-tied experts at top-k boundaries.
        for q in range(0,t,gemm_rows):
            torch.mm(ws.router_x[q:q+gemm_rows],w.router.t(),out=ws.logits[q:q+gemm_rows])
        _route[(t,)](ws.logits,w.bias,ws.ids,ws.route_weights,e,groups,e//groups,keep,top,scale,
                     num_warps=1,enable_fp_fusion=False)
        # Prepared GU must be ready before its GEMM; other paths join at combine.
        # Shared output is added after the rounded routed sum.
        if prepare_gu:
            root.wait_event(ws.done)
            expert_fn(x,ws.ids,ws.route_weights,w.gu,w.gu_scale,w.down,w.down_scale,
                      out,ws.routed,group,down_dtype=="fp8",ws.shared_out,ws.done,preunpacked=True)
        else:
            expert_fn(x,ws.ids,ws.route_weights,w.gu,w.gu_scale,w.down,w.down_scale,
                      out,ws.routed,group,down_dtype=="fp8",ws.shared_out,ws.done)
        if all_reduce is not None:all_reduce(out)
    return out
