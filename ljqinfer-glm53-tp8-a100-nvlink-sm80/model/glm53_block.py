"""Rank-local GLM53 Transformer block with explicit sparse binding, using the existing MLA ABI.
Attention and KV use FP16 without attention requantization; MoE uses BF16 activations and INT4/FP8.
The caller owns paged KV, collective scheduling and cross-layer scratch reuse.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _norm(X,W,Y,H:tl.constexpr,B:tl.constexpr):
    t=tl.program_id(0);i=tl.arange(0,B)
    x=tl.load(X+t*H+i,i<H,0).to(tl.float32)
    w=tl.load(W+i,i<H,0).to(tl.float32)
    z=x*tl.rsqrt(tl.sum(x*x,0)/H+1e-5)*w
    tl.store(Y+t*H+i,z,i<H)


@triton.jit(do_not_specialize=['N'], do_not_specialize_on_alignment=['N'])
def _residual(X,A,Y,N,B:tl.constexpr):
    i=tl.program_id(0)*B+tl.arange(0,B)
    x=tl.load(X+i,i<N,0).to(tl.float32)
    a=tl.load(A+i,i<N,0).to(tl.float32)
    tl.store(Y+i,x+a,i<N)


def add_residual(x,a):
    y=torch.empty_like(x)
    _residual[(triton.cdiv(x.numel(),1024),)](x,a,y,x.numel(),1024)
    return y
from ops.moe_layer import MoELayerWorkspace
from ops.moe_prefill import moe_prefill
from ops.moe_decode import moe_decode


class TransformerBlock:
    def __init__(self, attn, ffn_norm, moe, *, prefill_op, decode_op,
                 all_reduce, tp_size=8, down_dtype='int4', group=64, sparse=None):
        self.attn, self.ffn_norm, self.moe = attn, ffn_norm, moe
        self.sparse = sparse
        self.prefill_op, self.decode_op = prefill_op, decode_op
        self.all_reduce, self.tp_size = all_reduce, tp_size
        self.down_dtype, self.group = down_dtype, group
        if tp_size > 1 and all_reduce is None:
            raise ValueError('TP requires in-place all_reduce')

    def workspace(self, tokens):
        return MoELayerWorkspace.create(tokens, self.moe.router.shape[0],
            self.moe.router.shape[1], self.moe.gu.shape[1]//2,
            self.moe.shared_gu.shape[0]//2, self.ffn_norm.device)

    def attention(self, x, positions, kv, page_table, context, *, phase):
        a = self.attn
        if self.sparse is not None:
            partial = self.sparse(self.attn,x,positions,kv,page_table,context,phase=phase)
        elif phase == 'prefill':
            if not isinstance(context, int):
                raise TypeError('prefill context must be a host integer')
            _, partial = self.prefill_op.forward_rank_paged_inplace_half(
                x, positions, kv, page_table, context,
                *[a[k] for k in ['norm','q_a','q_a_norm','q_b','kv_a',
                                 'kv_a_norm','k_b','v_b','o']])
        elif phase == 'decode':
            _, partial = self.decode_op.forward_rank_paged_batch_half(
                x, positions, kv, [page_table], [context],
                *[a[k] for k in ['norm','q_a','q_a_norm','q_b','kv_a',
                                 'kv_a_norm','k_b','v_b','o']])
        else:
            raise ValueError(phase)
        if self.all_reduce is not None:
            self.all_reduce(partial)
        return (x + partial).half()

    def feed_forward(self, residual, workspace, out, *, phase):
        h = torch.empty_like(residual,dtype=torch.bfloat16)
        _norm[(h.shape[0],)](residual,self.ffn_norm,h,h.shape[1],triton.next_power_of_2(h.shape[1]),enable_fp_fusion=False)
        fn = moe_prefill if phase == 'prefill' else moe_decode
        fn(h, self.moe, out, workspace, self.down_dtype, group=self.group,
           groups=1, group_topk=1, routed_scale=2.5,
           tp_size=self.tp_size, all_reduce=self.all_reduce)
        return add_residual(residual,out)

    def prefill(self, x, positions, kv, page_table, context, workspace, out):
        h = self.attention(x, positions, kv, page_table, context, phase='prefill')
        return self.feed_forward(h, workspace, out, phase='prefill')

    def decode(self, x, positions, kv, page_table, context, workspace, out):
        h = self.attention(x, positions, kv, page_table, context, phase='decode')
        return self.feed_forward(h, workspace, out, phase='decode')


@triton.jit
def _cache_rows(X, POOL, TABLE, POS, D:tl.constexpr, PAGE:tl.constexpr,
                PAGES:tl.constexpr, PHYSICAL:tl.constexpr, B:tl.constexpr):
    t=tl.program_id(0);d=tl.arange(0,B)
    pos=tl.load(POS+t)
    valid=(pos>=0)&(pos<PAGES*PAGE)
    page=tl.load(TABLE+pos//PAGE,valid,other=0)
    value=tl.load(X+t*D+d,d<D,other=0)
    tl.store(POOL+(page*PAGE+pos%PAGE)*D+d,value,
             valid&(page>=0)&(page<PHYSICAL)&(d<D))


def _write_cache(rows,pool,table,positions):
    _cache_rows[(rows.shape[0],)](rows,pool,table,positions,rows.shape[1],
        pool.shape[1],table.numel(),pool.shape[0],triton.next_power_of_2(rows.shape[1]))


@triton.jit(do_not_specialize=['N', 'S'], do_not_specialize_on_alignment=['N', 'S'])
def _rms_apply(X,W,A,Y,H:tl.constexpr,S,N,B:tl.constexpr):
 j=tl.program_id(0)*B+tl.arange(0,B);row=j//H;col=j%H
 x=tl.load(X+row*S+col,j<N,0).to(tl.float32)
 a=tl.load(A+row,j<N,0)
 w=tl.load(W+col,j<N,0).to(tl.float32)
 z=(x*a)*w
 tl.store(Y+j,z,j<N)
def _rms_half(x,w):
 # Preserve reduction/rsqrt order; changing it perturbs downstream MoE routes.
 a=torch.rsqrt(x.float().square().mean(-1,keepdim=True)+1e-5)
 y=torch.empty(x.shape,dtype=torch.float16,device=x.device)
 _rms_apply[(triton.cdiv(x.numel(),1024),)](x,w,a,y,x.shape[1],x.stride(0),x.numel(),1024,enable_fp_fusion=False)
 return y


def _rope_interleaved(x,positions,inv_freq):
    angle=positions.double()[:,None]*inv_freq[None,:]
    shape=(x.shape[0],)+(1,)*(x.ndim-2)+(32,)
    c=angle.cos().float().reshape(shape);s=angle.sin().float().reshape(shape)
    e=x[...,0::2].float();o=x[...,1::2].float()
    return torch.stack((e*c-o*s,e*s+o*c),-1).flatten(-2).half()


class SparseAttentionBinding:
    """Fixed-shape TP8 single-sequence binding; caller owns the layer loop.

    Full indexers own a separate 128-wide cache. Shared layers reuse the preceding
    full binding's IDs, never its MLA KV. Use one binding per concurrent graph;
    run/capture producer then consumers each step, with identical metadata buffers.
    Warm up before capture. Device position/context values may change on replay.
    No multi-request scheduling or automatic dense fallback is provided here.
    """
    def __init__(self, *, tokens, capacity, parallel, index_weights=None,
                 index_pool=None, index_table=None, shared_from=None, topk=2048,
                 pair_group=None):
        from ops.sparse_topk_v2 import extension,workspace
        from ops.sparse_mla_cuda import extension as mla_extension
        if parallel.world!=8 or tokens<1 or not 0<topk<=capacity:
            raise ValueError('TP8, positive token count and topk<=capacity required')
        if shared_from is None:
            if index_weights is None or index_pool is None or index_table is None:
                raise ValueError('full indexer requires weights and explicit cache')
            device=index_pool.device
            if index_pool.dtype!=torch.float16 or index_pool.ndim!=3 or index_pool.shape[-1]!=128:
                raise ValueError('index cache must be FP16 [pages,page,128]')
            if capacity>index_table.numel()*index_pool.shape[1]:
                raise ValueError('index capacity exceeds page table')
        else:
            if any(v is not None for v in (index_weights,index_pool,index_table)):
                raise ValueError('shared layer must not create another indexer')
            if (tokens,capacity,topk)!=(shared_from.tokens,shared_from.capacity,shared_from.topk):
                raise ValueError('shared selection shape mismatch')
            device=shared_from.ids.device
        self.tokens,self.capacity,self.topk=tokens,capacity,topk
        self.parallel,self.index_weights=parallel,index_weights
        self.index_pool,self.index_table=index_pool,index_table
        self.shared_from=shared_from
        self.ids=(torch.empty((tokens,topk),device=device,dtype=torch.int64)
                  if shared_from is None else shared_from.ids)
        self.inv_freq=8000000.**(-torch.arange(0,64,2,device=device,dtype=torch.float64)/64)
        self.topk_workspace=(workspace(min(triton.cdiv(tokens,8),256),capacity,device=device)
                             if shared_from is None else None)
        self.metadata=None
        self.prefill_elementwise = None
        extension();mla_extension()
        self.projected_pair=None
        self.pair=None
        if pair_group is not None:
            from ops.sparse_mla_pair import PairWorkspace
            q=torch.empty((tokens,8,576),device=device,dtype=torch.float16)
            self.pair=PairWorkspace(q,self.ids,pair_group)

    def __call__(self,a,x,positions,kv,page_table,context,*,phase):
        from ops.sparse_index_decode import select_decode
        from ops.sparse_index_query import select_prefill
        from ops.sparse_mla_cuda import forward
        if phase not in ('decode','prefill'):raise ValueError(phase)
        if x.shape[0]!=self.tokens or x.dtype!=torch.float16 or not x.is_contiguous():
            raise ValueError('binding requires fixed token count and contiguous FP16 input')
        if not isinstance(context,torch.Tensor) or context.dtype!=torch.int64 or context.numel()!=1:
            raise TypeError('sparse context must be device int64 scalar for both phases')
        if context.device!=x.device or positions.device!=x.device or positions.dtype!=torch.int64:
            raise ValueError('metadata device/dtype mismatch')
        if positions.shape!=(self.tokens,) or not positions.is_contiguous():
            raise ValueError('positions must be contiguous [tokens]')
        if kv.dtype!=torch.float16 or kv.ndim!=3 or kv.shape[-1]!=576 or not kv.is_contiguous():
            raise ValueError('MLA cache must be contiguous FP16 [pages,page,576]')
        if page_table.dtype!=torch.int64 or not page_table.is_contiguous() or page_table.device!=x.device or kv.device!=x.device:
            raise ValueError('invalid MLA page table/device')
        if self.capacity>page_table.numel()*kv.shape[1]:
            raise ValueError('MLA capacity exceeds page table')
        rms, rope = _rms_half, _rope_interleaved
        if phase in ('prefill','decode'):
            if self.prefill_elementwise is None:
                from ops.prefill_elementwise import ElementwiseFusion
                self.prefill_elementwise=ElementwiseFusion(_rms_apply,self.tokens,x.device)
            self.prefill_elementwise.begin(positions,self.inv_freq)
            rms, rope = self.prefill_elementwise.rms, self.prefill_elementwise.rope
        projection=getattr(self,'token_projection',None)
        use_projection=(phase=='prefill' and self.tokens==12288
                        and self.prefill_elementwise.first_chunk and projection is not None)
        local_input=use_projection and self.shared_from is not None
        xn=x if local_input else rms(x,a['norm'])
        qa=(projection.begin(xn,a,normalize_input=local_input) if use_projection
            else rms(xn@a['q_a'].t(),a['q_a_norm']))
        qb=(qa@a['q_b'].t()).view(self.tokens,8,256)
        ql=torch.bmm(qb[...,:192].transpose(0,1),a['k_b'].transpose(1,2)).transpose(0,1)
        if phase in ('prefill','decode'):
            q=self.prefill_elementwise.query(ql,qb[...,192:])
        else:
            qr=rope(qb[...,192:],positions,self.inv_freq)
            q=torch.cat((ql,qr),-1).contiguous()
        raw=projection.finish() if use_projection else xn@a['kv_a'].t()
        if phase in ('prefill','decode'):
            self.prefill_elementwise.cache(rms(raw[:,:512],a['kv_a_norm']),raw[:,512:],kv,page_table,positions)
        else:
            rows=torch.cat((rms(raw[:,:512],a['kv_a_norm']),
                            rope(raw[:,512:],positions,self.inv_freq)),-1)
            _write_cache(rows,kv,page_table,positions)
        metadata=(positions.data_ptr(),context.data_ptr(),phase)
        if self.shared_from is None:
            w=self.index_weights
            iq=(qa@w['wq_b'].t()).view(self.tokens,4,128)
            iq=(self.prefill_elementwise.index_query(iq) if phase in ('prefill','decode') else
                torch.cat((rope(iq[...,:64],positions,self.inv_freq),iq[...,64:]),-1))
            ik=torch.nn.functional.layer_norm(xn@w['wk'].t(),(128,),w['k_norm'],w['k_bias'],1e-6)
            if phase in ('prefill','decode'):
                self.prefill_elementwise.index_cache(ik,self.index_pool,self.index_table,positions)
            else:
                ik=torch.cat((rope(ik[:,:64],positions,self.inv_freq),ik[:,64:]),-1)
                _write_cache(ik,self.index_pool,self.index_table,positions)
            weights=(xn.float()@w['weights_proj'].t())*(32**-0.5)
            kw=dict(parallel=self.parallel,logical_capacity=self.capacity)
            kw['scratch']=getattr(self,'index_scratch',None)
            if phase=='decode':kw['topk_workspace']=self.topk_workspace
            (select_decode if phase=='decode' else select_prefill)(
                iq,weights,self.index_pool,self.index_table,positions,context,self.ids,**kw)
        elif self.shared_from.metadata!=metadata:
            raise RuntimeError('shared index producer must run first with identical step metadata')
        self.metadata=metadata
        if (phase=='prefill' and self.projected_pair is not None and
                self.tokens==12288 and self.prefill_elementwise.first_chunk):
            heads=self.projected_pair(q,kv,page_table,self.ids,positions,context)
            return heads.reshape(self.tokens,2048)@a['o'].t()
        latent=getattr(self,'latent',None)
        if latent is None:
            latent=torch.empty((self.tokens,8,512),device=x.device,dtype=x.dtype)
        if phase=='prefill' and self.pair is not None:
            self.pair(q,kv,page_table,self.ids,positions,context,latent,first_chunk=self.prefill_elementwise.first_chunk)
        else:
            forward(q,kv,page_table,self.ids,positions,context,latent)
        heads=torch.bmm(latent.transpose(0,1),a['v_b'].transpose(1,2)).transpose(0,1)
        return heads.reshape(self.tokens,2048)@a['o'].t()
