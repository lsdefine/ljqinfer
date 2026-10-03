"""GLM53 rank-per-process TP8 engine: paged sparse prefill + fixed Q8 verify.
B1 execution contract. Call all ranks identically. No legacy native MTP path.
Verification writes scratch suffix KV; commit publishes only the consumed prefix.
DFlash features are post-layer, pre-final-norm at TARGET_LAYERS. The caller must
project/append committed features to mtp_kv, never tentative verify features.
"""
from dataclasses import dataclass
from math import ceil
from copy import copy
from contextlib import nullcontext
from model.config import DEFAULT_PREFILL_CHUNK_TOKENS
from ops.sparse_mla_pair import create_pair_group
import torch
import torch.distributed as dist
import triton
from model import weights as W
from model.glm53_block import TransformerBlock, SparseAttentionBinding, _norm, add_residual
from model.glm53_mtp_pool import MTPKVPool, VERIFY_Q, TARGET_LAYERS
from ops.sparse_index_tp import TPIndexParallel
from ops.moe_layer import MoELayerWorkspace


@dataclass
class Result:
    start: int
    logits: torch.Tensor
    features: tuple


class Engine:
    q = VERIFY_Q
    target_layer_ids = TARGET_LAYERS

    def __init__(self, weights, *, capacity=8192, prefill_chunk_tokens=DEFAULT_PREFILL_CHUNK_TOKENS):
        if not dist.is_initialized() or dist.get_world_size()!=8:
            raise RuntimeError('launch with torchrun --nproc-per-node=8')
        if capacity<2048 or capacity%64 or prefill_chunk_tokens<1:
            raise ValueError('capacity must be >=2048 and a multiple of 64')
        self.w=weights;self.rank=dist.get_rank();self.device=weights.embed.device
        if weights.rank!=self.rank:raise ValueError('rank/weights mismatch')
        self.capacity=capacity;self.prefill_chunk_tokens=prefill_chunk_tokens
        self.parallel=TPIndexParallel();self.length=0;self.pending=None
        self.pair_group=create_pair_group()
        self.decode_capacity=2048
        self.table=torch.arange(capacity//64,device=self.device,dtype=torch.int64)
        self.kv=torch.empty((78,capacity//64,64,576),device=self.device,dtype=torch.float16)
        self.index_layers=[l.idx for l in weights.layers if l.indexer is not None]
        self.index=torch.empty((len(self.index_layers),capacity//64,64,128),device=self.device,dtype=torch.float16)
        self.index_slot={l:i for i,l in enumerate(self.index_layers)}
        self.mtp_kv=MTPKVPool(max_tokens=capacity,max_sequence_tokens=capacity,device=self.device)
        self.plans={};self.graph=None;self.graph_result=None
        self.ids=torch.zeros(VERIFY_Q,device=self.device,dtype=torch.int64)
        self.positions=torch.arange(VERIFY_Q,device=self.device,dtype=torch.int64)
        self.context=torch.tensor(VERIFY_Q,device=self.device,dtype=torch.int64)
        from .glm53_workspace import Workspaces
        self.workspaces=Workspaces(self)

    @classmethod
    def load(cls, *, rank=None, capacity=8192, prefill_chunk_tokens=DEFAULT_PREFILL_CHUNK_TOKENS):
        rank=dist.get_rank() if rank is None else rank
        torch.cuda.set_device(rank);torch.set_num_threads(2)
        torch.backends.cuda.matmul.allow_tf32=False
        return cls(W.load_tp8(rank=rank),capacity=capacity,prefill_chunk_tokens=prefill_chunk_tokens)

    def _plan(self,tokens,phase='decode'):
        return self.workspaces.plan(tokens,phase)

    def _allocate_plan(self,tokens,phase='decode'):
        capacity=(self.decode_capacity if phase=='decode' else
                  max(2048, min(self.capacity, ceil((self.length+tokens)/64)*64)))
        blocks=[];last=None;template=None
        for layer in self.w.layers:
            if template is None:
                if layer.indexer is None:raise ValueError('first indexer must be a producer')
                b=SparseAttentionBinding(tokens=tokens,capacity=capacity,parallel=self.parallel,
                    index_weights=layer.indexer,index_pool=self.index[self.index_slot[layer.idx]],
                    index_table=self.table,pair_group=self.pair_group if phase=='prefill' else None)
                if phase in ('prefill','decode'):
                    from ops.prefill_elementwise import ElementwiseFusion
                    from model.glm53_block import _rms_apply
                    b.prefill_elementwise=ElementwiseFusion(_rms_apply,tokens,self.device)
                    if phase=='prefill' and tokens==12288:
                        from ops.prefill_token_projection import TokenProjection
                        b.token_projection=TokenProjection(self.device,self.pair_group)
                template=b
            else:
                b=copy(template)
                b.metadata=None
                b.shared_from=last if layer.indexer is None else None
                b.index_weights=layer.indexer
                b.index_pool=None if layer.indexer is None else self.index[self.index_slot[layer.idx]]
            if phase=='prefill' and tokens==12288:
                from ops.sparse_mla_pair import ProjectedPair
                # Sequential layers share pair output buffers, not their V weights.
                previous=blocks[0].sparse.projected_pair if blocks else None
                buffers=(previous.send,previous.recv) if previous is not None else None
                b.projected_pair=ProjectedPair(b.pair,layer.attn['v_b'],buffers)
            if layer.indexer is not None:last=b
            blocks.append(TransformerBlock(layer.attn,layer.ffn_norm,layer.ffn,
                prefill_op=None,decode_op=None,all_reduce=self.parallel,sparse=b))
        w=self.w.layers[3].ffn
        ws=MoELayerWorkspace.create(tokens,w.router.shape[0],W.D,w.gu.shape[1]//2,w.shared_gu.shape[0]//2,self.device)
        out=torch.empty((tokens,W.D),device=self.device,dtype=torch.float32)
        plan=(blocks,ws,out)
        return plan

    def prepare_decode(self,max_end):
        # Bound graph scans by request length, not the reserved KV allocation.
        capacity=min(self.capacity,max(2048,1 << (max_end-1).bit_length()))
        if self.graph is not None and capacity!=self.decode_capacity:
            torch.cuda.synchronize()
            self.graph.reset();self.graph=None;self.graph_result=None
            self.plans.pop('decode',None)
        self.decode_capacity=capacity

    @torch.no_grad()
    def _forward(self,ids,positions,context,phase,last_logits_only=False):
        blocks,ws,out=self._plan(ids.numel(),phase)
        valid=(ids>=self.w.vocab_start)&(ids<self.w.vocab_end)
        local=(ids-self.w.vocab_start).clamp(0,self.w.vocab_end-self.w.vocab_start-1)
        x=self.w.embed[local].half()*valid[:,None]
        self.parallel(x)
        features=[]
        fusion=blocks[0].sparse.prefill_elementwise
        scope=(fusion.chunk(positions,blocks[0].sparse.inv_freq,first_chunk=self.length==0)
               if fusion is not None else nullcontext())
        with scope:
            for layer,block in zip(self.w.layers,blocks):
                residual=block.attention(x,positions,self.kv[layer.idx],self.table,context,phase=phase)
                if layer.idx<3:
                    h=torch.empty_like(residual,dtype=torch.bfloat16)
                    _norm[(len(ids),)](residual,layer.ffn_norm,h,W.D,triton.next_power_of_2(W.D),enable_fp_fusion=False)
                    gu=h@layer.ffn.gu.T
                    gate,up=gu.chunk(2,-1)
                    act=(torch.nn.functional.silu(gate.float())*up.float()).bfloat16()
                    partial=act@layer.ffn.down.T
                    self.parallel(partial)
                    x=add_residual(residual,partial)
                else:
                    x=block.feed_forward(residual,ws,out,phase=phase)
                if layer.idx in TARGET_LAYERS:features.append(x)
        if last_logits_only:x=x[-1:]
        rows=len(x)
        normalized=torch.empty_like(x,dtype=self.w.lm_head.dtype)
        _norm[(rows,)](x,self.w.final_norm,normalized,W.D,triton.next_power_of_2(W.D),enable_fp_fusion=False)
        local_logits=normalized@self.w.lm_head.T
        # all_gather_into_tensor concatenates ranks along dimension 0.
        gathered=torch.empty((8*rows,local_logits.shape[1]),device=self.device,dtype=local_logits.dtype)
        dist.all_gather_into_tensor(gathered,local_logits)
        logits=gathered.view(8,rows,-1).permute(1,0,2).reshape(rows,W.VOCAB).float()
        return logits,tuple(features)

    def _validate(self,ids):
        ids=torch.as_tensor(ids,device=self.device,dtype=torch.int64)
        if ids.ndim!=1 or ids.numel()<1:raise ValueError('one nonempty sequence required')
        if bool(((ids<0)|(ids>=W.VOCAB)).any()):raise ValueError('token ID out of vocabulary')
        if self.pending is not None:raise RuntimeError('commit or abort outstanding verify first')
        if self.length+ids.numel()>self.capacity:raise ValueError('sequence capacity exhausted')
        return ids

    def prefill(self,ids,*,on_commit=None,last_logits_only=False):
        ids=self._validate(ids);last=None
        for offset in range(0,len(ids),self.prefill_chunk_tokens):
            part=ids[offset:offset+self.prefill_chunk_tokens];start=self.length
            pos=torch.arange(start,start+len(part),device=self.device,dtype=torch.int64)
            self.context.fill_(start+len(part))
            logits,features=self._forward(part,pos,self.context,'prefill',last_logits_only=last_logits_only)
            last=Result(start,logits,features)
            if on_commit is not None:on_commit(last)
            self.length+=len(part)
        return last

    def verify(self,ids):
        ids=self._validate(ids)
        if len(ids)!=VERIFY_Q:raise ValueError('decode contract is Q=8, anchor plus seven drafts')
        start=self.length
        if start+VERIFY_Q>self.decode_capacity:
            had_graph=self.graph is not None
            self.prepare_decode(start+VERIFY_Q)
            if had_graph:self.capture_verify()
        self.ids.copy_(ids);self.positions.copy_(torch.arange(start,start+VERIFY_Q,device=self.device))
        self.context.fill_(start+VERIFY_Q)
        if self.graph is None:
            logits,features=self._forward(self.ids,self.positions,self.context,'decode')
        else:
            self.graph.replay();logits,features=self.graph_result
        self.pending=Result(start,logits,features)
        return self.pending

    def commit(self,consumed,*,on_commit=None):
        if self.pending is None:raise RuntimeError('no pending verify')
        if not isinstance(consumed,int) or not 0<=consumed<=VERIFY_Q:raise ValueError('consumed must be 0..8')
        p=self.pending
        result=Result(p.start,p.logits[:consumed],tuple(t[:consumed] for t in p.features))
        if consumed and on_commit is not None:on_commit(result)
        self.length=p.start+consumed;self.pending=None
        return result

    def reset(self):
        self.length=0;self.pending=None;self.mtp_kv.lengths[0]=0

    def capture_verify(self):
        if self.pending is not None or self.graph is not None:raise RuntimeError('capture requires no pending verify and no graph')
        if self.length+VERIFY_Q>self.capacity:raise ValueError('no space for capture suffix')
        self.positions.copy_(torch.arange(self.length,self.length+VERIFY_Q,device=self.device))
        self.context.fill_(self.length+VERIFY_Q)
        self.prepare_decode(max(self.decode_capacity,self.length+VERIFY_Q))
        self._plan(VERIFY_Q)
        # A high-priority root cannot alias MoE normal-priority side streams.
        # CUDA stream pools wrap; the implicit process-global capture stream
        # can alias a side stream after repeated batch workspace allocation.
        stream=torch.cuda.Stream(device=self.device,priority=-1);stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):self._forward(self.ids,self.positions,self.context,'decode')
        torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize()
        self.graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph,stream=stream):
            self.graph_result=self._forward(self.ids,self.positions,self.context,'decode')
