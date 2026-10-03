"""GLM53 fixed-cohort Q8 batching, adapted from batch_decode's row contract.
Shared weights and lifetime KV allocation; independent request views and commits.
Target MoE/dense FFN, embedding and output projection execute on B*8 rows.
Attention projections/TP and DFlash execute jointly; KV selection is row-local.
No mid-cohort admission. Completed/cancelled rows leave at committed boundaries.
"""
from model.glm53_joint_attention import attention_joint
from model.glm53_joint_draft import JointDraft
from copy import copy
from contextlib import ExitStack
import time
import torch
import torch.distributed as dist
import triton
from model import weights as W
from model.glm53_engine import Result
from model.glm53_generate import Generation, accepted_prefix
from model.glm53_workspace import Workspaces, IndexBuffers
from model.glm53_block import _norm, add_residual
from model.glm53_mtp_pool import TARGET_LAYERS
from ops.moe_layer import MoELayerWorkspace
from ops.sparse_topk_v2 import workspace as topk_workspace


def _slot(base, draft, page_start, pages):
    """Slice preallocated pools, never clone model weights or resident KV."""
    e=copy(base);e.capacity=pages*64;e.length=0;e.pending=None
    e.decode_capacity=2048;e.graph=None;e.graph_result=None;e.plans={}
    e.table=torch.arange(pages,device=e.device,dtype=torch.int64)
    e.cache_page_span=(0,pages)
    e.kv=base.kv[:,page_start:page_start+pages]
    e.index=base.index[:,page_start:page_start+pages]
    e.ids=torch.zeros(8,device=e.device,dtype=torch.int64)
    e.positions=torch.arange(8,device=e.device,dtype=torch.int64)
    e.context=torch.zeros((),device=e.device,dtype=torch.int64)
    pool=copy(base.mtp_kv);pool.max_tokens=pool.max_sequence_tokens=e.capacity
    pool.num_pages=pool.logical_pages=pages;pool.lengths=[0];pool.free_pages=[]
    # Generator reserves every base MTP page. Its physical map need not be identity.
    # Use global pool storage with this slot's disjoint existing mapping.
    pool.page_table=base.mtp_kv.page_table[:,page_start:page_start+pages].clone()
    pool.host_page_table=[base.mtp_kv.host_page_table[0][page_start:page_start+pages].copy()]
    e.mtp_kv=pool
    ws=Workspaces.__new__(Workspaces);ws.engine=e;ws.views={}
    owner_blocks,owner_ws,owner_out=base.workspaces.owners['prefill']
    blocks=[];previous=None
    for owner in owner_blocks:
        block=copy(owner);b=copy(owner.sparse);b.metadata=None
        b.index_table=e.table
        b.index_pool=None if b.index_weights is None else e.index[e.index_slot[len(blocks)]]
        b.shared_from=previous if b.index_weights is None else None
        if b.index_weights is not None:previous=b
        block.sparse=b;blocks.append(block)
    ws.owners={'prefill':(blocks,owner_ws,owner_out),'decode':e._allocate_plan(8,'decode')}
    ws.topk=topk_workspace(1,e.capacity,device=e.device)
    ws.index={'prefill':base.workspaces.index['prefill'],
              'decode':IndexBuffers(8,e.capacity,'decode',e.device)}
    ws.latent={'prefill':base.workspaces.latent['prefill'],
               'decode':torch.empty((8,8,512),dtype=torch.float16,device=e.device)}
    e.workspaces=ws
    # Stream-local scratch shared across sequential layers; never allocate per layer.
    # Each fixed row owns its buffers, including across resident B1..B4 graph replay.
    from ops.sparse_ops import make_mla_workspace
    qshape=torch.empty((8,8,576),device=e.device,dtype=torch.float16)
    e.decode_mla_workspace=make_mla_workspace(qshape,splits=32)
    d=copy(draft);d.engine=e;d.pool=pool;d.graph=None;d.append_graphs=None
    d.anchor=torch.zeros_like(draft.anchor);d.context=torch.zeros_like(draft.context)
    return e,d


class BatchSession:
    def __init__(self,generator,capacities):
        self.base=generator;self.slots=[];self.graphs={};self.last_rounds=[]
        if not 1<=len(capacities)<=4:raise ValueError('batch must be 1..4')
        pages=[max(32,(int(n)+63)//64) for n in capacities]
        if sum(pages)>generator.engine.capacity//64:
            raise ValueError('batch exceeds shared KV page budget')
        if generator.engine.pending is not None:raise RuntimeError('outstanding B1 verify')
        # Base prefix cache refers to pages we are about to reuse.
        generator.cache.invalidate();offset=0
        for n in pages:
            self.slots.append(_slot(generator.engine,generator.draft,offset,n));offset+=n
        self.plans={};self.joint_drafts={}

    @torch.inference_mode()
    def _forward(self,active,ids):
        engines=[self.slots[i][0] for i in active];e=engines[0];rows=len(active)*8
        plans=[x._plan(8,'decode') for x in engines]
        if active not in self.plans:
            w=e.w.layers[3].ffn
            ws=MoELayerWorkspace.create(rows,w.router.shape[0],W.D,w.gu.shape[1]//2,
                                       w.shared_gu.shape[0]//2,e.device)
            self.plans[active]=(ws,torch.empty((rows,W.D),device=e.device,dtype=torch.float32))
        ws,out=self.plans[active]
        valid=(ids>=e.w.vocab_start)&(ids<e.w.vocab_end)
        local=(ids-e.w.vocab_start).clamp(0,e.w.vocab_end-e.w.vocab_start-1)
        x=e.w.embed[local].half()*valid[:,None];e.parallel(x);features=[]
        with ExitStack() as stack:
            for engine,plan in zip(engines,plans):
                b=plan[0][0].sparse
                stack.enter_context(b.prefill_elementwise.chunk(engine.positions,b.inv_freq,first_chunk=False))
            for l,layer in enumerate(e.w.layers):
                residual=attention_joint([p[0][l] for p in plans],x,engines,l)
                if l<3:
                    h=torch.empty_like(residual,dtype=torch.bfloat16)
                    _norm[(rows,)](residual,layer.ffn_norm,h,W.D,triton.next_power_of_2(W.D),enable_fp_fusion=False)
                    gate,up=(h@layer.ffn.gu.T).chunk(2,-1)
                    partial=(torch.nn.functional.silu(gate.float())*up.float()).bfloat16()@layer.ffn.down.T
                    e.parallel(partial);x=add_residual(residual,partial)
                else:x=plans[0][0][l].feed_forward(residual,ws,out,phase='decode')
                if l in TARGET_LAYERS:features.append(x)
        normalized=torch.empty_like(x,dtype=e.w.lm_head.dtype)
        _norm[(rows,)](x,e.w.final_norm,normalized,W.D,triton.next_power_of_2(W.D),enable_fp_fusion=False)
        local_logits=normalized@e.w.lm_head.T
        gathered=torch.empty((8*rows,local_logits.shape[1]),device=e.device,dtype=local_logits.dtype)
        dist.all_gather_into_tensor(gathered,local_logits)
        logits=gathered.view(8,rows,-1).permute(1,0,2).reshape(rows,W.VOCAB).float()
        return logits,tuple(features)

    @torch.inference_mode()
    def prepare(self,active):
        active=tuple(active)
        if active in self.graphs:return
        self.base.cache.drain()
        e=self.slots[0][0];ids=torch.zeros(len(active)*8,device=e.device,dtype=torch.long)
        # Warmup writes only the tentative suffix, never committed context.
        for i in active:
            eng=self.slots[i][0]
            eng.positions.copy_(torch.arange(eng.length,eng.length+8,device=e.device))
            eng.context.fill_(eng.length+8)
        stream=torch.cuda.Stream(priority=-1);stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):self._forward(active,ids)
        torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize()
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph,stream=stream):result=self._forward(active,ids)
        self.graphs[active]=(graph,ids,result)

    @torch.inference_mode()
    def verify(self,active,blocks,use_graph=True):
        active=tuple(active)
        if len(set(active))!=len(active) or len(active)!=len(blocks) or not active:
            raise ValueError('invalid active row mapping')
        for i,block in zip(active,blocks):
            eng=self.slots[i][0]
            if eng.pending is not None or len(block)!=8:raise ValueError('pending verify or invalid Q')
            if eng.length+8>eng.capacity:raise ValueError('row capacity exhausted')
            eng.positions.copy_(torch.arange(eng.length,eng.length+8,device=eng.device))
            eng.context.fill_(eng.length+8)
        ids=torch.tensor(blocks,device=eng.device,dtype=torch.long).flatten()
        if use_graph:
            self.prepare(active);graph,inputs,result=self.graphs[active]
            inputs.copy_(ids);graph.replay()
        else:result=self._forward(active,ids)
        logits,features=result
        for row,i in enumerate(active):
            eng=self.slots[i][0];s=slice(row*8,(row+1)*8)
            eng.pending=Result(eng.length,logits[s],tuple(t[s] for t in features))
        return logits.reshape(len(active),8,-1)

    @torch.inference_mode()
    def generate(self,prompts,limits,*,eos_token_ids=None,on_tokens=None,on_prefill=None,
                 should_stop=None,on_round=None,on_done=None,use_graph=True):
        if len(prompts)!=len(self.slots) or len(limits)!=len(prompts):raise ValueError('row count')
        for i,(prompt,limit) in enumerate(zip(prompts,limits)):
            if not prompt or limit<1 or len(prompt)+limit+8>self.slots[i][0].capacity:
                raise ValueError('invalid prompt/output capacity')
        started=time.perf_counter();outputs=[[] for _ in prompts];steps=[[] for _ in prompts]
        reasons=['length']*len(prompts);active=[];finished=set()
        def finish(i):
            if i not in finished:
                finished.add(i)
                if on_done:on_done(i,Generation(outputs[i],steps[i],time.perf_counter()-started,reasons[i]))
        stop=set(self.slots[0][1].eos if eos_token_ids is None else eos_token_ids)
        cancel=should_stop() if should_stop else [False]*len(prompts)
        for i,(prompt,limit) in enumerate(zip(prompts,limits)):
            if cancel[i]:reasons[i]='cancelled';finish(i);continue
            e,d=self.slots[i];e.prepare_decode(len(prompt)+limit+8)
            anchor,_=self.base.prefill(prompt,engine=e,draft=d)
            outputs[i].append(anchor)
            if on_prefill:on_prefill(i)
            if on_tokens:on_tokens(i,[anchor])
            if anchor in stop:reasons[i]='eos'
            elif limit>1:active.append(i)
            if i not in active:finish(i)
        if use_graph:
            for i in active:self.slots[i][1].capture()
            if active:self.prepare(tuple(active))
        while active:
            cancel=should_stop() if should_stop else [False]*len(prompts)
            for i in active:
                if cancel[i]:reasons[i]='cancelled';finish(i)
            active=[i for i in active if not cancel[i]]
            if not active:break
            # Capture shape transitions outside the timed complete round.
            key=tuple(active)
            if key not in self.joint_drafts:
                jd=JointDraft([self.slots[i][1] for i in active]);self.joint_drafts[key]=jd
                for i in active:
                    e,d=self.slots[i];d.anchor.fill_(outputs[i][-1]);d.context.fill_(e.length)
                if use_graph:jd.prepare()
            if use_graph:self.prepare(key)
            torch.cuda.synchronize();tick=time.perf_counter()
            jd=self.joint_drafts[tuple(active)]
            drafts=jd.run([outputs[i][-1] for i in active],use_graph).tolist()
            blocks=[[outputs[i][-1]]+draft for i,draft in zip(active,drafts)]
            predictions=self.verify(active,blocks,use_graph).argmax(-1).tolist()
            next_active=[]
            for i,block,pred in zip(active,blocks,predictions):
                e,d=self.slots[i];before=e.length;n=accepted_prefix(block,pred)
                extension=(block[1:n]+[pred[n-1]])[:limits[i]-len(outputs[i])]
                for j,token in enumerate(extension):
                    if token in stop:extension=extension[:j+1];reasons[i]='eos';break
                consumed=min(n,1+len(extension));e.commit(consumed,on_commit=d.append)
                if e.length!=d.pool.lengths[0]:raise RuntimeError('row target/draft divergence')
                outputs[i].extend(extension)
                steps[i].append(dict(start=before,consumed=consumed,accepted_drafts=n-1))
                if on_tokens:on_tokens(i,extension)
                if len(outputs[i])<limits[i] and reasons[i]!='eos':next_active.append(i)
                else:finish(i)
            torch.cuda.synchronize();ms=(time.perf_counter()-tick)*1000
            record=dict(active=active.copy(),batch=len(active),round_ms=ms)
            self.last_rounds.append(record)
            if on_round:on_round(record)
            active=next_active
        elapsed=time.perf_counter()-started
        return [Generation(o,s,elapsed,r) for o,s,r in zip(outputs,steps,reasons)]

    def close(self):
        torch.cuda.synchronize()
        for graph,_,_ in self.graphs.values():graph.reset()
        self.graphs.clear()
        for jd in self.joint_drafts.values():jd.close()
        self.joint_drafts.clear()
        for e,d in self.slots:d.close();e.pending=None
        self.slots.clear();self.plans.clear();self.base.cache.invalidate()


def generate_batch(generator,prompts,limits,**kwargs):
    session=BatchSession(generator,[len(p)+n+8 for p,n in zip(prompts,limits)])
    try:return session.generate(prompts,limits,**kwargs)
    finally:session.close()
