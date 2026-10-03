"""Resident Q8 executor: fixed graph rows, epoch page leases, commit-point boarding.
Port of GLM52 batch_decode's row/epoch contract to GLM53 DFlash2.
No weight/KV copies on compaction; only page maps and committed lengths move.
"""
import time
import torch
from model.glm53_batch import BatchSession, _slot
from model.glm53_joint_draft import JointDraft
from model.glm53_generate import Generation, accepted_prefix


class ResidentBatch(BatchSession):
    def __init__(self, generator, max_batch=4):
        self.generator=generator;self.base=generator;self.engine=generator.engine
        self.max_batch=max_batch;self.slots=[];self.plans={};self.last_rounds=[]
        self.graphs={};self.joint_drafts={};self.frozen=False
        self.used_pages=0;self.row_ids=[];self.capture_count=0;self.busy=False
        # Stable logical row buffers over the SAME physical pools. Page ownership
        # is assigned at admission, never by slicing the physical KV allocation.
        for _ in range(max_batch):
            self.slots.append(_slot(self.engine,generator.draft,0,self.engine.capacity//64))

    def _bind(self,row,start,pages):
        e,d=self.slots[row]
        e.cache_page_span=(start,pages)
        e.table.zero_();e.table[:pages].copy_(torch.arange(start,start+pages,device=e.device))
        # All scan buffers retain full logical capacity; unused mapped pages are
        # masked by positions/context, and always point to a valid physical page.
        mapping=self.engine.mtp_kv.host_page_table[0][start:start+pages]
        d.pool.page_table.zero_()
        d.pool.page_table[0,:pages].copy_(torch.tensor(mapping,device=e.device))
        d.pool.host_page_table[0]=mapping+[mapping[0]]*(d.pool.logical_pages-pages)
        e.length=0;e.pending=None;d.pool.lengths[0]=0
        e.positions.copy_(torch.arange(8,device=e.device));e.context.fill_(8)
        d.context.zero_();d.anchor.zero_()

    def _compact(self,keep):
        # keep is ascending: destination is never to the right of its source.
        for dst,src in enumerate(keep):
            if dst==src:continue
            e,d=self.slots[dst];se,sd=self.slots[src]
            assert se.pending is None and se.length==sd.pool.lengths[0]
            e.table.copy_(se.table);d.pool.page_table.copy_(sd.pool.page_table)
            d.pool.host_page_table[0]=sd.pool.host_page_table[0].copy()
            e.length=se.length;e.pending=None;d.pool.lengths[0]=se.length
        self.row_ids=[self.row_ids[i] for i in keep]

    def _resident_prepare(self,b,use_graph):
        rows=tuple(range(b))
        for e,d in self.slots[:b]:e.decode_capacity=e.capacity
        if use_graph:
            if rows not in self.graphs or rows not in self.joint_drafts:
                self.generator.cache.drain()
            for e,d in self.slots[:b]:
                if d.append_graphs is None:d.capture_append()
            if rows not in self.graphs:
                if self.frozen:raise RuntimeError("missing prebuilt batch graph")
                self.prepare(rows);self.capture_count+=1
        if rows not in self.joint_drafts:
            self.joint_drafts[rows]=JointDraft([self.slots[i][1] for i in rows])
        if use_graph:self.joint_drafts[rows].prepare()
        return rows

    @torch.inference_mode()
    def warm(self):
        if self.busy:raise RuntimeError('warm during active epoch')
        self.generator.reset()
        # Capture suffixes are disjoint even at startup.
        for row in range(self.max_batch):self._bind(row,row,1)
        for b in range(1,self.max_batch+1):self._resident_prepare(b,True)
        self.frozen=True
        self.generator.reset()

    @torch.inference_mode()
    def generate(self,prompts,limits,*,eos_token_ids=None,use_graph=True,
                 on_tokens=None,on_prefill=None,on_done=None,should_stop=None,
                 on_round=None,board_request=None,on_boarded=None):
        if self.busy:raise RuntimeError('overlapping resident epochs')
        if not prompts or len(prompts)!=len(limits) or len(prompts)>self.max_batch:
            raise ValueError('invalid initial batch')
        prompts=[list(p) for p in prompts];limits=list(limits)
        if any(not p or n<1 or len(p)+n+8>self.engine.capacity for p,n in zip(prompts,limits)):
            raise ValueError('invalid request capacity')
        if sum((len(p)+n+8+63)//64 for p,n in zip(prompts,limits))>self.engine.capacity//64:
            raise MemoryError('epoch page budget')
        self.busy=True;self.generator.cache.invalidate();self.used_pages=0;self.row_ids=[]
        self.last_rounds=[];outputs=[];steps=[];reasons=[];done=[];born=[]
        stop=set(self.generator.draft.eos if eos_token_ids is None else eos_token_ids)
        captures_before=self.capture_count
        def finish(rid,reason=None):
            if done[rid]:raise RuntimeError('duplicate completion')
            if reason:reasons[rid]=reason
            done[rid]=True
            if on_done:on_done(rid,Generation(outputs[rid],steps[rid],time.perf_counter()-born[rid],reasons[rid]))
        def admit(p,n,rid,cancelled=False,cohort=None):
            if cancelled:
                outputs.append([]);steps.append([]);reasons.append('cancelled');done.append(False);born.append(time.perf_counter())
                finish(rid,'semantic_eos' if cancelled==2 else 'cancelled');return None
            row=len(self.row_ids);pages=(len(p)+n+8+63)//64
            if self.used_pages+pages>self.engine.capacity//64:raise MemoryError('epoch leases exhausted')
            self._bind(row,self.used_pages,pages);self.used_pages+=pages
            self.row_ids.append(rid);outputs.append([]);steps.append([]);reasons.append('length');done.append(False);born.append(time.perf_counter())
            e,d=self.slots[row]
            anchor,metrics=self.generator.prefill(p,engine=e,draft=d)
            outputs[rid].append(anchor)
            if anchor in stop:reasons[rid]='eos'
            if on_prefill:on_prefill(rid,dict(metrics,seconds=metrics['model_prefill_seconds'],
                batch_size=len(self.row_ids) if cohort is None else cohort))
            if on_boarded:on_boarded(rid,len(self.row_ids))
            return anchor
        try:
            flags=should_stop() if should_stop else [False]*len(prompts)
            if len(flags)!=len(prompts):raise ValueError('cancel vector size')
            anchors=[admit(p,n,i,flags[i],sum(not f for f in flags)) for i,(p,n) in enumerate(zip(prompts,limits))]
            # A cold shape is prepared before emitting its first output. Startup
            # warm() makes every context length and B1..B4 capture-free.
            if self.row_ids:self._resident_prepare(len(self.row_ids),use_graph)
            for rid,anchor in enumerate(anchors):
                if on_tokens and anchor is not None:on_tokens(rid,[anchor])
            while self.row_ids:
                flags=should_stop() if should_stop else [False]*len(outputs)
                if len(flags)!=len(outputs):raise ValueError('cancel vector size')
                keep=[]
                for row,rid in enumerate(self.row_ids):
                    if flags[rid]:finish(rid,'semantic_eos' if flags[rid]==2 else 'cancelled')
                    elif outputs[rid][-1] in stop:finish(rid,'eos')
                    elif len(outputs[rid])>=limits[rid]:finish(rid)
                    else:keep.append(row)
                self._compact(keep)
                # All previous verify results have been committed; no graph runs
                # while page maps or active membership change. Match GLM52 epoch
                # lease policy: finished pages are released only at epoch end.
                while self.row_ids and len(self.row_ids)<self.max_batch and board_request:
                    remaining=(self.engine.capacity//64-self.used_pages)*64
                    request=board_request(remaining,self.engine.capacity)
                    if request is None:break
                    p,n=request;p=list(p);n=int(n)
                    if not p or n<1 or len(p)+n+8>remaining:raise ValueError('invalid boarded lease')
                    rid=len(outputs);prompts.append(p);limits.append(n)
                    anchor=admit(p,n,rid)
                    self._resident_prepare(len(self.row_ids),use_graph)
                    if on_tokens:on_tokens(rid,[anchor])
                # Terminal-at-anchor admissions are retired at next safe point;
                # never put them through verify (max_new_tokens=1 included).
                if any(len(outputs[i])>=limits[i] or outputs[i][-1] in stop for i in self.row_ids):continue
                if not self.row_ids:break
                b=len(self.row_ids);active=list(range(b));key=self._resident_prepare(b,use_graph)
                tick=time.perf_counter();emit_seconds=0.0
                proposals=self.joint_drafts[key].run([outputs[i][-1] for i in self.row_ids],use_graph).tolist()
                blocks=[[outputs[rid][-1]]+proposals[row] for row,rid in enumerate(self.row_ids)]
                predicted=self.verify(active,blocks,use_graph).argmax(-1).tolist()
                for row,rid in enumerate(self.row_ids):
                    e,d=self.slots[row];before=e.length
                    accept=accepted_prefix(blocks[row],predicted[row])
                    extension=(blocks[row][1:accept]+[predicted[row][accept-1]])[:limits[rid]-len(outputs[rid])]
                    for j,token in enumerate(extension):
                        if token in stop:extension=extension[:j+1];reasons[rid]='eos';break
                    consume=min(accept,1+len(extension))
                    e.commit(consume,on_commit=d.append)
                    assert e.length==d.pool.lengths[0]==before+consume
                    outputs[rid].extend(extension);steps[rid].append(dict(start=before,consumed=consume,accepted_drafts=accept-1))
                    if on_tokens:
                        emit_start=time.perf_counter()
                        on_tokens(rid,extension)
                        emit_seconds+=time.perf_counter()-emit_start
                if on_round is not None:
                    # Same host-call boundary as v41f obey(): no extra GPU fence.
                    rec=dict(active=self.row_ids.copy(),batch=b,round_ms=(time.perf_counter()-tick-emit_seconds)*1000,captures=self.capture_count-captures_before)
                    self.last_rounds.append(rec);on_round(rec)
            return [Generation(o,s,time.perf_counter()-t,r) for o,s,t,r in zip(outputs,steps,born,reasons)]
        finally:
            torch.cuda.synchronize()
            for e,d in self.slots:e.pending=None;e.length=0;d.pool.lengths[0]=0
            self.row_ids=[];self.used_pages=0;self.busy=False;self.generator.cache.invalidate()

    def close(self):
        if self.busy:raise RuntimeError('close during active epoch')
        torch.cuda.synchronize()
        super().close()
