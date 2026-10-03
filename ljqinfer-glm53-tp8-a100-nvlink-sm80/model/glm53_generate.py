"""B1 greedy DFlash2 generation; only target predictions authorize output."""
from dataclasses import dataclass
import time
import torch
from .glm53_engine import Engine
from model.config import DEFAULT_PREFILL_CHUNK_TOKENS
from .glm53_dflash import DFlash2
from .glm53_cache import PrefixState


def accepted_prefix(proposed, predictions):
    """Anchor is already target-authorized; subsequent proposals must match."""
    n=1
    while n<len(proposed) and proposed[n]==predictions[n-1]:n+=1
    return n


@dataclass
class Generation:
    token_ids:list
    steps:list
    elapsed_seconds:float
    finish_reason:str


class Generator:
    def __init__(self,engine,draft):
        self.engine=engine;self.draft=draft
        engine.mtp_kv.reserve(0,engine.capacity)
        self.cache=PrefixState(engine)
        self.prefill_stats={}

    @classmethod
    def load(cls,*,capacity=8192,prefill_chunk_tokens=DEFAULT_PREFILL_CHUNK_TOKENS,draft_dir=None):
        e=Engine.load(capacity=capacity,prefill_chunk_tokens=prefill_chunk_tokens)
        d=DFlash2(e,**({'model_dir':draft_dir} if draft_dir else {}))
        return cls(e,d)

    def reset(self):
        self.cache.drain()
        self.engine.reset();self.engine.mtp_kv.lengths[0]=0
        self.cache.clear()

    @torch.inference_mode()
    def prefill(self, prompt_ids, *, engine=None, draft=None):
        """Single-request prefill lifecycle, shared by single and batched decode.

        As in v41f Strategy: restore -> compute/commit/store each prompt chunk
        -> return continuation. Decode and retirement do not publish cold KV.
        """
        engine = self.engine if engine is None else engine
        draft = self.draft if draft is None else draft
        started = time.perf_counter()
        self.cache.drain()
        drained = time.perf_counter() - started
        cache = self.cache if engine is self.engine else self.cache.for_engine(engine)
        mark_restore = time.perf_counter()
        hit, source = cache.restore(prompt_ids)
        torch.cuda.current_stream(self.engine.device).synchronize()
        restored = time.perf_counter() - mark_restore
        compute_seconds = store_seconds = 0.0
        result = None
        try:
            for offset in range(hit, len(prompt_ids), engine.prefill_chunk_tokens):
                end = min(len(prompt_ids), offset + engine.prefill_chunk_tokens)
                mark = time.perf_counter()
                result = engine.prefill(prompt_ids[offset:end],
                    on_commit=draft.append, last_logits_only=True)
                # Drain compute before attributing time to cache storage.
                torch.cuda.current_stream(self.engine.device).synchronize()
                compute_seconds += time.perf_counter() - mark
                mark = time.perf_counter()
                cache.publish(prompt_ids[:end])
                store_seconds += time.perf_counter() - mark
            anchor = int(result.logits[-1].argmax().item())
        except BaseException:
            cache.invalidate()
            raise
        metrics = dict(cache_hit_tokens=hit, cache_source=source,
            cache_load_seconds=restored, cache_restore_seconds=restored,
            cache_store_seconds=store_seconds, cache_store_enqueue_seconds=store_seconds,
            cache_store_drain_seconds=drained, model_prefill_seconds=compute_seconds,
            prefill_tokens=len(prompt_ids)-hit, input_tokens=len(prompt_ids),
            strategy_seconds=time.perf_counter()-started)
        return anchor, metrics

    @torch.inference_mode()
    def generate(self,prompt_ids,*,max_new_tokens=64,use_graph=True,eos_token_ids=None,on_tokens=None,
                 on_prefill=None,should_stop=None,profile=False):
        prompt_ids=list(prompt_ids)
        if not prompt_ids:raise ValueError('empty prompt')
        if max_new_tokens<0:raise ValueError('negative generation length')
        if len(prompt_ids)+max_new_tokens+8>self.engine.capacity:
            raise ValueError('prompt+generation+verify scratch exceeds capacity')
        steps=[];output=[]
        if max_new_tokens==0:return Generation([],[],0.,'length')
        stop=set(self.draft.eos if eos_token_ids is None else eos_token_ids)
        started=time.perf_counter()
        self.engine.prepare_decode(len(prompt_ids)+max_new_tokens+8)
        anchor,self.prefill_stats=self.prefill(prompt_ids)
        if on_prefill:on_prefill()
        output.append(anchor)
        if on_tokens:on_tokens([anchor])
        if anchor in stop:
            return Generation(output,steps,time.perf_counter()-started,'eos')
        if use_graph:
            if self.engine.graph is None or self.draft.graph is None:self.cache.drain()
            if self.engine.graph is None:self.engine.capture_verify()
            if self.draft.graph is None:self.draft.capture()
        reason='length'
        while len(output)<max_new_tokens:
            if should_stop and should_stop():
                reason="cancelled";break
            before=self.engine.length
            if profile:torch.cuda.current_stream(self.engine.device).synchronize()
            t=time.perf_counter()
            proposals=self.draft.draft(anchor).tolist()
            if profile:torch.cuda.current_stream(self.engine.device).synchronize()
            draft_ms=(time.perf_counter()-t)*1000 if profile else None
            block=[anchor]+proposals
            t=time.perf_counter();checked=self.engine.verify(block)
            predicted=checked.logits.argmax(-1).tolist()
            if profile:torch.cuda.current_stream(self.engine.device).synchronize()
            verify_ms=(time.perf_counter()-t)*1000 if profile else None
            accepted=accepted_prefix(block,predicted)
            extension=block[1:accepted]+[predicted[accepted-1]]
            extension=extension[:max_new_tokens-len(output)]
            for i,token in enumerate(extension):
                if token in stop:extension=extension[:i+1];reason='eos';break
            # Consumed tokens: anchor plus accepted drafts emitted this round.
            # The bonus target token remains the next uncached anchor.
            consume=min(accepted,1+len(extension))
            t=time.perf_counter();self.engine.commit(consume,on_commit=self.draft.append)
            if profile:torch.cuda.current_stream(self.engine.device).synchronize()
            commit_ms=(time.perf_counter()-t)*1000 if profile else None
            assert self.engine.length==self.engine.mtp_kv.lengths[0]==before+consume
            output.extend(extension)
            steps.append(dict(start=before,consumed=consume,accepted_drafts=accepted-1,
                              draft_ms=draft_ms,verify_ms=verify_ms,commit_ms=commit_ms))
            if on_tokens:on_tokens(extension)
            if reason=='eos':break
            anchor=extension[-1]
        return Generation(output,steps,time.perf_counter()-started,reason)

    def close(self):
        self.cache.close()
        torch.cuda.current_stream(self.engine.device).synchronize();self.draft.close()
        if self.engine.graph is not None:
            self.engine.graph.reset();self.engine.graph=None;self.engine.graph_result=None
        self.engine.pending=None
