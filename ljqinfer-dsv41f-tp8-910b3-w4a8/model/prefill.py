"""DSv4.1 eager encoder/source20/CED. Past is the only persistent state owner.
The caller reserves pages, commits pos and restores cold carry. One serialized
call lane per Past; eager encoder, bounded CED graph, no private persistent KV.
"""
from dataclasses import dataclass
import torch
from ops.prefill import residual as r
from model.engram_prefetch import EngramPrefetch
from model.engram_workspace import EngramWorkspace
from model.index_workspace import IndexWorkspace
from model.prefill_trace import span
from model.prefill_attention import AttentionState

ENCODER_LAYERS, CED_ROWS = 20, 128


@dataclass
class PrefillOutput:
    logits: torch.Tensor | None
    main_hidden: torch.Tensor | None


class PrefillModel:
    def __init__(self, config, weights, blocks, parallel, *, past, phase, length,
                 capacity, device, library=None):
        self.c, self.weights, self.blocks, self.parallel = config, weights, blocks, parallel
        self.past, self.phase, self.length, self.capacity = past, phase, length, capacity
        self.device, self.library = torch.device(device), library
        self.encoder, self.decoder = blocks[:ENCODER_LAYERS], blocks[ENCODER_LAYERS:]
        self.engram_workspace = EngramWorkspace(length, config, parallel, self.device)
        self.index_workspace = IndexWorkspace(self.engram_workspace, capacity, config, parallel.world)
        self.prefetch = EngramPrefetch(self.encoder, device=self.device, length=length)
        self.closed = False
        self.ced_graph = None

    def select(self, *, past, length, phase, capacity=None):
        capacity = self.capacity if capacity is None else capacity
        if past is not self.past or not 1 <= length <= self.length:
            raise ValueError('selection exceeds this prefill lane')
        if phase != self.phase or capacity > self.capacity:
            raise ValueError('selection phase/capacity differs from this lane')
        return self

    def _check(self, past, slot, start, tokens, phase):
        if self.closed or past is not self.past or not 0 <= slot < past.n_slots or slot in past.free_slots:
            raise ValueError('prefill requires its live Past slot')
        end = start+len(tokens)
        if not 0 <= start < end <= self.capacity or len(tokens) > self.length:
            raise ValueError('invalid live encoder extent')
        if any(type(t) is not int or not 0 <= t < self.c['vocab_size'] for t in tokens):
            raise ValueError('invalid token id')
        if phase == 'encoder_append':
            if past.pos[slot] != start or slot in past.replay_pending:
                raise ValueError('append position/replay state mismatch')
        elif phase == 'encoder_resume':
            hit = past.pos[slot]
            if slot not in past.replay_pending or start != max(0, hit-CED_ROWS) or end <= hit:
                raise ValueError('resume must join bounded replay and the new suffix')
        elif end != past.pos[slot] or len(tokens) > CED_ROWS:
            raise ValueError('replay must be the bounded cached suffix')
        for source in past.sources.values():
            if past.pt.n_alloc(slot)*source.ckv_pool.rpp < end//source.ratio:
                raise ValueError('caller must reserve Past pages before prefill')
        return end

    @staticmethod
    def _image_rows_for(past, slot, start, length):
        result = []
        for span in getattr(past, 'image_spans', {}).get(slot, ()):
            lo, features = span
            begin, end = max(start, lo), min(start+length, lo+len(features))
            if begin < end:
                result.append((begin-start, features[begin-lo:end-lo]))
        return tuple(result)

    def _embed(self, tokens, slot, start):
        w, par, c = self.weights, self.parallel, self.c
        shard = w['embed.weight'].shape[0]
        ids = torch.tensor(tokens, dtype=torch.long, device=self.device)-par.rank*shard
        mask = (ids < 0) | (ids >= shard)
        hidden = w['embed.weight'][ids.masked_fill(mask, 0)].clone()
        hidden.masked_fill_(mask[:, None], 0)
        par.sum(hidden)
        images = self._image_rows_for(self.past, slot, start, len(tokens))
        for offset, features in images:
            hidden[offset:offset+len(features)].copy_(features.to(hidden))
        h = hidden[:, None].expand(-1, c['hc_mult'], -1).contiguous()
        pre = torch.zeros((len(tokens), c['hc_mult']), dtype=torch.float32, device=self.device)
        pre[:, 0] = 1
        return h, pre, images

    def _retain(self, slot, start, end, h, pre, replace=False):
        past, c = self.past, self.c
        if not hasattr(past, '_prefill_tail_storage'):
            shape = (past.n_slots, CED_ROWS, c['hc_mult'])
            past._prefill_tail_storage = (
                torch.empty((*shape, c['dim']), device=self.device, dtype=h.dtype),
                torch.empty(shape, device=self.device, dtype=pre.dtype))
        th, tp = past._prefill_tail_storage
        old = past.prefill_tails.get(slot)
        keep = 0
        if not replace and old is not None:
            if old['end'] != start:
                raise ValueError('nonconsecutive encoder tail')
            keep = min(CED_ROWS-min(len(h), CED_ROWS), old['end']-old['start'])
        previous_h = old['h'][-keep:].clone() if keep else h[:0]
        previous_p = old['pre'][-keep:].clone() if keep else pre[:0]
        n = min(len(h), CED_ROWS)
        th[slot].zero_()
        tp[slot].zero_()
        if keep:
            th[slot, CED_ROWS-n-keep:CED_ROWS-n].copy_(previous_h)
            tp[slot, CED_ROWS-n-keep:CED_ROWS-n].copy_(previous_p)
        th[slot, -n:].copy_(h[-n:])
        tp[slot, -n:].copy_(pre[-n:])
        past.prefill_tails[slot] = dict(h=th[slot], pre=tp[slot], start=end-n-keep, end=end)

    @span("chunk")
    def _encoder(self, tokens, *, past, slot, start, history_tokens=(), phase='encoder_append'):
        tokens, history_tokens = tuple(tokens), tuple(history_tokens)
        end = self._check(past, slot, start, tokens, phase)
        self.prefetch.submit(slot=slot, start=start, tokens=tokens, history_tokens=history_tokens)
        try:
            h, pre, images = self._embed(tokens, slot, start)
            state = AttentionState(past, slot, start, end, phase, self.library, self.index_workspace)
            for block in self.encoder:
                if block.engram is not None:
                    prepared = self.prefetch.wait(block.layer)
                    if images:
                        saved = h
                        h = block.engram(h, prepared, self.engram_workspace)
                        for offset, features in images:
                            h[offset:offset+len(features)].copy_(saved[offset:offset+len(features)])
                    else:
                        h = block.engram(h, prepared, self.engram_workspace)
                h, pre = block(h, pre, state)
            if phase in ('encoder_append', 'encoder_resume'):
                source = self.decoder[0]
                source.attention.source(source.norm(h, pre, 'attn'), state)
            self._retain(slot, start, end, h, pre, phase != 'encoder_append')
        finally:
            self.prefetch.drain()

    @torch.inference_mode()
    def forward(self, tokens, *, past, slot, start, history_tokens=()):
        self._encoder(tokens, past=past, slot=slot, start=start, history_tokens=history_tokens,
                      phase='encoder_resume' if slot in past.replay_pending else 'encoder_append')
        return PrefillOutput(None, None)

    @torch.inference_mode()
    def replay(self, tokens, *, past, slot, start, history_tokens=()):
        self._encoder(tokens, past=past, slot=slot, start=start, history_tokens=history_tokens,
                      phase='encoder_replay')
        # Cached globals, carry, pos and replay_pending remain untouched.
        return None

    @torch.inference_mode()
    def finish_prefill(self, *, past, slot):
        if self.closed or past is not self.past or slot in past.replay_pending:
            raise ValueError('cold replay/carry restoration must finish first')
        tail = past.prefill_tails.get(slot)
        if tail is None or tail['end'] != past.pos[slot]:
            raise ValueError('finish requires a committed encoder tail')
        end = tail['end']
        if tail['start'] != max(0, end-CED_ROWS):
            raise ValueError('finish requires the entire CED128 suffix')
        if self.ced_graph is None:
            from model.prefill_ced import CEDGraph
            self.ced_graph = CEDGraph(self)
        return self.ced_graph(slot)

    def close(self):
        if not self.closed:
            if self.ced_graph is not None:
                self.ced_graph.close()
                self.ced_graph = None
            self.prefetch.close()
            self.closed = True
