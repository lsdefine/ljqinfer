"""Lifetime-owned views over existing MoE/attention workspaces; no new kernels."""
from copy import copy
from math import prod
import torch
from ops import sparse_topk_v2


def rows_view(obj, rows, names):
    result = copy(obj)
    for name in names:
        setattr(result, name, getattr(obj, name)[:rows])
    return result


class IndexBuffers:
    """One phase-owned index scratch, sequentially reused by every full indexer."""
    def __init__(self, tokens, capacity, phase, device, topk=2048):
        self.tokens, self.capacity, self.phase = tokens, capacity, phase
        self.topk = topk
        local = (tokens+7)//8
        self.tile = 1536 if phase == 'prefill' else 256
        def alloc(shape, dtype):
            return torch.empty(shape, dtype=dtype, device=device)
        width = 128 if phase == 'prefill' else 144
        self.buffers = {
            'queries': alloc((8*tokens,4,width), torch.float16),
            'selected': alloc((local,topk), torch.int32),
            'pos': alloc((local,), torch.int64),
            'scores': alloc((min(local,self.tile),capacity), torch.float32),
            'gathered': alloc((8*local,topk), torch.int32),
        }
        if phase == 'prefill':
            self.buffers['weights'] = alloc((8*tokens,4), torch.float32)
        else:
            self.buffers['packed'] = alloc((tokens,4,144), torch.float16)

    def view(self, tokens, capacity):
        if not 0 < tokens <= self.tokens or not self.topk <= capacity <= self.capacity:
            raise ValueError('index scratch view exceeds lifetime owner')
        local = (tokens+7)//8
        width = 128 if self.phase == 'prefill' else 144
        shapes = {'queries': (8*tokens,4,width), 'selected': (local,self.topk),
                  'pos': (local,), 'scores': (min(local,self.tile),capacity),
                  'gathered': (8*local,self.topk)}
        shapes.update({'weights': (8*tokens,4)} if self.phase == 'prefill'
                      else {'packed': (tokens,4,144)})
        # Flatten first: narrow logical score widths must still be contiguous.
        return {k: self.buffers[k].view(-1)[:prod(shape)].view(shape)
                for k,shape in shapes.items()}


class Workspaces:
    def __init__(self, engine):
        self.engine = engine
        # Only these two calls allocate plans. Request lengths select views.
        self.owners = {
            'prefill': engine._allocate_plan(engine.prefill_chunk_tokens, 'prefill'),
            'decode': engine._allocate_plan(engine.q, 'decode'),
        }
        self.topk = sparse_topk_v2.workspace(1, engine.capacity, device=engine.device)
        self.index = {phase: IndexBuffers(tokens, engine.capacity, phase, engine.device)
                      for phase,tokens in [('prefill',engine.prefill_chunk_tokens),('decode',engine.q)]}
        self.latent = {phase: torch.empty((tokens,8,512),dtype=torch.float16,device=engine.device)
                       for phase,tokens in [('prefill',engine.prefill_chunk_tokens),('decode',engine.q)]}
        self.views = {}

    def plan(self, tokens, phase):
        e = self.engine
        capacity = (e.decode_capacity if phase == 'decode' else
                    max(2048, min(e.capacity, ((e.length+tokens+63)//64)*64)))
        old = self.views.get(phase)
        if old is not None and old[0] == (tokens, capacity):
            return old[1]
        owner_blocks, owner_ws, owner_out = self.owners[phase]
        ws = rows_view(owner_ws, tokens, ('logits', 'router_x', 'ids', 'route_weights',
                                         'shared_gu', 'shared_act', 'shared_out'))
        ws.routed = copy(owner_ws.routed)
        ws.routed.hidden = owner_ws.routed.hidden[:tokens*8]
        ws.routed.partial = owner_ws.routed.partial[:tokens*8]
        ws.routed.slots = owner_ws.routed.slots[:, :tokens*8]
        # Slots require contiguous rows. Reinterpret the fixed backing allocation.
        ws.routed.slots = owner_ws.routed.slots.reshape(-1)[:ws.routed.slots.numel()].view(ws.routed.slots.shape)
        scratch = self.index[phase].view(tokens, capacity)
        latent = self.latent[phase][:tokens]
        blocks = []
        previous = None
        for owner in owner_blocks:
            block = copy(owner)
            b = copy(owner.sparse)
            b.tokens, b.capacity = tokens, capacity
            b.index_scratch = scratch
            b.latent = latent
            b.ids = owner.sparse.ids[:tokens]
            b.metadata = None
            b.shared_from = previous if b.index_weights is None else None
            if phase == 'decode':
                chunks = (capacity+4095)//4096
                shapes = [(1,chunks,1024),(1,4),(1,chunks,2),(1,capacity),(1,capacity)]
                b.topk_workspace = tuple(v.reshape(-1)[:prod(shape)].view(shape)
                                          for v,shape in zip(self.topk,shapes))
            if b.pair is not None:
                pair = copy(b.pair)
                pair.t = tokens
                pair.half = (tokens+1)//2
                pair.lo = (b.pair.lo//b.pair.half)*pair.half
                for name in ('qsend','qrecv','osend','orecv','ids','pos'):
                    setattr(pair,name,getattr(b.pair,name)[:2*pair.half])
                if tokens % 2:
                    pair.qsend[tokens:].zero_()
                    pair.ids[tokens:].fill_(-1)
                    pair.pos[tokens:].fill_(-1)
                b.pair = pair
            if b.index_weights is not None:
                previous = b
            block.sparse = b
            blocks.append(block)
        plan = blocks, ws, owner_out[:tokens]
        self.views[phase] = ((tokens,capacity),plan)
        return plan
