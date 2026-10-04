"""CED128 paged graph: fixed buffers, dynamic device position and slot.

Sequential per-layer captures share a pool; encoder remains eager.
Past owns paged history; graph buffers hold only bounded CED working state.
"""
import torch
from model.prefill import PrefillOutput
from model.prefill_attention import AttentionState
from model.prefill_config import rotary_frequencies
from ops.prefill import attention as a, residual as r

class CEDState(AttentionState):
    def __init__(self, model):
        super().__init__(model.past, 0, 0, 128, 'ced', model.library, model.index_workspace)
        dev = model.device
        self.capacity = model.past.max_seq
        self.pos = torch.empty(128, device=dev, dtype=torch.int64)
        self.meta = torch.empty(2, device=dev, dtype=torch.int64)
        self.sid = torch.empty(1, device=dev, dtype=torch.int64)
        self.valid = self.meta[1:]
        self.query_length = torch.full((1,), 128, device=dev, dtype=torch.int32)
        self.subpages = torch.arange(2, device=dev, dtype=torch.int32)
        self.rotary = torch.empty((128, model.c['rope_head_dim']//2, 2), device=dev)

    def positions(self, device):
        return self.pos

    def attend(self, layer, q, kv, iq, iw, sink, parallel, total_heads, topk):
        from model.paged_attention import pack
        view = self.past.views[layer]
        source = self.past.sources[view.kv_source_layer]
        if view.ratio != 1:
            raise ValueError('CED decoder requires ratio-one global history')
        if view.mode != 'reuse':
            # Existing capture-safe TP gather; only the 128 query rows move.
            gathered_q = iq.new_empty((parallel.world, *iq.shape))
            gathered_w = iw.new_empty((parallel.world, *iw.shape))
            parallel.logits(iq.contiguous(), out=gathered_q)
            parallel.logits(iw.contiguous(), out=gathered_w)
            query = gathered_q.permute(1, 0, 2, 3).reshape(1, 128, total_heads, iq.shape[-1])
            weight = gathered_w.permute(1, 0, 2).reshape(1, 128, total_heads)
            index = source.index_pool
            if index.rpp != 2048 or index.data.dtype != torch.bfloat16:
                raise ValueError('paged CED expects canonical BF16 2048-row index pages')
            table = index.pt.table.index_select(0, self.sid).to(torch.int32)
            table = (table[:, :, None] * 2 + self.subpages).flatten(1)
            ids = a.native.paged_index(query.contiguous(), index.data.view(-1, 1024, 1, iq.shape[-1]),
                                       weight.contiguous(), self.query_length,
                                       self.valid.to(torch.int32), table, topk)
            self.selections[view.index_source_layer] = ids.reshape(128, topk)
        selected = self.selections[view.index_source_layer]
        bank = source.ckv_pool
        packed, indices, missing = pack(kv, bank.data, bank.pt.table, self.sid,
                                       selected, self.pos, self.meta[:1], self.valid)
        out, mx, sm = a.native.flash_sparse(q, packed, indices)
        out = a.native.joint_correct(out, mx, sm, missing, sink, indices.shape[-1])
        window = self.past.windows[layer]
        rows = self.sid * (window.pad + window.ring) + window.pad + self.pos.remainder(window.ring)
        dst = window.main_kv.flatten(0, 1)
        value = torch.where((self.pos >= 0)[:, None], kv, dst.index_select(0, rows))
        dst.index_copy_(0, rows, value)
        return out

class GraphSequence:

    def __init__(self):
        self.graphs = []

    def replay(self):
        for graph in self.graphs:
            graph.replay()

    def reset(self):
        for graph in reversed(self.graphs):
            graph.reset()
        self.graphs.clear()

class CEDGraph:

    def __init__(self, model):
        self.model = model
        self.state = CEDState(model)
        self.graph = None
        tail = next(iter(model.past.prefill_tails.values()))
        self.h = torch.empty_like(tail['h'])
        self.pre = torch.empty_like(tail['pre'])

    def prepare(self, slot):
        tail = self.model.past.prefill_tails[slot]
        end = tail['end']
        s = self.state
        assert 0 < end <= s.capacity
        self.h.copy_(tail['h'])
        self.pre.copy_(tail['pre'])
        s.sid.fill_(slot)
        s.meta.copy_(torch.tensor([end - 128, end], device=self.model.device))
        s.pos.copy_(torch.arange(end - 128, end, device=self.model.device))
        s.rotary.copy_(torch.view_as_real(rotary_frequencies(
            self.model.c, 20, 128, device=self.model.device, positions=s.pos)))

    def compute(self):
        m, s = (self.model, self.state)
        s.selections = {}
        s.freqs = {True: s.rotary}
        h, pre = (self.h.clone(), self.pre.clone())
        features = []
        for block in m.decoder:
            if block.layer in (37, 38, 39):
                features.append(h.mean(dim=1))
            h, pre = block(h, pre, s)
        hidden = r.collapse_norm(h[-1:], pre[-1:], m.weights['norm.weight'], m.c['norm_eps'])
        local = hidden.float() @ m.weights['head.weight'].float().t()
        logits = torch.empty((1, m.c['vocab_size']), device=m.device, dtype=torch.float32)
        m.parallel.logits(local, out=logits)
        return PrefillOutput(logits, torch.cat(features, dim=-1))

    def capture(self):
        self.compute()
        torch.npu.synchronize()
        m, s = (self.model, self.state)
        s.selections = {}
        s.freqs = {}
        self.stream = torch.npu.Stream()
        self.graph = GraphSequence()
        self.boundaries = []
        self.features = []
        pool = torch.npu.graph_pool_handle()
        h, pre = (self.h, self.pre)
        for start in range(0, len(m.decoder), 1):
            graph = torch.npu.NPUGraph()
            self.graph.graphs.append(graph)
            with torch.npu.graph(graph, stream=self.stream, pool=pool):
                if start == 0:
                    s.freqs = {True: s.rotary}
                    h, pre = (h.clone(), pre.clone())
                for block in m.decoder[start:start + 1]:
                    if block.layer in (37, 38, 39):
                        self.features.append(h.mean(dim=1))
                    h, pre = block(h, pre, s)
            self.boundaries.append((h, pre))
            torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        self.graph.graphs.append(graph)
        with torch.npu.graph(graph, stream=self.stream, pool=pool):
            self.hidden = r.collapse_norm(h[-1:], pre[-1:], m.weights['norm.weight'], m.c['norm_eps'])
            self.main_hidden = torch.cat(self.features, dim=-1)
        torch.npu.synchronize()
        self.graph.replay()
        torch.npu.synchronize()

    def __call__(self, slot):
        self.prepare(slot)
        if self.graph is None:
            windows = {i: self.model.past.windows[i] for i in range(20, 40)}
            saved = {i: window.main_kv.clone() for i, window in windows.items()}
            try:
                self.capture()
            except BaseException:
                self.close()
                raise
            finally:
                for i, value in saved.items():
                    windows[i].main_kv.copy_(value)
                torch.npu.synchronize()
            self.prepare(slot)
        self.graph.replay()
        # Callers may retain these after another slot replays the shared graph.
        m = self.model
        local = self.hidden.float() @ m.weights['head.weight'].float().t()
        logits = torch.empty((1, m.c['vocab_size']), device=m.device, dtype=torch.float32)
        m.parallel.logits(local, out=logits)
        return PrefillOutput(logits, self.main_hidden.clone())

    def close(self):
        if self.graph is not None:
            torch.npu.synchronize()
            self.graph.reset()
            self.graph = None
