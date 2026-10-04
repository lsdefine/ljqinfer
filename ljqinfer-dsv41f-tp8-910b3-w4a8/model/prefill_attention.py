"""Call-local Past adapter. No private persistent KV, frame or bound plan."""
import torch
from ops.prefill import attention as a


class AttentionState:
    def __init__(self, past, slot, start, end, phase, library, index_workspace):
        self.past, self.slot = past, slot
        self.start, self.end, self.phase = start, end, phase
        self.library = library
        self.index_workspace = index_workspace
        self.append_start = past.pos[slot] if phase == 'encoder_resume' else start
        self.banks, self.selections, self.candidates = {}, {}, None
        self.freqs = {}

    def positions(self, device):
        return torch.arange(self.start, self.end, dtype=torch.int64, device=device)

    def bank(self, layer):
        source = self.past.sources[layer]
        if layer not in self.banks:
            count = self.end // source.ratio
            kv = source.ckv_pool.read(self.slot, 0, count)
            index = source.index_pool.read(self.slot, 0, count)
            # Kernels require a non-null bank even before the first pair exists.
            if not count:
                kv = source.ckv_pool.data.new_zeros((1, kv.shape[-1]))
                index = source.index_pool.data.new_zeros((1, index.shape[-1]))
            self.banks[layer] = (kv.contiguous(), index.contiguous())
        return self.banks[layer]

    def publish(self, layer, kv, index):
        if self.phase not in ('encoder_append', 'encoder_resume'):
            raise ValueError('only encoder append publishes shared sources')
        source = self.past.sources[layer]
        offset = self.append_start // source.ratio
        if len(kv):
            source.ckv_pool.write(self.slot, offset, kv)
            source.index_pool.write(self.slot, offset, index)
        self.banks.pop(layer, None)

    def attend(self, layer, q, kv, iq, iw, sink, parallel, total_heads, topk):
        past, slot = self.past, self.slot
        view, window = past.views[layer], past.windows[layer]
        pos = self.positions(q.device)
        history = 127 if self.phase == 'encoder_append' else 0
        if history:
            old = torch.arange(self.start-history, self.start, device=q.device)
            prior = window.gather(slot, old).clone()
            prior.masked_fill_((old < 0)[:, None], 0)
            local = torch.cat((prior, kv))
        else:
            local = kv
        count = self.end // view.ratio if view.ratio else 0
        # sparse_meta = logical start represented by local[0], valid bank rows.
        meta = torch.tensor([self.start-history, count], dtype=torch.int64, device=q.device)
        if view.mode == 'swa':
            bank, selected = None, None
        else:
            bank, index = self.bank(view.kv_source_layer)
            if view.mode != 'reuse':
                valid = torch.tensor([count], device=q.device, dtype=torch.int64)
                kwargs = dict(ratio=view.ratio, total_heads=total_heads,
                              parallel=parallel, library=self.library, workspace=self.index_workspace)
                if self.phase == 'ced' and view.mode == 'full':
                    self.candidates, selected = a.select(
                        iq, iw, index, pos, valid, topk=topk, make_candidates=True, **kwargs)
                else:
                    selected = a.select(iq, iw, index, pos, valid, topk=topk,
                                        candidates=self.candidates, **kwargs)
                self.selections[view.index_source_layer] = selected
            selected = self.selections[view.index_source_layer]
        out = a.attend(q, local, bank, selected, pos, meta, sink,
                       ratio=view.ratio, library=self.library)
        lo = max(0, len(kv)-window.window, -self.start)
        window.write(slot, self.start+lo, kv[lo:])
        return out
