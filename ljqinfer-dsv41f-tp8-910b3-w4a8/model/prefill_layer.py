"""DSv4.1 projections: ratio2 encoder sources and standalone ratio1 source20."""
import torch
from ops.prefill import attention as a, residual as r
from model.prefill_linears import Linears
from model.prefill_config import rotary_frequencies


class PrefillAttention:
    def __init__(self, layer, config, weights, parallel, library=None):
        self.layer, self.c, self.w = layer, config, weights
        self.p, self.parallel = f'layers.{layer}.attn', parallel
        self.lin, self.library = Linears(weights, layer), library

    def norm(self, x, name):
        return r.rms(x, self.w[self.p+name], self.c['norm_eps'])

    def freqs(self, positions):
        # Tables are proportional to the visible end, not to model max capacity.
        table = rotary_frequencies(self.c, self.layer, self.end, device=positions.device)
        return a.frequencies(torch.view_as_real(table), positions, library=self.library)

    def source(self, x, state):
        c, lin = self.c, self.lin
        source = state.past.sources[self.layer]
        self.end = state.end
        x = x[state.append_start-state.start:]
        values = lin('c_wkv', x).float()
        if source.ratio == 2:
            scores = lin('c_wgate', x).float()
            raw = a.compress(values, scores, source.kv_state[state.slot],
                             source.score_state[state.slot], state.append_start, library=self.library)
        else:
            raw = values
        if not len(raw):
            return
        pos = torch.arange(state.append_start//source.ratio, state.end//source.ratio,
                           device=x.device, dtype=torch.int64) * source.ratio
        freqs = self.freqs(pos)
        latent = self.norm(raw.to(torch.bfloat16), '.compressor.norm.weight').to(torch.bfloat16)
        index = self.norm(lin('i_wk', latent), '.indexer.k_norm.weight')
        latent = a.qdq(a.rope(latent, freqs, library=self.library),
                       'compressed', library=self.library)
        index = a.qdq(a.rope(index, freqs, library=self.library), 'index', library=self.library)
        state.publish(self.layer, latent, index)

    def __call__(self, x, state):
        c, lin, world = self.c, self.lin, self.parallel.world
        self.end = state.end
        view = state.past.views[self.layer]
        if view.mode in ('source', 'full') and state.phase in ('encoder_append', 'encoder_resume'):
            self.source(x, state)
        compressed = bool(c['compress_ratios'][self.layer])
        if compressed not in state.freqs:
            state.freqs[compressed] = self.freqs(state.positions(x.device))
        freqs = state.freqs[compressed]
        qr = self.norm(lin('wq_a', x), '.q_norm.weight')
        q = lin('wq_b', qr).view(len(x), c['n_heads']//world, c['head_dim'])
        q = a.rope(q, freqs, library=self.library)
        kv = a.rope(self.norm(lin('wkv', x), '.kv_norm.weight'), freqs, library=self.library)
        kv = a.qdq(kv, 'local', library=self.library)
        iq = iw = None
        if view.mode in ('source', 'full', 'reindex'):
            iq = lin('i_wq_b', qr).view(len(x), c['index_n_heads']//world, c['index_head_dim'])
            iq = a.qdq(a.rope(iq, freqs, library=self.library), 'index', library=self.library)
            iw = lin('i_weights', x).float()
        out = state.attend(self.layer, q, kv, iq, iw, self.w[self.p+'.attn_sink'],
                           self.parallel, c['index_n_heads'], c['index_topk'])
        out = a.rope(out, freqs, inverse=True, library=self.library)
        projected = lin('wo_a', out.flatten(1))
        result = lin('wo_b', projected).float()
        self.parallel.sum(result)
        return result.to(x.dtype)
