"""DSv4.1 projections: ratio2 encoder sources and standalone ratio1 source20."""
import torch
from ops.prefill import attention as a, residual as r
from model.prefill_linears import Linears
from model.prefill_config import rotary_frequencies
from ops.prefill.attention_units import attention_prepare, attention_finish, source_append


class PrefillAttention:
    def __init__(self, layer, config, weights, parallel, library=None):
        self.layer, self.c, self.w = layer, config, weights
        self.p, self.parallel = f'layers.{layer}.attn', parallel
        self.lin, self.library = Linears(weights, layer), library
        self.norms = {name: weights[self.p+f'.{name}_norm.weight'] for name in ('q', 'kv')}

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
        rows = source_append(x, linear=lin,
            norm_weight=self.w[self.p+'.compressor.norm.weight'],
            index_norm_weight=self.w[self.p+'.indexer.k_norm.weight'], eps=c['norm_eps'],
            ratio=source.ratio,
            carry_kv=source.kv_state[state.slot] if source.ratio == 2 else None,
            carry_score=source.score_state[state.slot] if source.ratio == 2 else None,
            start=state.append_start, frequencies=self.freqs, library=self.library)
        if rows is not None:
            state.publish(self.layer, *rows)

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
        q, kv, iq, iw = attention_prepare(x, freqs, linear=lin,
            norms=self.norms,
            eps=c['norm_eps'], heads=c['n_heads']//world, head_dim=c['head_dim'],
            index_heads=c['index_n_heads']//world, index_dim=c['index_head_dim'],
            needs_index=view.mode in ('source', 'full', 'reindex'), library=self.library)
        out = state.attend(self.layer, q, kv, iq, iw, self.w[self.p+'.attn_sink'],
                           self.parallel, c['index_n_heads'], c['index_topk'])
        return attention_finish(out, freqs, linear=lin, comm=self.parallel,
                                dtype=x.dtype, library=self.library)
