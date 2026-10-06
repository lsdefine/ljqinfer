"""Residual block and TP Engram, driven exclusively by the incoming rows."""
import torch
from functools import partial
from ops.prefill import residual as r
from model.engram import EngramRows
from model.prefill_linears import projection
from model.prefill_layer import PrefillAttention
from ops.prefill.moe import PrefillMoE
from ops.prefill.hc import hc_prepare, hc_finish
from ops.prefill.engram import engram_apply


class PrefillBlock:
    def __init__(self, layer, config, weights, parallel, hasher=None, tables=None, library=None):
        self.layer, self.c, self.w = layer, config, weights
        self.p = f'layers.{layer}'
        self.hc = {name: partial(hc_prepare,
            fn=weights[f'{self.p}.hc_{name}_fn'],
            scale=weights[f'{self.p}.hc_{name}_scale'],
            base=weights[f'{self.p}.hc_{name}_base'],
            norm_weight=weights[f'{self.p}.{name}_norm.weight'],
            norm_eps=config['norm_eps'], hc_eps=config['hc_eps'],
            iters=config['hc_sinkhorn_iters']) for name in ('attn', 'ffn')}
        self.attention = PrefillAttention(layer, config, weights, parallel, library)
        self.ffn = PrefillMoE(layer, config, weights, self.attention.lin, parallel)
        self.engram = (PrefillEngram(layer, config, weights, parallel, hasher, tables[layer])
                       if layer in config['engram_layer_ids'] else None)

    def mix(self, h, name):
        p, c, w = self.p+'.hc_'+name, self.c, self.w
        return r.mixes(h, w[p+'_fn'].float(), w[p+'_scale'].float(), w[p+'_base'].float(),
                       norm_eps=c['norm_eps'], hc_eps=c['hc_eps'], iters=c['hc_sinkhorn_iters'])

    def norm(self, h, pre, name):
        return r.collapse_norm(h, pre, self.w[self.p+'.'+name+'_norm.weight'], self.c['norm_eps'])

    def __call__(self, h, pre, state):
        # The generated PRE gates the NEXT sublayer, not the producer itself.
        x, apre, post, comb = self.hc['attn'](h, pre)
        h = hc_finish(self.attention(x, state), h, post, comb)
        x, fpre, post, comb = self.hc['ffn'](h, apre)
        return hc_finish(self.ffn(x), h, post, comb), fpre


class PrefillEngram:
    def __init__(self, layer, config, weights, parallel, hasher, table):
        if hasher is None:
            raise ValueError('Engram requires its canonical host hasher')
        self.rows = EngramRows(hasher, layer, table, rank=parallel.rank)
        self.p, self.c, self.w = f'layers.{layer}.engram', config, weights
        self.parallel = parallel
        self.project = partial(projection, weights, self.p+'.wkv')
        self.weight = weights[self.p+'.q_weight'].float() * weights[self.p+'.k_weight'].float()
        self.rotation = weights['engram.rotation'].float()

    def __call__(self, h, prepared, workspace):
        return engram_apply(h, prepared, project=self.project,
            weight=self.weight, rotation=self.rotation, eps=self.c['norm_eps'],
            comm=self.parallel, workspace=workspace)
