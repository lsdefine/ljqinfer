"""Residual block and TP Engram, driven exclusively by the incoming rows."""
import torch
from ops.prefill import residual as r
from model.engram import EngramRows
from model.prefill_linears import projection
from model.prefill_layer import PrefillAttention
from ops.prefill.moe import PrefillMoE


class PrefillBlock:
    def __init__(self, layer, config, weights, parallel, hasher=None, tables=None, library=None):
        self.layer, self.c, self.w = layer, config, weights
        self.p = f'layers.{layer}'
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
        apre, post, comb = self.mix(h, 'attn')
        y = self.attention(self.norm(h, pre, 'attn'), state)
        h = r.expand(y, h, post, comb)
        fpre, post, comb = self.mix(h, 'ffn')
        y = self.ffn(self.norm(h, apre, 'ffn'))
        return r.expand(y, h, post, comb), fpre


class PrefillEngram:
    def __init__(self, layer, config, weights, parallel, hasher, table):
        if hasher is None:
            raise ValueError('Engram requires its canonical host hasher')
        self.rows = EngramRows(hasher, layer, table, rank=parallel.rank)
        self.p, self.c, self.w = f'layers.{layer}.engram', config, weights
        self.parallel = parallel

    def __call__(self, h, prepared, workspace):
        p, w, par = self.p, self.w, self.parallel
        live, copies, dim = h.shape
        share = (live+par.world-1)//par.world
        send, kv, bkv, local, gathered = workspace.views(h)
        send[live:].zero_()
        send[:live].copy_(projection(w, p+'.wkv', prepared))
        par.scatter(send, out=kv)
        lo, hi = min(par.rank*share, live), min((par.rank+1)*share, live)
        local[hi-lo:].zero_()
        local[:hi-lo].copy_(h[lo:hi])
        weight = w[p+'.q_weight'].float() * w[p+'.k_weight'].float()
        bkv.copy_(kv)
        gated = r.engram_gate(local, bkv, weight,
                             w['engram.rotation'].float(), self.c['norm_eps'])
        par.logits(gated, out=gathered)
        return gathered[:live].contiguous()
