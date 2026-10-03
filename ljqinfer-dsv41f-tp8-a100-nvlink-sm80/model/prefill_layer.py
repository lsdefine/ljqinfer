"""Prefill-only layer computation. `linear(name,x)` is the GEMM boundary.

Weights are rank-local in the established ABI. No token-count-based decode
branch. BF16/synthetic execution uses DenseLinear; packed GEMMs are injected
explicitly and never silently dequantized into full-model BF16 copies.
"""
import torch
import torch.nn.functional as F
from ops.prefill import attention as a, residual as r
from ops.prefill.quant import fp4_roundtrip
from model.prefill_attention import attend


class DenseLinear:
    """Small dense mathematical baseline, explicitly rejects packed weights."""
    def __init__(self, weights):
        self.weights = weights

    def __call__(self, name, x):
        w = self.weights[name+'.weight']
        if w.dtype not in (torch.float32, torch.bfloat16, torch.float16):
            raise TypeError('packed weight requires an explicit quantized GEMM binding')
        return F.linear(x, w)


class PrefillAttention:
    def __init__(self, layer, config, weights, linear, freqs, *, world=1, reduce_sum=None):
        self.layer, self.c, self.w, self.linear = layer, config, weights, linear
        self.freqs, self.world, self.reduce_sum = freqs, world, reduce_sum
        self.swa_workspace = None
        self.sparse_workspace = None
        if world != 1 and reduce_sum is None:
            raise ValueError('TP requires explicit sum collective')

    def project_global(self, x, *, past, slot, start):
        """Source-only CED projection: no query, local KV, output GEMM or MoE."""
        c, w, lin = self.c, self.w, self.linear
        eps = c['norm_eps']
        p = f'layers.{self.layer}.attn'
        view = past.views[self.layer]
        if view.mode != 'full':
            raise ValueError('only a KV source can project global rows')
        source = past.sources[self.layer]
        cp = p+'.compressor'
        values = lin(cp+'.wkv', x.float() if view.ratio == 2 else x)
        scores = lin(cp+'.wgate', x.float()) if view.ratio == 2 else None
        raw, origin = source.fold(slot, start, values, scores)
        if not len(raw):
            # Ratio-2 chunks that start on an even position complete no group:
            # compress() already folded the rows into the caller-owned carry, so
            # there is no global row to normalise, rotate or publish yet.
            return None, None
        latent = r.rms(raw.to(x.dtype), w[cp+'.norm.weight'], eps)
        cf = self.freqs[origin*view.ratio:(origin+len(latent))*view.ratio:view.ratio]
        ip = p+'.indexer'
        ik = fp4_roundtrip(a.rope(r.rms(lin(ip+'.wk', latent), w[ip+'.k_norm.weight'], eps), cf), block=32, e4m3_scale=False)
        ck = fp4_roundtrip(a.rope(latent, cf), block=16, e4m3_scale=True)
        return ck, ik

    def __call__(self, x, *, past, slot, start, scratch):
        c, w, lin = self.c, self.w, self.linear
        eps, d = c['norm_eps'], c['head_dim']
        p = f'layers.{self.layer}.attn'
        view = past.views[self.layer]
        t = len(x)
        f = self.freqs[start:start+t]
        qr = r.rms(lin(p+'.wq_a', x), w[p+'.q_norm.weight'], eps)
        q = a.rope(lin(p+'.wq_b', qr).unflatten(-1, (c['n_heads']//self.world, d)), f)
        kv = a.fp8_roundtrip(a.rope(r.rms(lin(p+'.wkv', x), w[p+'.kv_norm.weight'], eps), f))
        ck, ik, iq, iw = None, None, None, None
        if view.mode == 'full' and not scratch.read_global_only:
            ck, ik = self.project_global(x, past=past, slot=slot, start=start)
        elif view.mode == 'full' and view.ratio == 2:
            # Reconstruct only hot ring carry, never complete cached global rows.
            source = past.sources[self.layer]
            ring = 2 * view.ratio
            tail_start = max(0, t-ring)
            rows = torch.arange(start+tail_start, start+t, device=x.device) % ring
            cp = p+'.compressor'
            source.write_res(slot,
                             lin(cp+'.wkv', x[tail_start:].float()),
                             lin(cp+'.wgate', x[tail_start:].float()), rows)

        if view.mode in ('full', 'reindex'):
            ip = p+'.indexer'
            iq = lin(ip+'.wq_b', qr).unflatten(-1, (c['index_n_heads']//self.world,c['index_head_dim']))
            iq = a.rope(iq, f)
            iq = fp4_roundtrip(iq, block=32, e4m3_scale=False)
            iw = lin(ip+'.weights_proj', x).float()
        out = attend(past=past,slot=slot,start=start,layer=self.layer,q=q,kv=kv,
            sink=w[p+'.attn_sink'],scale=d**-.5,scratch=scratch,compressed=ck,index_keys=ik,
            index_q=iq,index_weight=iw,topk=c['index_topk'],total_index_heads=c['index_n_heads'],
            reduce_scores=self.reduce_sum,
            swa_workspace=self.swa_workspace, sparse_workspace=self.sparse_workspace)
        out = a.rope(out, f, inverse=True)
        groups, rank = c['o_groups']//self.world, c['o_lora_rank']
        grouped = out.reshape(t, groups, -1)
        wa = w[p+'.wo_a.weight']
        if wa.dtype not in (torch.bfloat16, torch.float32, torch.float16):
            raise TypeError('wo_a must be prepacked dense or bound to grouped GEMM')
        from ops.prefill.gemm import grouped_linear
        out = grouped_linear(grouped, wa.reshape(groups,rank,-1)).flatten(-2)
        out = lin(p+'.wo_b', out)
        if self.reduce_sum is not None:
            self.reduce_sum(out)
        return out
