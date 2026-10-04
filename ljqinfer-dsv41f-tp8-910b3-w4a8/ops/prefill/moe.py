"""Shared TP plus complete routed expert result, with released rounding."""
from ops.prefill import residual as r
from ops.prefill.npu_routed import PackedRouted


class PrefillMoE:
    def __init__(self, layer, config, weights, linear, parallel):
        self.p, self.c, self.w = f'layers.{layer}.ffn', config, weights
        self.linear, self.parallel = linear, parallel
        self.routed = PackedRouted(self.p, config, weights, parallel)

    def __call__(self, x):
        c, lin, w, p = self.c, self.linear, self.w, self.p
        act = r.swiglu(lin('s_w1', x), lin('s_w3', x), c['swiglu_limit'])
        shared = lin('s_w2', act).float()
        self.parallel.sum(shared)
        probabilities, ids = r.route(
            x, w[p+'.gate.weight'], w[p+'.gate.bias'],
            topk=c['n_activated_experts'], temperature=c['gate_temp'],
            scale=c['route_scale'], normalize=c['norm_topk_prob'], score=c['score_func'])
        return r.moe_add(self.routed(x, ids, probabilities), shared, x.dtype)
