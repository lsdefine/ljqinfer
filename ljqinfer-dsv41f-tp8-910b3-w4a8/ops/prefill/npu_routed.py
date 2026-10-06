"""Live-T expert dispatch; optimized HP W4A8 and released activation/combine."""
import torch
import torch_npu
from ops.prefill import residual as r
from ops.prefill.moe_units import dispatch_quant, grouped_linear_activation, grouped_linear_combine


class PackedRouted:
    def __init__(self, prefix, config, weights, parallel):
        self.p, self.w, self.parallel = prefix, weights, parallel
        # Immutable weight representation; independent of live T and storage addresses.
        self.scales = {}
        for tag in ('w13', 'w2'):
            scale = weights[prefix+'.'+tag+'.scale']
            self.scales[tag] = torch_npu.npu_trans_quant_param(scale.reshape(-1)).reshape(
                scale.shape[0], 1, scale.shape[1])
        self.topk = config['n_activated_experts']
        if config['swiglu_limit'] != 10 or self.topk != 6:
            raise ValueError('released routed kernels require top6, limit10')

    def __call__(self, x, ids, probabilities):
        p, w = self.p, self.w
        pairs, dim = ids.numel(), x.shape[-1]
        experts = w[p+'.w13.weight'].shape[0]
        inter = w[p+'.w13.scale'].shape[1] // 2
        if (dim, inter) != (5120, 288):
            raise ValueError('released routed kernels require D5120/I288')
        quantized, token_scale, counts, order = dispatch_quant(x, ids, experts=experts)
        sorted_prob = probabilities.flatten()[order].contiguous()
        quantized, token_scale = grouped_linear_activation(
            quantized, w[p+'.w13.weight'], self.scales['w13'],
            w[p+'.w13.hp_bias'], token_scale, counts, sorted_prob,
            padded_dim=w[p+'.w2.weight'].shape[1])
        result = grouped_linear_combine(quantized, w[p+'.w2.weight'], self.scales['w2'],
                                        w[p+'.w2.hp_bias'], token_scale, counts, order)
        self.parallel.sum(result)
        return result
