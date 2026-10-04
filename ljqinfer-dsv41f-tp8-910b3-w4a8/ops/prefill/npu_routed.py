"""Live-T expert dispatch; optimized HP W4A8 and released activation/combine."""
import torch
import torch_npu
from ops.prefill import residual as r
from ops.prefill.w4a8 import linear


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
        # Token-major stable expert order matches the released combine ABI.
        if experts > 2**24:
            raise ValueError('expert IDs exceed exact FP32 sort range')
        order = ids.flatten().float().argsort(stable=True)
        sorted_ids = ids.flatten()[order]
        # Quantize each token once, then dispatch INT8 rows and their scales.
        raw, token_scale = r.native.dynamic_quant(x)
        rows = order // self.topk
        quantized = raw[rows].contiguous()
        token_scale = token_scale[rows].contiguous()
        counts = torch.zeros(experts, dtype=torch.int64, device=x.device)
        counts.scatter_add_(0, sorted_ids, torch.ones_like(sorted_ids))
        hidden = r.native.grouped_int4(
            quantized, w[p+'.w13.weight'], self.scales['w13'],
            w[p+'.w13.hp_bias'], token_scale, counts).to(torch.bfloat16)
        sorted_prob = probabilities.flatten()[order].contiguous()
        gated = r.native.routed_swiglu(hidden, sorted_prob)
        values = self.project('w2', gated, counts)
        inverse = torch.empty_like(order)
        inverse.scatter_(0, order, torch.arange(pairs, device=order.device))
        result = r.native.routed_combine(values, inverse)
        self.parallel.sum(result)
        return result

    def project(self, tag, x, counts):
        key = self.p+'.'+tag
        # HP GMM returns FP16, but the released activation/combine read BF16.
        return linear(x, self.w[key+'.weight'], self.scales[tag],
                      self.w[key+'.hp_bias'], counts).to(torch.bfloat16)
