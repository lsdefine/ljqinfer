"""Checkpoint naming only: live inputs determine every projection's T."""
from ops.prefill.gemm import linear

NAMES = {
    'wq_a': 'attn.wq_a', 'wq_b': 'attn.wq_b', 'wkv': 'attn.wkv',
    'wo_a': 'attn.wo_a', 'wo_b': 'attn.wo_b',
    'c_wkv': 'attn.compressor.wkv', 'c_wgate': 'attn.compressor.wgate',
    'i_wk': 'attn.indexer.wk', 'i_wq_b': 'attn.indexer.wq_b',
    'i_weights': 'attn.indexer.weights_proj',
    's_w1': 'ffn.shared_experts.w1', 's_w3': 'ffn.shared_experts.w3',
    's_w2': 'ffn.shared_experts.w2',
}


def projection(weights, name, x):
    return linear(x, weights[name + '.weight'], weights.get(name + '.scale'))


class Linears:
    def __init__(self, weights, layer):
        self.weights, self.prefix = weights, f'layers.{layer}.'

    def __call__(self, short, x):
        name = self.prefix + NAMES[short]
        if short in ('c_wkv', 'c_wgate') and self.prefix + NAMES['c_wgate'] + '.weight' in self.weights:
            return projection(self.weights, name, x.float())
        return projection(self.weights, name, x)
