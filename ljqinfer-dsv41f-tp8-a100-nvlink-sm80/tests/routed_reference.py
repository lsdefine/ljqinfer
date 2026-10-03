"""Slow comparison oracle, never imported by production."""
import torch
from ops.prefill.gemm import packed_linear
from ops.prefill import residual as r


class BankRouted:
    """Test-only canonical EP bank oracle. Inputs replicated, experts rank-owned.

    Compute only selected local experts, sum FP32 outputs across EP8. A future
    dispatch/grouped-GEMM/combine may replace this entire boundary.
    """
    def __init__(self, prefix, config, weights, parallel):
        self.prefix, self.c, self.w, self.parallel = prefix, config, weights, parallel

    def __call__(self, x, ids, probabilities):
        count = self.c['n_routed_experts']//8
        out = torch.zeros_like(x,dtype=torch.float32)
        base = self.prefix+'.local_experts.'
        # Mapping may load bounded layer units from wcache on demand.
        for local in range(count):
            token, choice = torch.where(ids == self.parallel.rank*count+local)
            if not len(token):
                continue
            z = x[token]
            w13, s13 = self.w[base+'w13.weight'][local], self.w[base+'w13.scale'][local]
            y = r.swiglu(packed_linear(z,w13[0],s13[0]),
                          packed_linear(z,w13[1],s13[1]), self.c['swiglu_limit'],
                          probabilities[token,choice,None])
            y = packed_linear(y,self.w[base+'w2.weight'][local],
                              self.w[base+'w2.scale'][local])
            out.index_add_(0,token,y.float())
        self.parallel.sum(out)
        return out
