"""Device-only Engram; prepared rows must be ready on the current stream.

h BF16[T,H,D], prepared BF16[T,F]. workspace supplies padded send, KV,
BF16 KV, local residual and all-gather buffers. Output borrows gathered
storage and remains valid until workspace reuse. Hash/table IO stays outside.
"""
from . import residual as r


def engram_apply(h, prepared, *, project, weight, rotation,
                 eps, comm, workspace):
    par = comm
    live, copies, dim = h.shape
    share = (live+par.world-1)//par.world
    send, kv, bkv, local, gathered = workspace.views(h)
    send[live:].zero_()
    send[:live].copy_(project(prepared))
    par.scatter(send, out=kv)
    lo, hi = min(par.rank*share, live), min((par.rank+1)*share, live)
    local[hi-lo:].zero_()
    local[:hi-lo].copy_(h[lo:hi])
    bkv.copy_(kv)
    gated = r.engram_gate(local, bkv, weight,
                         rotation, eps)
    par.logits(gated, out=gathered)
    return gathered[:live].contiguous()
