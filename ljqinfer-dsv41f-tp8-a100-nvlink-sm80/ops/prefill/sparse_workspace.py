"""Bounded serial sparse attention: BF16 tensor-core QK/PV, FP32 softmax.

Output is borrowed until the next call. Nothing is graph-captured: the kernel is a single
launch, so indices, positions and KV contents are plain per-call copies.
"""
import torch

from . import sparse_attn


class Workspace:
    def __init__(self, capacity, heads, dim, device, topk, window=128,
                 global_capacity=None):
        self.capacity, self.h, self.d = capacity, heads, dim
        self.gcap = capacity if global_capacity is None else max(global_capacity, capacity)
        self.w, self.topk, self.k = window, topk, window + topk
        self.local_capacity = capacity + window - 1
        self.scale = dim ** -.5
        self.calls = 0

        def alloc(shape, dtype=torch.float32):
            return torch.empty(shape, device=device, dtype=dtype)

        self.block = 512
        self.q = alloc((capacity, heads, dim), torch.bfloat16)
        self.bank = alloc((self.local_capacity + self.gcap + 1, dim), torch.bfloat16)
        self.ix = alloc((capacity, self.k), torch.long)
        self.bad = alloc((capacity, self.k), torch.bool)
        self.delta = torch.arange(window - 1, -1, -1, device=device)
        self.out = alloc((capacity, heads, dim), torch.bfloat16)
        self.sink = alloc((heads,))
        self.mod = sparse_attn.extension()

    def __call__(self, q, history, kv, global_kv, selected, positions, *,
                 start, local_start, sink, scale, ratio):
        t, old, count = len(q), len(history), len(global_kv)
        if (q.device != self.q.device or q.dtype != torch.bfloat16
                or q.shape != (t, self.h, self.d) or not 0 < t <= self.capacity
                or kv.shape != (t, self.d) or history.shape != (old, self.d)
                or old != start - local_start or not 0 <= old < self.w
                or local_start < 0 or global_kv.shape != (count, self.d)
                or count > self.gcap or selected.shape != (t, self.topk)
                or positions.shape != (t,) or sink.shape != (self.h,)
                or scale != self.scale or ratio <= 0
                or selected.dtype != torch.long or positions.dtype != torch.long
                or any(x.device != q.device for x in
                       (history, kv, global_kv, selected, positions, sink))
                or any(x.dtype != q.dtype for x in (history, kv, global_kv))):
            raise ValueError('invalid sparse workspace geometry/device/dtype/position')
        self.q[:t].copy_(q)
        self.bank[:old].copy_(history)
        self.bank[old:old+t].copy_(kv)
        base = self.local_capacity
        self.bank[base:base+count].copy_(global_kv)
        if count == 0:
            self.bank[base].zero_()
        self.sink.copy_(sink)
        swa = positions[:, None] - self.delta
        self.ix[:t, :self.w].copy_((swa-local_start).clamp(0, old+t-1))
        self.bad[:t, :self.w].copy_((swa < local_start) | (swa >= start+t))
        self.ix[:t, self.w:].copy_(selected.clamp(0, max(0, count-1)) + base)
        self.bad[:t, self.w:].copy_((selected < 0) | (selected >= count)
                                  | (selected >= (positions[:, None]+1)//ratio))
        # One kernel launch per call: capturing a CUDA graph per distinct t
        # (each new chunk length) cost sync+2 warmups+capture ~100ms per layer,
        # which dominated TTFT (py-spy 2026-09-15). Launch directly instead.
        self.run(t)
        self.calls += 1
        return self.out[:t]

    def run(self, t):
        self.mod.sparse_attn_flat(self.q[:t], self.bank, self.ix[:t],
                                  self.bad[:t], self.sink, self.out[:t],
                                  self.scale)
