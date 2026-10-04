"""Fixed Engram lane workspace, allocated before decode/graph budgeting.

Reduce-scatter send and all-gather output do not overlap in lifetime. Their
storage aliases; all operations use the same serialized compute stream.
"""
import torch


class EngramWorkspace:
    def __init__(self, length, config, parallel, device):
        self.length, self.world = length, parallel.world
        self.copies, self.dim = config['hc_mult'], config['dim']
        share = (length + self.world - 1) // self.world
        width = (self.copies + 1) * self.dim
        # One allocation, unchanged total budget; index reuses the inactive tail.
        send_n, lane_n = share*self.world*width, share*width
        local_n = share*self.copies*self.dim
        self.storage = torch.empty(send_n*4 + lane_n*6 + local_n*2,
                                   dtype=torch.uint8, device=device)
        offset = 0
        def take(shape, dtype):
            nonlocal offset
            count = 1
            for n in shape:
                count *= n
            size = count * (4 if dtype == torch.float32 else 2)
            view = self.storage[offset:offset+size].view(dtype).view(shape)
            offset += size
            return view
        self.send = take((share*self.world, width), torch.float32)
        self.kv = take((share, width), torch.float32)
        self.bkv = take((share, width), torch.bfloat16)
        self.local = take((share, self.copies, self.dim), torch.bfloat16)
        self.gathered = self.send.view(torch.bfloat16).flatten()[:share*self.world*self.copies*self.dim].view(
            share*self.world, self.copies, self.dim)

    def views(self, h):
        live, copies, dim = h.shape
        if not 0 < live <= self.length or (copies, dim) != (self.copies, self.dim):
            raise ValueError('Engram extent exceeds startup pool')
        if h.dtype != self.local.dtype or h.device != self.local.device:
            raise ValueError('Engram pool dtype/device mismatch')
        share = (live+self.world-1)//self.world
        return (self.send[:share*self.world], self.kv[:share], self.bkv[:share],
                self.local[:share], self.gathered[:share*self.world])
