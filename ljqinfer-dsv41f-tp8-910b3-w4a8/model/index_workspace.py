"""Encoder index scratch: fixed query tile, shared inactive Engram arena tail.

The gathered residual remains live in the send prefix through attention.
Only its disjoint tail may be overwritten, on the serialized compute stream.
Narrow-key batches reuse spare capacity; no runtime workspace allocation.
"""
import torch


class IndexWorkspace:
    tile = 32
    group = 128

    def __init__(self, engram, capacity, config, world):
        self.heads = config['index_n_heads'] // world
        self.dim = config['index_head_dim']
        self.topk = config['index_topk']
        self.width = max(32, ((capacity // 2) + 31) // 32 * 32)
        storage = engram.storage
        offset = engram.gathered.numel() * engram.gathered.element_size()
        def take(shape, dtype):
            nonlocal offset
            size = 1
            for n in shape:
                size *= n
            unit = {torch.float32: 4, torch.bfloat16: 2, torch.int64: 8}[dtype]
            offset = (offset + 511) // 512 * 512
            end = offset + size * unit
            if end > storage.numel():
                raise ValueError('encoder index scratch exceeds startup shared pool')
            view = storage[offset:end].view(dtype).view(shape)
            offset = end
            return view
        self.dot = take((self.tile * self.heads * self.width,), torch.float32)
        self.score = take((self.group * self.width,), torch.float32)
        self.bank = take((self.width, self.dim), torch.bfloat16)
        # Use only the inactive arena tail; full-width buffers never grow.
        available = storage.numel() - ((offset + 511) // 512 * 512) - 511
        self.order_rows = max(self.group, min(2048, available // (12 * self.topk)))
        self.values = take((self.order_rows * self.topk,), torch.float32)
        self.ids = take((self.order_rows * self.topk,), torch.int64)

    def layout(self, rows, width):
        if rows >= 2048 and 0 < width <= 4096 and 128*self.heads*width <= self.dot.numel():
            group = min(2048, self.order_rows, self.score.numel() // width) // 128 * 128
            return 128, group
        return self.tile, self.group

    def views(self, rows, heads, width, topk, tile=32):
        if not (0 < rows <= self.order_rows and heads == self.heads and
                0 < width <= self.width and 0 < topk <= self.topk and
                tile in (32, 128) and min(rows,tile)*heads*width <= self.dot.numel() and
                rows*width <= self.score.numel()):
            raise ValueError('index extent exceeds startup workspace contract')
        return (self.dot[:min(rows,tile)*heads*width].view(min(rows,tile)*heads, width),
                self.score[:rows*width].view(rows, width),
                self.values[:rows*topk].view(rows, topk),
                self.ids[:rows*topk].view(rows, topk))
