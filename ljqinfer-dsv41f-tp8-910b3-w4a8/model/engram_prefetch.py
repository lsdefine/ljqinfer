"""One-layer-ahead Engram upload; pinned sources live until copies finish."""
from concurrent.futures import ThreadPoolExecutor, wait
import torch


class EngramPrefetch:
    def __init__(self, blocks, *, device, length):
        self.entries = {b.layer: b.engram.rows for b in blocks if b.engram is not None}
        self.pool = ThreadPoolExecutor(max_workers=max(1, len(self.entries)))
        self.uploader = ThreadPoolExecutor(max_workers=1)
        self.device = device
        self.stream = torch.npu.Stream(device=device)
        self.pending, self.ready, self.sources = {}, {}, []
        self.layers = iter(())
        # The released encoder has two Engram layers. Fixed upload slots are
        # owned by this lane, not the allocator; never grow them in submit().
        self.buffers = {}
        for b in blocks:
            if b.engram is None:
                continue
            width = b.engram.w[b.engram.p + '.wkv.weight'].shape[1]
            self.buffers[b.layer] = dict(
                host=torch.empty((length, width), dtype=torch.bfloat16, pin_memory=True),
                device=torch.empty((length, width), dtype=torch.bfloat16, device=device),
                copied=torch.npu.Event(), consumed=torch.npu.Event(), used=False)
        self.borrowed = set()

    def submit(self, *, slot, start, tokens, history_tokens=()):
        self.drain()
        for layer, lookup in self.entries.items():
            self.pending[layer] = self.pool.submit(lookup, slot, start, tuple(tokens), tuple(history_tokens))
        self.layers = iter(self.entries)
        self._advance()

    @torch.inference_mode()
    def _upload(self, layer, future):
        torch.npu.set_device(self.device)
        source = future.result().flatten(1)
        buf = self.buffers[layer]
        if source.shape[1] != buf['host'].shape[1] or len(source) > len(buf['host']):
            raise ValueError('Engram upload exceeds startup pool')
        host = buf['host'][:len(source)]
        host.copy_(source)
        with torch.npu.stream(self.stream):
            if buf['used']:
                self.stream.wait_event(buf['consumed'])
            rows = buf['device'][:len(source)]
            rows.copy_(host, non_blocking=True)
            buf['copied'].record(self.stream)
        return rows, buf['copied']

    def _advance(self):
        layer = next(self.layers, None)
        if layer is not None:
            self.ready[layer] = self.uploader.submit(self._upload, layer, self.pending[layer])

    def wait(self, layer):
        rows, done = self.ready.pop(layer).result()
        consumer = torch.npu.current_stream(self.device)
        consumer.wait_event(done)
        self.borrowed.add(layer)
        self._advance()
        return rows

    def drain(self):
        # Includes partial/failed consumers. Next upload waits on their stream,
        # so persistent storage can be reused without record_stream allocation.
        for layer in self.borrowed:
            buf = self.buffers[layer]
            buf['consumed'].record(torch.npu.current_stream(self.device))
            buf['used'] = True
        self.borrowed.clear()
        futures = [*self.pending.values(), *self.ready.values()]
        if futures:
            wait(futures)
        try:
            for future in futures:
                future.result()
        finally:
            # Also joins copies from already-consumed futures, including errors.
            self.stream.synchronize()
            self.pending.clear()
            self.ready.clear()
            self.sources.clear()
            self.layers = iter(())

    def close(self):
        try:
            self.drain()
        finally:
            self.uploader.shutdown(wait=True)
            self.pool.shutdown(wait=True)
