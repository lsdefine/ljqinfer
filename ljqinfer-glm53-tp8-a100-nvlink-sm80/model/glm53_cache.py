"""GLM53 state geometry over the DSv4.1 prefix cache/host arena.
Target, index and rank-local draft KV share one prefill commit boundary.
Cold transactions stay serial; bounded writeback overlaps subsequent computation.
"""
import logging
import torch
from strategy.cold_kv import ColdCache, Field, CacheCapacityError
from strategy.host_arena import HostArena
from strategy.async_writeback import AsyncWriteback
from model.glm53_cold_transfer import ColdTransfer


# Total host cold-cache bytes across all ranks (same setting name as A2).
COLD_CACHE_BYTES = 88_370_380_800
LOGITS_KEY = ('logits', 0)


class PrefixState:
    def __init__(self, engine):
        self.engine = engine
        self.namespace = 'glm53-int4-dflash2-tp8-v3-endpoint-logits'
        self.resident = []
        self.last_logits = None
        self.fields = {}
        for i in range(engine.kv.shape[0]):
            self.fields[('target', i)] = engine.kv[i].view(-1, 576)
        for i in range(engine.index.shape[0]):
            self.fields[('index', i)] = engine.index[i].view(-1, 128)
        pool = engine.mtp_kv
        # The B1 engine leases this slot for its entire lifetime, in page order.
        assert pool.host_page_table[0] == list(range(pool.logical_pages))
        for name in ('k', 'v'):
            data = getattr(pool, name)
            for i in range(pool.layers):
                self.fields[(name, i)] = data[i].view(-1, pool.kv_heads, pool.head_dim)
        self.transfer = ColdTransfer(self.fields, engine.device)
        schema = [Field(k, (None, *self.fields[k].shape[1:]), self.fields[k].dtype)
                  for k in self.transfer.local_keys]
        row_bytes = sum(self.fields[k][0].numel() * self.fields[k].element_size()
                        for k in self.transfer.local_keys)
        # Five contexts (~82.3 GiB total at capacity=147520), allocated on demand.
        # Public layers have one CPU owner; draft heads retain rank-local shards.
        total_row_bytes = sum(
            value[0].numel() * value.element_size() *
            (1 if key in self.transfer.owners else self.transfer.world)
            for key, value in self.fields.items())
        budget = COLD_CACHE_BYTES * row_bytes // total_row_bytes
        self.arena = HostArena(budget)
        warm = self.arena.alloc(512)
        self.arena.free(warm, 512)
        # Endpoint payload, atomically accounted/evicted with its KV.
        schema.append(Field(LOGITS_KEY, (None,), torch.float32))
        self.cold = ColdCache(schema, budget, namespace=self.namespace,
                              pin_memory=True, shm=self.arena)
        self.copy_stream = torch.cuda.Stream(device=engine.device)
        self.writer = AsyncWriteback(max_pending=2)

    def for_engine(self, engine):
        """Bind the same cold store to a request's physical state, not its batch.

        Current GLM leases are contiguous. Do not silently treat a future
        non-contiguous lease as a contiguous tensor view.
        """
        import copy
        bound = copy.copy(self)
        bound.engine = engine
        bound.resident = []
        bound.last_logits = None
        bound.fields = {}
        start, pages = engine.cache_page_span
        begin, end = start * 64, (start + pages) * 64
        for i in range(engine.kv.shape[0]):
            bound.fields[('target', i)] = engine.kv[i].view(-1, 576)[begin:end]
        for i in range(engine.index.shape[0]):
            bound.fields[('index', i)] = engine.index[i].view(-1, 128)[begin:end]
        pool = engine.mtp_kv
        mapping = pool.host_page_table[0][:pages]
        if mapping != list(range(mapping[0], mapping[0] + pages)):
            raise ValueError('non-contiguous GLM draft lease')
        begin, end = mapping[0] * pool.page_size, (mapping[-1] + 1) * pool.page_size
        for name in ('k', 'v'):
            for i in range(pool.layers):
                bound.fields[(name, i)] = getattr(pool, name)[i].view(
                    -1, pool.kv_heads, pool.head_dim)[begin:end]
        # Aliasing request views invalidate the base engine's hot-state identity.
        self.invalidate()
        return bound

    def restore(self, tokens):
        # Drain before reusing source rows or inspecting endpoint payloads.
        self.drain()
        hot = 0
        for a, b in zip(tokens, self.resident):
            if a != b:
                break
            hot += 1
        hot_logits = self.last_logits if hot == len(self.resident) else None
        if hot == len(tokens) and hot_logits is None:
            hot = max(0, hot - 1)
        self.last_logits = None
        with self.cold.lookup(tokens, namespace=self.namespace) as lease:
            cold_count = lease.token_count
            endpoint_logits = None
            if cold_count == len(tokens):
                for entry_id, begin, end in reversed(self.cold.spans(lease)):
                    if end == cold_count == self.cold.entries[entry_id].end:
                        value = self.cold.storage[LOGITS_KEY][entry_id]
                        if value.numel():
                            endpoint_logits = value
                        break
                # Internal prefixes have KV but no matching endpoint logits.
                if endpoint_logits is None:
                    cold_count = max(0, cold_count - 1)
            hot, cold = self.transfer.common_prefix(hot, cold_count)
            count = max(hot, cold)
            source = 'hot' if hot >= cold else 'cold'
            if source == 'cold':
                lease.end = count
                self.transfer.restore(self.fields, self.cold, lease, self.copy_stream)
            if count == len(tokens):
                self.last_logits = (hot_logits if source == 'hot' else
                    endpoint_logits.to(self.engine.device, non_blocking=False))
            self.engine.length = count
            self.engine.pending = None
            self.engine.mtp_kv.lengths[0] = count
        self.resident = list(tokens[:count])
        return count, source if count else 'miss'

    def publish(self, tokens, logits=None):
        count = self.engine.length
        assert count == self.engine.mtp_kv.lengths[0] and count <= len(tokens)
        tokens = list(tokens[:count])
        self.resident = tokens
        # Own the endpoint before another forward can reuse its storage.
        self.last_logits = None if logits is None else logits.detach().float().clone()
        ready = torch.cuda.Event()
        ready.record(torch.cuda.current_stream(self.engine.device))
        self.writer.submit(self._publish_async, tuple(tokens), ready, self.last_logits)

    def _publish_async(self, tokens, ready, logits):
        with torch.inference_mode(), torch.cuda.device(self.engine.device), torch.cuda.stream(self.copy_stream):
            self.copy_stream.wait_event(ready)
            try:
                self._store(tokens, logits)
            except CacheCapacityError as exc:
                # Store context has settled DMA and rolled back all staging.
                # Cold publication is optional; active GPU KV remains valid.
                logging.getLogger(__name__).warning(
                    'cold cache admission skipped after eviction: tokens=%d, %s',
                    len(tokens), exc)

    def _store(self, tokens, logits):
        # Never consult mutable engine.length/page maps in the worker.
        count = len(tokens)
        with self.cold.lookup(tokens, namespace=self.namespace) as parent:
            start = parent.token_count
            if start == count:
                return
            with self.cold.prepare(parent, tokens[start:]) as store:
                for key in self.cold.fields:
                    if key == LOGITS_KEY:
                        value = logits if logits is not None else torch.empty(0, dtype=torch.float32)
                    else:
                        value = self.fields[key][start:count]
                    store.write(key, value)
                store.commit()

    def drain(self):
        self.writer.drain()

    def invalidate(self):
        self.drain()
        self.resident = []
        self.last_logits = None

    def clear(self):
        self.invalidate()
        self.cold.clear()

    def close(self):
        self.writer.close()
