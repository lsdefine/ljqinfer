"""Synchronous ljqinfer orchestration, independent of model computation.

One step processes at most one chunk. Callers may interleave live sessions;
this is not a threaded scheduler, decoding loop or protocol server.
"""
from dataclasses import dataclass, field
from threading import RLock
from time import perf_counter
from model.model_api import PrefillCancelled
from model.cold import fields_for, restore_prefix_tp, store_chunk


@dataclass
class QueryMetrics:
    """What one prefill cost, published by the strategy and copied by servers.

    Wall clock around orchestration calls, not device time: the strategy owns
    no stream and must not synchronize one to measure it. Chunk compute is the
    dominant term and its launch queue drains inside the cold-cache store
    that follows it, so the wall clock tracks device time closely enough to
    compare chunk sizes. Without that store the queue may still be draining
    when the reading is taken.

    Chunk detail is kept per chunk rather than averaged because the first
    chunk of a process pays one-off warmup and would otherwise be smeared
    across the rate a caller reads; steady_prefill_tps excludes it.
    """

    input_tokens: int = 0
    cache_hit_tokens: int = 0
    prefill_tokens: int = 0
    chunks: int = 0
    cache_stored_blocks: int = 0
    cache_lookup_seconds: float = 0.0
    cache_load_seconds: float = 0.0
    model_prefill_seconds: float = 0.0
    cache_store_seconds: float = 0.0
    finish_prefill_seconds: float = 0.0
    strategy_seconds: float = 0.0
    chunk_seconds: list = field(default_factory=list)
    chunk_tokens: list = field(default_factory=list)

    @property
    def cache_hit_rate(self):
        return self.cache_hit_tokens / self.input_tokens if self.input_tokens else 0.0

    @property
    def first_chunk_seconds(self):
        return self.chunk_seconds[0] if self.chunk_seconds else 0.0

    @property
    def model_prefill_tps(self):
        s = self.model_prefill_seconds
        return self.prefill_tokens / s if s > 0 else 0.0

    @property
    def steady_prefill_tps(self):
        """Rate after warmup. 0.0 when one chunk carries the whole prompt."""
        if len(self.chunk_seconds) < 2:
            return 0.0
        seconds = sum(self.chunk_seconds[1:])
        return sum(self.chunk_tokens[1:]) / seconds if seconds > 0 else 0.0

    @property
    def effective_prefill_tps(self):
        """Prompt tokens served per second, counting cache hits as served."""
        s = self.strategy_seconds
        return self.input_tokens / s if s > 0 else 0.0


class Strategy:
    def __init__(self, model, cold_kv=None, *, namespace, rank=0, world=1,
                 device=None, ctrl=None, cold=None):
        self.model, self.cold_kv, self.namespace = model, cold_kv, namespace
        self.rank, self.world, self.ctrl = rank, world, ctrl
        self.device = device
        # Rank 0 alone holds the cache, so `cold_kv is not None` differs across
        # ranks and cannot gate a collective.  `cold` is derived from a shared
        # constant instead, giving every rank the same answer.
        self.cold = (cold_kv is not None) if cold is None else bool(cold)
        if self.cold and rank == 0 and cold_kv is None:
            raise ValueError('rank 0 must own the cold cache')
        self.lock = RLock()
        if cold_kv is not None:
            if cold_kv.namespace != namespace:
                raise ValueError('cache namespace mismatch')
            if cold_kv.fields != {f.key: f for f in fields_for(model.past)}:
                raise ValueError('cache layout mismatch')

    def open(self, input_ids, *, cancel=None):
        tokens = tuple(int(t) for t in input_ids)
        if not tokens or len(tokens) > self.model.past.row_cap:
            raise ValueError('input length must be 1..row_cap')
        metrics = QueryMetrics(input_tokens=len(tokens))
        opened_at = perf_counter()
        with self.lock:
            if cancel is not None and cancel.is_set():
                raise PrefillCancelled()
            pool = self.model.past
            if not pool.free_slots:
                raise MemoryError('no free sequence slots')
            slot = pool.alloc()
            lease = None
            try:
                replayed = 0
                if self.cold:
                    mark = perf_counter()
                    if self.cold_kv is not None:
                        lease = self.cold_kv.lookup(tokens, namespace=self.namespace)
                        keep = max(0, lease.token_count - pool.ring)
                    else:
                        keep = 0
                    if lease is not None and keep < lease.token_count:
                        # KV inside the sliding window is reduced element by
                        # element, so it must come from this pass with a single
                        # layout: a restored window reduces in another order and
                        # the 1-ulp bf16 drift is amplified by 40 layers into
                        # different greedy tokens. Older tokens only take part
                        # in compressed form, which is layout independent.
                        lease.close()
                        lease = self.cold_kv.lookup(tokens[:keep], namespace=self.namespace)
                    metrics.cache_lookup_seconds += perf_counter() - mark
                    mark = perf_counter()
                    # Followers hold no lease, so the hit length comes back
                    # from the broadcast rather than from the cache.
                    hit = restore_prefix_tp(
                        self.cold_kv, pool, slot, lease, rank=self.rank,
                        world=self.world, device=self.device, group=self.ctrl)
                    if hit:
                        replayed = self.model.replay_prefix(
                            slot, tokens[:hit], cancel=cancel)
                    metrics.cache_load_seconds += perf_counter() - mark
                if cancel is not None and cancel.is_set():
                    raise PrefillCancelled()
                session = PrefillSession(self, slot, tokens, lease, cancel, metrics)
                session.replayed_tokens = replayed
                metrics.cache_hit_tokens = session.hit_tokens
                metrics.strategy_seconds += perf_counter() - opened_at
                return session
            except BaseException:
                if lease is not None:
                    lease.close()
                pool.release(slot)
                raise


class PrefillSession:
    def __init__(self, strategy, slot, tokens, lease, cancel, metrics=None):
        self.strategy, self.slot, self.tokens = strategy, slot, tokens
        self.lease, self.cancel = lease, cancel
        self.metrics = metrics if metrics is not None else QueryMetrics(
            input_tokens=len(tokens))
        self.closed = False
        self.hit_tokens = strategy.model.past.pos[slot]
        self.matched_tokens = lease.matched_tokens if lease is not None else 0
        self.computed_tokens, self.chunks = 0, 0
        # Attention cache does NOT contain logits or model continuation sidecars.
        # Replay rebuilds hot state only; a full hit does not invent logits.
        self.output = None

    def _live(self):
        if self.closed:
            raise RuntimeError('session is closed')

    @property
    def position(self):
        self._live()
        return self.strategy.model.past.pos[self.slot]

    @property
    def done(self):
        return self.position == len(self.tokens)

    def step(self):
        s = self.strategy
        with s.lock:
            self._live()
            try:
                if self.cancel is not None and self.cancel.is_set():
                    raise PrefillCancelled()
                if self.done:
                    return None
                start = self.position
                if (0 < start < s.model.past.ring
                        and s.model.prefill_chunk_tokens >= s.model.past.ring):
                    # See Strategy.open: an incremental chunk on a partially
                    # filled ring is not numerically equivalent to one pass,
                    # so recompute the whole prefix instead of extending it.
                    s.model.past.set_pos(self.slot, 0)
                    if self.lease is not None:
                        self.lease.close()
                        self.lease = s.cold_kv.lookup((), namespace=s.namespace)
                    start = 0
                end = min(len(self.tokens), start + s.model.prefill_chunk_tokens)
                suffix = self.tokens[start:end]
                began = perf_counter()
                result = s.model.prefill_chunk(self.slot, suffix,
                    history_tokens=self.tokens[max(0,start-3):start], cancel=self.cancel)
                computed_at = perf_counter()
                if s.cold_kv is not None:
                    drained = []
                    new = store_chunk(s.cold_kv, s.model.past, self.slot,
                                      self.lease, suffix, namespace=s.namespace,
                                      on_drain=drained.append)
                    # The chunk's kernels retire inside store_chunk: the wait
                    # belongs to the compute that queued them, not to the store.
                    computed_at = drained[0]
                    self.lease.close()
                    self.lease = new
                    self.metrics.cache_store_seconds += perf_counter() - computed_at
                    self.metrics.cache_stored_blocks += 1
                self.output = result.output
                self.computed_tokens += end - start
                self.chunks += 1
                m = self.metrics
                m.chunk_seconds.append(computed_at - began)
                m.chunk_tokens.append(end - start)
                m.model_prefill_seconds += computed_at - began
                m.prefill_tokens, m.chunks = self.computed_tokens, self.chunks
                m.strategy_seconds += perf_counter() - began
                return result
            except BaseException:
                # No rollback guess after failed compute/copy: release this slot.
                # Earlier published, complete chunk endpoints remain valid.
                self.close()
                raise

    def run(self):
        while self.step() is not None:
            pass
        return self.output

    def finish_prefill(self):
        """Prepare decoder tail/logits after run(), including a full cold hit.

        Explicit so callers filling the prefix cache do not pay decoder cost.
        A future generation worker must call this before consuming first logits.
        """
        with self.strategy.lock:
            self._live()
            if not self.done:
                raise RuntimeError('finish input before preparing generation')
            try:
                began = perf_counter()
                self.output = self.strategy.model.finish_prefill(self.slot, cancel=self.cancel)
                self.metrics.finish_prefill_seconds += perf_counter() - began
                self.metrics.strategy_seconds += perf_counter() - began
                return self.output
            except BaseException:
                self.close()
                raise

    def extend(self, token_ids):
        """Append to a fully consumed live session, keeping its Past and lease."""
        with self.strategy.lock:
            self._live()
            if not self.done:
                raise RuntimeError('finish current input before extending')
            suffix = tuple(int(t) for t in token_ids)
            if not suffix or len(self.tokens) + len(suffix) > self.strategy.model.past.row_cap:
                raise ValueError('invalid extension length')
            self.tokens += suffix
            self.output = None

    def close(self):
        with self.strategy.lock:
            if self.closed:
                return
            try:
                if self.lease is not None:
                    self.lease.close()
            finally:
                self.strategy.model.past.release(self.slot)
                self.closed = True
                self.output = None

    def __enter__(self):
        self._live()
        return self

    def __exit__(self, *args):
        self.close()
