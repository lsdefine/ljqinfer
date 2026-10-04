"""Synchronous ljqinfer orchestration, independent of model computation.

One step processes at most one chunk. Callers may interleave live sessions;
this is not a threaded scheduler, decoding loop or protocol server.
"""
from threading import RLock
from model.model_api import PrefillCancelled
from model.cold import fields_for, restore_prefix, store_chunk


class Strategy:
    def __init__(self, model, cold_kv=None, *, namespace):
        self.model, self.cold_kv, self.namespace = model, cold_kv, namespace
        self.lock = RLock()
        if cold_kv is not None:
            if cold_kv.namespace != namespace:
                raise ValueError('cache namespace mismatch')
            if cold_kv.fields != {f.key: f for f in fields_for(model.past)}:
                raise ValueError('cache layout mismatch')

    def open(self, input_ids, *, cancel=None):
        tokens = tuple(int(t) for t in input_ids)
        if not tokens or len(tokens) > self.model.past.max_seq:
            raise ValueError('input length must be 1..max_seq')
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
                if self.cold_kv is not None:
                    lease = self.cold_kv.lookup(tokens, namespace=self.namespace)
                    restore_prefix(self.cold_kv, pool, slot, lease)
                    replayed = self.model.replay_prefix(
                        slot, tokens[:lease.token_count], cancel=cancel)
                if cancel is not None and cancel.is_set():
                    raise PrefillCancelled()
                session = PrefillSession(self, slot, tokens, lease, cancel)
                session.replayed_tokens = replayed
                return session
            except BaseException:
                if lease is not None:
                    lease.close()
                pool.release(slot)
                raise


class PrefillSession:
    def __init__(self, strategy, slot, tokens, lease, cancel):
        self.strategy, self.slot, self.tokens = strategy, slot, tokens
        self.lease, self.cancel = lease, cancel
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
                end = min(len(self.tokens), start + s.model.prefill_chunk_tokens)
                suffix = self.tokens[start:end]
                result = s.model.prefill_chunk(self.slot, suffix,
                    history_tokens=self.tokens[max(0,start-3):start], cancel=self.cancel)
                if s.cold_kv is not None:
                    new = store_chunk(s.cold_kv, s.model.past, self.slot,
                                      self.lease, suffix, namespace=s.namespace)
                    self.lease.close()
                    self.lease = new
                self.output = result.output
                self.computed_tokens += end - start
                self.chunks += 1
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
                self.output = self.strategy.model.finish_prefill(self.slot, cancel=self.cancel)
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
            if not suffix or len(self.tokens) + len(suffix) > self.strategy.model.past.max_seq:
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
