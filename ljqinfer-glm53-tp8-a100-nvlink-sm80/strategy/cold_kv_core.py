"""通用冷缓存内存管理：新模型、新硬件无需修改本文件，两仓保持一致。
模型布局与设备搬运在外部适配；公共修复须同步并通过同一套回归。
Backend.copy 须在异步复制前登记可 synchronize() 的 waiter（含异常路径）。
"""
from dataclasses import dataclass, field
from threading import RLock


class ArenaCapacityError(MemoryError):
    """Bounded arena pressure, NOT OS/device allocation failure."""


class CacheCapacityError(MemoryError):
    """Cache admission failed; live leases and staged copies stay protected."""


@dataclass(frozen=True)
class Field:
    key: tuple
    shape: tuple  # None permits a variable leading row count
    dtype: object

    def validate(self, value):
        if value.dtype != self.dtype or value.ndim != len(self.shape):
            raise ValueError('cold field shape/dtype mismatch')
        if any(n is not None and n != got for n, got in zip(self.shape, value.shape)):
            raise ValueError('cold field shape/dtype mismatch')


@dataclass
class Entry:
    id: int
    parent: int
    start: int
    end: int
    node: int
    touched: int
    nbytes: int
    pins: int = 0
    children: set = field(default_factory=set)


@dataclass
class Node:
    parent: int
    token: object
    children: dict = field(default_factory=dict)
    entry: int = 0


class Lease:
    def __init__(self, cache, ids, tokens, matched_tokens, end=None):
        self.cache, self.ids = cache, tuple(ids)
        self.tokens, self.matched_tokens = tuple(tokens), matched_tokens
        self.end = matched_tokens if end is None else end
        self.closed = False

    @property
    def token_count(self):
        return self.end

    def close(self):
        with self.cache.lock:
            if self.closed:
                return
            if self.cache.active is not None and self.cache.active.parent is self:
                raise RuntimeError('close store before its parent lease')
            for i in self.ids:
                self.cache.entries[i].pins -= 1
            self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class Store:
    def __init__(self, cache, parent, tokens):
        self.cache, self.parent, self.tokens = cache, parent, tokens
        self.written, self.closed = {}, False
        self.staged = {}
        self.waiters = set()

    def _check(self):
        if self.closed or self.cache.active is not self:
            raise RuntimeError('store is closed')

    def _alloc(self, key, value):
        c = self.cache
        if c.shm is None:
            return c.backend.empty(value, c.pin_memory)
        old = self.staged.pop(key, None)
        if old is not None:
            c.shm.free(*old)
        nbytes = c.backend.nbytes(value)
        if not nbytes:
            return c.backend.empty(value, False)
        # Payload accounting cannot predict alignment waste or fragmentation.
        # Retry only bounded-arena pressure, not OS/device allocation failures.
        while True:
            try:
                offset = c.shm.alloc(nbytes)
                break
            except ArenaCapacityError:
                c._evict_one()
        self.staged[key] = (offset, nbytes)
        return c.shm.view(offset, tuple(value.shape), value.dtype)

    def _release_staged(self):
        if self.cache.shm is not None:
            for offset, nbytes in self.staged.values():
                self.cache.shm.free(offset, nbytes)
        self.staged = {}

    def write(self, key, value):
        with self.cache.lock:
            self._check()
            c = self.cache
            c.fields[key].validate(value)
            if key in self.written:
                self._settle()  # Never recycle a destination with IO in flight.
                del self.written[key]
            size = c.backend.nbytes(value)
            staged = sum(c.backend.nbytes(t) for k, t in self.written.items() if k != key)
            c._make_room(staged + size)
            out = self._alloc(key, value)
            # Hold the destination even if enqueue fails, until abort settles IO.
            self.written[key] = out
            c.backend.copy(out, value, self.waiters)

    def commit(self):
        with self.cache.lock:
            self._check()
            self._settle()
            c = self.cache
            if set(self.written) != set(c.fields):
                raise RuntimeError('cannot publish incomplete fields')
            parent = self.parent.ids[-1] if self.parent.ids else 0
            end = self.parent.token_count + len(self.tokens)
            node = c._lease_node(self.parent)
            for token in self.tokens:
                if token not in c.nodes[node].children:
                    new = c.next_node
                    c.next_node += 1
                    c.nodes[new] = Node(node, token)
                    c.nodes[node].children[token] = new
                node = c.nodes[node].children[token]
            if c.nodes[node].entry:
                raise ValueError('prefix already stored; obtain a new lookup')
            i = c.next_id
            c.next_id += 1
            parent = self.parent.ids[-1] if self.parent.ids else 0
            size = sum(c.backend.nbytes(t) for t in self.written.values())
            c.clock += 1
            c.entries[i] = Entry(i, parent, self.parent.token_count, end, node, c.clock, size)
            if parent:
                c.entries[parent].children.add(i)
            c.nodes[node].entry = i
            for key, value in self.written.items():
                c.storage[key][i] = value
            if c.shm is not None:
                c.blocks[i] = tuple(self.staged.values())
            c.used_bytes += size
            c.active, self.closed = None, True
            self.written, self.staged = {}, {}
            return i

    def _settle(self):
        """Await the staged device copies before their bytes are used.

        Publishing or freeing staging while a DMA is still in flight would
        hand out half-written rows, so every exit from the transaction goes
        through here first.
        """
        for waiter in self.waiters:
            waiter.synchronize()
        self.waiters.clear()

    def abort(self):
        with self.cache.lock:
            if not self.closed:
                self._check()
                self._settle()
                self._release_staged()
                self.written.clear()
                self.cache.active, self.closed = None, True

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.abort()


class ColdCache:
    def __init__(self, fields, budget_bytes, *, namespace, pin_memory=False,
                 shm=None, backend):
        fields = tuple(fields)
        self.fields = {f.key: f for f in fields}
        if len(self.fields) != len(fields) or not fields or budget_bytes <= 0:
            raise ValueError('unique fields and positive budget required')
        self.namespace, self.budget_bytes = namespace, int(budget_bytes)
        self.backend = backend
        self.pin_memory, self.used_bytes = pin_memory, 0
        self.storage = {key: {} for key in self.fields}
        self.nodes, self.next_node = {0: Node(-1, None)}, 1
        self.entries, self.next_id, self.clock = {}, 1, 0
        self.active, self.lock = None, RLock()
        # Arena/rank ownership is a caller policy, not a cache assumption.
        self.shm, self.blocks = shm, {}

    def _validate(self, lease):
        if lease.cache is not self or lease.closed:
            raise RuntimeError('foreign or closed cache lease')

    def lookup(self, tokens, *, namespace):
        if namespace != self.namespace:
            raise ValueError('cache model/layout namespace mismatch')
        tokens = tuple(int(t) for t in tokens)
        with self.lock:
            node, matched, best = 0, 0, 0
            for token in tokens:
                next_node = self.nodes[node].children.get(token)
                if next_node is None:
                    break
                node, matched = next_node, matched + 1
                if self.nodes[node].entry:
                    best = self.nodes[node].entry
            # A token trie match inside a stored segment is now restorable.
            if matched:
                pending = [node]
                while pending:
                    candidate = pending.pop()
                    best = self.nodes[candidate].entry
                    if best:
                        break
                    pending.extend(self.nodes[candidate].children.values())
                if not best:
                    raise RuntimeError('trie path has no live backing entry')
            lease = self.pin_endpoint(best, namespace=namespace, end=matched)
            lease.tokens = tokens
            return lease

    def _lease_node(self, lease):
        node = self.entries[lease.ids[-1]].node if lease.ids else 0
        end = self.entries[lease.ids[-1]].end if lease.ids else 0
        for _ in range(end - lease.token_count):
            node = self.nodes[node].parent
        return node

    def spans(self, lease):
        """Disjoint backing slices, clipping covering ancestors at branch starts."""
        self._validate(lease)
        end, spans = lease.token_count, []
        for i in reversed(lease.ids):
            e = self.entries[i]
            if end > e.start:
                spans.append((i, e.start, end))
                end = e.start
        if end:
            raise ValueError('incomplete cold backing chain')
        return list(reversed(spans))

    def lookup_suffix(self, parent, tokens, *, namespace):
        """Return a lease for an exact appended endpoint, or None."""
        if namespace != self.namespace:
            raise ValueError('cache model/layout namespace mismatch')
        with self.lock:
            self._validate(parent)
            node = self._lease_node(parent)
            for token in tokens:
                node = self.nodes[node].children.get(token)
                if node is None:
                    return None
            entry = self.nodes[node].entry
            return self.pin_endpoint(entry, namespace=namespace) if entry else None

    def pin_endpoint(self, entry_id, *, namespace, end=None):
        """Pin only the backing slices needed through ``end`` (default: full).

        A trie descendant can cover a shorter matched prefix without supplying
        any of its bytes. Pinning that descendant would also make a new fork
        depend on an unrelated old suffix, building an unevictable branch chain.
        Use the same clipping rule as spans(), before acquiring pins: the final
        lease ID is the actual provider of the prefix, hence the correct parent
        of a subsequent store. Exact endpoint handoffs keep their full extent.
        """
        if namespace != self.namespace:
            raise ValueError('cache model/layout namespace mismatch')
        with self.lock:
            if entry_id and entry_id not in self.entries:
                raise KeyError('cold endpoint has been evicted')
            full_end = self.entries[entry_id].end if entry_id else 0
            end = full_end if end is None else end
            if not 0 <= end <= full_end:
                raise ValueError('prefix outside cold endpoint')
            ids, needed = [], end
            while entry_id and needed:
                e = self.entries[entry_id]
                if needed > e.start:
                    if needed > e.end:
                        raise ValueError('incomplete cold backing chain')
                    ids.append(entry_id)
                    needed = e.start
                entry_id = e.parent
            if needed:
                raise ValueError('incomplete cold backing chain')
            for i in ids:
                e = self.entries[i]
                self.clock += 1
                e.touched, e.pins = self.clock, e.pins + 1
            return Lease(self, reversed(ids), (), end)

    def prepare(self, parent, tokens):
        tokens = tuple(int(t) for t in tokens)
        with self.lock:
            self._validate(parent)
            if self.active is not None:
                raise RuntimeError('store transaction already active')
            if not tokens:
                raise ValueError('nonempty token suffix required')
            self.active = Store(self, parent, tokens)
            return self.active

    def _make_room(self, staged_bytes):
        if staged_bytes > self.budget_bytes:
            raise CacheCapacityError('payload exceeds cold byte budget')
        while self.used_bytes + staged_bytes > self.budget_bytes:
            self._evict_one()

    def _evict_one(self):
        """Reclaim one LRU leaf; caller holds lock, pins protect restore chains."""
        candidates = [e for e in self.entries.values() if not e.pins and not e.children]
        if not candidates:
            raise CacheCapacityError('no unleased cache leaf can satisfy allocation')
        e = min(candidates, key=lambda x: x.touched)
        if e.parent:
            self.entries[e.parent].children.remove(e.id)
        del self.entries[e.id]
        for values in self.storage.values():
            del values[e.id]
        for block in self.blocks.pop(e.id, ()):
            self.shm.free(*block)
        self.used_bytes -= e.nbytes
        node = e.node
        self.nodes[node].entry = 0
        while node and not self.nodes[node].entry and not self.nodes[node].children:
            old = self.nodes.pop(node)
            del self.nodes[old.parent].children[old.token]
            node = old.parent


    def read(self, lease, segment):
        with self.lock:
            self._validate(lease)
            if not 0 <= segment < len(lease.ids):
                raise IndexError('segment outside lease')
            i = lease.ids[segment]
            return {k: values[i].clone() for k, values in self.storage.items()}

    def clear(self):
        with self.lock:
            if self.active is not None or any(e.pins for e in self.entries.values()):
                raise RuntimeError('cannot clear live leases or store')
            self.entries.clear()
            for values in self.storage.values():
                values.clear()
            if self.shm is not None:
                for blocks in self.blocks.values():
                    for block in blocks:
                        self.shm.free(*block)
            self.blocks.clear()
            self.nodes = {0: Node(-1, None)}
            self.used_bytes = 0
