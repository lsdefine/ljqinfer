"""Bounded CPU prefix pool for Qwen target, GDN and DFlash state.

Like the GLM52 backend, only unprotected LRU leaves may be recycled. Tensor
storage is allocated at startup and reused, never retained from exporters.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import copy
import hashlib
import struct
import threading
from typing import Any, Callable, Sequence

DEFAULT_TARGET_CACHE_BYTES = 80 << 30


@dataclass(frozen=True)
class KVLayout:
    full_kv_layout: str = "block_layer_token_head_dim"
    gdn_layout: str = "boundary_layer_sequence_state"


@dataclass(frozen=True)
class ColdMatch:
    token_count: int
    digests: tuple[bytes, ...]
    generation: int
    versions: tuple[int, ...] = ()


@dataclass
class _Entry:
    parent: bytes
    token_ids: tuple[int, ...]
    record: Any
    last_used: int
    slot: int
    version: int
    children: set[bytes] = field(default_factory=set)


class PrefixColdCache:
    """Fixed-byte pool plus exact-prefix index and deterministic leaf LRU.

    Production supplies a model-owned record template at startup. The template
    describes tensor shapes/dtypes, not GPU storage. The small metadata-only
    mode exists for callers storing plain Python records and is slot-bounded.
    Export/restore operations are serialized by the strategy execution lock;
    a restored record is borrowed until the next store/clear operation.
    """
    def __init__(self, block_size: int = 1024, *, record_template=None,
                 target_bytes: int = DEFAULT_TARGET_CACHE_BYTES,
                 max_blocks: int | None = None, pin_memory: bool = False):
        if block_size <= 0 or target_bytes <= 0:
            raise ValueError("block size and pool byte budget must be positive")
        if max_blocks is not None and max_blocks <= 0:
            raise ValueError("max_blocks must be positive")
        self.block_size = int(block_size)
        self.layout = KVLayout()
        self.target_bytes = int(target_bytes)
        self._max_blocks = max_blocks
        self._pin_memory = bool(pin_memory)
        self._entries: dict[bytes, _Entry] = {}
        self._generation = 0
        self._clock = 0
        self._version = 0
        self._lock = threading.RLock()
        self._pool = None
        self._template = None
        self._tensor_specs = []
        self.page_bytes = 0
        self.capacity_blocks = int(max_blocks or 2048)
        self._free = list(range(self.capacity_blocks - 1, -1, -1))
        self.evicted_blocks = 0
        if record_template is not None:
            self._configure(record_template)

    @staticmethod
    def _tensor(value):
        # CPU unit tests may use the metadata-only backend without torch.
        try:
            import torch
            return torch.is_tensor(value)
        except ImportError:
            return False

    @classmethod
    def _walk(cls, value):
        if cls._tensor(value):
            yield value
        elif isinstance(value, dict):
            for child in value.values():
                yield from cls._walk(child)
        elif isinstance(value, (tuple, list)):
            for child in value:
                yield from cls._walk(child)

    def _configure(self, template):
        import torch
        tensors = list(self._walk(template))
        if not tensors:
            return
        offset = 0
        specs = []
        for tensor in tensors:
            # Align typed views independently (including mixed BF16/FP32).
            offset = (offset + 63) // 64 * 64
            size = tensor.numel() * tensor.element_size()
            specs.append((offset, size, tuple(tensor.shape), tensor.dtype))
            offset += size
        page = (offset + 63) // 64 * 64
        if page <= 0 or page > self.target_bytes:
            raise ValueError("cold record exceeds pool byte budget")
        count = self.target_bytes // page
        if self._max_blocks is not None:
            count = min(count, self._max_blocks)
        # No background growth: all ranks have identical capacity before serving.
        pool = torch.empty((count, page), dtype=torch.uint8, device="cpu",
                           pin_memory=self._pin_memory)
        self._pool = pool
        self._tensor_specs = specs
        self._template = self._schema(template)
        self.page_bytes = page
        self.capacity_blocks = count
        self._free = list(range(count - 1, -1, -1))
        print(f"[cold-kv] fixed pool blocks={count} page_bytes={page} "
              f"capacity_bytes={self.capacity_bytes} pinned={self._pin_memory}",
              flush=True)

    @classmethod
    def _schema(cls, value):
        if cls._tensor(value):
            return ("tensor", tuple(value.shape), value.dtype)
        if isinstance(value, dict):
            return ("dict", tuple((k, cls._schema(v)) for k, v in value.items()))
        if isinstance(value, (tuple, list)):
            return (type(value).__name__, tuple(cls._schema(v) for v in value))
        if isinstance(value, (str, int, float, bool, bytes, type(None))):
            return ("scalar", type(value))
        raise TypeError(f"unsupported cold record value: {type(value)}")

    def _copy_record(self, value, slot):
        specs = iter(self._tensor_specs)
        def visit(v):
            if self._tensor(v):
                offset, size, shape, dtype = next(specs)
                dest = self._pool[slot, offset:offset + size].view(dtype).reshape(shape)
                dest.copy_(v.detach(), non_blocking=False)
                return dest
            if isinstance(v, dict):
                return {k: visit(x) for k, x in v.items()}
            if isinstance(v, tuple):
                return tuple(visit(x) for x in v)
            if isinstance(v, list):
                return [visit(x) for x in v]
            return copy.deepcopy(v)
        return visit(value)

    @staticmethod
    def _digest(parent: bytes, tokens: tuple[int, ...]) -> bytes:
        h = hashlib.blake2b(digest_size=32, person=b"ljq-qwen-cold-v1")
        h.update(parent)
        h.update(struct.pack("<I", len(tokens)))
        h.update(struct.pack(f"<{len(tokens)}q", *tokens))
        return h.digest()

    def begin(self, input_ids: Sequence[int]) -> ColdMatch:
        ids = tuple(int(x) for x in input_ids)
        parent = b""
        hits, versions = [], []
        with self._lock:
            self._clock += 1
            for start in range(0, len(ids) - self.block_size + 1, self.block_size):
                block = ids[start:start + self.block_size]
                digest = self._digest(parent, block)
                entry = self._entries.get(digest)
                if entry is None or entry.parent != parent or entry.token_ids != block:
                    break
                entry.last_used = self._clock
                hits.append(digest)
                versions.append(entry.version)
                parent = digest
            return ColdMatch(len(hits) * self.block_size, tuple(hits),
                             self._generation, tuple(versions))

    def _validate(self, match):
        if match.generation != self._generation:
            raise RuntimeError("cold-cache match was invalidated")
        if match.token_count != len(match.digests) * self.block_size:
            raise ValueError("invalid matched prefix length")
        if match.versions and len(match.versions) != len(match.digests):
            raise ValueError("invalid cold-cache match versions")
        parent = b""
        for i, digest in enumerate(match.digests):
            entry = self._entries.get(digest)
            if (entry is None or entry.parent != parent or
                    (match.versions and entry.version != match.versions[i])):
                raise RuntimeError("cold-cache entry changed after lookup")
            parent = digest

    def restore(self, match: ColdMatch,
                importer: Callable[[Any, bool], int]) -> int:
        with self._lock:
            self._validate(match)
            cursor = 0
            for i, digest in enumerate(match.digests):
                cursor = int(importer(self._entries[digest].record,
                                      i + 1 == len(match.digests)))
            if cursor != match.token_count:
                raise RuntimeError("cold-cache restore cursor mismatch")
            return cursor

    def _select(self, protected):
        if self._free:
            return self._free.pop()
        candidates = [(entry.last_used, entry.version, digest)
                      for digest, entry in self._entries.items()
                      if not entry.children and digest not in protected]
        if not candidates:
            return None
        _, _, digest = min(candidates)
        entry = self._entries.pop(digest)
        if entry.parent:
            self._entries[entry.parent].children.remove(digest)
        self.evicted_blocks += 1
        return entry.slot

    def _store(self, match, input_ids, exporter):
        ids = tuple(int(x) for x in input_ids)
        if match.token_count > len(ids) or match.token_count % self.block_size:
            raise ValueError("invalid matched prefix length")
        stored = 0
        with self._lock:
            self._validate(match)
            # Reject a match made against a different request prefix.
            parent = b""
            for i, digest in enumerate(match.digests):
                block = ids[i * self.block_size:(i + 1) * self.block_size]
                if self._entries[digest].token_ids != block:
                    raise ValueError("cold-cache match belongs to another prefix")
                parent = digest
            protected = set(match.digests)
            for start in range(match.token_count,
                               len(ids) - self.block_size + 1, self.block_size):
                end = start + self.block_size
                block = ids[start:end]
                digest = self._digest(parent, block)
                existing = self._entries.get(digest)
                if existing is not None:
                    if existing.parent != parent or existing.token_ids != block:
                        raise RuntimeError("cold-cache digest collision")
                    self._clock += 1
                    existing.last_used = self._clock
                else:
                    # Never export an entire prompt into unbounded transient RAM.
                    # One block bounds temporary allocations even on a full pool.
                    if not self._free and not any(
                            not e.children and d not in protected
                            for d, e in self._entries.items()):
                        break
                    record = exporter(start, end)
                    tensors = list(self._walk(record))
                    if self._pool is None and tensors:
                        if self._entries:
                            raise ValueError("cannot change cold record layout")
                        self._configure(record)
                    if self._pool is not None and self._schema(record) != self._template:
                        raise ValueError("cold record layout differs from startup pool")
                    slot = self._select(protected)
                    if slot is None:
                        break
                    try:
                        owned = self._copy_record(record, slot)
                    except BaseException:
                        self._free.append(slot)
                        raise
                    self._clock += 1
                    self._version += 1
                    self._entries[digest] = _Entry(
                        parent, block, owned, self._clock, slot, self._version)
                    if parent:
                        self._entries[parent].children.add(digest)
                    stored += 1
                    del record, tensors
                protected.add(digest)
                parent = digest
        return stored

    def store(self, match: ColdMatch, input_ids: Sequence[int],
              exporter: Callable[[int, int], Any]) -> int:
        return self._store(match, input_ids, exporter)

    def store_owned_batch(self, match: ColdMatch, input_ids: Sequence[int],
                          exporter: Callable[[int, int], Sequence[Any]]) -> int:
        """Compatibility entry point; copy into the pool, never adopt views."""
        def one(start, end):
            records = tuple(exporter(start, end))
            if len(records) != 1:
                raise RuntimeError("batch exporter returned the wrong record count")
            return records[0]
        return self._store(match, input_ids, one)

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            self._free = list(range(self.capacity_blocks - 1, -1, -1))
            self._generation += 1

    @property
    def capacity_bytes(self) -> int:
        return self.capacity_blocks * self.page_bytes

    @property
    def entry_count(self) -> int:
        with self._lock:
            return len(self._entries)
