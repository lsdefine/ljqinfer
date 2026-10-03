#!/usr/bin/env python3
"""Replaceable cold-KV storage backends for the strategy layer."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import hashlib
import struct
import threading
from typing import Callable, Dict, List, Sequence, Set, Tuple

import torch

DEFAULT_KV_BLOCK_SIZE = 2 * 1024
DEFAULT_INITIAL_CACHE_BYTES = 768 << 20
DEFAULT_TARGET_CACHE_BYTES = 80 << 30
ROOT_CACHE_ID = 0
BlockHash = bytes
IndexKey = Tuple[BlockHash, int]
LoadBlock = Callable[[int, int, torch.Tensor], None]
StoreBlock = Callable[[int, int, torch.Tensor], None]


@dataclass(frozen=True, slots=True)
class KVLayout:
    """Physical shape and dtype of one token's portable KV record."""
    layers: int
    width: int
    dtype: torch.dtype


@dataclass(frozen=True, slots=True)
class CacheLookup:
    """Opaque longest-prefix match returned to strategy code.

    ``_entry_tokens`` records how much of each entry belongs to this match.  The
    last entry may be consumed only partly when a query diverges inside a block.
    """
    block_count: int
    token_count: int
    _entries: tuple["_Entry", ...]
    _entry_tokens: tuple[int, ...]
    _generation: int


@dataclass(frozen=True, slots=True)
class CacheStoreResult:
    stored_blocks: int
    evicted_blocks: int


class ColdKVBackend(ABC):
    block_size: int

    @abstractmethod
    def begin(self, input_ids: Sequence[int]) -> CacheLookup:
        """Match and pin the longest cached prefix for one query."""

    @abstractmethod
    def restore(self, match: CacheLookup, load_block: LoadBlock) -> None:
        """Restore a matched prefix through the model-supplied callback."""

    @abstractmethod
    def store(self, match: CacheLookup, input_ids: Sequence[int],
              store_block: StoreBlock) -> CacheStoreResult:
        """Persist the canonical prefix through the model callback."""

    @abstractmethod
    def clear(self) -> None:
        """Invalidate all cached prefixes while retaining storage."""

    @property
    @abstractmethod
    def token_count(self) -> int:
        """Number of valid cached tokens."""

    @property
    @abstractmethod
    def capacity_blocks(self) -> int:
        """Currently available physical blocks."""


@dataclass(frozen=True, slots=True)
class _Ref:
    location_id: int
    id: int


@dataclass(slots=True)
class _Entry:
    id: int
    location_id: int
    block_hash: BlockHash
    previous_id: int
    last_used: int
    tensor: torch.Tensor
    is_valid: bool
    token_ids: tuple[int, ...] = ()
    next_ids: Set[int] = field(default_factory=set)

    @property
    def valid_tokens(self) -> int:
        return len(self.token_ids)


class PinnedMemoryKVCache(ColdKVBackend):
    """LRU radix-prefix cache backed by fixed-size CPU pinned KV pages.

    Entries form a tree.  Full blocks use ``(hash, parent_id)`` for the fast
    path; child token arrays provide collision checking and prefix matches inside
    either full or partial blocks.  Stored paths are canonical: full blocks
    followed by at most one partial leaf.  Growing a partial tail creates (or
    reuses) a full sibling rather than appending fragmented pages.

    A lookup touches every entry from left to right.  Consequently an ancestor
    is never older than a reachable descendant, so eviction can safely choose
    the least-recently-used unpinned leaf without orphaning live nodes.
    """

    def __init__(self, layout: KVLayout,
                 block_size: int = DEFAULT_KV_BLOCK_SIZE,
                 initial_bytes: int = DEFAULT_INITIAL_CACHE_BYTES,
                 target_bytes: int = DEFAULT_TARGET_CACHE_BYTES):
        self.layout = layout
        self.block_size = int(block_size)
        if self.block_size <= 0:
            raise ValueError("block_size must be positive")
        self.page_bytes = (self.block_size * layout.layers * layout.width
                           * torch.empty((), dtype=layout.dtype).element_size())
        initial_bytes, target_bytes = int(initial_bytes), int(target_bytes)
        if initial_bytes <= 0 or target_bytes <= 0:
            raise ValueError("cache byte targets must be positive")
        self.target_blocks = max(1, target_bytes // self.page_bytes)
        self.initial_blocks = min(
            self.target_blocks, max(1, initial_bytes // self.page_bytes))
        self.hash_index: Dict[IndexKey, _Ref] = {}
        self._entries: List[_Entry] = []
        self._root_next_ids: Set[int] = set()
        self._next_id = ROOT_CACHE_ID + 1
        self._last_used = 0
        self._generation = 0
        self._lock = threading.RLock()
        self._ready = threading.Event()
        self._allocation_error: BaseException | None = None

        for _ in range(self.initial_blocks):
            self._append_entry()
        self._ready.set()
        self._allocator = threading.Thread(
            target=self._grow, name="cold-kv-pinned-allocator", daemon=True)
        self._allocator.start()
        print("[cold-kv] pinned pool ready "
              f"blocks={self.initial_blocks}/{self.target_blocks} "
              f"page_mib={self.page_bytes / (1 << 20):.2f}", flush=True)

    def _new_tensor(self) -> torch.Tensor:
        backing = torch.empty(
            (self.layout.layers, self.block_size, self.layout.width),
            dtype=self.layout.dtype, device="cpu", pin_memory=True)
        return backing.permute(1, 0, 2)

    def _append_entry(self) -> None:
        tensor = self._new_tensor()
        with self._lock:
            location_id = len(self._entries)
            self._entries.append(_Entry(
                id=ROOT_CACHE_ID, location_id=location_id, block_hash=b"",
                previous_id=ROOT_CACHE_ID, last_used=0, tensor=tensor,
                is_valid=False))

    def _grow(self) -> None:
        try:
            while self.capacity_blocks < self.target_blocks:
                self._append_entry()
                count = self.capacity_blocks
                if count % 32 == 0 or count == self.target_blocks:
                    print(f"[cold-kv] pinned pool blocks={count}/"
                          f"{self.target_blocks}", flush=True)
        except BaseException as exc:
            self._allocation_error = exc
            print(f"[cold-kv] background growth stopped: "
                  f"{type(exc).__name__}: {exc}", flush=True)

    @staticmethod
    def _hash_tokens(tokens: Sequence[int]) -> BlockHash:
        values = tuple(int(token) for token in tokens)
        payload = struct.pack(f"<{len(values)}q", *values)
        return hashlib.blake2b(payload, digest_size=16).digest()

    def _children(self, parent_id: int) -> Set[int]:
        if parent_id == ROOT_CACHE_ID:
            return self._root_next_ids
        for entry in self._entries:
            if entry.is_valid and entry.id == parent_id:
                return entry.next_ids
        return set()

    def _entry_by_id(self, entry_id: int) -> _Entry | None:
        for entry in self._entries:
            if entry.is_valid and entry.id == entry_id:
                return entry
        return None

    def _indexed_exact(self, parent_id: int,
                       tokens: tuple[int, ...]) -> _Entry | None:
        block_hash = self._hash_tokens(tokens)
        key = (block_hash, parent_id)
        ref = self.hash_index.get(key)
        if ref is not None:
            entry = self._entries[ref.location_id]
            if (entry.is_valid and entry.id == ref.id
                    and entry.previous_id == parent_id
                    and entry.block_hash == block_hash
                    and entry.token_ids == tokens):
                return entry
            if self.hash_index.get(key) == ref:
                self.hash_index.pop(key, None)
        # Hashes are accelerators, not identities.  A collision may leave only
        # one child in the single-value index; token equality remains decisive.
        for child_id in tuple(self._children(parent_id)):
            entry = self._entry_by_id(child_id)
            if entry is not None and entry.token_ids == tokens:
                self.hash_index[key] = _Ref(entry.location_id, entry.id)
                return entry
        return None

    def _best_child_prefix(self, parent_id: int,
                           remaining: tuple[int, ...]) -> tuple[_Entry | None, int]:
        best, best_len = None, 0
        stale = []
        for child_id in tuple(self._children(parent_id)):
            entry = self._entry_by_id(child_id)
            if entry is None or entry.previous_id != parent_id:
                stale.append(child_id)
                continue
            limit = min(len(remaining), entry.valid_tokens)
            common = 0
            while common < limit and remaining[common] == entry.token_ids[common]:
                common += 1
            if common > best_len:
                best, best_len = entry, common
        for child_id in stale:
            self._children(parent_id).discard(child_id)
        return best, best_len

    def begin(self, input_ids: Sequence[int]) -> CacheLookup:
        ids = tuple(int(token) for token in input_ids)
        with self._lock:
            self._last_used += 1
            stamp = self._last_used
            hits: List[_Entry] = []
            consumed: List[int] = []
            parent_id = ROOT_CACHE_ID
            position = 0
            while position < len(ids):
                remaining = ids[position:]
                entry = None
                if len(remaining) >= self.block_size:
                    chunk = remaining[:self.block_size]
                    entry = self._indexed_exact(parent_id, chunk)
                if entry is not None:
                    matched = self.block_size
                else:
                    entry, matched = self._best_child_prefix(parent_id, remaining)
                    if entry is None or matched == 0:
                        break
                entry.last_used = stamp
                hits.append(entry)
                consumed.append(matched)
                position += matched
                if matched < entry.valid_tokens:
                    break
                parent_id = entry.id
            return CacheLookup(len(hits), position, tuple(hits),
                               tuple(consumed), self._generation)

    def restore(self, match: CacheLookup, load_block: LoadBlock) -> None:
        with self._lock:
            self._validate_match(match)
            start = 0
            for entry, count in zip(match._entries, match._entry_tokens):
                if not entry.is_valid or count <= 0 or count > entry.valid_tokens:
                    raise RuntimeError("cold KV lookup entry was evicted or changed")
                end = start + count
                load_block(start, end, entry.tensor[:count])
                start = end
            if start != match.token_count:
                raise RuntimeError("cold KV lookup token count is inconsistent")

    def _invalidate_leaf(self, entry: _Entry) -> bool:
        if not entry.is_valid:
            return False
        if entry.next_ids:
            raise RuntimeError("cannot evict a non-leaf cold KV entry")
        key = (entry.block_hash, entry.previous_id)
        ref = _Ref(entry.location_id, entry.id)
        if self.hash_index.get(key) == ref:
            self.hash_index.pop(key, None)
        self._children(entry.previous_id).discard(entry.id)
        entry.is_valid = False
        entry.id = ROOT_CACHE_ID
        entry.block_hash = b""
        entry.previous_id = ROOT_CACHE_ID
        entry.last_used = 0
        entry.token_ids = ()
        entry.next_ids.clear()
        return True

    def _select_entry(self, pinned_ids: Set[int]) -> _Entry | None:
        for entry in self._entries:
            if not entry.is_valid:
                return entry
        leaves = [entry for entry in self._entries
                  if not entry.next_ids and entry.id not in pinned_ids]
        if not leaves:
            return None
        return min(leaves, key=lambda entry: (entry.last_used, entry.id))

    def _install(self, entry: _Entry, parent_id: int,
                 tokens: tuple[int, ...]) -> None:
        entry_id = self._next_id
        self._next_id += 1
        block_hash = self._hash_tokens(tokens)
        entry.id = entry_id
        entry.block_hash = block_hash
        entry.previous_id = parent_id
        entry.last_used = self._last_used
        entry.token_ids = tokens
        entry.next_ids.clear()
        entry.is_valid = True
        self.hash_index[(block_hash, parent_id)] = _Ref(
            entry.location_id, entry_id)
        self._children(parent_id).add(entry_id)

    def store(self, match: CacheLookup, input_ids: Sequence[int],
              store_block: StoreBlock) -> CacheStoreResult:
        ids = tuple(int(token) for token in input_ids)
        with self._lock:
            self._validate_match(match)
            # Re-walk canonical block boundaries.  Only selected ancestors are
            # pinned.  A matched old partial remains an evictable leaf, allowing
            # an in-place-capacity upgrade when the pool is already full.
            parent_id = ROOT_CACHE_ID
            pinned_ids: Set[int] = set()
            stored = evicted = 0
            for start in range(0, len(ids), self.block_size):
                end = min(start + self.block_size, len(ids))
                tokens = ids[start:end]
                entry = self._indexed_exact(parent_id, tokens)
                if entry is None:
                    entry = self._select_entry(pinned_ids)
                    if entry is None:
                        break
                    evicted += int(self._invalidate_leaf(entry))
                    try:
                        store_block(start, end, entry.tensor[:end - start])
                    except Exception:
                        entry.is_valid = False
                        entry.id = ROOT_CACHE_ID
                        entry.block_hash = b""
                        entry.previous_id = ROOT_CACHE_ID
                        entry.last_used = 0
                        entry.token_ids = ()
                        entry.next_ids.clear()
                        raise
                    self._install(entry, parent_id, tokens)
                    stored += 1
                else:
                    entry.last_used = self._last_used
                pinned_ids.add(entry.id)
                parent_id = entry.id
            return CacheStoreResult(stored, evicted)

    def _validate_match(self, match: CacheLookup) -> None:
        if match._generation != self._generation:
            raise RuntimeError("cold KV lookup was invalidated")

    def clear(self) -> None:
        with self._lock:
            self.hash_index.clear()
            self._root_next_ids.clear()
            self._generation += 1
            for entry in self._entries:
                entry.is_valid = False
                entry.id = ROOT_CACHE_ID
                entry.block_hash = b""
                entry.previous_id = ROOT_CACHE_ID
                entry.last_used = 0
                entry.token_ids = ()
                entry.next_ids.clear()

    @property
    def valid_slot_count(self) -> int:
        with self._lock:
            return sum(int(entry.is_valid) for entry in self._entries)

    @property
    def token_count(self) -> int:
        with self._lock:
            return sum(entry.valid_tokens for entry in self._entries
                       if entry.is_valid)

    @property
    def capacity_blocks(self) -> int:
        with self._lock:
            return len(self._entries)

    @property
    def capacity_bytes(self) -> int:
        return self.capacity_blocks * self.page_bytes

    @property
    def allocation_error(self) -> BaseException | None:
        return self._allocation_error
