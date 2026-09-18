"""Replaceable 1024-token CPU-DRAM prefix cache for hybrid Qwen state."""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import struct
import threading
from typing import Any, Callable, Sequence


@dataclass(frozen=True)
class KVLayout:
    full_kv_layout: str = "block_layer_token_head_dim"
    gdn_layout: str = "boundary_layer_sequence_state"


@dataclass(frozen=True)
class ColdMatch:
    token_count: int
    digests: tuple[bytes, ...]
    generation: int


@dataclass
class _Entry:
    parent: bytes
    token_ids: tuple[int, ...]
    record: Any
    last_used: int


class PrefixColdCache:
    """Exact longest-prefix cache with immutable CPU-owned records.

    Entries form a hash chain, so a block is reusable only under the exact same
    earlier prefix. Hash collisions are rejected by checking parent and token
    IDs before an entry is accepted.
    """
    def __init__(self, block_size: int = 1024):
        if block_size <= 0:
            raise ValueError("block_size must be positive")
        self.block_size = int(block_size)
        self.layout = KVLayout()
        self._entries: dict[bytes, _Entry] = {}
        self._generation = 0
        self._clock = 0
        self._lock = threading.RLock()

    @staticmethod
    def _digest(parent: bytes, tokens: tuple[int, ...]) -> bytes:
        h = hashlib.blake2b(digest_size=32, person=b"ljq-qwen-cold-v1")
        h.update(parent)
        h.update(struct.pack("<I", len(tokens)))
        for token in tokens:
            h.update(struct.pack("<q", int(token)))
        return h.digest()

    @staticmethod
    def _own_cpu(value):
        try:
            import torch
            if torch.is_tensor(value):
                return value.detach().to("cpu").clone()
        except ImportError:
            pass
        if isinstance(value, tuple):
            return tuple(PrefixColdCache._own_cpu(x) for x in value)
        if isinstance(value, list):
            return [PrefixColdCache._own_cpu(x) for x in value]
        if isinstance(value, dict):
            return {k: PrefixColdCache._own_cpu(v) for k, v in value.items()}
        return value

    def begin(self, input_ids: Sequence[int]) -> ColdMatch:
        ids = tuple(int(x) for x in input_ids)
        parent = b""
        hits: list[bytes] = []
        with self._lock:
            for start in range(0, len(ids) - self.block_size + 1, self.block_size):
                block = ids[start:start + self.block_size]
                digest = self._digest(parent, block)
                entry = self._entries.get(digest)
                if entry is None or entry.parent != parent or entry.token_ids != block:
                    break
                self._clock += 1
                entry.last_used = self._clock
                hits.append(digest)
                parent = digest
            return ColdMatch(len(hits) * self.block_size, tuple(hits), self._generation)

    def restore(self, match: ColdMatch,
                importer: Callable[[Any, bool], int]) -> int:
        with self._lock:
            if match.generation != self._generation:
                raise RuntimeError("cold-cache match was invalidated")
            records = []
            parent = b""
            for digest in match.digests:
                entry = self._entries.get(digest)
                if entry is None or entry.parent != parent:
                    raise RuntimeError("cold-cache entry changed after lookup")
                records.append(entry.record)
                parent = digest
        cursor = 0
        for i, record in enumerate(records):
            cursor = int(importer(record, i + 1 == len(records)))
        if cursor != match.token_count:
            raise RuntimeError("cold-cache restore cursor mismatch")
        return cursor

    def store(self, match: ColdMatch, input_ids: Sequence[int],
              exporter: Callable[[int, int], Any]) -> int:
        ids = tuple(int(x) for x in input_ids)
        if match.token_count > len(ids) or match.token_count % self.block_size:
            raise ValueError("invalid matched prefix length")
        parent = match.digests[-1] if match.digests else b""
        stored = 0
        with self._lock:
            if match.generation != self._generation:
                raise RuntimeError("cold-cache match was invalidated")
            for start in range(match.token_count,
                               len(ids) - self.block_size + 1,
                               self.block_size):
                end = start + self.block_size
                block = ids[start:end]
                digest = self._digest(parent, block)
                existing = self._entries.get(digest)
                if existing is not None:
                    if existing.parent != parent or existing.token_ids != block:
                        raise RuntimeError("cold-cache digest collision")
                else:
                    record = self._own_cpu(exporter(start, end))
                    self._clock += 1
                    self._entries[digest] = _Entry(parent, block, record, self._clock)
                    stored += 1
                parent = digest
        return stored

    def store_owned_batch(self, match: ColdMatch, input_ids: Sequence[int],
                          exporter: Callable[[int, int], Sequence[Any]]) -> int:
        """Store a batch of immutable CPU-owned records without cloning them.

        This is the high-throughput model path: the exporter performs batched
        device-to-host copies and transfers ownership of the resulting CPU
        allocations to the cache.  The legacy ``store`` method remains the
        defensive clone-on-insert API for generic callers.
        """
        ids = tuple(int(x) for x in input_ids)
        if match.token_count > len(ids) or match.token_count % self.block_size:
            raise ValueError("invalid matched prefix length")
        final_end = len(ids) - len(ids) % self.block_size
        starts = tuple(range(match.token_count, final_end, self.block_size))
        if not starts:
            return 0
        records = tuple(exporter(starts[0], final_end))
        if len(records) != len(starts):
            raise RuntimeError("batch exporter returned the wrong record count")
        parent = match.digests[-1] if match.digests else b""
        stored = 0
        with self._lock:
            if match.generation != self._generation:
                raise RuntimeError("cold-cache match was invalidated")
            for start, record in zip(starts, records):
                end = start + self.block_size
                block = ids[start:end]
                digest = self._digest(parent, block)
                existing = self._entries.get(digest)
                if existing is not None:
                    if existing.parent != parent or existing.token_ids != block:
                        raise RuntimeError("cold-cache digest collision")
                else:
                    self._clock += 1
                    self._entries[digest] = _Entry(
                        parent, block, record, self._clock)
                    stored += 1
                parent = digest
        return stored

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            self._generation += 1

    @property
    def entry_count(self) -> int:
        with self._lock:
            return len(self._entries)
