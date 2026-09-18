#!/usr/bin/env python3
"""Cold KV Format V2: 128-token radix cache over field-major pinned slabs.

The cache index remains block granular.  Physical tensors are grouped by field,
so prefill can stream each model field directly from CUDA into its final pinned
location and restore can copy pinned fields directly into a GPU packet.  There
is no per-block torch.cat and no CPU gather staging buffer.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import math
import struct
import threading
from collections import deque

import numpy as np
from typing import Dict, Iterable, List, Sequence, Set, Tuple

import torch

from strategy.cold_kv_cache import (
    CacheLookup, CacheStoreResult, ColdKVBackend, ROOT_CACHE_ID, _Ref,
)

FieldKey = Tuple[str, int, str]


@dataclass(frozen=True, slots=True)
class KVFieldSpec:
    key: FieldKey
    shape: tuple[int, ...]
    dtype: torch.dtype

    @property
    def nbytes(self) -> int:
        return math.prod(self.shape) * torch.empty((), dtype=self.dtype).element_size()


@dataclass(slots=True)
class _Slab:
    fields: Dict[FieldKey, torch.Tensor]
    backings: Dict[str, torch.Tensor]
    rows: int

    def backing(self, group: str) -> torch.Tensor:
        return self.backings[group]


@dataclass(slots=True)
class _EntryV2:
    id: int
    location_id: int
    slab_id: int
    row: int
    block_hash: bytes = b""
    previous_id: int = ROOT_CACHE_ID
    last_used: int = 0
    is_valid: bool = False
    token_ids: tuple[int, ...] = ()
    next_ids: Set[int] = field(default_factory=set)

    @property
    def valid_tokens(self) -> int:
        return len(self.token_ids)


@dataclass(frozen=True, slots=True)
class KVRunV2:
    start_block: int
    block_count: int
    slab: _Slab
    slab_row: int

    def field(self, key: FieldKey) -> torch.Tensor:
        return self.slab.fields[key][self.slab_row:self.slab_row + self.block_count]

    def group(self, name: str, *, last_only: bool = False) -> torch.Tensor:
        rows = self.slab.backing(name)
        start = self.slab_row + (self.block_count - 1 if last_only else 0)
        stop = self.slab_row + self.block_count
        return rows[start:stop]


class StorePlanV2:
    """Two-phase append transaction.  GPU D2H writes happen before commit."""
    def __init__(self, cache: "PinnedMemoryKVCacheV2", match: CacheLookup,
                 entries: tuple[_EntryV2, ...], tokens: tuple[tuple[int, ...], ...],
                 first_block: int, evicted: int):
        self.cache = cache
        self.match = match
        self.entries = entries
        self.tokens = tokens
        self.first_block = first_block
        self.evicted = evicted
        self._done = False

    @property
    def stored_blocks(self) -> int:
        return len(self.entries)

    def _runs(self, absolute_block: int, block_count: int):
        lo = max(absolute_block, self.first_block)
        hi = min(absolute_block + block_count, self.first_block + len(self.entries))
        if hi <= lo:
            return
        i = lo - self.first_block
        end = hi - self.first_block
        while i < end:
            e = self.entries[i]
            j = i + 1
            while j < end:
                n = self.entries[j]
                if n.slab_id != e.slab_id or n.row != e.row + (j - i):
                    break
                j += 1
            yield lo + (i - (lo - self.first_block)), i, j, e
            i = j

    def write(self, key: FieldKey, absolute_block: int, source: torch.Tensor) -> None:
        """Asynchronously copy ``source[blocks,...]`` into final pinned slabs."""
        if self._done:
            raise RuntimeError("cold KV V2 plan is already closed")
        n = int(source.shape[0])
        for abs0, i, j, entry in self._runs(absolute_block, n):
            src0 = abs0 - absolute_block
            count = j - i
            dst = self.cache._slabs[entry.slab_id].fields[key][entry.row:entry.row + count]
            dst.copy_(source[src0:src0 + count].reshape_as(dst), non_blocking=True)

    def commit(self) -> CacheStoreResult:
        if self._done:
            raise RuntimeError("cold KV V2 plan is already closed")
        self._done = True
        return self.cache._commit(self)

    def abort(self) -> None:
        if self._done:
            return
        self._done = True
        self.cache._abort(self)


class PinnedMemoryKVCacheV2(ColdKVBackend):
    """LRU radix cache whose physical allocation is field-major pinned slabs."""
    def __init__(self, fields: Sequence[KVFieldSpec], block_size: int = 128,
                 initial_bytes: int = 768 << 20, target_bytes: int = 80 << 30,
                 slab_blocks: int = 128):
        self.fields = tuple(fields)
        if not self.fields:
            raise ValueError("V2 field layout must not be empty")
        self.block_size = int(block_size)
        self.block_bytes = sum(f.nbytes for f in self.fields)
        self.initial_blocks = max(1, int(initial_bytes) // self.block_bytes)
        self.target_blocks = max(self.initial_blocks, int(target_bytes) // self.block_bytes)
        self.slab_blocks = max(1, int(slab_blocks))
        self.hash_index: Dict[tuple[bytes, int], _Ref] = {}
        self._root_next_ids: Set[int] = set()
        self._entries: List[_EntryV2] = []
        self._by_id: Dict[int, _EntryV2] = {}
        # Row-parallel last_used mirror: -1 marks a free row.  Kept in
        # sync with _EntryV2.last_used so the eviction scan never has to
        # walk python objects.
        self._hot = np.full(0, -1, dtype=np.int64)
        self._slab_starts_cache: Set[int] = set()
        self._free_cursor = 0
        self._free_rows = 0
        self._slabs: List[_Slab] = []
        self._next_id = ROOT_CACHE_ID + 1
        self._last_used = 0
        self._generation = 0
        self._lock = threading.RLock()
        self._active_plan: StorePlanV2 | None = None
        self._allocation_error: BaseException | None = None
        self._grow_to(self.initial_blocks)
        self._allocator = threading.Thread(target=self._grow, name="cold-kv-v2-allocator", daemon=True)
        self._allocator.start()
        print(f"[cold-kv-v2] field-major slabs ready blocks={self.capacity_blocks}/"
              f"{self.target_blocks} block_mib={self.block_bytes/2**20:.2f} "
              f"slab_blocks={self.slab_blocks}", flush=True)

    @staticmethod
    def _hash_tokens(tokens: Sequence[int]) -> bytes:
        vals = tuple(int(x) for x in tokens)
        return hashlib.blake2b(struct.pack(f"<{len(vals)}q", *vals), digest_size=16).digest()

    @staticmethod
    def _group(key: FieldKey) -> str:
        kind, _, name = key
        return "tail" if kind == "tail" else ("main" if name == "main_kv" else "aux")

    def _append_slab(self) -> None:
        remain = self.target_blocks - len(self._entries)
        n = min(self.slab_blocks, remain)
        if n <= 0:
            return
        tensors: Dict[FieldKey, torch.Tensor] = {}
        backings: Dict[str, torch.Tensor] = {}
        for group in ("main", "aux", "tail"):
            specs = [f for f in self.fields if self._group(f.key) == group]
            row_bytes = sum(f.nbytes for f in specs)
            # One cache block is one contiguous row.  Field views are strided
            # across rows, allowing direct D2H writes without pack/cat while a
            # restore reads any consecutive block run with one DMA per group.
            backing = torch.empty((n, row_bytes), dtype=torch.uint8,
                                  device="cpu", pin_memory=True)
            backings[group] = backing
            offset = 0
            for f in specs:
                part = backing[:, offset:offset + f.nbytes]
                tensors[f.key] = part.view(f.dtype).view(n, *f.shape)
                offset += f.nbytes
            assert offset == row_bytes
        with self._lock:
            sid = len(self._slabs)
            self._slabs.append(_Slab(tensors, backings, n))
            for row in range(n):
                loc = len(self._entries)
                self._entries.append(_EntryV2(ROOT_CACHE_ID, loc, sid, row))
            self._hot = np.concatenate(
                (self._hot, np.full(n, -1, dtype=np.int64)))

    def _grow_to(self, blocks: int) -> None:
        while len(self._entries) < min(blocks, self.target_blocks):
            self._append_slab()

    def _grow(self) -> None:
        try:
            while self.capacity_blocks < self.target_blocks:
                self._append_slab()
                n = self.capacity_blocks
                if n % (self.slab_blocks * 2) == 0 or n == self.target_blocks:
                    print(f"[cold-kv-v2] pinned blocks={n}/{self.target_blocks}", flush=True)
        except BaseException as exc:
            self._allocation_error = exc
            print(f"[cold-kv-v2] background growth stopped: {type(exc).__name__}: {exc}", flush=True)

    def _children(self, parent: int) -> Set[int]:
        if parent == ROOT_CACHE_ID:
            return self._root_next_ids
        e = self._entry_by_id(parent)
        return e.next_ids if e is not None else set()

    def _entry_by_id(self, entry_id: int) -> _EntryV2 | None:
        return self._by_id.get(entry_id)

    def _exact(self, parent: int, tokens: tuple[int, ...]) -> _EntryV2 | None:
        key = (self._hash_tokens(tokens), parent)
        ref = self.hash_index.get(key)
        if ref is None or ref.location_id >= len(self._entries):
            return None
        e = self._entries[ref.location_id]
        if e.is_valid and e.id == ref.id and e.previous_id == parent and e.token_ids == tokens:
            return e
        self.hash_index.pop(key, None)
        return None

    def begin(self, input_ids: Sequence[int]) -> CacheLookup:
        ids = tuple(int(x) for x in input_ids)
        with self._lock:
            self._last_used += 1
            stamp = self._last_used
            hits = []
            pos = 0
            parent = ROOT_CACHE_ID
            while pos + self.block_size <= len(ids):
                tok = ids[pos:pos + self.block_size]
                e = self._exact(parent, tok)
                if e is None:
                    break
                e.last_used = stamp
                self._hot[e.location_id] = stamp
                hits.append(e)
                parent = e.id
                pos += self.block_size
            return CacheLookup(len(hits), pos, tuple(hits),
                               (self.block_size,) * len(hits), self._generation)

    def _validate(self, match: CacheLookup) -> None:
        if match._generation != self._generation:
            raise RuntimeError("cold KV V2 lookup was invalidated")

    def _invalidate(self, e: _EntryV2) -> bool:
        if not e.is_valid:
            return False
        if e.next_ids:
            raise RuntimeError("cannot evict non-leaf cold KV V2 entry")
        key = (e.block_hash, e.previous_id)
        ref = _Ref(e.location_id, e.id)
        if self.hash_index.get(key) == ref:
            self.hash_index.pop(key, None)
        self._children(e.previous_id).discard(e.id)
        self._by_id.pop(e.id, None)
        self._free_rows += 1
        e.id = ROOT_CACHE_ID; e.block_hash = b""; e.previous_id = ROOT_CACHE_ID
        e.last_used = 0; e.is_valid = False; e.token_ids = (); e.next_ids.clear()
        self._hot[e.location_id] = -1
        return True

    def _evict_tree(self, e: _EntryV2) -> int:
        """Recycle an entry along with every descendant that depends on it.

        _invalidate refuses parents, so the chain is dropped leaves-first; this
        is what actually frees a contiguous row range for the next sequence.
        """
        if not e.is_valid:
            return 0
        if not e.next_ids:
            return int(self._invalidate(e))
        by_id = self._by_id
        chain: List[_EntryV2] = []
        stack = [e]
        while stack:
            cur = stack.pop()
            chain.append(cur)
            for cid in tuple(cur.next_ids):
                child = by_id.get(cid)
                if child is not None:
                    stack.append(child)
        freed = 0
        for cur in reversed(chain):
            freed += int(self._invalidate(cur))
        return freed

    def _select(self, pinned_ids: Set[int], reserved: Set[int]) -> _EntryV2 | None:
        for e in self._entries:
            if not e.is_valid and e.location_id not in reserved:
                return e
        leaves = [e for e in self._entries if not e.next_ids and e.id not in pinned_ids
                  and e.location_id not in reserved]
        return min(leaves, key=lambda e: (e.last_used, e.id)) if leaves else None

    def _slab_starts(self) -> Set[int]:
        if len(self._slab_starts_cache) != len(self._slabs):
            starts: Set[int] = set()
            base = 0
            for slab in self._slabs:
                starts.add(base)
                base += slab.rows
            self._slab_starts_cache = starts
        return self._slab_starts_cache

    def _free_span(self, need: int, reserved: Set[int],
                   starts: Set[int], n: int) -> tuple[int, int] | None:
        """Amortised O(1) path for the common case of spare rows.

        A fully free stretch of `need` rows scores (-need, 0, 0), which is the
        best key _select_span can produce, so it can be returned without
        building the eviction cost map at all.  The cursor keeps the scan from
        restarting at row 0 while a cache is being filled.
        """
        if self._free_rows < need:
            return None
        entries = self._entries
        i = self._free_cursor if 0 <= self._free_cursor < n else 0
        run = 0
        run_start = i
        for _ in range(n):
            if i in starts:
                run = 0
            e = entries[i]
            if (not e.is_valid) and (e.location_id not in reserved):
                if run == 0:
                    run_start = i
                run += 1
                if run >= need:
                    self._free_cursor = i + 1
                    return run_start, need
            else:
                run = 0
            i += 1
            if i >= n:
                i = 0
                run = 0
        return None

    @staticmethod
    def _sliding_max(a: "np.ndarray", w: int) -> "np.ndarray":
        """Windowed maximum via doubling: O(n log w) vectorised, no python loop."""
        m = a
        step = 1
        while step * 2 <= w:
            m = np.maximum(m[:-step], m[step:])
            step *= 2
        out = len(a) - w + 1
        rest = w - step
        if rest:
            m = np.maximum(m[:out], m[rest:rest + out])
        return m[:out]

    def _select_span(self, need: int, pinned_ids: Set[int],
                     reserved: Set[int]) -> tuple[int, int] | None:
        """Pick the coldest recyclable run of rows, vectorised.

        Equivalent to _select_span_ref but replaces its four python passes over
        every entry with numpy work on the persistent `_hot` mirror.  Only the
        pinned/reserved ancestor chains are walked, since those are exactly the
        rows a pin protects.
        """
        n = len(self._entries)
        if n == 0 or need <= 0:
            return None
        starts = self._slab_starts()
        fast = self._free_span(need, reserved, starts, n)
        if fast is not None:
            return fast

        hot = self._hot[:n]
        cost = np.where(hot < 0, 0, np.maximum(hot, 1))
        blocked = np.zeros(n, dtype=bool)
        by_id = self._by_id
        seeds = []
        for cid in pinned_ids:
            e = by_id.get(cid)
            if e is not None:
                seeds.append(e)
        for loc in reserved:
            if 0 <= loc < n:
                blocked[loc] = True
                e = self._entries[loc]
                if e.is_valid:
                    seeds.append(e)
        for e in seeds:
            cur = e
            while cur is not None:
                loc = cur.location_id
                if blocked[loc]:
                    break  # this ancestor chain was already marked
                blocked[loc] = True
                cur = by_id.get(cur.previous_id)

        # Maximal runs of usable rows, further cut at slab boundaries.
        avail = (~blocked).astype(np.int8)
        d = np.diff(np.concatenate((np.zeros(1, np.int8), avail,
                                    np.zeros(1, np.int8))))
        run_lo = np.flatnonzero(d == 1)
        run_hi = np.flatnonzero(d == -1)
        cuts = np.array(sorted(starts), dtype=np.int64)
        segs = []
        for lo, hi in zip(run_lo.tolist(), run_hi.tolist()):
            a = int(lo)
            for c in cuts[(cuts > lo) & (cuts < hi)].tolist():
                segs.append((a, int(c)))
                a = int(c)
            segs.append((a, int(hi)))
        if not segs:
            return None

        best_len = max(min(need, b - a) for a, b in segs)
        best = None  # (evicts, worst, head)
        for a, b in segs:
            if min(need, b - a) != best_len:
                continue
            if b - a == best_len:
                heads = np.array([a], dtype=np.int64)
                ev = np.array([int(np.count_nonzero(cost[a:b]))], dtype=np.int64)
                wo = np.array([int(cost[a:b].max())], dtype=np.int64)
            else:
                seg = cost[a:b]
                nz = np.concatenate((np.zeros(1, np.int64),
                                     np.cumsum(seg != 0, dtype=np.int64)))
                ev = nz[best_len:] - nz[:-best_len]
                wo = self._sliding_max(seg, best_len)
                heads = a + np.arange(len(ev), dtype=np.int64)
            k = int(np.lexsort((wo, ev))[0])  # min evicts, then min worst, then lowest head
            cand = (int(ev[k]), int(wo[k]), int(heads[k]))
            if best is None or cand[:2] < best[:2]:
                best = cand
        if best is None:
            return None
        return best[2], best_len


    def _select_span_ref(self, need: int, pinned_ids: Set[int],
                     reserved: Set[int]) -> tuple[int, int] | None:
        """Reserve one physically contiguous row segment inside a single slab.

        A restore issues one DMA per contiguous run, so scattering a sequence's
        blocks turns a bandwidth-bound copy into a launch-bound loop.  Blocks are
        therefore claimed in the longest contiguous stretch available, preferring
        free rows and otherwise the least recently used evictable window.
        """
        n = len(self._entries)
        starts = self._slab_starts()
        fast = self._free_span(need, reserved, starts, n)
        if fast is not None:
            return fast
        by_id = self._by_id
        # A row may be recycled together with its whole descendant chain, so a
        # full sequence can be dropped at once instead of leaving its interior
        # blocks stuck as non-evictable parents (which is what fragments rows).
        order: List[_EntryV2] = []
        for e in self._entries:
            if not e.is_valid or e.previous_id in by_id:
                continue
            stack = [(e, False)]
            while stack:
                cur, done = stack.pop()
                if done:
                    order.append(cur)          # post-order: children first
                    continue
                stack.append((cur, True))
                for cid in cur.next_ids:
                    child = by_id.get(cid)
                    if child is not None:
                        stack.append((child, False))
        free_ok: Dict[int, bool] = {}
        hot: Dict[int, int] = {}
        for cur in order:
            ok = cur.id not in pinned_ids and cur.location_id not in reserved
            heat = cur.last_used
            for cid in cur.next_ids:
                child = by_id.get(cid)
                if child is None:
                    continue
                ok = ok and free_ok.get(child.id, False)
                heat = max(heat, hot.get(child.id, 0))
            free_ok[cur.id] = ok
            hot[cur.id] = heat
        cost: List[int | None] = [None] * n
        for idx, e in enumerate(self._entries):
            if e.location_id in reserved:
                continue
            if not e.is_valid:
                cost[idx] = 0
            elif free_ok.get(e.id, False):
                cost[idx] = max(1, hot.get(e.id, e.last_used))
        best: tuple[tuple[int, int, int], int, int] | None = None
        i = 0
        while i < n:
            if cost[i] is None:
                i += 1
                continue
            j = i + 1
            while j < n and cost[j] is not None and j not in starts:
                j += 1
            length = min(need, j - i)
            evicts = 0
            # Window maximum via a monotonic deque: the previous code re-scanned
            # the whole window at every slide position, making a long stretch
            # quadratic.
            dq: deque[int] = deque()
            for k in range(i, i + length):  # first window of the stretch
                c = cost[k] or 0
                evicts += int(c > 0)
                while dq and (cost[dq[-1]] or 0) <= c:
                    dq.pop()
                dq.append(k)
            worst = cost[dq[0]] or 0 if dq else 0
            key = (-length, evicts, worst)
            head = i
            for s in range(i + 1, j - length + 1):  # slide, keep the coldest one
                if key[1] == 0 and key[2] == 0:
                    break  # an all-free window is already optimal for this length
                out_c = cost[s - 1] or 0
                tail = s + length - 1
                in_c = cost[tail] or 0
                evicts += int(in_c > 0) - int(out_c > 0)
                while dq and dq[0] < s:
                    dq.popleft()
                while dq and (cost[dq[-1]] or 0) <= in_c:
                    dq.pop()
                dq.append(tail)
                worst = cost[dq[0]] or 0
                cand = (-length, evicts, worst)
                if cand < key:
                    key, head = cand, s
            if best is None or key < best[0]:
                best = (key, head, length)
            i = j
        if best is None:
            return None
        return best[1], best[2]

    def prepare_store(self, match: CacheLookup, input_ids: Sequence[int],
                      protect: Sequence[CacheLookup] = ()) -> StorePlanV2:
        """Open the append transaction for ONE sequence.

        Sequences are prefilled one at a time, so at most one plan is ever
        active -- the check below is an invariant, not a batch-size limit.

        ``protect`` carries the lookups of the other sequences of the same
        batch that have not been stored yet.  Their entries must survive this
        transaction: a sibling appends onto that chain later, and evicting it
        here would leave the sibling committing under a dead parent.
        """
        ids = tuple(int(x) for x in input_ids)
        with self._lock:
            self._validate(match)
            if self._active_plan is not None:
                raise RuntimeError("cold KV V2 store transaction already active")
            full_blocks = len(ids) // self.block_size
            first = match.block_count
            if first > full_blocks:
                raise RuntimeError("cold KV V2 match exceeds input")
            pinned = {e.id for e in match._entries}
            for other in protect:
                self._validate(other)
                pinned.update(e.id for e in other._entries)
            reserved: Set[int] = set()
            entries = []
            toks = []
            evicted = 0
            b = first
            while b < full_blocks:
                span = self._select_span(full_blocks - b, pinned, reserved)
                if span is None:
                    break
                head, length = span
                for k in range(head, head + length):
                    e = self._entries[k]
                    evicted += self._evict_tree(e)
                    reserved.add(e.location_id)
                    entries.append(e)
                    toks.append(ids[b*self.block_size:(b+1)*self.block_size])
                    b += 1
            plan = StorePlanV2(self, match, tuple(entries), tuple(toks), first, evicted)
            self._active_plan = plan
            return plan

    def _install(self, e: _EntryV2, parent: int, tokens: tuple[int, ...]) -> None:
        eid = self._next_id; self._next_id += 1
        h = self._hash_tokens(tokens)
        e.id = eid; e.block_hash = h; e.previous_id = parent
        e.last_used = self._last_used; e.token_ids = tokens; e.next_ids.clear(); e.is_valid = True
        self._hot[e.location_id] = self._last_used
        self.hash_index[(h, parent)] = _Ref(e.location_id, eid)
        self._by_id[eid] = e
        self._free_rows -= 1
        self._children(parent).add(eid)

    def _commit(self, plan: StorePlanV2) -> CacheStoreResult:
        with self._lock:
            if self._active_plan is not plan:
                raise RuntimeError("cold KV V2 store transaction is not active")
            try:
                self._validate(plan.match)
                parent = plan.match._entries[-1].id if plan.match._entries else ROOT_CACHE_ID
                for e, tokens in zip(plan.entries, plan.tokens):
                    self._install(e, parent, tokens)
                    parent = e.id
                return CacheStoreResult(len(plan.entries), plan.evicted)
            finally:
                self._active_plan = None

    def _abort(self, plan: StorePlanV2) -> None:
        with self._lock:
            if self._active_plan is plan:
                # Reserved entries were deliberately left invalid until commit.
                self._active_plan = None

    def restore_runs(self, match: CacheLookup, block_limit: int | None = None) -> list[KVRunV2]:
        with self._lock:
            self._validate(match)
            entries = list(match._entries[:block_limit])
            out: list[KVRunV2] = []
            b = 0
            while b < len(entries):
                e = entries[b]
                j = b + 1
                while j < len(entries):
                    n = entries[j]
                    if n.slab_id != e.slab_id or n.row != e.row + (j - b):
                        break
                    j += 1
                out.append(KVRunV2(b, j - b, self._slabs[e.slab_id], e.row))
                b = j
            return out

    # Legacy callback API remains intentionally available for diagnostics only.
    def restore(self, match: CacheLookup, load_block) -> None:
        raise RuntimeError("Cold KV V2 requires restore_runs/load_kv_v2")

    def store(self, match: CacheLookup, input_ids: Sequence[int], store_block) -> CacheStoreResult:
        raise RuntimeError("Cold KV V2 requires prepare_store/capture/commit")

    def clear(self) -> None:
        with self._lock:
            if self._active_plan is not None:
                raise RuntimeError("cannot clear cold KV V2 during an active store transaction")
            self.hash_index.clear(); self._root_next_ids.clear(); self._generation += 1
            for e in self._entries:
                e.id = ROOT_CACHE_ID; e.block_hash = b""; e.previous_id = ROOT_CACHE_ID
                e.last_used = 0; e.is_valid = False; e.token_ids = (); e.next_ids.clear()
            self._hot[:] = -1

    @property
    def token_count(self) -> int:
        with self._lock:
            return sum(e.valid_tokens for e in self._entries if e.is_valid)

    @property
    def capacity_blocks(self) -> int:
        with self._lock:
            return len(self._entries)

    @property
    def capacity_bytes(self) -> int:
        return self.capacity_blocks * self.block_bytes

    @property
    def valid_slot_count(self) -> int:
        with self._lock:
            return sum(int(e.is_valid) for e in self._entries)

    @property
    def allocation_error(self):
        return self._allocation_error
