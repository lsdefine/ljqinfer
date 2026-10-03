"""Unified paged fixed-width decode: one B-agnostic state machine.

The scheduler supplies rows, per-row page tables and candidate tokens. The B1
graph replays against a rebound paged view; MTP verify, commit and streaming
share one fixed-width state machine.
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, List, Optional, Sequence, Tuple, Union

import torch

from model.runtime import KVCache
from model.blocks import _rmsnorm
from model.prefill import BatchPrefillState, lm_head, prefill_batch_chunked
from model.decode_backend import (ensure_mtp_graphs, mtp_draft_chain_batched,
                                  mtp_forward, replay_decode)

from model.config import *  # noqa: F401,F403 — 本机固定常量, 全部写死

_MTP_GREEDY = os.getenv("LJQ_MTP_GREEDY", "0") == "1"


if TYPE_CHECKING:
    from model.model import Engine


@dataclass
class BatchDecodeState:
    """Fixed-slot batched decode state with provisional width-Q computation.

    Every graph replay computes ``graph.Q`` rows per sequence, while verification
    may commit an independent prefix for each sequence, so logical lengths may
    diverge after every step.
    ``k0s[r][i]`` is a persistent CUDA int32 scalar for sequence ``i`` on
    rank ``r``.  Keeping the logical prefix length on device is required by
    the graph-safe paged MLA leaf; host ``lengths`` remains bookkeeping only.
    """
    caches: List[KVCache]
    lengths: List[int]
    capacities: List[int]
    graph: object = field(repr=False)
    page_indices: List[List[int]] = field(default_factory=list)
    k0s: List[List[torch.Tensor]] = field(default_factory=list)
    tables: List[List[torch.Tensor]] = field(default_factory=list)
    mtp_caches: List[KVCache] = field(default_factory=list)
    released: bool = False
    _decode_pending: bool = field(default=False, repr=False)

    @property
    def batch_size(self) -> int:
        return len(self.caches)

    def release(self) -> None:
        if self.released:
            return
        for cache in self.caches:
            cache.length = 0
        for cache in self.mtp_caches:
            cache.length = 0
        self.mtp_caches.clear()
        self.graph = None
        self.released = True


def _merge_decode_rows(engine: "Engine", left: BatchDecodeState,
                       right: BatchDecodeState) -> BatchDecodeState:
    """Append committed rows without moving KV or releasing either lease."""
    if left.released or right.released:
        raise RuntimeError("cannot merge a released decode state")
    if left._decode_pending or right._decode_pending:
        raise RuntimeError("decode rows may merge only at a commit boundary")
    total = left.batch_size + right.batch_size
    graph = engine.base_graphs.get((total, Q_MAX))
    if graph is None:
        raise ValueError(f"no resident B{total}Q{Q_MAX} graph")
    pages = left.page_indices + right.page_indices
    physical = [page for mapping in pages for page in mapping]
    if len(set(physical)) != len(physical):
        raise ValueError("boarding page mappings overlap live epoch pages")
    return BatchDecodeState(
        caches=left.caches + right.caches,
        lengths=left.lengths + right.lengths,
        capacities=left.capacities + right.capacities,
        graph=graph,
        page_indices=pages,
        k0s=[a + b for a, b in zip(left.k0s, right.k0s)],
        tables=[a + b for a, b in zip(left.tables, right.tables)],
        mtp_caches=left.mtp_caches + right.mtp_caches,
    )


def select_decode_rows(engine: "Engine", state: BatchDecodeState,
                       rows: Sequence[int]) -> BatchDecodeState:
    """Return a smaller committed view without moving or releasing any KV page."""
    if state.released:
        raise RuntimeError("cannot select rows from a released decode state")
    if state._decode_pending:
        raise RuntimeError("cannot select rows during a provisional decode")
    selected = [int(row) for row in rows]
    if not selected or len(set(selected)) != len(selected):
        raise ValueError("rows must be unique and non-empty")
    if any(row < 0 or row >= state.batch_size for row in selected):
        raise IndexError("decode row outside the active batch")
    graph = engine.base_graphs.get((len(selected), Q_MAX))
    if graph is None:
        raise ValueError(f"no resident B{len(selected)}Q{Q_MAX} graph")
    return BatchDecodeState(
        caches=[state.caches[row] for row in selected],
        lengths=[state.lengths[row] for row in selected],
        capacities=[state.capacities[row] for row in selected],
        graph=graph,
        page_indices=[state.page_indices[row] for row in selected],
        k0s=[[rank[row] for row in selected] for rank in state.k0s],
        tables=[[rank[row] for row in selected] for rank in state.tables],
        mtp_caches=[state.mtp_caches[row] for row in selected],
    )


def _batch_request_caches(engine: "Engine", lengths: Sequence[int],
                          capacities: Optional[Sequence[int]] = None, *,
                          page_indices: Optional[Sequence[Sequence[int]]] = None
                          ) -> BatchPrefillState:
    """Partition the existing physical KV pool into disjoint whole-page runs.

    ``capacities`` are total per-sequence token budgets (prefill plus all future
    decode tokens).  All pages are assigned now; there is no later allocation.
    """
    kv = engine.kv
    lengths = [int(n) for n in lengths]
    capacities = lengths if capacities is None else [int(n) for n in capacities]
    if len(capacities) != len(lengths):
        raise ValueError("capacities must have one value per sequence")
    if any(cap < n for cap, n in zip(capacities, lengths)):
        raise ValueError("each capacity must cover its prefill length")
    pool_pages = kv.logical_pool_pages
    page_counts = [(cap + KV_PAGE_SIZE - 1) // KV_PAGE_SIZE for cap in capacities]
    need_pages = sum(page_counts)
    if need_pages > pool_pages:
        raise ValueError(
            f"batch needs {need_pages} KV pages, engine pool has {pool_pages}; "
            "load the engine with a larger max_len")

    offsets = [0]
    for n in lengths:
        offsets.append(offsets[-1] + int(n))
    supplied = ([[] for _ in lengths] if page_indices is None
                else [[int(page) for page in pages] for pages in page_indices])
    if len(supplied) != len(lengths):
        raise ValueError("page_indices must contain one mapping per sequence")
    used = [page for pages in supplied for page in pages]
    if len(set(used)) != len(used) or any(page < 0 or page >= pool_pages
                                         for page in used):
        raise ValueError("preloaded page mappings overlap or leave the KV pool")
    if any(len(pages) > count for pages, count in zip(supplied, page_counts)):
        raise ValueError("preloaded page mapping exceeds sequence capacity")
    free = iter(page for page in range(pool_pages) if page not in set(used))
    mappings: List[List[int]] = []
    for pages, count in zip(supplied, page_counts):
        mapping = list(pages)
        mapping.extend(next(free) for _ in range(count - len(mapping)))
        mappings.append(mapping)

    caches: List[KVCache] = []
    for capacity, mapping in zip(capacities, mappings):
        # The owner creates the request-local page-table view.  TP8 shares
        # both persistent pool and attention workspace without copying them.
        caches.append(kv.bind_request(capacity, mapping))
    return BatchPrefillState(caches=caches, offsets=offsets,
                             lengths=lengths, capacities=list(capacities),
                             page_indices=mappings)


def decode_candidates(tokens: torch.Tensor,
                      state: BatchDecodeState) -> Tuple[torch.Tensor, torch.Tensor]:
    """Provisionally append one width-Q candidate block per live sequence.

    ``tokens`` is integer ``[B,Q]`` where ``Q == state.graph.Q``.  Attention is
    isolated by each sequence's page table and current length.  A successful
    replay advances all logical lengths by Q; ``commit_candidates`` must then
    commit a prefix of 1..Q tokens independently for every row.
    """
    if state.released:
        raise RuntimeError("batched decode state has been released")
    if state._decode_pending:
        raise RuntimeError("previous decode step must be committed before replay")
    if state.batch_size == 0:
        raise ValueError("batched decode requires at least one sequence")
    Q = int(state.graph.Q)
    if Q < 1:
        raise ValueError(f"decode graph has invalid Q={Q}")
    ids = torch.as_tensor(tokens, dtype=torch.long)
    if ids.dim() != 2 or tuple(ids.shape) != (state.batch_size, Q):
        raise ValueError(f"batched decode expects tokens [{state.batch_size},{Q}]")
    flat_ids = ids.to(device="cpu").contiguous().reshape(-1)
    if int(flat_ids.min()) < 0 or int(flat_ids.max()) >= VOCAB:
        raise IndexError("token id outside vocabulary")
    old_lengths = [int(n) for n in state.lengths]
    if len(old_lengths) != state.batch_size or len(state.capacities) != state.batch_size:
        raise ValueError("batched decode state metadata size mismatch")
    if any(n != cache.length for n, cache in zip(old_lengths, state.caches)):
        raise RuntimeError("batched decode state/cache lengths diverged")
    if any(n + Q > cap for n, cap in zip(old_lengths, state.capacities)):
        raise ValueError(f"Q={Q} decode exceeds one or more sequence capacities")
    if len(state.k0s) != TP or any(len(v) != state.batch_size for v in state.k0s):
        raise RuntimeError("batch decode device K0 metadata is missing")

    logits, hidden = replay_decode(
        state.graph, state, flat_ids.reshape(state.batch_size, Q))

    # Commit host bookkeeping only after a successful full replay.
    new_lengths = [n + Q for n in old_lengths]
    for cache, n in zip(state.caches, new_lengths):
        cache.length = n
    state.lengths[:] = new_lengths
    state._decode_pending = True
    return logits, hidden


def commit_candidates(state: BatchDecodeState,
                      accepted: Sequence[int]) -> None:
    """Commit a 1..Q prefix after a provisional BxQ candidate replay."""
    if state.released:
        raise RuntimeError("batched decode state has been released")
    if not state._decode_pending:
        raise RuntimeError("no provisional decode step to commit")
    Q = int(state.graph.Q)
    counts = [int(n) for n in accepted]
    if len(counts) != state.batch_size:
        raise ValueError("accepted must have one value per sequence")
    if any(n < 1 or n > Q for n in counts):
        raise ValueError(f"each accepted count must be in [1,{Q}]")
    if any(int(length) != cache.length for length, cache in zip(state.lengths, state.caches)):
        raise RuntimeError("batched decode state/cache lengths diverged")

    new_lengths = [int(length) - (Q - count)
                   for length, count in zip(state.lengths, counts)]
    if any(length < 0 for length in new_lengths):
        raise ValueError("batched commit would make a sequence length negative")
    for cache, length in zip(state.caches, new_lengths):
        cache.length = length
    state.lengths[:] = new_lengths
    state._decode_pending = False


def _alloc_mtp_caches(engine: Engine,
                      capacities: Sequence[int], *,
                      page_indices: Optional[Sequence[Sequence[int]]] = None
                      ) -> List[KVCache]:
    """Bind disjoint request-local page runs in the one-layer MTP pool.

    Used both during chunked batch prefill (so interior MTP slots can be
    written before decode) and as a lazy fallback when prime is invoked on a
    state that never bound MTP views.
    """
    owner = engine.mtp_kv
    if owner is None:
        raise RuntimeError("MTP head/cache not loaded")
    page_counts = [(int(cap) + KV_PAGE_SIZE - 1) // KV_PAGE_SIZE
                   for cap in capacities]
    pool_pages = owner.logical_pool_pages
    if sum(page_counts) > pool_pages:
        raise ValueError("batched MTP capacities exceed the private KV pool")
    mappings = ([list(range(sum(page_counts[:i]), sum(page_counts[:i + 1])))
                 for i in range(len(page_counts))]
                if page_indices is None
                else [[int(page) for page in pages] for pages in page_indices])
    if len(mappings) != len(capacities) or any(
            len(mapping) != count
            for mapping, count in zip(mappings, page_counts)):
        raise ValueError("MTP page mappings must match sequence capacities")
    return [owner.bind_request(int(capacity), mapping)
            for capacity, mapping in zip(capacities, mappings)]


def _ensure_batched_mtp_caches(engine: Engine,
                                state: BatchDecodeState) -> None:
    """Bind request-local MTP views if the prefill stage did not already."""
    if state.mtp_caches:
        if len(state.mtp_caches) != state.batch_size:
            raise RuntimeError("partial batched MTP cache state")
        return
    state.mtp_caches.extend(_alloc_mtp_caches(
        engine, state.capacities, page_indices=state.page_indices))


def _recurse_drafts(engine: Engine, cache: KVCache, token: int,
                    hidden: torch.Tensor, count: int = 3) -> List[int]:
    """Run one MTP head recurrently and return ``count`` greedy drafts."""
    drafts: List[int] = []
    for _ in range(count):
        token, _, hidden = mtp_forward(
            engine, [token], cache.length, hidden, cache=cache,
            return_hidden=True)
        drafts.append(token)
    return drafts


def batched_mtp_prime(engine: Engine, state: BatchDecodeState,
                      sequences: Sequence[torch.Tensor],
                      residuals: Sequence[torch.Tensor]) -> Tuple[List[int], List[List[int]]]:
    """Prime each shifted MTP stream and recursively produce ``Q_MAX - 1`` drafts."""
    if state.released or state._decode_pending:
        raise RuntimeError("batched MTP prime requires an idle live state")
    if len(sequences) != state.batch_size or len(residuals) != state.batch_size:
        raise ValueError("prime inputs must have one row per sequence")
    _ensure_batched_mtp_caches(engine, state)
    ensure_mtp_graphs(engine)
    base_tokens: List[int] = []
    draft_rows: List[List[int]] = []
    for i, (ids, resid, cache) in enumerate(zip(sequences, residuals,
                                                state.mtp_caches)):
        ids = torch.as_tensor(ids, dtype=torch.long).reshape(-1)
        length = int(state.lengths[i])
        if ids.numel() != length:
            raise ValueError("prime sequence length disagrees with state")
        if resid.dim() != 2 or resid.shape[-1] != D or resid.shape[0] < 1:
            raise ValueError("prime residual must be [*, D] with at least one row")
        base = int(_pick_tokens(lm_head(engine.rt, engine.w, resid)).item())
        last_hidden = _rmsnorm(resid[-1:], engine.w.final_norm).to(torch.float16)
        mtp_len = int(cache.length)
        if mtp_len == 0:
            if resid.shape[0] != length:
                raise ValueError(
                    "full-stream MTP prime requires residual length == prompt length")
            hidden = _rmsnorm(resid, engine.w.final_norm).to(torch.float16)
            stream = torch.cat([ids[1:], torch.tensor([base], dtype=torch.long)])
            first, _, mtp_hidden = mtp_forward(
                engine, stream, 0, hidden, cache=cache, return_hidden=True)
        elif mtp_len in (length - 1, length):
            first, _, mtp_hidden = mtp_forward(
                engine, [base], length - 1, last_hidden, cache=cache,
                return_hidden=True)
        else:
            raise RuntimeError(
                f"MTP cache length {mtp_len} incompatible with prompt length {length}")
        # Prime has already produced the first draft.  Keep the compatibility
        # path here because long prompt streams are not resident CUDA graphs;
        # steady-state decode below uses the fused device-recursive chain.
        drafts = [first]
        drafts.extend(_recurse_drafts(
            engine, cache, first, mtp_hidden, count=Q_MAX - 2))
        base_tokens.append(base)
        draft_rows.append(drafts)
    return base_tokens, draft_rows


def _pick_tokens(logits: torch.Tensor) -> torch.Tensor:
    """Pick one token per logits row; sampling is temperature=1 by default."""
    if _MTP_GREEDY:
        return logits.argmax(dim=-1)
    shape = logits.shape[:-1]
    flat = logits.reshape(-1, logits.shape[-1])
    probs = torch.softmax(flat, dim=-1)
    return torch.multinomial(probs, 1).reshape(shape)


def _sample(logits: torch.Tensor, base_tokens: Sequence[int],
            draft_tokens: Sequence[Sequence[int]]):
    """Accept drafts while they match temperature=1 Base samples."""
    predicted = _pick_tokens(logits).tolist()
    emitted, next_base, mtp_inputs, accepted = [], [], [], []
    for base, drafts, row in zip(base_tokens, draft_tokens, predicted):
        if len(drafts) != Q_MAX - 1:
            raise ValueError(f"expected {Q_MAX - 1} drafts, got {len(drafts)}")
        count = 1
        for offset, draft in enumerate(drafts):
            if int(row[offset]) != int(draft):
                break
            count += 1
        following = int(row[count - 1])
        emitted.append([int(base)] + [int(x) for x in drafts[:count - 1]])
        next_base.append(following)
        mtp_inputs.append([int(x) for x in drafts[:count - 1]] + [following])
        accepted.append(count)
    return emitted, next_base, mtp_inputs, accepted


def mtp_step(engine: Engine, state: BatchDecodeState,
             base_tokens: Sequence[int],
             draft_tokens: Sequence[Sequence[int]]) -> Tuple[List[List[int]], List[int], List[List[int]]]:
    """Verify one base token plus ``Q_MAX - 1`` drafts, then advance the MTP chain."""
    if state.graph.Q != Q_MAX:
        raise RuntimeError(f"fixed-width MTP policy requires graph Q={Q_MAX}")
    if len(state.mtp_caches) != state.batch_size:
        raise RuntimeError("batched MTP must be primed before verify")
    if len(base_tokens) != state.batch_size or len(draft_tokens) != state.batch_size:
        raise ValueError("verify tokens must have one row per sequence")
    candidates = torch.tensor(
        [[int(base), *map(int, drafts)]
         for base, drafts in zip(base_tokens, draft_tokens)], dtype=torch.long)
    base_lengths = [int(length) for length in state.lengths]
    base_cache_lengths = [int(cache.length) for cache in state.caches]
    mtp_lengths = [int(cache.length) for cache in state.mtp_caches]
    pending = bool(state._decode_pending)
    try:
        logits, hidden = decode_candidates(candidates, state)
        emitted, next_base, mtp_inputs, accepted = _sample(
            logits, base_tokens, draft_tokens)
        chain_tokens: List[List[int]] = []
        chain_starts: List[int] = []
        chain_hidden: List[torch.Tensor] = []
        for i, (cache, tokens, count) in enumerate(
                zip(state.mtp_caches, mtp_inputs, accepted)):
            # Base replay is provisional at L+Q.  Rewind the recurrent MTP
            # tail slots to committed L; the fused chain graph rebuilds the
            # accepted prefix and drafts the next tokens for every row.
            start = int(state.lengths[i]) - Q_MAX
            if start < 0:
                raise RuntimeError("provisional base length is smaller than Q")
            cache.length = start
            chain_tokens.append(list(map(int, tokens)))
            chain_starts.append(start)
            chain_hidden.append(hidden[i, :count])
        next_drafts = mtp_draft_chain_batched(
            engine, chain_tokens, chain_starts, chain_hidden,
            list(state.mtp_caches))
        commit_candidates(state, accepted)
        return emitted, next_base, next_drafts
    except Exception:
        # Provisional KV bytes past these lengths are unreachable and will be
        # overwritten by a retry.  Restore all host-side transaction metadata.
        state.lengths[:] = base_lengths
        for cache, length in zip(state.caches, base_cache_lengths):
            cache.length = length
        for cache, length in zip(state.mtp_caches, mtp_lengths):
            cache.length = length
        state._decode_pending = pending
        raise


def generate_mtp_batch(engine: Engine, input_ids,
                       max_new_tokens: Union[int, Sequence[int]] = 16,
                       eos_token_id: int = EOS_DEFAULT,
                       stats: Optional[dict] = None, *,
                       cancel_events: Optional[Sequence[threading.Event]] = None,
                       emit: Optional[Callable[[int, List[int]], None]] = None,
                       on_prefill: Optional[Callable[[], None]] = None,
                       on_prefill_state: Optional[Callable[[BatchPrefillState], None]] = None,
                       page_indices: Optional[Sequence[Sequence[int]]] = None,
                       loaded_lengths: Optional[Sequence[int]] = None,
                       select_active_rows: Optional[
                           Callable[[Sequence[int], Sequence[bool]], Sequence[int]]
                       ] = None,
                       board_row: Optional[Callable[[int], Optional[tuple]]] = None,
                       on_boarded_state: Optional[
                           Callable[[int, BatchPrefillState], None]
                       ] = None,
                       on_boarded: Optional[Callable[[int, int], None]] = None,
                       boarding_interval_steps: int = 128
                       ) -> List[List[int]]:
    """Greedy fixed-width MTP generation with policy-controlled active rows.

    ``select_active_rows`` is called only at committed step boundaries.  It may
    reorder or retain rows, but cannot introduce rows or drop a live row.  The
    original batch keeps ownership of every KV page until all rows finish.
    """
    sequences = [torch.as_tensor(ids, dtype=torch.long).reshape(-1)
                 for ids in input_ids]
    B = len(sequences)
    graph = engine.base_graphs.get((B, Q_MAX))
    if graph is None:
        raise ValueError(f"no resident B{B}Q{Q_MAX} graph")
    if any(ids.numel() == 0 for ids in sequences):
        raise ValueError("input sequences must be non-empty")
    limits = ([int(max_new_tokens)] * B if isinstance(max_new_tokens, int)
              else [int(n) for n in max_new_tokens])
    if len(limits) != B or any(n < 0 for n in limits):
        raise ValueError("max_new_tokens must contain one non-negative limit per row")
    cancels = (list(cancel_events) if cancel_events is not None
               else [threading.Event() for _ in range(B)])
    if len(cancels) != B:
        raise ValueError("cancel_events must contain one event per row")
    boarding_interval_steps = int(boarding_interval_steps)
    if boarding_interval_steps <= 0:
        raise ValueError("boarding_interval_steps must be positive")
    if all(n == 0 or event.is_set() for n, event in zip(limits, cancels)):
        if stats is not None:
            stats.update(steps=0, row_steps=[0] * B, accepts=[0] * B,
                         row_decode_seconds=[0.0] * B)
        return [[] for _ in range(B)]

    Q = int(graph.Q)
    capacities = [ids.numel() + Q * limit + Q
                  for ids, limit in zip(sequences, limits)]
    # Memory-bounded: rows prefill serially in chunks, decode stays batched.
    residuals, prefill_state = prefill_batch_chunked(
        engine, sequences, max_lengths=capacities,
        page_indices=page_indices, loaded_lengths=loaded_lengths)
    state: Optional[BatchDecodeState] = None
    lease_caches: List[KVCache] = []
    lease_mtp_caches: List[KVCache] = []
    try:
        if on_prefill_state is not None:
            on_prefill_state(prefill_state)
        state = prefill_state.decode_state(graph)
        lease_caches = list(state.caches)
        lease_mtp_caches = list(state.mtp_caches)
        base, draft = batched_mtp_prime(engine, state, sequences, residuals)
        lease_mtp_caches = list(state.mtp_caches)
        if on_prefill is not None:
            for device in engine.rt.devices:
                torch.cuda.synchronize(device)
            on_prefill()
        outputs: List[List[int]] = [[] for _ in range(B)]
        done = [limit == 0 or event.is_set()
                for limit, event in zip(limits, cancels)]
        accepts = [0] * B
        row_steps = [0] * B
        active_rows = list(range(B))

        def board_one() -> bool:
            """Synchronously append one pre-restored row at a commit boundary."""
            nonlocal state, base, draft
            if board_row is None or state.batch_size >= 4:
                return False
            original_i = len(sequences)
            payload = board_row(original_i)
            if payload is None:
                return False
            ids, limit, event, mapping, loaded = payload
            ids = torch.as_tensor(ids, dtype=torch.long).reshape(-1)
            limit, loaded = int(limit), int(loaded)
            if ids.numel() == 0 or limit < 0 or loaded < 0:
                raise ValueError("invalid boarded row")
            capacity = int(ids.numel()) + Q * limit + Q
            residuals_new, prefill_new = prefill_batch_chunked(
                engine, [ids], max_lengths=[capacity],
                page_indices=[mapping], loaded_lengths=[loaded])
            if on_boarded_state is not None:
                on_boarded_state(original_i, prefill_new)
            right = prefill_new.decode_state(
                engine.base_graphs[(1, Q_MAX)])
            right_base, right_draft = batched_mtp_prime(
                engine, right, [ids], residuals_new)
            state = _merge_decode_rows(engine, state, right)
            lease_caches.extend(right.caches)
            lease_mtp_caches.extend(right.mtp_caches)
            sequences.append(ids)
            limits.append(limit)
            cancels.append(event)
            outputs.append([])
            done.append(limit == 0 or event.is_set())
            accepts.append(0)
            row_steps.append(0)
            row_decode_seconds.append(0.0)
            active_rows.append(original_i)
            base.extend(right_base)
            draft.extend(right_draft)
            if on_boarded is not None:
                on_boarded(original_i, state.batch_size)
            return True

        def policy_selection() -> List[int]:
            if select_active_rows is None:
                return list(active_rows)
            selected_rows = [int(row) for row in
                             select_active_rows(tuple(active_rows), tuple(done))]
            if len(set(selected_rows)) != len(selected_rows):
                raise ValueError("active-row policy returned duplicate rows")
            active_set = set(active_rows)
            if any(row not in active_set for row in selected_rows):
                raise ValueError("active-row policy introduced an inactive row")
            live_rows = {row for row in active_rows if not done[row]}
            if not live_rows.issubset(selected_rows):
                raise ValueError("active-row policy dropped a live row")
            return selected_rows

        if select_active_rows is not None and any(done):
            selected_rows = policy_selection()
            selected = {row: i for i, row in enumerate(active_rows)}
            selected_indices = [selected[row] for row in selected_rows]
            active_rows = selected_rows
            state = select_decode_rows(engine, state, selected_indices)
            base = [base[i] for i in selected_indices]
            draft = [draft[i] for i in selected_indices]
        steps = 0
        row_decode_seconds = [0.0] * len(sequences)
        while active_rows:
            step_started = time.perf_counter()
            emitted, base, draft = mtp_step(
                engine, state, base, draft)
            step_seconds = time.perf_counter() - step_started
            steps += 1
            for original_i in active_rows:
                row_steps[original_i] += 1
                row_decode_seconds[original_i] += step_seconds
            for active_i, row in enumerate(emitted):
                original_i = active_rows[active_i]
                if done[original_i] or cancels[original_i].is_set():
                    done[original_i] = True
                    continue
                accepts[original_i] += max(0, len(row) - 1)
                chunk: List[int] = []
                for token in row:
                    if token == eos_token_id:
                        done[original_i] = True
                        break
                    if len(outputs[original_i]) >= limits[original_i]:
                        done[original_i] = True
                        break
                    outputs[original_i].append(token)
                    chunk.append(token)
                    if len(outputs[original_i]) >= limits[original_i]:
                        done[original_i] = True
                        break
                if chunk and not cancels[original_i].is_set():
                    if emit is not None:
                        emit(original_i, chunk)
                elif cancels[original_i].is_set():
                    done[original_i] = True
            for i, event in enumerate(cancels):
                if event.is_set():
                    done[i] = True
            if all(done):
                break
            if select_active_rows is not None:
                selected_rows = policy_selection()
                if selected_rows != active_rows:
                    selected = {row: i for i, row in enumerate(active_rows)}
                    selected_indices = [selected[row] for row in selected_rows]
                    active_rows = selected_rows
                    if not active_rows:
                        break
                    state = select_decode_rows(engine, state, selected_indices)
                    base = [base[i] for i in selected_indices]
                    draft = [draft[i] for i in selected_indices]
            if (board_row is not None and
                    steps % boarding_interval_steps == 0):
                while state.batch_size < 4 and board_one():
                    pass
        if stats is not None:
            stats.update(steps=steps, row_steps=row_steps, accepts=accepts,
                         row_decode_seconds=row_decode_seconds)
        return outputs
    finally:
        if state is not None:
            for cache in lease_caches:
                cache.length = 0
            mtp_lease = list(lease_mtp_caches)
            for cache in state.mtp_caches:
                if not any(cache is owned for owned in mtp_lease):
                    mtp_lease.append(cache)
            for cache in mtp_lease:
                cache.length = 0
            state.mtp_caches.clear()
            state.graph = None
            state.released = True
        elif not prefill_state.released:
            prefill_state.release()
