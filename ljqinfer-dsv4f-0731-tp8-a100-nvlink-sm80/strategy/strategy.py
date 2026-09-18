#!/usr/bin/env python3
"""Block-granular cold KV reuse around one streamed model execution slot.

The strategy imports only the stable model-layer API and treats the model as an
opaque execution-slot provider.
"""
from __future__ import annotations

import os

from dataclasses import asdict, dataclass
from queue import Queue
import threading
import time
import traceback
from typing import Optional, Sequence

import torch

from strategy.batch_scheduler import BatchJob, BatchScheduler
from strategy.cold_kv_cache import ColdKVBackend, KVLayout, PinnedMemoryKVCache
from strategy.cold_kv_v2 import KVFieldSpec, PinnedMemoryKVCacheV2
from model.model_api import BoardingRequest, KV_FORMAT, ModelExecution


BOARDING_GRACE_S = 0.10
# Sliding-window (128) attention only reads pool1 near the sequence tail;
# loading the last two 2048-token pages always covers the window.
POOL1_TAIL_PAGES = 2


@dataclass(frozen=True, slots=True)
class QueryMetrics:
    """Synchronous strategy work completed before streamed decode starts.

    Rates are fractions in ``[0, 1]``.  ``submit_to_prefill_seconds`` measures
    from entry into :meth:`Strategy.query` until model prefill has completed on
    every execution device.  ``effective_prefill_tps`` uses all input tokens,
    while ``model_prefill_tps`` uses only tokens actually scheduled for prefill.
    """

    input_tokens: int
    cache_block_size: int
    cache_hit_tokens: int
    cache_hit_rate: float
    prefill_tokens: int
    cache_stored_blocks: int
    cache_evicted_blocks: int
    cache_lookup_seconds: float
    cache_load_seconds: float
    model_prefill_seconds: float
    cache_store_seconds: float
    submit_to_prefill_seconds: float
    strategy_seconds: float
    effective_prefill_tps: float
    model_prefill_tps: float


class RequestHandle:
    """Cancellation state for one pending or running request."""

    def __init__(self, request_id: str):
        self.request_id = request_id
        self.cancel_event = threading.Event()
        self.semantic_stop = False
        self.submitted_at = time.perf_counter()
        self.state = "pending"

    def cancel(self) -> bool:
        first = not self.cancel_event.is_set()
        self.cancel_event.set()
        return first

    def stop_at_semantic_eos(self) -> bool:
        """Stop decode cooperatively without classifying normal EOS as disconnect."""
        self.semantic_stop = True
        first = not self.cancel_event.is_set()
        self.cancel_event.set()
        return first


_MAX_TOKEN_ID = None


def _vocab_size():
    """Vocabulary size of the served model (cached, defaults to DSv4's)."""
    global _MAX_TOKEN_ID
    if _MAX_TOKEN_ID is None:
        try:
            from model.args_dsv4 import make_args
            _MAX_TOKEN_ID = int(make_args().vocab_size)
        except Exception:
            _MAX_TOKEN_ID = 129280
    return _MAX_TOKEN_ID


class Strategy:
    """One execution slot plus a chained, block-granular cold KV cache."""

    def __init__(self, model: ModelExecution,
                 cold_kv: Optional[ColdKVBackend] = None):
        self.model = model
        if cold_kv is None:
            fields = tuple(KVFieldSpec(key, tuple(shape), dtype)
                           for key, shape, dtype in model.kv_v2_fields)
            # Dev knob: pinning the full 80 GiB target dominates startup time.
            # Default is unchanged (80); export LJQINFER_COLD_KV_GB=8 to iterate fast.
            target_bytes = int(float(os.environ.get("LJQINFER_COLD_KV_GB", "80")) * (1 << 30))
            cold_kv = PinnedMemoryKVCacheV2(
                fields, block_size=KV_FORMAT.block_tokens,
                target_bytes=target_bytes)
        self.cold_kv = cold_kv
        self._cold_kv_v2 = isinstance(self.cold_kv, PinnedMemoryKVCacheV2)
        caps = model.capabilities
        self._scheduler = BatchScheduler(
            max_batch_size=caps.max_batch_size,
            pool_pages=caps.kv_pool_pages)
        self._lock = threading.Lock()
        self._worker: Optional[threading.Thread] = None
        self._pending = 0
        self._seq = 0
        self._handles = {}
        self._span_chunk = caps.kv_page_size
        self._kv_sections = model.kv_block_sections
        self._span_staging = None
        if not self._cold_kv_v2:
            main_n, aux_n, tail_n = self._kv_sections
            blocks_per_page = self._span_chunk // KV_FORMAT.block_tokens
            compact_numel = blocks_per_page * (main_n + aux_n) + tail_n
            self._span_staging = torch.empty(
                compact_numel, dtype=KV_FORMAT.dtype,
                device="cpu", pin_memory=True)
            print(f"[cold-kv] compact staging preallocated tokens={self._span_chunk} "
                  f"main={main_n} aux={aux_n} tail={tail_n} "
                  f"mib={self._span_staging.numel() * self._span_staging.element_size() / 2**20:.2f}",
                  flush=True)

    @property
    def pending(self) -> int:
        """Requests admitted but not yet finished (includes the running one)."""
        with self._lock:
            return self._pending

    def query(self, input_ids: Sequence[int],
              max_new_tokens: int = 16,
              temperature: float = 1.0) -> Queue:
        """Admit one request and return its output Queue immediately.

        The execution slot is single-threaded, so requests are served strictly
        FIFO by one worker.  This call never blocks on a busy slot: it returns
        right away and the caller simply waits on the Queue.

        The Queue streams ``{"token_ids": [int]}`` events, then a final
        ``{"end": True}``.  On failure the final event carries ``error``.
        The ``metrics`` attribute is populated before the first token event
        and is only meaningful once an event has been received.
        """
        ids = tuple(int(token) for token in input_ids)
        if not ids:
            raise ValueError("input_ids must not be empty")
        limit = _vocab_size()
        bad = next((t for t in ids if t < 0 or t >= limit), None)
        if bad is not None:
            # An out-of-range id would trip a device-side assert inside the
            # embedding gather and poison the CUDA context of every rank.
            raise ValueError("token id %d out of range [0, %d)" % (bad, limit))
        n = int(max_new_tokens)
        if n < 0:
            raise ValueError("max_new_tokens must be non-negative")
        temp = float(temperature)
        if temp < 0.0:
            raise ValueError("temperature must be non-negative")

        out: Queue = Queue()
        out.metrics = None
        with self._lock:
            self._seq += 1
            handle = RequestHandle(f"req_{self._seq:08d}")
            self._handles[handle.request_id] = handle
            self._pending += 1
            depth = self._pending
            self._start_worker()
            self._scheduler.put(BatchJob(
                input_ids=ids,
                max_new_tokens=n,
                temperature=temp,
                required_pages=self.model.required_pages(len(ids), n),
                output=out,
                handle=handle))
        out.queue_depth_on_submit = depth
        out.request_id = handle.request_id
        out.cancel_handle = handle
        print(f"[strategy] submit id={handle.request_id} input={len(ids)} "
              f"max_new={n} depth={depth}", flush=True)
        return out

    def cancel(self, request_id: str) -> bool:
        """Cancel a pending request or request running at its next safe point."""
        with self._lock:
            handle = self._handles.get(str(request_id))
        if handle is None:
            return False
        first = handle.cancel()
        print(f"[strategy] cancel id={handle.request_id} state={handle.state} "
              f"first={first}", flush=True)
        return True

    def _start_worker(self) -> None:
        """Launch the single FIFO worker once; caller must hold ``_lock``."""
        if self._worker is not None and self._worker.is_alive():
            return
        self._worker = threading.Thread(
            target=self._serve_forever, name="strategy-worker", daemon=True)
        self._worker.start()

    def _serve_forever(self) -> None:
        while True:
            jobs = [self._scheduler.take_anchor()]
            try:
                anchor = jobs[0]
                if anchor.handle.cancel_event.is_set():
                    self._finish_cancelled(anchor)
                elif anchor.required_pages > self.model.capabilities.kv_pool_pages:
                    # pool is a fixed budget now: a single request may not fit at
                    # all. Reject explicitly instead of asserting inside ensure().
                    raise ValueError(
                        f'request needs {anchor.required_pages} kv pages, '
                        f'pool has {self.model.capabilities.kv_pool_pages}')
                else:
                    anchor.handle.state = "running"
                    self._execute(anchor, jobs)
            except Exception as exc:                    # noqa: BLE001
                error = f"{type(exc).__name__}: {exc}"
                print(f"[strategy] request-failed ids="
                      f"{[j.handle.request_id for j in jobs]} {error}",
                      flush=True)
                traceback.print_exc()
                for job in jobs:
                    job.output.put({"type": "error", "error": error})
            finally:
                with self._lock:
                    for job in jobs:
                        self._pending -= 1
                        self._handles.pop(job.handle.request_id, None)

    @staticmethod
    def _finish_cancelled(job: BatchJob) -> None:
        job.handle.state = "cancelled"
        job.output.put({"type": "end", "cancelled": True})
        print(f"[strategy] skip-cancelled id={job.handle.request_id}", flush=True)

    def _execute(self, anchor: BatchJob, jobs: list[BatchJob]) -> None:
        self.model.set_input(anchor.input_ids)
        match, loaded_hit, lookup_s, load_s, free_pages = self._load_old_kv(0, anchor)
        matches = [match]
        loaded_hits = [loaded_hit]
        lookups = [lookup_s]
        loads = [load_s]

        loaded_pages = self.model.capabilities.kv_pool_pages - free_pages
        available_pages = free_pages - max(0, anchor.required_pages - loaded_pages)
        time.sleep(BOARDING_GRACE_S)

        while len(jobs) < self.model.capabilities.max_batch_size:
            job = self._scheduler.take_next(available_pages)
            if job is None:
                break
            job.handle.state = "running"
            jobs.append(job)
            # Boarding rows need the same slot reset as the anchor: set_input
            # releases the row's slot (clearing stale pool.pos from the previous
            # request) before any cold-KV load writes into it.
            self.model.set_input(job.input_ids, row=len(jobs) - 1)
            match, loaded_hit, lookup_s, load_s, _ = self._load_old_kv(len(jobs) - 1, job)
            matches.append(match)
            loaded_hits.append(loaded_hit)
            lookups.append(lookup_s)
            loads.append(load_s)
            available_pages -= job.required_pages

        self._generate(jobs, matches, loaded_hits, lookups, loads)

    def _load_old_kv(self, row: int, job: BatchJob):
        started = time.perf_counter()
        match = self.cold_kv.begin(job.input_ids)
        looked_up = time.perf_counter()
        free_pages = self.model.capabilities.kv_pool_pages

        # V2 restores contiguous group rows directly from their pinned slabs.
        # As in V1, deliberately recompute the final matched 128-token block.
        total = max(0, match.token_count - self.cold_kv.block_size)
        if self._cold_kv_v2:
            if total:
                block_limit = total // self.cold_kv.block_size
                runs = self.cold_kv.restore_runs(match, block_limit)
                for run in runs:  # aux reconstructs the full resumable history
                    start = run.start_block * self.cold_kv.block_size
                    free_pages = self.model.load_kv_v2_group(
                        row, start, run.group("aux"), "aux").free_pages
                main_blocks = POOL1_TAIL_PAGES * self._span_chunk // self.cold_kv.block_size
                main_first = max(0, block_limit - main_blocks)
                for run in runs:
                    lo = max(run.start_block, main_first)
                    hi = run.start_block + run.block_count
                    if lo >= hi:
                        continue
                    offset = lo - run.start_block
                    src = run.group("main")[offset:offset + hi - lo]
                    free_pages = self.model.load_kv_v2_group(
                        row, lo * self.cold_kv.block_size, src, "main").free_pages
                last = runs[-1]
                free_pages = self.model.load_kv_v2_group(
                    row, total - self.cold_kv.block_size,
                    last.group("tail", last_only=True), "tail").free_pages
            return (match, total, looked_up - started,
                    time.perf_counter() - looked_up, free_pages)

        blocks: list[tuple[int, int, torch.Tensor]] = []

        def gather(start: int, end: int, source: torch.Tensor) -> None:
            blocks.append((start, end, source))

        # A portable past is sliceable only at 128-token boundaries.  Keep the
        # full lookup pinned for cache-chain/store bookkeeping, but deliberately
        # recompute its final matched block.  This avoids resuming directly from
        # a historical boundary exported after a larger continuous prefill.
        total = max(0, match.token_count - self.cold_kv.block_size)
        self.cold_kv.restore(match, gather)
        blocks = [block for block in blocks if block[1] <= total]
        if blocks:
            chunk = self._span_chunk
            staging = self._span_staging
            main_n, aux_n, tail_n = self._kv_sections
            block_numel = main_n + aux_n + tail_n
            index = 0
            for base in range(0, total, chunk):
                stop = min(base + chunk, total)
                nblocks = (stop - base) // KV_FORMAT.block_tokens
                # Match broken's conservative page rule: any page intersecting
                # the final N-page window carries main KV.
                include_main = stop > total - POOL1_TAIL_PAGES * chunk
                include_tail = stop == total
                row_n = (main_n if include_main else 0) + aux_n
                needed = nblocks * row_n + (tail_n if include_tail else 0)
                packet = staging[:needed]
                out = 0
                last_src = None
                while index < len(blocks) and blocks[index][1] <= stop:
                    start, end, source = blocks[index]
                    src = source.reshape(-1)
                    count = (end - start) // KV_FORMAT.block_tokens
                    assert src.numel() == count * block_numel
                    src = src.view(count, block_numel)
                    if include_main:
                        take = src[:, :main_n + aux_n].reshape(-1)
                    else:
                        take = src[:, main_n:main_n + aux_n].reshape(-1)
                    packet[out:out + take.numel()].copy_(take)
                    out += take.numel()
                    last_src = src[-1]
                    index += 1
                assert out == nblocks * row_n
                if include_tail:
                    assert last_src is not None
                    packet[out:out + tail_n].copy_(
                        last_src[main_n + aux_n:main_n + aux_n + tail_n])
                    out += tail_n
                assert out == needed
                free_pages = self.model.load_kv_span(
                    row, base, stop, packet,
                    pool1_tail_pages=POOL1_TAIL_PAGES,
                    include_main=include_main,
                    include_tail=include_tail).free_pages
        return (match, total, looked_up - started,
                time.perf_counter() - looked_up, free_pages)

    def _generate(self, jobs: list[BatchJob], matches: list,
                  loaded_hits: list[int], lookups: list[float],
                  loads: list[float]) -> None:
        started = time.perf_counter()
        counts = [0] * len(jobs)
        batch_at_prefill = [len(jobs)] * len(jobs)
        boarding_started: dict[int, float] = {}
        # Cold KV V2 keeps one store transaction per *sequence*: it is opened just
        # before that row's chunked prefill and closed right after the row's last
        # chunk.  Rows are prefilled one at a time, so transactions never overlap
        # and batching changes only how many run back to back -- never semantics.
        v2_plans: dict = {}
        v2_results: dict = {}
        v2_store_seconds = [0.0] * len(jobs)

        def row_prefill_begin(row: int) -> None:
            if not self._cold_kv_v2:
                return
            # Siblings still in flight must keep their matched chain alive: this
            # transaction is allowed to evict, and a sibling would otherwise commit
            # onto a dead parent.  Pin every other row's lookup while we append.
            protect = [m for i, m in enumerate(matches) if i != row]
            plan = self.cold_kv.prepare_store(matches[row], jobs[row].input_ids,
                                              protect=protect)
            try:
                self.model.begin_kv_v2_capture(plan)
            except Exception:
                plan.abort()
                raise
            v2_plans[row] = plan

        def row_prefill_end(row: int, ok: bool) -> None:
            if not self._cold_kv_v2:
                return
            plan = v2_plans.pop(row, None)
            if plan is None:
                return
            t_store = time.perf_counter()
            result = self.model.finish_kv_v2_capture(commit=ok)
            v2_store_seconds[row] = time.perf_counter() - t_store
            if ok:
                v2_results[row] = result

        def emit(row: int, token_ids: list[int]) -> None:
            counts[row] += len(token_ids)
            jobs[row].output.put({"type": "token", "token_ids": token_ids})

        def publish_prefill(row: int, model_s: float,
                            model_tokens: int, *, batch_size: int) -> None:
            finished = time.perf_counter()
            job = jobs[row]
            store_started = time.perf_counter()
            if self._cold_kv_v2:
                # Committed by row_prefill_end when this row's prefill finished.
                result = v2_results.pop(row, None)
                if result is None:
                    raise RuntimeError(
                        f"cold KV V2 store result missing for row={row}")
                store_s = v2_store_seconds[row]
            else:
                result = self.cold_kv.store(
                    matches[row], job.input_ids,
                    lambda start, end, destination:
                    self.model.export_kv(row, start, end, destination))
                store_s = time.perf_counter() - store_started
            length = len(job.input_ids)
            hit = loaded_hits[row]
            total_s = finished - job.handle.submitted_at
            job.output.metrics = QueryMetrics(
                input_tokens=length,
                cache_block_size=self.cold_kv.block_size,
                cache_hit_tokens=hit,
                cache_hit_rate=(hit / length if length else 0.0),
                prefill_tokens=length - hit,
                cache_stored_blocks=result.stored_blocks,
                cache_evicted_blocks=result.evicted_blocks,
                cache_lookup_seconds=lookups[row],
                cache_load_seconds=loads[row],
                model_prefill_seconds=model_s,
                cache_store_seconds=store_s,
                submit_to_prefill_seconds=total_s,
                strategy_seconds=total_s,
                effective_prefill_tps=(length / total_s if total_s else 0.0),
                model_prefill_tps=(model_tokens / model_s if model_s else 0.0))
            print(
                f"[strategy] prefill id={job.handle.request_id} "
                f"row={row} batch={batch_size} input={length} hit={hit} "
                f"lookup={lookups[row]:.6f}s load={loads[row]:.6f}s "
                f"model={model_s:.6f}s store={store_s:.6f}s "
                f"total={total_s:.6f}s "
                f"model_tps={job.output.metrics.model_prefill_tps:.2f} "
                f"effective_tps={job.output.metrics.effective_prefill_tps:.2f}",
                flush=True)
            job.output.put({"type": "prefill",
                            "metrics": asdict(job.output.metrics),
                            "batch_size": batch_size})

        def on_prefill() -> None:
            model_s = time.perf_counter() - started
            batch_tokens = sum(len(job.input_ids) - loaded_hits[row]
                               for row, job in enumerate(jobs))
            for row in range(len(jobs)):
                publish_prefill(row, model_s, batch_tokens,
                                batch_size=len(jobs))

        def board_request(row: int) -> Optional[BoardingRequest]:
            while True:
                job = self._scheduler.take_next(self.model.free_pages)
                if job is None:
                    return None
                if job.handle.cancel_event.is_set():
                    self._finish_cancelled(job)
                    with self._lock:
                        self._pending -= 1
                        self._handles.pop(job.handle.request_id, None)
                    continue
                job.handle.state = "running"
                jobs.append(job)
                counts.append(0)
                batch_at_prefill.append(0)
                v2_store_seconds.append(0.0)
                # Same slot reset as the initial batch rows (see _run_batch):
                # set_input releases any stale slot bound to this row before
                # the cold-KV load writes into a fresh one.
                self.model.set_input(job.input_ids, row=row)
                match, loaded_hit, lookup_s, load_s, _ = self._load_old_kv(row, job)
                matches.append(match)
                loaded_hits.append(loaded_hit)
                lookups.append(lookup_s)
                loads.append(load_s)
                boarding_started[row] = time.perf_counter()
                print(f"[strategy] board id={job.handle.request_id} row={row} "
                      f"free_pages={self.model.free_pages}", flush=True)
                return BoardingRequest(job.input_ids, job.max_new_tokens,
                                       job.handle.cancel_event,
                                       job.temperature)

        def on_boarded(row: int, active_batch_size: int) -> None:
            model_s = time.perf_counter() - boarding_started.pop(row)
            batch_at_prefill[row] = active_batch_size
            publish_prefill(row, model_s,
                            len(jobs[row].input_ids) - loaded_hits[row],
                            batch_size=active_batch_size)

        def select_active_rows(active_rows: Sequence[int],
                               done: Sequence[bool]) -> list[int]:
            """Keep only live original rows; model storage remains batch-owned."""
            return [row for row in active_rows if not done[row]]

        decode_stats: dict = {}
        try:
            self.model.generate_batch(
                [job.input_ids for job in jobs],
                [job.max_new_tokens for job in jobs],
                [job.handle.cancel_event for job in jobs], emit,
                temperatures=[job.temperature for job in jobs],
                on_prefill=on_prefill, stats=decode_stats,
                select_active_rows=select_active_rows,
                board_request=board_request, on_boarded=on_boarded,
                boarding_interval_steps=128,
                row_prefill_begin=row_prefill_begin,
                row_prefill_end=row_prefill_end)
        except Exception:
            # Any transaction still open (the row that raised) must be rolled back.
            for open_row in list(v2_plans):
                try:
                    row_prefill_end(open_row, False)
                except Exception as exc:
                    print(f"[strategy] cold-kv-v2 abort failed row={open_row}: {exc}",
                          flush=True)
            self.model.reset()
            raise

        elapsed = time.perf_counter() - started
        row_seconds = decode_stats.get("row_decode_seconds", [0.0] * len(jobs))
        for row, job in enumerate(jobs):
            cancelled = job.handle.cancel_event.is_set() and not job.handle.semantic_stop
            job.handle.state = "cancelled" if cancelled else "done"
            row_steps = decode_stats.get("row_steps")
            steps = int(row_steps[row] if row_steps is not None
                        else decode_stats.get("steps", 0))
            accepted = int(decode_stats.get("accepts", [0] * len(jobs))[row])
            decode_s = float(row_seconds[row])
            accepted_per_step = accepted / steps if steps else 0.0
            tokens_per_step = counts[row] / steps if steps else 0.0
            decode_ms_per_step = 1000.0 * decode_s / steps if steps else 0.0
            decode_tps = counts[row] / decode_s if decode_s else 0.0
            reason = ("cancelled" if cancelled else
                      "semantic_eos" if job.handle.semantic_stop else
                      "max_tokens" if counts[row] >= job.max_new_tokens else
                      "model_eos")
            job.output.decode_stats = {"decode_steps": steps,
                                       "accepted_tokens": accepted}
            job.output.put({"type": "end", "cancelled": cancelled,
                            "reason": reason,
                            "decode_steps": steps,
                            "accepted_tokens": accepted})
            print(f"[strategy] end id={job.handle.request_id} "
                  f"state={job.handle.state} reason={reason} "
                  f"batch_at_prefill={batch_at_prefill[row]} "
                  f"output={counts[row]} decode_steps={steps} "
                  f"accepted={accepted} accepted_per_step={accepted_per_step:.3f} "
                  f"tokens_per_step={tokens_per_step:.3f} "
                  f"decode_seconds={decode_s:.6f} "
                  f"decode_ms_per_step={decode_ms_per_step:.3f} "
                  f"decode_tps={decode_tps:.2f} elapsed={elapsed:.3f}s",
                  flush=True)


_runtime: Optional[Strategy] = None


def startup(devices: Optional[Sequence[int]] = None,
            cold_kv: Optional[ColdKVBackend] = None,
            prefill_chunk_tokens: Optional[int] = None) -> Strategy:
    """Start model and strategy layers with an injectable cold-KV backend."""
    global _runtime
    devices = None if devices is None else tuple(int(d) for d in devices)
    kwargs = {}
    if prefill_chunk_tokens is not None:
        kwargs["prefill_chunk_tokens"] = int(prefill_chunk_tokens)
    started = time.perf_counter()
    print("[startup] strategy.startup begin", flush=True)
    model = ModelExecution.startup(devices=devices, **kwargs)
    print(f"[startup] strategy model_ready total={time.perf_counter()-started:.3f}s", flush=True)
    # Batch graphs must be captured while every slot is still empty (capture warms up
    # on the slot KV); at request time prepare_batch then only re-binds row state.
    if int(getattr(model.capabilities, "max_batch_size", 1)) > 1:
        _t = model.warmup_batch_graphs()
        print(f"[startup] batch graphs captured in {_t:.2f}s", flush=True)
    _runtime = Strategy(model, cold_kv=cold_kv)
    print(f"[startup] strategy ready total={time.perf_counter()-started:.3f}s", flush=True)
    return _runtime


def query(input_ids: Sequence[int], max_new_tokens: int = 16,
          temperature: float = 1.0) -> Queue:
    """Submit input ids through the strategy instance created by startup."""
    if _runtime is None:
        raise RuntimeError("strategy.startup() has not been called")
    return _runtime.query(input_ids, max_new_tokens=max_new_tokens,
                          temperature=temperature)


def cancel(request_id: str) -> bool:
    """Cancel a request by public strategy request id."""
    if _runtime is None:
        return False
    return _runtime.cancel(request_id)
