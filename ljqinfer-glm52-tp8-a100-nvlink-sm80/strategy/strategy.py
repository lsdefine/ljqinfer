#!/usr/bin/env python3
"""Block-granular cold KV reuse around one streamed model execution slot.

The strategy imports only the stable model-layer API and treats the model as an
opaque execution-slot provider.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from queue import Queue
import threading
import time
import traceback
from typing import Optional, Sequence

import torch

from strategy.batch_scheduler import BatchJob, BatchScheduler
from strategy.cold_kv_cache import ColdKVBackend, KVLayout, PinnedMemoryKVCache
from model.model_api import BoardingRequest, KV_FORMAT, ModelExecution


BOARDING_GRACE_S = 0.30


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


class Strategy:
    """One execution slot plus a chained, block-granular cold KV cache."""

    def __init__(self, model: ModelExecution,
                 cold_kv: Optional[ColdKVBackend] = None):
        self.model = model
        self.cold_kv = cold_kv or PinnedMemoryKVCache(KVLayout(
            layers=KV_FORMAT.layers, width=KV_FORMAT.width,
            dtype=KV_FORMAT.dtype))
        caps = model.capabilities
        self._scheduler = BatchScheduler(
            max_batch_size=caps.max_batch_size,
            pool_pages=caps.kv_pool_pages)
        self._lock = threading.Lock()
        self._worker: Optional[threading.Thread] = None
        self._pending = 0
        self._seq = 0
        self._handles = {}

    @property
    def pending(self) -> int:
        """Requests admitted but not yet finished (includes the running one)."""
        with self._lock:
            return self._pending

    def query(self, input_ids: Sequence[int],
              max_new_tokens: int = 16) -> Queue:
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
        n = int(max_new_tokens)
        if n < 0:
            raise ValueError("max_new_tokens must be non-negative")

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
        match, lookup_s, load_s, free_pages = self._load_old_kv(0, anchor)
        matches = [match]
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
            match, lookup_s, load_s, _ = self._load_old_kv(len(jobs) - 1, job)
            matches.append(match)
            lookups.append(lookup_s)
            loads.append(load_s)
            available_pages -= job.required_pages

        self._generate(jobs, matches, lookups, loads)

    def _load_old_kv(self, row: int, job: BatchJob):
        started = time.perf_counter()
        match = self.cold_kv.begin(job.input_ids)
        looked_up = time.perf_counter()
        free_pages = self.model.capabilities.kv_pool_pages

        def load(start: int, end: int, source: torch.Tensor) -> None:
            nonlocal free_pages
            free_pages = self.model.load_kv_new(row, start, end, source).free_pages

        self.cold_kv.restore(match, load)
        return match, looked_up - started, time.perf_counter() - looked_up, free_pages

    def _generate(self, jobs: list[BatchJob], matches: list,
                  lookups: list[float], loads: list[float]) -> None:
        started = time.perf_counter()
        counts = [0] * len(jobs)
        batch_at_prefill = [len(jobs)] * len(jobs)
        boarding_started: dict[int, float] = {}

        def emit(row: int, token_ids: list[int]) -> None:
            counts[row] += len(token_ids)
            jobs[row].output.put({"type": "token", "token_ids": token_ids})

        def publish_prefill(row: int, model_s: float,
                            model_tokens: int, *, batch_size: int) -> None:
            finished = time.perf_counter()
            job = jobs[row]
            store_started = time.perf_counter()
            result = self.cold_kv.store(
                matches[row], job.input_ids,
                lambda start, end, destination:
                self.model.export_kv(row, start, end, destination))
            store_s = time.perf_counter() - store_started
            length = len(job.input_ids)
            hit = matches[row].token_count
            total_s = finished - job.handle.submitted_at
            job.output.metrics = QueryMetrics(
                input_tokens=length,
                cache_block_size=self.cold_kv.block_size,
                cache_hit_tokens=hit,
                cache_hit_rate=(hit / length if length else 0.0),
                prefill_tokens=length - hit + int(hit > 0),
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
            batch_tokens = sum(len(job.input_ids) for job in jobs)
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
                match, lookup_s, load_s, _ = self._load_old_kv(row, job)
                matches.append(match)
                lookups.append(lookup_s)
                loads.append(load_s)
                boarding_started[row] = time.perf_counter()
                print(f"[strategy] board id={job.handle.request_id} row={row} "
                      f"free_pages={self.model.free_pages}", flush=True)
                return BoardingRequest(job.input_ids, job.max_new_tokens,
                                       job.handle.cancel_event)

        def on_boarded(row: int, active_batch_size: int) -> None:
            model_s = time.perf_counter() - boarding_started.pop(row)
            batch_at_prefill[row] = active_batch_size
            publish_prefill(row, model_s, len(jobs[row].input_ids),
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
                on_prefill=on_prefill, stats=decode_stats,
                select_active_rows=select_active_rows,
                board_request=board_request, on_boarded=on_boarded,
                boarding_interval_steps=128)
        except Exception:
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
    _runtime = Strategy(model, cold_kv=cold_kv)
    print(f"[startup] strategy ready total={time.perf_counter()-started:.3f}s", flush=True)
    return _runtime


def query(input_ids: Sequence[int], max_new_tokens: int = 16) -> Queue:
    """Submit input ids through the strategy instance created by startup."""
    if _runtime is None:
        raise RuntimeError("strategy.startup() has not been called")
    return _runtime.query(input_ids, max_new_tokens=max_new_tokens)


def cancel(request_id: str) -> bool:
    """Cancel a request by public strategy request id."""
    if _runtime is None:
        return False
    return _runtime.cancel(request_id)
