"""Persistent serialized strategy backed exclusively by Q8+DFlash2 decode."""
from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
from queue import Queue
from threading import Condition, Event, Lock, Thread
from types import SimpleNamespace
import os
import time
import uuid
from typing import Callable, Deque, Optional, Sequence

from model.model_api import BoardingRequest, ModelExecution
from strategy.cold_kv_cache import PrefixColdCache
from strategy.control_plane import Rank0Coordinator, FollowerControlServer


BOARDING_GRACE_S = 0.30


@dataclass(frozen=True)
class QueryMetrics:
    input_tokens: int
    output_tokens: int
    backend: str
    verify_width: int
    verify_capacity: int
    cache_block_size: int
    cache_hit_tokens: int
    cache_hit_rate: float
    prefill_tokens: int
    cache_stored_blocks: int
    cache_entries: int
    dflash_loaded_pages: int
    dflash_resident_pages: int
    cache_lookup_seconds: float
    cache_load_seconds: float
    model_prefill_seconds: float
    cache_store_seconds: float
    decode_seconds: float
    draft_seconds: float
    verify_seconds: float
    engine_timing_source: str
    engine_profiled_rounds: int
    engine_phase_ms: list[dict[str, float]]
    draft_npu_seconds: float
    verify_npu_seconds: float
    commit_npu_seconds: float
    append_npu_seconds: float
    decode_generated_tokens: int
    decode_tps: float
    speculative_rounds: int
    proposed_draft_tokens: int
    accepted_draft_tokens: int
    acceptance_rate: float
    mean_accepted_per_round: float
    total_seconds: float


class _CancelHandle:
    """Soft cancel handle. TP4 lockstep cannot hard-stop mid-round safely."""

    def __init__(self, request_id: str):
        self.request_id = request_id
        self._state = "pending"
        self._lock = Lock()
        self.cancel_event = Event()
        self.reason: Optional[str] = None

    @property
    def state(self) -> str:
        with self._lock:
            return self._state

    def cancel(self, reason: str = "cancelled") -> bool:
        with self._lock:
            if self._state == "done":
                return False
            if self._state == "cancelled":
                return True
            self.reason = str(reason)
            self._state = "cancelled"
            self.cancel_event.set()
        print(f"[strategy] cancel id={self.request_id} reason={self.reason} "
              f"soft=True", flush=True)
        return True

    def start(self) -> bool:
        with self._lock:
            if self._state in ("done", "cancelled"):
                return False
            self._state = "running"
            return True

    def finish(self) -> tuple[bool, str]:
        with self._lock:
            cancelled = self.cancel_event.is_set()
            self._state = "cancelled" if cancelled else "done"
            reason = self.reason if cancelled else "completed"
            return cancelled, reason or "cancelled"

    def fail(self) -> None:
        with self._lock:
            if self._state != "cancelled":
                self._state = "done"

    def stop_at_semantic_eos(self) -> None:
        self.cancel("semantic_eos")


@dataclass
class _Job:
    ids: tuple[int, ...]
    n_new: int
    out: Queue
    handle: _CancelHandle


class Strategy:
    """Own one Q8+DFlash2 slot with block-granular radix cold-KV reuse."""

    def __init__(self, model: Optional[ModelExecution] = None):
        self.model = model or ModelExecution.startup()
        block = int(self.model.engine.engine_config.cold_checkpoint_interval)
        template = getattr(self.model, "cold_record_template", None)
        self.cold_cache = PrefixColdCache(
            block_size=block,
            record_template=template() if template is not None else None,
            pin_memory=template is not None)
        # Serialize model execution itself. Admission uses a separate FIFO.
        self._lock = Lock()
        self._ready = Condition(Lock())
        self._jobs: Deque[_Job] = deque()
        # Engine-global request registry; every access is protected by _ready.
        self._handles: dict[str, _CancelHandle] = {}
        self._pending = 0
        self._closing = False
        self._closed = False
        self._close_lock = Lock()
        self._worker: Optional[Thread] = None
        control_dir = os.environ.get("LJQ_CONTROL_DIR")
        rank = int(self.model.rt.rank)
        self._coordinator = (Rank0Coordinator(control_dir, self.model.rt.world)
                             if control_dir and rank == 0 else None)

    def close(self) -> None:
        """Stop admission, cancel at safe points, then collectively release."""
        with self._close_lock:
            if self._closed:
                return
            with self._ready:
                self._closing = True
                for handle in self._handles.values():
                    handle.cancel("shutdown")
                self._ready.notify_all()
                worker = self._worker
            if worker is not None:
                worker.join(timeout=60.0)
                if worker.is_alive():
                    raise TimeoutError("strategy worker did not drain; model kept alive")
            with self._lock:
                if self._coordinator is not None:
                    self._coordinator.shutdown(self.model.close)
                else:
                    self.model.close()
                self._closed = True

    def validate_request(self, input_ids: Sequence[int],
                         max_new_tokens: int = 64
                         ) -> tuple[tuple[int, ...], int]:
        """Validate synchronously before a streaming response sends headers."""
        ids = tuple(int(x) for x in input_ids)
        n_new = int(max_new_tokens)
        if not ids:
            raise ValueError("input_ids must not be empty")
        if n_new < 0:
            raise ValueError("max_new_tokens must be non-negative")
        if len(ids) + n_new > self.model.context_capacity:
            raise ValueError(
                f"engine context requires prompt+max_new <= "
                f"{self.model.context_capacity}, got {len(ids)}+{n_new}")
        return ids, n_new

    def query(self, input_ids: Sequence[int], max_new_tokens: int = 64, *,
              request_id: Optional[str] = None) -> Queue:
        """Ref-compatible event queue: prefill -> token* -> end|error.

        Admission is a strict FIFO with exactly one persistent worker. This
        matches S:/ljqinfer_tp8: query() only enqueues; generation never races
        on the TP control plane.
        """
        ids, n_new = self.validate_request(input_ids, max_new_tokens)

        out: Queue = Queue()
        rid = str(request_id) if request_id is not None else (
            "req_" + uuid.uuid4().hex[:16])
        if not rid:
            raise ValueError("request_id must not be empty")
        handle = _CancelHandle(rid)
        out.request_id = rid
        out.metrics = None
        out.cancel_handle = handle
        job = _Job(ids=ids, n_new=n_new, out=out, handle=handle)
        with self._ready:
            if self._closing:
                raise RuntimeError("strategy is shutting down")
            if rid in self._handles:
                raise ValueError(f"request_id is already active: {rid}")
            self._handles[rid] = handle
            self._jobs.append(job)
            self._pending += 1
            out.queue_depth_on_submit = self._pending
            print(
                f"[strategy] submit id={rid} input={len(ids)} max_new={n_new} "
                f"depth={out.queue_depth_on_submit}",
                flush=True)
            self._start_worker()
            self._ready.notify()
        return out

    def cancel_request(self, request_id: str,
                       reason: str = "cancelled") -> bool:
        """Mark a queued/running request for removal at the next safe point."""
        with self._ready:
            handle = self._handles.get(str(request_id))
            if handle is None:
                return False
            cancelled = handle.cancel(reason)
            self._ready.notify_all()
            return cancelled

    def _forget_request(self, job: _Job) -> None:
        with self._ready:
            current = self._handles.get(job.handle.request_id)
            if current is job.handle:
                del self._handles[job.handle.request_id]

    def generate(self, input_ids: Sequence[int], max_new_tokens: int = 64) -> dict:
        """Blocking helper used by followers and non-stream callers."""
        ids = tuple(int(x) for x in input_ids)
        n_new = int(max_new_tokens)
        if not ids:
            raise ValueError("input_ids must not be empty")
        if n_new < 0:
            raise ValueError("max_new_tokens must be non-negative")
        if len(ids) + n_new > self.model.context_capacity:
            raise ValueError(
                f"engine context requires prompt+max_new <= "
                f"{self.model.context_capacity}, got {len(ids)}+{n_new}")
        with self._lock:
            if self._closing:
                raise RuntimeError("strategy is shutting down")
            activate = getattr(self.model.rt, "activate", None)
            if activate is not None:
                activate()
            if self._coordinator is not None:
                return self._coordinator.run(
                    ids, n_new, lambda: self._generate_coordinated(ids, n_new))
            return self._generate_locked(ids, n_new)

    def _start_worker(self) -> None:
        """Launch the single FIFO worker once; caller must hold ``_ready``."""
        if self._worker is not None and self._worker.is_alive():
            return
        self._worker = Thread(
            target=self._serve_forever, name="strategy-fifo", daemon=True)
        self._worker.start()

    def _serve_forever(self) -> None:
        while True:
            with self._ready:
                while not self._jobs and not self._closing:
                    self._ready.wait()
                if not self._jobs:
                    return
                job = self._jobs.popleft()
            if not job.handle.start():
                self._finish_cancelled(job)
                self._forget_request(job)
                with self._ready:
                    self._pending = max(0, self._pending - 1)
                continue
            try:
                # Streaming query path owns one dynamic epoch. Later FIFO jobs
                # may board into the same epoch; static generate() stays B1.
                self._run_epoch(job)
            except Exception as exc:
                self._fail_job(job, exc)
                self._forget_request(job)
                with self._ready:
                    self._pending = max(0, self._pending - 1)

    @staticmethod
    def _finish_cancelled(job: _Job) -> None:
        cancelled, reason = job.handle.finish()
        if not cancelled:
            raise RuntimeError("non-cancelled job reached cancelled finish path")
        job.out.put({
            "type": "end",
            "cancelled": True,
            "reason": reason,
        })
        print(f"[strategy] skip-cancelled id={job.handle.request_id}", flush=True)

    def _decrement_pending(self) -> None:
        with self._ready:
            self._pending = max(0, self._pending - 1)

    def _publish_dynamic_prefill(self, job: _Job, *, hit: int,
                                 lookup_s: float = 0.0,
                                 load_s: float = 0.0,
                                 store_s: float = 0.0,
                                 stored: int = 0,
                                 active_batch_size: int = 1) -> None:
        metrics = {
            "input_tokens": len(job.ids),
            "cache_hit_tokens": int(hit),
            "cache_hit_rate": (hit / len(job.ids) if job.ids else 0.0),
            "prefill_tokens": max(0, len(job.ids) - int(hit)),
            "cache_stored_blocks": int(stored),
            "cache_entries": self.cold_cache.entry_count,
            "cache_block_size": self.cold_cache.block_size,
            "cache_lookup_seconds": float(lookup_s),
            "cache_load_seconds": float(load_s),
            "cache_store_seconds": float(store_s),
            "active_batch_size": int(active_batch_size),
        }
        job.out.metrics = SimpleNamespace(**metrics)
        job.out.put({"type": "prefill", "metrics": metrics})

    @staticmethod
    def _fail_job(job: _Job, exc: Exception) -> None:
        job.out.put({"type": "error",
                     "error": f"{type(exc).__name__}: {exc}"})
        job.handle.fail()
        print(f"[strategy] error id={job.handle.request_id} "
              f"err={type(exc).__name__}: {exc}", flush=True)

    def _run_epoch(self, anchor: _Job) -> None:
        started = time.perf_counter()
        jobs = [anchor]
        lookup_started = time.perf_counter()
        match = self.cold_cache.begin(anchor.ids)
        lookup_s = time.perf_counter() - lookup_started
        records = []
        load_started = time.perf_counter()
        if match.token_count:
            def collect(record, final):
                records.append(record)
                return int(record["end"])
            self.cold_cache.restore(match, collect)
        load_s = time.perf_counter() - load_started

        def execute_local() -> dict:
            # Match the follower GO handler before entering any model collective.
            self.model.rt.barrier()
            state = self.model.prefill_batch(
                [anchor.ids], sequence_ids=[0],
                max_lengths=[len(anchor.ids) + anchor.n_new],
                restored_records=[records])
            store_started = time.perf_counter()
            stored = self.cold_cache.store_owned_batch(
                match, anchor.ids,
                lambda start, end:
                state.export_prefix_records(0, start, end))
            store_s = time.perf_counter() - store_started
            # Ref admission contract: once the anchor owns/restores its cache
            # pages, give concurrent arrivals 300 ms to join the first B1-4
            # epoch. decode_dflash_batch_dynamic drains the ready FIFO before
            # its first MTP round, so these requests start as one batch.
            time.sleep(BOARDING_GRACE_S)
            self._publish_dynamic_prefill(
                anchor, hit=match.token_count, lookup_s=lookup_s,
                load_s=load_s, store_s=store_s, stored=stored,
                active_batch_size=1)

            def emit(row: int, token_ids: Sequence[int]) -> None:
                ids_list = [int(x) for x in token_ids]
                if ids_list:
                    jobs[row].out.put({"type": "token", "token_ids": ids_list})

            def request_board(row: int) -> Optional[BoardingRequest]:
                while True:
                    with self._ready:
                        if not self._jobs:
                            return None
                        candidate = self._jobs[0]
                        if candidate.handle.state != "cancelled":
                            return BoardingRequest(
                                candidate.ids, candidate.n_new,
                                candidate.handle.cancel_event)
                        self._jobs.popleft()
                    self._finish_cancelled(candidate)
                    self._forget_request(candidate)
                    self._decrement_pending()

            def boarded(row: int, active_batch_size: int) -> None:
                with self._ready:
                    if not self._jobs:
                        raise RuntimeError("boarded request disappeared from FIFO")
                    job = self._jobs.popleft()
                if row != len(jobs):
                    raise RuntimeError(
                        f"boarded row mismatch: model={row} strategy={len(jobs)}")
                job.handle.start()
                jobs.append(job)
                # Boarded rows intentionally miss cold KV in this first minimal
                # wiring; the model has completed their full prefill here.
                self._publish_dynamic_prefill(
                    job, hit=0, active_batch_size=active_batch_size)

            return self.model.decode_dflash_batch_dynamic(
                state, [anchor.n_new],
                cancel_events=[anchor.handle.cancel_event],
                on_tokens=emit, board_request=request_board,
                on_boarded=boarded, boarding_interval_steps=128)

        try:
            with self._lock:
                activate = getattr(self.model.rt, "activate", None)
                if activate is not None:
                    activate()
                if self._coordinator is not None:
                    result = self._coordinator.run(
                        anchor.ids, anchor.n_new, execute_local,
                        op="generate_dynamic")
                else:
                    result = execute_local()
            rows = result["rows"]
            if len(rows) != len(jobs):
                raise RuntimeError(
                    f"dynamic result/job mismatch: {len(rows)} != {len(jobs)}")
            for job, row in zip(jobs, rows):
                cancelled, reason = job.handle.finish()
                rounds = int(row.get("rounds", 0))
                accepted = int(row.get("accepted_draft_tokens", 0))
                job.out.decode_stats = {
                    "decode_steps": rounds,
                    "accepted_tokens": accepted,
                }
                steps = rounds
                decode_s = float(row.get("decode_seconds", 0.0))
                output_count = len(row.get("token_ids", ()))
                # The first token comes from prefill, not from any decode
                # step, and its cost sits outside decode_seconds. Exclude
                # it from decode-rate numerators so these match the
                # engine-level decode_tps in model_api.
                decode_tokens = max(0, output_count - 1)
                accepted_per_step = accepted / steps if steps else 0.0
                tokens_per_step = decode_tokens / steps if steps else 0.0
                decode_ms_per_step = 1000.0 * decode_s / steps if steps else 0.0
                decode_tps = decode_tokens / decode_s if decode_s else 0.0
                elapsed = time.perf_counter() - started
                print(f"[strategy] end id={job.handle.request_id} "
                      f"state={job.handle.state} reason={reason} "
                      f"batch_at_prefill={job.out.metrics.active_batch_size} "
                      f"output={output_count} decode_steps={steps} "
                      f"accepted={accepted} accepted_per_step={accepted_per_step:.3f} "
                      f"tokens_per_step={tokens_per_step:.3f} "
                      f"decode_seconds={decode_s:.6f} "
                      f"decode_ms_per_step={decode_ms_per_step:.3f} "
                      f"decode_tps={decode_tps:.2f} elapsed={elapsed:.3f}s",
                      flush=True)
                self._forget_request(job)
                self._decrement_pending()
                job.out.put({
                    "type": "end", "cancelled": cancelled,
                    "reason": reason or "cancelled",
                    "decode_steps": rounds,
                    "accepted_tokens": accepted,
                })
        except Exception as exc:
            reset = getattr(self.model, "reset", None)
            if reset is not None:
                reset()
            for job in jobs:
                self._fail_job(job, exc)
                self._forget_request(job)
                self._decrement_pending()

    def _run_epoch_follower(self, ids: tuple[int, ...], n_new: int) -> dict:
        match = self.cold_cache.begin(ids)
        records = []
        if match.token_count:
            def collect(record, final):
                records.append(record)
                return int(record["end"])
            self.cold_cache.restore(match, collect)
        state = self.model.prefill_batch(
            [ids], sequence_ids=[0], max_lengths=[len(ids) + n_new],
            restored_records=[records])
        self.cold_cache.store_owned_batch(
            match, ids,
            lambda start, end:
            state.export_prefix_records(0, start, end))
        return self.model.decode_dflash_batch_dynamic(
            state, [n_new], boarding_interval_steps=128)

    def _run_generate(self, ids: tuple[int, ...], n_new: int, out: Queue) -> dict:
        with self._lock:
            activate = getattr(self.model.rt, "activate", None)
            if activate is not None:
                activate()
            if self._coordinator is not None:
                return self._coordinator.run(
                    ids, n_new,
                    lambda: self._generate_coordinated(ids, n_new, out=out))
            return self._generate_locked(ids, n_new, out=out)

    def _generate_coordinated(self, ids: tuple[int, ...], n_new: int,
                              out: Optional[Queue] = None) -> dict:
        self.model.rt.barrier()
        return self._generate_locked(ids, n_new, out=out)

    def _generate_locked(self, ids: tuple[int, ...], n_new: int,
                         out: Optional[Queue] = None) -> dict:
        total_t0 = time.perf_counter()
        lookup_t0 = time.perf_counter()
        match = self.cold_cache.begin(ids)
        lookup_s = time.perf_counter() - lookup_t0
        records = []
        load_t0 = time.perf_counter()
        if match.token_count:
            def collect(record, final):
                records.append(record)
                return int(record["end"])
            self.cold_cache.restore(match, collect)
        load_s = time.perf_counter() - load_t0
        stored = 0
        store_s = 0.0
        prefill_emitted = False

        def store_ready(exporter) -> None:
            nonlocal stored, store_s
            started = time.perf_counter()
            stored = self.cold_cache.store_owned_batch(match, ids, exporter)
            store_s = time.perf_counter() - started

        def emit_prefill():
            nonlocal prefill_emitted
            if out is None or prefill_emitted:
                return
            hit = int(match.token_count)
            metrics = {
                "input_tokens": len(ids),
                "cache_hit_tokens": hit,
                "cache_hit_rate": (hit / len(ids) if ids else 0.0),
                "prefill_tokens": max(0, len(ids) - hit),
                "cache_stored_blocks": stored,
                "cache_entries": self.cold_cache.entry_count,
                "cache_block_size": self.cold_cache.block_size,
                "cache_lookup_seconds": lookup_s,
                "cache_load_seconds": load_s,
                "cache_store_seconds": store_s,
            }
            out.metrics = SimpleNamespace(**metrics)
            out.put({"type": "prefill", "metrics": metrics})
            print(
                f"[strategy] prefill id={getattr(out, 'request_id', '-')} "
                f"input={len(ids)} hit={hit} stored={stored} "
                f"lookup={lookup_s:.6f}s load={load_s:.6f}s store={store_s:.6f}s",
                flush=True)
            prefill_emitted = True

        def on_tokens(token_ids: Sequence[int]) -> None:
            emit_prefill()
            if out is None:
                return
            ids_list = [int(x) for x in token_ids]
            if not ids_list:
                return
            out.put({"type": "token", "token_ids": ids_list})

        result = self.model.generate_dflash(
            ids, n_new, sequence_id=0, restored_records=records,
            on_prefill_ready=store_ready, on_tokens=on_tokens)
        emit_prefill()
        hit = int(result["cache_hit_tokens"])
        metrics = QueryMetrics(
            input_tokens=len(ids), output_tokens=len(result["token_ids"]),
            backend=result["backend"], verify_width=result["verify_width"],
            verify_capacity=result["verify_capacity"],
            cache_block_size=self.cold_cache.block_size,
            cache_hit_tokens=hit,
            cache_hit_rate=(hit / len(ids) if ids else 0.0),
            prefill_tokens=int(result["prefill_tokens"]),
            cache_stored_blocks=stored, cache_entries=self.cold_cache.entry_count,
            dflash_loaded_pages=int(result["dflash_loaded_pages"]),
            dflash_resident_pages=int(result["dflash_resident_pages"]),
            cache_lookup_seconds=lookup_s,
            cache_load_seconds=load_s + result["cache_load_seconds"],
            model_prefill_seconds=result["prefill_seconds"],
            cache_store_seconds=store_s, decode_seconds=result["decode_seconds"],
            draft_seconds=result["draft_seconds"],
            verify_seconds=result["verify_seconds"],
            engine_timing_source=result["engine_timing_source"],
            engine_profiled_rounds=result["engine_profiled_rounds"],
            engine_phase_ms=result["engine_phase_ms"],
            draft_npu_seconds=result["draft_npu_seconds"],
            verify_npu_seconds=result["verify_npu_seconds"],
            commit_npu_seconds=result["commit_npu_seconds"],
            append_npu_seconds=result["append_npu_seconds"],
            decode_generated_tokens=result["decode_generated_tokens"],
            decode_tps=result["decode_tps"],
            speculative_rounds=result["rounds"],
            proposed_draft_tokens=result["proposed_draft_tokens"],
            accepted_draft_tokens=result["accepted_draft_tokens"],
            acceptance_rate=result["acceptance_rate"],
            mean_accepted_per_round=result["mean_accepted_per_round"],
            total_seconds=time.perf_counter() - total_t0)
        metrics_dict = asdict(metrics)
        if out is not None:
            out.metrics = SimpleNamespace(**metrics_dict)
        return {"token_ids": result["token_ids"], "metrics": metrics_dict}


def follower_main() -> None:
    strategy = Strategy()
    control_dir = os.environ.get("LJQ_CONTROL_DIR")
    if not control_dir:
        raise RuntimeError("LJQ_CONTROL_DIR is required for follower ranks")
    server = FollowerControlServer(control_dir, strategy.model.rt.rank)

    def execute(ids, n_new, *, op="generate"):
        if op == "generate_dynamic":
            with strategy._lock:
                activate = getattr(strategy.model.rt, "activate", None)
                if activate is not None:
                    activate()
                strategy.model.rt.barrier()
                return strategy._run_epoch_follower(ids, n_new)
        strategy.model.rt.barrier()
        return strategy.generate(ids, n_new)

    server.serve(execute, shutdown=strategy.close)
