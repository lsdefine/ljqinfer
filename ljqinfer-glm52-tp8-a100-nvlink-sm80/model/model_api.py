#!/usr/bin/env python3
"""Stable model-layer API for one configurable GPU execution slot.

This module is the boundary above the model orchestration implementation.  A
caller can load input ids, run prefill, copy the public 79 x 576 MLA KV format,
and start streamed decode.  Details such as speculative decoding are private to
this module.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import threading
import time
from typing import Callable, Iterable, Optional, Sequence

import torch

from model import model as _impl
from model.config import EXECUTION_LEN, Q_MAX


class CancellationError(RuntimeError):
    """Cooperative cancellation observed at a model safe point."""
PUBLIC_KV_LAYERS = 79
PUBLIC_KV_WIDTH = 576


@contextmanager
def _stable_full_cache_write(cache: _impl.KVCache, start: int, count: int):
    """Keep graph-bound KV tensors stable during a full-cache prefill."""
    max_len = int(cache.max_len)
    protect = int(start) == 0 and int(count) == max_len
    if protect:
        cache.max_len = max_len + 1
    try:
        yield
    finally:
        cache.max_len = max_len


@dataclass(frozen=True)
class KVFormat:
    """Public cold-cache record format: [tokens, layers, width]."""

    layers: int = PUBLIC_KV_LAYERS
    width: int = PUBLIC_KV_WIDTH
    dtype: torch.dtype = torch.float16
    layout: str = "token_layer_width"


@dataclass(frozen=True)
class ModelCapabilities:
    """Scheduling geometry owned and reported by the model layer."""

    max_batch_size: int
    kv_page_size: int
    kv_pool_pages: int


@dataclass(frozen=True)
class KVLoadResult:
    free_pages: int


@dataclass(frozen=True)
class BoardingRequest:
    """One FIFO request restored by the strategy for safe-point boarding."""

    input_ids: tuple[int, ...]
    max_new_tokens: int
    cancel_event: threading.Event


KV_FORMAT = KVFormat()


class ModelExecution:
    """One model-owned execution slot.

    Public lifecycle for one request::

        set_input(input_ids) -> generate_batch(...)

    ``generate_batch`` is the only generation entrypoint. The production B1/B2 slot
    uses paged prefill plus the resident fixed-width verify/streaming state machine.
    """

    def __init__(self, engine: _impl.Engine):
        if _impl.N_LAYER != 78 or _impl.CACHE_DIM != PUBLIC_KV_WIDTH:
            raise RuntimeError(
                "public KV contract requires 78 base layers + 1 private slot, width 576")
        self.__engine = engine
        self.__capacity = int(engine.kv.max_len)
        self.__input_ids = torch.empty(
            self.__capacity, dtype=torch.long, device="cpu", pin_memory=True)
        self.__input_length = 0
        self.__pool_next_page = 0
        self.__pool_row_pages: dict[int, list[int]] = {}
        self.__pool_row_tokens: dict[int, int] = {}
        self.__prefill_row_pages: dict[int, tuple[int, ...]] = {}
        self.__prefill_row_lengths: dict[int, int] = {}
        self.__phase = "idle"
        self.__lock = threading.RLock()


    @classmethod
    def startup(cls, devices: Optional[Sequence[int]] = None,
                prefill_chunk_tokens: int = _impl.DEFAULT_PREFILL_CHUNK_TOKENS
                ) -> "ModelExecution":
        """Load parameters and allocate the replicated execution slot."""
        prefill_chunk_tokens = int(prefill_chunk_tokens)
        if prefill_chunk_tokens <= 0:
            raise ValueError("prefill_chunk_tokens must be positive")
        devices = None if devices is None else tuple(int(d) for d in devices)
        if devices is not None:
            if len(devices) != _impl.TP:
                raise ValueError(f"this model requires exactly {_impl.TP} devices")
            if len(set(devices)) != len(devices):
                raise ValueError("devices must be unique")
        started = time.perf_counter()
        print(f"[startup] ModelExecution.startup begin execution_len={EXECUTION_LEN}", flush=True)
        engine = _impl.Engine.load(
            devices=None if devices is None else list(devices),
            prefill_chunk_tokens=prefill_chunk_tokens)
        model = cls(engine)
        print(f"[startup] ModelExecution.startup ready total={time.perf_counter()-started:.3f}s", flush=True)
        return model

    @property
    def capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(
            max_batch_size=4,
            kv_page_size=int(_impl.KV_PAGE_SIZE),
            kv_pool_pages=self.__engine.kv.logical_pool_pages)

    def required_pages(self, input_length: int, max_new_tokens: int) -> int:
        """Whole pages reserved for one row for the entire decode epoch."""
        q = int(Q_MAX)
        capacity = int(input_length) + q * int(max_new_tokens) + q
        size = int(_impl.KV_PAGE_SIZE)
        return (capacity + size - 1) // size

    def reserve_row_capacity(self, row: int, page_count: int) -> KVLoadResult:
        """Append never-reused physical pages until ``row`` owns ``page_count``.

        This is deliberately a high-water allocator, not a dynamic allocator:
        pages are never removed or reused until the whole generation epoch ends.
        Cold-KV restore must finish before this method reserves the unwritten tail.
        """
        row, page_count = int(row), int(page_count)
        with self.__lock:
            if self.__phase not in ("input_ready", "decoding"):
                raise RuntimeError(
                    f"row reservation requires an active epoch, got {self.__phase}")
            if row < 0 or page_count < 0:
                raise ValueError("row and page_count must be non-negative")
            pages = self.__pool_row_pages.setdefault(row, [])
            if len(pages) > page_count:
                raise ValueError("row already owns more pages than requested")
            pool_pages = int(self.__engine.kv.logical_pool_pages)
            missing = page_count - len(pages)
            if self.__pool_next_page + missing > pool_pages:
                raise ValueError(
                    f"row reservation needs {missing} pages, only "
                    f"{pool_pages - self.__pool_next_page} remain")
            pages.extend(range(self.__pool_next_page,
                               self.__pool_next_page + missing))
            self.__pool_next_page += missing
            return KVLoadResult(pool_pages - self.__pool_next_page)

    @property
    def free_pages(self) -> int:
        """Unassigned high-water tail pages in the current generation epoch."""
        with self.__lock:
            return (int(self.__engine.kv.logical_pool_pages) -
                    self.__pool_next_page)

    def set_input(self, input_ids: Iterable[int]) -> int:
        """Copy a query's input ids into the model-owned input area."""
        ids = torch.as_tensor(input_ids, dtype=torch.long).reshape(-1)
        if ids.device.type != "cpu":
            ids = ids.cpu()
        n = int(ids.numel())
        if n <= 0:
            raise ValueError("input_ids must not be empty")
        if n > self.__capacity:
            raise ValueError(
                f"input length {n} exceeds execution capacity {self.__capacity}")

        with self.__lock:
            if self.__phase != "idle":
                raise RuntimeError(f"execution slot is busy ({self.__phase})")
            self.__input_ids[:n].copy_(ids)
            self.__input_length = n
            # A zero-hit request never calls load_kv_new(row=0,start=0), so the
            # request boundary itself must discard the previous static layout.
            self._clear_generation_pages_locked()
            self.__phase = "input_ready"
        return n

    def load_kv_new(self, row: int, start: int, end: int,
                    source: torch.Tensor) -> KVLoadResult:
        with self.__lock:
            if row < 0:
                raise ValueError("row must be non-negative")
            if start < 0 or end <= start:
                raise ValueError("invalid KV range")

            count = end - start
            page_size = int(_impl.KV_PAGE_SIZE)
            if start % page_size != 0 or count > page_size:
                raise ValueError("one cold KV block must fit one aligned KV page")
            expected = (count, PUBLIC_KV_LAYERS, PUBLIC_KV_WIDTH)
            if tuple(source.shape) != expected:
                raise ValueError(
                    f"source shape {tuple(source.shape)} does not match {expected}")
            if source.dtype != KV_FORMAT.dtype:
                raise ValueError(
                    f"source dtype {source.dtype} does not match {KV_FORMAT.dtype}")

            row_pages = self.__pool_row_pages.get(row, [])
            row_tokens = self.__pool_row_tokens.get(row, 0)
            if start != row_tokens or len(row_pages) != start // page_size:
                raise ValueError("cold KV blocks for a row must be loaded in order")

            engine = self.__engine
            pool_pages = engine.kv.logical_pool_pages
            page_index = self.__pool_next_page
            if page_index >= pool_pages:
                raise ValueError("cold KV block does not fit in the static KV pool")

            physical_start = page_index * page_size
            physical_end = physical_start + count
            has_private_layer = (
                engine.mtp_kv is not None and
                getattr(engine.w, "mtp", None) is not None)
            leader_device = engine.rt.devices[0]
            leader_stream = engine.rt.streams[0]
            with torch.cuda.device(leader_device), torch.cuda.stream(leader_stream):
                for layer in range(_impl.N_LAYER):
                    engine.kv.pool_tokens(layer, 0)[physical_start:physical_end].copy_(
                        source[:, layer, :], non_blocking=True)

                if has_private_layer:
                    if start == 0:
                        if count > 1:
                            engine.mtp_kv.pool_tokens(0, 0)[
                                physical_start:physical_start + count - 1].copy_(
                                    source[1:, _impl.N_LAYER, :], non_blocking=True)
                    else:
                        previous_last = (row_pages[-1] + 1) * page_size - 1
                        engine.mtp_kv.pool_tokens(0, 0)[previous_last].copy_(
                            source[0, _impl.N_LAYER, :], non_blocking=True)
                        if count > 1:
                            engine.mtp_kv.pool_tokens(0, 0)[
                                physical_start:physical_start + count - 1].copy_(
                                    source[1:, _impl.N_LAYER, :], non_blocking=True)
                ready = torch.cuda.Event()
                ready.record(leader_stream)

            for rank, device in enumerate(engine.rt.devices[1:], start=1):
                stream = engine.rt.streams[rank]
                with torch.cuda.device(device), torch.cuda.stream(stream):
                    stream.wait_event(ready)
                    for layer in range(_impl.N_LAYER):
                        engine.kv.pool_tokens(layer, rank)[physical_start:physical_end].copy_(
                            engine.kv.pool_tokens(layer, 0)[physical_start:physical_end],
                            non_blocking=True)
                    if has_private_layer:
                        if start > 0:
                            previous_last = (row_pages[-1] + 1) * page_size - 1
                            engine.mtp_kv.pool_tokens(0, rank)[previous_last].copy_(
                                engine.mtp_kv.pool_tokens(0, 0)[previous_last],
                                non_blocking=True)
                        if count > 1:
                            mtp_end = physical_start + count - 1
                            engine.mtp_kv.pool_tokens(0, rank)[physical_start:mtp_end].copy_(
                                engine.mtp_kv.pool_tokens(0, 0)[physical_start:mtp_end],
                                non_blocking=True)

            for stream in engine.rt.streams:
                stream.synchronize()

            self.__pool_row_pages.setdefault(row, []).append(page_index)
            self.__pool_row_tokens[row] = end
            self.__pool_next_page += 1
            return KVLoadResult(pool_pages - self.__pool_next_page)

    def export_kv(self, row: int, start: int, end: int,
                  destination: torch.Tensor) -> None:
        """Export one logical row from the paged GPU KV pool."""
        row, start, end = int(row), int(start), int(end)
        with self.__lock:
            if self.__phase != "decoding":
                raise RuntimeError(
                    f"export_kv requires active generation, got {self.__phase}")
            length = self.__prefill_row_lengths.get(row)
            pages = self.__prefill_row_pages.get(row)
            if length is None or pages is None:
                raise ValueError(f"no completed prefill state for row {row}")
            if start < 0 or end < start or end > length:
                raise ValueError(
                    f"KV range [{start}, {end}) outside row {row} length {length}")
            expected = (end - start, PUBLIC_KV_LAYERS, PUBLIC_KV_WIDTH)
            if tuple(destination.shape) != expected:
                raise ValueError(
                    f"destination shape {tuple(destination.shape)} != {expected}")
            if destination.dtype != KV_FORMAT.dtype:
                raise ValueError(
                    f"destination dtype {destination.dtype} != {KV_FORMAT.dtype}")
            if destination.device.type == "cuda":
                device = destination.device.index
                device = torch.cuda.current_device() if device is None else device
                try:
                    rank = list(self.__engine.rt.devices).index(device)
                except ValueError as exc:
                    raise ValueError(
                        f"destination cuda:{device} is not an execution device") from exc
            elif destination.device.type == "cpu":
                rank = 0
            else:
                raise ValueError("destination must be a CPU or CUDA tensor")

            page_size = self.capabilities.kv_page_size
            engine = self.__engine
            engine.rt.streams[rank].synchronize()

            def copy_paged(pool, source_layer: int, destination_layer: int,
                           source_start: int, source_end: int,
                           destination_start: int) -> None:
                source = source_start
                while source < source_end:
                    logical_page, in_page = divmod(source, page_size)
                    if logical_page >= len(pages):
                        raise RuntimeError(
                            f"row {row} page table ends before token {source}")
                    count = min(source_end - source, page_size - in_page)
                    physical_page = pages[logical_page]
                    out = destination_start + source - source_start
                    destination[out:out + count, destination_layer, :].copy_(
                        pool[source_layer][rank][
                            physical_page, in_page:in_page + count],
                        non_blocking=False)
                    source += count

            for layer in range(_impl.N_LAYER):
                copy_paged(engine.kv.pool, layer, layer, start, end, 0)
            destination[:, _impl.N_LAYER, :].zero_()
            if engine.mtp_kv is not None:
                private_start = max(start, 1)
                if private_start < end:
                    copy_paged(engine.mtp_kv.pool, 0, _impl.N_LAYER,
                               private_start - 1, end - 1,
                               private_start - start)
            if destination.device.type == "cuda":
                torch.cuda.current_stream(destination.device).synchronize()

    def _clear_generation_pages_locked(self) -> None:
        self.__pool_next_page = 0
        self.__pool_row_pages.clear()
        self.__pool_row_tokens.clear()
        self.__prefill_row_pages.clear()
        self.__prefill_row_lengths.clear()

    def _prepared_batch_locked(
            self, sequences: Sequence[tuple[int, ...]],
    ) -> tuple[list[list[int]], list[int]]:
        """Validate row 0 and snapshot every row's loaded KV prefix."""
        if self.__phase != "input_ready":
            raise RuntimeError(
                f"generation requires input_ready, got {self.__phase}")
        anchor = tuple(int(token) for token in
                       self.__input_ids[:self.__input_length].tolist())
        if sequences[0] != anchor:
            raise ValueError("batch row 0 does not match the prepared input")
        page_indices = [list(self.__pool_row_pages.get(row, []))
                        for row in range(len(sequences))]
        loaded_lengths = [int(self.__pool_row_tokens.get(row, 0))
                          for row in range(len(sequences))]
        if any(hit > len(ids) for hit, ids in zip(loaded_lengths, sequences)):
            raise ValueError("loaded KV prefix exceeds its batch row")
        return page_indices, loaded_lengths

    def generate_batch(self, input_ids: Sequence[Sequence[int]],
                       max_new_tokens: Sequence[int],
                       cancel_events: Sequence[threading.Event],
                       emit: Callable[[int, list[int]], None],
                       eos_token_id: Optional[int] = None,
                       on_prefill: Optional[Callable[[], None]] = None,
                       stats: Optional[dict] = None,
                       select_active_rows: Optional[
                           Callable[[Sequence[int], Sequence[bool]], Sequence[int]]
                       ] = None,
                       board_request: Optional[
                           Callable[[int], Optional[BoardingRequest]]
                       ] = None,
                       on_boarded: Optional[Callable[[int, int], None]] = None,
                       boarding_interval_steps: int = 128) -> None:
        """Run the unified paged batch state machine for any resident B graph."""
        sequences = [tuple(int(token) for token in ids) for ids in input_ids]
        limits = [int(limit) for limit in max_new_tokens]
        cancels = list(cancel_events)
        batch_size = len(sequences)
        if not sequences or any(not ids for ids in sequences):
            raise ValueError("batch rows must be non-empty")
        if batch_size > self.capabilities.max_batch_size:
            raise ValueError(f"unsupported batch size B={batch_size}")
        if len(limits) != batch_size or len(cancels) != batch_size:
            raise ValueError("batch arguments must have equal lengths")
        if any(limit < 0 for limit in limits):
            raise ValueError("max_new_tokens must be non-negative")

        pages = sum(self.required_pages(len(ids), limit)
                    for ids, limit in zip(sequences, limits))
        if pages > self.capabilities.kv_pool_pages:
            raise ValueError(
                f"batch needs {pages} pages, pool has "
                f"{self.capabilities.kv_pool_pages}")
        eos = _impl.EOS_DEFAULT if eos_token_id is None else int(eos_token_id)
        for row, (ids, limit) in enumerate(zip(sequences, limits)):
            self.reserve_row_capacity(
                row, self.required_pages(len(ids), limit))
        with self.__lock:
            page_indices, loaded_lengths = self._prepared_batch_locked(sequences)
            self.__phase = "decoding"

        try:
            def remember_prefill_pages(state) -> None:
                with self.__lock:
                    self.__prefill_row_pages = {
                        row: tuple(int(page) for page in pages)
                        for row, pages in enumerate(state.page_indices)}
                    self.__prefill_row_lengths = {
                        row: int(length)
                        for row, length in enumerate(state.lengths)}

            def prepare_boarded_row(row: int):
                if board_request is None:
                    return None
                request = board_request(row)
                if request is None:
                    return None
                pages = self.required_pages(
                    len(request.input_ids), request.max_new_tokens)
                self.reserve_row_capacity(row, pages)
                with self.__lock:
                    mapping = list(self.__pool_row_pages[row])
                    loaded = int(self.__pool_row_tokens.get(row, 0))
                return (request.input_ids, request.max_new_tokens,
                        request.cancel_event, mapping, loaded)

            def remember_boarded_pages(row: int, state) -> None:
                with self.__lock:
                    self.__prefill_row_pages[row] = tuple(
                        int(page) for page in state.page_indices[0])
                    self.__prefill_row_lengths[row] = int(state.lengths[0])

            _impl.generate_mtp_batch(
                self.__engine, sequences, max_new_tokens=limits,
                eos_token_id=eos, cancel_events=cancels, emit=emit,
                on_prefill=on_prefill,
                on_prefill_state=remember_prefill_pages,
                page_indices=page_indices, loaded_lengths=loaded_lengths,
                stats=stats, select_active_rows=select_active_rows,
                board_row=prepare_boarded_row,
                on_boarded_state=remember_boarded_pages,
                on_boarded=on_boarded,
                boarding_interval_steps=boarding_interval_steps)
        finally:
            with self.__lock:
                self.__phase = "idle"
                self._clear_generation_pages_locked()

    def reset(self) -> None:
        """Reset a non-decoding execution slot after a control-path failure."""
        with self.__lock:
            if self.__phase == "decoding":
                raise RuntimeError("cannot reset while generation is running")
            self.__phase = "idle"
            self.__input_length = 0
            self._clear_generation_pages_locked()
