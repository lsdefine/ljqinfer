"""Thread-safe FIFO admission queue for the strategy layer."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import threading
from typing import Any, Deque, Optional


@dataclass(slots=True)
class BatchJob:
    input_ids: tuple[int, ...]
    max_new_tokens: int
    temperature: float
    required_pages: int
    output: Any
    handle: Any


class BatchScheduler:
    """Thread-safe FIFO queue; cancellation is removed lazily at planning time."""

    def __init__(self, *, max_batch_size: int, pool_pages: int):
        self.max_batch_size = int(max_batch_size)
        self.pool_pages = int(pool_pages)
        if min(self.max_batch_size, self.pool_pages) <= 0:
            raise ValueError("scheduler capabilities must be positive")
        self._jobs: Deque[BatchJob] = deque()
        self._ready = threading.Condition()

    def put(self, job: BatchJob) -> None:
        with self._ready:
            self._jobs.append(job)
            self._ready.notify()

    def take_anchor(self) -> BatchJob:
        with self._ready:
            while not self._jobs:
                self._ready.wait()
            return self._jobs.popleft()

    def take_next(self, available_pages: int) -> Optional[BatchJob]:
        with self._ready:
            if not self._jobs:
                return None
            candidate = self._jobs[0]
            if candidate.handle.cancel_event.is_set():
                return self._jobs.popleft()
            if candidate.required_pages > available_pages:
                return None
            return self._jobs.popleft()
