#!/usr/bin/env python3
"""CPU contracts for FIFO admission and lazy cancellation."""
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from strategy.batch_scheduler import BatchJob, BatchScheduler


def _job(name: str, pages: int, *, cancelled: bool = False) -> BatchJob:
    event = threading.Event()
    if cancelled:
        event.set()
    return BatchJob(input_ids=(1,), max_new_tokens=1,
                    required_pages=pages, output=None,
                    handle=SimpleNamespace(name=name, cancel_event=event))


def test_cancelled_fifo_head_is_returned_for_cleanup():
    scheduler = BatchScheduler(max_batch_size=4, pool_pages=16)
    cancelled = _job("cancelled", 2, cancelled=True)
    live = _job("live", 2)
    scheduler.put(cancelled)
    scheduler.put(live)

    assert scheduler.take_next(16) is cancelled
    assert scheduler.take_next(16) is live


def test_many_cancelled_heads_can_be_drained_iteratively():
    scheduler = BatchScheduler(max_batch_size=4, pool_pages=16)
    cancelled = [_job(f"cancelled-{i}", 10_000, cancelled=True)
                 for i in range(2048)]
    tail = _job("tail", 1)
    for job in cancelled:
        scheduler.put(job)
    scheduler.put(tail)

    drained = []
    while True:
        job = scheduler.take_next(1)
        if job is None or not job.handle.cancel_event.is_set():
            break
        drained.append(job)
    assert drained == cancelled
    assert job is tail


def test_oversized_live_fifo_head_is_not_bypassed():
    scheduler = BatchScheduler(max_batch_size=4, pool_pages=16)
    head = _job("head", 9)
    tail = _job("tail", 1)
    scheduler.put(head)
    scheduler.put(tail)

    assert scheduler.take_next(8) is None
    assert scheduler.take_next(9) is head
    assert scheduler.take_next(1) is tail


if __name__ == "__main__":
    test_cancelled_fifo_head_is_returned_for_cleanup()
    test_many_cancelled_heads_can_be_drained_iteratively()
    test_oversized_live_fifo_head_is_not_bypassed()
    print("BATCH_SCHEDULER_CONTRACT_PASS")
