"""CPU contracts for strict FIFO single-worker admission."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from queue import Queue
from threading import Condition, Event, Lock, Thread
import time
import unittest
from typing import Deque, Optional


@dataclass
class _Job:
    rid: str
    out: Queue
    cancel: Event
    hold: Event
    started: Event
    finished: Event


class MiniFIFO:
    """Minimal mirror of Strategy admission: enqueue + one persistent worker."""

    def __init__(self):
        self._ready = Condition(Lock())
        self._jobs: Deque[_Job] = deque()
        self._pending = 0
        self._worker: Optional[Thread] = None
        self.active = 0
        self.max_active = 0
        self.order: list[str] = []

    def query(self, rid: str, *, hold: Event, cancel: Optional[Event] = None) -> Queue:
        out: Queue = Queue()
        job = _Job(
            rid=rid,
            out=out,
            cancel=cancel or Event(),
            hold=hold,
            started=Event(),
            finished=Event(),
        )
        with self._ready:
            self._jobs.append(job)
            self._pending += 1
            out.queue_depth_on_submit = self._pending
            if self._worker is None or not self._worker.is_alive():
                self._worker = Thread(target=self._serve, name="mini-fifo", daemon=True)
                self._worker.start()
            self._ready.notify()
        return out

    def _serve(self) -> None:
        while True:
            with self._ready:
                while not self._jobs:
                    self._ready.wait()
                job = self._jobs.popleft()
            if job.cancel.is_set():
                job.out.put({"type": "end", "cancelled": True})
                job.finished.set()
                with self._ready:
                    self._pending = max(0, self._pending - 1)
                continue
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            self.order.append(job.rid)
            job.started.set()
            job.hold.wait(timeout=2.0)
            job.out.put({"type": "end"})
            job.finished.set()
            self.active -= 1
            with self._ready:
                self._pending = max(0, self._pending - 1)


class FIFOAdmissionTest(unittest.TestCase):
    def test_concurrent_queries_never_overlap_execution(self):
        fifo = MiniFIFO()
        holds = [Event() for _ in range(3)]
        outs = [fifo.query(f"r{i}", hold=holds[i]) for i in range(3)]
        deadline = time.time() + 2
        while time.time() < deadline and "r0" not in fifo.order:
            time.sleep(0.01)
        self.assertEqual(fifo.order, ["r0"])
        self.assertEqual(fifo.active, 1)
        self.assertEqual(outs[1].queue_depth_on_submit, 2)
        self.assertEqual(outs[2].queue_depth_on_submit, 3)
        holds[0].set(); holds[1].set(); holds[2].set()
        for out in outs:
            self.assertEqual(out.get(timeout=2)["type"], "end")
        self.assertEqual(fifo.order, ["r0", "r1", "r2"])
        self.assertEqual(fifo.max_active, 1)

    def test_cancelled_pending_job_is_skipped(self):
        fifo = MiniFIFO()
        hold0 = Event(); cancel1 = Event()
        out0 = fifo.query("r0", hold=hold0)
        out1 = fifo.query("r1", hold=Event(), cancel=cancel1)
        out2 = fifo.query("r2", hold=Event())
        deadline = time.time() + 2
        while time.time() < deadline and fifo.active != 1:
            time.sleep(0.01)
        cancel1.set(); hold0.set()
        self.assertEqual(out0.get(timeout=2)["type"], "end")
        self.assertTrue(out1.get(timeout=2).get("cancelled"))
        self.assertEqual(out2.get(timeout=2)["type"], "end")
        self.assertEqual(fifo.order, ["r0", "r2"])
        self.assertEqual(fifo.max_active, 1)


if __name__ == "__main__":
    unittest.main()
