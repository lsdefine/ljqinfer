"""Bounded serial cache work; no model, rank or batch scheduling knowledge."""
from collections import deque
from concurrent.futures import ThreadPoolExecutor


class AsyncWriteback:
    def __init__(self, max_pending=2):
        if max_pending < 1:
            raise ValueError('positive pending limit required')
        self.max_pending = max_pending
        self.pending = deque()
        self.worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix='cold-writeback')
        self.closed = False

    def submit(self, function, *args):
        if self.closed:
            raise RuntimeError('closed cache writer')
        while self.pending and self.pending[0].done():
            self.pending.popleft().result()
        if len(self.pending) >= self.max_pending:
            self.pending.popleft().result()
        self.pending.append(self.worker.submit(function, *args))

    def drain(self):
        # Wait for ALL source readers even when an earlier transaction failed.
        error = None
        while self.pending:
            try:
                self.pending.popleft().result()
            except BaseException as exc:
                if error is None:
                    error = exc
        if error is not None:
            raise error

    def close(self):
        self.closed = True
        try:
            self.drain()
        finally:
            self.worker.shutdown(wait=True)
