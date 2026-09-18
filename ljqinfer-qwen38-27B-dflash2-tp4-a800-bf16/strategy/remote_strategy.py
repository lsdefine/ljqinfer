"""Localhost proxy presenting the strategy Queue interface."""
from __future__ import annotations

import json
import threading
import uuid
from queue import Queue
from types import SimpleNamespace

import requests


class _Handle:
    state = "pending"

    def __init__(self, base, request_id):
        self.base = base
        self.request_id = request_id

    def _stop(self, semantic=False):
        if self.state == "done":
            return
        path = "/stop" if semantic else "/cancel"
        try:
            response = requests.post(
                f"{self.base}{path}/{self.request_id}",
                params={"reason": "semantic_eos"} if semantic else None,
                timeout=5)
            if response.status_code not in (200, 404):
                response.raise_for_status()
        except requests.RequestException as exc:
            print(f"[remote] stop failed id={self.request_id}: {exc}", flush=True)

    def cancel(self):
        if self.state in ("done", "cancelled"):
            return
        self._stop()
        self.state = "cancelled"

    def stop_at_semantic_eos(self):
        if self.state not in ("done", "cancelled"):
            self._stop(True)


class RemoteStrategy:
    def __init__(self, base: str = "http://127.0.0.1:62001"):
        self.base = base.rstrip("/")

    def generate(self, input_ids, max_new_tokens: int = 64) -> dict:
        """Blocking compatibility path for existing non-stream callers."""
        queue = self.query(input_ids, max_new_tokens)
        token_ids = []
        metrics = {}
        while True:
            event = queue.get()
            kind = event.get("type")
            if kind == "prefill":
                metrics = event.get("metrics") or metrics
                continue
            if kind == "token":
                token_ids.extend(event.get("token_ids") or [])
                continue
            if kind == "end":
                if hasattr(queue, "metrics") and queue.metrics is not None:
                    metrics = vars(queue.metrics) if not isinstance(queue.metrics, dict) else queue.metrics
                return {"token_ids": token_ids, "metrics": metrics}
            if kind == "error":
                raise RuntimeError(event.get("error") or "generation failed")
            raise RuntimeError(f"unknown strategy event: {kind!r}")

    def query(self, input_ids, max_new_tokens: int = 64) -> Queue:
        out, rid = Queue(), "rpc_" + uuid.uuid4().hex[:16]
        response = requests.post(
            f"{self.base}/generate",
            stream=True,
            json={"request_id": rid,
                  "input_ids": list(input_ids),
                  "max_new_tokens": int(max_new_tokens)},
            timeout=None)
        response.raise_for_status()
        out.request_id = rid
        out.metrics = None
        out.queue_depth_on_submit = 0
        out.cancel_handle = handle = _Handle(self.base, rid)

        def pump():
            try:
                for line in response.iter_lines(chunk_size=1, decode_unicode=True):
                    if not line:
                        continue
                    event = json.loads(line)
                    if event.get("type") == "prefill":
                        out.metrics = SimpleNamespace(**(event.get("metrics") or {}))
                    if event.get("type") == "end":
                        out.decode_stats = {
                            k: event[k]
                            for k in ("decode_steps", "accepted_tokens")
                            if k in event
                        }
                    if event.get("type") in ("end", "error"):
                        # Queue.put publishes to the service thread; expose the
                        # terminal state before that event becomes observable.
                        handle.state = (
                            "cancelled" if event.get("cancelled") else "done")
                        out.put(event)
                        return
                    out.put(event)
                raise RuntimeError("engine stream closed without terminal event")
            except Exception as exc:
                out.put({"type": "error", "error": repr(exc)})
            finally:
                if handle.state not in ("done", "cancelled"):
                    handle.cancel()
                response.close()

        threading.Thread(target=pump, name=rid, daemon=True).start()
        return out

    def health(self) -> dict:
        response = requests.get(self.base + "/health", timeout=5)
        response.raise_for_status()
        return response.json()
