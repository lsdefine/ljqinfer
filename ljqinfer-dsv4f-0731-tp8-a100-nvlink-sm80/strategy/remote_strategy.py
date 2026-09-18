"""Localhost proxy presenting the strategy Queue interface."""
import json, threading, uuid
from queue import Queue
from types import SimpleNamespace
import requests

class _Handle:
    state = "pending"
    def __init__(self, base, request_id): self.base, self.request_id = base, request_id
    def _stop(self, semantic=False):
        if self.state == "done": return
        if semantic:
            response = requests.post(f"{self.base}/stop/{self.request_id}",
                                     params={"reason": "semantic_eos"})
            if response.status_code == 404:
                response = requests.post(f"{self.base}/cancel/{self.request_id}",
                                         params={"semantic": True})
        else:
            response = requests.post(f"{self.base}/cancel/{self.request_id}")
        if response.status_code != 404: response.raise_for_status()
    def cancel(self):
        if self.state in ("done", "cancelled"): return
        self._stop(); self.state = "cancelled"
    def stop_at_semantic_eos(self):
        if self.state not in ("done", "cancelled"): self._stop(True)

class RemoteStrategy:
    def __init__(self, base="http://127.0.0.1:62001"): self.base = base.rstrip("/")
    def query(self, input_ids, max_new_tokens, temperature=1.0):
        out, rid = Queue(), "rpc_" + uuid.uuid4().hex[:16]
        response = requests.post(f"{self.base}/generate", stream=True,
            json={"request_id": rid, "input_ids": list(input_ids),
                  "max_new_tokens": max_new_tokens,
                  "temperature": temperature})
        response.raise_for_status()
        out.request_id, out.metrics, out.queue_depth_on_submit = rid, None, 0
        out.cancel_handle = handle = _Handle(self.base, rid)
        def pump():
            try:
                for line in response.iter_lines(chunk_size=1):
                    if not line: continue
                    event = json.loads(line)
                    if event.get("type") == "prefill":
                        out.metrics = SimpleNamespace(**event.get("metrics", {}))
                    if event.get("type") == "end":
                        out.decode_stats = {
                            k: event[k]
                            for k in ("decode_steps", "accepted_tokens")
                            if k in event}
                    out.put(event)
                    if event.get("type") in ("end", "error"):
                        handle.state = "done"; return
                raise RuntimeError("engine stream closed without terminal event")
            except Exception as exc: out.put({"type": "error", "error": repr(exc)})
            finally: response.close()
        threading.Thread(target=pump, name=rid, daemon=True).start()
        return out
