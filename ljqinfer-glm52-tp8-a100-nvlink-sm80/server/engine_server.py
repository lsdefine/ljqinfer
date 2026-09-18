"""Private localhost RPC wrapper around the unchanged strategy runtime."""
from __future__ import annotations

import json
import threading
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse

from strategy import strategy

_state = {"config": {}, "handles": {}, "lock": threading.Lock(), "ready": False}


@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = _state["config"]
    started = time.perf_counter()
    print("[startup] engine_server lifespan begin", flush=True)
    strategy.startup(devices=cfg["devices"],
                     prefill_chunk_tokens=cfg["prefill_chunk_tokens"])
    _state["ready"] = True
    print(f"[startup] engine_server ready total={time.perf_counter()-started:.3f}s", flush=True)
    yield
    _state["ready"] = False


app = FastAPI(title="ljqinfer-engine", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok" if _state["ready"] else "starting"}


@app.post("/generate")
def generate(body: dict):
    request_id = str(body["request_id"])
    queue = strategy.query(body["input_ids"], int(body["max_new_tokens"]))
    with _state["lock"]:
        if request_id in _state["handles"]:
            raise HTTPException(409, "duplicate request_id")
        _state["handles"][request_id] = queue.cancel_handle

    def events():
        complete = False
        try:
            while True:
                event = queue.get()
                yield json.dumps(event, separators=(",", ":")) + "\n"
                if event.get("type") in ("end", "error"):
                    complete = True
                    return
        finally:
            with _state["lock"]:
                handle = _state["handles"].pop(request_id, None)
            if handle is not None and not complete:
                handle.cancel()

    return StreamingResponse(events(), media_type="application/x-ndjson")


def _request_handle(request_id: str):
    with _state["lock"]:
        handle = _state["handles"].get(request_id)
    if handle is None:
        raise HTTPException(404, "request not found")
    return handle


@app.post("/stop/{request_id}")
def stop(request_id: str, reason: str = "semantic_eos"):
    if reason != "semantic_eos":
        raise HTTPException(400, f"unsupported stop reason: {reason}")
    changed = _request_handle(request_id).stop_at_semantic_eos()
    return {"stopped": changed, "reason": reason}


@app.post("/cancel/{request_id}")
def cancel(request_id: str, semantic: bool = False):
    handle = _request_handle(request_id)
    changed = (handle.stop_at_semantic_eos() if semantic else handle.cancel())
    return {"cancelled": changed}


# Fixed single-machine deployment: no CLI args, no env switches.
HOST = "127.0.0.1"
PORT = 62001
DEVICES = None
PREFILL_CHUNK_TOKENS = None


def main(argv=None):
    _state["config"] = {"devices": DEVICES,
                        "prefill_chunk_tokens": PREFILL_CHUNK_TOKENS}
    import uvicorn
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")


if __name__ == "__main__":
    main()
