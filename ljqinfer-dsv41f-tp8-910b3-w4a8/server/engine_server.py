"""Original localhost generation RPC; unavailable until decode is integrated."""
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
    # A generation backend is attached only after real model/decode integration.
    _state["ready"] = callable(getattr(strategy, "query", None))
    yield
    _state["ready"] = False


app = FastAPI(title="ljqinfer-engine", lifespan=lifespan)


@app.get("/health")
def health():
    if not _state["ready"]:
        raise HTTPException(503, "V4.1 generation backend is not integrated")
    return {"status": "ok"}


@app.post("/generate")
def generate(body: dict):
    if not _state["ready"]:
        raise HTTPException(503, "V4.1 generation backend is not integrated")
    try:
        request_id = str(body["request_id"])
        input_ids = body["input_ids"]
        max_new = body["max_new_tokens"]
        temperature = body.get("temperature", 0.0)
    except (KeyError, TypeError, ValueError) as exc:
        raise HTTPException(400, "invalid request: %s" % exc)
    # query only validates/enqueues CPU work. Serialize it with registration so
    # duplicate IDs cannot create an unreachable generation job.
    with _state["lock"]:
        if request_id in _state["handles"]:
            raise HTTPException(409, "duplicate request_id")
        try:
            queue = strategy.query(input_ids, max_new, temperature,
                                   images=body.get("images"))
        except (KeyError, TypeError, ValueError) as exc:
            raise HTTPException(400, "invalid request: %s" % exc)
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
    """Live handle for request_id, or None if it already finished.

    Cancellation is IDEMPOTENT. Handles are popped the moment a request
    ends, so a stop/cancel that loses the race against a natural EOS finds
    nothing -- that is normal client timing, not an error. Returning 404
    here forced callers to guess whether the id was bogus or merely done,
    which showed up as false alarms in upstream logs.
    """
    with _state["lock"]:
        return _state["handles"].get(request_id)


@app.post("/stop/{request_id}")
def stop(request_id: str, reason: str = "semantic_eos"):
    if reason != "semantic_eos":
        raise HTTPException(400, f"unsupported stop reason: {reason}")
    handle = _request_handle(request_id)
    if handle is None:
        return {"stopped": False, "reason": reason,
                "state": "already_finished"}
    return {"stopped": handle.stop_at_semantic_eos(), "reason": reason,
            "state": "running"}


@app.post("/cancel/{request_id}")
def cancel(request_id: str, semantic: bool = False):
    handle = _request_handle(request_id)
    if handle is None:
        return {"cancelled": False, "state": "already_finished"}
    changed = (handle.stop_at_semantic_eos() if semantic else handle.cancel())
    return {"cancelled": changed, "state": "running"}


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
