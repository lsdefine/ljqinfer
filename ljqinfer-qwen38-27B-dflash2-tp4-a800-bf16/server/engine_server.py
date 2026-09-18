"""Rank0 localhost RPC front-end for the persistent strategy runtime."""
from __future__ import annotations

from contextlib import asynccontextmanager
import json
import logging
from typing import Iterator, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from strategy.strategy import Strategy

_state = {"strategy": None}
_log = logging.getLogger("ljqinfer.engine")


class GenerateRequest(BaseModel):
    request_id: Optional[str] = None
    input_ids: list[int] = Field(min_length=1)
    max_new_tokens: int = Field(default=64, ge=0, le=16384)


@asynccontextmanager
async def lifespan(app: FastAPI):
    _state["strategy"] = Strategy()
    try:
        yield
    finally:
        _state["strategy"].close()
        _state["strategy"] = None


app = FastAPI(title="ljqinfer-qwen-tp4-engine", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok", "strategy": _state["strategy"] is not None,
            "backend": "dflash2_q8", "verify_width": 8}


def _event_stream(queue, request_id: str) -> Iterator[bytes]:
    strategy = _state["strategy"]
    rid = request_id
    terminal = False
    print(f"[engine] stream start id={rid}", flush=True)
    try:
        while True:
            event = queue.get()
            is_terminal = event.get("type") in ("end", "error")
            if is_terminal:
                # The consumer may close while this generator is suspended at
                # yield, so publish terminal state before exposing the event.
                terminal = True
            yield (json.dumps(event, ensure_ascii=False) + "\n").encode("utf-8")
            if is_terminal:
                print(f"[engine] stream end id={rid} type={event.get('type')}",
                      flush=True)
                return
    except Exception as exc:
        terminal = True
        _log.exception("engine stream failed id=%s", rid)
        yield (json.dumps({"type": "error",
                           "error": f"{type(exc).__name__}: {exc}"}) + "\n").encode("utf-8")
    finally:
        if not terminal:
            strategy.cancel_request(rid, reason="stream_disconnected")
            print(f"[engine] stream disconnect id={rid} cancel=True", flush=True)


@app.post("/generate")
def generate(request: GenerateRequest):
    try:
        strategy = _state["strategy"]
        if strategy is None:
            raise HTTPException(503, "engine strategy is not ready")
        # Generator bodies run after StreamingResponse sends 200 headers.
        # Validate now so capacity failures are proper HTTP 400 responses,
        # never truncated successful streams.
        strategy.validate_request(request.input_ids, request.max_new_tokens)
        queue = strategy.query(
            request.input_ids, request.max_new_tokens,
            request_id=request.request_id)
        return StreamingResponse(
            _event_stream(queue, queue.request_id),
            media_type="application/x-ndjson",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.post("/cancel/{request_id}")
def cancel(request_id: str):
    strategy = _state["strategy"]
    if strategy is None:
        raise HTTPException(503, "engine strategy is not ready")
    if not strategy.cancel_request(request_id):
        raise HTTPException(404, f"request is not active: {request_id}")
    print(f"[engine] cancel id={request_id} soft=True", flush=True)
    return {"ok": True, "request_id": request_id, "soft": True}


@app.post("/stop/{request_id}")
def stop(request_id: str, reason: str = "semantic_eos"):
    strategy = _state["strategy"]
    if strategy is None:
        raise HTTPException(503, "engine strategy is not ready")
    if not strategy.cancel_request(request_id, reason=reason):
        raise HTTPException(404, f"request is not active: {request_id}")
    print(f"[engine] stop id={request_id} reason={reason}", flush=True)
    return {"ok": True, "request_id": request_id, "reason": reason}


def main(argv=None):
    import signal
    import threading
    from contextlib import contextmanager
    import uvicorn

    class EngineServer(uvicorn.Server):
        @contextmanager
        def capture_signals(self):
            # Own the supervised process status instead of re-raising SIGTERM
            # after lifespan teardown, as Uvicorn 0.44 normally does.
            if threading.current_thread() is not threading.main_thread():
                yield
                return
            previous = {sig: signal.signal(sig, self.handle_exit)
                        for sig in (signal.SIGINT, signal.SIGTERM)}
            try:
                yield
            finally:
                for sig, handler in previous.items():
                    signal.signal(sig, handler)

    server = EngineServer(uvicorn.Config(
        app, host="127.0.0.1", port=62001, log_level="info", lifespan="on"))
    server.run()
    life = server.lifespan
    if (not server.started or server.force_exit or life.startup_failed
            or life.shutdown_failed or life.error_occurred):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
