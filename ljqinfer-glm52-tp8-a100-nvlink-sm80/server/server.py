"""Anthropic HTTP API backed by the separately running GPU engine.

This process owns only protocol, templates and tokenization.  The model and
strategy runtime stay alive behind localhost RPC when this service restarts.
"""

from __future__ import annotations

import json
import os
import threading
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional

import anyio
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from server import service
from server.openai_protocol import (
    StreamAdapter,
    completion_response,
    to_service_request,
)
from server.service import ServiceError, ServiceLayer

DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8000
DEFAULT_ENGINE_URL = "http://127.0.0.1:62001"
API_KEY = "dummy_key"
MODEL_NAME = "ljqinfer-glm-5.2"
MAX_OUTPUT_TOKENS = 8192
MAX_TOTAL_TOKENS = 68 * 1024

_state: Dict[str, Any] = {"service": None, "config": {}}


def _expected_key() -> Optional[str]:
    return API_KEY or None


async def authorize(authorization: Optional[str] = Header(default=None),
                    x_api_key: Optional[str] = Header(default=None,
                                                      alias="x-api-key")) -> None:
    """Accept either ``Authorization: Bearer <k>`` or ``x-api-key: <k>``."""
    expected = _expected_key()
    if expected is None:                      # unset key == open endpoint
        return
    supplied = x_api_key
    if not supplied and authorization:
        parts = authorization.split(None, 1)
        if len(parts) == 2 and parts[0].lower() == "bearer":
            supplied = parts[1]
        else:
            supplied = authorization
    if not supplied or supplied.strip() != expected:
        raise HTTPException(status_code=401, detail={
            "type": "error",
            "error": {"type": "authentication_error",
                      "message": "invalid api key"}})


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Attach the lightweight service layer to the persistent GPU engine."""
    from strategy.remote_strategy import RemoteStrategy
    cfg = _state["config"]
    _state["service"] = ServiceLayer(
        RemoteStrategy(cfg.get("engine_url", DEFAULT_ENGINE_URL)),
        model_name=cfg.get("model_name", "ljqinfer-glm52"),
        max_output_tokens=cfg.get("max_output_tokens", 4096),
        max_total_tokens=cfg.get("max_total_tokens", 68 * 1024))
    yield
    _state["service"] = None


app = FastAPI(title="ljqinfer", version="1.0", lifespan=lifespan)


@app.exception_handler(HTTPException)
async def _http_error(request: Request, exc: HTTPException) -> JSONResponse:
    """Keep Anthropic's flat error envelope instead of FastAPI's detail wrap."""
    detail = exc.detail
    if isinstance(detail, dict) and "error" in detail:
        return JSONResponse(status_code=exc.status_code, content=detail)
    return JSONResponse(status_code=exc.status_code, content={
        "type": "error",
        "error": {"type": "api_error", "message": str(detail)}})


def _layer() -> ServiceLayer:
    layer = _state.get("service")
    if layer is None:
        raise HTTPException(status_code=503, detail={
            "type": "error",
            "error": {"type": "overloaded_error",
                      "message": "service layer is not ready"}})
    return layer


@app.get("/health")
async def health() -> Dict[str, Any]:
    return {"status": "ok" if _state.get("service") else "starting",
            "model": _state["config"].get("model_name", "ljqinfer-glm52")}


@app.get("/v1/models", dependencies=[Depends(authorize)])
async def models() -> Dict[str, Any]:
    name = _state["config"].get("model_name", "ljqinfer-glm52")
    return {"data": [{"id": name, "type": "model",
                      "display_name": name}], "has_more": False}


def _error_response(status: int, kind: str, message: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={
        "type": "error", "error": {"type": kind, "message": message}})


def _standard_cache_usage(usage: Dict[str, Any], stats: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Add Anthropic-compatible prompt-cache counters to a usage object."""
    out = dict(usage or {})
    stats = stats or {}
    total_input_tokens = int(
        out.get("input_tokens", stats.get("input_tokens", 0)) or 0)
    read_tokens = max(0, int(stats.get("cache_hit_tokens", 0) or 0))
    block_size = max(1, int(stats.get("cache_block_size", 0) or 1))
    stored_blocks = max(0, int(stats.get("cache_stored_blocks", 0) or 0))
    creation_tokens = min(stored_blocks * block_size,
                          max(0, total_input_tokens - read_tokens))
    # Claude/Anthropic input_tokens excludes prompt-cache tokens.
    out["input_tokens"] = max(
        0, total_input_tokens - read_tokens - creation_tokens)
    out["cache_creation_input_tokens"] = creation_tokens
    out["cache_read_input_tokens"] = read_tokens
    return out


def _protocol_event(event: Dict[str, Any]) -> Dict[str, Any]:
    """Adapt service events to Anthropic usage fields without mutating them."""
    out = dict(event)
    stats = out.get("ljqinfer")
    if out.get("type") == "message_start":
        message = dict(out.get("message") or {})
        message["usage"] = _standard_cache_usage(
            message.get("usage") or {}, stats if isinstance(stats, dict) else None)
        out["message"] = message
    elif isinstance(stats, dict):
        out["usage"] = _standard_cache_usage(out.get("usage") or {}, stats)
    return out


@app.post("/v1/messages", dependencies=[Depends(authorize)])
async def messages(request: Request):
    """Anthropic-compatible messages endpoint (streaming and blocking)."""
    try:
        body = await request.json()
    except Exception:
        return _error_response(400, "invalid_request_error", "malformed json")
    if not isinstance(body, dict):
        return _error_response(400, "invalid_request_error",
                               "body must be an object")
    layer = _layer()

    if not bool(body.get("stream")):
        try:
            result = await anyio.to_thread.run_sync(layer.generate, body)
        except ServiceError as exc:
            return _error_response(400, "invalid_request_error", str(exc))
        except Exception as exc:                       # pragma: no cover
            return _error_response(500, "api_error", repr(exc))
        payload = result.to_dict()
        payload["usage"] = _standard_cache_usage(
            payload.get("usage") or {}, payload.get("ljqinfer"))
        return JSONResponse(content=payload)

    send, receive = anyio.create_memory_object_stream(64)

    class _CancelRelay:
        """Thread-safe bridge from ASGI disconnect to a late query handle."""

        def __init__(self) -> None:
            self._lock = threading.Lock()
            self._handle = None
            self._cancelled = False

        def bind(self, handle: Any) -> None:
            with self._lock:
                if self._cancelled:
                    if handle is not None:
                        handle.cancel()
                else:
                    self._handle = handle

        def cancel(self) -> None:
            with self._lock:
                self._cancelled = True
                if self._handle is not None:
                    self._handle.cancel()

    cancel_relay = _CancelRelay()

    def _produce() -> None:
        events = layer.stream(body, on_submit=cancel_relay.bind)
        try:
            for event in events:
                anyio.from_thread.run(send.send, _protocol_event(event))
        except (anyio.BrokenResourceError, anyio.ClosedResourceError):
            print("[server] client disconnected; closing generation", flush=True)
        except ServiceError as exc:
            anyio.from_thread.run(send.send, {
                "type": "error",
                "error": {"type": "invalid_request_error",
                          "message": str(exc)}})
        except Exception as exc:                       # pragma: no cover
            anyio.from_thread.run(send.send, {
                "type": "error",
                "error": {"type": "api_error", "message": repr(exc)}})
        finally:
            events.close()
            try:
                anyio.from_thread.run(send.aclose)
            except (anyio.BrokenResourceError, anyio.ClosedResourceError):
                pass

    async def _sse():
        disconnected = anyio.Event()

        async def _watch_disconnect():
            while not disconnected.is_set():
                if await request.is_disconnected():
                    print("[server] client disconnected; closing generation", flush=True)
                    cancel_relay.cancel()
                    disconnected.set()
                    await receive.aclose()
                    return
                await anyio.sleep(0.05)

        async with anyio.create_task_group() as group:
            group.start_soon(anyio.to_thread.run_sync, _produce)
            group.start_soon(_watch_disconnect)
            try:
                async with receive:
                    while True:
                        event = None
                        with anyio.move_on_after(10) as timeout_scope:
                            try:
                                event = await receive.receive()
                            except anyio.EndOfStream:
                                break
                        if timeout_scope.cancel_called:
                            # SSE comment: keeps proxies alive without adding a
                            # non-Anthropic protocol event for clients to parse.
                            yield ": heartbeat\n\n"
                            continue
                        payload = json.dumps(event, ensure_ascii=False)
                        yield f"event: {event['type']}\ndata: {payload}\n\n"
            finally:
                # StreamingResponse cancellation is often delivered here before
                # Request.is_disconnected() observes the socket close.
                cancel_relay.cancel()
                disconnected.set()
                await receive.aclose()

    return StreamingResponse(_sse(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})


def _openai_error_response(status: int, kind: str,
                           message: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": {
        "message": message, "type": kind, "param": None, "code": None}})


@app.post("/v1/chat/completions", dependencies=[Depends(authorize)])
async def chat_completions(request: Request):
    """OpenAI-compatible Chat Completions endpoint."""
    try:
        body = await request.json()
    except Exception:
        return _openai_error_response(
            400, "invalid_request_error", "malformed json")
    try:
        service_request = to_service_request(body)
    except ServiceError as exc:
        return _openai_error_response(400, "invalid_request_error", str(exc))

    layer = _layer()
    if not bool(body.get("stream")):
        try:
            result = await anyio.to_thread.run_sync(
                layer.generate, service_request)
        except ServiceError as exc:
            return _openai_error_response(
                400, "invalid_request_error", str(exc))
        except Exception as exc:                       # pragma: no cover
            return _openai_error_response(500, "server_error", repr(exc))
        return JSONResponse(content=completion_response(
            result, max_tokens=service_request.get("max_tokens")))

    send, receive = anyio.create_memory_object_stream(64)

    class _CancelRelay:
        def __init__(self) -> None:
            self._lock = threading.Lock()
            self._handle = None
            self._cancelled = False

        def bind(self, handle: Any) -> None:
            with self._lock:
                if self._cancelled:
                    if handle is not None:
                        handle.cancel()
                else:
                    self._handle = handle

        def cancel(self) -> None:
            with self._lock:
                self._cancelled = True
                if self._handle is not None:
                    self._handle.cancel()

    cancel_relay = _CancelRelay()
    adapter = StreamAdapter(
        include_usage=bool((body.get("stream_options") or {}).get(
            "include_usage")),
        max_tokens=service_request.get("max_tokens"))

    def _produce() -> None:
        events = layer.stream(service_request, on_submit=cancel_relay.bind)
        try:
            for event in events:
                for chunk in adapter.feed(event):
                    anyio.from_thread.run(send.send, chunk)
        except (anyio.BrokenResourceError, anyio.ClosedResourceError):
            print("[server] OpenAI client disconnected; closing generation",
                  flush=True)
        except ServiceError as exc:
            anyio.from_thread.run(send.send, {"error": {
                "message": str(exc), "type": "invalid_request_error",
                "param": None, "code": None}})
        except Exception as exc:                       # pragma: no cover
            anyio.from_thread.run(send.send, {"error": {
                "message": repr(exc), "type": "server_error",
                "param": None, "code": None}})
        finally:
            events.close()
            try:
                anyio.from_thread.run(send.aclose)
            except (anyio.BrokenResourceError, anyio.ClosedResourceError):
                pass

    async def _sse():
        disconnected = anyio.Event()

        async def _watch_disconnect():
            while not disconnected.is_set():
                if await request.is_disconnected():
                    print("[server] OpenAI client disconnected; closing generation",
                          flush=True)
                    cancel_relay.cancel()
                    disconnected.set()
                    await receive.aclose()
                    return
                await anyio.sleep(0.05)

        async with anyio.create_task_group() as group:
            group.start_soon(anyio.to_thread.run_sync, _produce)
            group.start_soon(_watch_disconnect)
            try:
                async with receive:
                    while True:
                        chunk = None
                        with anyio.move_on_after(10) as timeout_scope:
                            try:
                                chunk = await receive.receive()
                            except anyio.EndOfStream:
                                break
                        if timeout_scope.cancel_called:
                            yield ": heartbeat\n\n"
                            continue
                        yield "data: " + json.dumps(
                            chunk, ensure_ascii=False) + "\n\n"
                yield "data: [DONE]\n\n"
            finally:
                cancel_relay.cancel()
                disconnected.set()
                await receive.aclose()

    return StreamingResponse(_sse(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})


def main(argv=None) -> None:
    _state["config"] = {
        "engine_url": DEFAULT_ENGINE_URL,
        "model_name": MODEL_NAME,
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "max_total_tokens": MAX_TOTAL_TOKENS,
    }
    import uvicorn
    uvicorn.run(app, host=DEFAULT_HOST, port=DEFAULT_PORT, log_level="info")


if __name__ == "__main__":
    main()