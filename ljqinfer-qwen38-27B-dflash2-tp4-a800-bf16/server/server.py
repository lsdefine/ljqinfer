"""OpenAI-compatible HTTP surface; owns only protocol and tokenization."""
from __future__ import annotations

from contextlib import asynccontextmanager
import json
import os
import secrets
import time
from typing import Any, AsyncIterator, Dict, List, Optional, Union

import anyio
from anyio.abc import ObjectReceiveStream
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from server.openai_protocol import StreamAdapter, _usage
from server.service import ServiceError, ServiceLayer

API_KEY = os.environ.get("LJQINFER_API_KEY", "dummy_key")
MODEL_NAME = os.environ.get("LJQINFER_MODEL_NAME", "qwen3.8-27b")
ENGINE_URL = os.environ.get("LJQINFER_ENGINE_URL", "http://127.0.0.1:62001")
HEARTBEAT_SECONDS = float(os.environ.get("LJQINFER_SSE_HEARTBEAT_SECONDS", "15"))

_state: Dict[str, Any] = {"layer": None}


class ChatMessage(BaseModel):
    role: str
    content: Optional[Union[str, List[dict]]] = None
    name: Optional[str] = None
    tool_call_id: Optional[str] = None
    tool_calls: Optional[List[dict]] = None
    reasoning_content: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    model: Optional[str] = None
    messages: List[ChatMessage]
    # Newer OpenAI clients send max_completion_tokens.  Keep both fields and
    # use the same precedence as Ref's to_service_request adapter.
    max_completion_tokens: Optional[int] = Field(default=None, ge=1, le=16384)
    max_tokens: Optional[int] = Field(default=None, ge=1, le=16384)
    stream: bool = False
    stream_options: Optional[Dict[str, Any]] = None
    tools: Optional[List[dict]] = None
    tool_choice: Optional[Union[str, dict]] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    # OpenAI-compatible reasoning controls.  GenericAgent sends
    # reasoning_effort="none" for its non-thinking Qwen backend; this must not
    # be silently discarded by Pydantic or the service will waste the output
    # budget inside <think> and may never reach a tool call/final answer.
    reasoning_effort: Optional[str] = None
    thinking: Optional[Union[bool, str, dict]] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    from strategy.remote_strategy import RemoteStrategy
    _state["layer"] = ServiceLayer(RemoteStrategy(ENGINE_URL))
    yield


app = FastAPI(title="ljqinfer-qwen-tp4", version="1.0", lifespan=lifespan)


def require_api_key(
        authorization: Optional[str] = Header(default=None),
        x_api_key: Optional[str] = Header(default=None, alias="x-api-key")):
    token = x_api_key
    if authorization and authorization.startswith("Bearer "):
        token = authorization[len("Bearer "):].strip()
    if not token or not secrets.compare_digest(token, API_KEY):
        raise HTTPException(status_code=401, detail="Invalid API key")


def _layer() -> ServiceLayer:
    layer = _state.get("layer")
    if layer is None:
        raise HTTPException(503, "service not ready")
    return layer


def _openai_error_response(status_code: int, err_type: str, message: str):
    return JSONResponse(
        status_code=status_code,
        content={"error": {"message": message, "type": err_type,
                           "param": None, "code": None}})


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_NAME}


@app.get("/v1/models", dependencies=[Depends(require_api_key)])
def list_models():
    return {"object": "list",
            "data": [{"id": MODEL_NAME, "object": "model"}]}


def _completion_json(result: dict, model: str) -> dict:
    message = {"role": "assistant", "content": result.get("text") or ""}
    if result.get("reasoning"):
        message["reasoning_content"] = result["reasoning"]
    if result.get("tool_calls"):
        message["tool_calls"] = result["tool_calls"]
        message["content"] = result.get("text") or None
    finish = "tool_calls" if result.get("tool_calls") else "stop"
    return {
        "id": "chatcmpl_" + secrets.token_hex(12),
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "message": message,
            "finish_reason": finish,
            "logprobs": None,
        }],
        "usage": _usage(result["input_tokens"], result["output_tokens"],
                        result.get("metrics")),
        "ljqinfer": result.get("metrics") or {},
    }


@app.post("/v1/chat/completions", dependencies=[Depends(require_api_key)])
async def chat_completions(body: ChatCompletionRequest, request: Request):
    layer = _layer()
    model = body.model or MODEL_NAME
    messages = [m.model_dump(exclude_none=True) for m in body.messages]
    # Match Ref/OpenAI compatibility: the newer field wins when both exist;
    # if neither exists, default to 8192 when the client omits both fields.
    requested_max_tokens = (
        body.max_completion_tokens
        if body.max_completion_tokens is not None
        else body.max_tokens)
    max_tokens = int(requested_max_tokens or 8192)

    if not body.stream:
        try:
            result = await anyio.to_thread.run_sync(
                lambda: layer.complete(
                    messages, max_tokens,
                    tools=body.tools, tool_choice=body.tool_choice,
                    reasoning_effort=body.reasoning_effort,
                    thinking=body.thinking))
        except ServiceError as exc:
            return _openai_error_response(502, "server_error", str(exc))
        except ValueError as exc:
            return _openai_error_response(400, "invalid_request_error", str(exc))
        except Exception as exc:
            return _openai_error_response(500, "server_error",
                                          f"{type(exc).__name__}: {exc}")
        return _completion_json(result, model)

    include_usage = bool((body.stream_options or {}).get("include_usage"))
    adapter = StreamAdapter(include_usage=include_usage, max_tokens=max_tokens)
    import queue as pyqueue
    bridge: pyqueue.Queue = pyqueue.Queue()
    sentinel = object()
    disconnected = anyio.Event()

    def worker():
        try:
            for event in layer.stream(
                    messages, max_tokens,
                    tools=body.tools, tool_choice=body.tool_choice,
                    model=model, reasoning_effort=body.reasoning_effort,
                    thinking=body.thinking):
                if disconnected.is_set():
                    break
                bridge.put(event)
        except Exception as exc:
            bridge.put({"type": "error",
                        "error": {"type": "server_error",
                                  "message": f"{type(exc).__name__}: {exc}"}})
        finally:
            bridge.put(sentinel)

    # Start the backend before emitting SSE headers. Prefetch the first event
    # so template/service failures return a normal HTTP error instead of an
    # empty 200 stream.
    import threading
    threading.Thread(target=worker, name="openai-stream-worker", daemon=True).start()
    try:
        first = await anyio.to_thread.run_sync(bridge.get)
    except Exception as exc:
        disconnected.set()
        return _openai_error_response(
            500, "server_error", f"{type(exc).__name__}: {exc}")

    if first is sentinel:
        disconnected.set()
        return _openai_error_response(
            502, "server_error", "backend closed before first event")
    if isinstance(first, dict) and first.get("type") == "error":
        disconnected.set()
        err = first.get("error") or {}
        if isinstance(err, dict):
            msg = str(err.get("message") or err)
            err_type = str(err.get("type") or "server_error")
        else:
            msg = str(err)
            err_type = "server_error"
        status = 400 if err_type == "invalid_request_error" else 502
        return _openai_error_response(status, err_type, msg)

    async def _sse() -> AsyncIterator[bytes]:
        try:
            event = first
            while True:
                if await request.is_disconnected():
                    disconnected.set()
                    break
                for payload in adapter.feed(event):
                    if "error" in payload:
                        yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n".encode()
                        yield b"data: [DONE]\n\n"
                        disconnected.set()
                        return
                    yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n".encode()
                if event.get("type") == "message_stop":
                    yield b"data: [DONE]\n\n"
                    return
                try:
                    with anyio.fail_after(HEARTBEAT_SECONDS):
                        event = await anyio.to_thread.run_sync(bridge.get)
                except TimeoutError:
                    if await request.is_disconnected():
                        disconnected.set()
                        break
                    yield b": keepalive\n\n"
                    continue
                if event is sentinel:
                    break
        finally:
            disconnected.set()

    return StreamingResponse(
        _sse(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache",
                 "X-Accel-Buffering": "no"})



def _anthropic_error(status_code: int, err_type: str, message: str):
    return JSONResponse(
        status_code=status_code,
        content={"type": "error", "error": {"type": err_type,
                                                "message": message}})


def _anthropic_request(body: dict):
    raw_messages = body.get("messages")
    if not isinstance(raw_messages, list) or not raw_messages:
        raise ServiceError("'messages' must be a non-empty array")

    messages = []
    system = body.get("system")
    if system:
        if isinstance(system, str):
            system_text = system
        elif isinstance(system, list):
            system_text = "".join(
                str(block.get("text", "")) for block in system
                if isinstance(block, dict) and block.get("type") == "text")
        else:
            raise ServiceError("'system' must be a string or text block array")
        messages.append({"role": "system", "content": system_text})

    # Convert Anthropic blocks to the same Qwen-native history contract used
    # by Ref: assistant tool_use -> assistant.tool_calls with object arguments;
    # user tool_result -> tool turns consumed by the Qwen template.
    for entry in raw_messages:
        if not isinstance(entry, dict):
            raise ServiceError("each message must be an object")
        role = entry.get("role")
        content = entry.get("content")
        if role == "assistant" and isinstance(content, list):
            text = "".join(str(b.get("text", "")) for b in content
                           if isinstance(b, dict) and b.get("type") == "text")
            reasoning = "".join(str(b.get("thinking", "")) for b in content
                                if isinstance(b, dict) and b.get("type") == "thinking")
            calls = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_use":
                    calls.append({
                        "id": block.get("id"),
                        "type": "function",
                        "function": {"name": block.get("name", ""),
                                     "arguments": block.get("input") or {}},
                    })
            turn = {"role": "assistant", "content": text,
                    "tool_calls": calls}
            if reasoning:
                turn["reasoning_content"] = reasoning
            messages.append(turn)
            continue
        if role == "user" and isinstance(content, list):
            text_parts = []
            for block in content:
                if not isinstance(block, dict):
                    text_parts.append(str(block))
                elif block.get("type") == "tool_result":
                    value = block.get("content", "")
                    if isinstance(value, list):
                        value = "".join(str(x.get("text", "")) for x in value
                                        if isinstance(x, dict))
                    messages.append({"role": "tool", "content": str(value),
                                     "tool_call_id": block.get("tool_use_id")})
                elif block.get("type") == "text":
                    text_parts.append(str(block.get("text", "")))
            if "".join(text_parts).strip():
                messages.append({"role": "user", "content": "".join(text_parts)})
            continue
        messages.append({"role": role, "content": content})

    max_tokens = int(body.get("max_tokens") or 8192)
    if max_tokens <= 0:
        raise ServiceError("'max_tokens' must be positive")
    if max_tokens > 16384:
        raise ServiceError("'max_tokens' must be <= 16384")

    tools = None
    if body.get("tools") is not None:
        tools = []
        for tool in body.get("tools") or []:
            if not isinstance(tool, dict) or not tool.get("name"):
                raise ServiceError("each Anthropic tool requires a name")
            tools.append({"type": "function", "function": {
                "name": tool["name"],
                "description": tool.get("description", ""),
                "parameters": tool.get("input_schema") or {"type": "object"},
            }})

    choice = body.get("tool_choice")
    if isinstance(choice, dict):
        kind = choice.get("type", "auto")
        if kind == "auto":
            choice = "auto"
        elif kind == "any":
            choice = "required"
        elif kind == "tool":
            choice = {"type": "function",
                      "function": {"name": choice.get("name", "")}}
        elif kind == "none":
            choice = "none"
        else:
            raise ServiceError(f"unsupported tool_choice type: {kind!r}")
    return messages, max_tokens, tools, choice


def _anthropic_completion(result: dict, model: str) -> dict:
    content = []
    if result.get("reasoning"):
        content.append({"type": "thinking", "thinking": result["reasoning"]})
    if result.get("text"):
        content.append({"type": "text", "text": result["text"]})
    for call in result.get("tool_calls") or []:
        raw = (call.get("function") or {}).get("arguments", "{}")
        try:
            parsed = json.loads(raw)
        except (TypeError, json.JSONDecodeError):
            parsed = {}
        content.append({"type": "tool_use",
                        "id": call.get("id") or "toolu_" + secrets.token_hex(12),
                        "name": (call.get("function") or {}).get("name", ""),
                        "input": parsed})
    return {
        "id": "msg_" + secrets.token_hex(12),
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content,
        "stop_reason": "tool_use" if result.get("tool_calls") else "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": result["input_tokens"],
                  "output_tokens": result["output_tokens"]},
        "ljqinfer": result.get("metrics") or {},
    }


@app.post("/v1/messages", dependencies=[Depends(require_api_key)])
async def messages_api(request: Request):
    try:
        body = await request.json()
        if not isinstance(body, dict):
            raise ServiceError("request body must be a JSON object")
        messages, max_tokens, tools, tool_choice = _anthropic_request(body)
    except (ServiceError, ValueError, json.JSONDecodeError) as exc:
        return _anthropic_error(400, "invalid_request_error", str(exc))

    model = str(body.get("model") or MODEL_NAME)
    layer = _layer()
    if not bool(body.get("stream")):
        try:
            result = await anyio.to_thread.run_sync(
                lambda: layer.complete(messages, max_tokens,
                                       tools=tools, tool_choice=tool_choice))
        except ServiceError as exc:
            return _anthropic_error(400, "invalid_request_error", str(exc))
        except Exception as exc:
            return _anthropic_error(
                500, "api_error", f"{type(exc).__name__}: {exc}")
        return _anthropic_completion(result, model)

    async def _anthropic_sse() -> AsyncIterator[bytes]:
        import queue as pyqueue
        bridge: pyqueue.Queue = pyqueue.Queue()
        sentinel = object()
        disconnected = anyio.Event()

        def worker():
            try:
                for event in layer.stream(messages, max_tokens, tools=tools,
                                          tool_choice=tool_choice, model=model):
                    if disconnected.is_set():
                        break
                    bridge.put(event)
            except Exception as exc:
                bridge.put({"type": "error",
                            "error": {"type": "api_error",
                                      "message": f"{type(exc).__name__}: {exc}"}})
            finally:
                bridge.put(sentinel)

        async with anyio.create_task_group() as tg:
            tg.start_soon(anyio.to_thread.run_sync, worker)
            try:
                while True:
                    try:
                        with anyio.fail_after(HEARTBEAT_SECONDS):
                            event = await anyio.to_thread.run_sync(bridge.get)
                    except TimeoutError:
                        if await request.is_disconnected():
                            disconnected.set()
                            break
                        yield b": keepalive\n\n"
                        continue
                    if event is sentinel:
                        break
                    if await request.is_disconnected():
                        disconnected.set()
                        break
                    name = str(event.get("type", "message"))
                    data = json.dumps(event, ensure_ascii=False,
                                      separators=(",", ":"))
                    yield f"event: {name}\ndata: {data}\n\n".encode("utf-8")
                    if name in ("message_stop", "error"):
                        break
            finally:
                disconnected.set()

    return StreamingResponse(
        _anthropic_sse(), media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


def main(argv=None):
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=18084, log_level="info")


if __name__ == "__main__":
    main()
