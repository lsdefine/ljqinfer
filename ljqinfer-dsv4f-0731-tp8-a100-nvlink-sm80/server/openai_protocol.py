"""OpenAI Chat Completions protocol adapters for the ljqinfer service layer.

The GPU-facing service speaks one internal Messages-shaped request/event
contract.  This module keeps OpenAI wire compatibility at the HTTP boundary.
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Any, Dict, List, Optional

from server.service import GenerationResult, ServiceError


def _text_content(content: Any) -> Any:
    """Normalize OpenAI text content while preserving the simple string form."""
    if content is None or isinstance(content, str):
        return content or ""
    if not isinstance(content, list):
        raise ServiceError("message 'content' must be a string or an array")
    blocks: List[Dict[str, Any]] = []
    for item in content:
        if isinstance(item, str):
            blocks.append({"type": "text", "text": item})
        elif isinstance(item, dict) and item.get("type") in ("text", "input_text"):
            blocks.append({"type": "text", "text": item.get("text", "")})
        else:
            raise ServiceError("only text message content is supported")
    return blocks


def _tool_arguments(raw: Any) -> Dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if raw in (None, ""):
        return {}
    if not isinstance(raw, str):
        raise ServiceError("tool call arguments must be a JSON object string")
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ServiceError(f"invalid tool call arguments JSON: {exc.msg}") from exc
    if not isinstance(value, dict):
        raise ServiceError("tool call arguments must decode to an object")
    return value


def to_service_request(body: Dict[str, Any]) -> Dict[str, Any]:
    """Convert an OpenAI chat request to the existing service request contract."""
    if not isinstance(body, dict):
        raise ServiceError("request body must be a JSON object")
    if body.get("n", 1) != 1:
        raise ServiceError("only n=1 is supported")
    response_format = body.get("response_format")
    if response_format not in (None, {"type": "text"}):
        raise ServiceError("only response_format.type='text' is supported")
    if body.get("stop") not in (None, [], ""):
        raise ServiceError("custom stop sequences are not supported")
    tool_choice = body.get("tool_choice", "auto")
    if tool_choice not in (None, "auto", "none"):
        raise ServiceError("only tool_choice='auto' or 'none' is supported")
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ServiceError("'messages' must be a non-empty array")

    system_parts: List[str] = []
    converted: List[Dict[str, Any]] = []
    for entry in messages:
        if not isinstance(entry, dict):
            raise ServiceError("each message must be an object")
        role = entry.get("role")
        if role in ("system", "developer"):
            value = _text_content(entry.get("content"))
            if isinstance(value, list):
                value = "".join(block.get("text", "") for block in value)
            system_parts.append(value)
            continue
        if role == "user":
            converted.append({"role": "user",
                              "content": _text_content(entry.get("content"))})
            continue
        if role == "assistant":
            content: List[Dict[str, Any]] = []
            text = _text_content(entry.get("content"))
            if isinstance(text, str):
                if text:
                    content.append({"type": "text", "text": text})
            else:
                content.extend(text)
            reasoning = entry.get("reasoning_content")
            if reasoning:
                content.insert(0, {"type": "thinking", "thinking": str(reasoning)})
            tool_calls = entry.get("tool_calls") or []
            if not isinstance(tool_calls, list):
                raise ServiceError("assistant 'tool_calls' must be an array")
            for call in tool_calls:
                if not isinstance(call, dict):
                    raise ServiceError("each tool call must be an object")
                function = call.get("function") or {}
                name = function.get("name")
                if not name:
                    raise ServiceError("tool call function name is required")
                content.append({"type": "tool_use",
                                "id": call.get("id") or f"call_{uuid.uuid4().hex[:24]}",
                                "name": name,
                                "input": _tool_arguments(function.get("arguments"))})
            converted.append({"role": "assistant", "content": content})
            continue
        if role == "tool":
            converted.append({
                "role": "user",
                "content": [{"type": "tool_result",
                             "tool_use_id": entry.get("tool_call_id") or "",
                             "content": _text_content(entry.get("content"))}],
            })
            continue
        raise ServiceError(f"unsupported message role: {role!r}")

    if not converted:
        raise ServiceError("at least one user, assistant, or tool message is required")
    request: Dict[str, Any] = {
        "model": body.get("model"),
        "messages": converted,
        "max_tokens": body.get("max_completion_tokens",
                               body.get("max_tokens")),
        "stream": bool(body.get("stream")),
        "temperature": body.get("temperature"),
    }
    if system_parts:
        request["system"] = "\n\n".join(system_parts)
    if tool_choice != "none" and "tools" in body:
        request["tools"] = body["tools"]
    for key in ("thinking", "reasoning_effort", "reasoning",
                "clear_thinking"):
        if key in body:
            request[key] = body[key]
    return request


def _usage(input_tokens: int, output_tokens: int,
           stats: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    stats = stats or {}
    prompt = int(input_tokens or 0)
    completion = int(output_tokens or 0)
    usage: Dict[str, Any] = {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": prompt + completion,
    }
    cached = max(0, int(stats.get("cache_hit_tokens", 0) or 0))
    if cached:
        usage["prompt_tokens_details"] = {"cached_tokens": cached}
    return usage


def completion_response(result: GenerationResult, *,
                        max_tokens: Optional[int] = None) -> Dict[str, Any]:
    text_parts: List[str] = []
    reasoning_parts: List[str] = []
    tool_calls: List[Dict[str, Any]] = []
    for block in result.content:
        kind = block.get("type")
        if kind == "text":
            text_parts.append(block.get("text", ""))
        elif kind == "thinking":
            reasoning_parts.append(block.get("thinking", ""))
        elif kind == "tool_use":
            tool_calls.append({
                "id": block.get("id") or f"call_{uuid.uuid4().hex[:24]}",
                "type": "function",
                "function": {
                    "name": block.get("name", ""),
                    "arguments": json.dumps(block.get("input") or {},
                                            ensure_ascii=False,
                                            separators=(",", ":")),
                },
            })
    message: Dict[str, Any] = {
        "role": "assistant",
        "content": "".join(text_parts) or None,
        "refusal": None,
    }
    if reasoning_parts:
        message["reasoning_content"] = "".join(reasoning_parts)
    if tool_calls:
        message["tool_calls"] = tool_calls
    finish_reason = (
        "tool_calls" if tool_calls else
        "length" if max_tokens is not None and result.output_tokens >= max_tokens else
        "stop")
    return {
        "id": result.message_id.replace("msg_", "chatcmpl_", 1),
        "object": "chat.completion",
        "created": int(time.time()),
        "model": result.model,
        "choices": [{"index": 0, "message": message,
                     "logprobs": None, "finish_reason": finish_reason}],
        "usage": _usage(result.input_tokens, result.output_tokens, result.stats),
        "ljqinfer": result.stats,
    }


class StreamAdapter:
    """Stateful Anthropic-event to OpenAI ChatCompletionChunk adapter."""

    def __init__(self, *, include_usage: bool = False,
                 max_tokens: Optional[int] = None) -> None:
        self.include_usage = include_usage
        self.max_tokens = max_tokens
        self.id = f"chatcmpl_{uuid.uuid4().hex[:24]}"
        self.model = ""
        self.created = int(time.time())
        self.block_types: Dict[int, str] = {}
        self.tool_indexes: Dict[int, int] = {}
        self.input_tokens = 0
        self.output_tokens = 0
        self.stats: Dict[str, Any] = {}
        self.started = False

    def _chunk(self, delta: Dict[str, Any], finish_reason: Optional[str] = None,
               usage: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "id": self.id,
            "object": "chat.completion.chunk",
            "created": self.created,
            "model": self.model,
            "choices": [{"index": 0, "delta": delta,
                         "logprobs": None, "finish_reason": finish_reason}],
        }
        if usage is not None:
            payload["usage"] = usage
        return payload

    def feed(self, event: Dict[str, Any]) -> List[Dict[str, Any]]:
        kind = event.get("type")
        if kind == "message_start":
            message = event.get("message") or {}
            self.id = str(message.get("id") or self.id).replace(
                "msg_", "chatcmpl_", 1)
            self.model = message.get("model") or self.model
            self.input_tokens = int((message.get("usage") or {}).get(
                "input_tokens", 0) or 0)
            self.stats.update(event.get("ljqinfer") or {})
            self.started = True
            return [self._chunk({"role": "assistant", "content": ""})]
        if kind == "content_block_start":
            index = int(event.get("index", 0))
            block = event.get("content_block") or {}
            block_kind = block.get("type")
            self.block_types[index] = block_kind
            if block_kind == "tool_use":
                tool_index = len(self.tool_indexes)
                self.tool_indexes[index] = tool_index
                return [self._chunk({"tool_calls": [{
                    "index": tool_index,
                    "id": block.get("id"),
                    "type": "function",
                    "function": {"name": block.get("name", ""),
                                 "arguments": ""},
                }]})]
            return []
        if kind == "content_block_delta":
            index = int(event.get("index", 0))
            delta = event.get("delta") or {}
            delta_kind = delta.get("type")
            if delta_kind == "text_delta":
                return [self._chunk({"content": delta.get("text", "")})]
            if delta_kind == "thinking_delta":
                return [self._chunk({"reasoning_content":
                                    delta.get("thinking", "")})]
            if delta_kind == "input_json_delta":
                return [self._chunk({"tool_calls": [{
                    "index": self.tool_indexes.get(index, index),
                    "function": {"arguments": delta.get("partial_json", "")},
                }]})]
            return []
        if kind == "message_delta":
            delta = event.get("delta") or {}
            stop_reason = delta.get("stop_reason")
            usage = event.get("usage") or {}
            self.input_tokens = int(usage.get("input_tokens", self.input_tokens) or 0)
            self.output_tokens = int(usage.get("output_tokens", 0) or 0)
            self.stats.update(event.get("ljqinfer") or {})
            finish = (
                "tool_calls" if stop_reason == "tool_use" else
                "length" if self.max_tokens is not None and
                self.output_tokens >= self.max_tokens else
                "stop")
            return [self._chunk({}, finish)]
        if kind == "message_stop":
            if self.include_usage:
                payload = self._chunk({}, None,
                    _usage(self.input_tokens, self.output_tokens, self.stats))
                payload["choices"] = []
                return [payload]
            return []
        if kind == "error":
            error = event.get("error") or {}
            return [{"error": {"message": error.get("message", "generation failed"),
                               "type": error.get("type", "server_error"),
                               "param": None, "code": None}}]
        return []
