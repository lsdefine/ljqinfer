"""Service layer for the ljqinfer stack.

Strict layering: talks to the strategy layer through exactly one entry point,
``strategy.query(input_ids, max_new_tokens) -> Queue``.  Never imports torch,
the model layer, or any kernel module.

Stages: intake -> template -> tokenize -> decode/parse.
"""

from __future__ import annotations

import json
import re
import os
import time
import uuid
from dataclasses import dataclass, field
from queue import Queue
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple

from server.encoding_dsv41 import (dsml_token, encode_messages, eos_token,
                                  tool_calls_block_name,
                                  parse_tool_calls as official_parse_tool_calls)
from tokenizers import Tokenizer

__all__ = ["ServiceLayer", "ServiceError", "GenerationResult", "SurfaceParser"]

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_TOKENIZER = "/mnt/data/kw/models/DeepSeek-V4.1-Flash/tokenizer.json"
MODEL_NAME = "ljqinfer-dsv41f"

# V4.1 official generate.py terminates on eos_token_id only.
# Agentic workloads (long reasoning + tool loops) routinely need more than a
# few hundred tokens; 1024 silently truncated them. Callers that want a short
# answer still pass max_tokens explicitly.
DEFAULT_MAX_TOKENS = 16384

# Official V4.1 effort (model encoding.py): low=50, high=75, max=100.
# There is no "xhigh" level; legacy aliases collapse onto "max".
DEFAULT_REASONING_EFFORT = "high"
_EFFORT_ALIASES = {
    "minimal": "low", "low": "low",
    "medium": "high", "high": "high",
    "max": "max", "maximum": "max", "xhigh": "max", "very_high": "max",
    "veryhigh": "max", "ultra": "max",
}
_EFFORT_OFF = {"none", "off", "disabled", "no"}
# Anthropic sends a thinking token budget instead of an effort label; map it
# onto the three template levels.
LOW_EFFORT_BUDGET = 2048     # budget <= this -> low
HIGH_EFFORT_BUDGET = 8192    # budget <= this -> high, above -> max




class ServiceError(Exception):
    """Malformed request or failed generation."""


@dataclass
class GenerationResult:
    message_id: str
    model: str
    content: List[Dict[str, Any]] = field(default_factory=list)
    stop_reason: str = "end_turn"
    input_tokens: int = 0
    output_tokens: int = 0
    stats: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Anthropic ``message`` object as returned by /v1/messages."""
        return {
            "id": self.message_id,
            "type": "message",
            "role": "assistant",
            "model": self.model,
            "content": self.content,
            "stop_reason": self.stop_reason,
            "stop_sequence": None,
            "usage": {"input_tokens": self.input_tokens,
                      "output_tokens": self.output_tokens},
            "ljqinfer": self.stats,
        }


_DSML_START = "<" + dsml_token + tool_calls_block_name
_DSML_END = "</" + dsml_token + tool_calls_block_name + ">"
_TAGS = ("</think>", "<think>", _DSML_START, _DSML_END)
_MAX_TAG = max(len(t) for t in _TAGS)


def parse_tool_calls(text: str) -> List[Dict[str, Any]]:
    """Decode a raw DSML ``tool_calls`` block into Anthropic ``tool_use``."""
    if not text:
        return []
    index = text.index(_DSML_START) + len(_DSML_START)
    _, _, raw_calls = official_parse_tool_calls(index, text)
    return [{"type": "tool_use",
             "id": "toolu_" + uuid.uuid4().hex[:24],
             "name": call["name"],
             "input": json.loads(call["arguments"])}
            for call in raw_calls]


class SurfaceParser:
    """Split the raw model stream into thinking / text / tool-call segments.

    The prompt ends with an opening think tag, so generation starts inside the
    reasoning block.  Text is released only once it can no longer be the start
    of a control tag, which keeps deltas free of markup.
    """

    THINKING = "thinking"
    TEXT = "text"
    TOOL = "tool"

    def __init__(self, thinking: bool = True) -> None:
        self.state = self.THINKING if thinking else self.TEXT
        self._buf = ""
        self.tool_source = ""

    def feed(self, chunk: str) -> List[Tuple[str, str]]:
        self._buf += chunk
        return self._drain(final=False)

    def flush(self) -> List[Tuple[str, str]]:
        return self._drain(final=True)

    def _drain(self, final: bool) -> List[Tuple[str, str]]:
        out: List[Tuple[str, str]] = []
        while True:
            if self.state == self.THINKING:
                head, sep, tail = self._buf.partition("</think>")
                if sep:
                    if head:
                        out.append((self.THINKING, head))
                    self._buf, self.state = tail, self.TEXT
                    continue
                safe = self._safe_split(final)
                if safe:
                    out.append((self.THINKING, safe))
                return out

            if self.state == self.TEXT:
                pos = self._buf.find(_DSML_START)
                if pos != -1:
                    if pos:
                        out.append((self.TEXT, self._buf[:pos]))
                    self.tool_source += _DSML_START
                    self._buf = self._buf[pos + len(_DSML_START):]
                    self.state = self.TOOL
                    continue
                safe = self._safe_split(final)
                if safe:
                    out.append((self.TEXT, safe))
                return out

            # TOOL: accumulate verbatim, never stream to the client.
            head, sep, tail = self._buf.partition(_DSML_END)
            if sep:
                self.tool_source += head + _DSML_END
                self._buf, self.state = tail, self.TEXT
                continue
            if final:
                self.tool_source += self._buf
                self._buf = ""
            return out

    def _safe_split(self, final: bool) -> str:
        """Emit buffered text minus a possible partial tag at the tail."""
        if final:
            text, self._buf = self._buf, ""
            return text
        keep = 0
        for size in range(min(_MAX_TAG, len(self._buf)), 0, -1):
            tail = self._buf[-size:]
            if any(tag.startswith(tail) for tag in _TAGS):
                keep = size
                break
        if keep == 0:
            text, self._buf = self._buf, ""
            return text
        text, self._buf = self._buf[:-keep], self._buf[-keep:]
        return text


class ServiceLayer:
    """Turns Anthropic-style requests into strategy-layer generations."""

    def __init__(self, strategy: Any,
                 tokenizer_path: str = DEFAULT_TOKENIZER,
                 model_name: str = MODEL_NAME,
                 max_output_tokens: int = 16384,
                 max_total_tokens: int = 1024 * 1024) -> None:
        if strategy is None or not hasattr(strategy, "query"):
            raise ServiceError("strategy must expose .query(input_ids, ...)")
        self._strategy = strategy
        self.model_name = model_name
        self.max_output_tokens = int(max_output_tokens)
        self.max_total_tokens = int(max_total_tokens)
        if self.max_total_tokens <= 0:
            raise ServiceError("max_total_tokens must be positive")
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        eos_id = self.tokenizer.token_to_id(eos_token)
        if eos_id is None:
            raise ServiceError('tokenizer lacks V4.1 EOS token')
        self.stop_ids = {eos_id}

    # -- stage 1: intake ---------------------------------------------------

    @staticmethod
    def _flatten(content: Any) -> str:
        """Anthropic content blocks -> plain text."""
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        parts: List[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
        return "".join(parts)

    @staticmethod
    def _normalise_tools(tools: Any) -> Optional[List[Dict[str, Any]]]:
        """Convert Anthropic/OpenAI declarations to official OpenAI tool shape."""
        if not tools:
            return None
        out: List[Dict[str, Any]] = []
        for raw in tools:
            if not isinstance(raw, dict):
                continue
            tool = raw.get("function", raw)
            function = {"name": tool.get("name"),
                        "description": tool.get("description", "")}
            schema = tool.get("input_schema") or tool.get("parameters")
            if schema:
                function["parameters"] = schema
            out.append({"type": "function", "function": function})
        return out or None

    @staticmethod
    def _reasoning(message: Dict[str, Any]) -> str:
        """Recover an assistant turn's reasoning text for the template.

        Accepts Anthropic ``thinking``/``redacted_thinking`` content blocks as
        well as the OpenAI-style ``reasoning_content`` field.  Inline
        ``<think>..</think>`` text is left alone: the template splits it.
        """
        explicit = message.get("reasoning_content")
        if isinstance(explicit, str) and explicit.strip():
            return explicit
        content = message.get("content")
        if not isinstance(content, list):
            return ""
        parts: List[str] = []
        for item in content:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "thinking":
                parts.append(item.get("thinking") or "")
            elif item.get("type") == "redacted_thinking":
                parts.append(item.get("data") or "")
        return "".join(parts)

    @staticmethod
    def _thinking_options(request: Dict[str, Any],
                          max_tokens: int) -> Dict[str, Any]:
        """Resolve every thinking-related knob the chat template exposes.

        Supported spellings, in precedence order:
          * ``thinking`` = ``{"type": "enabled"|"disabled", "budget_tokens": N,
            "effort": ..., "clear_thinking": bool}`` (Anthropic)
          * ``thinking`` = bool (convenience)
          * ``reasoning_effort`` / ``reasoning: {"effort": ...}`` (OpenAI)
          * ``clear_thinking`` at the top level
        """
        thinking = request.get("thinking")
        # Preserve the original service default: explicit opt-in thinking.
        enable = False
        explicitly_disabled = False
        effort: Any = None
        budget: Optional[int] = None
        clear_thinking: Optional[bool] = None

        if isinstance(thinking, bool):
            enable = thinking
            explicitly_disabled = not thinking
        elif isinstance(thinking, dict):
            kind = thinking.get("type")
            if kind is not None:
                if kind not in ("enabled", "disabled"):
                    raise ServiceError(
                        "'thinking.type' must be 'enabled' or 'disabled'")
                enable = kind == "enabled"
                explicitly_disabled = kind == "disabled"
            if thinking.get("budget_tokens") is not None:
                try:
                    budget = int(thinking["budget_tokens"])
                except (TypeError, ValueError):
                    raise ServiceError(
                        "'thinking.budget_tokens' must be an integer")
                if budget < 0:
                    raise ServiceError(
                        "'thinking.budget_tokens' must not be negative")
                if budget == 0:
                    enable = False
                    explicitly_disabled = True
                elif not explicitly_disabled:
                    enable = True
            effort = thinking.get("effort", thinking.get("reasoning_effort"))
            if thinking.get("clear_thinking") is not None:
                clear_thinking = bool(thinking["clear_thinking"])
            elif thinking.get("clear_history") is not None:
                clear_thinking = bool(thinking["clear_history"])
        elif thinking is not None:
            raise ServiceError("'thinking' must be an object or a boolean")

        if effort is None:
            effort = request.get("reasoning_effort")
        if effort is None and isinstance(request.get("reasoning"), dict):
            effort = request["reasoning"].get("effort")

        resolved = DEFAULT_REASONING_EFFORT
        if effort is not None:
            if type(effort) is int:
                if not 1 <= effort <= 100:
                    raise ServiceError("'reasoning_effort' must be within 1..100")
                resolved = effort
                if not explicitly_disabled:
                    enable = True
                key = None
            elif isinstance(effort, str):
                key = effort.strip().lower()
            else:
                raise ServiceError("'reasoning_effort' must be a label or integer 1..100")
            if key is None:
                pass
            elif key in _EFFORT_OFF:
                enable = False
                explicitly_disabled = True
            elif key in _EFFORT_ALIASES:
                resolved = _EFFORT_ALIASES[key]
                if not explicitly_disabled:
                    enable = True
            else:
                raise ServiceError(f"unsupported reasoning effort: {effort!r}")
        elif budget is not None and budget > 0:
            resolved = ("low" if budget <= LOW_EFFORT_BUDGET
                        else "high" if budget <= HIGH_EFFORT_BUDGET else "max")

        if clear_thinking is None and request.get("clear_thinking") is not None:
            clear_thinking = bool(request["clear_thinking"])
        if clear_thinking is None:
            clear_thinking = True          # template default: keep only the
                                           # current turn's reasoning

        if enable and budget is not None and budget > 0 and budget >= max_tokens:
            raise ServiceError(
                f"'thinking.budget_tokens' ({budget}) must be smaller than "
                f"'max_tokens' ({max_tokens})")

        return {"enable_thinking": enable,
                "reasoning_effort": resolved if enable else None,
                "clear_thinking": clear_thinking,
                "thinking_budget_tokens": budget if enable else None}

    def intake(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and normalise a ``/v1/messages`` payload."""
        if not isinstance(request, dict):
            raise ServiceError("request body must be a JSON object")
        raw_messages = request.get("messages")
        if not isinstance(raw_messages, list) or not raw_messages:
            raise ServiceError("'messages' must be a non-empty array")

        messages: List[Dict[str, Any]] = []
        system = request.get("system")
        if system:
            messages.append({"role": "system",
                             "content": self._flatten(system)})

        for entry in raw_messages:
            if not isinstance(entry, dict):
                raise ServiceError("each message must be an object")
            role, content = entry.get("role"), entry.get("content")
            if role not in ("user", "assistant"):
                raise ServiceError(f"unsupported message role: {role!r}")

            if isinstance(content, list) and any(isinstance(b, dict) and
                    b.get('type') in ('image', 'image_url') for b in content):
                if role != 'user' or any(not isinstance(b, dict) or b.get('type') not in
                                        ('text', 'image', 'image_url') for b in content):
                    raise ServiceError('images require user text/image blocks')
                messages.append({'role': role, 'content': content})
                continue

            # A user turn carrying tool_result blocks is really a tool turn.
            if role == "user" and isinstance(content, list):
                results = [b for b in content if isinstance(b, dict)
                           and b.get("type") == "tool_result"]
                if results:
                    for block in results:
                        messages.append({
                            "role": "tool",
                            "tool_call_id": block.get("tool_use_id", ""),
                            "content": self._flatten(block.get("content"))})
                    leftover = self._flatten(content)
                    if leftover.strip():
                        messages.append({"role": "user", "content": leftover})
                    continue

            if role == "assistant":
                turn: Dict[str, Any] = {"role": "assistant",
                                        "content": self._flatten(content)}
                # Keep the turn's own chain of thought: the template replays it
                # verbatim for the current turn (and for every turn when
                # clear_thinking is false), which is what tool-call loops need.
                reasoning = self._reasoning(entry)
                if reasoning.strip():
                    turn["reasoning_content"] = reasoning
                if isinstance(content, list):
                    uses = [b for b in content if isinstance(b, dict)
                            and b.get("type") == "tool_use"]
                    if uses:
                        turn["tool_calls"] = [
                            {"id": u.get("id", ""),
                             "type": "function",
                             "function": {
                                 "name": u.get("name"),
                                 "arguments": json.dumps(
                                     u.get("input") or {}, ensure_ascii=False)}}
                            for u in uses]
                messages.append(turn)
                continue

            messages.append({"role": role, "content": self._flatten(content)})

        try:
            max_tokens = request.get("max_tokens")
            max_tokens = DEFAULT_MAX_TOKENS if max_tokens is None else max_tokens
            if isinstance(max_tokens, bool) or not isinstance(max_tokens, int):
                raise ValueError("not an integer")
        except (TypeError, ValueError):
            raise ServiceError("'max_tokens' must be an integer")
        if max_tokens <= 0:
            raise ServiceError("'max_tokens' must be positive")

        thinking = self._thinking_options(request, max_tokens)

        temperature = request.get("temperature")
        try:
            temperature = 1.0 if temperature is None else float(temperature)
        except (TypeError, ValueError):
            raise ServiceError("'temperature' must be a number")
        if temperature < 0.0:
            raise ServiceError("'temperature' must be non-negative")

        plan = {"messages": messages,
                "tools": self._normalise_tools(request.get("tools")),
                "max_tokens": min(max_tokens, self.max_output_tokens),
                "temperature": temperature,
                "model": request.get("model") or self.model_name}
        plan.update(thinking)
        return plan

    # -- stage 2 + 3: template and tokenizer -------------------------------

    def render(self, plan: Dict[str, Any]) -> str:
        """Render via the official DeepSeek-V4.1 encoder (no jinja template)."""
        messages = plan["messages"]
        if plan["tools"]:
            # The official encoder only renders tools off a leading
            # system/developer message; on a user message they are dropped.
            if messages[0]["role"] not in ("system", "developer"):
                messages = [{"role": "system", "content": ""}] + messages
            messages = [dict(messages[0], tools=plan["tools"])] + messages[1:]
        rendered = encode_messages(
            messages,
            thinking_mode="thinking" if plan["enable_thinking"] else "chat",
            drop_thinking=plan.get("clear_thinking", True),
            reasoning_effort=plan.get("reasoning_effort"),
            return_multi_modal_data=True)
        if isinstance(rendered, tuple):
            prompt, images = rendered
            plan["_images"] = images["images"]
            return prompt
        return rendered

    def encode(self, prompt: str) -> List[int]:
        return self.tokenizer.encode(prompt, add_special_tokens=False).ids

    def build(self, request: Dict[str, Any]) -> Tuple[List[int], Dict[str, Any]]:
        plan = self.intake(request)
        input_ids = self.encode(self.render(plan))
        if plan.get("_images") or 129264 in input_ids:
            from server.image_input import expand_images
            try:
                input_ids, payload = expand_images(input_ids, plan.pop("_images", []))
            except (ValueError, TypeError, OSError) as exc:
                raise ServiceError(str(exc)) from exc
            if payload is not None:
                plan["image_payload"] = payload
        total_tokens = len(input_ids) + plan["max_tokens"]
        if total_tokens > self.max_total_tokens:
            raise ServiceError(
                f"input tokens ({len(input_ids)}) + max output tokens "
                f"({plan['max_tokens']}) = {total_tokens} exceeds "
                f"--max-total-tokens ({self.max_total_tokens})")
        return input_ids, plan

    # -- stage 4: strategy hand-off ----------------------------------------

    def _consume(self, queue: Queue,
                 counter: Optional[List[int]] = None,
                 ) -> Iterator[Tuple[List[int], bool]]:
        """Consume explicit strategy dict events; prefill is metadata only."""
        while True:
            event = queue.get()
            kind = event.get("type")
            if kind == "error":
                raise ServiceError(f"generation failed: {event.get('error')}")
            if kind == "prefill":
                continue
            if kind == "token":
                ids = event.get("token_ids") or []
                kept: List[int] = []
                for token in ids:
                    if token in self.stop_ids:
                        if counter is not None:
                            counter[0] += len(kept) + 1
                        handle = getattr(queue, "cancel_handle", None)
                        if handle is not None:
                            handle.stop_at_semantic_eos()
                        yield kept, True
                        while True:
                            terminal = queue.get()
                            terminal_kind = terminal.get("type")
                            if terminal_kind == "error":
                                raise ServiceError(
                                    f"generation failed: {terminal.get('error')}")
                            if terminal_kind == "end":
                                queue.stop_reason = terminal.get("reason")
                                return
                    kept.append(token)
                if counter is not None:
                    counter[0] += len(kept)
                if kept:
                    yield kept, False
                continue
            if kind == "end":
                queue.stop_reason = event.get("reason")
                yield [], True
                return
            raise ServiceError(f"unknown strategy event type: {kind!r}")

    def _decode_stream(self, queue: Queue,
                       counter: Optional[List[int]] = None,
                       ) -> Iterator[str]:
        """Incremental text, safe against multi-byte pieces being split."""
        pending: List[int] = []
        emitted = ""
        for ids, stopped in self._consume(queue, counter):
            if ids:
                pending.extend(ids)
                text = self.tokenizer.decode(pending, skip_special_tokens=False)
                if text.endswith("\ufffd"):
                    continue          # partial UTF-8, wait for more tokens
                if len(text) > len(emitted):
                    yield text[len(emitted):]
                    emitted = text
            if stopped:
                text = self.tokenizer.decode(pending, skip_special_tokens=False)
                if len(text) > len(emitted):
                    yield text[len(emitted):]
                # Resume _consume once more so it can drain the strategy's
                # terminal event.  Returning here would close the generator
                # while the handle is still running and misclassify a normal
                # EOS/tool completion as a client cancellation.
                continue

    # -- stage 5: observability -------------------------------------------

    @staticmethod
    def _response_stop_reason(queue: Queue,
                              calls: List[Dict[str, Any]]) -> str:
        """Map the engine terminal reason to the Anthropic response contract."""
        if calls:
            return "tool_use"
        if getattr(queue, "stop_reason", None) in ("max_tokens", "length"):
            return "max_tokens"
        return "end_turn"

    @staticmethod
    def _stats(queue: Queue, submitted_at: float, first_token_at: Optional[float],
               finished_at: float, output_tokens: int) -> Dict[str, Any]:
        """Flatten strategy QueryMetrics plus service-side wall clock timings.

        The service layer stays ignorant of how the numbers were produced: it
        only copies whatever public fields the strategy chose to publish.
        """
        stats: Dict[str, Any] = {}
        metrics = getattr(queue, "metrics", None)
        for name in ("input_tokens", "cache_hit_tokens", "cache_hit_rate",
                     "prefill_tokens",
                     "cache_stored_blocks", "cache_evicted_blocks",
                     "cache_lookup_seconds", "cache_load_seconds",
                     "model_prefill_seconds", "cache_store_seconds",
                     "finish_prefill_seconds", "prefill_compute_seconds",
                     "submit_to_prefill_seconds", "strategy_seconds",
                     "effective_prefill_tps", "model_prefill_tps",
                     "chunks", "first_chunk_seconds", "steady_prefill_tps",
                     "chunk_seconds", "chunk_tokens"):
            value = getattr(metrics, name, None)
            if value is not None:
                stats[name] = round(value, 6) if isinstance(value, float) else value
        decode_seconds = (finished_at - first_token_at) if first_token_at else 0.0
        stats.update({
            "output_tokens": output_tokens,
            "queue_wait_seconds": round(
                (first_token_at or finished_at) - submitted_at, 6),
            "time_to_first_token_seconds": round(
                (first_token_at or finished_at) - submitted_at, 6),
            "decode_seconds": round(decode_seconds, 6),
            "total_seconds": round(finished_at - submitted_at, 6),
            "decode_tps": round(output_tokens / decode_seconds, 3)
            if decode_seconds > 0 else 0.0})
        decode_stats = getattr(queue, "decode_stats", None)
        if decode_stats:
            stats.update(decode_stats)
            steps = int(decode_stats.get("decode_steps") or 0)
            if steps > 0:
                stats["mtp_accepted_per_step"] = round(
                    decode_stats.get("accepted_tokens", 0) / steps, 3)
                stats["decode_tokens_per_step"] = round(
                    output_tokens / steps, 3)
        return stats

    # -- public API --------------------------------------------------------

    def generate(self, request: Dict[str, Any]) -> GenerationResult:
        """Non-streaming ``/v1/messages`` generation."""
        input_ids, plan = self.build(request)
        submitted_at = time.perf_counter()
        queue = self._strategy.query(input_ids, plan["max_tokens"],
                                     plan["temperature"],
                                     **({"image_payload": plan["image_payload"]} if "image_payload" in plan else {}))
        print(f"[service] blocking id={queue.request_id} input={len(input_ids)} "
              f"max_new={plan['max_tokens']}", flush=True)

        parser = SurfaceParser(plan["enable_thinking"])
        thinking: List[str] = []
        text: List[str] = []
        counter = [0]
        first_token_at: Optional[float] = None
        for chunk in self._decode_stream(queue, counter):
            if first_token_at is None:
                first_token_at = time.perf_counter()
            for kind, piece in parser.feed(chunk):
                (thinking if kind == SurfaceParser.THINKING else text).append(piece)
        for kind, piece in parser.flush():
            (thinking if kind == SurfaceParser.THINKING else text).append(piece)

        message_id = f"msg_{uuid.uuid4().hex[:24]}"
        content: List[Dict[str, Any]] = []
        joined_think = "".join(thinking).strip()
        if joined_think:
            content.append({"type": "thinking", "thinking": joined_think,
                            "signature": uuid.uuid4().hex[:24]})
        joined_text = "".join(text).strip()
        if joined_text:
            content.append({"type": "text", "text": joined_text})

        calls = parse_tool_calls(parser.tool_source)
        content.extend(calls)

        stats = self._stats(queue, submitted_at, first_token_at,
                            time.perf_counter(), counter[0])
        stats["reasoning_effort"] = plan.get("reasoning_effort") or "off"
        print(f"[ljqinfer] id={queue.request_id} "
              f"metrics={json.dumps(stats, ensure_ascii=False, sort_keys=True)}",
              flush=True)
        return GenerationResult(
            message_id=message_id,
            model=plan["model"],
            content=content or [{"type": "text", "text": ""}],
            stop_reason=self._response_stop_reason(queue, calls),
            input_tokens=len(input_ids),
            output_tokens=counter[0],
            stats=stats)

    def stream(self, request: Dict[str, Any], *,
               on_submit: Optional[Callable[[Any], None]] = None,
               ) -> Iterator[Dict[str, Any]]:
        """Streaming generation as Anthropic SSE events (dicts, not wire text).

        Thinking and text are streamed live; tool_use blocks are only known
        once complete, so they are emitted at the end.  ``on_submit`` exposes
        only the cancellation handle to the HTTP transport, allowing a queued
        request to be abandoned while this generator is blocked on its queue.
        """
        input_ids, plan = self.build(request)
        message_id = f"msg_{uuid.uuid4().hex[:24]}"

        submitted_at = time.perf_counter()
        queue = self._strategy.query(input_ids, plan["max_tokens"],
                                     plan["temperature"],
                                     **({"image_payload": plan["image_payload"]} if "image_payload" in plan else {}))
        if on_submit is not None:
            on_submit(getattr(queue, "cancel_handle", None))
        print(f"[service] stream id={queue.request_id} input={len(input_ids)} "
              f"max_new={plan['max_tokens']}", flush=True)

        # message_start is driven by the explicit prefill queue event.
        prefill_event = queue.get()
        prefill_kind = prefill_event.get("type")
        if prefill_kind == "error":
            raise ServiceError(
                f"generation failed: {prefill_event.get('error')}")
        if prefill_kind != "prefill":
            raise ServiceError(
                f"expected prefill event, got {prefill_kind!r}")
        start_stats = dict(prefill_event.get("metrics") or {})
        yield {"type": "message_start",
               "message": {"id": message_id, "type": "message",
                           "role": "assistant", "model": plan["model"],
                           "content": [],
                           "usage": {"input_tokens": len(input_ids),
                                     "output_tokens": 0}},
               "ljqinfer": start_stats}
        parser = SurfaceParser(plan["enable_thinking"])
        counter = [0]
        first_token_at: Optional[float] = None
        index = -1
        open_kind: Optional[str] = None

        def _close() -> Iterator[Dict[str, Any]]:
            if open_kind is not None:
                yield {"type": "content_block_stop", "index": index}

        def _emit(kind: str, piece: str) -> Iterator[Dict[str, Any]]:
            nonlocal index, open_kind
            if kind != open_kind:
                yield from _close()
                index += 1
                block = ({"type": "thinking", "thinking": "",
                          "signature": uuid.uuid4().hex[:24]}
                         if kind == SurfaceParser.THINKING
                         else {"type": "text", "text": ""})
                yield {"type": "content_block_start",
                       "index": index, "content_block": block}
                open_kind = kind
            delta = ({"type": "thinking_delta", "thinking": piece}
                     if kind == SurfaceParser.THINKING
                     else {"type": "text_delta", "text": piece})
            yield {"type": "content_block_delta",
                   "index": index, "delta": delta}

        try:
            for chunk in self._decode_stream(queue, counter):
                if first_token_at is None:
                    first_token_at = time.perf_counter()
                for kind, piece in parser.feed(chunk):
                    yield from _emit(kind, piece)
            for kind, piece in parser.flush():
                yield from _emit(kind, piece)
        except ServiceError as exc:
            yield {"type": "error",
                   "error": {"type": "api_error", "message": str(exc)}}
            return
        finally:
            handle = getattr(queue, "cancel_handle", None)
            if handle is not None and handle.state not in ("done", "cancelled"):
                print(f"[service] generator-close id={handle.request_id} "
                      f"state={handle.state}; cancelling", flush=True)
                handle.cancel()

        yield from _close()
        open_kind = None

        calls = parse_tool_calls(parser.tool_source)
        for call in calls:
            index += 1
            yield {"type": "content_block_start", "index": index,
                   "content_block": {"type": "tool_use",
                                     "id": call["id"],
                                     "name": call["name"], "input": {}}}
            yield {"type": "content_block_delta", "index": index,
                   "delta": {"type": "input_json_delta",
                             "partial_json": json.dumps(call["input"],
                                                        ensure_ascii=False)}}
            yield {"type": "content_block_stop", "index": index}

        stats = self._stats(queue, submitted_at, first_token_at,
                            time.perf_counter(), counter[0])
        stats["reasoning_effort"] = plan.get("reasoning_effort") or "off"
        print(f"[ljqinfer] id={queue.request_id} "
              f"metrics={json.dumps(stats, ensure_ascii=False, sort_keys=True)}",
              flush=True)
        yield {"type": "message_delta",
               "delta": {"stop_reason": self._response_stop_reason(queue, calls),
                         "stop_sequence": None},
               "usage": {"input_tokens": len(input_ids),
                         "output_tokens": counter[0]},
               "ljqinfer": stats}
        yield {"type": "message_stop"}