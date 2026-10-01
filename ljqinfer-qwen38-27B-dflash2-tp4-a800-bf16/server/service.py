"""Protocol-independent Qwen chat tokenization and result decoding.

Only declarative model configuration is imported; no torch or execution implementation.
"""
from __future__ import annotations

import json
import re
import time
import uuid
from pathlib import Path
from queue import Queue
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

from jinja2 import Environment, StrictUndefined
from tokenizers import Tokenizer

from strategy.remote_strategy import RemoteStrategy

from model.config import MODEL_DIR
STOP_IDS = {248044, 248046}
_TOOL_CALL_RE = re.compile(
    r"<tool_call>\s*<function=([^>\n]+)>\s*(.*?)</function>\s*</tool_call>",
    re.DOTALL)
_TOOL_PARAMETER_RE = re.compile(
    r"<parameter=([^>\n]+)>\s*(.*?)\s*</parameter>", re.DOTALL)
_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"
_MAX_TAG = max(len(_THINK_OPEN), len(_THINK_CLOSE), len("<tool_call>"))
_REASONING_OFF = {"none", "off", "disabled", "no", "false"}
_REASONING_ON = {"on", "enabled", "yes", "true"}
_REASONING_EFFORTS = {
    "minimal": "low", "low": "low", "medium": "medium",
    "high": "xhigh", "max": "xhigh", "maximum": "xhigh",
    "xhigh": "xhigh", "very_high": "xhigh", "veryhigh": "xhigh",
    "ultra": "xhigh",
}


class ServiceError(Exception):
    """Malformed request or failed generation."""


def _raise_template_error(message):
    raise ValueError(str(message))


def _argument_value(source: str):
    source = source.strip()
    try:
        return json.loads(source)
    except (TypeError, ValueError):
        return source


def split_reasoning(text: str):
    """Split Qwen thinking from user-visible assistant content."""
    marker = "</think>"
    if marker not in text:
        return "", text.strip()
    reasoning, content = text.rsplit(marker, 1)
    if reasoning.lstrip().startswith("<think>"):
        reasoning = reasoning.lstrip()[len("<think>"):]
    return reasoning.strip(), content.strip()


def parse_tool_calls(text: str) -> tuple[str, list[dict]]:
    """Parse Qwen 3.5/3.8 XML tool calls into OpenAI response objects."""
    calls = []
    for match in _TOOL_CALL_RE.finditer(text):
        arguments = {}
        for parameter in _TOOL_PARAMETER_RE.finditer(match.group(2)):
            arguments[parameter.group(1).strip()] = _argument_value(parameter.group(2))
        calls.append({
            "id": "call_" + uuid.uuid4().hex[:24],
            "type": "function",
            "function": {
                "name": match.group(1).strip(),
                "arguments": json.dumps(arguments, ensure_ascii=False,
                                        separators=(",", ":")),
            },
        })
    content = _TOOL_CALL_RE.sub("", text).strip()
    return content, calls


class SurfaceParser:
    """Incremental Qwen surface parser for thinking/text/tool regions."""

    THINKING = "thinking"
    TEXT = "text"
    TOOL = "tool"

    def __init__(self, thinking: bool = True) -> None:
        # Qwen's chat template ends the prompt with an opening <think> tag.
        # Generated tokens therefore begin *inside* reasoning, exactly as in
        # the Ref service and vLLM's qwen3 reasoning parser.
        self.state = self.THINKING if thinking else self.TEXT
        self._buf = ""
        self.tool_source = ""
        self.seen_think_open = thinking

    def feed(self, chunk: str):
        self._buf += chunk
        return self._drain(final=False)

    def flush(self):
        return self._drain(final=True)

    def _drain(self, final: bool):
        out = []
        while True:
            if self.state == self.TEXT:
                # Detect optional <think> at the very beginning of generation.
                if not self.seen_think_open and self._buf.lstrip().startswith(_THINK_OPEN):
                    stripped = self._buf.lstrip()
                    self._buf = stripped[len(_THINK_OPEN):]
                    self.seen_think_open = True
                    self.state = self.THINKING
                    continue
                head, sep, tail = self._buf.partition("<tool_call>")
                if sep:
                    if head:
                        out.append((self.TEXT, head))
                    self.tool_source += "<tool_call>"
                    self._buf, self.state = tail, self.TOOL
                    continue
                if self._buf.startswith(_THINK_OPEN) or (
                        not self.seen_think_open and _THINK_OPEN.startswith(self._buf) and self._buf):
                    # partial tag; wait
                    if final and self._buf:
                        out.append((self.TEXT, self._buf))
                        self._buf = ""
                    return out
                safe = self._safe_split(final)
                if safe:
                    out.append((self.TEXT, safe))
                return out

            if self.state == self.THINKING:
                head, sep, tail = self._buf.partition(_THINK_CLOSE)
                if sep:
                    if head:
                        out.append((self.THINKING, head))
                    self._buf, self.state = tail, self.TEXT
                    continue
                safe = self._safe_split(final, tags=(_THINK_CLOSE,))
                if safe:
                    out.append((self.THINKING, safe))
                return out

            # TOOL: accumulate verbatim, never stream to the client.
            head, sep, tail = self._buf.partition("</tool_call>")
            if sep:
                self.tool_source += head + "</tool_call>"
                self._buf, self.state = tail, self.TEXT
                continue
            if final:
                self.tool_source += self._buf
                self._buf = ""
            return out

    def _safe_split(self, final: bool, tags=None) -> str:
        if final:
            text, self._buf = self._buf, ""
            return text
        tags = tags or (_THINK_OPEN, _THINK_CLOSE, "<tool_call>")
        keep = 0
        for size in range(min(_MAX_TAG, len(self._buf)), 0, -1):
            tail = self._buf[-size:]
            if any(tag.startswith(tail) for tag in tags):
                keep = size
                break
        if keep == 0:
            text, self._buf = self._buf, ""
            return text
        text, self._buf = self._buf[:-keep], self._buf[-keep:]
        return text


class ServiceLayer:
    def __init__(self, strategy=None):
        self.strategy = strategy or RemoteStrategy()
        self.tokenizer = Tokenizer.from_file(str(MODEL_DIR / "tokenizer.json"))
        config = json.loads((MODEL_DIR / "tokenizer_config.json").read_text())
        env = Environment(
            trim_blocks=True, lstrip_blocks=True, undefined=StrictUndefined)
        env.filters["tojson"] = (
            lambda value, **kwargs: json.dumps(value, ensure_ascii=False, **{
                k: v for k, v in kwargs.items() if k in ("indent",)}))
        env.globals["raise_exception"] = _raise_template_error
        chat_template = config.get("chat_template")
        if not chat_template:
            raise RuntimeError("tokenizer_config.json missing chat_template")
        self.template = env.from_string(chat_template)
        tool_template_path = MODEL_DIR / "tool_chat_template.jinja"
        if tool_template_path.exists():
            self.tool_template = env.from_string(tool_template_path.read_text())
        else:
            self.tool_template = self.template

    def _render_plain_chat(self, messages, *, enable_thinking=True,
                           reasoning_effort=None):
        # The Qwen template uses StrictUndefined and directly reads
        # assistant.tool_calls even for ordinary multi-turn text chats.
        # OpenAI/Anthropic clients normally omit that optional field.
        normalized = []
        for message in messages:
            entry = dict(message)
            if entry.get("role") == "assistant":
                entry.setdefault("tool_calls", [])
            normalized.append(entry)
        options = {
            "messages": normalized, "tools": None,
            "add_generation_prompt": True,
            "enable_thinking": enable_thinking,
        }
        if reasoning_effort is not None:
            options["reasoning_effort"] = reasoning_effort
        return self.template.render(**options)

    def _template_messages(self, messages):
        normalized = []
        for message in messages:
            role = message.get("role")
            if role == "tool":
                normalized.append({
                    "role": "tool",
                    "content": str(message.get("content") or ""),
                    "name": message.get("name"),
                    "tool_call_id": message.get("tool_call_id"),
                })
                continue
            entry = {"role": role, "content": message.get("content")}
            if role == "assistant":
                # Qwen's tool template reads this optional field directly
                # under StrictUndefined, including for ordinary text-only
                # assistant turns replayed in a tools-enabled conversation.
                entry["tool_calls"] = []
            reasoning = message.get("reasoning_content")
            if reasoning:
                entry["reasoning_content"] = str(reasoning)
            if message.get("tool_calls"):
                # OpenAI returns function.arguments as a JSON string, while
                # Qwen's standard tool template iterates it as a mapping.
                # Normalize the replayed assistant turn to the model-native
                # object form used by Ref and vLLM before rendering.
                normalized_calls = []
                for call in message["tool_calls"]:
                    copied = dict(call)
                    fn = dict(copied.get("function") or {})
                    args = fn.get("arguments")
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except Exception as exc:
                            raise ValueError("tool call arguments must be valid JSON") from exc
                    if args is None:
                        args = {}
                    if not isinstance(args, dict):
                        raise ValueError("tool call arguments must decode to an object")
                    fn["arguments"] = args
                    copied["function"] = fn
                    normalized_calls.append(copied)
                entry["tool_calls"] = normalized_calls
            if isinstance(message.get("content"), list):
                entry["content"] = "".join(
                    part.get("text", "") if isinstance(part, dict) else str(part)
                    for part in message["content"])
            normalized.append(entry)
        return normalized

    def _select_tools(self, tools, tool_choice):
        if not tools:
            return None
        if tool_choice == "none":
            return None
        if isinstance(tool_choice, dict):
            name = (tool_choice.get("function") or {}).get("name")
            if name:
                selected = [t for t in tools
                            if (t.get("function") or t).get("name") == name]
                return selected or tools
        return tools

    @staticmethod
    def _thinking_options(reasoning_effort=None, thinking=None):
        """Resolve OpenAI/Anthropic reasoning controls for Qwen's template.

        Omitted controls preserve the historical service default (thinking on).
        An explicit off value wins; otherwise effort aliases are normalized to
        the only values accepted by Qwen3.8's shipped template.
        """
        enable = True
        explicitly_disabled = False
        raw_effort = reasoning_effort

        if isinstance(thinking, bool):
            enable = thinking
            explicitly_disabled = not thinking
        elif isinstance(thinking, dict):
            kind = thinking.get("type")
            if kind is not None:
                value = str(kind).strip().lower()
                if value in _REASONING_OFF:
                    enable = False
                    explicitly_disabled = True
                elif value in _REASONING_ON:
                    enable = True
                else:
                    raise ValueError(
                        "thinking.type must be enabled/on or disabled/off")
            nested_effort = thinking.get("effort") or thinking.get("reasoning_effort")
            if nested_effort is not None:
                raw_effort = nested_effort
        elif thinking is not None:
            # Keep compatibility with clients that use string thinking knobs.
            raw_effort = thinking

        effort = None
        if raw_effort is not None:
            value = str(raw_effort).strip().lower()
            if value in _REASONING_OFF:
                enable = False
                explicitly_disabled = True
            elif value in _REASONING_ON:
                if not explicitly_disabled:
                    enable = True
            elif value in _REASONING_EFFORTS:
                effort = _REASONING_EFFORTS[value]
                if not explicitly_disabled:
                    enable = True
            else:
                raise ValueError(
                    f"unsupported reasoning_effort {raw_effort!r}; expected "
                    "none/off, low, medium, high, or xhigh")
        return enable, effort if enable else None

    def render_chat(self, messages, tools=None, tool_choice=None, *,
                    reasoning_effort=None, thinking=None) -> str:
        if not messages:
            raise ValueError("messages must not be empty")
        enable_thinking, resolved_effort = self._thinking_options(
            reasoning_effort, thinking)
        selected_tools = self._select_tools(tools, tool_choice)
        uses_tool_protocol = bool(selected_tools) or any(
            message.get("role") == "tool" or message.get("tool_calls")
            for message in messages)
        if not uses_tool_protocol:
            return self._render_plain_chat(
                messages, enable_thinking=enable_thinking,
                reasoning_effort=resolved_effort)
        rendered_messages = self._template_messages(messages)
        if tool_choice == "required" or isinstance(tool_choice, dict):
            name = (tool_choice.get("function", {}).get("name")
                    if isinstance(tool_choice, dict) else None)
            instruction = (f"You must call the {name} tool for this request."
                           if name else "You must call one available tool for this request.")
            if rendered_messages and rendered_messages[0].get("role") == "system":
                rendered_messages[0]["content"] = (
                    str(rendered_messages[0].get("content", "")) + "\n\n" + instruction)
            else:
                rendered_messages.insert(0, {"role": "system", "content": instruction})
        options = {
            "messages": rendered_messages, "tools": selected_tools,
            "add_generation_prompt": True,
            "enable_thinking": enable_thinking,
            "preserve_thinking": True, "add_vision_id": False,
        }
        if resolved_effort is not None:
            options["reasoning_effort"] = resolved_effort
        return self.tool_template.render(**options)

    def _consume(self, queue: Queue, counter: Optional[List[int]] = None):
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
                    if token in STOP_IDS:
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
                                return
                    kept.append(token)
                if counter is not None:
                    counter[0] += len(kept)
                if kept:
                    yield kept, False
                continue
            if kind == "end":
                yield [], True
                return
            raise ServiceError(f"unknown strategy event type: {kind!r}")

    def _decode_stream(self, queue: Queue, counter: Optional[List[int]] = None):
        pending: List[int] = []
        emitted = ""
        for ids, stopped in self._consume(queue, counter):
            if ids:
                pending.extend(ids)
                text = self.tokenizer.decode(pending)
                if text.endswith("\ufffd"):
                    continue
                if len(text) > len(emitted):
                    yield text[len(emitted):]
                    emitted = text
            if stopped:
                text = self.tokenizer.decode(pending)
                if len(text) > len(emitted):
                    yield text[len(emitted):]
                continue

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
                     "submit_to_prefill_seconds", "strategy_seconds",
                     "effective_prefill_tps", "model_prefill_tps"):
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

    def complete(self, messages, max_tokens: int = 1024, *, tools=None,
                 tool_choice=None, reasoning_effort=None, thinking=None, temperature=1.0) -> dict:
        prompt = self.render_chat(
            messages, tools, tool_choice,
            reasoning_effort=reasoning_effort, thinking=thinking)
        input_ids = self.tokenizer.encode(prompt).ids
        submitted_at = time.perf_counter()
        queue = self.strategy.query(input_ids, max_tokens, temperature=temperature)
        print(f"[service] blocking id={queue.request_id} input={len(input_ids)} "
              f"max_new={max_tokens}", flush=True)
        counter = [0]
        first_token_at = None
        token_ids: List[int] = []
        for ids, stopped in self._consume(queue, counter):
            if ids and first_token_at is None:
                first_token_at = time.perf_counter()
            token_ids.extend(ids)
            if stopped:
                break
        finished_at = time.perf_counter()
        visible = list(token_ids)
        while visible and visible[-1] in STOP_IDS:
            visible.pop()
        text = self.tokenizer.decode(visible)
        reasoning, visible_text = split_reasoning(text)
        content, tool_calls = parse_tool_calls(visible_text)
        metrics = self._stats(queue, submitted_at, first_token_at, finished_at, counter[0])
        print(f"[service] done id={queue.request_id} metrics="
              f"{json.dumps(metrics, ensure_ascii=False, sort_keys=True)}", flush=True)
        return {"text": content,
                "reasoning": reasoning,
                "tool_calls": tool_calls,
                "token_ids": token_ids,
                "input_tokens": len(input_ids),
                "output_tokens": counter[0],
                "metrics": metrics}

    def stream(self, messages, max_tokens: int = 1024, *, tools=None,
               tool_choice=None, model: str = "qwen3.8-27b",
               reasoning_effort=None, thinking=None, temperature=1.0) -> Iterator[dict]:
        """Yield OpenAI-friendly internal events for the HTTP SSE adapter."""
        prompt = self.render_chat(
            messages, tools, tool_choice,
            reasoning_effort=reasoning_effort, thinking=thinking)
        input_ids = self.tokenizer.encode(prompt).ids
        submitted_at = time.perf_counter()
        queue = self.strategy.query(input_ids, max_tokens, temperature=temperature)
        print(f"[service] stream id={queue.request_id} input={len(input_ids)} "
              f"max_new={max_tokens}", flush=True)
        message_id = "msg_" + uuid.uuid4().hex[:24]

        # Match Ref: the public stream starts only after the backend confirms
        # that prefill completed.  A backend error before prefill therefore
        # cannot be mistaken for a valid message_start event.
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
               "message": {"id": message_id, "model": model,
                           "usage": {"input_tokens": len(input_ids)}},
               "ljqinfer": start_stats}

        enable_thinking, _ = self._thinking_options(
            reasoning_effort, thinking)
        parser = SurfaceParser(enable_thinking)
        counter = [0]
        first_token_at = None
        open_kind = None
        index = -1

        def _close():
            nonlocal open_kind
            if open_kind is not None:
                yield {"type": "content_block_stop", "index": index}
                open_kind = None

        def _emit(kind, piece):
            nonlocal open_kind, index
            if not piece:
                return
            if open_kind != kind:
                yield from _close()
                index += 1
                open_kind = kind
                block = ({"type": "thinking", "thinking": ""}
                         if kind == SurfaceParser.THINKING
                         else {"type": "text", "text": ""})
                yield {"type": "content_block_start", "index": index,
                       "content_block": block}
            delta = ({"type": "thinking_delta", "thinking": piece}
                     if kind == SurfaceParser.THINKING
                     else {"type": "text_delta", "text": piece})
            yield {"type": "content_block_delta", "index": index, "delta": delta}

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
        _, calls = parse_tool_calls(parser.tool_source)
        for call in calls:
            index += 1
            yield {"type": "content_block_start", "index": index,
                   "content_block": {"type": "tool_use",
                                     "id": call["id"],
                                     "name": call["function"]["name"],
                                     "input": {}}}
            yield {"type": "content_block_delta", "index": index,
                   "delta": {"type": "input_json_delta",
                             "partial_json": call["function"]["arguments"]}}
            yield {"type": "content_block_stop", "index": index}

        stats = self._stats(queue, submitted_at, first_token_at,
                            time.perf_counter(), counter[0])
        print(f"[ljqinfer] id={queue.request_id} "
              f"metrics={json.dumps(stats, ensure_ascii=False, sort_keys=True)}",
              flush=True)
        yield {"type": "message_delta",
               "delta": {"stop_reason": "tool_use" if calls else "end_turn",
                         "stop_sequence": None},
               "usage": {"input_tokens": len(input_ids),
                         "output_tokens": counter[0]},
               "ljqinfer": stats}
        yield {"type": "message_stop"}
