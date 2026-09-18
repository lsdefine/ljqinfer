#!/usr/bin/env python3
"""Reusable black-box acceptance suite for the running ljqinfer HTTP server.

This suite talks only to ``/health`` and ``/v1/messages``. It deliberately does
not import model/CUDA code, inspect GPUs, parse engine logs, or manage processes.
The Q4 and concurrent labels describe server-visible production requests; this
script validates HTTP behavior, not internal CUDA graph capture.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class CaseResult:
    name: str
    seconds: float
    detail: str


class TestFailure(AssertionError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise TestFailure(message)


class ServerClient:
    def __init__(self, base_url: str, api_key: str, model: str,
                 timeout: float) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    def _request(self, path: str, *, payload: dict[str, Any] | None = None,
                 timeout: float | None = None) -> tuple[int, Any]:
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        headers = {"accept": "application/json"}
        if payload is not None:
            headers["content-type"] = "application/json"
        if self.api_key:
            headers["x-api-key"] = self.api_key
        request = urllib.request.Request(self.base_url + path, data=data,
                                         headers=headers,
                                         method="POST" if data is not None else "GET")
        try:
            with urllib.request.urlopen(request, timeout=timeout or self.timeout) as response:
                raw = response.read().decode("utf-8", "replace")
                return response.status, json.loads(raw)
        except urllib.error.HTTPError as exc:
            raw = exc.read().decode("utf-8", "replace")
            try:
                body: Any = json.loads(raw)
            except json.JSONDecodeError:
                body = raw
            raise TestFailure(f"HTTP {exc.code} {path}: {body}") from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            raise TestFailure(f"request failed {path}: {exc}") from exc

    def health(self) -> dict[str, Any]:
        status, body = self._request("/health")
        require(status == 200, f"health returned HTTP {status}")
        require(isinstance(body, dict) and body.get("status") == "ok",
                f"unexpected health response: {body!r}")
        return body

    def messages(self, *, messages: list[dict[str, Any]], max_tokens: int,
                 tools: list[dict[str, Any]] | None = None,
                 tool_choice: dict[str, Any] | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": messages,
        }
        if tools is not None:
            payload["tools"] = tools
        if tool_choice is not None:
            payload["tool_choice"] = tool_choice
        status, body = self._request("/v1/messages", payload=payload)
        require(status == 200, f"messages returned HTTP {status}")
        require(isinstance(body, dict), f"response is not an object: {body!r}")
        require(body.get("type") == "message", f"unexpected response type: {body!r}")
        require(body.get("role") == "assistant", f"unexpected response role: {body!r}")
        require(isinstance(body.get("content"), list), "response content is not a list")
        return body


def text_content(response: dict[str, Any]) -> str:
    return "".join(
        block.get("text", "")
        for block in response.get("content", [])
        if isinstance(block, dict) and block.get("type") == "text"
    ).strip()


def metrics(response: dict[str, Any]) -> dict[str, Any]:
    value = response.get("ljqinfer")
    require(isinstance(value, dict), "response is missing ljqinfer metrics")
    return value


def weather_tool() -> dict[str, Any]:
    return {
        "name": "get_weather",
        "description": "Get current weather for a location",
        "input_schema": {
            "type": "object",
            "properties": {"location": {"type": "string"}},
            "required": ["location"],
        },
    }


def run_case(name: str, fn: Callable[[], str]) -> CaseResult:
    started = time.monotonic()
    detail = fn()
    elapsed = time.monotonic() - started
    print(f"PASS {name:<24} {elapsed:7.3f}s  {detail}", flush=True)
    return CaseResult(name, elapsed, detail)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Black-box acceptance suite for a running ljqinfer server")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--api-key",
                        default="devkey")
    parser.add_argument("--model", default="ljqinfer-glm-5.2")
    parser.add_argument("--timeout", type=float, default=300.0,
                        help="per-request timeout in seconds (default: 300)")
    args = parser.parse_args(argv)
    client = ServerClient(args.base_url, args.api_key, args.model, args.timeout)
    results: list[CaseResult] = []
    tool_state: dict[str, Any] = {}

    print("ljqinfer server black-box acceptance")
    print(f"target={client.base_url} model={client.model}")
    print("scope=HTTP protocol/concurrency/cache only; no CUDA or engine-log assertions")

    def health_case() -> str:
        body = client.health()
        return f"status={body.get('status')} model={body.get('model', '-') }"

    def tool_use_case() -> str:
        response = client.messages(
            messages=[{
                "role": "user",
                "content": "What is the weather in Tokyo? Use get_weather. Do not answer from memory.",
            }],
            max_tokens=64,
            tools=[weather_tool()],
            tool_choice={"type": "tool", "name": "get_weather"},
        )
        calls = [b for b in response["content"]
                 if isinstance(b, dict) and b.get("type") == "tool_use"]
        require(response.get("stop_reason") == "tool_use",
                f"expected stop_reason=tool_use, got {response.get('stop_reason')!r}")
        require(len(calls) == 1, f"expected exactly one tool_use, got {calls!r}")
        call = calls[0]
        require(call.get("name") == "get_weather", f"wrong tool: {call!r}")
        require(isinstance(call.get("id"), str) and call["id"],
                f"missing tool_use id: {call!r}")
        location = call.get("input", {}).get("location", "")
        require(str(location).lower() == "tokyo", f"wrong tool input: {call!r}")
        tool_state["call"] = call
        return f"tool={call['name']} location={location} stop=tool_use"

    def tool_result_case() -> str:
        call = tool_state["call"]
        response = client.messages(
            messages=[
                {"role": "user", "content": "What is the weather in Tokyo? Use get_weather."},
                {"role": "assistant", "content": [call]},
                {"role": "user", "content": [{
                    "type": "tool_result",
                    "tool_use_id": call["id"],
                    "content": "Tokyo: 27 C, sunny.",
                }]},
            ],
            max_tokens=64,
            tools=[weather_tool()],
        )
        text = text_content(response)
        require(text, f"tool result continuation returned no text: {response!r}")
        require("27" in text or "sunny" in text.lower(),
                f"answer did not use tool result: {text!r}")
        require(response.get("stop_reason") == "end_turn",
                f"continuation did not end normally: {response.get('stop_reason')!r}")
        return f"stop=end_turn text={text[:70]!r}"

    def q4_case() -> str:
        expected = "HELLO"
        response = client.messages(
            messages=[{"role": "user",
                       "content": f"Reply with exactly: {expected}"}],
            max_tokens=16,
        )
        text = text_content(response)
        stat = metrics(response)
        output_tokens = int(stat.get("output_tokens", 0) or 0)
        require(text == expected,
                f"Q4 short-sentence mismatch: expected {expected!r}, got {text!r}")
        require(output_tokens >= 2,
                f"Q4 request emitted fewer than 2 tokens: {output_tokens}")
        return f"output_tokens={output_tokens} exact_text={text!r}"

    def concurrent_case() -> str:
        barrier = threading.Barrier(2)

        def one(label: str) -> tuple[str, dict[str, Any]]:
            barrier.wait(timeout=10)
            response = client.messages(
                messages=[{"role": "user",
                           "content": f"Reply with exactly two words: {label} OK"}],
                max_tokens=16,
            )
            return label, response

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(one, "FIRST"), pool.submit(one, "SECOND")]
            pairs = [future.result(timeout=args.timeout + 15) for future in futures]
        details = []
        for label, response in pairs:
            text = text_content(response)
            output_tokens = int(metrics(response).get("output_tokens", 0) or 0)
            require(text, f"concurrent request {label} returned no text")
            require(output_tokens >= 2,
                    f"concurrent request {label} emitted {output_tokens} tokens")
            details.append(f"{label}:{output_tokens}tok")
        return "concurrent=2 " + ",".join(details)

    def cold_hot_case() -> str:
        nonce = uuid.uuid4().hex
        prompt = (
            f"Cache acceptance nonce {nonce}. "
            "Read this stable sentence and reply with exactly: CACHE OK. "
            "The request is intentionally repeated byte-for-byte to verify server cache metrics."
        )
        kwargs = {
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 16,
        }
        cold = client.messages(**kwargs)
        hot = client.messages(**kwargs)
        cold_m, hot_m = metrics(cold), metrics(hot)
        cold_input = int(cold_m.get("input_tokens", 0) or 0)
        hot_input = int(hot_m.get("input_tokens", 0) or 0)
        cold_hit = int(cold_m.get("cache_hit_tokens", 0) or 0)
        hot_hit = int(hot_m.get("cache_hit_tokens", 0) or 0)
        hot_rate = float(hot_m.get("cache_hit_rate", 0.0) or 0.0)
        require(text_content(cold) and text_content(hot),
                "cold/hot request returned empty text")
        require(cold_input > 0 and hot_input == cold_input,
                f"input metric mismatch: cold={cold_input}, hot={hot_input}")
        require(hot_hit > cold_hit,
                f"hot repeat did not improve cache hits: cold={cold_hit}, hot={hot_hit}")
        require(hot_rate >= 0.90,
                f"hot repeat cache_hit_rate below 90%: {hot_rate:.6f}")
        return (f"cold={cold_hit}/{cold_input} hot={hot_hit}/{hot_input} "
                f"rate={hot_rate:.1%}")

    cases = [
        ("health", health_case),
        ("real_tool_use", tool_use_case),
        ("tool_result_roundtrip", tool_result_case),
        ("q4_server_shape", q4_case),
        ("concurrent_requests", concurrent_case),
        ("short_cold_hot_repeat", cold_hot_case),
    ]

    suite_started = time.monotonic()
    try:
        for name, function in cases:
            results.append(run_case(name, function))
    except Exception as exc:
        elapsed = time.monotonic() - suite_started
        print(f"FAIL {name:<24} {type(exc).__name__}: {exc}", file=sys.stderr)
        print(f"SUMMARY FAIL passed={len(results)}/{len(cases)} elapsed={elapsed:.3f}s",
              file=sys.stderr)
        return 1

    elapsed = time.monotonic() - suite_started
    print(f"SUMMARY PASS passed={len(results)}/{len(cases)} elapsed={elapsed:.3f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
