"""Exercise actual endpoint generators without loading GPU weights."""
import json
import threading
import time

import anyio
import pytest
from server import server as api


class Request:
    async def is_disconnected(self):
        return False

    async def json(self):
        return {"messages": [{"role": "user", "content": "test"}],
                "max_tokens": 32, "stream": True}


@pytest.mark.parametrize("protocol", ["openai", "anthropic"])
@pytest.mark.parametrize("ending", ["stop", "error"])
def test_idle_heartbeat_preserves_events(monkeypatch, protocol, ending):
    release = threading.Event()
    events = [
        {"type": "message_start", "message": {"id": "msg_test", "model": "test",
                                                   "usage": {"input_tokens": 1}}},
        {"type": "content_block_start", "index": 0,
         "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 0,
         "delta": {"type": "text_delta", "text": "before"}},
        {"type": "content_block_delta", "index": 0,
         "delta": {"type": "text_delta", "text": "after"}},
    ]

    class Layer:
        def stream(self, *args, **kwargs):
            yield from events[:3]
            release.wait(1.0)  # Simulate boarding/prefill after decode has begun.
            yield events[3]
            if ending == "error":
                raise RuntimeError("injected_failure")
            yield {"type": "message_delta", "delta": {"stop_reason": "end_turn"},
                   "usage": {"output_tokens": 2}}
            yield {"type": "message_stop"}

    monkeypatch.setitem(api._state, "layer", Layer())
    monkeypatch.setattr(api, "HEARTBEAT_SECONDS", 0.03)

    async def collect():
        req = Request()
        if protocol == "openai":
            body = api.ChatCompletionRequest(**(await req.json()))
            response = await api.chat_completions(body, req)
        else:
            response = await api.messages_api(req)
        chunks = []
        heartbeat_times = []
        start = time.monotonic()
        async for chunk in response.body_iterator:
            chunks.append(chunk)
            if chunk == b": keepalive\n\n":
                heartbeat_times.append(time.monotonic() - start)
                if len(heartbeat_times) == 3:
                    release.set()
        assert len(heartbeat_times) >= 3, "backend silence must not suppress heartbeats"
        assert heartbeat_times[0] < 0.5
        data = []
        for line in b"".join(chunks).decode().splitlines():
            if line.startswith("data: ") and line != "data: [DONE]":
                data.append(json.loads(line[6:]))
        if protocol == "openai":
            text = "".join(x["choices"][0]["delta"].get("content", "")
                           for x in data if x.get("choices"))
            assert b"".join(chunks).count(b"data: [DONE]") == 1
            assert sum(x.get("choices", [{}])[0].get("delta", {}).get("role") == "assistant"
                       for x in data) == 1
        else:
            text = "".join(x.get("delta", {}).get("text", "") for x in data)
            assert sum(x.get("type") == "message_start" for x in data) == 1
        assert text == "beforeafter", "heartbeats must neither repeat nor steal tokens"
        assert any("error" in x for x in data) == (ending == "error")
    try:
        anyio.run(collect)
    finally:
        release.set()
