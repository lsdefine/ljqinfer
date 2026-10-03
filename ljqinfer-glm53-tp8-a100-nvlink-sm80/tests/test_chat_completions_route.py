#!/usr/bin/env python3
import json

from fastapi.testclient import TestClient
from openai.types.chat import ChatCompletion, ChatCompletionChunk

from server.server import _state, app
from server.service import GenerationResult


class FakeLayer:
    def __init__(self):
        self.requests = []
        self._strategy = self

    def check_ready(self):
        return None

    def build(self, request):
        return [], request

    def generate(self, request, *, on_submit=None):
        if on_submit:
            on_submit(None)
        self.requests.append(request)
        if request.get("tools"):
            return GenerationResult(
                message_id="msg_tool", model="ljqinfer-glm-5.2",
                content=[{"type": "tool_use", "id": "call_weather",
                          "name": "get_weather", "input": {"city": "Tokyo"}}],
                stop_reason="tool_use", input_tokens=20, output_tokens=6)
        return GenerationResult(
            message_id="msg_text", model="ljqinfer-glm-5.2",
            content=[{"type": "text", "text": "hello"}],
            stop_reason="end_turn", input_tokens=10, output_tokens=2)

    def stream(self, request, *, on_submit=None):
        self.requests.append(request)
        if on_submit:
            on_submit(None)
        yield {"type": "message_start", "message": {
            "id": "msg_stream", "model": "ljqinfer-glm-5.2",
            "usage": {"input_tokens": 11, "output_tokens": 0}}}
        yield {"type": "content_block_start", "index": 0,
               "content_block": {"type": "text", "text": ""}}
        yield {"type": "content_block_delta", "index": 0,
               "delta": {"type": "text_delta", "text": "hi"}}
        yield {"type": "content_block_stop", "index": 0}
        yield {"type": "message_delta",
               "delta": {"stop_reason": "end_turn"},
               "usage": {"output_tokens": 1}}
        yield {"type": "message_stop"}


def _client():
    fake = FakeLayer()
    _state["service"] = fake
    _state["config"] = {"model_name": "ljqinfer-glm-5.2"}
    return TestClient(app), fake


def test_auth_and_validation():
    client, _ = _client()
    response = client.post("/v1/chat/completions", json={"messages": []})
    assert response.status_code == 401
    response = client.post(
        "/v1/chat/completions", headers={"Authorization": "Bearer devkey"},
        json={"model": "x", "messages": [], "n": 2})
    assert response.status_code == 400
    assert response.json()["error"]["type"] == "invalid_request_error"


def test_blocking_text_and_tool():
    client, fake = _client()
    headers = {"Authorization": "Bearer devkey"}
    response = client.post("/v1/chat/completions", headers=headers, json={
        "model": "ljqinfer-glm-5.2", "messages": [
            {"role": "system", "content": "brief"},
            {"role": "user", "content": "hello"}]})
    assert response.status_code == 200
    parsed = ChatCompletion.model_validate(response.json())
    assert parsed.choices[0].message.content == "hello"
    assert parsed.choices[0].finish_reason == "stop"
    assert fake.requests[-1]["system"] == "brief"

    response = client.post("/v1/chat/completions", headers=headers, json={
        "model": "ljqinfer-glm-5.2", "messages": [
            {"role": "user", "content": "weather"}],
        "tools": [{"type": "function", "function": {
            "name": "get_weather", "parameters": {"type": "object"}}}]})
    parsed = ChatCompletion.model_validate(response.json())
    call = parsed.choices[0].message.tool_calls[0]
    assert parsed.choices[0].finish_reason == "tool_calls"
    assert call.function.name == "get_weather"
    assert json.loads(call.function.arguments) == {"city": "Tokyo"}


def test_streaming_sse():
    client, _ = _client()
    with client.stream("POST", "/v1/chat/completions",
                       headers={"x-api-key": "devkey"}, json={
                           "model": "ljqinfer-glm-5.2", "stream": True,
                           "stream_options": {"include_usage": True},
                           "messages": [{"role": "user", "content": "hi"}]}) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        lines = [line for line in response.iter_lines() if line.startswith("data: ")]
    assert lines[-1] == "data: [DONE]"
    chunks = [json.loads(line[6:]) for line in lines[:-1]]
    for chunk in chunks:
        ChatCompletionChunk.model_validate(chunk)
    assert chunks[0]["choices"][0]["delta"]["role"] == "assistant"
    assert any(c.get("choices") == [] and c.get("usage", {}).get(
        "completion_tokens") == 1 for c in chunks)


if __name__ == "__main__":
    test_auth_and_validation()
    test_blocking_text_and_tool()
    test_streaming_sse()
    print("CHAT_COMPLETIONS_ROUTE_PASS")
