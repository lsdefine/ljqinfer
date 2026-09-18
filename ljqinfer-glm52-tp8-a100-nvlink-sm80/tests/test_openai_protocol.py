#!/usr/bin/env python3
import json

from openai.types.chat import ChatCompletion, ChatCompletionChunk

from server.openai_protocol import StreamAdapter, completion_response, to_service_request
from server.service import GenerationResult, ServiceError


def test_request_conversion():
    body = {
        "model": "ljqinfer-glm-5.2",
        "max_completion_tokens": 123,
        "messages": [
            {"role": "system", "content": "Be concise."},
            {"role": "developer", "content": "Use tools."},
            {"role": "user", "content": "Weather?"},
            {"role": "assistant", "content": None, "tool_calls": [{
                "id": "call_1", "type": "function",
                "function": {"name": "weather", "arguments": "{\"city\":\"Tokyo\"}"},
            }]},
            {"role": "tool", "tool_call_id": "call_1", "content": "sunny"},
        ],
        "tools": [{"type": "function", "function": {
            "name": "weather", "parameters": {"type": "object"}}}],
    }
    got = to_service_request(body)
    assert got["system"] == "Be concise.\n\nUse tools."
    assert got["max_tokens"] == 123
    assert got["messages"][1]["content"][0] == {
        "type": "tool_use", "id": "call_1", "name": "weather",
        "input": {"city": "Tokyo"}}
    assert got["messages"][2]["content"][0]["type"] == "tool_result"
    assert got["tools"] == body["tools"]


def test_blocking_text_and_tool_shapes():
    text = GenerationResult(
        message_id="msg_abc", model="m",
        content=[{"type": "thinking", "thinking": "hmm"},
                 {"type": "text", "text": "hello"}],
        input_tokens=10, output_tokens=2, stats={"cache_hit_tokens": 4})
    payload = completion_response(text)
    parsed = ChatCompletion.model_validate(payload)
    assert parsed.id == "chatcmpl_abc"
    assert parsed.choices[0].message.content == "hello"
    assert payload["choices"][0]["message"]["reasoning_content"] == "hmm"
    assert parsed.usage.total_tokens == 12
    assert parsed.usage.prompt_tokens_details.cached_tokens == 4
    limited = completion_response(text, max_tokens=2)
    assert limited["choices"][0]["finish_reason"] == "length"

    tool = GenerationResult(
        message_id="msg_tool", model="m",
        content=[{"type": "tool_use", "id": "toolu_1",
                  "name": "weather", "input": {"city": "Tokyo"}}],
        input_tokens=8, output_tokens=5)
    payload = completion_response(tool)
    parsed = ChatCompletion.model_validate(payload)
    assert parsed.choices[0].finish_reason == "tool_calls"
    call = parsed.choices[0].message.tool_calls[0]
    assert call.function.name == "weather"
    assert json.loads(call.function.arguments) == {"city": "Tokyo"}


def test_stream_text_tool_and_usage_shapes():
    adapter = StreamAdapter(include_usage=True)
    events = [
        {"type": "message_start", "message": {
            "id": "msg_stream", "model": "m", "usage": {"input_tokens": 9}},
         "ljqinfer": {"cache_hit_tokens": 3}},
        {"type": "content_block_start", "index": 0,
         "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 0,
         "delta": {"type": "text_delta", "text": "hi"}},
        {"type": "content_block_start", "index": 1,
         "content_block": {"type": "tool_use", "id": "toolu_2",
                           "name": "weather", "input": {}}},
        {"type": "content_block_delta", "index": 1,
         "delta": {"type": "input_json_delta",
                   "partial_json": "{\"city\":\"Tokyo\"}"}},
        {"type": "message_delta", "delta": {"stop_reason": "tool_use"},
         "usage": {"input_tokens": 9, "output_tokens": 4},
         "ljqinfer": {"cache_hit_tokens": 3}},
        {"type": "message_stop"},
    ]
    chunks = [chunk for event in events for chunk in adapter.feed(event)]
    for chunk in chunks:
        ChatCompletionChunk.model_validate(chunk)
    assert chunks[0]["choices"][0]["delta"]["role"] == "assistant"
    assert chunks[1]["choices"][0]["delta"]["content"] == "hi"
    assert chunks[-2]["choices"][0]["finish_reason"] == "tool_calls"
    assert chunks[-1]["choices"] == []
    assert chunks[-1]["usage"]["total_tokens"] == 13

    limited = StreamAdapter(max_tokens=4)
    limited_chunks = [chunk for event in events for chunk in limited.feed(event)]
    assert limited_chunks[-1]["choices"][0]["finish_reason"] == "tool_calls"
    text_events = [event for event in events
                   if event.get("content_block", {}).get("type") != "tool_use"
                   and event.get("delta", {}).get("type") != "input_json_delta"]
    text_events[-2] = {"type": "message_delta",
                       "delta": {"stop_reason": "end_turn"},
                       "usage": {"input_tokens": 9, "output_tokens": 4}}
    limited = StreamAdapter(max_tokens=4)
    limited_chunks = [chunk for event in text_events for chunk in limited.feed(event)]
    assert limited_chunks[-1]["choices"][0]["finish_reason"] == "length"


def test_rejections():
    disabled = to_service_request({
        "messages": [{"role": "user", "content": "x"}],
        "tools": [{"type": "function", "function": {"name": "f"}}],
        "tool_choice": "none",
    })
    assert "tools" not in disabled

    for body in (
        {"messages": [{"role": "user", "content": "x"}], "n": 2},
        {"messages": [{"role": "assistant", "content": None,
                       "tool_calls": [{"function": {"name": "x", "arguments": "[]"}}]}]},
        {"messages": [{"role": "user", "content": [{"type": "image_url"}]}]},
        {"messages": [{"role": "user", "content": "x"}],
         "response_format": {"type": "json_object"}},
        {"messages": [{"role": "user", "content": "x"}], "stop": ["END"]},
        {"messages": [{"role": "user", "content": "x"}],
         "tool_choice": "required"},
    ):
        try:
            to_service_request(body)
        except ServiceError:
            pass
        else:
            raise AssertionError(body)


if __name__ == "__main__":
    test_request_conversion()
    test_blocking_text_and_tool_shapes()
    test_stream_text_tool_and_usage_shapes()
    test_rejections()
    print("OPENAI_PROTOCOL_CONTRACT_PASS")
