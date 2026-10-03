"""Original HTTP/service stack with real V4.1 tokenizer, fake decode events."""
import importlib.util
import json
from queue import Queue
from pathlib import Path
import pytest
from fastapi.testclient import TestClient
from server import server, engine_server
from server.service import ServiceLayer, ServiceError, SurfaceParser, parse_tool_calls
from server.encoding_dsv41 import encode_messages, eos_token
from strategy.remote_strategy import BackendUnavailable

MODEL = Path('/mnt/data/kw/models/DeepSeek-V4.1-Flash')


class Handle:
    state = 'running'
    def cancel(self): self.state = 'cancelled'
    def stop_at_semantic_eos(self): self.state = 'done'


class Backend:
    def __init__(self):
        self.text = '你好，世界！'
        self.calls = 0
        self.available = True
    def check_ready(self):
        if not self.available:
            raise BackendUnavailable('model not loaded')
    def query(self, input_ids, max_new_tokens, temperature=1.0):
        self.calls += 1
        q = Queue()
        q.cancel_handle = Handle()
        q.request_id = 'test'
        q.put({'type': 'prefill', 'metrics': {}})
        ids = self.layer.tokenizer.encode(self.text + eos_token, add_special_tokens=False).ids
        for token in ids:
            q.put({'type': 'token', 'token_ids': [token]})
        q.put({'type': 'end', 'reason': 'semantic_eos'})
        return q


@pytest.fixture
def layer():
    if not (MODEL/'tokenizer.json').exists():
        pytest.skip('requires downloaded V4.1 tokenizer')
    b = Backend()
    b.layer = ServiceLayer(b, tokenizer_path=str(MODEL/'tokenizer.json'))
    return b.layer


@pytest.mark.parametrize('effort', ['low', 'high', 'max', 1, 42, 100])
def test_official_template(layer, effort):
    spec = importlib.util.spec_from_file_location('official_test', MODEL/'encoding/encoding.py')
    official = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(official)
    req = {'messages': [{'role': 'user', 'content': '你好'}],
           'thinking': {'type': 'enabled'}, 'reasoning_effort': effort}
    plan = layer.intake(req)
    expected = official.encode_messages(plan['messages'], thinking_mode='thinking',
        drop_thinking=True, reasoning_effort=effort)
    assert layer.render(plan) == expected
    ids, _ = layer.build(req)
    assert layer.tokenizer.decode(ids, skip_special_tokens=False) == expected
    assert layer.stop_ids == {1}


@pytest.mark.parametrize('effort', [0, 101, 1.5, True, 'bogus'])
def test_invalid_effort(layer, effort):
    with pytest.raises(ServiceError):
        layer.build({'messages': [{'role': 'user', 'content': 'hi'}], 'reasoning_effort': effort})
    assert layer._strategy.calls == 0


def test_tool_stream(layer):
    raw = '<｜DSML｜ calls>\n<｜DSML｜ invoke name="weather">\n<｜DSML｜ parameter name="city" string="true">上海</｜DSML｜ parameter>\n</｜DSML｜ invoke>\n</｜DSML｜ calls>'
    parser = SurfaceParser(True)
    out = []
    for char in '推理</think>答复\n\n' + raw:
        out += parser.feed(char)
    out += parser.flush()
    assert ''.join(v for k, v in out if k == parser.THINKING) == '推理'
    assert ''.join(v for k, v in out if k == parser.TEXT) == '答复\n\n'
    assert parse_tool_calls(parser.tool_source)[0]['input'] == {'city': '上海'}
    layer._strategy.text = '推理</think>答复\n\n' + raw
    req = {'messages': [{'role': 'user', 'content': '天气'}], 'thinking': {'type': 'enabled'}}
    result = layer.generate(req)
    assert result.stop_reason == 'tool_use'
    assert result.content[-1]['input'] == {'city': '上海'}
    events = list(layer.stream(req))
    assert events[-1]['type'] == 'message_stop'
    assert any(e.get('content_block', {}).get('type') == 'tool_use' for e in events)


@pytest.mark.parametrize('path', ['/v1/messages', '/v1/chat/completions'])
@pytest.mark.parametrize('stream', [False, True])
def test_http(layer, monkeypatch, path, stream):
    monkeypatch.setitem(server._state, 'service', layer)
    c = TestClient(server.app)
    headers = {'x-api-key': server.API_KEY}
    req = {'model': 'ljqinfer-dsv41f', 'messages': [{'role': 'user', 'content': 'hi'}],
           'max_tokens': 32, 'stream': stream}
    assert c.post(path, json=req).status_code == 401
    r = c.post(path, json=req, headers=headers)
    assert r.status_code == 200, r.text
    assert '你好' in r.text
    if stream:
        assert ('message_stop' if path.endswith('messages') else '[DONE]') in r.text
        assert 'error' not in r.text
    layer._strategy.available = False
    r = c.post(path, json=req, headers=headers)
    assert r.status_code == 503
    req['max_tokens'] = 0
    assert c.post(path, json=req, headers=headers).status_code == 400
    count = c.post('/v1/messages/count_tokens', json={'messages': req['messages']}, headers=headers)
    assert count.status_code == 200 and count.json()['input_tokens'] > 0


def test_rpc_unavailable():
    with TestClient(engine_server.app) as c:
        assert c.get('/health').status_code == 503
        assert c.post('/generate', json={}).status_code == 503
        assert c.post('/cancel/gone').json()['state'] == 'already_finished'
