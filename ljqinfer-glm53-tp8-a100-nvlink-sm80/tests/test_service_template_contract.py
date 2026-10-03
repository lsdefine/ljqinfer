"""CPU regression of the shipped GLM-5.3 reasoning/template contract."""
from pathlib import Path
from queue import Queue
import json
import pytest
from server.service import DEFAULT_TEMPLATE, ServiceLayer, SurfaceParser, ServiceError
from server.openai_protocol import to_service_request, completion_response, StreamAdapter


class StubStrategy:
    def check_ready(self):
        return None

    def query(self, ids, max_tokens):
        q = Queue()
        q.request_id = 'contract'
        q.put({'type': 'prefill', 'metrics': {}})
        # Split the think delimiter across token events on purpose.
        for token in self.tokens:
            q.put({'type': 'token', 'token_ids': [token]})
        q.put({'type': 'end'})
        return q


@pytest.fixture
def service():
    strategy = StubStrategy()
    layer = ServiceLayer(strategy, model_name='glm-5.3')
    strategy.tokens = layer.encode('Check the sum.</think>42')
    return layer


@pytest.mark.parametrize('effort', ['low', 'high', 'max'])
@pytest.mark.parametrize('field', ['reasoning_effort', 'reasoning', 'output_config', 'chat_template_kwargs'])
def test_effort_inputs(service, effort, field):
    value = effort if field == 'reasoning_effort' else {
        'reasoning_effort' if field == 'chat_template_kwargs' else 'effort': effort}
    body = {'messages': [{'role': 'user', 'content': 'hello'}], field: value}
    for request in (body, to_service_request(body)):
        plan = service.intake(request)
        prompt = service.render(plan)
        assert plan['enable_thinking'] is True
        assert plan['reasoning_effort'] == effort
        assert f'Reasoning Effort: {effort.capitalize()}' in prompt
        assert prompt.endswith('<|assistant|><think>')


def test_default_template_and_parser_contract(service):
    plan = service.intake({'messages':[{'role':'user','content':'hello'}]})
    assert plan['enable_thinking'] is True
    assert plan['reasoning_effort'] == 'max'
    prompt = service.render(plan)
    assert 'Reasoning Effort: Max' in prompt
    assert prompt.endswith('<think>') and not prompt.endswith('<think></think>')
    parser = SurfaceParser(True)
    pieces = parser.feed('reason</thi') + parser.feed('nk>READY') + parser.flush()
    assert ''.join(v for k,v in pieces if k == SurfaceParser.THINKING) == 'reason'
    assert ''.join(v for k,v in pieces if k == SurfaceParser.TEXT) == 'READY'


@pytest.mark.parametrize('kind', ['enabled', 'adaptive'])
def test_thinking_mode(service, kind):
    plan = service.intake({'messages':[{'role':'user','content':'hello'}],
                           'thinking': {'type':kind}, 'output_config': {'effort':'low'}})
    assert plan['enable_thinking'] and plan['reasoning_effort'] == 'low'


@pytest.mark.parametrize('options', [
    {'thinking': {'type':'disabled'}}, {'thinking': {'type':[]}},
    {'thinking': {'type':'adaptive','budget_tokens':1024}},
    {'thinking': {'type':'enabled','budget_tokens':1024}},
    {'enable_thinking':False}, {'enable_thinking':'false'},
    {'chat_template_kwargs':{'enable_thinking':False}},
    {'reasoning_effort':'none'}, {'reasoning_effort':'medium'},
    {'reasoning_effort':'xhigh'}, {'reasoning_effort':[]},
    {'output_config':'low'}, {'reasoning':[]},
    {'reasoning_effort':'low','output_config':{'effort':'high'}},
    {'clear_thinking':'false'},
    {'clear_thinking':True,'chat_template_kwargs':{'clear_thinking':False}},
])
def test_unsupported_controls_are_explicit(service, options):
    with pytest.raises(ServiceError):
        service.intake({'messages':[{'role':'user','content':'hello'}], **options})


def test_native_template_source(service):
    source = Path('/mnt/data2/kw/GLM-5.3-UNCENSORED-FP8/chat_template.jinja')
    assert Path(DEFAULT_TEMPLATE).read_bytes() == source.read_bytes()


def test_blocking_and_stream_outputs(service):
    request = {'messages':[{'role':'user','content':'hello'}], 'output_config':{'effort':'low'}}
    result = service.generate(request)
    assert result.stats['reasoning_effort'] == 'low'
    assert [(b['type'], b.get('thinking', b.get('text'))) for b in result.content] == [
        ('thinking','Check the sum.'), ('text','42')]
    message = completion_response(result)['choices'][0]['message']
    assert message['reasoning_content'] == 'Check the sum.' and message['content'] == '42'
    events = list(service.stream(request))
    assert ''.join(e.get('delta',{}).get('thinking','') for e in events) == 'Check the sum.'
    assert ''.join(e.get('delta',{}).get('text','') for e in events) == '42'
    adapter = StreamAdapter()
    chunks = [c for e in events for c in adapter.feed(e)]
    deltas = [c['choices'][0]['delta'] for c in chunks if c.get('choices')]
    assert ''.join(d.get('reasoning_content','') for d in deltas) == 'Check the sum.'
    assert ''.join(d.get('content','') for d in deltas) == '42'


@pytest.mark.parametrize('endpoint', ['/v1/messages', '/v1/chat/completions'])
@pytest.mark.parametrize('stream', [False, True])
def test_http_reasoning(service, monkeypatch, endpoint, stream):
    from fastapi.testclient import TestClient
    from server.server import app, _state
    monkeypatch.setitem(_state, 'service', service)
    monkeypatch.setitem(_state, 'config', {'model_name':'glm-5.3'})
    options = {'thinking':{'type':'adaptive'}, 'output_config':{'effort':'low'}} if endpoint.endswith('messages') else {'reasoning_effort':'low'}
    response = TestClient(app).post(endpoint, headers={'x-api-key':'devkey'}, json={
        'model':'glm-5.3', 'messages':[{'role':'user','content':'hello'}],
        'max_tokens':128, 'stream':stream, **options})
    assert response.status_code == 200, response.text
    if stream:
        events = [json.loads(l[6:]) for l in response.text.splitlines()
                  if l.startswith('data: ') and l != 'data: [DONE]']
        if endpoint.endswith('messages'):
            assert ''.join(e.get('delta',{}).get('thinking','') for e in events) == 'Check the sum.'
            assert ''.join(e.get('delta',{}).get('text','') for e in events) == '42'
        else:
            deltas = [e['choices'][0]['delta'] for e in events if e.get('choices')]
            assert ''.join(d.get('reasoning_content','') for d in deltas) == 'Check the sum.'
            assert ''.join(d.get('content','') for d in deltas) == '42'
    elif endpoint.endswith('messages'):
        blocks = response.json()['content']
        assert [(b['type'], b.get('thinking', b.get('text'))) for b in blocks] == [('thinking','Check the sum.'), ('text','42')]
    else:
        m = response.json()['choices'][0]['message']
        assert m['reasoning_content'] == 'Check the sum.' and m['content'] == '42'
    bad = TestClient(app).post(endpoint, headers={'x-api-key':'devkey'}, json={
        'messages':[{'role':'user','content':'hello'}], 'stream':stream,
        'thinking':{'type':'disabled'}})
    assert bad.status_code == 400 and bad.json()['error']['type'] == 'invalid_request_error'
