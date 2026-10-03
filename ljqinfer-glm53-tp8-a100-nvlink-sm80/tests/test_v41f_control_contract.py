"""Model-independent contracts inherited from v41f."""
from types import SimpleNamespace
import pytest
import requests
from strategy import remote_strategy as remote


@pytest.mark.parametrize('body', [{'status': 'ok'}, {'status': 'loading'}])
def test_backend_readiness(monkeypatch, body):
    calls = []
    def get(url, **kwargs):
        calls.append((url, kwargs))
        return SimpleNamespace(raise_for_status=lambda: None, json=lambda: body)
    monkeypatch.setattr(remote.requests, 'get', get)
    proxy = remote.RemoteStrategy()
    if body['status'] == 'ok':
        proxy.check_ready()
    else:
        with pytest.raises(remote.BackendUnavailable):
            proxy.check_ready()
    assert calls[0][1]['timeout'] == (3, 5)


def test_backend_unavailable(monkeypatch):
    def get(*args, **kwargs):
        raise requests.ConnectionError('offline')
    monkeypatch.setattr(remote.requests, 'get', get)
    with pytest.raises(remote.BackendUnavailable):
        remote.RemoteStrategy().check_ready()


def test_rpc_preserves_prefill_metrics(monkeypatch):
    import json
    calls = []
    metrics = dict(cache_hit_tokens=21, prefill_tokens=1, input_tokens=22)
    events = [{'type': 'prefill', 'metrics': metrics},
              {'type': 'end', 'decode_tps': 41.0, 'prefill_seconds': .2}]
    response = SimpleNamespace(raise_for_status=lambda: None,
        iter_lines=lambda **kwargs: (json.dumps(x) for x in events), close=lambda: None)
    def post(url, **kwargs):
        calls.append((url, kwargs))
        return response
    monkeypatch.setattr(remote.requests, 'post', post)
    q = remote.RemoteStrategy().query([1, 2], 3)
    assert q.get(timeout=3)['type'] == 'prefill'
    assert q.get(timeout=3)['type'] == 'end'
    assert vars(q.metrics) == metrics
    assert q.decode_stats['decode_tps'] == 41.0
    assert calls[0][1]['timeout'] == (3, 300)
    assert q.cancel_handle.state == 'done'


def test_semantic_stop_fallback(monkeypatch):
    calls = []
    def post(url, **kwargs):
        calls.append((url, kwargs))
        return SimpleNamespace(status_code=404 if len(calls)==1 else 200,
                               raise_for_status=lambda: None)
    monkeypatch.setattr(remote.requests, 'post', post)
    remote._Handle('http://localhost', 'request').stop_at_semantic_eos()
    assert calls[0][0].endswith('/stop/request')
    assert calls[1][0].endswith('/cancel/request')
    assert calls[1][1]['params'] == {'semantic': True}


@pytest.mark.parametrize('value', [True, False, 0, -1, 2.5, '12'])
def test_max_tokens_strict(value):
    from server.service import ServiceLayer, ServiceError
    layer = object.__new__(ServiceLayer)
    layer.max_output_tokens = 4096
    layer.model_name = 'glm-5.3'
    with pytest.raises(ServiceError):
        layer.intake({'messages': [{'role': 'user', 'content': 'hi'}],
                      'max_tokens': value})
