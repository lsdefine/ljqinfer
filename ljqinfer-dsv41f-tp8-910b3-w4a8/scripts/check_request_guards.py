"""CPU-only request rejection regression; no model/device commands."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import queue
from types import SimpleNamespace
from strategy.decode_worker import Engine, QueueStrategy, RequestRejected, RequestBusy


def main():
    eng = Engine.__new__(Engine)
    eng.rows, eng.failed, eng.row_cap = {}, None, 1024*1024
    eng.reservations = {0: 15}
    eng.past = SimpleNamespace(page_tokens=16, pt=SimpleNamespace(n_pages=16))
    def forbidden(*args, **kwargs):
        raise AssertionError('bad request reached device command')
    eng.command = forbidden
    jobs = queue.Queue()
    backend = QueueStrategy(eng, jobs)
    bad = [dict(temperature=t) for t in (99.9999, 100, -1, float('nan'), float('inf'), True, '1')]
    bad += [dict(max_new_tokens=n) for n in (0, -1, 1.5, True, '8')]
    bad += [dict(input_ids=t) for t in ([], None, [True], [-1], [129280], ['1'])]
    bad += [dict(images=p) for p in ({}, [], {'version':1,'images':[None]}, {'version':2,'images':[]})]
    bad += [dict(input_ids=[129264])]
    for change in bad:
        args = dict(input_ids=[1,2], max_new_tokens=8, temperature=0)
        args.update(change)
        try:
            backend.query(**args)
        except (ValueError, TypeError):
            pass
        else:
            raise AssertionError(f'accepted bad request: {change}')
        assert jobs.empty() and backend.failure is None and eng.failed is None
    try:
        eng.admit([1], 8, temperature=99.9999)
    except RequestRejected:
        pass
    else:
        raise AssertionError('admit did not reject bad temperature')
    try:
        eng.admit([1]*16, 8)
    except RequestBusy:
        pass
    else:
        raise AssertionError('KV budget was not enforced before command')
    assert eng.validate_request([1]*(600*1024), 8, 0) == 8
    backend.query([1,2], 8, 0)
    assert jobs.qsize() == 1 and backend.failure is None and eng.failed is None
    print(f'PASS {len(bad)} bad requests rejected; next valid request accepted; 600K admission; KV wait')

def http_check():
    import threading
    from fastapi.testclient import TestClient
    import server.engine_server as es
    eng = Engine.__new__(Engine)
    eng.rows, eng.failed, eng.row_cap = {}, None, 1024*1024
    jobs = queue.Queue()
    backend = QueueStrategy(eng, jobs)
    es.strategy = backend
    with TestClient(es.app) as client:
        base = dict(request_id='test', input_ids=[1,2], max_new_tokens=8)
        for change in (dict(temperature=99.9999), dict(images={}),
                       dict(images={'version':1,'images':[None]}),
                       dict(input_ids=[129264]), dict(max_new_tokens=True)):
            response = client.post('/generate', json=dict(base, **change))
            assert response.status_code == 400, response.text
            assert jobs.empty() and not es._state['handles']
            assert client.get('/health').status_code == 200
        assert client.post('/generate', json={}).status_code == 400
        def complete():
            job = jobs.get(timeout=10)
            job.out.put({'type':'end', 'reason':'length'})
        worker = threading.Thread(target=complete)
        worker.start()
        response = client.post('/generate', json=base)
        worker.join(timeout=10)
        assert response.status_code == 200 and 'end' in response.text
        assert not worker.is_alive() and not es._state['handles']
        assert client.get('/health').status_code == 200
    print('PASS HTTP invalid requests 400; next valid request 200; handles released')

if __name__ == '__main__':
    main()
    http_check()
