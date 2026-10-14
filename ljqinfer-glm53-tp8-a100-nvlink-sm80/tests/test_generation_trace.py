import json
from queue import Queue
from types import SimpleNamespace
import pytest
from server.generation_trace import GenerationTrace
from server.service import ServiceLayer, ServiceError
import server.generation_trace as trace_module

class Tokenizer:
    def decode(self, ids, **kwargs): return ''.join('The' if i == 785 else f'[{i}]' for i in ids)

@pytest.mark.parametrize('end', ['end','error','close'])
def test_trace_in_decoder_finally(tmp_path, monkeypatch, end):
    monkeypatch.setattr(trace_module, 'DEFAULT_DIR', tmp_path)
    queue=Queue(); queue.request_id='test'
    queue.generation_trace=GenerationTrace([10,20], {'temperature':1,'max_tokens':8192}, 'test')
    queue.put({'type':'token','token_ids':[785]*8192})
    queue.put({'type':end,'error':'injected'})
    svc=object.__new__(ServiceLayer);svc.tokenizer=Tokenizer()
    gen=svc._decode_stream(queue,[0])
    assert next(gen)=='The'*8192
    if end=='close': gen.close()
    elif end=='error':
        with pytest.raises(ServiceError): list(gen)
    else: list(gen)
    files=list(tmp_path.glob('*.json'));assert len(files)==1
    data=json.loads(files[0].read_text())
    assert data['input_ids']==[10,20]
    assert data['output_ids']==[785]*8192
    assert data['longest_same_token_run']=={'token_id':785,'count':8192}
    assert data['terminal'] is None if end=='close' else data['terminal']['type']==end
    assert files[0].stat().st_mode & 0o777 == 0o600


def test_short_request_no_file(tmp_path, monkeypatch):
    monkeypatch.setattr(trace_module,'DEFAULT_DIR',tmp_path)
    t=GenerationTrace([10],{},'short');t.event({'type':'token','token_ids':[785]*8191})
    t.finish(Tokenizer(),SimpleNamespace())
    assert not list(tmp_path.iterdir())


def test_eos_preserved_in_raw_trace(tmp_path, monkeypatch):
    monkeypatch.setattr(trace_module,'DEFAULT_DIR',tmp_path)
    q=Queue();q.generation_trace=GenerationTrace([10],{},'eos')
    q.put({'type':'token','token_ids':[785]*8191+[154820]});q.put({'type':'end'})
    svc=object.__new__(ServiceLayer);svc.tokenizer=Tokenizer()
    list(svc._decode_stream(q,[0]))
    data=json.loads(next(tmp_path.glob('*.json')).read_text())
    assert data['output_ids'][-1]==154820 and data['terminal']['type']=='end'
