"""Small state-contract tests; not a substitute for whole-model parity."""
from types import SimpleNamespace
import torch
import pytest
from model.engram import EngramRows
from model.prefill import PrefillModel


class RecordingHash:
    layout = SimpleNamespace(layer_ids=(1,))
    def __init__(self):
        self.image_spans = {2: ((3, torch.zeros(4, 2)),)}
        self.calls = []
    def __call__(self, tokens, **kw):
        self.calls.append((tokens, kw))
        return torch.zeros(len(tokens), 1, 2, dtype=torch.int64)


@pytest.mark.parametrize('start,length', [(0,4),(4,3),(7,1),(8,4),(10,4)])
def test_current_and_speculative_history_are_position_owned(start, length):
    h=RecordingHash(); rows=EngramRows(h,1,None)
    # Same reserved token both inside and outside the immutable prompt span.
    tokens=(129264,)*length
    history=(129264,)*min(3,start)
    rows.ids(start,tokens,history,slot=2)
    _,kw=h.calls[-1]
    assert kw['token_mask']==tuple(not 3<=p<7 for p in range(start,start+length))
    assert kw['history_mask']==tuple(not 3<=p<7 for p in range(start-len(history),start))


def test_mixed_batch_and_slot_reuse():
    h=RecordingHash(); rows=EngramRows(h,1,None)
    rows.ids((4,4),((129264,),(129264,)),((1,2,3),(1,2,3)),slot=(2,8))
    assert h.calls[0][1]['token_mask']==(False,)
    assert 'token_mask' not in h.calls[1][1]
    del h.image_spans[2]
    rows.ids(4,(129264,),(1,2,3),slot=2)
    assert 'token_mask' not in h.calls[-1][1]


def test_embedding_chunk_boundaries_and_ced_tail():
    m=object.__new__(PrefillModel)
    m.c={'dim':2,'hc_mult':1,'window_size':5}
    m.embed=lambda tokens: torch.tensor([[float(t),-float(t)] for t in tokens])
    features=torch.arange(8,dtype=torch.float32).reshape(4,2)+100
    spans=((3,features),)
    p=SimpleNamespace(prefill_tails={})
    chunks=[]
    for start,end in [(0,4),(4,6),(6,10)]:
        h,pre=m._embed(tuple(range(start,end)),spans,start)
        chunks.append(h[:,0])
        m._retain_encoder(p,2,start,h,pre,[])
    expected=torch.tensor([[float(i),-float(i)] for i in range(10)])
    expected[3:7]=features
    torch.testing.assert_close(torch.cat(chunks),expected,rtol=0,atol=0)
    tail=p.prefill_tails[2]
    assert tail['end']==10
    torch.testing.assert_close(tail['h'][:,0],expected[5:],rtol=0,atol=0)
    mask=m._image_mask(spans,tail['end']-len(tail['h']),len(tail['h']),'cpu')
    assert mask.tolist()==[True,True,False,False,False]
    assert m._image_mask((),0,5,'cpu') is None


def test_prefetch_rejects_reused_slot_layout():
    from model.prefill_block import PrefillEngram
    h=RecordingHash(); rows=EngramRows(h,1,None)
    e=object.__new__(PrefillEngram); e.rows=rows
    tokens=(129264,); hist=(1,2,3)
    class Future:
        cancelled=False
        def result(self): raise AssertionError('stale rows were consumed')
        def cancel(self): self.cancelled=True
    future=Future()
    e._pending=((2,4,tokens,hist,e._mask_key(2,4)),future)
    h.image_spans.clear()
    e._rows_host=lambda *args: 'fresh'
    assert e._gathered(2,4,tokens,{'history_tokens':hist})=='fresh'
    assert future.cancelled


@pytest.mark.parametrize('cancelled', [False, True])
def test_image_request_bypasses_cold_flow_until_release(monkeypatch, cancelled):
    from unittest.mock import Mock
    import strategy.decode_worker as dw
    from model.past import SlotPool
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda: None)
    monkeypatch.setattr(dw, 'sample_rows', lambda *a: torch.tensor([7]))
    past=SlotPool(1,32,device='cpu')
    flow=Mock()
    trunk=SimpleNamespace(prefill_chunk=Mock(), finish_prefill=lambda slot:
                          SimpleNamespace(logits=torch.zeros(1,10),main_hidden=None))
    spec=SimpleNamespace(open=lambda *a,**kw: object())
    e=dw.Engine(0,torch.device('cpu'),past,trunk,spec,9,32,3,flow=flow)
    tokens=(1,129264,129264,129264,129264,2)
    for color in (10.,20.):
        spans=((1,torch.full((4,2),color)),)
        row=e.open_row(tokens,image_spans=spans)
        assert row.session is None and past.image_spans[row.slot] is spans
        if cancelled:
            lane=dw.Lane(0,row,9,10,cancel=SimpleNamespace(is_set=lambda:True))
            assert lane.check_cancel() and lane.reason=='cancelled'
        e.close_row(row)
        assert not past.image_spans and past.free_slots==[0]
    assert flow.mock_calls==[]
    # The same engine retains the original text-cache path.
    slot=past.alloc()
    session=SimpleNamespace(slot=slot,hit_tokens=len(tokens),position=len(tokens),
                            metrics=SimpleNamespace(cache_store_seconds=0.,cache_stored_blocks=0),
                            close=lambda:past.release(slot))
    flow.open.return_value=session
    row=e.open_row(tokens)
    flow.open.assert_called_once_with(tokens)
    e.close_row(row)
    assert not past.image_spans and past.free_slots==[0]
