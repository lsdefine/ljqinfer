"""Released-shape decoder finish, independently scheduled from encoder replay."""
import pytest
import torch
from released_random import build
from numeric_checks import assert_parity_modulo_index_rounding
from model.past import SlotPool
from model.model_api import ModelExecution
from strategy.strategy import Strategy

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='released shapes require CUDA')


@pytest.mark.parametrize('chunks', [(3,), (129,), (127, 2), (129, 3)])
def test_decoder_finish_retains_encoder_tail(chunks):
    model, w = build()
    p = SlotPool(2,256,page_tokens=16,device='cuda').configure_default(
        window_dtype=torch.float32,ckv_dtype=torch.float32,index_dtype=torch.float32)
    ex = ModelExecution(model,p)
    slot = p.alloc()
    tokens = tuple(range(7,7+sum(chunks)))
    offset = 0
    for size in chunks:
        ex.prefill_chunk(slot,tokens[offset:offset+size],
                         history_tokens=tokens[max(0,offset-3):offset]); offset += size
    end = len(tokens); start = max(0,end-128)
    tail = p.prefill_tails[slot]
    assert tail['end'] == end and tail['h'].shape == (end-start,4,5120)
    saved = {k:v.clone() for k,v in tail.items() if torch.is_tensor(v)}
    global_before = {i:(s.ckv(slot,end).clone(),s.index_k(slot,end).clone()) for i,s in p.sources.items()}
    encoder_before = {i:p.windows[i].main_kv[slot].clone() for i in range(20)}
    w.accesses.clear()
    output = ex.finish_prefill(slot)
    assert output.logits.shape == (1,129280) and torch.isfinite(output.logits).all()
    assert output.main_hidden.shape == (end-start,3*5120)
    assert [int(n.split('.')[1]) for n in w.accesses if n.endswith('.ffn.gate.weight')] == list(range(20,40))
    assert not any(n.startswith(tuple(f'layers.{i}.' for i in range(20))) for n in w.accesses)
    assert p.pos[slot] == end
    for k,v in saved.items(): torch.testing.assert_close(tail[k],v,rtol=0,atol=0)
    for i,s in p.sources.items():
        torch.testing.assert_close(s.ckv(slot,end),global_before[i][0],rtol=0,atol=0)
        torch.testing.assert_close(s.index_k(slot,end),global_before[i][1],rtol=0,atol=0)
    for i,v in encoder_before.items(): torch.testing.assert_close(p.windows[i].main_kv[slot],v,rtol=0,atol=0)
    for i in range(20,40): assert torch.isfinite(p.windows[i].read(slot,start,end)).all()
    repeated = ex.finish_prefill(slot)
    torch.testing.assert_close(output.logits,repeated.logits,rtol=0,atol=0)
    if end <= 128:
        # Within one window there is no bounded-encoder approximation.
        replay = model.replay(tokens,past=p,slot=slot,start=0,history_tokens=())
        torch.testing.assert_close(output.logits,replay.logits,rtol=0,atol=0)
    blob = p.export_cold(slot,0)
    p.release(slot)
    assert slot not in p.prefill_tails
    slot = p.alloc(); p.import_cold(slot,[blob])
    with pytest.raises(ValueError): ex.finish_prefill(slot)
    ex.replay_prefix(slot,tokens)
    assert ex.finish_prefill(slot).logits.shape == (1,129280)
    p.release(slot)
    assert not p.prefill_tails


def test_session_explicit_generation_preparation():
    model, _ = build()
    p = SlotPool(1,256,page_tokens=16,device='cuda').configure_default(
        window_dtype=torch.float32,ckv_dtype=torch.float32,index_dtype=torch.float32)
    strategy = Strategy(ModelExecution(model,p,prefill_chunk_tokens=2),namespace='random-ced')
    with strategy.open((7,8,9)) as session:
        with pytest.raises(RuntimeError): session.finish_prefill()
        assert session.run().logits is None
        assert session.finish_prefill().logits.shape == (1,129280)
        session.extend((10,))
        session.run()
        assert session.finish_prefill().logits.shape == (1,129280)
    assert not p.prefill_tails


@pytest.mark.parametrize('length', [3, 129])
def test_finish_independent_attention(length, monkeypatch):
    from replay_attention_reference import sparse_reference
    from ops.prefill import attention
    m,_ = build()
    p=SlotPool(1,256,page_tokens=16,device='cuda').configure_default(
        window_dtype=torch.float32,ckv_dtype=torch.float32,index_dtype=torch.float32)
    slot=p.alloc(); e=ModelExecution(m,p)
    e.prefill_chunk(slot,tuple(range(7,7+length)))
    actual=e.finish_prefill(slot)
    monkeypatch.setattr(attention,'sparse',sparse_reference)
    reference=e.finish_prefill(slot)
    assert_parity_modulo_index_rounding(actual, reference)
    p.release(slot)
