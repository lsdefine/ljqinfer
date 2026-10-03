"""Full released text backbone, canonical mixed precision and real Engram hash.
Random weight values only; this is single-rank computation, not EP8 acceptance.
"""
import pytest
import torch
from released_random import build
from model.past import SlotPool
from model.model_api import ModelExecution
from ops.prefill import residual


def test_rms_bf16_rounds_after_weight():
    g=torch.Generator().manual_seed(102)
    x=torch.randn(3,5120,generator=g).bfloat16()
    w=torch.randn(5120,generator=g).bfloat16()
    expected=(x.float()*torch.rsqrt(x.float().square().mean(-1,keepdim=True)+1e-6)*w.float()).bfloat16()
    torch.testing.assert_close(residual.rms(x,w),expected,rtol=0,atol=0)
    wrong=(x.float()*torch.rsqrt(x.float().square().mean(-1,keepdim=True)+1e-6)).bfloat16()*w
    assert not torch.equal(wrong,expected)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='full released shapes require CUDA')
@pytest.mark.parametrize('chunks',[(3,),(127,2)])
def test_mixed_finish_and_cold_replay(chunks):
    model,w=build(mixed=True)
    past=SlotPool(1,256,page_tokens=16,device='cuda').configure_default()
    execution=ModelExecution(model,past)
    slot=past.alloc()
    tokens=tuple(range(7,7+sum(chunks)))
    start=0
    for size in chunks:
        output=execution.prefill_chunk(slot,tokens[start:start+size],
            history_tokens=tokens[max(0,start-3):start])
        assert output.output.logits is None
        start+=size
    assert w.dtype_hits=={'fp4','fp8','bf16'}
    assert past.prefill_tails[slot]['h'].dtype==torch.bfloat16
    before={i:(s.ckv(slot,start).clone(),s.index_k(slot,start).clone()) for i,s in past.sources.items()}
    w.accesses.clear()
    out=execution.finish_prefill(slot)
    assert out.logits.shape==(1,129280) and out.logits.dtype==torch.float32
    assert out.main_hidden.shape==(min(start,128),3*5120)
    assert out.main_hidden.dtype==torch.bfloat16
    assert torch.isfinite(out.logits).all() and torch.isfinite(out.main_hidden).all()
    assert [int(n.split('.')[1]) for n in w.accesses if n.endswith('.ffn.gate.weight')]==list(range(20,40))
    assert past.pos[slot]==start
    for i,s in past.sources.items():
        torch.testing.assert_close(s.ckv(slot,start),before[i][0],rtol=0,atol=0)
        torch.testing.assert_close(s.index_k(slot,start),before[i][1],rtol=0,atol=0)
    repeated=execution.finish_prefill(slot)
    torch.testing.assert_close(repeated.logits,out.logits,rtol=0,atol=0)
    blob=past.export_cold(slot,0)
    past.release(slot);slot=past.alloc();past.import_cold(slot,[blob])
    execution.replay_prefix(slot,tokens)
    restored=execution.finish_prefill(slot)
    assert torch.isfinite(restored.logits).all()
    if start<=128:
        torch.testing.assert_close(restored.logits,out.logits,rtol=0,atol=0)
    execution.prefill_chunk(slot,(190,),history_tokens=tokens[-3:])
    assert past.pos[slot]==start+1
    assert torch.isfinite(execution.finish_prefill(slot).logits).all()
    past.release(slot)
