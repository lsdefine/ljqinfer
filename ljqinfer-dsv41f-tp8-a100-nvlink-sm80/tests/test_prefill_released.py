"""Released 40-layer dimensions with lazy random weights, never micro shapes."""
import pytest
import torch
from released_random import build
from numeric_checks import assert_parity_modulo_index_rounding
from model.past import SlotPool
from model.model_api import ModelExecution
from model.prefill_config import released_config, validate_config, rotary_frequencies


def test_released_configuration_and_rope():
    c= released_config()
    validate_config(c)
    assert (c['dim'],c['n_heads'],c['head_dim'],c['n_routed_experts'],c['n_activated_experts']) == (5120,64,512,384,6)
    assert c['engram_layer_ids'] == [1,14]
    assert c['compress_ratios'][20:40] == [1]*20
    assert rotary_frequencies(c,0,8,device='cpu').shape == (8,32)
    assert not torch.equal(rotary_frequencies(c,0,8,device='cpu'),rotary_frequencies(c,2,8,device='cpu'))
    for key in ['dim','n_routed_experts','hc_mult']:
        bad=dict(c);bad[key]//=2
        with pytest.raises(ValueError):validate_config(bad)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='full-dimension random GEMMs require CUDA')
@pytest.mark.parametrize("length", [3, 129])
def test_released_ced_forward_replay(length, monkeypatch):
    model,w=build()
    past=SlotPool(2,256,page_tokens=16,device='cuda').configure_default(
        window_dtype=torch.float32,ckv_dtype=torch.float32,index_dtype=torch.float32)
    slot=past.alloc();ex=ModelExecution(model,past)
    tokens=tuple(range(7,7+length))
    out=ex.prefill_chunk(slot,tokens)
    assert out.output.logits is None and past.pos[slot]==length
    gates=[int(n.split('.')[1]) for n in w.accesses if n.endswith('.ffn.gate.weight')]
    assert gates==list(range(20))
    layer20=[n for n in w.accesses if n.startswith('layers.20.')]
    assert layer20 and all(('.compressor.' in n or '.indexer.wk.' in n or '.indexer.k_norm.' in n or n=='layers.20.attn_norm.weight') for n in layer20)
    assert not any(int(n.split('.')[1])>20 for n in w.accesses)
    for i in range(20,40): assert not past.windows[i].main_kv[slot].count_nonzero()
    globals_before={i:(s.ckv(slot,length).clone(),s.index_k(slot,length).clone()) for i,s in past.sources.items()}
    w.accesses.clear()
    # Simulate missing hot state after importing global-only cold cache.
    for window in past.windows.values():window.main_kv[slot].fill_(float('nan'))
    for source in past.sources.values():
        if source.ratio==2:
            source.kv_state[slot].fill_(float('nan'));source.score_state[slot].fill_(float('nan'))
    past.replay_pending.add(slot)
    start=max(0,length-128)
    output=model.replay(tokens[start:],past=past,slot=slot,start=start,history_tokens=tokens[max(0,start-3):start])
    assert past.pos[slot]==length and slot in past.replay_pending
    assert output.logits.shape==(1,129280) and torch.isfinite(output.logits).all()
    assert output.main_hidden.shape==(min(length,128),3*5120)
    assert [int(n.split('.')[1]) for n in w.accesses if n.endswith('.ffn.gate.weight')]==list(range(40))
    for i in range(40):assert torch.isfinite(past.windows[i].read(slot,start,length)).all()
    for i,source in past.sources.items():
        torch.testing.assert_close(source.ckv(slot,length),globals_before[i][0],rtol=0,atol=0)
        torch.testing.assert_close(source.index_k(slot,length),globals_before[i][1],rtol=0,atol=0)
        if source.ratio==2:
            assert torch.isfinite(source.kv_state[slot]).all()
            assert torch.isfinite(source.score_state[slot]).all()
    from replay_attention_reference import sparse_reference
    from ops.prefill import attention
    actual_sparse = attention.sparse
    compared = []
    def checked_sparse(*args, **kwargs):
        actual = actual_sparse(*args, **kwargs)
        expected = sparse_reference(*args, **kwargs)
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
        compared.append(float((actual-expected).abs().max()))
        # Identical inputs per layer: don't confuse quantizer/top-k branch
        # sensitivity with attention formula correctness.
        return actual
    with monkeypatch.context() as patch:
        patch.setattr(attention, 'sparse', checked_sparse)
        repeated = model.replay(tokens[start:], past=past, slot=slot, start=start,
                                history_tokens=tokens[max(0,start-3):start])
    assert len(compared) == 40
    torch.testing.assert_close(output.logits, repeated.logits, rtol=0, atol=0)
    torch.testing.assert_close(output.main_hidden, repeated.main_hidden, rtol=0, atol=0)
    with pytest.raises(AssertionError):
        torch.testing.assert_close(output.logits, repeated.logits+1, rtol=2e-3, atol=2e-4)
    # Execution owns clearing replay_pending; the compute method never clears it.
    assert ex.replay_prefix(slot,tokens)==min(length,128)
    assert slot not in past.replay_pending
    assert ex.prefill_chunk(slot,(17,),history_tokens=tokens[-3:]).end==length+1
    assert past.pos[slot]==length+1


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')
@pytest.mark.parametrize('length', [3, 129])
def test_free_propagation_reference_parity(length, monkeypatch):
    from replay_attention_reference import sparse_reference
    from ops.prefill import attention
    model, _ = build()
    past = SlotPool(1,256,page_tokens=16,device='cuda').configure_default(
        window_dtype=torch.float32,ckv_dtype=torch.float32,index_dtype=torch.float32)
    slot = past.alloc()
    tokens = tuple(range(7,7+length))
    ModelExecution(model,past).prefill_chunk(slot,tokens)
    start = max(0,length-128)
    kw = dict(past=past,slot=slot,start=start,history_tokens=tokens[max(0,start-3):start])
    actual = model.replay(tokens[start:], **kw)
    monkeypatch.setattr(attention,'sparse',sparse_reference)
    reference = model.replay(tokens[start:], **kw)
    assert_parity_modulo_index_rounding(actual, reference)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')
@pytest.mark.parametrize('length', [3, 129])
def test_replay_without_swa_rounding_diagnostic(length, monkeypatch):
    # Diagnostic only, NOT model acceptance: all model dimensions stay intact,
    # but removing the SWA quantizer isolates discontinuous rounding effects.
    from ops.prefill import attention
    monkeypatch.setattr(attention, 'fp8_roundtrip', lambda x: x)
    test_free_propagation_reference_parity(length, monkeypatch)
