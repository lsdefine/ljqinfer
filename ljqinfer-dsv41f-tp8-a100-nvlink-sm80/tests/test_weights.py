"""Byte layout plus independent dequantized-matmul oracles, no model weights."""
import pytest
import torch
from model.weights import placement, shard, quant_pair, pack_experts, prepare_rank


def dequant(w, s):
    if w.dtype == torch.int8:
        table = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, 0, -.5, -1, -1.5, -2, -3, -4, -6])
        b = w.to(torch.uint8)
        v = table[torch.stack((b & 15, b >> 4), -1).long()].flatten(-2)
        scale = s.float().sub(127).exp2().repeat_interleave(32, 1)
    else:
        v = w.float()
        scale = s.float().sub(127).exp2().repeat_interleave(32, 0).repeat_interleave(32, 1)
    return v * scale[:v.shape[0], :v.shape[1]]


@pytest.mark.parametrize('prefix,axis', [
    ('layers.2.attn.wq_b', 0), ('layers.2.attn.wo_b', 1),
    ('layers.1.engram.wkv', 1), ('layers.0.ffn.shared_experts.w1', 0),
    ('layers.0.ffn.shared_experts.w2', 1)])
@pytest.mark.parametrize('fp4', [False, True])
def test_quantized_tp_matmul(prefix, axis, fp4):
    torch.manual_seed(3)
    n, k = 256, 512
    w = (torch.randint(-128, 128, (n, k // 2), dtype=torch.int8) if fp4 else
         torch.randn(n, k).to(torch.float8_e4m3fn))
    s = torch.randint(125, 129, (n if fp4 else n // 32, k // 32), dtype=torch.uint8)
    x = torch.randn(6, k)
    expected = x @ dequant(w, s).T
    ys, ws, ss = [], [], []
    for rank in range(8):
        a, b = quant_pair(prefix, w, s, rank)
        ws.append(a); ss.append(b)
        local_x = x if axis == 0 else x[:, rank * (k // 8):(rank + 1) * (k // 8)]
        ys.append(local_x @ dequant(a, b).T)
    actual = torch.cat(ys, 1) if axis == 0 else torch.stack(ys).sum(0)
    torch.testing.assert_close(actual, expected, atol=2e-3, rtol=2e-4)
    assert torch.equal(torch.cat(ws, axis).view(torch.uint8), w.view(torch.uint8))
    assert torch.equal(torch.cat(ss, axis), s)


def test_fp8_shared_288_slice_and_block_rejection():
    w = torch.zeros(2304, 256).to(torch.float8_e4m3fn)
    s = torch.full((72, 8), 127, dtype=torch.uint8)
    a, b = quant_pair('layers.0.ffn.shared_experts.w1', w, s, 7)
    assert a.shape == (288, 256) and b.shape == (9, 8)
    with pytest.raises(ValueError, match='boundary'):
        quant_pair('layers.0.attn.wq_b', w[:264], s[:9], 0)
    with pytest.raises(ValueError, match='scale'):
        quant_pair('layers.0.attn.wq_b', w, s.float(), 0)


# TP8 over the expert intermediate I: every rank owns every expert and keeps
# I/8 of the gate/up rows plus the matching down-projection columns.
I, K = 256, 64


def expert_bank(group, count):
    tensors = {}
    for e in range(count):
        for c, shape in [('w1', (I, K // 2)), ('w3', (I, K // 2)), ('w2', (K, I // 2))]:
            p = f'{group}.ffn.experts.{e}.{c}'
            tensors[p + '.weight'] = torch.full(shape, (e + int(c[-1])) % 120, dtype=torch.int8)
            tensors[p + '.scale'] = torch.full((shape[0], shape[1] * 2 // 32), 127, dtype=torch.uint8)
    return tensors


@pytest.mark.parametrize('group,count', [('layers.0', 384), ('mtp.2', 128)])
@pytest.mark.parametrize('rank', [0, 7])
def test_local_expert_bank_order(group, count, rank):
    tensors = expert_bank(group, count)
    out = pack_experts(tensors, group + '.ffn', rank)
    assert out['w13.weight'].shape == (count, 2, I // 8, K // 2)
    assert out['w2.weight'].shape == (count, K, I // 16)
    assert out['w13.scale'].shape == (count, 2, I // 8, K // 32)
    assert out['w2.scale'].shape == (count, K, I // 256)
    for i in (0, count // 2, count - 1):
        for j, c in enumerate(('w1', 'w3')):
            full = tensors[f'{group}.ffn.experts.{i}.{c}.weight']
            assert torch.equal(out['w13.weight'][i, j], full.narrow(0, rank * (I // 8), I // 8))
        full2 = tensors[f'{group}.ffn.experts.{i}.w2.weight']
        assert torch.equal(out['w2.weight'][i], full2.narrow(1, rank * (I // 16), I // 16))
    prepared = prepare_rank(tensors, rank)
    assert len(prepared) == 4
    del tensors[next(iter(tensors))]
    with pytest.raises(KeyError):
        pack_experts(tensors, group + '.ffn', rank)


def test_incomplete_bank_is_rejected():
    """Every rank needs every expert now; a partial bank must not pad zeros."""
    tensors = expert_bank('layers.0', 384)
    for c in ('w1', 'w3', 'w2'):
        for suffix in ('.weight', '.scale'):
            del tensors[f'layers.0.ffn.experts.383.{c}{suffix}']
    with pytest.raises(KeyError):
        pack_experts(tensors, 'layers.0.ffn', 0)


def test_host_exclusion_and_closed_mapping():
    name = 'layers.1.engram.embed.weight'
    assert placement(name).kind == 'host'
    assert prepare_rank({name: torch.empty(2, 256)}, 0) == {}
    # Experts are no longer expert-parallel: every rank shards every expert.
    gate = shard('layers.0.ffn.experts.48.w1.weight', torch.ones(16, 4), 3)
    assert gate.shape == (2, 4) and placement('layers.0.ffn.experts.48.w1.weight').axis == 0
    assert placement('layers.0.ffn.experts.48.w2.weight').axis == 1
    assert placement('layers.0.ffn.experts.48.w1.weight').owner is None
    with pytest.raises(ValueError): placement('layers.0.unrecognized.weight')
    with pytest.raises(ValueError): placement('layers.0.ffn.experts.384.w1.weight')
    with pytest.raises(ValueError): prepare_rank({}, 8)
    with pytest.raises(ValueError): prepare_rank({'layers.0.attn.wq_b.scale': torch.ones(2)}, 0)
    assert placement('mtp.2.markov_head.head.weight').axis == 0
