"""Independent leaf numerical checks, not whole-model/CED parity claims."""
import ast
import os
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
import torch.nn.functional as F
from test_engram_hash import oracle
from released_random import released_hasher
from model.engram import EngramRows
from model.prefill_block import PrefillMoE
from ops.prefill import residual


@pytest.mark.parametrize('score', ['sqrtsoftplus', 'sigmoid', 'softmax'])
def test_mixed_image_route_against_official_forward(monkeypatch, score):
    root = Path(os.environ.get('DSV41_REFERENCE',
        '/mnt/data/kw/models/DeepSeek-V4.1-Flash/inference'))
    source = root / 'model.py'
    if not source.exists():
        pytest.skip('official Gate source unavailable')
    tree = ast.parse(source.read_text())
    gate = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Gate')
    forward = next(n for n in gate.body if isinstance(n, ast.FunctionDef) and n.name == 'forward')
    # Execute the unmodified official method, with its dense linear dependency.
    module = ast.Module(body=[forward], type_ignores=[])
    namespace = dict(torch=torch, F=F, linear=F.linear)
    exec(compile(ast.fix_missing_locations(module), str(source), 'exec'), namespace)
    gen = torch.Generator().manual_seed(912)
    x = torch.randn(19, 32, generator=gen)
    weight = torch.randn(16, 32, generator=gen)
    bias = torch.linspace(-3, 3, 16)
    bias_vl = bias.flip(0)
    mask = torch.tensor([False]*4 + [True]*7 + [False]*8)
    cfg = dict(n_activated_experts=4, gate_temp=1.2, route_scale=1.7,
               norm_topk_prob=True, score_func=score, swiglu_limit=0.)
    official = SimpleNamespace(weight=weight, bias=bias, bias_vl=bias_vl,
        topk=4, gate_temp=1.2, route_scale=1.7, norm_topk_prob=True, score_func=score)
    expected_p, expected_ids = namespace['forward'](official, x, mask)
    captured = {}
    def routed(x, ids, probabilities, **kwargs):
        captured.update(ids=ids.clone(), probabilities=probabilities.clone())
        return torch.zeros_like(x)
    monkeypatch.setattr(residual, 'swiglu', lambda a, b, limit: torch.zeros_like(a))
    weights = {'layers.0.ffn.gate.weight':weight, 'layers.0.ffn.gate.bias':bias,
               'layers.0.ffn.gate.bias_vl':bias_vl}
    moe = PrefillMoE(0, cfg, weights, lambda name, x:torch.zeros_like(x), routed)
    moe(x, image_mask=mask)
    torch.testing.assert_close(captured['ids'], expected_ids, rtol=0, atol=0)
    torch.testing.assert_close(captured['probabilities'], expected_p, rtol=0, atol=0)
    _, plain_ids = namespace['forward'](official, x, None)
    assert (plain_ids[mask] != expected_ids[mask]).any(), 'fixture must expose wrong bias'


def test_image_engram_rows_chunks_ced_tail_and_spec_batch(oracle, monkeypatch):
    hasher = released_hasher()
    tokens = tuple(range(7, 20)) + (129264,)*11 + tuple(range(40, 59))
    mask = torch.ones(len(tokens), dtype=torch.bool)
    mask[13:24] = False
    # Full-sequence official state is independent of chunk/microbatch slicing.
    expected_image = oracle(torch.tensor([tokens]), 0, mask[None])[0].clone()
    expected_text = oracle(torch.tensor([tokens]), 0, torch.ones_like(mask)[None])[0].clone()
    monkeypatch.setattr(hasher, 'image_spans', {2:((13, torch.empty(11, 1)),)}, raising=False)
    for layer_index, layer in enumerate(hasher.layout.layer_ids):
        rows = EngramRows(hasher, layer, None)
        for a,b in [(0,16), (16,24), (24,25), (25,26), (26,29), (29,len(tokens)),
                    (18,len(tokens))]:
            history = tokens[max(0,a-3):a]
            actual = rows.ids(a,tokens[a:b],history,slot=2)
            torch.testing.assert_close(actual,expected_image[a:b,layer_index],rtol=0,atol=0)
        # Identical raw IDs, different slot semantics, post-image speculative window.
        a,b = 24,27
        actual = rows.ids((a,a),(tokens[a:b],tokens[a:b]),
                          (tokens[a-3:a],tokens[a-3:a]),slot=(2,3))
        expected = torch.cat((expected_image[a:b,layer_index],expected_text[a:b,layer_index]))
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    assert (expected_image[24:27] != expected_text[24:27]).any()
