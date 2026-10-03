"""TP8 MoE workspace: real-bank parity oracle and CUDA graph capture.

Skipped unless the released checkpoint is present; this is the only coverage of
WorkspaceRouted.__call__, the device routed path used by prefill and CED.
"""
import os
import pytest
import torch

WEIGHTS = os.environ.get('DSV41F_WEIGHTS', '/mnt/data/kw/models/DeepSeek-V4.1-Flash')
pytestmark = pytest.mark.skipif(not (torch.cuda.is_available() and os.path.isdir(WEIGHTS)),
                                reason='released checkpoint and CUDA required')


def oracle(c, x, ids, probabilities, b13, bs13, b2, bs2):
    from ops.prefill.grouped_moe import unpack_fp4
    from ops.prefill import residual as r
    out = torch.zeros(x.shape[0], c['dim'], device=x.device, dtype=torch.float32)
    for e in range(c['n_routed_experts']):
        token, slot = torch.where(ids == e)
        if not len(token):
            continue
        rows = x[token]
        w13 = unpack_fp4(b13[e], bs13[e])
        y = r.swiglu(torch.nn.functional.linear(rows, w13[0]),
                     torch.nn.functional.linear(rows, w13[1]),
                     c['swiglu_limit'], probabilities[token, slot, None])
        w2 = unpack_fp4(b2[e][None], bs2[e][None])[0]
        out.index_add_(0, token, torch.nn.functional.linear(y, w2).float())
    return out


@pytest.mark.parametrize('length', [128])
def test_routed_workspace_matches_bank_oracle_and_captures(length):
    from model.prefill_config import released_config
    from model.checkpoint_weights import CheckpointWeights
    from ops.prefill.moe_workspace import WorkspaceRouted
    device = torch.device('cuda', 0)
    torch.cuda.set_device(device)
    c = released_config()
    weights = CheckpointWeights(WEIGHTS, device, 0)
    weights.load_experts()

    class Serial:
        rank = 0

        def sum(self, tensor):
            return None

    op = WorkspaceRouted('layers.0.ffn', c, weights, Serial(), length)
    gen = torch.Generator(device=device).manual_seed(817)
    x = (torch.randn(length, c['dim'], device=device, generator=gen)*0.05).bfloat16()
    ids = torch.rand(length, c['n_routed_experts'], device=device, generator=gen).topk(c['n_activated_experts'], -1).indices
    p = torch.rand(length, ids.shape[1], device=device, generator=gen)
    p = p/p.sum(-1, keepdim=True)
    base = 'layers.0.ffn.local_experts.'
    banks = (weights[base+'w13.weight'], weights[base+'w13.scale'],
             weights[base+'w2.weight'], weights[base+'w2.scale'])
    reference = oracle(c, x, ids, p, *banks)
    actual = op(x, ids, p).clone()
    # FP8 activation quantisation sets the floor; anything mis-routed is orders larger.
    assert (actual-reference).norm()/reference.norm() < 0.06

    for _ in range(3):
        op(x, ids, p)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = op(x, ids, p)
    torch.cuda.synchronize()
    # Routing must stay data dependent inside the graph: replace it and replay.
    ids.copy_(torch.rand(length, c['n_routed_experts'], device=device, generator=gen).topk(ids.shape[1], -1).indices)
    graph.replay()
    torch.cuda.synchronize()
    replayed = oracle(c, x, ids, p, *banks)
    assert (captured-replayed).norm()/replayed.norm() < 0.06
