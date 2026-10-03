"""Regression: top-k padding is an interior hole, not the end of live keys."""
import pytest
import torch
from ops.decode.sparse_attn import attend


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@pytest.mark.parametrize('rows', [1, 6])
@pytest.mark.parametrize('layout', ['interior_hole', 'compact', 'empty'])
def test_sparse_attention_holes(rows, layout):
    device = 'cuda'
    g = torch.Generator(device=device).manual_seed(719)
    q = torch.randn(rows, 8, 512, device=device, generator=g).bfloat16()
    pool = torch.randn(2, 1024, 512, device=device, generator=g).bfloat16()
    tail = torch.randn(129, 512, device=device, generator=g).bfloat16()
    page_table = torch.tensor([1, 0], device=device, dtype=torch.int64)
    pos = torch.arange(307, 307 + rows, device=device, dtype=torch.int64)
    sink = torch.randn(8, device=device, generator=g)
    ids = torch.full((rows, 640), -1, device=device, dtype=torch.int64)
    if layout == 'interior_hole':
        ids[:, :154] = torch.arange(154, device=device)
        ids[:, 512:] = torch.arange(154, 282, device=device)
    elif layout == 'compact':
        ids[:, :282] = torch.arange(282, device=device)
    # pos[0] // ratio = 153 committed rows; row 153 is staged in tail[0].
    keys = torch.cat((pool[1, :153], tail), dim=0).double()
    gathered = keys[ids.clamp_min(0)]
    scores = torch.einsum('thd,tkd->thk', q.double(), gathered) * (512 ** -0.5)
    scores.masked_fill_((ids < 0)[:, None, :], -torch.inf)
    logits = torch.cat((scores, sink.double()[None, :, None].expand(rows, -1, -1)), dim=-1)
    probability = logits.softmax(dim=-1)[..., :-1]
    expected = torch.einsum('thk,tkd->thd', probability, gathered)
    out = torch.empty_like(q)
    attend(q, pool, page_table, tail, sink, ids, out, pos, 2, 512 ** -0.5)
    assert torch.isfinite(out).all()
    if layout == 'empty':
        assert torch.count_nonzero(out) == 0
    else:
        relative_l2 = (out.double() - expected).norm() / expected.norm()
        assert relative_l2.item() < 0.01, relative_l2.item()
        torch.testing.assert_close(out.double(), expected, atol=0.006, rtol=0.02)
