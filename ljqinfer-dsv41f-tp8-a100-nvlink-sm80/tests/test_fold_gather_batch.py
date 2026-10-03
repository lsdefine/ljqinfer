"""fold_gather serves a batch in one launch; each request keeps its own ring."""
import pytest
import torch

from ops.decode.fold_fused import fold_gather

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='needs cuda')


def _case(ratio, n, d, slots, phases):
    dev = 'cuda'
    g = torch.Generator(device=dev).manual_seed(1234)
    pool = 6
    kv = torch.randn(pool, 2 * ratio, d, device=dev, generator=g)
    sc = torch.randn(pool, 2 * ratio, d, device=dev, generator=g)
    b = len(slots)
    values = torch.randn(b * n, d, device=dev, generator=g, dtype=torch.bfloat16)
    scores = torch.randn(b * n, d, device=dev, generator=g, dtype=torch.bfloat16)
    pos = torch.cat([torch.arange(p, p + n, device=dev) for p in phases])
    rows = torch.tensor(slots, device=dev, dtype=torch.int32)
    return kv, sc, values, scores, pos, rows, n


@pytest.mark.parametrize('ratio', [2, 4])
@pytest.mark.parametrize('phases', [(0, 1, 3), (5, 5, 5), (7, 2, 4)])
def test_batch_matches_one_at_a_time(ratio, phases):
    slots = (3, 0, 2)
    kv, sc, values, scores, pos, rows, n = _case(ratio, 6, 512, slots, phases)
    v, s = fold_gather(values, scores, kv, sc, pos, rows, ratio,
                       rows_per_request=n)
    out = v.shape[0] // len(slots)
    for i in range(len(slots)):
        vi, si = fold_gather(values[i * n:(i + 1) * n], scores[i * n:(i + 1) * n],
                             kv, sc, pos[i * n:(i + 1) * n], rows[i:i + 1], ratio,
                             rows_per_request=n)
        torch.testing.assert_close(v[i * out:(i + 1) * out], vi, rtol=0, atol=0)
        torch.testing.assert_close(s[i * out:(i + 1) * out], si, rtol=0, atol=0)


def test_batch_rows_read_their_own_slot():
    """Swapping two requests' pool rows swaps their folded output."""
    ratio, n = 2, 4
    kv, sc, values, scores, pos, rows, n = _case(ratio, n, 256, (1, 4), (3, 3))
    v, _ = fold_gather(values, scores, kv, sc, pos, rows, ratio, rows_per_request=n)
    sw = torch.tensor((4, 1), device=rows.device, dtype=torch.int32)
    v2, _ = fold_gather(values, scores, kv, sc, pos, sw, ratio, rows_per_request=n)
    assert not torch.equal(v, v2)
