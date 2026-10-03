"""GPU regression: a stacked page table batches index scoring row-wise.

One launch over B rows of QWIN tokens must equal B single-row launches: the
kernel reads B off the table's first dimension, so the batched decode path and
the single-send path are the same code.
"""
import pytest
import torch
from ops.decode.live_index import scores

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


@pytest.mark.parametrize('ratio', [1, 2])
@pytest.mark.parametrize('b,qwin', [(1, 6), (2, 6), (3, 1), (8, 6)])
def test_stacked_table_matches_per_row_calls(b, qwin, ratio):
    torch.manual_seed(20260918)
    device = 'cuda'
    n, rpp, h, d = 4096, 128, 4, 128
    pool = torch.randn(n // rpp, rpp, d, device=device, dtype=torch.bfloat16)
    # Every row owns a different page permutation, so a row that read its
    # neighbour's table row would score different keys and fail below.
    table = torch.stack([torch.randperm(n // rpp, device=device) for _ in range(b)])
    q = torch.randn(b * qwin, h, d, device=device, dtype=torch.bfloat16)
    w = torch.rand(b * qwin, h, device=device)
    # Unrelated live prefixes per row, including a very short one: the window
    # limit is read per token from pos, never from the batch.
    starts = [17, 2039, 128, 4090, 63, 1024, 333, 2][:b]
    pos = torch.cat([torch.arange(s, s + qwin, device=device) for s in starts])
    got = scores(q, w, pool, table, pos, ratio, n, h)
    for i in range(b):
        rows = slice(i * qwin, (i + 1) * qwin)
        ref = scores(q[rows], w[rows], pool, table[i], pos[rows], ratio, n, h)
        assert torch.equal(got[rows], ref), (b, qwin, ratio, i)
