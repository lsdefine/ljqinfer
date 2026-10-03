import torch
import pytest
from ops.decode.attention import prepare_candidates
from ops.decode.cand_blocks import candidate_prep
from ops.decode.live_index import candidate_rows

CASES = [(4, 98304, 4, 8, 2048), (3, 4096, 4, 8, 64), (2, 1000, 4, 8, 16), (5, 77, 1, 8, 4)]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs gpu")
@pytest.mark.parametrize("t,n,ratio,block,topk", CASES)
def test_fused_matches_torch_chain(t, n, ratio, block, topk):
    """The fused kernel must agree with the ATen chain it replaces.

    Both paths keep the same number of live rows; the only tolerated
    divergence is which blocks are picked from the tie class sitting
    exactly on the 16-bit score threshold -- an arbitrary choice the
    reference ``topk`` also makes, bounded here at 1% of the budget.
    """
    torch.manual_seed(1234 + n)
    score = torch.randn(t, n, device="cuda")
    pos = torch.randint(n // 2, n, (t,), device="cuda", dtype=torch.int32)
    rows = candidate_rows(score, pos, ratio, block, topk)
    want_ids, want_bad = prepare_candidates(rows, pos, ratio)
    got_ids, got_bad = candidate_prep(score, pos, ratio, block, topk)
    assert got_ids.shape == want_ids.shape and got_ids.dtype == want_ids.dtype
    assert int((~got_bad).sum()) == int((~want_bad).sum())
    for q in range(t):
        want = set(want_ids[q][~want_bad[q]].tolist())
        got = set(got_ids[q][~got_bad[q]].tolist())
        assert len(want - got) == len(got - want)
        assert len(want - got) <= max(1, topk * block // 100)
