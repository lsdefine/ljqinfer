"""GPU regression for single-graph live-prefix paged index scoring."""
import pytest
import torch
from ops.decode.live_index import scores, candidate_rows
from ops.prefill.candidates import build

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')

@pytest.mark.parametrize('ratio', [1, 2])
@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_graph_live_positions_and_permuted_pages(ratio, dtype):
    torch.manual_seed(20260915)
    device = 'cuda'
    n, rpp, h, d, t = 4096, 128, 4, 128, 7
    pool = torch.randn(n // rpp, rpp, d, device=device, dtype=dtype)
    table = torch.randperm(n // rpp, device=device)
    # Query head stride differs from a fully contiguous tensor.
    q = torch.randn(t, h * 2, d, device=device, dtype=dtype)[:, ::2]
    w = torch.rand(t, h, device=device)
    pos = torch.arange(t, device=device, dtype=torch.long)
    ix = torch.arange(n, device=device)
    keys = pool[table[ix // rpp], ix % rpp].float()
    for _ in range(2):
        scores(q, w, pool, table, pos, ratio, n, h)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        got = scores(q, w, pool, table, pos, ratio, n, h)
    # Increasing AND decreasing lengths: no stale live rows survive a replay.
    for start in [0, 16, 127 * ratio, 2047 * ratio, 8, n * ratio - t]:
        pos.copy_(torch.arange(start, start + t, device=device))
        graph.replay()
        # The scan is a bf16 tensor-core mma whatever the pool dtype, so the
        # reference rounds its inputs the same way: these scores only rank
        # blocks, and the index query is fp4-rounded upstream anyway.
        ref = (torch.einsum('thd,kd->thk', q.bfloat16().float(),
                            keys.bfloat16().float()).relu()
               * w[:, :, None]).sum(1) * d**-.5 * h**-.5
        ref.masked_fill_(ix[None, :] >= ((pos[:, None] + 1) // ratio), -torch.inf)
        # Loose only by the mma's fp32 accumulation order, not by its inputs.
        torch.testing.assert_close(got, ref, atol=2e-5, rtol=1e-4)
        for block_size, top_blocks in [(16, 4), (128, 2)]:
            expected = build(q, w, keys, pos, ratio, block_size=block_size,
                             top_blocks=top_blocks, total_heads=h)
            actual = candidate_rows(got, pos, ratio, block_size, top_blocks)
            assert torch.equal(actual, expected), (dtype, ratio, start, block_size)
