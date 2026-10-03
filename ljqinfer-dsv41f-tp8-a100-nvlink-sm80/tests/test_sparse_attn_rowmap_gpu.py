"""rowmap: one launch,每个请求读自己那一行的池状态.

A batch never owns neighbouring slots, so the batched launch is handed the
whole page table / tail bank plus a request -> row map.  Reading row map[b]
must give exactly what a lone single-request launch on that row gives.
"""
import pytest
import torch
from ops.decode.sparse_attn import attend

SLOTS = 8
PAGES = 6
QWIN = 6


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@pytest.mark.parametrize('ratio', [1, 2])
@pytest.mark.parametrize('nreq', [1, 2, 3, 4])
def test_rowmap_matches_single_request_launches(nreq, ratio):
    device = 'cuda'
    g = torch.Generator(device=device).manual_seed(4177)
    scale = 512 ** -0.5
    pool = torch.randn(PAGES, 1024, 512, device=device, generator=g).bfloat16()
    tails = torch.randn(SLOTS, 129, 512, device=device, generator=g).bfloat16()
    # Every slot spells its history out of different physical pages, so a map
    # that misses would read another request's keys.
    table = torch.stack([torch.randperm(PAGES, device=device, generator=g)[:2]
                         for _ in range(SLOTS)]).contiguous()
    sink = torch.randn(8, device=device, generator=g)

    slots = [3, 0, 6, 1][:nreq]
    starts = [614, 1226, 842, 910][:nreq]
    q = torch.randn(nreq * QWIN, 8, 512, device=device, generator=g).bfloat16()
    ids = torch.full((nreq * QWIN, 640), -1, device=device, dtype=torch.int64)
    pos = torch.empty(nreq * QWIN, device=device, dtype=torch.int64)
    for b, start in enumerate(starts):
        rows = slice(b * QWIN, (b + 1) * QWIN)
        pos[rows] = torch.arange(start, start + QWIN, device=device)
        total = start // ratio
        live = 100 + b * 7            # a different live count per request
        ids[rows, :live] = torch.arange(live, device=device)
        # ids >= total address the tail bank; keep the interior hole too.
        ids[rows, 512:520] = torch.arange(total, total + 8, device=device)

    out = torch.empty_like(q)
    attend(q, pool, table, tails, sink, ids, out, pos, ratio, scale,
           qwin=QWIN, rowmap=torch.tensor(slots, device=device, dtype=torch.int64))
    assert torch.isfinite(out).all()

    for b, slot in enumerate(slots):
        rows = slice(b * QWIN, (b + 1) * QWIN)
        alone = torch.empty_like(q[rows])
        attend(q[rows].contiguous(), pool, table[slot].contiguous(),
               tails[slot].contiguous(), sink, ids[rows].contiguous(), alone,
               pos[rows].contiguous(), ratio, scale)
        assert torch.equal(alone, out[rows]), f'request {b} on slot {slot}'
