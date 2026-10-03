import torch
import pytest

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="gpu only")

SLOTS, TOKENS, PAGES, PAGE = 4, 16384, 64, 256


def _fixture(dtype=torch.bfloat16, ratio=2, dim_a=512, dim_b=128):
    from model.paging import PageTable, PagedPool
    pt = PageTable(SLOTS, TOKENS, PAGES, PAGE, device="cuda")
    a = PagedPool(pt, ratio, dim_a, device="cuda", dtype=dtype)
    b = PagedPool(pt, ratio, dim_b, device="cuda", dtype=dtype)
    return pt, a, b


def _publish(pt, a, b, ck, ik, slots, starts, counts):
    from ops.decode.paged_commit import commit_pair
    commit_pair(a.data, b.data, pt.table, ck, ik, slots, starts, counts)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32, torch.float16])
def test_single_request_matches_pool_write(dtype):
    torch.manual_seed(0)
    pt, a, b = _fixture(dtype)
    pt.ensure(1, 4096)
    ref_a, ref_b = a.data.clone(), b.data.clone()
    per, count, start = 6, 4, 37
    ck = torch.randn(per, a.data.shape[2], device="cuda", dtype=dtype)
    ik = torch.randn(per, b.data.shape[2], device="cuda", dtype=dtype)
    _publish(pt, a, b, ck, ik, (1,), (start,), (count,))
    a.data, b.data, kept = ref_a, ref_b, (a.data, b.data)
    a.write(1, start, ck[:count])
    b.write(1, start, ik[:count])
    assert torch.equal(kept[0], a.data) and torch.equal(kept[1], b.data)


def test_rows_beyond_count_are_untouched():
    torch.manual_seed(1)
    pt, a, b = _fixture()
    pt.ensure(0, 4096)
    per, count, start = 8, 3, 100
    ck = torch.randn(per, a.data.shape[2], device="cuda", dtype=a.data.dtype)
    ik = torch.randn(per, b.data.shape[2], device="cuda", dtype=b.data.dtype)
    _publish(pt, a, b, ck, ik, (0,), (start,), (count,))
    assert torch.equal(a.read(0, start, start + count), ck[:count])
    assert torch.equal(b.read(0, start, start + count), ik[:count])
    assert not a.read(0, start + count, start + per).any()


def test_zero_count_is_a_noop():
    pt, a, b = _fixture()
    pt.ensure(2, 2048)
    ck = torch.randn(4, a.data.shape[2], device="cuda", dtype=a.data.dtype)
    ik = torch.randn(4, b.data.shape[2], device="cuda", dtype=b.data.dtype)
    _publish(pt, a, b, ck, ik, (2,), (10,), (0,))
    assert not a.data.any() and not b.data.any()


def test_batch_equals_per_slot_writes():
    """One launch for three requests must land exactly where three separate
    writes would, including a request that accepted nothing."""
    torch.manual_seed(2)
    pt, a, b = _fixture()
    slots, starts, counts, per = (0, 1, 3), (5, 64, 130), (2, 4, 0), 4
    for s in slots:
        pt.ensure(s, 4096)
    ck = torch.randn(len(slots) * per, a.data.shape[2], device="cuda", dtype=a.data.dtype)
    ik = torch.randn(len(slots) * per, b.data.shape[2], device="cuda", dtype=b.data.dtype)
    _publish(pt, a, b, ck, ik, slots, starts, counts)
    got_a, got_b = a.data.clone(), b.data.clone()
    a.data.zero_(); b.data.zero_()
    for i, s in enumerate(slots):
        c = counts[i]
        if c:
            a.write(s, starts[i], ck[i * per:i * per + c])
            b.write(s, starts[i], ik[i * per:i * per + c])
    assert torch.equal(got_a, a.data) and torch.equal(got_b, b.data)


def test_batch_exceeding_the_slot_bound_is_rejected():
    pt, a, b = _fixture()
    pt.ensure(0, 1024)
    ck = torch.randn(9, a.data.shape[2], device="cuda", dtype=a.data.dtype)
    ik = torch.randn(9, b.data.shape[2], device="cuda", dtype=b.data.dtype)
    with pytest.raises(RuntimeError):
        _publish(pt, a, b, ck, ik, tuple(range(9)), (0,) * 9, (1,) * 9)
