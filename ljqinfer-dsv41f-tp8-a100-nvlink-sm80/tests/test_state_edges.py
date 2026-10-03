"""Adversarial state tests independent of the retired engine."""
import pytest
import torch

from model.paging import PageTable, PagedPool
from model.past import SlotPool


@pytest.mark.parametrize('pos', [127, 128, 129, 255])
@pytest.mark.parametrize('accepted', range(6))
def test_rejected_ring_overwrites_are_restored(pos, accepted):
    pool = SlotPool(1, 512, page_tokens=128).configure_default(
        kv_dim=4, index_dim=2, window_dtype=torch.float32)
    slot = pool.alloc()
    pool.ensure(slot, pos + 5)
    pool.set_pos(slot, pos)
    begin = max(0, pos - 128)
    for layer, window in pool.windows.items():
        history = torch.arange(begin, pos).float()[:, None].repeat(1, 4) + layer * 1000
        window.write(slot, begin, history)
    snapshot = pool.snapshot_verify(slot)
    for layer, window in pool.windows.items():
        pending = torch.arange(pos, pos + 5).float()[:, None].repeat(1, 4) + layer * 1000
        window.write(slot, pos, pending)
    projected = {layer: (torch.zeros(5, 4), torch.zeros(5, 4)) for layer in (2, 8, 14)}
    pool.commit_verify_prefix(snapshot, accepted, projected)
    end = pos + accepted
    for layer, window in pool.windows.items():
        expected = torch.arange(max(0, end - 128), end).float()[:, None].repeat(1, 4) + layer * 1000
        assert torch.equal(window.read(slot, max(0, end - 128), end), expected)


def test_page_exhaustion_does_not_partially_allocate():
    pt = PageTable(2, 512, 3, 128)
    pt.ensure(0, 256)
    before = pt.table.clone()
    with pytest.raises(MemoryError):
        pt.ensure(1, 256)
    assert torch.equal(pt.table, before)
    assert pt.n_alloc(1) == 0
    pt.ensure(1, 128)
    assert pt.n_alloc(1) == 1
    pt.release(0)
    pt.ensure(1, 384)
    assert pt.n_alloc(1) == 3


@pytest.mark.parametrize('ratio', [1, 2])
def test_cross_page_slot_isolation(ratio):
    pt = PageTable(2, 512, 8, 128)
    pool = PagedPool(pt, ratio, 3, dtype=torch.float32)
    pt.ensure(0, 257)
    pt.ensure(1, 257)
    count = 256 // ratio
    x = torch.arange(count * 3).reshape(count, 3).float()
    pool.write(0, 0, x)
    pool.write(1, 0, x + 10000)
    assert torch.equal(pool.read(0, 0, count), x)
    pt.release(0)
    pt.ensure(0, 257)
    pool.write(0, 0, -x)
    assert torch.equal(pool.read(1, 0, count), x + 10000)
    assert torch.equal(pool.read(0, 0, count), -x)


def test_bounds_and_unallocated_access():
    pt = PageTable(1, 256, 2, 128)
    pool = PagedPool(pt, 2, 3)
    with pytest.raises(RuntimeError):
        pool.read(0, 0, 1)
    for pos in (-1, 257):
        with pytest.raises(ValueError):
            pt.ensure(0, pos)
    with pytest.raises(IndexError):
        pt.ensure(1, 1)
