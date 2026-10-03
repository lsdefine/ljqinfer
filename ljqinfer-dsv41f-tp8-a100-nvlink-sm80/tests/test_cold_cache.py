import pytest
import torch
from strategy.cold_kv import ColdCache, Field


def make_cache(capacity=3):
    return ColdCache([Field(('k',), (2, 3), torch.float32)], 24 * capacity,
                     namespace='test')


def put(c, prefix, block, value):
    with c.lookup(prefix, namespace='test') as lease:
        with c.prepare(lease, block) as plan:
            plan.write(('k',), torch.full((2, 3), float(value)))
            plan.commit()


def test_atomic_publication_abort_and_validation():
    c = make_cache()
    with c.lookup([], namespace='test') as lease:
        with c.prepare(lease, [1, 2]) as plan:
            with pytest.raises(RuntimeError, match='incomplete'):
                plan.commit()
            with pytest.raises(ValueError):
                plan.write(('k',), torch.zeros(3))
            with pytest.raises(RuntimeError):
                lease.close()
            with c.lookup([1, 2], namespace='test') as miss:
                assert miss.token_count == 0
        assert c.used_bytes == 0
    with pytest.raises(RuntimeError):
        c.prepare(lease, [1, 2])
    with pytest.raises(ValueError):
        c.lookup([], namespace='other-model')
    assert not c.entries


def test_leased_leaves_cannot_be_evicted():
    c = make_cache(1)
    put(c, [], [1, 2], 5)
    with c.lookup([1, 2], namespace='test') as lease:
        with pytest.raises(MemoryError):
            with c.prepare(lease, [3, 4]) as plan:
                plan.write(('k',), torch.zeros(2, 3))
        with pytest.raises(RuntimeError):
            c.clear()
        assert torch.equal(c.read(lease, 0)[('k',)], torch.full((2, 3), 5.))
    put(c, [], [9, 8], 7)
    with c.lookup([1, 2], namespace='test') as miss:
        assert miss.token_count == 0
    assert len(c.entries) == 1
    assert c.used_bytes == 24


def test_branches_exact_tokens_and_read_ownership():
    c = make_cache()
    put(c, [], [1, 2], 1)
    put(c, [1, 2], [3, 4], 2)
    put(c, [1, 2], [3, 5], 3)
    for tokens, value in [([1, 2, 3, 4], 2), ([1, 2, 3, 5], 3)]:
        with c.lookup(tokens + [7], namespace='test') as lease:
            assert lease.token_count == 4
            got = c.read(lease, 1)[('k',)]
            assert (got == value).all()
            got.zero_()
            assert (c.read(lease, 1)[('k',)] == value).all()
    with c.lookup([1, 2, 3, 9], namespace='test') as lease:
        assert lease.token_count == 3
    c.clear()
    assert c.used_bytes == 0 and len(c.nodes) == 1


def test_single_writer_and_foreign_lease():
    c, other = make_cache(), make_cache()
    with other.lookup([], namespace='test') as foreign:
        with pytest.raises(RuntimeError):
            c.prepare(foreign, [1, 2])
    with c.lookup([], namespace='test') as lease:
        with c.prepare(lease, [1, 2]):
            with pytest.raises(RuntimeError):
                c.prepare(lease, [3, 4])


def test_interior_lease_pins_covering_branch_dependency():
    c = make_cache(2)
    put(c, [], [1, 2, 3, 4], 1)
    # The short branch depends on the longer backing entry, not a saved node 2.
    put(c, [1, 2], [9], 2)
    with c.lookup([1, 2, 9], namespace='test') as branch:
        assert branch.token_count == 3 and len(branch.ids) == 2
        assert [(a, b) for _, a, b in c.spans(branch)] == [(0, 2), (2, 3)]
        with pytest.raises(MemoryError):
            put(c, [], [7], 3)
    # Both unpinned leaves can now be reclaimed, in dependency order.
    put(c, [], [7], 3)
    with c.lookup([1, 2, 3], namespace='test') as prefix:
        assert prefix.token_count == 3
        assert [(a, b) for _, a, b in c.spans(prefix)] == [(0, 3)]
    put(c, [], [8], 4)  # evicts 7: prefix lookup refreshed the old backing
    put(c, [], [6], 5)  # now the original backing is the oldest
    with c.lookup([1, 2], namespace='test') as gone:
        assert gone.token_count == 0
    assert c.used_bytes == 48
