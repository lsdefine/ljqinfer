import pytest
import torch
from model.past import SlotPool
from model.cold import fields_for, store_chunk, restore_prefix
from strategy.cold_kv import ColdCache


def setup(budget=1000000):
    p = SlotPool(2, 32768).configure_default(kv_dim=4, index_dim=2)
    a = p.alloc()
    c = ColdCache(fields_for(p), budget, namespace='test')
    return p, a, c


def advance(p, a, end):
    p.ensure(a, end)
    p.set_pos(a, end)


def test_chunk_lease_chain_and_branch():
    p, a, c = setup()
    root = c.pin_endpoint(0, namespace='test')
    advance(p, a, 3)
    first = store_chunk(c, p, a, root, [1, 2, 3], namespace='test')
    root.close()
    advance(p, a, 5)
    left = store_chunk(c, p, a, first, [4, 5], namespace='test')
    right = store_chunk(c, p, a, first, [6, 7], namespace='test')
    first.close()
    with c.lookup([1, 2, 3, 4, 5], namespace='test') as found:
        assert found.ids == left.ids and found.token_count == 5
    with c.lookup([1, 2, 3, 6, 8], namespace='test') as partial:
        assert partial.matched_tokens == 4 and partial.token_count == 4
    b = p.alloc()
    assert restore_prefix(c, p, b, right) == 5
    with pytest.raises(RuntimeError):
        c.clear()
    left.close()
    right.close()
    c.clear()


@pytest.mark.parametrize('tokens,end,namespace', [([], 0, 'test'),
    (range(12289), 12289, 'test'), ([1], 2, 'test'), ([1], 1, 'other')])
def test_invalid_chunk_keeps_parent(tokens, end, namespace):
    p, a, c = setup()
    with c.pin_endpoint(0, namespace='test') as parent:
        advance(p, a, end)
        with pytest.raises(ValueError):
            store_chunk(c, p, a, parent, tokens, namespace=namespace)
        assert not parent.closed and not c.entries and c.active is None


def test_chunk_oom_keeps_previous_endpoint():
    p, a, c = setup()
    with c.pin_endpoint(0, namespace='test') as root:
        advance(p, a, 3)
        parent = store_chunk(c, p, a, root, [1, 2, 3], namespace='test')
    c.budget_bytes = c.used_bytes
    advance(p, a, 4)
    with pytest.raises(MemoryError):
        store_chunk(c, p, a, parent, [4], namespace='test')
    assert c.active is None and not parent.closed
    b = p.alloc()
    assert restore_prefix(c, p, b, parent) == 3
    old = parent.ids[-1]
    parent.close()
    c.clear()
    with pytest.raises(KeyError):
        c.pin_endpoint(old, namespace='test')


def test_chunk_rejects_closed_and_foreign_leases():
    p, a, c = setup()
    _, _, other = setup()
    advance(p, a, 1)
    parent = c.pin_endpoint(0, namespace='test')
    parent.close()
    with pytest.raises(RuntimeError):
        store_chunk(c, p, a, parent, [1], namespace='test')
    with other.pin_endpoint(0, namespace='test') as foreign:
        with pytest.raises(RuntimeError):
            store_chunk(c, p, a, foreign, [1], namespace='test')
