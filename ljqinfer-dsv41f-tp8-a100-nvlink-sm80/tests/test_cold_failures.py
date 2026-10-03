import pytest
import torch
from strategy.cold_kv import ColdCache, Field
from model.past import SlotPool
from model.cold import fields_for, store_prefix, restore_prefix


def test_two_fields_must_both_complete():
    fields = [Field((name,), (2,), torch.float32) for name in ('a', 'b')]
    c = ColdCache(fields, 16, namespace='test')
    with c.lookup([], namespace='test') as lease:
        with c.prepare(lease, [1, 2]) as plan:
            plan.write(('a',), torch.ones(2))
            with pytest.raises(RuntimeError, match='incomplete'):
                plan.commit()
            with c.lookup([1, 2], namespace='test') as missing:
                assert missing.token_count == 0
            plan.write(('b',), torch.full((2,), 2.))
            plan.commit()
    with c.lookup([1, 2], namespace='test') as hit:
        assert hit.token_count == 2
        assert (c.read(hit, 0)[('b',)] == 2).all()


def test_restore_oom_does_not_publish_partial_position():
    src = SlotPool(1, 384, page_tokens=128).configure_default(kv_dim=2, index_dim=2)
    dst = SlotPool(1, 384, pool_tokens=128, page_tokens=128).configure_default(kv_dim=2, index_dim=2)
    a, b = src.alloc(), dst.alloc()
    fields = fields_for(src)
    c = ColdCache(fields, 2_000_000, namespace='test')
    for end in (128, 256):
        src.ensure(a, end)
        src.set_pos(a, end)
        store_prefix(c, src, a, range(end), namespace='test')
    with c.lookup(range(256), namespace='test') as lease:
        with pytest.raises(MemoryError):
            restore_prefix(c, dst, b, lease)
    assert dst.pos[b] == 0 and dst.pt.n_alloc(b) == 0
