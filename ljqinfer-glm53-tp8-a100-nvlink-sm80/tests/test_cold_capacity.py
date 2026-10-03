"""Bounded cold-cache admission: alignment, fragmentation, leases and rollback."""
import torch
import pytest
from strategy.cold_kv import ColdCache, Field, CacheCapacityError
from strategy.host_arena import HostArena


def cache(cap=2048, segment=1024, fields=('kv',)):
    arena = HostArena(cap, segment_bytes=segment, pin_memory=False)
    c = ColdCache([Field(k, (None,), torch.uint8) for k in fields],
                  cap, namespace='test', shm=arena)
    return c, arena


def put(c, token, size=1):
    with c.lookup([token], namespace='test') as lease:
        with c.prepare(lease, [token]) as store:
            for key in c.fields:
                store.write(key, torch.full((size,), token % 251, dtype=torch.uint8))
            return store.commit()


def test_alignment_churn():
    c, arena = cache()
    for i in range(500):
        put(c, i)
        with c.lookup([i], namespace='test') as lease:
            assert lease.token_count == 1
            assert c.read(lease, 0)['kv'].item() == i % 251
        assert arena.used_bytes <= arena.capacity
        assert len(c.entries) <= 4
    c.clear()
    assert arena.used_bytes == c.used_bytes == 0


def test_fragmentation_needs_eviction_below_logical_budget():
    c, arena = cache(4096, 2048)
    for i in range(8):
        put(c, i)
    # Touch odd entries: oldest leaves are now alternate physical extents.
    for i in [1, 3, 5, 7]:
        with c.lookup([i], namespace='test'):
            pass
    for _ in range(4):
        c._evict_one()
    assert arena.used_bytes == 2048
    assert all(size == 512 for holes in arena._free for _, size in holes)
    put(c, 99, 1024)
    with c.lookup([99], namespace='test') as lease:
        assert torch.equal(c.read(lease, 0)['kv'], torch.full((1024,), 99, dtype=torch.uint8))
    c.clear()
    assert arena.used_bytes == 0


def test_pinned_entries_survive_and_retry_after_release():
    c, arena = cache(1024, 1024)
    put(c, 1); put(c, 2)
    a = c.lookup([1], namespace='test'); b = c.lookup([2], namespace='test')
    with pytest.raises(CacheCapacityError):
        put(c, 3)
    assert c.active is None and arena.used_bytes == 1024
    assert c.read(a, 0)['kv'].item() == 1
    assert c.read(b, 0)['kv'].item() == 2
    a.close(); b.close()
    put(c, 3)
    c.clear()
    assert arena.used_bytes == 0


def test_partial_staging_aborts_without_leak_or_publication():
    c, arena = cache(1024, 1024, fields=('a', 'b', 'c'))
    for i in range(8):
        with pytest.raises(CacheCapacityError):
            put(c, i)
        assert c.active is None and not c.entries
        assert arena.used_bytes == c.used_bytes == 0
        with c.lookup([i], namespace='test') as lease:
            assert lease.token_count == 0


def test_real_allocation_error_is_not_hidden(monkeypatch):
    c, arena = cache()
    def fail(_):
        raise MemoryError('injected real allocation failure')
    monkeypatch.setattr(arena, 'alloc', fail)
    with pytest.raises(MemoryError, match='injected real') as exc:
        put(c, 1)
    assert not isinstance(exc.value, CacheCapacityError)
    assert c.active is None


def test_parent_chain_pinned_during_extension():
    c, arena = cache(1536, 1536)
    put(c, 10)
    for tokens in [[10, 11], [10, 11, 12]]:
        with c.lookup(tokens, namespace='test') as parent:
            with c.prepare(parent, tokens[parent.token_count:]) as store:
                store.write('kv', torch.tensor([tokens[-1]], dtype=torch.uint8))
                store.commit()
    with c.lookup([10, 11, 12, 13], namespace='test') as parent:
        with pytest.raises(CacheCapacityError):
            with c.prepare(parent, [13]) as store:
                store.write('kv', torch.tensor([13], dtype=torch.uint8))
                store.commit()
        assert [int(c.read(parent, i)['kv'].item()) for i in range(3)] == [10, 11, 12]
    put(c, 99)
    c.clear()
    assert arena.used_bytes == 0


def test_cuda_async_pressure_recovers(caplog):
    from types import SimpleNamespace
    from strategy.async_writeback import AsyncWriteback
    from model.glm53_cache import PrefixState
    from model.glm53_cold_transfer import ColdTransfer
    device = torch.device('cuda:0')
    state = PrefixState.__new__(PrefixState)
    state.engine = SimpleNamespace(device=device, length=128, pending=None,
                                  mtp_kv=SimpleNamespace(lengths=[128]))
    state.namespace = 'pressure-test'
    state.resident = []
    # 128 rows need 1536 aligned bytes and fail midway; 32 need only 1024.
    source = torch.arange(128, dtype=torch.float32, device=device)
    state.fields = {'a': source, 'b': source + 1000}
    state.transfer = ColdTransfer(state.fields, device)
    state.arena = HostArena(1024, segment_bytes=1024)
    state.cold = ColdCache([Field(k, (None,), torch.float32) for k in state.fields],
                           1024, namespace=state.namespace, shm=state.arena)
    state.copy_stream = torch.cuda.Stream(device=device)
    state.writer = AsyncWriteback(2)
    tokens = list(range(128))
    try:
        # Hold one full allocation as a reader: no eviction may take it.
        state.engine.length = state.engine.mtp_kv.lengths[0] = 32
        state.publish(tokens[:32]); state.drain()
        lease = state.cold.lookup(tokens[:32], namespace=state.namespace)
        state.engine.length = state.engine.mtp_kv.lengths[0] = 64
        state.publish(tokens[:64]); state.drain()
        assert 'cold cache admission skipped' in caplog.text
        assert state.resident == tokens[:64]
        assert torch.equal(state.fields['a'], source)
        assert lease.token_count == 32 and state.cold.active is None
        lease.close()
        # Different prefix can evict and publish after the refusal.
        state.engine.length = state.engine.mtp_kv.lengths[0] = 32
        other = list(range(1000, 1032))
        state.publish(other); state.drain()
        with state.cold.lookup(other, namespace=state.namespace) as lease:
            assert lease.token_count == 32
            for key in state.fields:
                assert torch.equal(state.cold.read(lease, 0)[key], state.fields[key][:32].cpu())
        state.clear()
        assert state.arena.used_bytes == 0
        # Logical budget is larger than arena: actual failure after first D2H.
        state.cold.budget_bytes = 4096
        state.engine.length = state.engine.mtp_kv.lengths[0] = 128
        # Three fields force partial physical staging to fail.
        extra = source + 2000
        state.fields['c'] = extra
        state.cold.fields['c'] = Field('c', (None,), torch.float32)
        state.cold.storage['c'] = {}
        state.publish(tokens); state.drain()
        assert state.cold.active is None and state.arena.used_bytes == 0
        assert not state.cold.entries
        state._store = lambda tokens: (_ for _ in ()).throw(ValueError('real fault'))
        state.publish(tokens)
        with pytest.raises(ValueError, match='real fault'):
            state.drain()
    finally:
        state.close()
