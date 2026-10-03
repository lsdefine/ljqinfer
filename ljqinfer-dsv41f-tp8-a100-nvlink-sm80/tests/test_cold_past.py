import pytest
import torch
from model.past import SlotPool
from model.cold import fields_for, store_prefix, restore_prefix
from strategy.cold_kv import ColdCache


def pool():
    return SlotPool(2, 768, page_tokens=128).configure_default(kv_dim=4, index_dim=2)


def rows(start, end, dim, offset):
    return torch.arange(start, end).float()[:, None].expand(-1, dim) + offset


def append(p, slot, count):
    start, end = p.pos[slot], p.pos[slot] + count
    p.ensure(slot, end)
    for layer, w in p.windows.items():
        # Segment writes explicitly: reference rings do not retain old chunks.
        for a in range(start, end, 128):
            w.write(slot, a, rows(a, min(a + 128, end), 4, layer * 1000))
    projected = {}
    for layer, s in p.sources.items():
        a, b = start // s.ratio, end // s.ratio
        s.write_ckv(slot, a, rows(a, b, 4, layer * 1000))
        s.write_index_k(slot, a, rows(a, b, 2, layer * 2000))
        if s.ratio > 1:
            kv, score = rows(start, end, 4, layer * 100), rows(start, end, 4, -layer * 100)
            s.replay_partial(slot, start, count, kv, score)
            projected[layer] = (kv, score)
    p.set_pos(slot, end)
    return projected


def replay_hot(p, slot):
    # Deterministic toy projections, not a claim of exact real-model replay.
    end = p.pos[slot]
    start = max(0, end - p.ring)
    assert slot in p.replay_pending
    for layer, w in p.windows.items():
        w.write(slot, start, rows(start, end, 4, layer * 1000))
    for layer, source in p.sources.items():
        if source.ratio > 1:
            source.replay_partial(slot, start, end-start,
                rows(start, end, 4, layer*100), rows(start, end, 4, -layer*100))
    p.replay_pending.remove(slot)


def equal(a, sa, b, sb):
    end = a.pos[sa]
    assert b.pos[sb] == end
    for layer, s in a.sources.items():
        other = b.sources[layer]
        assert torch.equal(s.ckv(sa, end), other.ckv(sb, end))
        assert torch.equal(s.index_k(sa, end), other.index_k(sb, end))
        if s.ratio > 1:
            # Replay reconstructs the canonical ring, not a packed carry.
            assert torch.equal(s.kv_state[sa], other.kv_state[sb])
            assert torch.equal(s.score_state[sa], other.score_state[sb])
    for layer, w in a.windows.items():
        assert torch.equal(w.read(sa, max(0, end - 128), end),
                           b.windows[layer].read(sb, max(0, end - 128), end))


@pytest.mark.parametrize('blocks', [1, 2, 4])
@pytest.mark.parametrize('accepted', [0, 1, 3, 5])
def test_restore_continue_and_reject(blocks, accepted):
    original, restored = pool(), pool()
    source, dest = original.alloc(), restored.alloc()
    fields = fields_for(original)
    cache = ColdCache(fields, 4_000_000, namespace='weights:layout:rank0')
    tokens = tuple(range(128 * blocks))
    for block in range(blocks):
        append(original, source, 128)
        assert store_prefix(cache, original, source, tokens[:128 * (block + 1)],
                              namespace=cache.namespace)
    assert not store_prefix(cache, original, source, tokens, namespace=cache.namespace)
    # Restore at every historical boundary, not just the latest ring contents.
    for n in range(1, blocks + 1):
        with cache.lookup(tokens[:128*n], namespace=cache.namespace) as lease:
            assert restore_prefix(cache, restored, dest, lease) == 128*n
        replay_hot(restored, dest)
        assert (restored.windows[0].read(dest, 128*(n-1), 128*n)[:, 0]
                == torch.arange(128*(n-1), 128*n)).all()
        if n != blocks:
            restored.release(dest)
            dest = restored.alloc()
    equal(original, source, restored, dest)
    snap = restored.snapshot_verify(dest)
    projected = append(restored, dest, 5)
    # Execution reserves/writes speculative rows, but commit owns logical pos.
    restored.set_pos(dest, snap.pos)
    restored.commit_verify_prefix(snap, accepted, projected)
    append(original, source, accepted)
    equal(original, source, restored, dest)
    append(original, source, 7)
    append(restored, dest, 7)
    equal(original, source, restored, dest)



@pytest.mark.parametrize('end', [1, 2, 3, 63, 127, 128, 129, 255, 256, 257, 511])
@pytest.mark.parametrize('accepted', [0, 1, 2, 5])
def test_arbitrary_endpoint_roundtrip(end, accepted):
    a, b = pool(), pool()
    sa, sb = a.alloc(), b.alloc()
    c = ColdCache(fields_for(a), 4_000_000, namespace='test')
    append(a, sa, end)
    assert store_prefix(c, a, sa, range(end), namespace='test')
    assert len(c.entries) == 1  # no forced intermediate 128 checkpoints
    with c.lookup(range(end), namespace='test') as lease:
        assert lease.matched_tokens == lease.token_count == end
        assert restore_prefix(c, b, sb, lease) == end
    replay_hot(b, sb)
    equal(a, sa, b, sb)
    assert all(key[0] == 'sources' for key in c.storage)
    snap = b.snapshot_verify(sb)
    projected = append(b, sb, 5)
    b.set_pos(sb, end)
    b.commit_verify_prefix(snap, accepted, projected)
    append(a, sa, accepted)
    equal(a, sa, b, sb)
    append(a, sa, 3)
    append(b, sb, 3)
    equal(a, sa, b, sb)


def test_irregular_segments_and_partial_text_match():
    a = pool()
    sa = a.alloc()
    c = ColdCache(fields_for(a), 4_000_000, namespace='test')
    ends = [1, 2, 3, 129, 132, 257]
    for end in ends:
        append(a, sa, end - a.pos[sa])
        store_prefix(c, a, sa, range(end), namespace='test')
    # Source histories are incremental even across odd-to-even endpoints.
    for layer, source in a.sources.items():
        assert sum(t.shape[0] for t in c.storage[('sources', layer, 'main_ckv')].values()) == 257 // source.ratio
    for end in ends:
        expected, actual = pool(), pool()
        se, sd = expected.alloc(), actual.alloc()
        append(expected, se, end)
        with c.lookup(range(end), namespace='test') as lease:
            assert restore_prefix(c, actual, sd, lease) == end
        replay_hot(actual, sd)
        equal(expected, se, actual, sd)
    with c.lookup(list(range(131)) + [9999], namespace='test') as lease:
        assert lease.matched_tokens == 131
        assert lease.token_count == 131  # restore an interior prefix without a saved tail
    with c.lookup(range(17), namespace='test') as lease:
        assert lease.matched_tokens == 17 and lease.token_count == 17


def test_new_endpoint_inside_existing_edge():
    a = pool()
    sa = a.alloc()
    c = ColdCache(fields_for(a), 4_000_000, namespace='test')
    append(a, sa, 257)
    store_prefix(c, a, sa, range(257), namespace='test')
    b = pool()
    sb = b.alloc()
    append(b, sb, 131)
    store_prefix(c, b, sb, range(131), namespace='test')
    for end in [131, 257]:
        with c.lookup(range(end), namespace='test') as lease:
            assert lease.token_count == end
    assert len(c.entries) == 1  # shorter prefix already covered


@pytest.mark.parametrize('end', [1, 2, 127, 128, 129])
def test_real_nonoverlap_pooling_after_restore(end):
    a, b = pool(), pool()
    sa, sb = a.alloc(), b.alloc()
    c = ColdCache(fields_for(a), 4_000_000, namespace='test')
    append(a, sa, end)
    store_prefix(c, a, sa, range(end), namespace='test')
    with c.lookup(range(end), namespace='test') as lease:
        restore_prefix(c, b, sb, lease)
    replay_hot(b, sb)
    # Independent softmax pooling oracle for the next completed r=2 group.
    # Deliberately poison every dead row to prove restored carry is sufficient.
    for layer, source in b.sources.items():
        if source.ratio != 2:
            continue
        first = end - end % source.ratio
        stop = first + source.ratio
        live = torch.arange(first, end) % (2 * source.ratio)
        dead = torch.ones(2 * source.ratio, dtype=torch.bool)
        dead[live] = False
        source.kv_state[sb, dead] = float('nan')
        source.score_state[sb, dead] = float('nan')
        actual, group_start = source.fold(
            sb, end, rows(end, stop, 4, layer * 100),
            rows(end, stop, 4, -layer * 100), write=False)
        assert group_start == first // source.ratio
        assert actual.shape == (1, 4)
        actual = actual[0]
        kv = rows(first, stop, 4, layer * 100)
        scores = rows(first, stop, 4, -layer * 100)
        expected = (kv * scores.softmax(dim=0)).sum(dim=0)
        assert torch.equal(actual, expected)
    accounted = sum(t.numel() * t.element_size() for values in c.storage.values() for t in values.values())
    assert c.used_bytes == accounted <= c.budget_bytes
