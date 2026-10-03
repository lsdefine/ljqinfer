# CPU leaf tests for the DSV4.1 shared-source past contract.
import torch

from model.past import (
    INDEX_SOURCE_LAYERS, KV_SOURCE_LAYERS, KV_SOURCE_RATIOS,
    N_ATTENTION_LAYERS, WINDOW_TOKENS, DECODE_BAND, SlotPool, default_layer_views,
)


def make_pool(n_slots=2, max_seq=512):
    return SlotPool(
        n_slots, max_seq, pool_tokens=n_slots * max_seq,
        ring=WINDOW_TOKENS + DECODE_BAND, page_tokens=128,
    ).configure_default(
        kv_dim=4, index_dim=2,
        window_dtype=torch.float32,
        ckv_dtype=torch.float32,
        index_dtype=torch.float32,
    )


def test_layer_views_match_csa2_modes():
    views = default_layer_views()
    assert len(views) == 43
    assert views[0].mode == views[1].mode == "swa"
    assert [i for i, v in views.items() if v.mode == "full"] == list(KV_SOURCE_LAYERS)
    assert [i for i, v in views.items() if v.mode == "reindex"] == [24, 28, 32, 36]
    assert [i for i, v in views.items() if v.mode == "dspark"] == [40, 41, 42]
    assert views[7].kv_source_layer == 2 and views[7].index_source_layer == 2
    assert views[19].kv_source_layer == 14 and views[19].ratio == 2
    assert views[27].kv_source_layer == 20 and views[27].index_source_layer == 24
    assert set(INDEX_SOURCE_LAYERS) == {2, 8, 14, 20, 24, 28, 32, 36}


def test_storage_is_four_sources_plus_thin_windows():
    pool = make_pool()
    assert set(pool.sources) == set(KV_SOURCE_LAYERS)
    assert len(pool.windows) == N_ATTENTION_LAYERS
    for layer, source in pool.sources.items():
        assert source.ratio == KV_SOURCE_RATIOS[layer]
        assert not hasattr(source, "res_x")
        assert not hasattr(source, "derived")
    assert pool.sources[20].kv_state is None
    assert pool.sources[2].score_state.shape == (2, 4, 4)


def fill(pool, slot, pos):
    pool.ensure(slot, pos)
    for layer, source in pool.sources.items():
        n = pos // source.ratio
        base = layer * 10000
        source.write_ckv(slot, 0, torch.arange(n * 4).view(n, 4) + base)
        source.write_index_k(slot, 0, torch.arange(n * 2).view(n, 2) + base + 5000)
        if source.ratio == 2:
            source.kv_state[slot].copy_(torch.arange(16).view(4, 4) + base + 100)
            source.score_state[slot].copy_(torch.arange(16).view(4, 4) + base + 200)
    for layer, window in pool.windows.items():
        value = torch.arange(128 * 4).view(128, 4) + layer * 1000
        window.write(slot, pos - 128, value)
    pool.set_pos(slot, pos)


def assert_equal(a, sa, b, sb, pos):
    for layer in KV_SOURCE_LAYERS:
        x, y = a.sources[layer], b.sources[layer]
        assert torch.equal(x.ckv(sa, pos), y.ckv(sb, pos))
        assert torch.equal(x.index_k(sa, pos), y.index_k(sb, pos))
        if x.ratio == 2:
            assert torch.equal(x.kv_state[sa], y.kv_state[sb])
            assert torch.equal(x.score_state[sa], y.score_state[sb])
    for layer in range(N_ATTENTION_LAYERS):
        assert torch.equal(
            a.windows[layer].read(sa, pos - 128, pos),
            b.windows[layer].read(sb, pos - 128, pos),
        )


def test_chunked_cold_roundtrip():
    a, b = make_pool(), make_pool()
    sa, sb = a.alloc(), b.alloc()
    fill(a, sa, 256)
    segments = [a.export_cold(sa, 0, 128), a.export_cold(sa, 128, 256)]
    assert all("tail" not in segment for segment in segments)
    assert b.import_cold(sb, segments) == 256
    assert sb in b.replay_pending
    for layer, x in a.sources.items():
        y = b.sources[layer]
        assert torch.equal(x.ckv(sa, 256), y.ckv(sb, 256))
        assert torch.equal(x.index_k(sa, 256), y.index_k(sb, 256))
    assert torch.count_nonzero(b.windows[0].main_kv[sb]) == 0


def test_verify_prefix_repairs_only_partial_state():
    pool = make_pool(1, 256)
    slot = pool.alloc()
    pool.ensure(slot, 128)
    pool.set_pos(slot, 101)  # ratio-2 sources already carry token 100 in row 0
    for source in pool.sources.values():
        if source.ratio == 2:
            source.kv_state[slot, 0].fill_(7)
            source.score_state[slot, 0].fill_(8)
    snap = pool.snapshot_verify(slot)
    projected = {}
    for layer in (2, 8, 14):
        kv = torch.arange(20).view(5, 4).float() + layer * 100
        score = kv + 1000
        projected[layer] = (kv, score)

    assert pool.commit_verify_prefix(snap, 1, projected) == 102
    for layer in (2, 8, 14):
        source = pool.sources[layer]
        assert torch.equal(source.kv_state[slot, 0], torch.full((4,), 7.0))
        assert torch.equal(source.kv_state[slot, 1], projected[layer][0][0])

    # Re-applying the same snapshot must not accumulate rejected rows.
    pool.set_pos(slot, snap.pos)
    assert pool.commit_verify_prefix(snap, 4, projected) == 105
    for layer in (2, 8, 14):
        source = pool.sources[layer]
        # accepted abs positions 101..104: final open group contains token 104 in row 0
        assert torch.equal(source.kv_state[slot, 0], projected[layer][0][3])
        assert torch.equal(source.score_state[slot, 0], projected[layer][1][3])


def test_release_resets_authoritative_state():
    pool = make_pool(1, 256)
    slot = pool.alloc()
    pool.ensure(slot, 128)
    pool.set_pos(slot, 17)
    pool.sources[2].kv_state[slot].fill_(9)
    pool.release(slot)
    assert pool.pos[slot] == 0 and pool.pt.n_alloc(slot) == 0
    assert torch.count_nonzero(pool.sources[2].kv_state[slot]) == 0
    assert torch.isneginf(pool.sources[2].score_state[slot]).all()
    assert pool.alloc() == slot


def test_default_pool_budget_holds_every_slot_at_max_seq():
    """max_seq straddling a page boundary must not cost a slot its last page."""
    pool = SlotPool(3, 3 * 128 + 1, page_tokens=128).configure_default(
        kv_dim=2, index_dim=2)
    for slot in range(pool.n_slots):
        pool.pt.ensure(slot, pool.max_seq)
    assert [pool.pt.n_alloc(s) for s in range(3)] == [4, 4, 4]
