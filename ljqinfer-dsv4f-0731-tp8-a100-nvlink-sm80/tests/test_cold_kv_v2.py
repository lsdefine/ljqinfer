import pytest
import torch

from strategy.cold_kv_v2 import KVFieldSpec, PinnedMemoryKVCacheV2


def make_cache(blocks=4):
    fields = (
        KVFieldSpec(("layers", 0, "aux"), (2,), torch.uint8),
        KVFieldSpec(("tail", 0, "res_x"), (3,), torch.uint8),
    )
    block_bytes = sum(field.nbytes for field in fields)
    return PinnedMemoryKVCacheV2(
        fields,
        block_size=4,
        initial_bytes=blocks * block_bytes,
        target_bytes=blocks * block_bytes,
        slab_blocks=2,
    )


def test_store_transaction_is_exclusive_and_abort_releases_it():
    cache = make_cache()
    first = cache.begin(range(4))
    plan = cache.prepare_store(first, range(4))

    with pytest.raises(RuntimeError, match="transaction already active"):
        cache.prepare_store(cache.begin(range(10, 14)), range(10, 14))
    with pytest.raises(RuntimeError, match="active store transaction"):
        cache.clear()

    plan.abort()
    second = cache.prepare_store(cache.begin(range(10, 14)), range(10, 14))
    result = second.commit()
    assert result.stored_blocks == 1
    assert cache.begin(range(10, 14)).token_count == 4
    assert cache.begin(range(4)).token_count == 0


def test_commit_releases_transaction_and_partial_block_is_not_stored():
    cache = make_cache()
    ids = tuple(range(6))
    plan = cache.prepare_store(cache.begin(ids), ids)
    assert plan.stored_blocks == 1
    plan.commit()

    assert cache.begin(ids).token_count == 4
    next_plan = cache.prepare_store(cache.begin(range(20, 24)), range(20, 24))
    next_plan.abort()


def test_field_major_write_and_restore_views_round_trip():
    cache = make_cache()
    ids = tuple(range(8))
    plan = cache.prepare_store(cache.begin(ids), ids)
    aux = torch.tensor([[1, 2], [3, 4]], dtype=torch.uint8)
    tail = torch.tensor([[5, 6, 7], [8, 9, 10]], dtype=torch.uint8)
    plan.write(("layers", 0, "aux"), 0, aux)
    plan.write(("tail", 0, "res_x"), 0, tail)
    plan.commit()

    match = cache.begin(ids)
    runs = cache.restore_runs(match)
    assert match.token_count == 8
    assert sum(run.block_count for run in runs) == 2
    got_aux = torch.cat([run.field(("layers", 0, "aux")).clone() for run in runs])
    got_tail = torch.cat([run.field(("tail", 0, "res_x")).clone() for run in runs])
    assert torch.equal(got_aux, aux)
    assert torch.equal(got_tail, tail)


def test_clear_invalidates_old_lookup_after_transaction_closes():
    cache = make_cache()
    lookup = cache.begin(range(4))
    plan = cache.prepare_store(lookup, range(4))
    plan.abort()
    cache.clear()
    with pytest.raises(RuntimeError, match="lookup was invalidated"):
        cache.prepare_store(lookup, range(4))
