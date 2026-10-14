"""Shared-prefix forks must remain independent LRU leaves under pressure."""
import random
from types import SimpleNamespace

import pytest
import torch

from model.glm53_cache import PrefixState
from strategy.cold_kv import ColdCache, Field, CacheCapacityError
from strategy.host_arena import HostArena


KEYS = ('target', 'draft')
ROWS = 80
WIDTH = 256


def payload(tokens, key):
    # Independent full-prefix oracle; values depend on all preceding tokens.
    x = torch.tensor(tokens, dtype=torch.int64).cumsum(0)
    return ((x[:, None] + torch.arange(WIDTH) + KEYS.index(key) * 17) % 251).to(torch.uint8)


def state_for(capacity_rows=160, arena=True):
    budget = capacity_rows * WIDTH * len(KEYS)
    state = object.__new__(PrefixState)
    state.namespace = 'branch-lru-test'
    state.cold = ColdCache([Field(k, (None, WIDTH), torch.uint8) for k in KEYS],
                           budget, namespace=state.namespace,
                           shm=HostArena(budget, segment_bytes=budget, pin_memory=False) if arena else None)
    state.fields = {}
    return state


def store(state, tokens):
    state.fields = {k: payload(tokens, k) for k in KEYS}
    state._store(tokens)


def check(state, tokens, expected_hit=None):
    c = state.cold
    with c.lookup(tokens, namespace=state.namespace) as lease:
        if expected_hit is not None:
            assert lease.token_count == expected_hit
        spans = c.spans(lease)
        assert lease.ids == tuple(i for i, _, _ in spans)
        for key in KEYS:
            parts = [c.storage[key][i][:end-start] for i, start, end in spans]
            actual = torch.cat(parts) if parts else torch.empty((0, WIDTH), dtype=torch.uint8)
            assert torch.equal(actual, payload(tokens[:lease.token_count], key))
    assert all(e.pins == 0 for e in c.entries.values())
    assert c.used_bytes == sum(e.nbytes for e in c.entries.values())
    assert c.used_bytes <= c.budget_bytes
    if c.shm:
        assert c.shm.used_bytes <= c.shm.capacity


@pytest.mark.parametrize('arena', [False, True])
def test_shared_prefix_fanout_survives_continuous_eviction(arena):
    state = state_for(arena=arena)
    root = [1] * 16 + [100] * 64
    store(state, root)
    first = next(iter(state.cold.entries))
    for n in range(500):
        tokens = [1] * 16 + [1000+n] * 64
        store(state, tokens)
        check(state, tokens, ROWS)
        check(state, [1] * 16 + [-1] * 64, 16)
        assert len(state.cold.entries) <= 2
        for entry in state.cold.entries.values():
            if entry.id != first:
                assert entry.parent == first
    state.cold.clear()
    assert state.cold.used_bytes == 0
    if state.cold.shm:
        assert state.cold.shm.used_bytes == 0


def test_nested_forks_retain_only_contributing_chain():
    state = state_for(capacity_rows=320)
    a = list(range(80))
    b = a[:16] + [100] * 64
    c = b[:40] + [200] * 40
    for tokens in (a, b, c):
        store(state, tokens)
    for end, chain_length in [(0, 0), (8, 1), (16, 1), (30, 2), (40, 2), (60, 3), (80, 3)]:
        with state.cold.lookup(c[:end], namespace=state.namespace) as lease:
            assert lease.token_count == end
            assert len(lease.ids) == chain_length
        check(state, c[:end], end)
    # Branch before both previous fork points: do not attach to b or c.
    d = a[:8] + [300] * 72
    store(state, d)
    for tokens in (a, b, c, d):
        check(state, tokens, 80)


@pytest.mark.parametrize('arena', [False, True])
def test_random_forks_eviction_restore_matches_independent_oracle(arena):
    state = state_for(capacity_rows=1200, arena=arena)
    rng = random.Random(51012)
    history = [list(range(80))]
    store(state, history[0])
    for n in range(600):
        parent = rng.choice(history[-40:])
        cut = rng.randrange(0, 80)
        tokens = parent[:cut] + [1000+n] * (80-cut)
        store(state, tokens)
        history.append(tokens)
        check(state, tokens, 80)
        for old in rng.sample(history, min(3, len(history))):
            check(state, old)
    state.cold.clear()


def test_live_reader_remains_protected_then_eviction_recovers():
    state = state_for()
    a = [1] * 16 + [100] * 64
    b = [1] * 16 + [200] * 64
    c = [1] * 16 + [300] * 64
    store(state, a)
    store(state, b)
    with state.cold.lookup(b, namespace=state.namespace) as reader:
        with pytest.raises(CacheCapacityError):
            store(state, c)
        assert state.cold.active is None
        assert reader.token_count == 80
        for key in KEYS:
            parts = [state.cold.storage[key][i][:end-start]
                     for i, start, end in state.cold.spans(reader)]
            assert torch.equal(torch.cat(parts), payload(b, key))
    store(state, c)
    check(state, c, 80)
    state.cold.clear()


def test_exact_suffix_endpoint_and_clipped_lease_handoff():
    state = state_for(capacity_rows=320)
    tokens = list(range(80))
    store(state, tokens)
    c = state.cold
    with c.lookup(tokens[:16], namespace=state.namespace) as parent:
        with c.lookup_suffix(parent, tokens[16:], namespace=state.namespace) as exact:
            assert exact.token_count == 80
            assert c.spans(exact) == [(exact.ids[0], 0, 80)]
        branch = tokens[:16] + [101] * 64
        store(state, branch)
        with c.lookup_suffix(parent, branch[16:], namespace=state.namespace) as exact:
            assert exact.token_count == 80
            assert len(exact.ids) == 2
    check(state, tokens, 80)
    check(state, branch, 80)
