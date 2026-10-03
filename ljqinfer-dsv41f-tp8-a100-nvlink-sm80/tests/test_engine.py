"""Compute-only mock exercises the real Strategy/ModelExecution entrypoints."""
import os
from threading import Event
import pytest
import torch
from model.past import SlotPool
from model.model_api import ModelExecution, PrefillCancelled
from model.cold import fields_for
from strategy.cold_kv import ColdCache
from strategy.strategy import Strategy


class MockCompute:
    def __init__(self):
        self.calls = []
        self.fail_at = None
        self.cancel_during = None

    def forward(self, tokens, *, past, slot, start, history_tokens=()):
        self.calls.append((slot, start, len(tokens)))
        if self.fail_at == start:
            past.windows[0].main_kv[slot].fill_(-999)
            raise RuntimeError('injected compute failure')
        dim = past.windows[0].main_kv.shape[-1]
        prev = past.windows[0].read(slot, start-1, start)[0, 0].float() if start else 0
        x = (torch.tensor(tokens, device=past.device).cumsum(0) + prev) % 127
        raw = x[:, None].float() + torch.arange(dim, device=past.device)[None, :] % 7
        end = start + len(tokens)
        for layer, source in past.sources.items():
            values = raw + layer
            r = source.ratio
            if start % r:
                values = torch.cat([source.kv_state[slot, :1], values])
            n = end // r - start // r
            pooled = values[:n*r].reshape(n, r, dim).sum(1)
            source.ckv_pool.write(slot, start // r, pooled)
            source.index_pool.write(slot, start // r, pooled[:, :source.index_pool.data.shape[-1]])
            if r > 1 and end % r:
                source.kv_state[slot, 0].copy_(values[-1])
                source.score_state[slot, 0].fill_(1)
        for layer, w in past.windows.items():
            w.write(slot, start, raw + layer)
        if self.cancel_during is not None:
            self.cancel_during.set()
        return int(x[-1].item())

    def replay(self, tokens, *, past, slot, start, history_tokens):
        # This mock recurrence can recover its running sum from ratio-1 global
        # rows. Real model bounded replay is approximate, unlike this toy oracle.
        self.replays = getattr(self, 'replays', []) + [(slot, start, len(tokens), history_tokens)]
        end = start + len(tokens)
        source = past.sources[20]
        assert source.ratio == 1
        raw = source.ckv_pool.read(slot, start, end).float() - 20
        for layer, w in past.windows.items():
            w.write(slot, start, raw + layer)
        for layer, source in past.sources.items():
            n = end % source.ratio
            if n:
                source.kv_state[slot, :n].copy_(raw[-n:] + layer)
                source.score_state[slot, :n].fill_(1)


def oracle(p, slot, tokens):
    # Independent full-sequence recurrence, never reads restored state as truth.
    x = torch.tensor(tokens, dtype=torch.int64).cumsum(0) % 127
    dim = p.windows[0].main_kv.shape[-1]
    n = len(tokens)
    def rows(a, b, layer):
        return (x[a:b, None] + torch.arange(dim)[None, :] % 7 + layer).float()
    for layer, s in p.sources.items():
        r = s.ratio
        for a in range(0, n // r, 12288):
            b = min(n // r, a + 12288)
            expected = rows(a*r, b*r, layer).reshape(b-a, r, dim).sum(1)
            for field in [s.ckv_pool, s.index_pool]:
                got = field.read(slot, a, b).cpu()
                assert torch.equal(got, expected[:, :got.shape[-1]].to(got.dtype))
        if n % r:
            assert torch.equal(s.kv_state[slot, 0].cpu(), rows(n-1, n, layer)[0])
            assert torch.all(s.score_state[slot, 0] == 1)
    for layer, w in p.windows.items():
        a = max(0, n-p.ring)
        got = w.read(slot, a, n).cpu()
        assert torch.equal(got, rows(a, n, layer).to(got.dtype))
    assert p.pos[slot] == n
    return int(x[-1])


def engine(device='cpu', *, chunk=127, cold=True, budget=64*2**20,
           max_seq=32768, pool_tokens=65536, dim=4, index_dim=2):
    p = SlotPool(4, max_seq, pool_tokens=pool_tokens, device=device).configure_default(
        kv_dim=dim, index_dim=index_dim, window_dtype=torch.bfloat16,
        ckv_dtype=torch.bfloat16, index_dtype=torch.bfloat16)
    m = MockCompute()
    c = ColdCache(fields_for(p), budget, namespace='mock', pin_memory=device=='cuda') if cold else None
    return Strategy(ModelExecution(m, p, prefill_chunk_tokens=chunk), c, namespace='mock')


def clean(s):
    p = s.model.past
    assert len(p.free_slots) == p.n_slots
    assert len(p.pt._free) == p.pt.n_pages
    if s.cold_kv is not None:
        assert s.cold_kv.active is None
        assert all(e.pins == 0 for e in s.cold_kv.entries.values())


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('chunk', [1, 127, 12288])
def test_engine_roundtrip(device, chunk):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA required')
    s = engine(device, chunk=chunk)
    tokens = [i % 11 for i in range(17 if chunk == 1 else 24583)]
    with s.open(tokens) as q:
        assert q.hit_tokens == 0
        assert q.run() == oracle(s.model.past, q.slot, tokens)
        assert all(n <= chunk for _, _, n in s.model.compute.calls)
    calls = len(s.model.compute.calls)
    with s.open(tokens) as q:
        assert q.hit_tokens == max(0, len(tokens) - s.model.past.ring)
        assert q.run() == oracle(s.model.past, q.slot, tokens)
        assert all(n <= s.model.past.ring for _, _, n in s.model.compute.calls[calls:])
        q.extend([3, 7])
        assert q.run() == oracle(s.model.past, q.slot, tokens + [3, 7])
    clean(s)


def test_partial_branch_and_interleaved_duplicate():
    s = engine(chunk=3)
    with s.open([1,2,3,4,5,6,7]) as a, s.open([1,2,3,4,5,6,7]) as b:
        while not a.done:
            a.step()
            b.step()
        assert a.output == b.output == 28
        assert len(s.cold_kv.entries) == 3
    with s.open([1,2,3,4,9]) as q:
        assert q.run() == oracle(s.model.past, q.slot, [1,2,3,4,9])
    clean(s)


@pytest.mark.parametrize('cold', [False, True])
def test_cancel_and_compute_failure(cold):
    s = engine(chunk=3, cold=cold)
    cancel = Event()
    q = s.open(range(7), cancel=cancel)
    q.step()
    s.model.compute.cancel_during = cancel
    with pytest.raises(PrefillCancelled):
        q.step()
    assert q.closed
    clean(s)
    cancel.clear()
    s.model.compute.cancel_during = None
    s.model.compute.fail_at = 3
    q = s.open(range(7))
    with pytest.raises(RuntimeError, match='compute failure'):
        q.run()
    clean(s)
    s.model.compute.fail_at = None
    with s.open(range(7)) as q:
        assert q.run() == oracle(s.model.past, q.slot, list(range(7)))
    clean(s)


@pytest.mark.parametrize('failure', ['pages', 'cold', 'slots', 'restore'])
def test_resource_failure_cleanup(failure):
    s = engine(chunk=2048, pool_tokens=2048 if failure in ('pages', 'restore') else 65536,
               budget=1 if failure == 'cold' else 64*2**20)
    if failure == 'slots':
        sessions = [s.open([1]) for _ in range(4)]
        with pytest.raises(MemoryError):
            s.open([2])
        for q in sessions:
            q.close()
    elif failure == 'restore':
        with s.open([1]*2048) as q:
            q.run()
        with s.open([2]) as occupied:
            occupied.step()
            with pytest.raises(MemoryError):
                s.open([1]*2048)
            assert not occupied.closed
    else:
        q = s.open([1]*4096)
        with pytest.raises(MemoryError):
            q.run()
        assert q.closed
    clean(s)


def test_precancel_limits_and_no_cache():
    s = engine(cold=False, chunk=3)
    event = Event()
    event.set()
    with pytest.raises(PrefillCancelled):
        s.open([1], cancel=event)
    for tokens in [[], [1]*32769]:
        with pytest.raises(ValueError):
            s.open(tokens)
    with s.open([1,2,3,4,5]) as q:
        assert q.run() == 15
        q.extend([6])
        assert q.run() == 21
    clean(s)


@pytest.mark.skipif(os.environ.get('V41_FULL_GPU') != '1', reason='opt-in 4Mi GPU pool')
def test_full_capacity_engine():
    assert torch.cuda.is_available()
    s = engine('cuda', chunk=12288, max_seq=2**20, pool_tokens=4*2**20,
               dim=512, index_dim=128, budget=20*2**30)
    # 4 different requests, interleaved through the REAL execution API.
    tokens = [[(i + k) % 11 for i in range(2**20)] for k in range(4)]
    sessions = [s.open(t) for t in tokens]
    try:
        while not all(q.done for q in sessions):
            for q in sessions:
                q.step()
        assert len(s.model.past.pt._free) == 0
        for q, t in zip(sessions, tokens):
            assert q.chunks == 86
            assert q.output == oracle(s.model.past, q.slot, t)
    finally:
        for q in sessions:
            q.close()
    clean(s)
    calls = len(s.model.compute.calls)
    # Full hit: only the sliding window may be recomputed.
    with s.open(tokens[0]) as q:
        assert q.hit_tokens == max(0, 2**20 - s.model.past.ring)
        assert q.run() == oracle(s.model.past, q.slot, tokens[0])
        # oracle consumed by the assert above
    assert all(n <= s.model.past.ring for _, _, n in s.model.compute.calls[calls:])
    # Odd saved endpoint at max_seq, then a divergent shorter request.
    branch = tokens[0][:12288] + [100, 101, 102]
    with s.open(branch) as q:
        assert q.hit_tokens == max(0, 12288 - s.model.past.ring)
        assert q.run() == oracle(s.model.past, q.slot, branch)
    with s.open(branch + [3, 4]) as q:
        assert q.hit_tokens == max(0, len(branch) - s.model.past.ring)
        assert q.run() == oracle(s.model.past, q.slot, branch + [3, 4])
    clean(s)
    s.cold_kv.clear()


def test_metrics_are_an_accounting_identity():
    # service._stats only copies these numbers, so drift here ships to users.
    # QueryMetrics sat unimplemented behind that copy once already.
    s = engine(chunk=127)
    tokens = tuple(range(1, 600))
    with s.open(tokens) as q:
        while q.step() is not None:
            pass
        m = q.metrics  # finish_prefill has no mock; timed on real weights
        assert m.input_tokens == len(tokens)
        assert m.cache_hit_tokens == q.hit_tokens
        assert m.prefill_tokens == q.computed_tokens
        assert m.chunks == q.chunks == len(m.chunk_seconds) == len(m.chunk_tokens)
        assert sum(m.chunk_tokens) == m.prefill_tokens
        # nothing may be computed twice or silently skipped
        assert m.cache_hit_tokens + m.prefill_tokens == m.input_tokens
        assert m.cache_stored_blocks == m.chunks
        assert m.cache_hit_rate == m.cache_hit_tokens / m.input_tokens
        assert m.model_prefill_tps > 0 and m.effective_prefill_tps > 0
        assert m.steady_prefill_tps > 0  # needs >= 2 chunks to mean anything
        assert m.strategy_seconds >= m.model_prefill_seconds
        cold_computed = m.prefill_tokens
    clean(s)
    with s.open(tokens) as q:
        while q.step() is not None:
            pass
        m = q.metrics
        # the cache path has to be visible in the numbers, not just in behaviour
        assert m.cache_hit_tokens == q.hit_tokens > 0
        assert m.cache_lookup_seconds > 0 and m.cache_load_seconds > 0
        assert m.prefill_tokens < cold_computed
        assert m.cache_hit_tokens + m.prefill_tokens == m.input_tokens
    clean(s)
