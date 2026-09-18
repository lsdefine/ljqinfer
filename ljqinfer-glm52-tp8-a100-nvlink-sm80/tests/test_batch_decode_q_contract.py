#!/usr/bin/env python3
"""CPU contract for width-Q provisional decode and per-row rollback."""
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import model.batch_decode as batch_decode
from model.config import D, TP, VOCAB


class Cache:
    def __init__(self, length):
        self.length = length


def _state(q=3):
    caches = [Cache(5), Cache(7)]
    return batch_decode.BatchDecodeState(
        caches=caches,
        lengths=[cache.length for cache in caches],
        capacities=[20, 20],
        graph=SimpleNamespace(Q=q),
        k0s=[[object(), object()] for _ in range(TP)],
    )


def test_width_q_replay_and_independent_commit():
    state = _state(q=3)
    calls = []
    real_replay = batch_decode.replay_decode

    def replay(graph, replay_state, tokens):
        calls.append((graph.Q, tuple(tokens.shape), list(replay_state.lengths)))
        return (torch.zeros(2, graph.Q, VOCAB),
                torch.zeros(2, graph.Q, D))

    batch_decode.replay_decode = replay
    try:
        logits, hidden = batch_decode.decode_candidates(
            torch.tensor([[1, 2, 3], [4, 5, 6]]), state)
    finally:
        batch_decode.replay_decode = real_replay

    assert calls == [(3, (2, 3), [5, 7])]
    assert tuple(logits.shape) == (2, 3, VOCAB)
    assert tuple(hidden.shape) == (2, 3, D)
    assert state.lengths == [8, 10]
    assert [cache.length for cache in state.caches] == [8, 10]
    assert state._decode_pending

    batch_decode.commit_candidates(state, [1, 3])
    assert state.lengths == [6, 10]
    assert [cache.length for cache in state.caches] == [6, 10]
    assert not state._decode_pending


def test_contract_rejects_wrong_width_and_commit_count():
    state = _state(q=3)
    try:
        batch_decode.decode_candidates(torch.tensor([[1, 2], [3, 4]]), state)
    except ValueError as exc:
        assert "[2,3]" in str(exc)
    else:
        raise AssertionError("wrong query width was accepted")

    state._decode_pending = True
    try:
        batch_decode.commit_candidates(state, [0, 3])
    except ValueError as exc:
        assert "[1,3]" in str(exc)
    else:
        raise AssertionError("zero-token commit was accepted")


def test_mtp_step_restores_transaction_metadata_on_failure():
    q = batch_decode.Q_MAX
    state = _state(q=q)
    state.mtp_caches = [Cache(5), Cache(7)]
    originals = (list(state.lengths),
                 [cache.length for cache in state.caches],
                 [cache.length for cache in state.mtp_caches],
                 state._decode_pending)
    real_decode = batch_decode.decode_candidates
    real_sample = batch_decode._sample
    real_mtp = batch_decode.mtp_draft_chain_batched

    def decode(tokens, replay_state):
        replay_state.lengths[:] = [length + q for length in replay_state.lengths]
        for cache in replay_state.caches:
            cache.length += q
        replay_state._decode_pending = True
        return (torch.empty(2, q, 1), torch.empty(2, q, D))

    def sample(*_args):
        return ([[1], [2]], [3, 4], [[5], [6]], [1, 1])

    def fail_mtp(*_args, **_kwargs):
        raise RuntimeError("injected MTP failure")

    batch_decode.decode_candidates = decode
    batch_decode._sample = sample
    batch_decode.mtp_draft_chain_batched = fail_mtp
    try:
        try:
            batch_decode.mtp_step(None, state, [1, 2],
                                  [[3] * (q - 1), [4] * (q - 1)])
        except RuntimeError as exc:
            assert "injected MTP failure" in str(exc)
        else:
            raise AssertionError("injected MTP failure did not propagate")
    finally:
        batch_decode.decode_candidates = real_decode
        batch_decode._sample = real_sample
        batch_decode.mtp_draft_chain_batched = real_mtp

    restored = (state.lengths,
                [cache.length for cache in state.caches],
                [cache.length for cache in state.mtp_caches],
                state._decode_pending)
    assert restored == originals


def test_q6_prime_produces_full_draft_width():
    q = batch_decode.Q_MAX
    assert q == 6
    state = _state(q=q)
    state.mtp_caches = [Cache(5), Cache(7)]
    engine = SimpleNamespace(rt=None, w=SimpleNamespace(final_norm=torch.ones(D)))
    sequences = [torch.arange(5), torch.arange(7)]
    residuals = [torch.zeros(1, D), torch.zeros(1, D)]
    calls = []
    real_ensure = batch_decode.ensure_mtp_graphs
    real_lm = batch_decode.lm_head
    real_pick = batch_decode._pick_tokens
    real_rms = batch_decode._rmsnorm
    real_forward = batch_decode.mtp_forward
    real_recurse = batch_decode._recurse_drafts
    batch_decode.ensure_mtp_graphs = lambda _engine: None
    batch_decode.lm_head = lambda *_args: torch.tensor([0.0, 1.0])
    batch_decode._pick_tokens = lambda logits: torch.ones(
        logits.shape[:-1], dtype=torch.long)
    batch_decode._rmsnorm = lambda x, _w: x
    batch_decode.mtp_forward = lambda *_args, **_kwargs: (11, None, torch.zeros(1, D))
    def recurse(_engine, _cache, _token, _hidden, count):
        calls.append(count)
        return list(range(20, 20 + count))
    batch_decode._recurse_drafts = recurse
    try:
        base, drafts = batch_decode.batched_mtp_prime(
            engine, state, sequences, residuals)
    finally:
        batch_decode.ensure_mtp_graphs = real_ensure
        batch_decode.lm_head = real_lm
        batch_decode._pick_tokens = real_pick
        batch_decode._rmsnorm = real_rms
        batch_decode.mtp_forward = real_forward
        batch_decode._recurse_drafts = real_recurse
    assert base == [1, 1]
    assert calls == [q - 2, q - 2]
    assert all(len(row) == q - 1 for row in drafts)


def test_select_decode_rows_is_a_non_owning_view():
    graphs = {(b, batch_decode.Q_MAX): SimpleNamespace(B=b, Q=batch_decode.Q_MAX)
              for b in range(1, 5)}
    engine = SimpleNamespace(base_graphs=graphs)
    caches = [Cache(10 + row) for row in range(4)]
    mtp_caches = [Cache(20 + row) for row in range(4)]
    state = batch_decode.BatchDecodeState(
        caches=caches,
        lengths=[cache.length for cache in caches],
        capacities=[100 + row for row in range(4)],
        graph=graphs[(4, batch_decode.Q_MAX)],
        page_indices=[[row] for row in range(4)],
        k0s=[[object() for _ in range(4)] for _ in range(TP)],
        tables=[[object() for _ in range(4)] for _ in range(TP)],
        mtp_caches=mtp_caches,
    )

    selected = batch_decode.select_decode_rows(engine, state, [0, 2, 3])

    assert selected.graph is graphs[(3, batch_decode.Q_MAX)]
    assert selected.caches == [caches[0], caches[2], caches[3]]
    assert selected.mtp_caches == [mtp_caches[0], mtp_caches[2], mtp_caches[3]]
    assert selected.lengths == [10, 12, 13]
    assert selected.page_indices == [[0], [2], [3]]
    assert [cache.length for cache in caches] == [10, 11, 12, 13]
    assert [cache.length for cache in mtp_caches] == [20, 21, 22, 23]
    assert not state.released


def test_compaction_tracks_original_rows_and_keeps_batch_lease():
    graphs = {(b, batch_decode.Q_MAX): SimpleNamespace(B=b, Q=batch_decode.Q_MAX)
              for b in range(1, 5)}
    engine = SimpleNamespace(base_graphs=graphs)
    caches = [Cache(10) for _ in range(4)]
    mtp_caches = [Cache(10) for _ in range(4)]
    for row, cache in enumerate(caches):
        cache.row = row
    state = batch_decode.BatchDecodeState(
        caches=caches,
        lengths=[10] * 4,
        capacities=[100] * 4,
        graph=graphs[(4, batch_decode.Q_MAX)],
        page_indices=[[row] for row in range(4)],
        k0s=[[object() for _ in range(4)] for _ in range(TP)],
        tables=[[object() for _ in range(4)] for _ in range(TP)],
        mtp_caches=mtp_caches,
    )

    class PrefillState:
        released = False

        def decode_state(self, graph):
            assert graph is graphs[(4, batch_decode.Q_MAX)]
            self.released = True
            return state

        def release(self):
            raise AssertionError("decode owns the lease after decode_state")

    widths = []
    emitted = []
    retired = set()
    real_prefill = batch_decode.prefill_batch_chunked
    real_prime = batch_decode.batched_mtp_prime
    real_step = batch_decode.mtp_step

    def prefill(*args, **kwargs):
        return [object()] * 4, PrefillState()

    def prime(engine_arg, state_arg, sequences, residuals):
        return [1] * state_arg.batch_size, [[2] * (batch_decode.Q_MAX - 1)
                                            for _ in range(state_arg.batch_size)]

    def step(engine_arg, state_arg, base, draft):
        active = {cache.row for cache in state_arg.caches}
        assert all(caches[row].length > 0 for row in retired)
        retired.update(set(range(4)) - active)
        widths.append(state_arg.batch_size)
        assert state_arg.graph is graphs[(state_arg.batch_size, batch_decode.Q_MAX)]
        for cache in state_arg.caches:
            cache.length += 1
        state_arg.lengths[:] = [cache.length for cache in state_arg.caches]
        return ([[100 + cache.row] for cache in state_arg.caches],
                list(base), list(draft))

    batch_decode.prefill_batch_chunked = prefill
    batch_decode.batched_mtp_prime = prime
    batch_decode.mtp_step = step
    try:
        outputs = batch_decode.generate_mtp_batch(
            engine, [[1], [2], [3], [4]], max_new_tokens=[1, 2, 3, 4],
            eos_token_id=-1, emit=lambda row, chunk: emitted.append((row, chunk)),
            select_active_rows=lambda active, done:
                [row for row in active if not done[row]])
    finally:
        batch_decode.prefill_batch_chunked = real_prefill
        batch_decode.batched_mtp_prime = real_prime
        batch_decode.mtp_step = real_step

    assert widths == [4, 3, 2, 1]
    assert outputs == [[100], [101, 101], [102, 102, 102], [103, 103, 103, 103]]
    assert emitted == [(0, [100]), (1, [101]), (2, [102]), (3, [103]),
                       (1, [101]), (2, [102]), (3, [103]),
                       (2, [102]), (3, [103]), (3, [103])]
    assert [cache.length for cache in caches] == [0, 0, 0, 0]
    assert [cache.length for cache in mtp_caches] == [0, 0, 0, 0]


def test_active_row_policy_cannot_drop_a_live_row():
    graphs = {(b, batch_decode.Q_MAX): SimpleNamespace(B=b, Q=batch_decode.Q_MAX)
              for b in range(1, 3)}
    engine = SimpleNamespace(base_graphs=graphs)
    caches = [Cache(10), Cache(10)]
    for row, cache in enumerate(caches):
        cache.row = row
    state = batch_decode.BatchDecodeState(
        caches=caches, lengths=[10, 10], capacities=[100, 100],
        graph=graphs[(2, batch_decode.Q_MAX)], page_indices=[[0], [1]],
        k0s=[[object(), object()] for _ in range(TP)],
        tables=[[object(), object()] for _ in range(TP)],
        mtp_caches=[Cache(10), Cache(10)])

    class PrefillState:
        released = False

        def decode_state(self, graph):
            self.released = True
            return state

    real_prefill = batch_decode.prefill_batch_chunked
    real_prime = batch_decode.batched_mtp_prime
    real_step = batch_decode.mtp_step
    batch_decode.prefill_batch_chunked = lambda *args, **kwargs: (
        [object(), object()], PrefillState())
    batch_decode.batched_mtp_prime = lambda *args, **kwargs: (
        [1, 1], [[2] * (batch_decode.Q_MAX - 1) for _ in range(2)])
    batch_decode.mtp_step = lambda *args, **kwargs: (
        [[100], [101]], args[2], args[3])
    try:
        try:
            batch_decode.generate_mtp_batch(
                engine, [[1], [2]], max_new_tokens=[1, 2], eos_token_id=-1,
                select_active_rows=lambda active, done: [])
        except ValueError as exc:
            assert "dropped a live row" in str(exc)
        else:
            raise AssertionError("unsafe active-row policy was accepted")
    finally:
        batch_decode.prefill_batch_chunked = real_prefill
        batch_decode.batched_mtp_prime = real_prime
        batch_decode.mtp_step = real_step

    assert [cache.length for cache in caches] == [0, 0]


def test_boarding_appends_at_commit_boundary_and_keeps_epoch_lease():
    graphs = {(b, batch_decode.Q_MAX): SimpleNamespace(B=b, Q=batch_decode.Q_MAX)
              for b in range(1, 5)}
    engine = SimpleNamespace(base_graphs=graphs)
    base_caches = []
    mtp_caches = []
    prefill_calls = []

    class PrefillState:
        def __init__(self, row, pages):
            self.released = False
            self.page_indices = [list(pages)]
            self.lengths = [1]
            self.state = batch_decode.BatchDecodeState(
                caches=[Cache(1)], lengths=[1], capacities=[64],
                graph=graphs[(1, batch_decode.Q_MAX)],
                page_indices=[list(pages)],
                k0s=[[object()] for _ in range(TP)],
                tables=[[object()] for _ in range(TP)],
                mtp_caches=[Cache(1)])
            self.state.caches[0].row = row
            self.state.mtp_caches[0].row = row
            base_caches.extend(self.state.caches)
            mtp_caches.extend(self.state.mtp_caches)

        def decode_state(self, graph):
            assert graph is graphs[(1, batch_decode.Q_MAX)]
            self.released = True
            self.state.graph = graph
            return self.state

        def release(self):
            raise AssertionError("decode owns every boarded lease")

    real_prefill = batch_decode.prefill_batch_chunked
    real_prime = batch_decode.batched_mtp_prime
    real_step = batch_decode.mtp_step
    real_perf_counter = batch_decode.time.perf_counter
    clock = iter([0.0, 0.1, 0.1, 0.3, 0.3, 0.6])
    widths = []
    emitted = []
    boarded = []
    boarded_batches = []
    stats = {}
    offered = False

    def prefill(_engine, sequences, **kwargs):
        row = len(prefill_calls)
        pages = kwargs["page_indices"][0]
        prefill_calls.append((len(sequences), list(pages),
                              list(kwargs["loaded_lengths"])))
        return [object()], PrefillState(row, pages)

    def prime(_engine, state, sequences, residuals):
        assert len(sequences) == state.batch_size == len(residuals)
        return ([100 + cache.row for cache in state.caches],
                [[200 + cache.row] * (batch_decode.Q_MAX - 1)
                 for cache in state.caches])

    def step(_engine, state, base, draft):
        widths.append(state.batch_size)
        assert state.graph is graphs[(state.batch_size, batch_decode.Q_MAX)]
        tokens = ([10] if state.batch_size == 1 and len(widths) == 1 else
                  [11, 20] if state.batch_size == 2 else [12])
        return ([[token] for token in tokens], list(base), list(draft))

    def board(row):
        nonlocal offered
        if offered:
            return None
        assert row == 1
        offered = True
        return ([2], 1, threading.Event(), [1], 0)

    batch_decode.prefill_batch_chunked = prefill
    batch_decode.batched_mtp_prime = prime
    batch_decode.mtp_step = step
    batch_decode.time.perf_counter = lambda: next(clock)
    try:
        outputs = batch_decode.generate_mtp_batch(
            engine, [[1]], max_new_tokens=[3], eos_token_id=-1,
            page_indices=[[0]], loaded_lengths=[0],
            emit=lambda row, chunk: emitted.append((row, chunk)),
            select_active_rows=lambda active, done:
                [row for row in active if not done[row]],
            board_row=board,
            on_boarded_state=lambda row, state: boarded.append(
                (row, list(state.page_indices[0]))),
            on_boarded=lambda row, batch_size: boarded_batches.append(
                (row, batch_size)),
            stats=stats, boarding_interval_steps=1)
    finally:
        batch_decode.prefill_batch_chunked = real_prefill
        batch_decode.batched_mtp_prime = real_prime
        batch_decode.mtp_step = real_step
        batch_decode.time.perf_counter = real_perf_counter

    assert widths == [1, 2, 1]
    assert outputs == [[10, 11, 12], [20]]
    assert emitted == [(0, [10]), (0, [11]), (1, [20]), (0, [12])]
    assert prefill_calls == [(1, [0], [0]), (1, [1], [0])]
    assert boarded == [(1, [1])]
    assert boarded_batches == [(1, 2)]
    assert stats["steps"] == 3
    assert stats["row_steps"] == [3, 1]
    assert stats["accepts"] == [0, 0]
    assert abs(stats["row_decode_seconds"][0] - 0.6) < 1e-12
    assert abs(stats["row_decode_seconds"][1] - 0.2) < 1e-12
    assert [cache.length for cache in base_caches] == [0, 0]
    assert [cache.length for cache in mtp_caches] == [0, 0]


def test_merge_decode_rows_rejects_overlapping_pages():
    graphs = {(b, batch_decode.Q_MAX): SimpleNamespace(B=b, Q=batch_decode.Q_MAX)
              for b in range(1, 3)}
    engine = SimpleNamespace(base_graphs=graphs)

    def state(page):
        return batch_decode.BatchDecodeState(
            caches=[Cache(1)], lengths=[1], capacities=[64],
            graph=graphs[(1, batch_decode.Q_MAX)], page_indices=[[page]],
            k0s=[[object()] for _ in range(TP)],
            tables=[[object()] for _ in range(TP)], mtp_caches=[Cache(1)])

    try:
        batch_decode._merge_decode_rows(engine, state(7), state(7))
    except ValueError as exc:
        assert "overlap" in str(exc)
    else:
        raise AssertionError("overlapping live epoch pages were accepted")


if __name__ == "__main__":
    test_width_q_replay_and_independent_commit()
    test_contract_rejects_wrong_width_and_commit_count()
    test_mtp_step_restores_transaction_metadata_on_failure()
    test_q6_prime_produces_full_draft_width()
    test_select_decode_rows_is_a_non_owning_view()
    test_compaction_tracks_original_rows_and_keeps_batch_lease()
    test_active_row_policy_cannot_drop_a_live_row()
    test_boarding_appends_at_commit_boundary_and_keeps_epoch_lease()
    test_merge_decode_rows_rejects_overlapping_pages()
    print("BATCH_DECODE_Q_CONTRACT_PASS")
