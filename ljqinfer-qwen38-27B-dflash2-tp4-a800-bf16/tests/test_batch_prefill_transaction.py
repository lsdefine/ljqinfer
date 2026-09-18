import unittest
from unittest.mock import patch
from types import MethodType, SimpleNamespace

import torch

from model.model_api import ModelExecution


class _Pool:
    def __init__(self, max_sequences=4, logical_pages=8, page_size=4,
                 physical_pages=24):
        self.page_size = page_size
        self.logical_pages = logical_pages
        self.host_page_table = [
            [-1] * logical_pages for _ in range(max_sequences)]
        self.page_table = torch.full(
            (max_sequences, logical_pages), -1, dtype=torch.int64)
        self.free_pages = list(range(physical_pages - 1, -1, -1))

    def _reserve(self, sid, capacity):
        count = (int(capacity) + self.page_size - 1) // self.page_size
        for logical in range(count):
            if self.host_page_table[sid][logical] < 0:
                if not self.free_pages:
                    raise MemoryError("fake page pool exhausted")
                page = self.free_pages.pop()
                self.host_page_table[sid][logical] = page
                self.page_table[sid, logical] = page
        return tuple(self.host_page_table[sid][:count])

    def reserve_sequence_pages(self, sid, capacity):
        return self._reserve(int(sid), int(capacity))

    def page_indices(self, sid):
        return tuple(page for page in self.host_page_table[int(sid)]
                     if page >= 0)

    def release_sequence(self, sid):
        sid = int(sid)
        owned = self.page_indices(sid)
        self.free_pages.extend(owned)
        self.host_page_table[sid] = [-1] * self.logical_pages
        self.page_table[sid].fill_(-1)


class _Cache(_Pool):
    def __init__(self):
        super().__init__()
        self.spec = SimpleNamespace(page_size=self.page_size)
        self.lengths = torch.zeros(4, dtype=torch.int64)
        self.hot_gdn_conv = torch.zeros((2, 4, 3), dtype=torch.float32)
        self.hot_gdn_recurrent = torch.zeros((2, 4, 5), dtype=torch.float32)

    def release_sequence(self, sid):
        super().release_sequence(sid)
        self.lengths[int(sid)] = 0
        self.hot_gdn_conv[:, int(sid)].zero_()
        self.hot_gdn_recurrent[:, int(sid)].zero_()


class _Drafter:
    def __init__(self):
        self.kv_pool = _Pool()
        self.context_lengths = [0] * 4

    def reset(self, sid):
        self.kv_pool.release_sequence(int(sid))
        self.context_lengths[int(sid)] = 0


class _Engine:
    def __init__(self):
        self.cache = _Cache()
        self.engine_config = SimpleNamespace(
            max_sequences=4, max_sequence_tokens=32,
            max_cached_tokens=96)

    def release_sequence(self, sid):
        self.cache.release_sequence(int(sid))


class _RT:
    device = "cpu"

    def __init__(self, calls):
        self.calls = calls
        self.syncs = 0

    def synchronize(self):
        self.syncs += 1
        self.calls.append(("sync", self.syncs))


class BatchPrefillTransactionTest(unittest.TestCase):
    def _execution(self, *, fail_sid=None):
        calls = []
        engine, drafter, rt = _Engine(), _Drafter(), _RT(calls)
        execution = ModelExecution(
            engine=engine, rt=rt,
            verify=SimpleNamespace(aux_layer_ids=()), drafter=drafter)

        def restore(this, records, sequence_id=0):
            sid = int(sequence_id)
            restored = int(records[0]["restored"]) if records else 0
            boundary = (torch.full((1, 3), float(100 + sid))
                        if restored else None)
            engine.cache.lengths[sid] = restored
            drafter.context_lengths[sid] = restored
            calls.append(("restore", sid, restored))
            return restored, boundary

        def stream(this, suffix, absolute_start, sequence_id=0):
            sid = int(sequence_id)
            calls.append(("stream", sid, int(absolute_start), tuple(suffix)))
            if sid == fail_sid:
                raise RuntimeError("injected row failure")
            end = int(absolute_start) + len(suffix)
            engine.cache.lengths[sid] = end
            drafter.context_lengths[sid] = end
            return torch.full((1, 3), float(10 + sid))

        execution._restore_prefix_records = MethodType(restore, execution)
        execution._stream_prefill = MethodType(stream, execution)
        return execution, calls

    def test_dynamic_allocator_trim_orders_sync_gc_cache_sync(self):
        execution, calls = self._execution()
        with patch("model.model_api.gc.collect",
                   side_effect=lambda: calls.append(("gc",))), \
             patch("model.model_api.torch.cuda.empty_cache",
                   side_effect=lambda: calls.append(("empty_cache",))):
            execution._trim_dynamic_epoch_allocator()
        self.assertEqual(calls, [
            ("sync", 1), ("gc",), ("empty_cache",), ("sync", 2)])

    def test_exact_r_resume_sid_isolation_and_handoff_views(self):
        execution, calls = self._execution()
        state = execution.prefill_batch(
            [[1, 2, 3, 4, 5, 6], [7, 8, 9, 10]],
            sequence_ids=[2, 0], max_lengths=[12, 8],
            restored_records=[({"restored": 4},), ({"restored": 4},)])

        self.assertEqual(state.sequence_ids, (2, 0))
        self.assertEqual(state.restored_lengths, (4, 4))
        # Partial row resumes at exactly R=4; exact hit performs no stream call.
        self.assertIn(("stream", 2, 4, (5, 6)), calls)
        self.assertFalse(any(call[0] == "stream" and call[1] == 0
                             for call in calls))
        self.assertEqual(execution.rt.syncs, 2)
        restore_indices = [i for i, call in enumerate(calls)
                           if call[0] == "restore"]
        first_sync = calls.index(("sync", 1))
        first_stream = next(i for i, call in enumerate(calls)
                            if call[0] == "stream")
        self.assertLess(max(restore_indices), first_sync)
        self.assertLess(first_sync, first_stream)
        self.assertEqual(calls[-1], ("sync", 2))

        target2 = set(state.rows[0].target_resident_pages)
        target0 = set(state.rows[1].target_resident_pages)
        draft2 = set(state.rows[0].dflash_resident_pages)
        draft0 = set(state.rows[1].dflash_resident_pages)
        self.assertTrue(target2.isdisjoint(target0))
        self.assertTrue(draft2.isdisjoint(draft0))

        handoff = state.decode_handoff()
        self.assertEqual(handoff["lengths"], (6, 4))
        self.assertEqual(tuple(handoff["last_hidden"].shape), (2, 3))
        self.assertEqual(
            handoff["target_page_tables"][0].data_ptr(),
            execution.engine.cache.page_table[2].data_ptr())
        before = int(handoff["target_page_tables"][0][0])
        execution.engine.cache.page_table[2, 0] = before + 10
        self.assertEqual(int(handoff["target_page_tables"][0][0]), before + 10)

        state.release()
        state.release()  # idempotent; no duplicate free-page return.
        self.assertTrue(state.released)
        self.assertEqual(execution.engine.cache.page_indices(2), ())
        self.assertEqual(execution.engine.cache.page_indices(0), ())
        self.assertEqual(execution.drafter.kv_pool.page_indices(2), ())
        self.assertEqual(execution.drafter.kv_pool.page_indices(0), ())
        with self.assertRaises(RuntimeError):
            state.decode_handoff()

    def test_live_state_rejects_sid_reuse_and_stale_release_spares_new_pages(self):
        execution, _ = self._execution()
        old = execution.prefill_batch([[1, 2, 3]], sequence_ids=[0],
                                      max_lengths=[8])
        old_target = execution.engine.cache.page_indices(0)
        old_draft = execution.drafter.kv_pool.page_indices(0)
        with self.assertRaisesRegex(RuntimeError, "already owned"):
            execution.prefill_batch([[4, 5]], sequence_ids=[0],
                                    max_lengths=[8])
        self.assertEqual(execution.engine.cache.page_indices(0), old_target)
        self.assertEqual(execution.drafter.kv_pool.page_indices(0), old_draft)

        old.release()
        new = execution.prefill_batch([[4, 5]], sequence_ids=[0],
                                      max_lengths=[8])
        new_target = execution.engine.cache.page_indices(0)
        new_draft = execution.drafter.kv_pool.page_indices(0)
        old.release()  # Stale idempotent release must not return the new lease.
        self.assertEqual(execution.engine.cache.page_indices(0), new_target)
        self.assertEqual(execution.drafter.kv_pool.page_indices(0), new_draft)
        new.release()

    def test_failure_rolls_back_all_rows_and_publishes_no_state(self):
        execution, calls = self._execution(fail_sid=1)
        target_free = len(execution.engine.cache.free_pages)
        draft_free = len(execution.drafter.kv_pool.free_pages)
        with self.assertRaisesRegex(RuntimeError, "injected row failure"):
            execution.prefill_batch(
                [[1, 2, 3], [4, 5, 6]], sequence_ids=[3, 1],
                max_lengths=[8, 12])
        self.assertIn(("stream", 3, 0, (1, 2, 3)), calls)
        self.assertIn(("stream", 1, 0, (4, 5, 6)), calls)
        for sid in (3, 1):
            self.assertEqual(execution.engine.cache.page_indices(sid), ())
            self.assertEqual(execution.drafter.kv_pool.page_indices(sid), ())
            self.assertEqual(int(execution.engine.cache.lengths[sid]), 0)
            self.assertEqual(execution.drafter.context_lengths[sid], 0)
        self.assertEqual(len(execution.engine.cache.free_pages), target_free)
        self.assertEqual(len(execution.drafter.kv_pool.free_pages), draft_free)
        self.assertEqual(execution.rt.syncs, 0)

    def test_batched_decode_freezes_completed_row_and_replays_once(self):
        execution, calls = self._execution()
        state = execution.prefill_batch(
            [[1, 2], [3, 4]], sequence_ids=[2, 0], max_lengths=[3, 5])
        graph_calls = []

        class FakeGraph:
            batch_size = 2
            max_prefix = 16
            aux_layer_ids = ()
            aux_hidden = torch.empty((0, 2, 8, 1))
            local_logits = torch.empty((2, 8, 32))
            _pending = False

            def prepare(self, ids, positions):
                graph_calls.append(("prepare", ids, positions))

            def replay(self):
                self._pending = True
                graph_calls.append(("replay",))

            def commit(self, counts):
                graph_calls.append(("commit", tuple(counts)))
                self._pending = False

            def rollback(self):
                graph_calls.append(("rollback",))
                self._pending = False

            def reset(self):
                graph_calls.append(("reset",))

        graph = FakeGraph()
        execution._select_verify = MethodType(
            lambda this, tokens, batch_size=1: (
                graph if (tokens, batch_size) == (5, 2) else
                (_ for _ in ()).throw(AssertionError((tokens, batch_size)))),
            execution)
        execution._prime_verify_batch = MethodType(
            lambda this, owned, verify=None: graph, execution)
        execution.engine.local_logits = lambda hidden: torch.empty((2, 32))
        argmax_results = iter((
            [10, 20],
            list(range(8)) + [21, 22, 23, 24, 25, 26, 27, 28],
        ))
        execution._global_argmax_rows = MethodType(
            lambda this, logits: next(argmax_results), execution)

        draft_calls, append_calls = [], []

        def draft_batch(anchors, sids, active_rows=None):
            draft_calls.append((tuple(anchors), tuple(sids), tuple(active_rows)))
            return (None, ([21, 22, 23, 24, 25, 26, 27], None, None))

        def append_batch(features, sids):
            append_calls.append((tuple(features), tuple(sids)))

        execution.drafter.draft_batch = draft_batch
        execution.drafter.append_context_graph_batch = append_batch
        streamed = []
        result = execution.decode_dflash_batch(
            state, [1, 3], on_tokens=lambda row, ids: streamed.append(
                (row, tuple(ids))))

        self.assertEqual([row["token_ids"] for row in result["rows"]],
                         [[10], [20, 21, 22]])
        self.assertEqual(draft_calls,
                         [((10, 20), (2, 0), (False, True))])
        self.assertEqual(
            [call for call in graph_calls if call[0] == "replay"],
            [("replay",)])
        self.assertIn(("commit", (0, 2)), graph_calls)
        self.assertEqual(streamed, [(0, (10,)), (1, (20,)), (1, (21, 22))])
        self.assertEqual(append_calls[0][1], (2, 0))
        self.assertIsNone(append_calls[0][0][0])
        self.assertEqual(append_calls[0][0][1], [])
        self.assertTrue(state.released)
        for sid in (2, 0):
            self.assertEqual(execution.engine.cache.page_indices(sid), ())
            self.assertEqual(execution.drafter.kv_pool.page_indices(sid), ())


if __name__ == "__main__":
    unittest.main()
