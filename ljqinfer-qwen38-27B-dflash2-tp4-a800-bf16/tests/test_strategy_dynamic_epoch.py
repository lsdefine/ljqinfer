"""CPU contracts for Strategy's ref-compatible dynamic epoch wiring."""
from __future__ import annotations

from queue import Empty
import threading
import time
import types
import unittest


from strategy.strategy import Strategy  # noqa: E402


class _RT:
    rank = 0
    world = 1

    def __init__(self):
        self.barriers = 0
        self.activations = 0

    def barrier(self):
        self.barriers += 1

    def activate(self):
        self.activations += 1


class _State:
    def export_prefix_records(self, row, start, end):
        return tuple({"start": pos, "end": pos + 4}
                     for pos in range(start, end, 4))


class _FakeModel:
    context_capacity = 128

    def __init__(self, *, fail_after_board=False,
                 pause_before_safe_board=False):
        config = types.SimpleNamespace(cold_checkpoint_interval=4)
        self.engine = types.SimpleNamespace(engine_config=config)
        self.rt = _RT()
        self.prefill_entered = threading.Event()
        self.allow_decode = threading.Event()
        self.fail_after_board = fail_after_board
        self.pause_before_safe_board = pause_before_safe_board
        self.initial_board_probe_done = threading.Event()
        self.allow_safe_board = threading.Event()
        if not pause_before_safe_board:
            self.allow_safe_board.set()
        self.reset_calls = 0
        self.decode_entered = threading.Event()
        self.decode_entered_at = None

    def prefill_batch(self, rows, **kwargs):
        self.prefill_entered.set()
        if not self.allow_decode.wait(2):
            raise TimeoutError("test did not release prefill")
        return _State()

    def decode_dflash_batch_dynamic(
            self, state, limits, *, cancel_events, on_tokens,
            board_request, on_boarded, boarding_interval_steps):
        self.decode_entered_at = time.perf_counter()
        self.decode_entered.set()
        requests = []
        while len(requests) < 3:
            request = board_request(1 + len(requests))
            if request is None:
                break
            requests.append(request)
            on_boarded(len(requests), 1 + len(requests))
        if self.pause_before_safe_board:
            self.initial_board_probe_done.set()
            if not self.allow_safe_board.wait(2):
                raise TimeoutError("test did not release safe-point boarding")
            request = board_request(1 + len(requests))
            if request is not None:
                requests.append(request)
                on_boarded(len(requests), 1 + len(requests))
        if self.fail_after_board:
            raise RuntimeError("decode boom")
        rows = []
        all_limits = [int(limits[0])] + [r.max_new_tokens for r in requests]
        for row, limit in enumerate(all_limits):
            if limit:
                on_tokens(row, [100 + row])
            rows.append({
                "token_ids": ([100 + row] if limit else []),
                "rounds": 1 if limit else 0,
                "accepted_draft_tokens": row,
            })
        return {"rows": rows}

    def reset(self):
        self.reset_calls += 1


class DynamicEpochStrategyTest(unittest.TestCase):
    @staticmethod
    def _drain(queue):
        events = []
        while True:
            try:
                events.append(queue.get(timeout=2))
            except Empty:
                raise AssertionError("output queue did not terminate")
            if events[-1]["type"] in ("end", "error"):
                return events

    def test_anchor_waits_boarding_grace_before_first_decode(self):
        model = _FakeModel()
        strategy = Strategy(model)
        output = strategy.query((1, 2), 1)
        self.assertTrue(model.prefill_entered.wait(2))
        released_at = time.perf_counter()
        model.allow_decode.set()
        self.assertTrue(model.decode_entered.wait(2))
        self.assertGreaterEqual(model.decode_entered_at - released_at, 0.28)
        self._drain(output)
        self.assertEqual(strategy._pending, 0)

    def test_queued_requests_board_same_epoch_and_keep_event_ownership(self):
        model = _FakeModel()
        strategy = Strategy(model)
        queues = [strategy.query((row + 1, row + 2), 2) for row in range(3)]
        self.assertTrue(model.prefill_entered.wait(2))
        model.allow_decode.set()
        events = [self._drain(queue) for queue in queues]

        self.assertEqual([[event["type"] for event in row]
                          for row in events],
                         [["prefill", "token", "end"]] * 3)
        self.assertEqual([row[1]["token_ids"] for row in events],
                         [[100], [101], [102]])
        self.assertEqual([queue.metrics.active_batch_size for queue in queues],
                         [1, 2, 3])
        self.assertEqual(strategy._pending, 0)
        self.assertEqual(model.rt.barriers, 1)

    def test_post_grace_request_boards_at_later_safe_point(self):
        model = _FakeModel(pause_before_safe_board=True)
        strategy = Strategy(model)
        anchor = strategy.query((1, 2), 4)
        self.assertTrue(model.prefill_entered.wait(2))
        model.allow_decode.set()
        self.assertTrue(model.initial_board_probe_done.wait(2))

        late = strategy.query((3, 4), 1)
        model.allow_safe_board.set()
        anchor_events = self._drain(anchor)
        late_events = self._drain(late)

        self.assertEqual([event["type"] for event in anchor_events],
                         ["prefill", "token", "end"])
        self.assertEqual([event["type"] for event in late_events],
                         ["prefill", "token", "end"])
        self.assertEqual(anchor_events[1]["token_ids"], [100])
        self.assertEqual(late_events[1]["token_ids"], [101])
        self.assertEqual(anchor.metrics.active_batch_size, 1)
        self.assertEqual(late.metrics.active_batch_size, 2)
        self.assertEqual(model.rt.barriers, 1)
        self.assertEqual(strategy._pending, 0)

    def test_cancelled_fifo_head_is_skipped_and_next_request_boards(self):
        model = _FakeModel()
        strategy = Strategy(model)
        first = strategy.query((1, 2), 2)
        self.assertTrue(model.prefill_entered.wait(2))
        cancelled = strategy.query((3, 4), 2)
        last = strategy.query((5, 6), 2)
        cancelled.cancel_handle.cancel()
        model.allow_decode.set()

        first_events = self._drain(first)
        cancelled_events = self._drain(cancelled)
        last_events = self._drain(last)
        self.assertEqual(first_events[1]["token_ids"], [100])
        self.assertTrue(cancelled_events[-1]["cancelled"])
        self.assertEqual(last_events[1]["token_ids"], [101])
        self.assertEqual(strategy._pending, 0)

    def test_cancel_by_external_request_id_removes_queued_job(self):
        model = _FakeModel(pause_before_safe_board=True)
        strategy = Strategy(model)
        first = strategy.query((1, 2), 2, request_id="running")
        self.assertTrue(model.prefill_entered.wait(2))
        second = strategy.query((3, 4), 2, request_id="queued")

        self.assertTrue(strategy.cancel_request("queued", reason="client_abort"))
        self.assertFalse(strategy.cancel_request("missing"))
        model.allow_decode.set()
        self.assertTrue(model.initial_board_probe_done.wait(2))
        model.allow_safe_board.set()

        first_events = self._drain(first)
        second_events = self._drain(second)
        self.assertEqual(first_events[-1]["type"], "end")
        self.assertEqual(second_events, [{
            "type": "end", "cancelled": True,
            "reason": "client_abort",
        }])
        self.assertNotIn("queued", strategy._handles)
        self.assertNotIn("running", strategy._handles)
        self.assertEqual(strategy._pending, 0)

    def test_duplicate_active_request_id_is_rejected_then_reusable(self):
        model = _FakeModel()
        strategy = Strategy(model)
        first = strategy.query((1, 2), 1, request_id="same")
        with self.assertRaisesRegex(ValueError, "already active"):
            strategy.query((3, 4), 1, request_id="same")
        self.assertTrue(strategy.cancel_request("same"))
        model.allow_decode.set()
        self.assertEqual(self._drain(first)[-1]["reason"], "cancelled")
        replacement = strategy.query((5, 6), 1, request_id="same")
        model.allow_decode.set()
        self.assertEqual(self._drain(replacement)[-1]["type"], "end")
        self.assertNotIn("same", strategy._handles)
        self.assertFalse(strategy.cancel_request("same"))

    def test_cancel_handle_publishes_reason_before_event_and_is_idempotent(self):
        model = _FakeModel()
        strategy = Strategy(model)
        queue = strategy.query((1, 2), 1, request_id="ordered")
        handle = queue.cancel_handle
        self.assertTrue(handle.cancel("client_abort"))
        self.assertTrue(handle.cancel("ignored"))
        self.assertTrue(handle.cancel_event.is_set())
        self.assertEqual(handle.reason, "client_abort")
        model.allow_decode.set()
        self.assertEqual(self._drain(queue)[-1]["reason"], "client_abort")
        self.assertTrue(handle.cancel("too_late"))
        self.assertEqual(handle.reason, "client_abort")

    def test_decode_failure_fails_anchor_and_already_boarded_rows_once(self):
        model = _FakeModel(fail_after_board=True)
        strategy = Strategy(model)
        first = strategy.query((1, 2), 2)
        self.assertTrue(model.prefill_entered.wait(2))
        second = strategy.query((3, 4), 2)
        model.allow_decode.set()

        first_events = self._drain(first)
        second_events = self._drain(second)
        self.assertEqual([event["type"] for event in first_events],
                         ["prefill", "error"])
        self.assertEqual([event["type"] for event in second_events],
                         ["prefill", "error"])
        time.sleep(0.02)
        self.assertTrue(first.empty())
        self.assertTrue(second.empty())
        self.assertEqual(strategy._pending, 0)
        self.assertEqual(model.reset_calls, 1)


if __name__ == "__main__":
    unittest.main()
