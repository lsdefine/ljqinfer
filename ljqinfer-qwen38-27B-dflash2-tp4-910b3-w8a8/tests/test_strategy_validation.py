import unittest
from types import SimpleNamespace

from fastapi import HTTPException
from server.engine_server import (
    GenerateRequest, _event_stream, _state, cancel, generate, stop)
from strategy.strategy import Strategy


class StrategyValidationTest(unittest.TestCase):
    def test_engine_capacity_is_rejected_before_coordinator(self):
        strategy = Strategy.__new__(Strategy)
        strategy.model = SimpleNamespace(context_capacity=131072)
        with self.assertRaisesRegex(
                ValueError, r"prompt\+max_new <= 131072, got 131000\+73"):
            strategy.generate([1] * 131000, 73)

    def test_engine_cancel_and_stop_target_active_request_ids(self):
        calls = []
        strategy = SimpleNamespace(
            cancel_request=lambda rid, reason="cancelled": (
                calls.append((rid, reason)) or rid != "missing"))
        previous = _state["strategy"]
        _state["strategy"] = strategy
        try:
            self.assertTrue(cancel("req-a")["ok"])
            self.assertEqual(stop("req-b", "semantic_eos")["reason"],
                             "semantic_eos")
            with self.assertRaises(HTTPException) as caught:
                cancel("missing")
            self.assertEqual(caught.exception.status_code, 404)
            self.assertEqual(calls, [
                ("req-a", "cancelled"),
                ("req-b", "semantic_eos"),
                ("missing", "cancelled"),
            ])
        finally:
            _state["strategy"] = previous

    def test_engine_stream_close_marks_request_cancelled(self):
        class FakeQueue:
            def get(self):
                return {"type": "token", "token_ids": [7]}

        calls = []
        strategy = SimpleNamespace(
            cancel_request=lambda rid, reason="cancelled":
            calls.append((rid, reason)))
        previous = _state["strategy"]
        _state["strategy"] = strategy
        stream = _event_stream(FakeQueue(), "req-close")
        try:
            self.assertIn(b'"type": "token"', next(stream))
            stream.close()
            self.assertEqual(calls, [
                ("req-close", "stream_disconnected")])
        finally:
            _state["strategy"] = previous

    def test_engine_terminal_event_close_does_not_cancel(self):
        class FakeQueue:
            def get(self):
                return {"type": "end", "cancelled": False}

        calls = []
        strategy = SimpleNamespace(
            cancel_request=lambda rid, reason="cancelled":
            calls.append((rid, reason)))
        previous = _state["strategy"]
        _state["strategy"] = strategy
        stream = _event_stream(FakeQueue(), "req-complete")
        try:
            self.assertIn(b'"type": "end"', next(stream))
            stream.close()
            self.assertEqual(calls, [])
        finally:
            _state["strategy"] = previous

    def test_engine_route_rejects_before_streaming_headers(self):
        strategy = Strategy.__new__(Strategy)
        strategy.model = SimpleNamespace(context_capacity=8)
        previous = _state["strategy"]
        _state["strategy"] = strategy
        try:
            with self.assertRaises(HTTPException) as caught:
                generate(GenerateRequest(
                    input_ids=[1] * 9, max_new_tokens=0))
            self.assertEqual(caught.exception.status_code, 400)
            self.assertIn("prompt+max_new <= 8", caught.exception.detail)
        finally:
            _state["strategy"] = previous


if __name__ == "__main__":
    unittest.main()
