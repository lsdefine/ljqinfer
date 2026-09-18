#!/usr/bin/env python3
"""CPU contracts for semantic stop routing and legacy cancellation."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

from fastapi import HTTPException

ROOT = Path(__file__).resolve().parents[1]


class _Response:
    def __init__(self, status_code=200):
        self.status_code = status_code
        self.raised = False

    def raise_for_status(self):
        self.raised = True
        if self.status_code >= 400:
            raise RuntimeError(self.status_code)


def _load_engine_server():
    package = ModuleType("strategy")
    package.strategy = SimpleNamespace()
    previous = sys.modules.get("strategy")
    sys.modules["strategy"] = package
    try:
        spec = importlib.util.spec_from_file_location(
            "engine_server_contract", ROOT / "server" / "engine_server.py")
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module
    finally:
        if previous is None:
            sys.modules.pop("strategy", None)
        else:
            sys.modules["strategy"] = previous


def _load_remote_strategy():
    spec = importlib.util.spec_from_file_location(
        "remote_strategy_contract", ROOT / "strategy" / "remote_strategy.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_engine_exposes_stop_and_preserves_legacy_cancel():
    module = _load_engine_server()
    paths = {getattr(route, "path", None): getattr(route, "methods", set())
             for route in module.app.routes}
    assert "POST" in paths["/stop/{request_id}"]
    assert "POST" in paths["/cancel/{request_id}"]

    class Handle:
        def __init__(self):
            self.semantic = 0
            self.cancelled = 0

        def stop_at_semantic_eos(self):
            self.semantic += 1
            return True

        def cancel(self):
            self.cancelled += 1
            return True

    handle = Handle()
    module._state["handles"]["request"] = handle
    assert module.stop("request") == {
        "stopped": True, "reason": "semantic_eos"}
    assert handle.semantic == 1 and handle.cancelled == 0
    assert module.cancel("request", semantic=True) == {"cancelled": True}
    assert handle.semantic == 2 and handle.cancelled == 0
    assert module.cancel("request") == {"cancelled": True}
    assert handle.semantic == 2 and handle.cancelled == 1

    try:
        module.stop("request", reason="other")
    except HTTPException as exc:
        assert exc.status_code == 400
    else:
        raise AssertionError("unsupported stop reason was accepted")


def test_remote_handle_uses_stop_and_falls_back_only_on_404():
    module = _load_remote_strategy()
    calls = []

    def post(url, params=None):
        calls.append((url, params))
        return _Response(200)

    module.requests.post = post
    module._Handle("http://engine", "semantic").stop_at_semantic_eos()
    module._Handle("http://engine", "cancel").cancel()
    assert calls == [
        ("http://engine/stop/semantic", {"reason": "semantic_eos"}),
        ("http://engine/cancel/cancel", None),
    ]

    calls.clear()
    responses = iter([_Response(404), _Response(200)])

    def fallback_post(url, params=None):
        calls.append((url, params))
        return next(responses)

    module.requests.post = fallback_post
    module._Handle("http://old-engine", "legacy").stop_at_semantic_eos()
    assert calls == [
        ("http://old-engine/stop/legacy", {"reason": "semantic_eos"}),
        ("http://old-engine/cancel/legacy", {"semantic": True}),
    ]


if __name__ == "__main__":
    test_engine_exposes_stop_and_preserves_legacy_cancel()
    test_remote_handle_uses_stop_and_falls_back_only_on_404()
    print("STOP_API_CONTRACT_PASS")
