"""CPU-only tests for the post-tokenization total-token admission limit."""
import pytest
from server.service import ServiceError, ServiceLayer


def layer(encoded_len, limit=68 * 1024):
    obj = ServiceLayer.__new__(ServiceLayer)
    obj.max_output_tokens = 8192
    obj.max_total_tokens = limit
    obj.model_name = "test"
    obj.render = lambda plan: "rendered"
    obj.encode = lambda prompt: list(range(encoded_len))
    return obj


def request(max_tokens):
    return {"messages": [{"role": "user", "content": "x"}],
            "max_tokens": max_tokens}


def test_total_equal_limit_is_allowed():
    ids, plan = layer(68 * 1024 - 100).build(request(100))
    assert len(ids) + plan["max_tokens"] == 68 * 1024


def test_total_above_limit_is_rejected_before_query():
    with pytest.raises(ServiceError, match=r"69633 exceeds --max-total-tokens \(69632\)"):
        layer(68 * 1024 - 99).build(request(100))
