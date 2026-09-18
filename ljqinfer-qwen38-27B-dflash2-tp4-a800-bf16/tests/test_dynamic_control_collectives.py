"""CPU contracts for dynamic-boarding control collectives."""
from __future__ import annotations

import threading
import unittest

import torch

from model.model_api import BoardingRequest, ModelExecution


class _FakeRT:
    rank = 0
    device = "cpu"

    def __init__(self):
        self.dtypes = []

    def all_gather(self, tensor):
        self.dtypes.append(tensor.dtype)
        return tensor.unsqueeze(0)

    def synchronize(self):
        pass


class DynamicControlCollectiveTest(unittest.TestCase):
    def setUp(self):
        self.model = ModelExecution.__new__(ModelExecution)
        self.model.rt = _FakeRT()

    def test_boarding_request_uses_supported_float_collectives(self):
        request = BoardingRequest(
            (1, 151643, 42), 4096, threading.Event())
        result = self.model._broadcast_boarding_request(request)

        self.assertEqual(result.input_ids, request.input_ids)
        self.assertEqual(result.max_new_tokens, request.max_new_tokens)
        self.assertFalse(result.cancel_event.is_set())
        self.assertEqual(self.model.rt.dtypes,
                         [torch.float32, torch.float32])

    def test_active_rows_use_supported_float_collective(self):
        result = self.model._broadcast_int_rows([3, 1], capacity=4)

        self.assertEqual(result, [3, 1])
        self.assertEqual(self.model.rt.dtypes, [torch.float32])


if __name__ == "__main__":
    unittest.main()
