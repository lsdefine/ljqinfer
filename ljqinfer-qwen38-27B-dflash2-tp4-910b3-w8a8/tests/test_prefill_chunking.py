import unittest
from types import SimpleNamespace

import torch

from model.config import CONFIG
from model.model_api import ModelExecution


class _Cache:
    def __init__(self, start):
        self.lengths = torch.tensor([start], dtype=torch.int64)
        self.begun = []
        self.flushes = 0
        self.waits = 0

    def begin_gdn_checkpoints(self, sequence_id, absolute, width):
        self.begun.append((int(sequence_id), int(absolute), int(width)))

    def flush_gdn_checkpoints(self):
        self.flushes += 1

    def wait_gdn_checkpoints(self):
        self.waits += 1


class _Engine:
    def __init__(self, start=0, chunk=12288):
        self.cache = _Cache(start)
        self.cache.spec = SimpleNamespace(checkpoint_interval=1024)
        self.cache.cold_boundary_hidden_host = torch.full(
            (256, CONFIG.hidden_size), -1, dtype=torch.bfloat16)
        self.engine_config = SimpleNamespace(
            cold_checkpoint_interval=1024,
            prefill_chunk_size=chunk,
        )
        self.widths = []
        self.positions = []
        self.sequence_ids = []

    def forward_tokens(self, ids, **kwargs):
        self.assert_collect = kwargs.get("collect_gdn_checkpoints") is True
        width = int(ids.numel())
        self.widths.append(width)
        positions = kwargs.get("positions")
        sequence_ids = kwargs.get("sequence_ids")
        if positions is not None:
            self.positions.append(positions.clone())
        if sequence_ids is not None:
            self.sequence_ids.append(sequence_ids.clone())
        self.cache.lengths[0] += width
        hidden = ids.to(torch.float32).reshape(-1, 1).expand(-1, CONFIG.hidden_size)
        return hidden, [hidden + 1000, hidden + 2000]


class _Drafter:
    def __init__(self):
        self.calls = []

    def append_context(self, aux, positions, sequence_id=0):
        self.calls.append((tuple(x.clone() for x in aux), positions.clone(),
                           int(sequence_id)))


class PrefillChunkingTest(unittest.TestCase):
    def runtime(self, start=0, chunk=12288):
        runtime = ModelExecution.__new__(ModelExecution)
        runtime.engine = _Engine(start, chunk)
        runtime.verify = SimpleNamespace(aux_layer_ids=(3, 7))
        runtime.rt = SimpleNamespace(device=torch.device("cpu"))
        runtime.drafter = _Drafter()
        runtime._prefill_token_in = torch.empty(chunk, dtype=torch.int64)
        runtime._prefill_position_in = torch.empty(chunk, dtype=torch.int64)
        runtime._prefill_position_base = torch.arange(chunk, dtype=torch.int64)
        runtime._prefill_sequence_in = torch.empty(chunk, dtype=torch.int64)
        runtime._prefill_last_hidden = torch.empty(
            (1, CONFIG.hidden_size), dtype=torch.bfloat16)
        return runtime

    def test_short_prefill_is_one_model_call(self):
        runtime = self.runtime()
        ids = list(range(2500))
        hidden, aux = runtime._prefill_with_aux(ids)
        self.assertEqual(runtime.engine.widths, [2500])
        self.assertEqual(runtime.engine.cache.flushes, 1)
        self.assertEqual(runtime.engine.cache.waits, 1)
        self.assertTrue(runtime.engine.assert_collect)
        torch.testing.assert_close(hidden[:, 0], torch.tensor(ids, dtype=torch.bfloat16))
        self.assertEqual(len(aux), 2)
        self.assertEqual(tuple(aux[0].shape), (2500, CONFIG.hidden_size))

    def test_cross_chunk_prefill_uses_12k_calls(self):
        runtime = self.runtime(start=512)
        runtime._prefill_with_aux(list(range(13000)))
        self.assertEqual(runtime.engine.widths, [12288, 712])
        self.assertEqual(runtime.engine.cache.flushes, 2)
        self.assertEqual(runtime.engine.cache.waits, 1)

    def test_stream_prefill_consumes_each_chunk_and_spills_boundaries(self):
        runtime = self.runtime(start=512)
        last = runtime._stream_prefill(
            list(range(13000)), absolute_start=512, sequence_id=3)

        self.assertEqual(runtime.engine.widths, [12288, 712])
        self.assertEqual(len(runtime.drafter.calls), 2)
        self.assertEqual([int(x.numel()) for _, x, _ in runtime.drafter.calls],
                         [12288, 712])
        self.assertEqual([sid for _, _, sid in runtime.drafter.calls], [3, 3])
        self.assertEqual((int(runtime.engine.positions[0][0]),
                          int(runtime.engine.positions[0][-1])), (512, 12799))
        self.assertEqual((int(runtime.engine.positions[1][0]),
                          int(runtime.engine.positions[1][-1])), (12800, 13511))
        for rows in runtime.engine.sequence_ids:
            self.assertTrue(torch.equal(rows, torch.full_like(rows, 3)))
        for boundary in range(1024, 13512, 1024):
            expected = torch.tensor(
                boundary - 512 - 1, dtype=torch.bfloat16)
            self.assertEqual(
                runtime.engine.cache.cold_boundary_hidden_host[
                    boundary // 1024 - 1, 0], expected)
        self.assertEqual(last[0, 0], torch.tensor(12999, dtype=torch.bfloat16))
        self.assertIs(last, runtime._prefill_last_hidden)
        self.assertEqual(runtime.engine.cache.flushes, 2)
        self.assertEqual(runtime.engine.cache.waits, 1)


if __name__ == "__main__":
    unittest.main()
