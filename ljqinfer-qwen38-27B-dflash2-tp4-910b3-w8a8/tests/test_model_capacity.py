import unittest
from math import ceil
from types import SimpleNamespace

from model.config import EngineConfig
from model.model_api import ModelExecution
from model.runtime import allocate_mock_cache


class ModelCapacityTest(unittest.TestCase):
    def runtime(self, capacity=131072, sequence_capacity=None):
        values = {"max_cached_tokens": capacity}
        if sequence_capacity is not None:
            values["max_sequence_tokens"] = sequence_capacity
        runtime = ModelExecution.__new__(ModelExecution)
        runtime.engine = SimpleNamespace(
            engine_config=SimpleNamespace(**values))
        return runtime

    def test_context_capacity_comes_from_engine_and_model(self):
        self.assertEqual(self.runtime().context_capacity, 131072)
        self.assertEqual(
            self.runtime(800_000, 262_144).context_capacity, 262_144)

    def test_physical_pool_is_independent_from_sequence_geometry(self):
        cfg = EngineConfig(max_sequences=1)
        spec = allocate_mock_cache(
            max_tokens=cfg.max_cached_tokens,
            max_sequences=cfg.max_sequences,
            page_size=cfg.kv_page_size,
            checkpoint_interval=cfg.cold_checkpoint_interval,
            prefill_chunk_size=cfg.prefill_chunk_size,
            max_sequence_tokens=cfg.max_sequence_tokens)

        self.assertEqual(cfg.max_cached_tokens, 800_000)
        self.assertEqual(cfg.max_sequence_tokens, 262_144)
        self.assertEqual(spec.num_pages, ceil(800_000 / 2048))
        self.assertEqual(spec.page_table.shape, (1, ceil(262_144 / 2048)))
        self.assertEqual(spec.checkpoint_table.shape,
                         (1, ceil(262_144 / 1024)))
        self.assertEqual(spec.transient_checkpoint_slots,
                         ceil(12_288 / 1024))

    def test_power_of_two_verify_buckets(self):
        runtime = self.runtime()
        for tokens, expected in ((1, 512), (512, 512), (513, 1024),
                                 (626, 1024), (65537, 131072),
                                 (131072, 131072)):
            self.assertEqual(runtime._verify_capacity(tokens), expected)

    def test_common_verify_capacities_are_resident(self):
        self.assertEqual(
            ModelExecution._resident_verify_capacities(262_144),
            (512, 1024))
        self.assertEqual(
            ModelExecution._resident_verify_capacities(700), (512,))
        self.assertEqual(
            ModelExecution._resident_verify_capacities(256), (256,))

    def test_capacity_overflow_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "got 131073"):
            self.runtime()._verify_capacity(131073)


if __name__ == "__main__":
    unittest.main()
