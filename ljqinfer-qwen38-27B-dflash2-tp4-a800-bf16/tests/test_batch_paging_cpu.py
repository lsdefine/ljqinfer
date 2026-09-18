import unittest
from types import SimpleNamespace

import torch

from model.dflash2 import DFlashKVPool
from model.runtime import RuntimeCache


class BatchPagingCPUContract(unittest.TestCase):
    def test_dflash_capacity_lease_is_nonidentity_and_released(self):
        pool = DFlashKVPool(layers=2, max_tokens=20, page_size=4,
                            max_sequence_tokens=12, max_sequences=3,
                            device="cpu", dtype=torch.float32, kv_heads=1)
        # Lease sid=1 first so sid=0's logical pages do not equal physical ids.
        self.assertEqual(pool.reserve_sequence_pages(1, 8), (0, 1))
        self.assertEqual(pool.reserve_sequence_pages(0, 9), (2, 3, 4))
        self.assertEqual(pool.page_indices(0), (2, 3, 4))
        self.assertEqual(pool.page_indices(1), (0, 1))

        k = torch.arange(2 * 5 * 1 * 128, dtype=torch.float32).reshape(2, 5, 1, 128)
        v = k + 1000
        pool.write(3, tuple(k[layer] for layer in range(2)),
                   tuple(v[layer] for layer in range(2)), sequence_id=0)
        # The write crosses a logical-page tail (3..8) through a nonidentity map.
        for layer in range(2):
            rk, rv = pool.read_layer(layer, 3, 8, sequence_id=0)
            self.assertTrue(torch.equal(rk, k[layer]))
            self.assertTrue(torch.equal(rv, v[layer]))

        free_before = len(pool.free_pages)
        pool.release_sequence(0)
        self.assertEqual(len(pool.free_pages), free_before + 3)
        self.assertEqual(pool.page_indices(0), ())
        # Release is idempotent and must not duplicate free pages.
        pool.release_sequence(0)
        self.assertEqual(len(pool.free_pages), free_before + 3)

    def test_dflash_write_layer_rejects_wrong_head_geometry_before_allocation(self):
        pool = DFlashKVPool(layers=1, max_tokens=8, page_size=4,
                            max_sequence_tokens=8, max_sequences=1,
                            device="cpu", dtype=torch.float32, kv_heads=1)
        free_before = tuple(pool.free_pages)
        for shape in ((2, 2, 128), (2, 1, 64)):
            key = torch.zeros(shape, dtype=torch.float32)
            with self.assertRaisesRegex(ValueError, r"\[T,H,D\]"):
                pool.write_layer(0, 0, key, key.clone())
        self.assertEqual(tuple(pool.free_pages), free_before)
        self.assertEqual(pool.page_indices(0), ())

    def test_target_capacity_lease_is_nonidentity_and_released(self):
        spec = SimpleNamespace(max_sequences=3, max_sequence_tokens=384,
                               page_size=128)
        cache = RuntimeCache(
            spec=spec, k=None, v=None,
            hot_gdn_conv=torch.zeros((1, 3, 1)),
            hot_gdn_recurrent=torch.zeros((1, 3, 1)),
            page_table=torch.full((3, 3), -1, dtype=torch.int64),
            lengths=torch.zeros(3, dtype=torch.int64),
            free_pages=list(range(4, -1, -1)),
            host_page_table=[[-1] * 3 for _ in range(3)],
            fia_page_table=torch.full((3, 3), -1, dtype=torch.int32),
            fia_subpage_lut=torch.arange(5, dtype=torch.int32).reshape(5, 1))
        self.assertEqual(cache.reserve_sequence_pages(2, 256), (0, 1))
        self.assertEqual(cache.reserve_sequence_pages(0, 257), (2, 3, 4))
        self.assertEqual(cache.page_indices(2), (0, 1))
        self.assertEqual(cache.page_indices(0), (2, 3, 4))
        self.assertEqual(cache.host_page_table[0], [2, 3, 4])
        free_before = len(cache.free_pages)
        cache.release_sequence(0)
        self.assertEqual(len(cache.free_pages), free_before + 3)
        self.assertEqual(cache.page_indices(0), ())
        self.assertEqual(cache.host_page_table[0], [-1, -1, -1])
        cache.release_sequence(0)
        self.assertEqual(len(cache.free_pages), free_before + 3)

    def test_reservation_failure_can_be_rolled_back_atomically(self):
        pool = DFlashKVPool(layers=1, max_tokens=8, page_size=4,
                            max_sequence_tokens=8, max_sequences=2,
                            device="cpu", dtype=torch.float32, kv_heads=1)
        leased = []
        try:
            pool.reserve_sequence_pages(0, 8)
            leased.append(0)
            with self.assertRaises(MemoryError):
                pool.reserve_sequence_pages(1, 4)
        finally:
            for sid in leased + [1]:
                pool.release_sequence(sid)
        self.assertEqual(sorted(pool.free_pages), [0, 1])
        self.assertEqual(pool.page_indices(0), ())
        self.assertEqual(pool.page_indices(1), ())


if __name__ == "__main__":
    unittest.main()
