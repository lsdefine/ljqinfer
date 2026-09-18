import unittest

import torch

from model.dflash2 import DFlashKVPool, HEAD_DIM, N_KV_HEADS


class DFlashKVPoolTest(unittest.TestCase):
    @staticmethod
    def _block(start: int, length: int, layers: int):
        base = torch.arange(
            start, start + length * N_KV_HEADS * HEAD_DIM,
            dtype=torch.float32).reshape(length, N_KV_HEADS, HEAD_DIM)
        keys = [base + layer * 100000 for layer in range(layers)]
        values = [tensor + 50000 for tensor in keys]
        return keys, values

    def test_cross_page_write_read_and_reset(self):
        pool = DFlashKVPool(layers=2, max_tokens=16, page_size=4,
                            device=torch.device("cpu"))
        keys, values = self._block(0, 6, 2)
        pool.write(0, keys, values)
        self.assertEqual(pool.resident_pages, 2)
        got_k, got_v = pool.read_layer(1, 1, 5)
        torch.testing.assert_close(got_k, keys[1][1:5].to(got_k.dtype))
        torch.testing.assert_close(got_v, values[1][1:5].to(got_v.dtype))
        pool.reset()
        self.assertEqual(pool.resident_pages, 0)
        self.assertTrue(torch.all(pool.page_table < 0))

    def test_import_arbitrary_logical_page(self):
        pool = DFlashKVPool(layers=2, max_tokens=16, page_size=4,
                            device=torch.device("cpu"))
        keys, values = self._block(8, 4, 2)
        pool.import_block(8, 12, keys, values)
        self.assertEqual(pool.resident_pages, 1)
        self.assertEqual(int(pool.page_table[0, 0]), -1)
        self.assertEqual(int(pool.page_table[0, 1]), -1)
        self.assertGreaterEqual(int(pool.page_table[0, 2]), 0)
        got_k, got_v = pool.read_layer(0, 8, 12)
        torch.testing.assert_close(got_k, keys[0].to(got_k.dtype))
        torch.testing.assert_close(got_v, values[0].to(got_v.dtype))
        pool.reset()
        self.assertEqual(pool.resident_pages, 0)


if __name__ == "__main__":
    unittest.main()
