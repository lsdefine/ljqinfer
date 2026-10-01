"""CPU-only regression: byte cap, leaf LRU, ownership and stale matches."""
import unittest
import torch
from strategy.cold_kv_cache import PrefixColdCache


def record(start, end, value=1):
    return {"start": start, "end": end,
            "target": (torch.full((4,), value, dtype=torch.bfloat16),),
            "dflash": (torch.full((2,), value, dtype=torch.float32),),
            "boundary_hidden": torch.full((3,), value, dtype=torch.bfloat16)}


class ColdPoolTest(unittest.TestCase):
    def pool(self, blocks):
        return PrefixColdCache(block_size=2, record_template=record(0, 2),
                               target_bytes=blocks * 192, max_blocks=blocks)

    def put(self, c, ids, value=1):
        return c.store_owned_batch(c.begin(ids), ids,
                                  lambda s, e: (record(s, e, value),))

    def test_capacity_reused_under_churn(self):
        c = self.pool(3)
        ptr = c._pool.data_ptr()
        self.assertEqual(c.capacity_bytes, 576)
        for i in range(200):
            self.put(c, [i, i])
            self.assertLessEqual(c.entry_count, 3)
            self.assertEqual(c._pool.data_ptr(), ptr)
        self.assertEqual(c.evicted_blocks, 197)
        self.assertEqual(len(c._entries) + len(c._free), 3)
        c.clear()
        self.assertEqual(c._pool.data_ptr(), ptr)
        self.assertEqual(c.entry_count, 0)

    def test_leaf_lru_and_ancestor_protection(self):
        c = self.pool(3)
        self.put(c, [1, 1, 2, 2])
        self.put(c, [3, 3])
        c.begin([1, 1, 2, 2])
        self.put(c, [4, 4])
        self.assertEqual(c.begin([3, 3]).token_count, 0)
        self.assertEqual(c.begin([1, 1, 2, 2]).token_count, 4)
        self.put(c, [5, 5])
        self.put(c, [6, 6])
        # Parent is never removed while a descendant is reachable.
        for d, e in c._entries.items():
            if e.parent:
                self.assertIn(e.parent, c._entries)
                self.assertIn(d, c._entries[e.parent].children)

    def test_long_prefix_stops_without_evicting_own_ancestors(self):
        c = self.pool(2)
        self.assertEqual(self.put(c, list(range(20))), 2)
        self.assertEqual(c.begin(list(range(20))).token_count, 4)
        self.assertEqual(c.evicted_blocks, 0)

    def test_no_alias_to_exporter_or_boundary_pool(self):
        c = self.pool(2)
        original = record(0, 2, 7)
        c.store_owned_batch(c.begin([1, 1]), [1, 1], lambda s, e: (original,))
        original['target'][0].fill_(99)
        original['boundary_hidden'].fill_(99)
        got = []
        c.restore(c.begin([1, 1]), lambda r, final: got.append(r) or r['end'])
        self.assertTrue(torch.all(got[0]['target'][0] == 7))
        self.assertTrue(torch.all(got[0]['boundary_hidden'] == 7))
        self.assertEqual(got[0]['target'][0].untyped_storage().data_ptr(),
                         c._pool.untyped_storage().data_ptr())

    def test_stale_match_rejected_even_after_same_digest_reinserted(self):
        c = self.pool(1)
        self.put(c, [1, 1])
        match = c.begin([1, 1])
        self.put(c, [2, 2])
        self.put(c, [1, 1])
        with self.assertRaises(RuntimeError):
            c.restore(match, lambda r, final: r['end'])

    def test_malformed_export_keeps_existing_entry(self):
        c = self.pool(1)
        self.put(c, [1, 1])
        with self.assertRaises(ValueError):
            c.store(c.begin([2, 2]), [2, 2], lambda s, e: {'x': torch.ones(100)})
        self.assertEqual(c.begin([1, 1]).token_count, 2)
        with self.assertRaises(ValueError):
            c.store(c.begin([1, 1]), [2, 2], record)

    def test_clear_invalidates_match_and_retains_pool(self):
        c = self.pool(2)
        self.put(c, [1, 1])
        match = c.begin([1, 1])
        c.clear()
        with self.assertRaises(RuntimeError):
            c.restore(match, lambda r, final: r['end'])

    def test_too_small_budget_rejected(self):
        with self.assertRaises(ValueError):
            PrefixColdCache(record_template=record(0, 2), target_bytes=1)


if __name__ == '__main__':
    unittest.main()
