#!/usr/bin/env python3
import unittest

import torch

from strategy.cold_kv_cache import KVLayout, PinnedMemoryKVCache


class PinnedMemoryKVCacheTest(unittest.TestCase):
    def make_cache(self, blocks: int = 3) -> PinnedMemoryKVCache:
        layout = KVLayout(layers=2, width=3, dtype=torch.float16)
        page_bytes = 4 * layout.layers * layout.width * 2
        cache = PinnedMemoryKVCache(
            layout, block_size=4,
            initial_bytes=page_bytes,
            target_bytes=page_bytes * blocks)
        cache._allocator.join(timeout=5)
        self.assertEqual(cache.capacity_blocks, blocks)
        return cache

    @staticmethod
    def tokens(count: int, offset: int = 0):
        return list(range(offset, offset + count))

    @staticmethod
    def source(count: int):
        return torch.arange(count * 2 * 3, dtype=torch.float16).reshape(count, 2, 3)

    def put(self, cache, tokens):
        source = self.source(len(tokens))
        result = cache.store(cache.begin(tokens), tokens,
                             lambda s, e, dst: dst.copy_(source[s:e]))
        return result, source

    def restored(self, cache, tokens):
        match = cache.begin(tokens)
        out = torch.empty((match.token_count, 2, 3), dtype=torch.float16)
        cache.restore(match, lambda s, e, src: out[s:e].copy_(src))
        return match, out

    def assert_tree(self, cache):
        valid = {e.id: e for e in cache._entries if e.is_valid}
        seen = set()
        stack = list(cache._root_next_ids)
        while stack:
            entry_id = stack.pop()
            self.assertNotIn(entry_id, seen)
            seen.add(entry_id)
            entry = valid[entry_id]
            for child_id in entry.next_ids:
                self.assertEqual(valid[child_id].previous_id, entry.id)
            stack.extend(entry.next_ids)
        self.assertEqual(seen, set(valid))
        for entry in valid.values():
            self.assertGreater(entry.valid_tokens, 0)
            self.assertLessEqual(entry.valid_tokens, cache.block_size)
            if entry.valid_tokens < cache.block_size:
                self.assertFalse(entry.next_ids)

    def test_background_growth_full_store_and_restore(self):
        cache = self.make_cache(3)
        result, source = self.put(cache, self.tokens(12))
        self.assertEqual((result.stored_blocks, result.evicted_blocks), (3, 0))
        hit, restored = self.restored(cache, self.tokens(12))
        self.assertEqual((hit.block_count, hit.token_count), (3, 12))
        self.assertTrue(torch.equal(restored, source))
        self.assertTrue(all(entry.tensor.is_pinned() for entry in cache._entries))
        self.assertTrue(all(entry.tensor[:, 0, :].is_contiguous()
                            for entry in cache._entries))
        self.assert_tree(cache)

    def test_partial_tail_exact_hit_and_restore(self):
        cache = self.make_cache(2)
        result, source = self.put(cache, self.tokens(7))
        self.assertEqual(result.stored_blocks, 2)
        hit, restored = self.restored(cache, self.tokens(7))
        self.assertEqual((hit.block_count, hit.token_count), (2, 7))
        self.assertTrue(torch.equal(restored, source))
        self.assertEqual(sorted(e.valid_tokens for e in cache._entries if e.is_valid), [3, 4])
        self.assertEqual(cache.token_count, 7)

    def test_prefix_match_inside_partial_and_full_block(self):
        cache = self.make_cache(4)
        _, source = self.put(cache, self.tokens(7))
        hit, restored = self.restored(cache, self.tokens(6))
        self.assertEqual(hit.token_count, 6)
        self.assertTrue(torch.equal(restored, source[:6]))

        cache2 = self.make_cache(2)
        _, source2 = self.put(cache2, self.tokens(8))
        query = self.tokens(6) + [999]
        hit2, restored2 = self.restored(cache2, query)
        self.assertEqual(hit2.token_count, 6)
        self.assertTrue(torch.equal(restored2, source2[:6]))

    def test_divergent_partial_branches_choose_longest_prefix(self):
        cache = self.make_cache(4)
        a = [0, 1, 2, 3, 10, 11, 12]
        b = [0, 1, 2, 3, 10, 11, 20]
        self.put(cache, a)
        self.put(cache, b)
        self.assertEqual(cache.begin([0, 1, 2, 3, 10, 11, 12, 99]).token_count, 7)
        self.assertEqual(cache.begin([0, 1, 2, 3, 10, 11, 20, 99]).token_count, 7)
        self.assertEqual(cache.begin([0, 1, 2, 3, 10, 11, 99]).token_count, 6)
        self.assert_tree(cache)

    def test_partial_upgrade_normalizes_and_reuses_full_sibling(self):
        cache = self.make_cache(4)
        self.put(cache, self.tokens(6))       # 4 + 2
        result, _ = self.put(cache, self.tokens(11))  # 4 + 4 + 3
        self.assertEqual(result.stored_blocks, 2)
        self.assertEqual(cache.begin(self.tokens(11)).token_count, 11)
        root = next(e for e in cache._entries if e.is_valid and e.previous_id == 0)
        child_lengths = sorted(cache._entry_by_id(i).valid_tokens for i in root.next_ids)
        self.assertEqual(child_lengths, [2, 4])
        full_child = next(cache._entry_by_id(i) for i in root.next_ids
                          if cache._entry_by_id(i).valid_tokens == 4)
        self.assertEqual([cache._entry_by_id(i).valid_tokens
                          for i in full_child.next_ids], [3])
        self.assert_tree(cache)

    def test_full_capacity_partial_upgrade_replaces_old_leaf(self):
        cache = self.make_cache(2)
        self.put(cache, self.tokens(6))
        result, _ = self.put(cache, self.tokens(8))
        self.assertEqual((result.stored_blocks, result.evicted_blocks), (1, 1))
        self.assertEqual(cache.begin(self.tokens(8)).token_count, 8)
        self.assertEqual(cache.valid_slot_count, 2)
        self.assert_tree(cache)

    def test_lru_keeps_ancestor_and_evicts_oldest_leaf(self):
        cache = self.make_cache(3)
        a = [0, 1, 2, 3, 4, 5, 6, 7]
        b = [100, 101, 102, 103]
        self.put(cache, a)  # root A -> A2
        self.put(cache, b)  # root B
        cache.begin(a)      # refresh both A nodes
        c = [200, 201, 202, 203]
        result, _ = self.put(cache, c)
        self.assertEqual(result.evicted_blocks, 1)
        self.assertEqual(cache.begin(a).token_count, 8)
        self.assertEqual(cache.begin(b).token_count, 0)
        self.assertEqual(cache.begin(c).token_count, 4)
        self.assert_tree(cache)

    def test_lru_leaf_order_preserves_shared_ancestor(self):
        cache = self.make_cache(3)
        a = [0, 1, 2, 3, 10, 11]
        b = [0, 1, 2, 3, 20, 21]
        self.put(cache, a)
        self.put(cache, b)
        cache.begin(a)  # A leaf newest, shared parent also newest
        c = [100, 101, 102, 103]
        self.put(cache, c)
        self.assertEqual(cache.begin(a).token_count, 6)
        self.assertEqual(cache.begin(b).token_count, 4)  # shared parent remains
        self.assertEqual(cache.begin(c).token_count, 4)
        self.assert_tree(cache)

    def test_repeated_partial_upgrades_at_fixed_capacity(self):
        cache = self.make_cache(3)
        for length in range(1, 13):
            tokens = self.tokens(length)
            self.put(cache, tokens)
            self.assertEqual(cache.begin(tokens).token_count, length)
            self.assertLessEqual(cache.valid_slot_count, 3)
            self.assert_tree(cache)
        self.assertEqual(cache.begin(self.tokens(12)).token_count, 12)

    def test_one_slot_partial_upgrade_and_divergence(self):
        cache = self.make_cache(1)
        self.put(cache, [1, 2])
        result, _ = self.put(cache, [1, 2, 3, 4])
        self.assertEqual((result.stored_blocks, result.evicted_blocks), (1, 1))
        self.assertEqual(cache.begin([1, 2, 3, 4]).token_count, 4)
        self.put(cache, [1, 9, 8])
        self.assertEqual(cache.begin([1, 9, 8]).token_count, 3)
        self.assertEqual(cache.begin([1, 2, 3, 4]).token_count, 1)
        self.assert_tree(cache)

    def test_hash_collision_uses_tokens_as_identity(self):
        cache = self.make_cache(3)
        cache._hash_tokens = lambda tokens: b"forced-collision"
        a = [1, 2, 3, 4]
        b = [5, 6, 7, 8]
        self.put(cache, a)
        self.put(cache, b)
        self.assertEqual(cache.begin(a).token_count, 4)
        self.assertEqual(cache.begin(b).token_count, 4)
        self.assert_tree(cache)

    def test_access_order_keeps_ancestor_no_older_than_descendant(self):
        cache = self.make_cache(4)
        paths = [self.tokens(12), self.tokens(7), self.tokens(5, 100)]
        for path in paths:
            self.put(cache, path)
            cache.begin(path)
        valid = {e.id: e for e in cache._entries if e.is_valid}
        for entry in valid.values():
            if entry.previous_id:
                self.assertGreaterEqual(
                    valid[entry.previous_id].last_used, entry.last_used)
        self.assert_tree(cache)

    def test_store_failure_leaves_tree_consistent(self):
        cache = self.make_cache(1)
        def fail(start, end, dst):
            raise RuntimeError("copy failed")
        with self.assertRaisesRegex(RuntimeError, "copy failed"):
            cache.store(cache.begin([1, 2]), [1, 2], fail)
        self.assertEqual(cache.valid_slot_count, 0)
        self.assertEqual(cache.begin([1, 2]).token_count, 0)
        self.assert_tree(cache)

    def test_clear_invalidates_entries_and_old_lookup(self):
        cache = self.make_cache(1)
        self.put(cache, self.tokens(3))
        old = cache.begin(self.tokens(3))
        cache.clear()
        self.assertEqual(cache.token_count, 0)
        self.assertEqual(cache.begin(self.tokens(3)).block_count, 0)
        with self.assertRaisesRegex(RuntimeError, "invalidated"):
            cache.restore(old, lambda start, end, src: None)
        self.assert_tree(cache)


if __name__ == "__main__":
    unittest.main()
