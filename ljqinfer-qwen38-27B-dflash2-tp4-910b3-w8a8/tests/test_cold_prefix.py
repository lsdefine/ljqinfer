import unittest
from strategy.cold_kv_cache import PrefixColdCache


class ColdPrefixTest(unittest.TestCase):
    def setUp(self):
        self.cache = PrefixColdCache(block_size=4)
        self.ids = list(range(8))
        self.records = [{"v": [1]}, {"v": [2]}]
        self.exported = []

    def exporter(self, start, end):
        record = self.records[start // 4]
        self.exported.append((start, end))
        return record

    def test_empty_partial_and_wrong_token(self):
        self.assertEqual(self.cache.begin([]).token_count, 0)
        self.cache.store(self.cache.begin(self.ids), self.ids, self.exporter)
        self.assertEqual(self.cache.begin(self.ids[:3]).token_count, 0)
        self.assertEqual(self.cache.begin(self.ids[:6]).token_count, 4)
        wrong = self.ids.copy(); wrong[4] = 99
        self.assertEqual(self.cache.begin(wrong).token_count, 4)

    def test_idempotent_and_owned_record(self):
        self.assertEqual(
            self.cache.store(self.cache.begin(self.ids), self.ids, self.exporter), 2)
        self.assertEqual(
            self.cache.store(self.cache.begin(self.ids), self.ids, self.exporter), 0)
        self.assertEqual(self.cache.entry_count, 2)
        self.records[0]["v"][0] = 999
        restored = []
        def importer(record, final):
            restored.append((record, final))
            return len(restored) * 4
        n = self.cache.restore(self.cache.begin(self.ids), importer)
        self.assertEqual(n, 8)
        self.assertEqual(restored[0][0]["v"][0], 1)
        self.assertEqual([x[1] for x in restored], [False, True])

    def test_invalidated_match_and_bad_match_length(self):
        match = self.cache.begin(self.ids)
        self.cache.clear()
        with self.assertRaises(RuntimeError):
            self.cache.store(match, self.ids, self.exporter)
        bad = type(match)(3, (), match.generation + 1)
        with self.assertRaises(ValueError):
            self.cache.store(bad, self.ids, self.exporter)


if __name__ == "__main__":
    unittest.main()
