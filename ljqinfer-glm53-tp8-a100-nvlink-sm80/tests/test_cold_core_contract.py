"""Same frozen-core regression in both repositories; no torch required."""
import ast
import hashlib
import pathlib
import unittest
from strategy.cold_kv_core import ColdCache, Field, CacheCapacityError


class Payload:
    dtype = 'byte'
    ndim = 1
    def __init__(self, data):
        self.data = list(data)
        self.shape = (len(self.data),)


class Backend:
    def __init__(self):
        self.pending = 0
        self.waits = 0
        self.fail = False
    def nbytes(self, value):
        return len(value.data)
    def empty(self, value, pin):
        return Payload([0] * len(value.data))
    def copy(self, dst, src, waiters):
        waiters.add(self)
        self.pending += 1
        dst.data[:] = src.data
        if self.fail:
            raise RuntimeError('injected IO error')
    def synchronize(self):
        self.pending = 0
        self.waits += 1


class CoreContract(unittest.TestCase):
    def make(self, budget=160):
        b = Backend()
        return ColdCache([Field('kv', (None,), 'byte')], budget,
                         namespace='test', backend=b), b
    def put(self, c, tokens):
        with c.lookup(tokens, namespace='test') as parent:
            if parent.token_count == len(tokens):
                return
            with c.prepare(parent, tokens[parent.token_count:]) as tx:
                tx.write('kv', Payload(tokens[parent.token_count:]))
                tx.commit()
    def check(self, c, tokens):
        with c.lookup(tokens, namespace='test') as lease:
            self.assertEqual(lease.token_count, len(tokens))
            values = []
            for index, start, end in c.spans(lease):
                values.extend(c.storage['kv'][index].data[:end-start])
            self.assertEqual(values, tokens)
    def test_forks_then_entire_ancestor_eviction(self):
        c, b = self.make()
        for i in range(501):
            tokens = list(range(16)) + [100+i]*64
            self.put(c, tokens)
            self.check(c, tokens)
            self.assertLessEqual(len(c.entries), 2)
            self.assertTrue(all(e.pins == 0 for e in c.entries.values()))
        self.put(c, [9999]*160)
        self.assertEqual(len(c.entries), 1)
        self.check(c, [9999]*160)
        c.clear()
        self.assertEqual(c.used_bytes, 0)
    def test_overwrite_settles_and_abort_waits(self):
        c, b = self.make()
        with c.lookup([1], namespace='test') as parent:
            with c.prepare(parent, [1]) as tx:
                tx.write('kv', Payload([1]))
                self.assertEqual(b.pending, 1)
                tx.write('kv', Payload([2]))
                self.assertEqual(b.waits, 1)
            self.assertEqual(b.pending, 0)
        self.assertFalse(c.entries)
        self.assertIsNone(c.active)
    def test_copy_exception_rolls_back(self):
        c, b = self.make()
        b.fail = True
        with self.assertRaisesRegex(RuntimeError, 'injected IO'):
            self.put(c, [1])
        self.assertEqual(b.pending, 0)
        self.assertEqual(b.waits, 1)
        self.assertFalse(c.entries)
        self.assertIsNone(c.active)
    def test_readers_block_eviction_only_until_release(self):
        c, b = self.make(1)
        self.put(c, [1])
        with c.lookup([1], namespace='test'):
            with self.assertRaises(CacheCapacityError):
                self.put(c, [2])
        self.put(c, [2])
        self.check(c, [2])
    def test_core_is_frozen_and_standard_library_only(self):
        root = pathlib.Path(__file__).resolve().parents[1]
        source = root/'strategy/cold_kv_core.py'
        expected = '4ea13d4028d84c7d85a4ff06b80f0af57d0577e94e83e88387ad0c05be51b730'
        self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), expected)
        tree = ast.parse(source.read_text())
        imports = [n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
        self.assertEqual(set(imports), {'dataclasses', 'threading'})
        self.assertFalse(any(isinstance(n, ast.Import) for n in ast.walk(tree)))


if __name__ == '__main__':
    unittest.main()
