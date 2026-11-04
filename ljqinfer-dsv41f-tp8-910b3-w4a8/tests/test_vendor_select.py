"""CPU regressions for compressed decode indexing; no CANN or weights needed.

Run from this instance: python -B -m unittest discover -s tests -p test_vendor_select.py -v
Only the native indexer is replaced. Preparation, page addressing, query
splitting and output assembly execute the real VendorSelect implementation.
The stub implements the vendor's right-down causal length convention; it
does not test NPU kernels or numerical top-k accuracy.
"""
import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

import torch


ROOT = Path(__file__).resolve().parents[1]


class IndexerStub:
    def __init__(self, query, key, weights, seq_q, seq_k, table, indices,
                 sparse_count):
        self.operands = (query, key, weights, seq_q, seq_k, table)
        self.indices = indices
        self.sparse_count = sparse_count

    def run(self):
        query, _, _, seq_q, seq_k, _ = self.operands
        # Return the most recent visible IDs to expose wrong causal lengths
        # and incorrect interleaving of even/odd query results.
        row = torch.arange(query.shape[1])
        limit = seq_k[:, None] - seq_q[:, None] + row[None, :] + 1
        ids = limit[:, :, None] - torch.arange(1, self.sparse_count + 1)
        self.indices.copy_(ids.clamp_min(-1)[:, :, None, :])

    def close(self):
        pass


class VendorSelectTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        self.plans = []
        native = types.ModuleType('ops.decode.vendor_indexer')
        native.VendorIndexer = self.make_indexer
        queued = types.ModuleType('ops.queued')
        queued.queued = lambda *a, **k: self.fail('native call in CPU test')
        imports = patch.dict(sys.modules, {
            'ops.queued': queued,
            'ops.decode.vendor_indexer': native,
        })
        imports.start()
        self.addCleanup(imports.stop)
        spec = importlib.util.spec_from_file_location(
            '_vendor_select_test_target', ROOT / 'ops/decode/vendor_select.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.select = module.VendorSelect

    def make_indexer(self, *args, **kwargs):
        plan = IndexerStub(*args, **kwargs)
        self.plans.append(plan)
        return plan

    def make_selector(self, ratio, batch, capacity=131072):
        page_tokens, dim = 2048, 128
        columns = capacity // page_tokens
        rpp = page_tokens // ratio
        table = torch.arange(batch * columns).reshape(batch, columns).flip(1)
        # Encode physical addresses exactly in BF16 so neighboring rows and
        # pages remain distinguishable even with a 1M-token capacity.
        bank = torch.zeros(batch * columns + 1, rpp, dim, dtype=torch.bfloat16)
        page = torch.arange(bank.shape[0])[:, None]
        lane = torch.arange(rpp)[None, :]
        bank[:, :, 0] = page // 128
        bank[:, :, 1] = page % 128
        bank[:, :, 2] = lane // 128
        bank[:, :, 3] = lane % 128
        pending = -torch.arange(1, batch * 6 + 1, dtype=torch.float32)
        pending = pending.reshape(batch, 6, 1).expand(-1, -1, dim).to(torch.bfloat16)
        starts = torch.zeros(batch, dtype=torch.int64)
        slots = torch.arange(batch).flip(0)
        obj = self.select(bank=bank, table=table, slots=slots, start=starts,
                          pending=pending, batch=batch, ratio=ratio,
                          world=1, heads_local=4, sparse_count=8, device='cpu')
        self.addCleanup(obj.close)
        return obj

    def check_positions(self, ratio, batch, capacity, positions):
        obj = self.make_selector(ratio, batch, capacity)
        base_table = obj.table.clone()
        committed = obj.bank[:-1].clone()
        addresses = [tuple(x.data_ptr() for x in p.operands) +
                     (p.indices.data_ptr(),) for *_, p in obj.parts]
        iq = torch.arange(batch * 6 * 4 * 128, dtype=torch.float32)
        iq = (iq.reshape(batch, 6, 4, 128) % 31).to(torch.bfloat16)
        weights = (torch.arange(batch * 6 * 4).reshape(batch, 6, 4) % 17).to(torch.bfloat16)
        selected = torch.empty(batch, 6, 8, dtype=torch.int64)
        for base in positions:
            with self.subTest(ratio=ratio, batch=batch, capacity=capacity, start=base):
                obj.start.copy_(torch.tensor([max(0, base - b) for b in range(batch)]))
                obj.table.copy_(base_table)
                # Future pages have not been allocated. Exercise real sentinel
                # entries as requests grow, move across pages and reuse slots.
                for b, start in enumerate(obj.start.tolist()):
                    allocated = (start + 6 + 2047) // 2048
                    obj.table[int(obj.slots[b]), allocated:] = -1
                obj.prepare()
                obj.dispatch(iq, weights, selected)

                limits = (obj.start[:, None] + torch.arange(6)[None, :] + 1) // ratio
                expected_ids = (limits[:, :, None] - torch.arange(1, 9)).clamp_min(-1)
                self.assertTrue(torch.equal(selected, expected_ids), 'causal range or query order')
                self.assertTrue(torch.equal(obj.seq_k, (obj.start + 6) // ratio))
                self.assertGreaterEqual(int(obj.block_table.min()), 0)
                self.assertLess(int(obj.block_table.max()), obj.total_blocks)
                for offset, pq, pw, _, ids, plan in obj.parts:
                    self.assertTrue(torch.equal(pq, iq[:, offset::ratio]))
                    self.assertTrue(torch.equal(pw, weights[:, offset::ratio]))
                    self.assertTrue(torch.equal(ids[:, :, 0], expected_ids[:, offset::ratio]))
                current = [tuple(x.data_ptr() for x in p.operands) +
                           (p.indices.data_ptr(),) for *_, p in obj.parts]
                self.assertEqual(current, addresses, 'bound operand addresses changed')

                for b, start in enumerate(obj.start.tolist()):
                    origin, end = start // ratio, (start + 6) // ratio
                    probes = set(range(max(0, origin - 3), end))
                    probes.update(x for x in (0, origin - 129) if 0 <= x < end)
                    for logical in probes:
                        block = int(obj.block_table[b, logical // obj.block])
                        actual = obj.key[block, logical % obj.block, 0]
                        if logical >= origin:
                            expected = obj.pending[b, logical - origin]
                        else:
                            rpp = obj.bank.shape[1]
                            page = int(base_table[int(obj.slots[b]), logical // rpp])
                            expected = committed[page, logical % rpp]
                        self.assertTrue(torch.equal(actual, expected),
                                        f'wrong key at batch={b}, row={logical}')
        self.assertTrue(torch.equal(obj.bank[:-1], committed), 'committed history was modified')

    def test_128k_coordinates_and_dispatch(self):
        positions = (0, 1, 126, 127, 128, 129, 254, 255, 256, 257,
                     2046, 2047, 2048, 65535, 130158, 131065, 131066)
        for ratio in (1, 2):
            for batch in (1, 4):
                self.check_positions(ratio, batch, 131072, positions)

    def test_default_1m_capacity(self):
        # A larger bank can hide the crash without fixing wrong row placement.
        for ratio in (1, 2):
            self.check_positions(ratio, 1, 1048576,
                                 (130158, 130159, 1048569, 1048570))

    def test_source_passes_its_compression_ratio(self):
        spec = importlib.util.spec_from_file_location(
            '_decode_source_test_target', ROOT / 'model/decode_source.py')
        source_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(source_module)
        # The Source constructor normally requires NPU tensors. Retain its
        # shape/dtype/layout checks while allowing CPU storage in this test.
        def check_tensor(value, shape, dtype, device, name):
            self.assertEqual(tuple(value.shape), tuple(shape), name)
            self.assertEqual(value.dtype, getattr(torch, dtype), name)
            self.assertEqual(value.device, device, name)
            self.assertTrue(value.is_contiguous(), name)
            return value

        gemm = types.ModuleType('ops.decode.gemm')
        gemm.Matmul = gemm.W8Matmul = lambda *args: types.SimpleNamespace(
            workspace_bytes=0, close=lambda: None)
        vendor = types.ModuleType('ops.decode.vendor_select')
        vendor.VendorSelect = self.select
        with patch.object(source_module, '_check_tensor', check_tensor), patch.dict(
                sys.modules, {'ops.decode.gemm': gemm, 'ops.decode.vendor_select': vendor}):
            for ratio in (1, 2):
                with self.subTest(ratio=ratio):
                    specs = source_module.source_resource_specs(batch=1, ratio=ratio, score_width=1024)
                    tensors = {k: torch.zeros(shape, dtype=getattr(torch, dtype))
                               for k, (shape, dtype) in specs.items()}
                    tensors.update(index_bank=torch.zeros(2, 2048 // ratio, 128, dtype=torch.bfloat16),
                                   table=torch.zeros(1, 1, dtype=torch.int64))
                    prefix = 'layers.2.attn.'
                    layout = {
                        'wq_a.weight': ((1280, 5120), torch.int8),
                        'wq_a.scale': ((1280, 1), torch.float32),
                        'indexer.wq_b.weight': ((512, 1280), torch.int8),
                        'indexer.wq_b.scale': ((512, 1), torch.float32),
                        'indexer.weights_proj.weight': ((4, 5120), torch.bfloat16),
                        'q_norm.weight': ((1280,), torch.float32),
                        'compressor.norm.weight': ((512,), torch.float32),
                        'indexer.k_norm.weight': ((128,), torch.float32),
                        'indexer.wk.weight': ((128, 512), torch.bfloat16),
                        'compressor.wkv.weight': ((512, 5120), torch.float32 if ratio == 2 else torch.bfloat16),
                    }
                    if ratio == 2:
                        layout['compressor.wgate.weight'] = ((512, 5120), torch.float32)
                        for key in ('carry_values', 'carry_scores'):
                            tensors[key] = torch.zeros(1, 4, 512, dtype=torch.float32)
                    data = {prefix + k: torch.empty(shape, dtype=dtype)
                            for k, (shape, dtype) in layout.items()}
                    bindings = {'q_norm': 'q_norm.weight', 'c_norm': 'compressor.norm.weight',
                                'i_norm': 'indexer.k_norm.weight'}
                    if ratio == 2:
                        bindings.update(c_wkv='compressor.wkv.weight', c_wgate='compressor.wgate.weight')
                    prepared = {k: data[prefix + v] for k, v in bindings.items()}
                    prepared.update(wqa_scale=torch.empty(1280, 1, dtype=torch.bfloat16),
                                    wiq_scale=torch.empty(512, 1, dtype=torch.bfloat16))
                    weights = types.SimpleNamespace(data=data, device=torch.device('cpu'), rank=None)
                    parallel = types.SimpleNamespace(lib=None, comm=types.SimpleNamespace(value=0))
                    source = source_module.Source(weights, 2, tensors, prepared, batch=1, ratio=ratio,
                                                  score_width=1024, norm_library=None, gemm_library=None,
                                                  eps=1e-6, parallel=parallel)
                    try:
                        self.assertEqual(source.vendor.ratio, ratio)
                        self.assertEqual(len(source.vendor.parts), ratio)
                    finally:
                        source.close()

    def test_unsupported_ratios_fail_before_native_construction(self):
        for ratio in (0, 3, 4, 8):
            with self.subTest(ratio=ratio), self.assertRaisesRegex(ValueError, 'ratio'):
                self.select(bank=None, table=None, slots=None, start=None,
                            pending=None, batch=1, ratio=ratio)
        self.assertEqual(self.plans, [])


if __name__ == '__main__':
    unittest.main()
