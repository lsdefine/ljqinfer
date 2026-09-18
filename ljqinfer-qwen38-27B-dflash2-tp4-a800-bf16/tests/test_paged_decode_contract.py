"""Execute CUDA attention adapters on CPU with only FlashInfer replaced.

These are plan/layout/dispatch contract gates, not CUDA numerical validation.
No CUDA context, GPU allocation or real FlashInfer kernel is used. GPU numerical
correctness (including bottom-right causal masking) must be validated separately.
"""
import importlib.util
from pathlib import Path
import sys
from types import ModuleType
import unittest
from unittest.mock import Mock, patch

import torch

import model.blocks as blocks
from model.runtime import RuntimeCache, allocate_mock_cache


class PagedDecodeContractTest(unittest.TestCase):
    def setUp(self):
        props = patch.object(torch.cuda, "get_device_properties",
                             return_value=Mock(multi_processor_count=108))
        props.start()
        self.addCleanup(props.stop)
        # Load the real adapter privately so a fake dependency cannot leak into
        # model.cuda_attention used by other tests in the same pytest process.
        self.flashinfer = ModuleType("flashinfer")
        self.wrapper = Mock()
        self.flashinfer.BatchPrefillWithPagedKVCacheWrapper = Mock(return_value=self.wrapper)
        self.flashinfer.single_prefill_with_kv_cache = Mock()
        spec = importlib.util.spec_from_file_location(
            "_cuda_attention_contract", Path(blocks.__file__).with_name("cuda_attention.py"))
        self.adapter = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {"flashinfer": self.flashinfer}):
            spec.loader.exec_module(self.adapter)
        spec = allocate_mock_cache(max_tokens=1024, max_sequences=2, page_size=128,
                                   max_sequence_tokens=384, prefill_chunk_size=128)
        # Only KV storage and page metadata are needed, not large GDN pools.
        shape = (2, spec.num_pages, spec.page_size, 2, 8)
        self.cache = RuntimeCache(
            spec, torch.zeros(shape), torch.ones(shape), torch.empty(0), torch.empty(0),
            torch.full(spec.page_table.shape, -1, dtype=torch.int64),
            torch.tensor([127, 3]), free_pages=[3, 5, 1, 4, 0, 7, 2, 6],
            host_page_table=[[-1] * 3 for _ in range(2)],
            fia_page_table=torch.full(spec.page_table.shape, -1, dtype=torch.int32),
            fia_subpage_lut=torch.arange(8, dtype=torch.int32).view(-1, 1))
        self.cache.reserve_sequence_pages(0, 256)
        self.cache.reserve_sequence_pages(1, 256)
        self.assertEqual(self.cache.page_indices(0), (6, 2))
        self.assertEqual(self.cache.page_indices(1), (7, 0))

    def invoke(self, tokens=1, total=129, sid=0, layer=1):
        q = torch.arange(tokens * 4 * 8, dtype=torch.float32).reshape(tokens, 4, 8)
        result = torch.full_like(q, 19)
        self.wrapper.run.return_value = result
        lengths = self.cache.lengths.clone()
        # A dense gather would conceal broken physical page mapping.
        with patch.object(self.cache, "read_kv", side_effect=AssertionError("dense gather")), \
                patch.object(self.cache, "read_kv_range", side_effect=AssertionError("dense gather")), \
                patch.dict(sys.modules, {"model.cuda_attention": self.adapter}):
            out = blocks._paged_attention(q, self.cache, layer, sid, total)
        self.assertIs(out, result)
        rq, (rk, rv) = self.wrapper.run.call_args.args
        self.assertIs(rq, q)
        for actual, pool in ((rk, self.cache.k), (rv, self.cache.v)):
            self.assertEqual(actual.data_ptr(), pool[layer].data_ptr())
            self.assertEqual(actual.shape, pool[layer].shape)
            self.assertEqual(actual.stride(), pool[layer].stride())
        torch.testing.assert_close(self.cache.lengths, lengths, rtol=0, atol=0)
        return q

    def assert_plan(self, *, queries, total, indices):
        args, kwargs = self.wrapper.plan.call_args
        expected = ([0, queries], [0, len(indices)], indices, [(total - 1) % 128 + 1])
        for tensor, values in zip(args[:4], expected):
            self.assertEqual(tensor.tolist(), list(values))
            self.assertEqual(tensor.dtype, torch.int32)
            self.assertEqual(tensor.device.type, "cpu")
        self.assertEqual(args[4:], (4, 2, 8, 128))
        self.assertEqual(kwargs, dict(causal=True, q_data_type=torch.float32,
                                     kv_data_type=torch.float32, sm_scale=8 ** -0.5,
                                     non_blocking=False))
        constructor = self.flashinfer.BatchPrefillWithPagedKVCacheWrapper
        constructor.assert_called_once()
        workspace = constructor.call_args.args[0]
        self.assertEqual(workspace.dtype, torch.uint8)
        self.assertEqual(workspace.device.type, "cpu")
        self.assertGreater(workspace.numel(), 0)
        self.assertEqual(constructor.call_args.kwargs, dict(kv_layout="NHD", backend="fa2"))

    def test_decode_cross_page_uses_nonidentity_table_and_uncommitted_total(self):
        before_k, before_v = self.cache.k.clone(), self.cache.v.clone()
        self.invoke(total=129)
        self.assert_plan(queries=1, total=129, indices=[6, 2])
        self.assertEqual(self.cache.lengths.tolist(), [127, 3])
        torch.testing.assert_close(self.cache.k, before_k, rtol=0, atol=0)
        torch.testing.assert_close(self.cache.v, before_v, rtol=0, atol=0)

    def test_chunk_prefill_is_causal_and_plan_reuse_does_not_freeze_layer(self):
        self.invoke(tokens=3, total=130, layer=0)
        self.assert_plan(queries=3, total=130, indices=[6, 2])
        self.invoke(tokens=3, total=130, layer=1)
        self.assertEqual(self.wrapper.plan.call_count, 1)
        self.assertEqual(self.wrapper.run.call_count, 2)

    def test_replan_on_length_query_count_sequence_and_page_mapping_changes(self):
        cases = [(1, 128, 0, [6]), (1, 129, 0, [6, 2]),
                 (2, 129, 0, [6, 2]), (2, 129, 1, [7, 0]),
                 (2, 256, 1, [7, 0])]
        for count, (queries, total, sid, indices) in enumerate(cases, 1):
            self.invoke(tokens=queries, total=total, sid=sid)
            self.assert_plan(queries=queries, total=total, indices=indices)
            self.assertEqual(self.wrapper.plan.call_count, count)
        # Same shape/length but different physical ownership must invalidate plan.
        self.cache.host_page_table[1][:2] = [0, 7]
        self.cache.page_table[1, :2] = torch.tensor([0, 7])
        self.invoke(tokens=2, total=256, sid=1)
        self.assert_plan(queries=2, total=256, indices=[0, 7])
        self.assertEqual(self.wrapper.plan.call_count, len(cases) + 1)

    def test_unallocated_page_fails_before_planning_or_running(self):
        with self.assertRaisesRegex(RuntimeError, "unallocated KV page"):
            self.adapter.paged_attention(torch.zeros(1, 4, 8), self.cache, 0, 0, 257)
        self.wrapper.plan.assert_not_called()
        self.wrapper.run.assert_not_called()
        self.flashinfer.BatchPrefillWithPagedKVCacheWrapper.assert_not_called()

    def test_dense_cuda_adapter_preserves_row_mask_polarity_and_causal_flag(self):
        q, k, v = torch.zeros(2, 2, 4, 8), torch.zeros(2, 3, 2, 8), torch.ones(2, 3, 2, 8)
        mask = torch.tensor([[[False, True, True], [False, False, True]],
                             [[False, False, True], [False, False, False]]])
        sentinels = [torch.full_like(q[0], 3), torch.full_like(q[1], 7)]
        leaf = self.flashinfer.single_prefill_with_kv_cache
        for excluded in (mask, mask[:1], None):
            with self.subTest(broadcast=excluded is not None and len(excluded) == 1):
                leaf.reset_mock()
                leaf.side_effect = sentinels
                out = self.adapter.attention(q, k, v, mask=excluded, causal=True, scale=0.25)
                torch.testing.assert_close(out, torch.stack(sentinels), rtol=0, atol=0)
                self.assertEqual(leaf.call_count, 2)
                for row, call in enumerate(leaf.call_args_list):
                    for actual, source in zip(call.args, (q[row], k[row], v[row])):
                        self.assertEqual(actual.data_ptr(), source.data_ptr())
                    kwargs = dict(call.kwargs)
                    allowed = kwargs.pop("custom_mask")
                    if excluded is None:
                        self.assertIsNone(allowed)
                    else:
                        torch.testing.assert_close(allowed, ~excluded[0 if len(excluded) == 1 else row])
                    self.assertEqual(kwargs, dict(causal=True, kv_layout="NHD", sm_scale=0.25, backend="fa2"))


if __name__ == "__main__":
    unittest.main()
