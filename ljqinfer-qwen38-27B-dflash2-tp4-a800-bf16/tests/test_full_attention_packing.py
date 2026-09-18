"""CPU gates for the production RuntimeCache packed-attention orchestration.

Only the hardware attention boundary is replaced. The independent prefix oracle
checks packing, causality and isolation under that CPU substitute; it does NOT
certify FlashInfer/CUDA operator numerics (GPU validation remains pending).
"""
import math
import unittest
from dataclasses import replace
from unittest.mock import patch

import torch

from model.blocks import _packed_sequence_attention
from model.config import CONFIG
from model.runtime import RuntimeCache, TensorSpec, allocate_mock_cache


def make_cache():
    # Real page allocator/read/write/context methods, without unrelated GDN pools.
    spec = allocate_mock_cache(max_tokens=1024, max_sequences=10, page_size=128,
                               max_sequence_tokens=384, prefill_chunk_size=128)
    shape = (2, spec.num_pages, spec.page_size, CONFIG.local_kv_heads, CONFIG.head_dim)
    spec = replace(spec, k=TensorSpec("k", shape, "float32"),
                   v=TensorSpec("v", shape, "float32"))
    return RuntimeCache(
        spec, torch.full(shape, -17.0), torch.full(shape, 23.0),
        torch.empty(0), torch.empty(0),
        torch.full(spec.page_table.shape, -1, dtype=torch.int64),
        torch.zeros(spec.max_sequences, dtype=torch.int64),
        # Allocation order 6,2,7,... is deliberately nonidentity/noncontiguous.
        free_pages=[3, 5, 1, 4, 0, 7, 2, 6],
        host_page_table=[[-1] * spec.page_table.shape[1] for _ in range(10)],
        fia_page_table=torch.full(spec.page_table.shape, -1, dtype=torch.int32),
        fia_subpage_lut=torch.arange(spec.num_pages, dtype=torch.int32).view(-1, 1))


class PackedFullAttentionTest(unittest.TestCase):
    def setUp(self):
        self.rng = torch.Generator(device="cpu").manual_seed(20260825)
        self.h, self.kh, self.d = CONFIG.local_q_heads, CONFIG.local_kv_heads, CONFIG.head_dim
        self.slot = 1
        self.layer = CONFIG.full_attention_layers[self.slot]

    def rand(self, tokens, heads):
        return torch.randn(tokens, heads, self.d, generator=self.rng, device="cpu") * 0.2

    def reference(self, q, k, v, ids, past):
        """Independent token-at-a-time oracle over original (not cache-read) KV."""
        out = torch.empty_like(q)
        for sid in dict.fromkeys(ids.tolist()):
            rows = (ids == sid).nonzero().flatten().tolist()
            pk, pv = past[sid]
            kk = torch.cat((pk, k[rows])).repeat_interleave(self.h // self.kh, dim=1)
            vv = torch.cat((pv, v[rows])).repeat_interleave(self.h // self.kh, dim=1)
            for offset, row in enumerate(rows):
                visible = len(pk) + offset + 1
                score = torch.einsum("hd,shd->hs", q[row], kk[:visible]) / math.sqrt(self.d)
                out[row] = torch.einsum("hs,shd->hd", score.softmax(-1), vv[:visible])
        return out

    def run_packed(self, cache, q, k, v, ids, past, *, update=True, host_sids=None):
        positions = torch.tensor([len(past[sid][0]) + ids[:i].tolist().count(sid)
                                  for i, sid in enumerate(ids.tolist())], dtype=torch.int64)
        ctx = cache.layer_context(self.layer, ids, positions, update_kv=update,
                                  host_sids=host_sids)
        old_lengths = cache.lengths.clone()
        old_context_lengths = dict(ctx.kv_old_lengths)
        calls = []

        def cpu_attention(sq, actual_cache, slot, sid, total):
            self.assertIs(actual_cache, cache)
            self.assertEqual(slot, self.slot)
            rows = (ids == sid).nonzero().flatten()
            torch.testing.assert_close(sq, q[rows], rtol=0, atol=0)
            self.assertEqual(total, len(past[sid][0]) + len(rows))
            calls.append(sid)
            # Real paging, then a vectorized CPU stand-in for hardware attention.
            kk, vv = actual_cache.read_kv(slot, sid, total)
            kk = kk.repeat_interleave(self.h // self.kh, dim=1)
            vv = vv.repeat_interleave(self.h // self.kh, dim=1)
            score = torch.einsum("thd,shd->hts", sq, kk) / math.sqrt(self.d)
            visible = torch.arange(total)[None, :] <= (total - len(sq) + torch.arange(len(sq)))[:, None]
            score = score.masked_fill(~visible[None], float("-inf"))
            return torch.einsum("hts,shd->thd", score.softmax(-1), vv)

        with patch("model.blocks._paged_attention", side_effect=cpu_attention), \
                patch("model.blocks._fused_causal_attention", side_effect=AssertionError("dense fallback")):
            out = _packed_sequence_attention(q, k, v, ctx)
        expected_order = list(host_sids) if host_sids is not None else list(dict.fromkeys(ids.tolist()))
        self.assertEqual(calls, expected_order)
        torch.testing.assert_close(cache.lengths, old_lengths, rtol=0, atol=0)
        self.assertEqual(ctx.kv_old_lengths, old_context_lengths)
        torch.testing.assert_close(out, self.reference(q, k, v, ids, past), rtol=1e-5, atol=1e-6)
        return out

    def seed(self, cache, lengths):
        past = {sid: (self.rand(n, self.kh), self.rand(n, self.kh)) for sid, n in lengths.items()}
        for sid, (pk, pv) in past.items():
            cache.write_kv(self.slot, sid, 0, pk, pv)
            cache.lengths[sid] = len(pk)  # Explicit caller-side commit, not attention.
        return past

    def assert_storage(self, cache, before, ids, past, k, v):
        # Check the entire physical pool, including other layer/rows/slack.
        expected_k, expected_v = (x.clone() for x in before)
        for sid in dict.fromkeys(ids.tolist()):
            rows = (ids == sid).nonzero().flatten().tolist()
            for offset, row in enumerate(rows, len(past[sid][0])):
                logical, within = divmod(offset, cache.spec.page_size)
                page = cache.host_page_table[sid][logical]
                self.assertGreaterEqual(page, 0)
                self.assertEqual(cache.page_table[sid, logical].item(), page)
                expected_k[self.slot, page, within] = k[row]
                expected_v[self.slot, page, within] = v[row]
        torch.testing.assert_close(cache.k, expected_k, rtol=0, atol=0)
        torch.testing.assert_close(cache.v, expected_v, rtol=0, atol=0)

    def test_contiguous_prefix_crosses_nonidentity_pages_without_committing(self):
        for update in (True, False):
            with self.subTest(update_kv=update):
                cache = make_cache()
                past = self.seed(cache, {0: 127})
                before = cache.k.clone(), cache.v.clone()
                ids = torch.zeros(7, dtype=torch.int64)
                q, k, v = self.rand(7, self.h), self.rand(7, self.kh), self.rand(7, self.kh)
                self.run_packed(cache, q, k, v, ids, past, update=update, host_sids=(0,))
                self.assertEqual(cache.page_indices(0), (6, 2))
                self.assert_storage(cache, before, ids, past, k, v)

    def test_interleaved_unequal_rows_are_causal_and_isolated(self):
        for host_sids in (None, (2, 5)):
            with self.subTest(host_sids=host_sids):
                cache = make_cache()
                past = self.seed(cache, {5: 127, 2: 3})
                before = cache.k.clone(), cache.v.clone()
                ids = torch.tensor([5, 2, 5, 2, 5, 2, 5])
                q, k, v = self.rand(7, self.h), self.rand(7, self.kh), self.rand(7, self.kh)
                baseline = self.run_packed(cache, q, k, v, ids, past, host_sids=host_sids)
                self.assert_storage(cache, before, ids, past, k, v)
                self.assertEqual(cache.page_indices(5), (6, 7))
                self.assertEqual(cache.page_indices(2), (2,))
                # Future KV and another stream must not affect earlier rows.
                changed_k, changed_v = k.clone(), v.clone()
                changed_k[-1] += 10
                changed_v[-1] += 50
                changed = self.run_packed(cache, q, changed_k, changed_v, ids, past, host_sids=host_sids)
                torch.testing.assert_close(changed[:-1], baseline[:-1], rtol=0, atol=0)
                self.assertFalse(torch.allclose(changed[-1], baseline[-1]))

                changed_k, changed_v = k.clone(), v.clone()
                changed_k[ids == 2] -= 7
                changed_v[ids == 2] += 30
                changed = self.run_packed(cache, q, changed_k, changed_v, ids, past, host_sids=host_sids)
                torch.testing.assert_close(changed[ids == 5], baseline[ids == 5], rtol=0, atol=0)
                self.assertFalse(torch.allclose(changed[ids == 2], baseline[ids == 2]))

    def test_zero_prefix_uses_only_causal_current_chunk(self):
        cache = make_cache()
        past = self.seed(cache, {0: 0})
        before = cache.k.clone(), cache.v.clone()
        ids = torch.zeros(3, dtype=torch.int64)
        q, k, v = self.rand(3, self.h), self.rand(3, self.kh), self.rand(3, self.kh)
        out = self.run_packed(cache, q, k, v, ids, past)
        self.assert_storage(cache, before, ids, past, k, v)
        # First query has exactly one visible key, independently of Q/K scores.
        torch.testing.assert_close(out[0], v[0].repeat_interleave(self.h // self.kh, dim=0),
                                   rtol=0, atol=0)

    def test_uncommitted_decode_writes_candidates_and_retry_overwrites_them(self):
        cache = make_cache()
        past = self.seed(cache, {9: 2, 4: 129})
        ids = torch.tensor([9, 4])
        q, k, v = self.rand(2, self.h), self.rand(2, self.kh), self.rand(2, self.kh)
        before = cache.k.clone(), cache.v.clone()
        self.run_packed(cache, q, k, v, ids, past, update=False)
        self.assert_storage(cache, before, ids, past, k, v)
        tables, free = cache.page_table.clone(), list(cache.free_pages)
        retry_k, retry_v = k + 1, v - 2
        self.run_packed(cache, q, retry_k, retry_v, ids, past, update=False)
        self.assert_storage(cache, before, ids, past, retry_k, retry_v)
        torch.testing.assert_close(cache.page_table, tables, rtol=0, atol=0)
        self.assertEqual(cache.free_pages, free)

    def test_empty_packed_batch_is_noop(self):
        for host_sids in (None, ()):
            with self.subTest(host_sids=host_sids):
                cache = make_cache()
                self.seed(cache, {2: 3})
                before = [x.clone() for x in (cache.k, cache.v, cache.page_table, cache.lengths)]
                host_before = [row[:] for row in cache.host_page_table]
                free = list(cache.free_pages)
                empty = torch.empty(0, dtype=torch.int64)
                out = self.run_packed(cache, self.rand(0, self.h), self.rand(0, self.kh),
                                      self.rand(0, self.kh), empty, {}, host_sids=host_sids)
                self.assertEqual(out.shape, (0, self.h, self.d))
                self.assertEqual(out.device.type, "cpu")
                self.assertEqual(out.dtype, cache.k.dtype)
                for actual, expected in zip((cache.k, cache.v, cache.page_table, cache.lengths), before):
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                self.assertEqual(cache.host_page_table, host_before)
                self.assertEqual(cache.free_pages, free)


if __name__ == "__main__":
    unittest.main()
