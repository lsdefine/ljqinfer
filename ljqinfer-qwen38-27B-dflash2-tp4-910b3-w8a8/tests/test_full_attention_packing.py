import math
import unittest

import torch

from model.blocks import LayerContext, _packed_sequence_attention
from model.config import CONFIG


class PackedFullAttentionTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(20260825)
        self.h = CONFIG.local_q_heads
        self.kh = CONFIG.local_kv_heads
        self.d = CONFIG.head_dim

    def _rand(self, tokens, heads):
        return (torch.randn(tokens, heads, self.d, dtype=torch.float32) * 0.2).to(torch.bfloat16)

    def _reference(self, q, k, v, sequence_ids, past_k, past_v):
        """Independent token-at-a-time definition with explicit visible prefixes."""
        out = torch.empty_like(q)
        for sid in dict.fromkeys(int(x) for x in sequence_ids.tolist()):
            index = torch.nonzero(sequence_ids == sid, as_tuple=False).flatten()
            sk, sv = k.index_select(0, index), v.index_select(0, index)
            pk, pv = past_k.get(sid), past_v.get(sid)
            prefix = 0 if pk is None else pk.shape[0]
            kk = sk if pk is None else torch.cat((pk, sk), dim=0)
            vv = sv if pv is None else torch.cat((pv, sv), dim=0)
            expanded_k = kk.repeat_interleave(self.h // self.kh, dim=1)
            expanded_v = vv.repeat_interleave(self.h // self.kh, dim=1)
            for offset, row in enumerate(index.tolist()):
                visible = prefix + offset + 1
                score = torch.einsum(
                    "hd,shd->hs", q[row].float(), expanded_k[:visible].float())
                score = score / math.sqrt(self.d)
                prob = torch.softmax(score, dim=-1).to(v.dtype)
                out[row] = torch.einsum("hs,shd->hd", prob, expanded_v[:visible])
        return out

    def test_contiguous_sequence_with_prefix_updates_complete_kv(self):
        q, k, v = self._rand(7, self.h), self._rand(7, self.kh), self._rand(7, self.kh)
        pk, pv = self._rand(3, self.kh), self._rand(3, self.kh)
        expected = self._reference(q, k, v, torch.zeros(7, dtype=torch.int64), {0: pk}, {0: pv})
        ctx = LayerContext(positions=torch.arange(3, 10),
                           sequence_ids=torch.zeros(7, dtype=torch.int64),
                           past_k={0: pk.clone()}, past_v={0: pv.clone()})
        got = _packed_sequence_attention(q, k, v, ctx)
        torch.testing.assert_close(got, expected, rtol=0.02, atol=0.02)
        self.assertTrue(torch.equal(ctx.past_k[0], torch.cat((pk, k))))
        self.assertTrue(torch.equal(ctx.past_v[0], torch.cat((pv, v))))

    def test_interleaved_sequences_are_causally_isolated(self):
        sequence_ids = torch.tensor([5, 2, 5, 2, 5, 2, 5], dtype=torch.int64)
        q, k, v = self._rand(7, self.h), self._rand(7, self.kh), self._rand(7, self.kh)
        past_k = {5: self._rand(3, self.kh), 2: self._rand(1, self.kh)}
        past_v = {5: self._rand(3, self.kh), 2: self._rand(1, self.kh)}
        expected = self._reference(q, k, v, sequence_ids, past_k, past_v)
        ctx = LayerContext(positions=torch.tensor([3, 1, 4, 2, 5, 3, 6]),
                           sequence_ids=sequence_ids,
                           past_k={sid: x.clone() for sid, x in past_k.items()},
                           past_v={sid: x.clone() for sid, x in past_v.items()})
        got = _packed_sequence_attention(q, k, v, ctx)
        torch.testing.assert_close(got, expected, rtol=0.02, atol=0.02)
        for sid in (5, 2):
            index = torch.nonzero(sequence_ids == sid, as_tuple=False).flatten()
            self.assertTrue(torch.equal(ctx.past_k[sid],
                                        torch.cat((past_k[sid], k.index_select(0, index)))))
            self.assertTrue(torch.equal(ctx.past_v[sid],
                                        torch.cat((past_v[sid], v.index_select(0, index)))))

    def test_batched_decode_does_not_mutate_cache_when_disabled(self):
        sequence_ids = torch.tensor([9, 4], dtype=torch.int64)
        q, k, v = self._rand(2, self.h), self._rand(2, self.kh), self._rand(2, self.kh)
        past_k = {9: self._rand(2, self.kh), 4: self._rand(5, self.kh)}
        past_v = {9: self._rand(2, self.kh), 4: self._rand(5, self.kh)}
        expected = self._reference(q, k, v, sequence_ids, past_k, past_v)
        ctx = LayerContext(positions=torch.tensor([2, 5]), sequence_ids=sequence_ids,
                           past_k={sid: x.clone() for sid, x in past_k.items()},
                           past_v={sid: x.clone() for sid, x in past_v.items()},
                           update_kv=False)
        got = _packed_sequence_attention(q, k, v, ctx)
        torch.testing.assert_close(got, expected, rtol=0.02, atol=0.02)
        for sid in (9, 4):
            self.assertTrue(torch.equal(ctx.past_k[sid], past_k[sid]))
            self.assertTrue(torch.equal(ctx.past_v[sid], past_v[sid]))

    def test_empty_packed_batch_is_safe(self):
        q, k, v = self._rand(0, self.h), self._rand(0, self.kh), self._rand(0, self.kh)
        ctx = LayerContext(positions=torch.empty(0, dtype=torch.int64),
                           sequence_ids=torch.empty(0, dtype=torch.int64),
                           past_k={}, past_v={})
        got = _packed_sequence_attention(q, k, v, ctx)
        self.assertEqual(got.shape, q.shape)
        self.assertEqual(ctx.past_k, {})
        self.assertEqual(ctx.past_v, {})


if __name__ == "__main__":
    unittest.main()
