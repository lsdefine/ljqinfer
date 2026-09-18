import unittest
from unittest.mock import patch

import torch

from ops.kernels import K, C


class RopeSemanticsTest(unittest.TestCase):
    def test_shared_frequencies_match_convenience_wrapper(self):
        torch.manual_seed(7)
        q = torch.randn(8, 6, 256, dtype=torch.bfloat16)
        k = torch.randn(8, 1, 256, dtype=torch.bfloat16)
        positions = torch.arange(11, 19, dtype=torch.int64)

        expected_q, expected_k = K.rope(q, k, positions, 64, 10_000_000.0)
        frequencies = K.rope_frequencies(
            positions, 64, 10_000_000.0, q.dtype)
        got_q = K.apply_rope(q, frequencies, 64)
        got_k = K.apply_rope(k, frequencies, 64)

        self.assertTrue(torch.equal(got_q, expected_q))
        self.assertTrue(torch.equal(got_k, expected_k))
        self.assertTrue(torch.equal(got_q[..., 64:], q[..., 64:]))
        self.assertTrue(torch.equal(got_k[..., 64:], k[..., 64:]))

    @staticmethod
    def _rms_reference(x, weight):
        # Frozen BF16 delta-weight and normalized-output rounding boundaries.
        return (x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6)
                * (1 + weight).float()).to(x.dtype)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA numerical test; requires coordinated GPU allocation")
    def test_qk_norm_rope_cuda_matches_independent_composition(self):
        torch.manual_seed(11)
        q = torch.randn(8, 12, 256, device="cuda", dtype=torch.bfloat16)[:, ::2]
        k = torch.randn(8, 2, 256, device="cuda", dtype=torch.bfloat16)[:, ::2]
        q_weight = torch.randn(256, device="cuda", dtype=torch.bfloat16)
        k_weight = torch.randn(256, device="cuda", dtype=torch.bfloat16)
        positions = torch.arange(101, 109, device="cuda")
        for rd in (2, 64, 256):
            with self.subTest(rotary_dim=rd):
                frequencies = K.rope_frequencies(positions, rd, 10_000_000.0, q.dtype)
                # Independent trig formula, not K.apply_rope/K.rms_norm as oracle.
                phase = positions.float()[:, None] * (10_000_000.0 **
                        (-torch.arange(0, rd, 2, device="cuda").float() / rd))
                c, sn = phase.cos().to(q.dtype), phase.sin().to(q.dtype)
                torch.testing.assert_close(frequencies[0], c, atol=0, rtol=0)
                torch.testing.assert_close(frequencies[1], sn, atol=0, rtol=0)
                old_q, old_k = q.clone(), k.clone()
                got_q, got_k = K.qk_rms_norm_rope_decode(q, k, q_weight, k_weight, frequencies, rd)
                for got, x, weight in ((got_q, q, q_weight), (got_k, k, k_weight)):
                    n = self._rms_reference(x, weight)
                    left, right = n[..., :rd//2].float(), n[..., rd//2:rd].float()
                    # CUDA rotates normalized BF16 values in FP32, then rounds
                    # once. Old eager BF16 product-by-product rounding is NOT
                    # the fused CUDA contract; do not loosen that old equality.
                    cf, sf = c.float()[:, None], sn.float()[:, None]
                    expected = torch.cat((left*cf-right*sf, right*cf+left*sf,
                                          n[..., rd:].float()), dim=-1).to(x.dtype)
                    torch.testing.assert_close(got, expected, atol=0.001, rtol=0.008)
                    torch.testing.assert_close(got[..., rd:], n[..., rd:], atol=0, rtol=0)
                    self.assertEqual(got.shape, x.shape)
                    self.assertEqual(got.dtype, x.dtype)
                torch.testing.assert_close(q, old_q, atol=0, rtol=0)
                torch.testing.assert_close(k, old_k, atol=0, rtol=0)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA numerical test; requires coordinated GPU allocation")
    def test_qk_norm_rope_zero_dimension_only_normalizes(self):
        torch.manual_seed(19)
        q = torch.randn(3, 2, 16, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(3, 1, 16, device="cuda", dtype=torch.bfloat16)
        q_weight = torch.randn(16, device="cuda", dtype=torch.bfloat16)
        k_weight = torch.randn(16, device="cuda", dtype=torch.bfloat16)
        got_q, got_k = K.qk_rms_norm_rope_decode(q, k, q_weight, k_weight, None, 0)
        self.assertTrue(torch.equal(got_q, K.rms_norm(q, q_weight)))
        self.assertTrue(torch.equal(got_k, K.rms_norm(k, k_weight)))
        torch.testing.assert_close(got_q, self._rms_reference(q, q_weight), atol=0, rtol=0)
        torch.testing.assert_close(got_k, self._rms_reference(k, k_weight), atol=0, rtol=0)

    def test_zero_dimension_routes_norm_not_rotary_leaf_without_cuda(self):
        q, k = torch.empty(3, 2, 16), torch.empty(3, 1, 16)
        qw, kw = torch.empty(16), torch.empty(16)
        oq, ok = torch.empty_like(q), torch.empty_like(k)
        with patch.object(K, "rms_norm", side_effect=[oq, ok]) as norm, patch.object(C, "qk_norm_rope") as rotary:
            got_q, got_k = K.qk_rms_norm_rope_decode(q, k, qw, kw, None, 0, 0.125)
            self.assertIs(got_q, oq)
            self.assertIs(got_k, ok)
            self.assertEqual(norm.call_count, 2)
            for call, x, w in zip(norm.call_args_list, (q, k), (qw, kw)):
                self.assertIs(call.args[0], x)
                self.assertIs(call.args[1], w)
                self.assertEqual(call.args[2], 0.125)
            rotary.assert_not_called()

    def test_nonzero_rotary_rejects_cpu_before_cuda_launch(self):
        q, k = torch.empty(3, 2, 16, dtype=torch.bfloat16), torch.empty(3, 1, 16, dtype=torch.bfloat16)
        w = torch.empty(16, dtype=torch.bfloat16)
        f = K.rope_frequencies(torch.arange(3), 8, 10000., q.dtype)
        with patch.object(C, "qk_norm_rope") as leaf:
            with self.assertRaisesRegex(ValueError, "same-device BF16"):
                K.qk_rms_norm_rope_decode(q, k, w, w, f, 8)
            leaf.assert_not_called()

    def test_zero_rotary_dimension_is_identity(self):
        q = torch.randn(3, 2, 16)
        k = torch.randn(3, 1, 16)
        positions = torch.arange(3)
        got_q, got_k = K.rope(q, k, positions, 0, 10_000.0)
        self.assertIs(got_q, q)
        self.assertIs(got_k, k)
        self.assertIsNone(K.rope_frequencies(
            positions, 0, 10_000.0, q.dtype))


if __name__ == "__main__":
    unittest.main()
