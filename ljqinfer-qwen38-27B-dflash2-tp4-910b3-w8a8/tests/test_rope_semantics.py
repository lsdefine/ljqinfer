import unittest

import torch

from ops.kernels import K


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

    def test_qk_norm_rope_fallback_matches_composition(self):
        torch.manual_seed(11)
        q = torch.randn(8, 6, 256, dtype=torch.bfloat16)
        k = torch.randn(8, 1, 256, dtype=torch.bfloat16)
        q_weight = torch.randn(256, dtype=torch.bfloat16)
        k_weight = torch.randn(256, dtype=torch.bfloat16)
        frequencies = K.rope_frequencies(
            torch.arange(101, 109), 64, 10_000_000.0, q.dtype)

        expected_q = K.apply_rope(K.rms_norm(q, q_weight), frequencies, 64)
        expected_k = K.apply_rope(K.rms_norm(k, k_weight), frequencies, 64)
        got_q, got_k = K.qk_rms_norm_rope_decode(
            q, k, q_weight, k_weight, frequencies, 64)

        self.assertTrue(torch.equal(got_q, expected_q))
        self.assertTrue(torch.equal(got_k, expected_k))

    def test_qk_norm_rope_zero_dimension_only_normalizes(self):
        torch.manual_seed(19)
        q = torch.randn(3, 2, 16, dtype=torch.bfloat16)
        k = torch.randn(3, 1, 16, dtype=torch.bfloat16)
        q_weight = torch.randn(16, dtype=torch.bfloat16)
        k_weight = torch.randn(16, dtype=torch.bfloat16)

        got_q, got_k = K.qk_rms_norm_rope_decode(
            q, k, q_weight, k_weight, None, 0)

        self.assertTrue(torch.equal(got_q, K.rms_norm(q, q_weight)))
        self.assertTrue(torch.equal(got_k, K.rms_norm(k, k_weight)))

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
