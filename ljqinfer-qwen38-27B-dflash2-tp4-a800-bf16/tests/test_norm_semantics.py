import unittest
from unittest.mock import patch

import torch

from ops.kernels import K, C


class NormSemanticsTest(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "CUDA numerical test; requires coordinated GPU allocation")
    def test_qwen35_rms_norm_uses_delta_weight(self):
        x = torch.tensor([[1.0, -2.0, 3.0, -4.0]], dtype=torch.float32, device="cuda")
        weight = torch.tensor([0.25, -0.5, 0.0, 1.0], dtype=torch.float32, device="cuda")
        got = K.rms_norm(x, weight, 1e-6)
        normalized = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6)
        expected = normalized * (1.0 + weight)
        wrong_direct_weight = normalized * weight
        torch.testing.assert_close(got, expected)
        self.assertGreater(float((got - wrong_direct_weight).abs().max()), 0.5)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA numerical test; requires coordinated GPU allocation")
    def test_gdn_gated_rms_norm_uses_direct_weight(self):
        x = torch.tensor([[1.0, -2.0, 3.0, -4.0]], dtype=torch.float32, device="cuda")
        z = torch.tensor([[0.5, -0.25, 1.0, -1.5]], dtype=torch.float32, device="cuda")
        weight = torch.tensor([0.75, 1.25, 0.5, 1.5], dtype=torch.float32, device="cuda")
        got = K.rmsnorm_gated(x, z, weight, 1e-6)
        normalized = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6)
        expected = normalized * weight * torch.nn.functional.silu(z)
        wrong_delta_weight = normalized * (1.0 + weight) * torch.nn.functional.silu(z)
        torch.testing.assert_close(got, expected)
        self.assertGreater(float((got - wrong_delta_weight).abs().max()), 0.1)

    def test_norm_routes_modes_eps_and_residual_without_cuda(self):
        x, w, z, residual = torch.empty(2, 4), torch.empty(4), torch.empty(2, 4), torch.empty(2, 4)
        out = torch.empty_like(x)
        with patch.object(C, "norm", return_value=out) as leaf:
            self.assertIs(K.rms_norm(x, w), out)
            leaf.assert_called_once_with(x, w, K.eps)
            leaf.reset_mock()
            self.assertIs(K.rms_norm(x, w, 0.125), out)
            leaf.assert_called_once_with(x, w, 0.125)
            leaf.reset_mock()
            got = K.rmsnorm_gated(x, z, w, 0.25)
            self.assertEqual(got.shape, x.shape)
            self.assertEqual(got.data_ptr(), out.data_ptr())
            leaf.assert_called_once_with(x, w, 0.25, mode=2, z=z)
            leaf.reset_mock()
            got, carry = K.residual_rms_norm(x, None, w, 0.5)
            self.assertIs(got, out)
            self.assertIs(carry, x)
            leaf.assert_called_once_with(x, w, 0.5)
        summed = torch.empty_like(x)
        with patch.object(C, "norm", return_value=(out, summed)) as leaf:
            got, carry = K.residual_rms_norm(x, residual, w, 0.5)
            self.assertIs(got, out)
            self.assertIs(carry, summed)
            leaf.assert_called_once_with(x, w, 0.5, residual=residual)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA numerical test; requires coordinated GPU allocation")
    def test_bf16_rounding_and_packed_rank4_inputs(self):
        torch.manual_seed(31)
        # Packed rows and heads, not only contiguous flat vectors.
        x = torch.randn(2, 8, 24, 128, device="cuda", dtype=torch.bfloat16)[:, :, ::2]
        z = torch.randn_like(x)
        w = torch.randn(128, device="cuda", dtype=torch.bfloat16)
        r = torch.randn_like(x)
        old = x.clone()
        def rms(a, delta):
            ww = (1 + w).float() if delta else w.float()
            return (a.float() * torch.rsqrt(a.float().square().mean(-1, keepdim=True) + 1e-6) * ww).to(a.dtype)
        got = K.rms_norm(x, w, 1e-6)
        torch.testing.assert_close(got, rms(x, True), atol=0, rtol=0)
        gated = (rms(x, False).float() * torch.nn.functional.silu(z.float())).to(x.dtype)
        torch.testing.assert_close(K.rmsnorm_gated(x, z, w), gated, atol=0.001, rtol=0.008)
        got, summed = K.add_rms_norm(x, r, w)
        torch.testing.assert_close(summed, x + r, atol=0, rtol=0)
        torch.testing.assert_close(got, rms(x + r, True), atol=0.001, rtol=0.008)
        torch.testing.assert_close(x, old, atol=0, rtol=0)
        self.assertEqual(got.shape, x.shape)


if __name__ == "__main__":
    unittest.main()
