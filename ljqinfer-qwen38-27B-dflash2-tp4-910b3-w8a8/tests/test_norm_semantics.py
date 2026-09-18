import unittest

import torch

from ops.kernels import K


class NormSemanticsTest(unittest.TestCase):
    def test_qwen35_rms_norm_uses_delta_weight(self):
        x = torch.tensor([[1.0, -2.0, 3.0, -4.0]], dtype=torch.float32)
        weight = torch.tensor([0.25, -0.5, 0.0, 1.0], dtype=torch.float32)
        got = K.rms_norm(x, weight, 1e-6)
        normalized = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6)
        expected = normalized * (1.0 + weight)
        wrong_direct_weight = normalized * weight
        torch.testing.assert_close(got, expected)
        self.assertGreater(float((got - wrong_direct_weight).abs().max()), 0.5)

    def test_gdn_gated_rms_norm_uses_direct_weight(self):
        x = torch.tensor([[1.0, -2.0, 3.0, -4.0]], dtype=torch.float32)
        z = torch.tensor([[0.5, -0.25, 1.0, -1.5]], dtype=torch.float32)
        weight = torch.tensor([0.75, 1.25, 0.5, 1.5], dtype=torch.float32)
        got = K.rmsnorm_gated(x, z, weight, 1e-6)
        normalized = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6)
        expected = normalized * weight * torch.nn.functional.silu(z)
        wrong_delta_weight = normalized * (1.0 + weight) * torch.nn.functional.silu(z)
        torch.testing.assert_close(got, expected)
        self.assertGreater(float((got - wrong_delta_weight).abs().max()), 0.1)


if __name__ == "__main__":
    unittest.main()
