import unittest
from unittest.mock import patch
import torch
from ops.kernels import K
from ops import cuda_ops as C

class VerifyNormPackedTest(unittest.TestCase):
    def test_cpu_guard_and_wrapper_contract(self):
        x = torch.empty(8, 17, dtype=torch.bfloat16)
        w = torch.empty(17, dtype=torch.bfloat16)
        with self.assertRaises(ValueError):
            C.norm_verify_packed(x, w)
        result = object()
        with patch.object(C, 'norm_verify_packed', return_value=result) as leaf:
            self.assertIs(K.rms_norm_verify_packed(x, w), result)
            leaf.assert_called_once_with(x, w, K.eps)
            leaf.reset_mock()
            self.assertIs(K.add_rms_norm_verify_packed(x, x, w, 0.125), result)
            leaf.assert_called_once_with(x, w, 0.125, residual=x)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_values_padding_residual_and_changed_input_graph(self):
        for seed in (140, 141, 142):
            torch.manual_seed(seed)
            for m in (8, 16, 24, 32):
                for add in (False, True):
                    with self.subTest(seed=seed, m=m, residual=add):
                        x = torch.randn(m, 10240, device='cuda', dtype=torch.bfloat16)[:, :5120]
                        w = torch.randn(5120, device='cuda', dtype=torch.bfloat16)
                        r = torch.randn(m, 5120, device='cuda', dtype=torch.bfloat16) if add else None
                        def run():
                            return K.add_rms_norm_verify_packed(x, r, w) if add else K.rms_norm_verify_packed(x, w)
                        def check(got):
                            reference = K.add_rms_norm(x, r, w) if add else K.rms_norm(x, w)
                            y, summed = got if add else (got, None)
                            ref, ref_sum = reference if add else (reference, None)
                            self.assertEqual(y.shape, (32, 5120))
                            self.assertTrue(torch.equal(y, K.pack_verify_rows(ref)))
                            self.assertEqual(torch.count_nonzero(y[m:]).item(), 0)
                            if add:
                                self.assertEqual(summed.shape, x.shape)
                                self.assertTrue(torch.equal(summed, ref_sum))
                        before = x.clone()
                        check(run())
                        self.assertTrue(torch.equal(before, x))
                        stream = torch.cuda.Stream()
                        stream.wait_stream(torch.cuda.current_stream())
                        with torch.cuda.stream(stream):
                            for _ in range(3): run()
                        torch.cuda.current_stream().wait_stream(stream)
                        graph = torch.cuda.CUDAGraph()
                        try:
                            with torch.cuda.graph(graph): got = run()
                            x.neg_()
                            if add: r.mul_(0.5)
                            graph.replay()
                            torch.cuda.synchronize()
                            check(got)
                        finally:
                            graph.reset()

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_fp32_reduction_preserves_both_bf16_roundings(self):
        torch.manual_seed(140)
        for m in (8, 16, 24, 32):
            for scale in (0.0, 1e-5, 1.0, 1000.0):
                x = torch.randn(m, 5120, device='cuda') * scale
                r = torch.randn(m, 5120, device='cuda', dtype=torch.bfloat16)
                w = torch.randn(5120, device='cuda', dtype=torch.bfloat16)
                got = K.add_rms_norm_verify_packed(x, r, w)
                ref = K.add_rms_norm_verify_packed(x.to(torch.bfloat16), r, w)
                for a, b in zip(got, ref):
                    self.assertEqual(a.dtype, torch.bfloat16)
                    self.assertTrue(torch.equal(a, b))
                self.assertEqual(torch.count_nonzero(got[0][m:]).item(), 0)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_invalid_cuda_payloads(self):
        x = torch.zeros(8, 17, device='cuda', dtype=torch.bfloat16)
        w = torch.zeros(17, device='cuda', dtype=torch.bfloat16)
        cases = [
            (x.float(), w, 1e-6, None),
            (x[:0], w, 1e-6, None),
            (x.repeat(5, 1), w, 1e-6, None),
            (x[:, ::2], w[:9], 1e-6, None),
            (x, w[:16], 1e-6, None),
            (x, w, 0.0, None),
            (x, w, -1.0, None),
            (x, w, float('nan'), None),
            (x, w, 1e-6, x[:1]),
            (x, w, 1e-6, torch.zeros(8, 34, device='cuda', dtype=x.dtype)[:, :17]),
        ]
        for i, args in enumerate(cases):
            with self.subTest(case=i), self.assertRaises(ValueError):
                C.norm_verify_packed(args[0], args[1], args[2], residual=args[3])
