import unittest
import torch
from ops.kernels import K


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class VerifyPackTest(unittest.TestCase):
    def test_values_strides_and_graph_replay(self):
        for m in (1, 8, 16, 24, 32):
            for d in (17, 128, 1024, 4096, 5120, 17408):
                for stride in (1, 2):
                    with self.subTest(m=m, d=d, stride=stride):
                        x = torch.randn((m, d * stride), device="cuda", dtype=torch.bfloat16)[:, :d]
                        gold = torch.zeros((32, d), device=x.device, dtype=x.dtype)
                        gold[:m].copy_(x)
                        got = K.pack_verify_rows(x)
                        self.assertTrue(torch.equal(gold, got))
                        stream = torch.cuda.Stream()
                        stream.wait_stream(torch.cuda.current_stream())
                        with torch.cuda.stream(stream):
                            for _ in range(3):
                                K.pack_verify_rows(x)
                        torch.cuda.current_stream().wait_stream(stream)
                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph):
                            got = K.pack_verify_rows(x)
                        x.neg_()
                        gold[:m].copy_(x)
                        graph.replay()
                        torch.cuda.synchronize()
                        self.assertTrue(torch.equal(gold, got))
                        graph.reset()
