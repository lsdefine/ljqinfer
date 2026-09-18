import unittest
import torch
from ops.kernels import KernelBackend


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
class AttentionGatePackTest(unittest.TestCase):
    def test_layout_padding_and_changed_input_graph(self):
        backend = KernelBackend()
        torch.manual_seed(140)
        for batch in (1, 2, 3, 4):
            for head_major in (False, True):
                with self.subTest(batch=batch, head_major=head_major):
                    y = torch.randn((batch, 8, 6, 256), device='cuda', dtype=torch.bfloat16)
                    if head_major:
                        y = y.transpose(1, 2).contiguous().transpose(1, 2)
                    backing = torch.randn((batch, 8, 6, 512), device='cuda', dtype=torch.bfloat16)
                    gate = backing[..., 256:]
                    def reference():
                        out = torch.zeros((32, 1536), device='cuda', dtype=torch.bfloat16)
                        out[:batch * 8] = (y * torch.sigmoid(gate)).reshape(batch * 8, 1536)
                        return out
                    result = backend.attention_gate_pack(y, gate)
                    self.assertTrue(torch.equal(result, reference()))
                    with torch.cuda.stream(torch.cuda.Stream()):
                        for _ in range(3):
                            backend.attention_gate_pack(y, gate)
                    torch.cuda.synchronize()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        result = backend.attention_gate_pack(y, gate)
                    y.normal_()
                    gate.normal_()
                    graph.replay()
                    torch.cuda.synchronize()
                    self.assertTrue(torch.equal(result, reference()))
                    self.assertTrue(torch.isfinite(result).all().item())
                    graph.reset()


if __name__ == '__main__':
    unittest.main()
