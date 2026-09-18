"""Exact payload/mask contract for Q8 fused logical KV gather."""
import unittest
import torch
import flashinfer
from ops.kernels import K


def transform(k, v, start):
    positions = start[:, None] + torch.arange(8, device=start.device)[None, :]
    lengths, packed = K.logical_attention_metadata(positions, k.shape[1])
    ko, vo = K.logical_kv_gather(k, v, lengths)
    return ko, vo, packed


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
class LogicalKVOrderTest(unittest.TestCase):
    def test_payload_mask_and_replay(self):
        torch.manual_seed(140)
        for batch in (1, 2, 3, 4):
            for length in (17, 520, 12296):
                with self.subTest(batch=batch, length=length):
                    k = torch.randn(batch, length, 1, 256, device='cuda', dtype=torch.bfloat16)
                    v = torch.randn_like(k)
                    start = torch.zeros(batch, device='cuda', dtype=torch.int64)
                    transform(k, v, start)
                    torch.cuda.synchronize()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        got = transform(k, v, start)
                    try:
                        for base in (0, 1, length - 9, length - 8):
                            values = [max(0, base - row) for row in range(batch)]
                            start.copy_(torch.tensor(values, device='cuda'))
                            graph.replay()
                            logical = torch.arange(length, device='cuda')[None, :].expand(batch, -1)
                            s = start[:, None]
                            idx = torch.where(logical < s, logical, length - 8 + logical - s).clamp(0, length - 1)
                            idx = idx[:, :, None, None].expand_as(k)
                            excluded = logical[:, None, None, :] >= (s[:, None, :, None] + torch.arange(8, device='cuda')[None, None, :, None] + 1)
                            packed = torch.stack([flashinfer.quantization.packbits((~excluded[row]).flatten(), bitorder='little') for row in range(batch)])
                            expected = k.gather(1, idx), v.gather(1, idx), packed
                            for actual, reference in zip(got, expected):
                                self.assertTrue(torch.equal(actual, reference))
                    finally:
                        graph.reset()


if __name__ == '__main__':
    unittest.main()
