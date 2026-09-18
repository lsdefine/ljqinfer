import unittest
import torch
from ops.kernels import K


def reference(scores, candidate):
    batch = candidate.shape[0]
    previous = torch.zeros(batch, device=scores.device, dtype=torch.long)
    rows = torch.arange(batch, device=scores.device)
    result = []
    for step in range(7):
        previous = scores[rows, step, previous].argmax(-1)
        result.append(candidate[rows, step, previous])
    return torch.stack(result, 1)


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
class DFlashPathSelectTest(unittest.TestCase):
    def test_layout_ties_and_changed_input_graph(self):
        for batch in (1, 2, 3, 4):
            for dtype in (torch.bfloat16, torch.float32):
                for strided in (False, True):
                    with self.subTest(batch=batch, dtype=dtype, strided=strided):
                        scores = torch.randn((batch, 7, 16, 16), device='cuda', dtype=dtype)
                        candidates = torch.randint(0, 248320, (batch, 7, 32), device='cuda')[:, :, ::2]
                        if strided:
                            scores = scores.transpose(2, 3)
                        self.assertTrue(torch.equal(reference(scores, candidates), K.dflash_path_select(scores, candidates)))
                        scores.zero_()
                        self.assertTrue(torch.equal(K.dflash_path_select(scores, candidates), candidates[:, :, 0]))
                        stream = torch.cuda.Stream()
                        stream.wait_stream(torch.cuda.current_stream())
                        with torch.cuda.stream(stream):
                            for _ in range(3):
                                K.dflash_path_select(scores, candidates)
                        torch.cuda.current_stream().wait_stream(stream)
                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph):
                            output = K.dflash_path_select(scores, candidates)
                        scores.normal_()
                        candidates.random_(0, 248320)
                        graph.replay()
                        torch.cuda.synchronize()
                        self.assertTrue(torch.equal(output, reference(scores, candidates)))
                        graph.reset()


if __name__ == '__main__':
    unittest.main()
