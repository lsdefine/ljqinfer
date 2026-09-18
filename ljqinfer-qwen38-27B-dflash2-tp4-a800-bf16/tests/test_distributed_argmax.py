import unittest
from types import SimpleNamespace
import torch
from model.model_api import ModelExecution


class DistributedArgmaxTests(unittest.TestCase):
    def test_matches_full_vocabulary(self):
        torch.manual_seed(140)
        for world in (1, 4):
            for rows in (1, 8, 16, 24, 32):
                for case in ('random', 'ties', 'negative', 'infinity', 'nan', 'strided'):
                    with self.subTest(world=world, rows=rows, case=case):
                        width = 37
                        x = torch.randn(world, rows, width, dtype=torch.bfloat16)
                        if case == 'ties':
                            x.fill_(-3)
                            x[:, :, 7] = 5
                            x[:, :, 9] = 5
                        elif case == 'negative':
                            x = -x.abs() - 1
                        elif case == 'infinity':
                            x.fill_(-float('inf'))
                            x[-1, 0, 4] = float('inf')
                        elif case == 'nan':
                            x[0, :, 6] = float('nan')
                            x[-1, :, 2] = float('nan')
                        elif case == 'strided':
                            storage = torch.empty(world, rows, width * 2, dtype=x.dtype)
                            storage[:, :, ::2] = x
                            x = storage[:, :, ::2]
                        values, indices = x.float().max(-1)
                        offsets = torch.arange(world)[:, None] * width
                        pairs = torch.stack((values, indices.float() + offsets), -1)
                        expected = x.permute(1, 0, 2).reshape(rows, -1).float().argmax(-1).tolist()
                        for rank in range(world):
                            def gather(local, rank=rank):
                                self.assertEqual(tuple(local.shape), (rows, 2))
                                self.assertEqual(local.dtype, torch.float32)
                                torch.testing.assert_close(local, pairs[rank], rtol=0, atol=0, equal_nan=True)
                                return pairs
                            execution = SimpleNamespace(rt=SimpleNamespace(
                                rank=rank, world=world, all_gather=gather))
                            actual = ModelExecution._global_argmax_rows(execution, x[rank])
                            self.assertEqual(actual, expected)

    def test_rejects_inexact_token_index_range(self):
        execution = SimpleNamespace(rt=SimpleNamespace(rank=0, world=2**24))
        with self.assertRaises(AssertionError):
            ModelExecution._global_argmax_rows(execution, torch.zeros(1, 1))
