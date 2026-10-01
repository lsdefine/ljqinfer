"""Candidate transport, nucleus boundary and distribution contracts."""
import math
import unittest
import torch
from model.sampling import sample_pairs, select_candidates, temperatures, TOP_K, TOP_P


class SamplingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def draw(self, logits, temps, world=4, seed=123):
        shards = logits.chunk(world, dim=-1)
        gathered = [sample_pairs(x, temps, rank, world,
                    generator=torch.Generator().manual_seed(seed + rank * 100003))
                    for rank, x in enumerate(shards)]
        return select_candidates(torch.stack(gathered).tolist(), temps)

    def reference(self, row, t):
        order = row.argsort(descending=True)[:TOP_K]
        if t == 0:
            p = torch.zeros_like(row);p[order[0]] = 1;return p
        probs = (row[order] / t).softmax(0)
        keep = probs.cumsum(0) - probs < TOP_P
        probs[~keep] = 0;probs /= probs.sum()
        p = torch.zeros_like(row);p[order] = probs;return p

    def test_distribution_and_no_input_mutation(self):
        base = torch.linspace(-4, 2, 80)
        # Top candidates deliberately spread across every shard.
        base = base[torch.randperm(80, generator=torch.Generator().manual_seed(22))]
        for t in (0.5, 1.0, 2.0):
            x = base.repeat(18000, 1);saved = x.clone()
            ids = torch.tensor(self.draw(x, [t]))
            expected = self.reference(base, t)
            observed = torch.bincount(ids, minlength=80).float() / len(ids)
            self.assertLess((observed - expected).abs().max().item(), 0.012)
            self.assertTrue((expected[ids] > 0).all())
            self.assertTrue(torch.equal(saved, x))

    def test_mixed_request_rows(self):
        base = torch.linspace(-3, 3, 80)
        n = 6000
        ids = torch.tensor(self.draw(base.repeat(4*n, 1), [0, 1, 0, 1]))
        for i in (0, 2):self.assertTrue((ids[i*n:(i+1)*n] == 79).all())
        for i in (1, 3):
            observed = torch.bincount(ids[i*n:(i+1)*n], minlength=80).float()/n
            self.assertLess((observed-self.reference(base, 1)).abs().max().item(), .018)

    def test_nucleus_includes_crossing_token_and_ignores_tail_noise(self):
        # Probabilities .90,.06,.04; keep first TWO, not only the first.
        pool = [[[[math.log(.90), 0., 1.], [math.log(.06), 1., 1e-8],
                  [math.log(.04), 2., 1e-35]]]]
        self.assertEqual(select_candidates(pool, [1]), [1])
        self.assertEqual(select_candidates(pool, [0]), [0])

    def test_global_topk_before_noise(self):
        x = torch.arange(80, dtype=torch.float32).reshape(1, 80)
        packets = [sample_pairs(shard, [1], rank, 4,
                   generator=torch.Generator().manual_seed(rank))
                   for rank, shard in enumerate(x.chunk(4, -1))]
        packets[0][:, :, 2] = 1e-35
        selected = select_candidates(torch.stack(packets).tolist(), [1])[0]
        self.assertGreaterEqual(selected, 60)
        self.assertEqual(tuple(packets[0].shape), (1, 20, 3))

    def test_same_packets_same_decision_despite_local_rng(self):
        x = torch.randn(32, 80, generator=torch.Generator().manual_seed(6))
        packets = [sample_pairs(shard, [0, 1, 1, 0], rank, 4,
                   generator=torch.Generator().manual_seed(rank+29))
                   for rank, shard in enumerate(x.chunk(4, -1))]
        host = torch.stack(packets).tolist()
        a = select_candidates(host, [0, 1, 1, 0])
        for seed in range(4):
            torch.manual_seed(seed)
            self.assertEqual(a, select_candidates(host, [0, 1, 1, 0]))

    def test_small_vocab_and_greedy(self):
        x = torch.tensor([[-1., 0., .5, 1.]])
        self.assertEqual(self.draw(x, [0], world=2), [3])
        self.assertEqual(self.draw(x, [0], world=1), [3])

    def test_mixed_greedy_ties_keep_lowest_global_id(self):
        x = torch.ones(8, 160)
        ids = self.draw(x, [0, 1], world=4)
        self.assertEqual(ids[:4], [0] * 4)

    def test_validation(self):
        self.assertEqual(temperatures(1., 2), [1., 1.])
        for value in (-1., float('nan'), float('inf'), [0.]):
            with self.assertRaises(ValueError):temperatures(value, 2)
        with self.assertRaises(ValueError):sample_pairs(torch.zeros(3, 80), [0, 1], 0, 1)


if __name__ == '__main__':unittest.main()
