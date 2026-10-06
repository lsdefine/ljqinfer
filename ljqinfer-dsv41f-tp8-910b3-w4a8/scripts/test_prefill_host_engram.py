"""CPU-only Engram regression: PYTHONPATH=. python scripts/test_prefill_host_engram.py.

No model weights, NPU allocation, or engine required. The full hash and Torch
INT8 dequantization are independent references for the selected native path.
"""
import json
from types import SimpleNamespace

import torch

from model.engram import EngramHash
from ops.prefill.host_engram import gather


def main():
    torch.set_num_threads(1)
    gen = torch.Generator().manual_seed(2701)
    counts = dict(gather=0, hash=0, rejected=0)

    def check(weight, scale, ids):
        ref = (weight[ids].float().unflatten(-1, (8, 32)) *
               scale[ids][..., None]).flatten(-2).to(torch.bfloat16)
        got = gather(weight, scale, ids)
        assert got.shape == ref.shape
        assert torch.equal(got.view(torch.int16), ref.view(torch.int16))
        counts['gather'] += 1

    def rejects(fn):
        try:
            fn()
        except ValueError:
            counts['rejected'] += 1
        else:
            raise AssertionError('invalid input accepted')

    weight = torch.randint(-128, 128, (521, 256), dtype=torch.int8, generator=gen)
    scale = torch.randn(521, 8, generator=gen)
    for n in (0, 1, 7, 127, 1023, 1024, 3071, 3072, 8192):
        ids = torch.randint(521, (n, 6), generator=gen)[:, ::2]
        check(weight, scale, ids)
    check(weight, scale, torch.tensor(520))
    check(weight, scale, torch.tensor([0, 520, 0, 520]))

    # Every INT8 value with signed zeros, tie-sensitive values, and nonfinite scales.
    values = torch.arange(-128, 128, dtype=torch.int16).to(torch.int8)
    edge = torch.tensor([0., -0., 1., -1., 1.00390625, 1.e-38,
                         float('inf'), float('-inf'), float('nan')])
    ew = values.repeat(len(edge), 1)
    es = edge[:, None].expand(-1, 8).contiguous()
    check(ew, es, torch.arange(len(edge)))
    for ids in (torch.tensor([-1]), torch.tensor([521])):
        rejects(lambda ids=ids: gather(weight, scale, ids))
    rejects(lambda: gather(weight, scale, torch.tensor([0.0])))
    rejects(lambda: gather(weight[:, ::2], scale, torch.tensor([0])))
    rejects(lambda: gather(weight, scale.double(), torch.tensor([0])))

    h = EngramHash.__new__(EngramHash)
    h.layout = SimpleNamespace(max_ngram_size=4, n_heads=8)
    h.token_map = torch.arange(97, dtype=torch.int64) % 31
    h.token_map[::11] = -1
    h.pad_id = 3
    h.primes = torch.tensor([101, 103, 107, 109, 113, 127, 131, 137]).repeat(2, 3, 1)
    flat = h.primes.flatten(1)
    h.offsets = flat.cumsum(-1) - flat
    h.multipliers = torch.randint(1, 2**62, (2, 4), generator=gen)
    for start in (0, 1, 2, 3, 8192):
        for n in (0, 1, 7, 127, 1024):
            tokens = torch.randint(97, (n,), generator=gen).tolist()
            history = torch.randint(97, (min(start, 3),), generator=gen).tolist()
            full = h(tokens, start=start, history_tokens=history)
            for layer in (0, 1):
                for rank in (None, *range(8)):
                    expected = full[:, layer]
                    if rank is not None:
                        expected = expected[:, rank * 3:(rank + 1) * 3]
                    got = h.selected_ids(layer, rank, tokens, start=start, history_tokens=history)
                    assert torch.equal(got, expected), (start, n, layer, rank)
                    counts['hash'] += 1
    rejects(lambda: h.selected_ids(0, 8, [1], start=0))
    rejects(lambda: h.selected_ids(0, 0, [97], start=0))
    rejects(lambda: h.selected_ids(0, 0, [1], start=1))
    print(json.dumps(dict(complete=True, **counts)))


if __name__ == '__main__':
    main()
