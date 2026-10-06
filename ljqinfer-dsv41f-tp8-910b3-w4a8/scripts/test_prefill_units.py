"""Single-NPU leaf acceptance; no model weights/server required.
Run: PYTHONPATH=. python scripts/test_prefill_units.py --device 0
Tests frozen composition parity; use real captured weights for GMM tuning.
"""
import argparse
import json
import statistics
import torch
import torch_npu
from ops.prefill import residual as ref
from ops.prefill.native import ops
from ops.prefill.hc import hc_prepare, hc_finish
from ops.prefill.moe_units import dispatch_quant, activation_quant, combine


def exact(a, b):
    if isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            exact(x, y)
    else:
        assert a.shape == b.shape and a.dtype == b.dtype
        assert torch.isfinite(a).all().item()
        assert torch.equal(a, b), (a.float()-b.float()).abs().max().item()


def timed(fn):
    for _ in range(3):
        fn()
    torch.npu.synchronize()
    base = torch.npu.memory_allocated()
    torch.npu.reset_peak_memory_stats()
    samples = []
    for _ in range(9):
        a, b = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
        a.record()
        fn()
        b.record()
        b.synchronize()
        samples.append(a.elapsed_time(b))
    return {'median_ms': statistics.median(samples),
            'peak_extra_bytes': torch.npu.max_memory_allocated()-base}


def run(device):
    torch.npu.set_device(device)
    torch.manual_seed(4201)
    result = []
    for t in (1, 127, 129, 513):
        h = torch.randn(t, 4, 5120, device='npu', dtype=torch.bfloat16)
        incoming = torch.rand(t, 4, device='npu')
        fn = torch.randn(24, 20480, device='npu') * 0.002
        scale = torch.ones(3, device='npu')
        base = torch.zeros(24, device='npu')
        norm = torch.ones(5120, device='npu', dtype=torch.bfloat16)
        kwargs = dict(norm_eps=1e-6, hc_eps=1e-6, iters=20)
        call = lambda: hc_prepare(h, incoming, fn, scale, base, norm, **kwargs)
        actual = call()
        gates = ref.mixes(h, fn, scale, base, **kwargs)
        exact(actual, (ref.collapse_norm(h, incoming, norm, 1e-6), *gates))
        x, _, post, comb = actual
        exact(hc_finish(x, h, post, comb), ref.expand(x, h, post, comb))
        result.append({'op': 'hc_prepare', 'rows': t, **timed(call), 'exact': True})
        del h, fn, actual, gates
        # Skewed routes leave empty experts and repeat IDs across rows.
        ids = torch.arange(t*6, device='npu').reshape(t, 6).remainder(7)
        actual = dispatch_quant(x, ids, experts=16)
        order = ids.flatten().float().argsort(stable=True)
        raw, scales = ops.dynamic_quant(x)
        counts = torch.zeros(16, dtype=torch.int64, device='npu')
        counts.scatter_add_(0, ids.flatten()[order], torch.ones_like(order))
        exact(actual, (raw[order//6].contiguous(), scales[order//6].contiguous(), counts, order))
        hidden = torch.randn(t*6, 576, device='npu', dtype=torch.bfloat16)
        prob = torch.rand(t*6, device='npu')
        gated = ops.routed_swiglu(hidden, prob)
        q, qs = ops.dynamic_quant(gated)
        padded = torch.zeros(t*6, 320, dtype=torch.int8, device='npu')
        padded[:, :288].copy_(q)
        exact(activation_quant(hidden, prob, padded_dim=320), (padded, qs))
        exact(activation_quant(hidden, prob, padded_dim=288), (q, qs))
        values = torch.randn(t*6, 5120, device='npu', dtype=torch.bfloat16)
        inverse = torch.empty_like(order)
        inverse.scatter_(0, order, torch.arange(t*6, device='npu'))
        exact(combine(values, order), ops.routed_combine(values, inverse))
        # Integer values also admit an independent exact FP32 sum oracle.
        integer = values.round()
        expected = integer[inverse].float().view(t, 6, 5120).sum(1)
        exact(combine(integer, order), expected)
        result.append({'op': 'dispatch_quant', 'rows': t,
                       **timed(lambda: dispatch_quant(x, ids, experts=16)), 'exact': True})
        result.append({'op': 'activation_quant', 'rows': t,
                       **timed(lambda: activation_quant(hidden, prob, padded_dim=320)), 'exact': True})
        result.append({'op': 'combine', 'rows': t,
                       **timed(lambda: combine(values, order)), 'exact': True})
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--device', type=int, default=0)
    p.add_argument('--output', default='/tmp/prefill_units.json')
    args = p.parse_args()
    with torch.inference_mode():
        result = run(args.device)
    with open(args.output, 'w') as f:
        json.dump({'pass': True, 'tests': result}, f, indent=2)
    print(json.dumps(result, indent=2))
