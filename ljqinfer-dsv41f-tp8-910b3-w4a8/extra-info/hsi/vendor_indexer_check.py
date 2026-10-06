"""Validate and time the vendor fused lightning indexer against our semantics.

Correctness: the op's top-k set must match a CPU reference of our own scoring
rule sum_h relu(q_h . k) * w_h under causal masking.
Timing: single-dispatch wall clock at the decode shapes we actually run.

Usage (on the A2 box, CANN env sourced):
    python extra-info/hsi/vendor_indexer_check.py
"""
import os
import sys
import time

import torch
import torch_npu  # noqa: F401  (registers the NPU backend)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from ops.decode.vendor_indexer import VendorIndexerPlan  # noqa: E402

HEADS, DIM, BLOCK = 32, 128, 128


def build(batch, q_len, kv_len, sparse_count, seed=0):
    """Allocate one decode-shaped operand set with paged keys."""
    torch.manual_seed(seed)
    dev = 'npu'
    query = torch.randn(batch, q_len, HEADS, DIM, dtype=torch.bfloat16, device=dev)
    weights = torch.randn(batch, q_len, HEADS, dtype=torch.bfloat16, device=dev)
    blocks_per_seq = (kv_len + BLOCK - 1) // BLOCK
    key = torch.randn(batch * blocks_per_seq, BLOCK, 1, DIM,
                      dtype=torch.bfloat16, device=dev)
    block_table = torch.arange(batch * blocks_per_seq, dtype=torch.int32,
                               device=dev).view(batch, blocks_per_seq)
    seq_key = torch.full((batch,), kv_len, dtype=torch.int32, device=dev)
    seq_query = torch.full((batch,), q_len, dtype=torch.int32, device=dev)
    indices = torch.zeros(batch, q_len, 1, sparse_count, dtype=torch.int32, device=dev)
    plan = VendorIndexerPlan(query, key, weights, seq_query, seq_key,
                             block_table, indices, sparse_count)
    return plan, query, key, weights, blocks_per_seq


def check(kv_len=1024, q_len=6, sparse_count=64):
    """Compare the op's selected set with the CPU reference set, per query row."""
    plan, query, key, weights, blocks = build(1, q_len, kv_len, sparse_count)
    plan.run()
    torch.npu.synchronize()
    got = plan.indices[0, :, 0, :].cpu()

    keys = key.view(blocks * BLOCK, DIM)[:kv_len].float().cpu()
    q = query[0].float().cpu()
    w = weights[0].float().cpu()
    ok = True
    for t in range(q_len):
        scores = (torch.relu(q[t] @ keys.T) * w[t, :, None]).sum(0)
        valid = kv_len - q_len + 1 + t          # causal tail alignment
        scores[valid:] = float('-inf')
        want = set(torch.topk(scores, sparse_count).indices.tolist())
        overlap = len(want & set(got[t].tolist()))
        ok &= overlap == sparse_count
        print(f'  t={t} overlap {overlap}/{sparse_count} valid={valid} '
              f'max_index={int(got[t].max())}')
    print('CHECK', 'PASS' if ok else 'FAIL')
    return ok


def bench(kv_lens=(4096, 32768, 131072), batch=1, q_len=6, sparse_count=2048,
          warmup=20, iters=100):
    """Median single-dispatch latency, steady state, synchronised per iteration."""
    results = {}
    for kv_len in kv_lens:
        plan = build(batch, q_len, kv_len, sparse_count)[0]
        for _ in range(warmup):
            plan.run()
        torch.npu.synchronize()
        samples = []
        for _ in range(iters):
            start = time.perf_counter()
            plan.run()
            torch.npu.synchronize()
            samples.append((time.perf_counter() - start) * 1e6)
        samples.sort()
        # Amortised: many launches behind a single sync, so host dispatch cost
        # is pipelined away. This is the number a captured graph would see.
        torch.npu.synchronize()
        start = time.perf_counter()
        for _ in range(iters):
            plan.run()
        torch.npu.synchronize()
        amortised = (time.perf_counter() - start) * 1e6 / iters
        results[kv_len] = (samples[len(samples) // 2], amortised)
        print(f'  kv={kv_len} sparse_count={sparse_count} '
              f'per-dispatch median {samples[len(samples) // 2]:.1f}us '
              f'amortised {amortised:.1f}us')
    return results


if __name__ == '__main__':
    torch.npu.set_device(0)
    print('correctness vs CPU reference:')
    passed = check()
    print('latency sweep:')
    bench()
    sys.exit(0 if passed else 1)
