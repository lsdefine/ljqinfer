"""Gate checks before wiring the vendor indexer into decode.

The index bank pages 2048 rows, but the op rejects block_size=2048, so this
sweeps the accepted block sizes (a page is contiguous, hence it can be
re-expressed as several smaller blocks) and then proves the dispatch survives
NPUGraph capture and replay at the decode sparse_count of 512.
"""
import os, sys, statistics, torch, torch_npu

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from ops.decode.vendor_indexer import VendorIndexerPlan

DEV, B, S, N, D = 'npu:0', 1, 6, 32, 128


def build(kv_len, block_size, sparse_count):
    blocks = (kv_len + block_size - 1) // block_size
    key = torch.randn(blocks, block_size, 1, D, dtype=torch.bfloat16, device=DEV)
    query = torch.randn(B, S, N, D, dtype=torch.bfloat16, device=DEV)
    weights = torch.randn(B, S, N, dtype=torch.bfloat16, device=DEV).abs()
    table = torch.arange(blocks, dtype=torch.int32, device=DEV).view(B, blocks)
    seq_q = torch.full((B,), S, dtype=torch.int32, device=DEV)
    seq_k = torch.full((B,), kv_len, dtype=torch.int32, device=DEV)
    idx = torch.zeros(B, S, 1, sparse_count, dtype=torch.int32, device=DEV)
    plan = VendorIndexerPlan(query, key, weights, seq_q, seq_k, table, idx, sparse_count)
    return plan, idx


def timed(fn, iters=30):
    for _ in range(5):
        fn()
    torch.npu.synchronize()
    samples = []
    for _ in range(iters):
        torch.npu.synchronize()
        a = torch.npu.Event(enable_timing=True); b = torch.npu.Event(enable_timing=True)
        a.record(); fn(); b.record(); torch.npu.synchronize()
        samples.append(a.elapsed_time(b) * 1e3)
    return statistics.median(samples)


print('--- accepted block sizes (kv=32768, sparse_count=512) ---', flush=True)
for block in (16, 32, 64, 128, 256, 512, 1024):
    try:
        plan, idx = build(32768, block, 512)
        us = timed(lambda: plan.run())
        row = idx[0, -1, 0]
        print(f'  block={block:5d} ok  median {us:7.1f}us valid={int(((row >= 0) & (row < 32768)).sum())}/512', flush=True)
    except RuntimeError as exc:
        print(f'  block={block:5d} rejected ({str(exc).split(";")[0].split(": ")[-1]})', flush=True)

print('--- sparse_count at 128K, block=128 ---', flush=True)
for sc in (512, 2048):
    plan, idx = build(131072, 128, sc)
    print(f'  sc={sc} median {timed(lambda: plan.run()):.1f}us', flush=True)

print('--- graph capture and replay (128K, block=128, sc=512) ---', flush=True)
plan, idx = build(131072, 128, 512)
plan.run(); torch.npu.synchronize()
reference = idx.clone()
idx.zero_()
graph = torch_npu.npu.NPUGraph()
stream = torch_npu.npu.Stream()
try:
    with torch_npu.npu.graph(graph, stream=stream):
        plan.run()
except Exception as exc:
    print(f'  CAPTURE FAILED: {type(exc).__name__}: {exc}', flush=True)
    raise SystemExit(1)
print('  capture ok', flush=True)
for i in range(3):
    idx.zero_()
    graph.replay()
    torch.npu.synchronize()
    print(f'  replay {i}: identical_to_eager={bool(torch.equal(idx, reference))} '
          f'valid={int(((idx >= 0) & (idx < 131072)).sum())}/{512 * S}', flush=True)
print(f'  replay median {timed(lambda: graph.replay()):.1f}us', flush=True)
print('DONE', flush=True)
