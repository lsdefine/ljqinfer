"""Leaf gate: ops.sparse_attn_flat (broken TC kernel) vs kernels_torch.sparse_attn (ref).
Run on GPU: python tests/test_sparse_attn_leaf.py"""
import sys, time, torch
sys.path.insert(0, '.')
import ops
from model import kernels_torch as K

torch.manual_seed(0)
dev = 'cuda'

def case(b, s, n, c, h, topk, dtype=torch.bfloat16):
    q = (torch.randn(b, s, h, 512, device=dev) * 0.5).to(dtype)
    kv = (torch.randn(b, n, 512, device=dev) * 0.5).to(dtype)
    kvc = (torch.randn(b, c, 512, device=dev) * 0.5).to(dtype) if c else None
    tot = n + c
    idx = torch.randint(0, tot, (b, s, topk), device=dev)
    idx[..., topk // 3:] = torch.where(torch.rand(b, s, topk - topk // 3, device=dev) < 0.2,
                                       torch.full_like(idx[..., topk // 3:], -1), idx[..., topk // 3:])
    idx[:, 0, 1:] = -1                       # near-empty row
    sink = torch.randn(h, device=dev) * 0.3
    scale = 512 ** -0.5
    kvcat = kv if kvc is None else torch.cat([kv, kvc], 1)
    ref = K.sparse_attn(q, kvcat, sink, idx.int(), scale)
    torch.cuda.synchronize()
    out = ops.sparse_attn_flat(q, kv, sink, idx.int(), scale, kvc)
    torch.cuda.synchronize()
    err = (out.float() - ref.float()).abs()
    rel = err.max().item() / (ref.float().abs().max().item() + 1e-9)
    # timing
    for f, name in ((lambda: K.sparse_attn(q, kvcat, sink, idx.int(), scale), 'ref'),
                    (lambda: ops.sparse_attn_flat(q, kv, sink, idx.int(), scale, kvc), 'leaf')):
        f(); torch.cuda.synchronize(); t0 = time.time()
        for _ in range(5): f()
        torch.cuda.synchronize(); print(f'    {name}: {(time.time()-t0)/5*1e3:.2f} ms')
    print(f'b={b} s={s} n={n} c={c} h={h} K={topk}: maxabs={err.max().item():.3e} mean={err.mean().item():.3e} rel={rel:.3e}')
    return rel

worst = 0
for args in [(1, 64, 64, 0, 8, 64), (1, 300, 300, 3, 8, 2048), (2, 128, 128, 1, 8, 1024), (1, 1, 4096, 32, 8, 2048)]:
    worst = max(worst, case(*args))
print('WORST REL', worst, 'PASS' if worst < 2e-2 else 'FAIL')
