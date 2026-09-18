import torch, time, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import ops
m = ops._mod()
torch.manual_seed(0)
b, s, h, d = 1, 8, 8, 512
K = 2048; npages = 4; page = 2048
q = torch.randn(b, s, h, d, device='cuda', dtype=torch.bfloat16)
pool = torch.randn(npages, page, d, device='cuda', dtype=torch.bfloat16)
table = torch.arange(npages, device='cuda').view(1, -1)
sink = torch.randn(h, device='cuda')
idxs = torch.randint(0, 4096, (b, s, K), device='cuda')
idxs[0, 0, 100:] = -1
total = 4096
T = b * s
o = m.sparse_attn_paged(q, pool, table, pool, table, sink, idxs, total, 0.05, None)
# keff == K must be bit-identical to keff == None (no truncation requested)
kf = torch.full((T,), K, device='cuda', dtype=torch.int64)
o_full = m.sparse_attn_paged(q, pool, table, pool, table, sink, idxs, total, 0.05, kf)
assert torch.equal(o, o_full), 'keff=K must match keff=None bitwise'
# row 0 has only 100 live ids (idxs[0,0,100:] == -1); truncating there must also
# be bit-identical -- this is the core correctness claim of the keff optimisation.
kt = kf.clone(); kt[0] = 100
o_tr = m.sparse_attn_paged(q, pool, table, pool, table, sink, idxs, total, 0.05, kt)
assert torch.equal(o, o_tr), 'keff truncation changed results'
print('KEFF EQUIV OK (None == K == truncated)')


def ref():
    ids = idxs.clone(); valid = ids >= 0; ids[~valid] = 0
    kv = pool.view(-1, d)[ids]
    sc = torch.einsum('bshd,bskd->bshk', q.float(), kv.float()) * 0.05
    sc = sc.masked_fill(~valid[:, :, None, :], float('-inf'))
    sc = torch.cat([sc, sink.view(1, 1, h, 1).expand(b, s, h, 1)], -1)
    p = torch.softmax(sc, -1)[..., :K]
    return torch.einsum('bshk,bskd->bshd', p, kv.float())


r = ref()
print('SPLITK', 8, '(compile-time fixed)', 'max abs diff vs torch ref:',
      (o.float() - r).abs().max().item(), 'ref scale', r.abs().max().item())
torch.cuda.synchronize()
for _ in range(5): m.sparse_attn_paged(q, pool, table, pool, table, sink, idxs, total, 0.05, kt)
torch.cuda.synchronize(); t = time.time()
for _ in range(200): m.sparse_attn_paged(q, pool, table, pool, table, sink, idxs, total, 0.05, kt)
torch.cuda.synchronize(); print('us/call', (time.time() - t) / 200 * 1e6)
