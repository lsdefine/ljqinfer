"""Which part of the last six key rows does the op disagree about?

Experiment A makes all six query rows identical, so a row-to-row misalignment
of the query cannot show up; anything left is a key-read problem.
"""
import sys, torch, torch_npu

sys.path.insert(0, '/data/ljqinfer_dsv41f_tp8')
from ops.decode.vendor_indexer import VendorIndexerPlan

B, R, H, D, BLK, SC = 1, 6, 32, 128, 128, 512
dev = torch.device('npu:0')
torch.manual_seed(0)
N = 4000                                     # committed rows
blocks = (N + 6 + BLK - 1) // BLK + 1
keys = torch.randn(blocks, BLK, 1, D, dtype=torch.bfloat16, device=dev)
table = torch.arange(blocks, dtype=torch.int32, device=dev)[None, :]
seq_q = torch.full((B,), R, dtype=torch.int32, device=dev)
seq_k = torch.full((B,), N + R, dtype=torch.int32, device=dev)

for tag in ('identical-rows', 'distinct-rows'):
    q = torch.randn(B, R, H, D, dtype=torch.bfloat16, device=dev)
    if tag == 'identical-rows':
        q[:] = q[:, :1]
    w = torch.ones(B, R, H, dtype=torch.bfloat16, device=dev)
    idx = torch.zeros(B, R, 1, SC, dtype=torch.int32, device=dev)
    VendorIndexerPlan(q, keys, w, seq_q, seq_k, table.repeat(B, 1), idx,
                      sparse_count=SC, sparse_mode=3).run()
    torch.npu.synchronize()
    flat = keys.view(-1, D).float()[:N + R]
    bad = []
    for j in range(R):
        s = torch.einsum('hd,kd->hk', q[0, j].float(), flat).relu().sum(0)[:N + j + 1]
        want = set(s.topk(SC).indices.tolist())
        got = set(idx[0, j, 0].tolist())
        bad.append((j, sorted(want - got)[:3], sorted(got - want)[:3]))
    print(tag, 'exact rows:', [j for j, m, e in bad if not m and not e])
    for j, m, e in bad:
        if m or e:
            print(f'  row{j} miss={m} extra={e}')
print('DONE')
