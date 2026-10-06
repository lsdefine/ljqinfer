"""Is the fused indexer's output ordered by score, and where do ties land?

Decode needs the six pending rows in the final id set. If the op returns its
positions score-descending, the cheapest exact-enough wiring overwrites the
weakest entries; that only holds if the ordering is really by score.
"""
import sys, torch, torch_npu

sys.path.insert(0, '/data/ljqinfer_dsv41f_tp8')
from ops.decode.vendor_indexer import VendorIndexerPlan

B, S, N, D, KV, BLK, SC = 1, 6, 32, 128, 8192, 128, 512
dev = torch.device('npu:0')
torch.manual_seed(0)
blocks = KV // BLK
keys = torch.randn(blocks, BLK, 1, D, dtype=torch.bfloat16, device=dev)
q = torch.randn(B, S, N, D, dtype=torch.bfloat16, device=dev)
w = torch.randn(B, S, N, dtype=torch.bfloat16, device=dev)
table = torch.arange(blocks, dtype=torch.int32, device=dev).view(B, blocks)
seq_q = torch.full((B,), S, dtype=torch.int32, device=dev)
seq_k = torch.full((B,), KV, dtype=torch.int32, device=dev)

idx = torch.zeros(B, S, 1, SC, dtype=torch.int32, device=dev)
plan = VendorIndexerPlan(q, keys, w, seq_q, seq_k, table, idx, SC)
plan.run()
torch.npu.synchronize()

# Reference scores for the last query row, computed the way the op defines them.
scores = (torch.einsum('nd,kd->nk', q[0, -1].float(), keys.view(-1, D).float())
          .relu() * w[0, -1].float()[:, None]).sum(0)
picked = idx[0, -1, 0].long()
vals = scores[picked]
print('first 8 ids  ', picked[:8].tolist())
print('first 8 score', [round(v, 3) for v in vals[:8].tolist()])
print('last 8 score ', [round(v, 3) for v in vals[-8:].tolist()])
print('descending:', bool(torch.all(vals[:-1] >= vals[1:])))
print('equals exact top-512 set:',
      bool(torch.equal(picked.sort().values, scores.topk(SC).indices.sort().values)))
print('DONE')
