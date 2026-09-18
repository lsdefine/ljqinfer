
import torch, time, os
import ops
m = ops._mod()
torch.manual_seed(0)
T, K, h, D = 12288, 640, 8, 512
start = 49152; ctx = start + T
page = 64
npages = (ctx + page - 1)//page + 8
pool = torch.randn(npages*page, D, device='cuda', dtype=torch.bfloat16)
table = torch.arange(npages, device='cuda', dtype=torch.int64).view(1, -1)
cpool = torch.randn(4096, D, device='cuda', dtype=torch.bfloat16)
ctable = torch.arange(64, device='cuda', dtype=torch.int64).view(1, -1)
sink = torch.zeros(h, device='cuda')
q = torch.randn(1, T, h, D, device='cuda', dtype=torch.bfloat16)
# idxs: window 128 (prev 128 positions) + 512 random topk over [0,pos)
pos = torch.arange(start, ctx, device='cuda')
win = pos[:, None] - torch.arange(128, 0, -1, device='cuda')[None, :]
topk = (torch.rand(T, 512, device='cuda') * pos[:, None].float()).long()
idxs = torch.cat([win, topk], 1).view(1, T, K)
keff = torch.full((T,), K, device='cuda', dtype=torch.int64)
for _ in range(3): o = m.sparse_attn_paged(q, pool, table, cpool, ctable, sink, idxs, ctx, 0.05, keff)
torch.cuda.synchronize()
t0=time.time(); n=20
for _ in range(n): o = m.sparse_attn_paged(q, pool, table, cpool, ctable, sink, idxs, ctx, 0.05, keff)
torch.cuda.synchronize(); dt=(time.time()-t0)/n
print('ms/call %.3f  GB/call %.2f  eff TB/s %.2f' % (dt*1e3, T*K*D*2/1e9, T*K*D*2/dt/1e12))
