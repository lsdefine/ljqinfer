import os, time, torch, torch.distributed as dist
import ops
dist.init_process_group('nccl'); rk = dist.get_rank(); torch.cuda.set_device(rk); dev = f'cuda:{rk}'
torch.manual_seed(0)
b, s, h, d, n, k = 1, int(os.environ.get('S', '5805')), 8, 128, int(os.environ.get('N', '67000')), 2048
q = torch.randn(b, s, h, d, device=dev, dtype=torch.bfloat16)
kv = torch.randn(b, n, d, device=dev, dtype=torch.bfloat16)
w = torch.rand(b, s, h, device=dev, dtype=torch.bfloat16)
torch.manual_seed(rk + 1); q = q + 0.01 * torch.randn_like(q)  # per-rank shard differs
def run(flag):
    ops._INDEX_TOPK_ROWSHARD = flag
    torch.cuda.synchronize(); dist.barrier(); t = time.time()
    out = ops.index_topk(q, kv, w, 1, n - s, 0, k)
    torch.cuda.synchronize(); dist.barrier(); return out, time.time() - t
a, ta = run(False); a2, _ = run(False)
bb, tb = run(True)
assert a.shape == bb.shape == (b, s, k), (a.shape, bb.shape)
sa = a.sort(-1)[0]; sb = bb.sort(-1)[0]
row_eq = (sa == sb).all(-1).float().mean().item()
elem_eq = (sa == sb).float().mean().item()
if rk == 0:
    print(f'S={s} N={n} AR={ta*1000:.1f}ms RS={tb*1000:.1f}ms  AR-repeat-eq={(a==a2).all().item()}  row_eq={row_eq:.4f} elem_eq={elem_eq:.6f}')
dist.destroy_process_group()
