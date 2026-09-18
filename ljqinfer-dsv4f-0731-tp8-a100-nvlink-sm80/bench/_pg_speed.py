
import torch, sys, time
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # repo root on sys.path (this file lives in bench/)
import ops
mod = ops._mod()
torch.manual_seed(0)
dev = "cuda:0"

def bench(fn, it=200, wu=30):
    for _ in range(wu): fn()
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
    e0.record()
    for _ in range(it): fn()
    e1.record(); torch.cuda.synchronize()
    return e0.elapsed_time(e1)/it*1000   # us

print("shape                 M  | dispatch(now)   k8-grouped      cuBLAS      | acc: k8%  cublas%  -> verdict", flush=True)
for (N, K, tag) in [(2048,7168,"sw1/sw3"), (7168,2048,"sw2"), (4096,2048,"ctl")]:
    w = (torch.randn(N, K, device=dev, dtype=torch.bfloat16)*0.02).contiguous()
    wf = w.float()
    for M in [8,16,24,32]:
        x = (torch.randn(M, K, device=dev, dtype=torch.bfloat16)*0.05).contiguous()
        ref = x.float() @ wf.t()
        f_dis = lambda: mod.bf16_gemm(x, w)
        f_k8  = lambda: torch.cat([mod.bf16_gemm(x[i:i+8].contiguous(), w) for i in range(0,M,8)], 0)
        f_cub = lambda: torch.matmul(x, w.t())
        t_dis, t_k8, t_cub = bench(f_dis), bench(f_k8), bench(f_cub)
        rel = lambda a: ((a.float()-ref).abs().max()/ref.abs().max()).item()*100
        r_k8, r_cub = rel(f_k8()), rel(f_cub())
        win = "k8 FASTER" if t_k8 < t_dis*0.99 else ("k8 slower x%.2f" % (t_k8/t_dis))
        print("%-8s[%4d,%4d] %3d | %8.1fus %8.1fus %8.1fus | %6.3f %6.3f  -> %s"
              % (tag,N,K,M,t_dis,t_k8,t_cub,r_k8,r_cub,win), flush=True)
print("SPEED_DONE", flush=True)
