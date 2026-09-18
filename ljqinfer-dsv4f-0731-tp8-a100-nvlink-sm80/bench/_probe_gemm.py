
import torch, sys
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # repo root on sys.path (this file lives in bench/)
import ops
mod = ops._mod()
torch.manual_seed(0)
dev = "cuda:0"
for (N, K) in [(2048, 7168), (4096, 2048)]:
    w = (torch.randn(N, K, device=dev, dtype=torch.bfloat16) * 0.02).contiguous()
    wf = w.float()
    print("\n=== w[N=%d,K=%d]  K%%256==%d" % (N, K, K & 255), flush=True)
    for M in [8, 16, 32]:
        x = (torch.randn(M, K, device=dev, dtype=torch.bfloat16) * 0.05).contiguous()
        ref = (x.float() @ wf.t())
        y_dispatch = mod.bf16_gemm(x, w)
        y_k8 = torch.cat([mod.bf16_gemm(x[i:i+8].contiguous(), w) for i in range(0, M, 8)], 0)
        y_cublas = torch.matmul(x, w.t())
        d = lambda a: (a.float() - ref).abs().max().item()
        rel = lambda a: ((a.float()-ref).abs().max()/ref.abs().max()).item()*100
        same = bool((y_dispatch == y_k8).all().item())
        print(" M=%3d | dispatch_vs_fp32=%.6f (%.3f%%)  k8_vs_fp32=%.6f (%.3f%%)  cublas_vs_fp32=%.6f | dispatch==k8? %s | k8_vs_cublas=%.6f"
              % (M, d(y_dispatch), rel(y_dispatch), d(y_k8), rel(y_k8), d(y_cublas), same,
                 (y_k8.float()-y_cublas.float()).abs().max().item()), flush=True)
print("PROBE_DONE", flush=True)
