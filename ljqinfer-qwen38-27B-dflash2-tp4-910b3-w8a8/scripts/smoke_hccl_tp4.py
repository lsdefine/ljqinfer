import argparse
import torch
import torch_npu
from model.spmd import SPMDRuntime

p = argparse.ArgumentParser()
p.add_argument("--rank", type=int, required=True)
p.add_argument("--device-base", type=int, default=0)
a = p.parse_args()
rt = SPMDRuntime(a.rank, world=4, device_index=a.device_base + a.rank,
                 root_file="/dev/shm/qtp4_smoke_root.bin")
x = torch.full((16,), float(a.rank + 1), dtype=torch.bfloat16,
               device=rt.device)
rt.all_reduce(x)
rt.synchronize()
value = float(x[0].cpu())
print(f"rank={a.rank} value={value} finite={bool(torch.isfinite(x).all().cpu())}",
      flush=True)
if value != 10.0:
    raise RuntimeError(f"rank {a.rank}: expected 10, got {value}")
rt.destroy()
