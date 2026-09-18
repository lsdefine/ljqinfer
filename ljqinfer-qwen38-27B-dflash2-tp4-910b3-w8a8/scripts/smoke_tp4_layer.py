import argparse
import time
import torch
import torch_npu

from model.blocks import block_forward
from model.config import EngineConfig
from model.model import Engine
from model.spmd import SPMDRuntime

p = argparse.ArgumentParser()
p.add_argument("--rank", type=int, required=True)
p.add_argument("--device-base", type=int, default=4)
a = p.parse_args()
device_index = a.device_base + a.rank
rt = SPMDRuntime(a.rank, world=4, device_index=device_index,
                 root_file="/dev/shm/qtp4_layer_root.bin")
cfg = EngineConfig(max_cached_tokens=4096, max_sequences=2,
                   prefill_chunk_size=1024, mtp_backend="disabled")
engine = Engine.load(a.rank, cfg, device=f"npu:{device_index}", collective=rt)
torch.manual_seed(20260824)
hidden = torch.randn((1, engine.config.hidden_size), dtype=torch.bfloat16,
                     device=engine.device)
# Force bit-identical replicated input independently on every rank.
hidden.fill_(0)
hidden[:, 0] = 1
positions = torch.zeros(1, dtype=torch.int64, device=engine.device)
sids = torch.zeros(1, dtype=torch.int64, device=engine.device)
ctx = engine.cache.layer_context(0, sids, positions, update_kv=False)
t0 = time.time()
out = block_forward(hidden, 0, engine.weights.layer(0), ctx, engine.all_reduce)
rt.synchronize()
dt = time.time() - t0
gathered = rt.all_gather(out)
rt.synchronize()
ref = gathered[0]
max_diff = float((gathered - ref).float().abs().max().cpu())
finite = bool(torch.isfinite(gathered).all().cpu())
print(f"rank={a.rank} shape={tuple(out.shape)} finite={finite} "
      f"max_rank_diff={max_diff} seconds={dt:.4f} "
      f"meanabs={float(out.float().abs().mean().cpu()):.7f}", flush=True)
if not finite or max_diff != 0.0:
    raise RuntimeError(f"TP4 layer mismatch finite={finite} diff={max_diff}")
rt.destroy()
