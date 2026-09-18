import argparse
import time
import torch
import torch_npu

from model.blocks import block_forward
from model.config import CONFIG, EngineConfig
from model.model import Engine
from model.spmd import SPMDRuntime
from ops.kernels import K

p = argparse.ArgumentParser()
p.add_argument("--rank", type=int, required=True)
p.add_argument("--device-base", type=int, default=4)
p.add_argument("--layers", type=int, default=64)
a = p.parse_args()

device_index = a.device_base + a.rank
root = f"/dev/shm/qtp4_engine{a.layers}_root.bin"
rt = SPMDRuntime(a.rank, world=4, device_index=device_index, root_file=root)
cfg = EngineConfig(max_cached_tokens=4096, max_sequences=2,
                   prefill_chunk_size=1024, mtp_backend="disabled")
engine = Engine.load(a.rank, cfg, device=f"npu:{device_index}", collective=rt)
input_ids = torch.tensor([1], dtype=torch.int64, device=engine.device)
sids = torch.zeros(1, dtype=torch.int64, device=engine.device)
positions = torch.zeros(1, dtype=torch.int64, device=engine.device)
hidden = K.embedding(input_ids, engine.weights.embedding)
old_lengths = {0: 0}
total_start = time.time()
for layer_idx in range(a.layers):
    start = time.time()
    ctx = engine.cache.layer_context(layer_idx, sids, positions, update_kv=True)
    hidden = block_forward(hidden, layer_idx, engine.weights.layer(layer_idx),
                           ctx, engine.all_reduce)
    if layer_idx in CONFIG.full_attention_layers:
        engine.cache.commit_full_attention(layer_idx, ctx, old_lengths)
    rt.synchronize()
    print(f"rank={a.rank} layer={layer_idx} type="
          f"{'full' if layer_idx in CONFIG.full_attention_layers else 'gdn'} "
          f"seconds={time.time()-start:.4f} "
          f"meanabs={float(hidden.float().abs().mean().cpu()):.7f}", flush=True)
hidden = K.rms_norm(hidden, engine.weights.final_norm, CONFIG.rms_norm_eps)
engine.cache.lengths[0] = 1
rt.synchronize()
gathered = rt.all_gather(hidden)
rt.synchronize()
max_diff = float((gathered - gathered[0]).float().abs().max().cpu())
finite = bool(torch.isfinite(gathered).all().cpu())
print(f"rank={a.rank} engine_layers={a.layers} shape={tuple(hidden.shape)} "
      f"finite={finite} max_rank_diff={max_diff} "
      f"seconds={time.time()-total_start:.4f} length="
      f"{int(engine.cache.lengths[0].cpu())} "
      f"meanabs={float(hidden.float().abs().mean().cpu()):.7f}", flush=True)
if not finite or max_diff != 0.0:
    raise RuntimeError(f"TP4 engine mismatch finite={finite} diff={max_diff}")
rt.destroy()
