import argparse
import time
import torch
import torch_npu

from model.config import EngineConfig
from model.decode import decode_step
from model.model import Engine
from model.prefill import prefill
from model.spmd import SPMDRuntime

p = argparse.ArgumentParser()
p.add_argument("--rank", type=int, required=True)
p.add_argument("--device-base", type=int, default=4)
a = p.parse_args()

root = "/dev/shm/qtp4_production_root.bin"
rt = SPMDRuntime(a.rank, device_index=a.device_base + a.rank, root_file=root)
rt.communicator()
# Two active sequences each need at least one physical KV page even when their
# prompts are short; total capacity therefore needs two 2048-token pages.
cfg = EngineConfig(max_cached_tokens=4096, max_sequences=2,
                   prefill_chunk_size=1024, mtp_backend="disabled")
engine = Engine.load(a.rank, cfg, device=str(rt.device), collective=rt)

started = time.time()
h0 = prefill(engine, [1, 2], sequence_id=0, return_all_hidden=False)
h1 = prefill(engine, [3, 4, 5], sequence_id=1, return_all_hidden=False)
prefill_seconds = time.time() - started
before = [int(x) for x in engine.cache.lengths.cpu().tolist()]

step_started = time.time()
local_logits = decode_step(
    engine,
    torch.tensor([6, 7], dtype=torch.int64, device=rt.device),
    torch.tensor([0, 1], dtype=torch.int64, device=rt.device),
    return_logits=True,
)
rt.synchronize()
decode_seconds = time.time() - step_started

gathered_logits = rt.all_gather(local_logits.contiguous())
rt.synchronize()
global_logits = gathered_logits.permute(1, 0, 2).reshape(2, -1)
tokens = global_logits.float().argmax(dim=-1)
token_probe = rt.all_gather(tokens.to(torch.float32).contiguous())
rt.synchronize()

lengths = [int(x) for x in engine.cache.lengths.cpu().tolist()]
pages = engine.cache.page_table[:, 0].cpu().tolist()
gdn_delta = float((engine.cache.hot_gdn_recurrent[:, 0] -
                   engine.cache.hot_gdn_recurrent[:, 1]).abs().mean().cpu())
finite = bool(torch.isfinite(global_logits).all().cpu())
token_rank_diff = float((token_probe - token_probe[0:1]).abs().max().cpu())

print(
    f"rank={a.rank} prefill_hidden={tuple(h0.shape)},{tuple(h1.shape)} "
    f"before={before} after={lengths} pages={pages} "
    f"logits={tuple(global_logits.shape)} finite={finite} "
    f"tokens={tokens.cpu().tolist()} token_rank_diff={token_rank_diff} "
    f"gdn_sid_delta={gdn_delta:.8f} "
    f"prefill_seconds={prefill_seconds:.4f} decode_seconds={decode_seconds:.4f} "
    f"total_seconds={time.time()-started:.4f}",
    flush=True,
)
if before != [2, 3] or lengths != [3, 4]:
    raise RuntimeError(f"length state mismatch before={before} after={lengths}")
if pages[0] < 0 or pages[1] < 0 or pages[0] == pages[1]:
    raise RuntimeError(f"sequence pages are not independent: {pages}")
if not finite or token_rank_diff != 0.0 or gdn_delta == 0.0:
    raise RuntimeError(
        f"production gate failed finite={finite} token_diff={token_rank_diff} "
        f"gdn_delta={gdn_delta}"
    )
rt.destroy()
