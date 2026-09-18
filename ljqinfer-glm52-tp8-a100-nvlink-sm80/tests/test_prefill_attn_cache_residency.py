#!/usr/bin/env python3
"""Cold-process TP8 regression for provider-private prefill Q8 cache residency.

Run on an idle 8-GPU node with the production model available:
    python tests/test_prefill_attn_cache_residency.py

The selected prefill provider embeds q8_cublas.cu and therefore owns a cache
that K.q8.clear_weight_cache() cannot reach.  A short prompt still traverses
all 78 layers and catches the old ~5 GiB/rank residency leak.
"""
import json
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.0")

from model.model import Engine
from model.prefill import prefill

PROMPT_TOKENS = 16
MAX_DELTA_MIB_PER_RANK = 256.0


def allocated(engine: Engine) -> list[int]:
    for device in engine.rt.devices:
        torch.cuda.synchronize(device)
    return [torch.cuda.memory_allocated(device) for device in engine.rt.devices]


def main() -> None:
    selected = json.loads((ROOT / "ops/selected_ops.lock.json").read_text())["operators"]["prefill_attn"]
    print(
        "PREFILL_PROVIDER "
        f"recipe={selected['recipe_id']} sha256={selected['sha256']} "
        f"artifact={selected['artifact']}",
        flush=True,
    )

    engine = Engine.load()
    before = allocated(engine)
    tokens = torch.arange(PROMPT_TOKENS, dtype=torch.long, device=engine.rt.devices[0])
    residual = prefill(engine, tokens)
    after = allocated(engine)

    if tuple(residual.shape) != (PROMPT_TOKENS, 6144):
        raise AssertionError(f"unexpected prefill residual shape: {tuple(residual.shape)}")

    delta_mib = [(end - start) / (1 << 20) for start, end in zip(before, after)]
    result = {
        "before_bytes": before,
        "after_bytes": after,
        "delta_mib": delta_mib,
        "limit_mib": MAX_DELTA_MIB_PER_RANK,
    }
    print("PREFILL_CACHE_RESIDENCY " + json.dumps(result), flush=True)
    if max(delta_mib) >= MAX_DELTA_MIB_PER_RANK:
        raise AssertionError(
            "prefill provider retained private dequantized weights: "
            f"delta_mib={delta_mib}, limit={MAX_DELTA_MIB_PER_RANK}"
        )
    print("PREFILL_CACHE_RESIDENCY_PASS", flush=True)


if __name__ == "__main__":
    main()
