#!/usr/bin/env python3
"""Direct Q4 MTP E2E through ModelExecution.generate_batch."""
import os
import sys
import threading
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.0")

from model.model_api import ModelExecution

PROMPT = [151331, 151333, 198, 2610, 525, 264, 10950, 17847]
N = 128


def run(model: ModelExecution, rows: list[list[int]]) -> tuple[list[list[int]], float]:
    output = [[] for _ in rows]
    model.set_input(rows[0])
    started = time.perf_counter()
    model.generate_batch(
        rows, [N] * len(rows), [threading.Event() for _ in rows],
        lambda row, tokens: output[row].extend(tokens))
    return output, time.perf_counter() - started


def main() -> None:
    started = time.perf_counter()
    model = ModelExecution.startup()
    print(f"LOAD_S={time.perf_counter() - started:.2f}", flush=True)

    output, elapsed = run(model, [PROMPT])
    if len(output[0]) != N:
        raise AssertionError(f"expected {N} tokens, got {len(output[0])}")
    print(f"B1Q4: {len(output[0])} tok in {elapsed:.3f}s = {len(output[0]) / elapsed:.2f} tps", flush=True)
    print("MTP_E2E_DONE", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
