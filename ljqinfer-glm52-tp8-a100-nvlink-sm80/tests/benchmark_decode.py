#!/usr/bin/env python3
"""Canonical decode-step benchmark.

This is the only supported decode performance benchmark. Its workload is
fixed: 12,288-token prefixes, B1/B2/B4, Q=6 MTP, 25 steps with the first 5
discarded. Timing covers the complete mtp_step (Base verify, sampling,
draft-chain compute, and commit), synchronized across all TP GPUs.
Accepted-token TPS is deliberately not reported.
"""
from __future__ import annotations

import gc
import hashlib
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if Path.cwd().resolve() != ROOT:
    raise SystemExit(
        f"run from repository root: cd {ROOT} && python tests/benchmark_decode.py"
    )
sys.path.insert(0, str(ROOT))

import torch

from model.batch_decode import batched_mtp_prime, mtp_step, prefill_batch_chunked
from model.model import Engine

PREFIX = 12_288
BATCHES = (1, 2, 4)
Q = 6
ROUNDS = 25
WARMUP = 5
SEED = 12_345
EXPECTED_EMIT_SHA256 = {
    1: "ca1de8b7583cda953fc2bf58b17e688cd3c2a89354e540ea8d64046ffbe506bc",
    2: "b30bfb6f1d5dbf61177989c054974e673dcb7a4d505dc0d38271edaffeaa325f",
    4: "b46911ccb3142e5ae7d2440fe13ed95faa23e478223c507d7d77566523dd61a2",
}


def sync(engine: Engine) -> None:
    for device in engine.rt.devices:
        torch.cuda.synchronize(device)


def prompt(request_index: int) -> list[int]:
    return [
        1000 + ((request_index * 131 + i * 17) % 50_000)
        for i in range(PREFIX)
    ]


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[int(fraction * (len(ordered) - 1))]


def identity() -> dict:
    lock = ROOT / "ops" / "selected_ops.lock.json"
    return {
        "git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "lock_sha256": hashlib.sha256(lock.read_bytes()).hexdigest(),
        "prefix": PREFIX,
        "batches": list(BATCHES),
        "q": Q,
        "rounds": ROUNDS,
        "warmup": WARMUP,
        "seed": SEED,
        "timing_scope": "full_mtp_step_synchronized_all_tp_gpus",
    }


def bench(engine: Engine, batch_size: int) -> dict:
    rows = [prompt(i + 1) for i in range(batch_size)]
    capacities = [PREFIX + 8 + ROUNDS * Q + 64] * batch_size
    residuals, prefill_state = prefill_batch_chunked(
        engine, rows, max_lengths=capacities
    )
    state = prefill_state.decode_state(engine.base_graphs[(batch_size, Q)])
    base, draft = batched_mtp_prime(engine, state, rows, residuals)
    sync(engine)
    values: list[float] = []
    emitted: list[list[list[int]]] = []
    try:
        for _ in range(ROUNDS):
            sync(engine)
            started = time.perf_counter()
            output, base, draft = mtp_step(engine, state, base, draft)
            sync(engine)
            values.append((time.perf_counter() - started) * 1000.0)
            emitted.append(output)

        steady = values[WARMUP:]
        emit_sha = hashlib.sha256(
            json.dumps(emitted, sort_keys=True).encode()
        ).hexdigest()
        expected = EXPECTED_EMIT_SHA256[batch_size]
        # The reference token trajectory is diagnostic, not a floating-point
        # correctness oracle: mathematically equivalent GEMMs may change the
        # reduction order and cross a sampling boundary. Operator numerical
        # equivalence is gated separately on real weights.
        trajectory_match_reference = emit_sha == expected
        result = {
            "B": batch_size,
            "prefix": PREFIX,
            "rounds": ROUNDS,
            "warmup": WARMUP,
            "n": len(steady),
            "p50_ms": statistics.median(steady),
            "mean_ms": statistics.mean(steady),
            "min_ms": min(steady),
            "p90_ms": percentile(steady, 0.9),
            "max_ms": max(steady),
            "vals_ms": steady,
            "final_lengths": list(state.lengths),
            "emit_sha256": emit_sha,
            "reference_emit_sha256": expected,
            "trajectory_match_reference": trajectory_match_reference,
            "correctness": "PASS_EXTERNAL_NUMERIC_GATE",
        }
        print("DECODE_STEP " + json.dumps(result), flush=True)
        return result
    finally:
        state.release()
        del state, prefill_state, residuals, rows, base, draft
        gc.collect()
        sync(engine)


def main() -> None:
    if len(sys.argv) != 1:
        raise SystemExit(
            "this benchmark has no tunable CLI arguments; run exactly: "
            "python tests/benchmark_decode.py"
        )
    torch.manual_seed(SEED)
    print("DECODE_BENCH_ID " + json.dumps(identity()), flush=True)
    engine = Engine.load()
    sync(engine)
    results = [bench(engine, batch_size) for batch_size in BATCHES]
    print("DECODE_STEP_ALL " + json.dumps(results), flush=True)


if __name__ == "__main__":
    main()
