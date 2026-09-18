#!/usr/bin/env python3
"""Long-context acceptance test against the private Strategy RPC.

The test bypasses tokenization and the public service layer: ``/generate`` is a
thin transport for ``Strategy.query(input_ids, max_new_tokens)``.  It covers B1 through B4 with cold, exact-hit, and partial-hit KV prefixes.

Decode rates use the wall interval between Strategy token events. A one-token
MTP verification is a reject; widths two through six are accepts. The reported
path rates use the actual emitted token count and exclude a final clipped event.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import threading
import time
from dataclasses import dataclass
from typing import Any

import requests


BASE_URL = "http://127.0.0.1:62001"
PREFILL_TOKENS = 32 * 1024
PARTIAL_HIT_TOKENS = 28 * 1024
DECODE_TOKENS = 384
REQUEST_TIMEOUT_SECONDS = 900
MIN_PREFILL_TPS = 1000.0
MIN_REJECT_TPS = 20.0
MIN_ACCEPT_TPS = 40.0
MIN_PATH_SAMPLES = 4


class TestFailure(AssertionError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise TestFailure(message)


@dataclass
class RowResult:
    batch_size: int
    metrics: dict[str, Any]
    chunks: list[tuple[float, list[int]]]
    prefill_at: float
    ended_at: float

    @property
    def output_tokens(self) -> int:
        return sum(len(ids) for _, ids in self.chunks)

    @property
    def decode_seconds(self) -> float:
        return self.ended_at - self.prefill_at


@dataclass
class PathRates:
    accepts: int
    rejects: int
    accept_tokens: int
    accept_seconds: float
    reject_seconds: float

    @property
    def accept_tps(self) -> float:
        return self.accept_tokens / self.accept_seconds if self.accept_seconds else 0.0

    @property
    def reject_tps(self) -> float:
        return self.rejects / self.reject_seconds if self.reject_seconds else 0.0

    @property
    def accept_rate(self) -> float:
        steps = self.accepts + self.rejects
        return self.accepts / steps if steps else 0.0


def random_ids(seed: int) -> list[int]:
    rng = random.Random(seed)
    return [rng.randrange(100, 30000) for _ in range(PREFILL_TOKENS)]


def changed_first(ids: list[int]) -> list[int]:
    changed = list(ids)
    changed[0] = 100 + ((changed[0] - 99) % 29900)
    require(changed[0] != ids[0], "failed to make a distinct cold prefix")
    return changed


def partial(prefix: list[int], tail: list[int]) -> list[int]:
    result = prefix[:PARTIAL_HIT_TOKENS] + tail[PARTIAL_HIT_TOKENS:]
    require(result[:PARTIAL_HIT_TOKENS] == prefix[:PARTIAL_HIT_TOKENS],
            "partial-hit prefix construction failed")
    require(result[PARTIAL_HIT_TOKENS:] != prefix[PARTIAL_HIT_TOKENS:],
            "partial-hit tail did not diverge")
    return result


def run_row(label: str, row: int, ids: list[int], barrier: threading.Barrier | None,
            sink: list[RowResult | BaseException | None]) -> None:
    request_id = f"stress_{label}_{row}_{time.time_ns()}"
    try:
        if barrier is not None:
            barrier.wait()
        with requests.post(
            f"{BASE_URL}/generate",
            json={"request_id": request_id, "input_ids": ids,
                  "max_new_tokens": DECODE_TOKENS},
            stream=True,
            timeout=(20, REQUEST_TIMEOUT_SECONDS),
        ) as response:
            response.raise_for_status()
            metrics = None
            batch_size = None
            prefill_at = None
            ended_at = None
            chunks: list[tuple[float, list[int]]] = []
            for raw in response.iter_lines(chunk_size=1, decode_unicode=True):
                if not raw:
                    continue
                now = time.perf_counter()
                event = json.loads(raw)
                kind = event.get("type")
                if kind == "prefill":
                    require(prefill_at is None, f"{label}[{row}] duplicate prefill event")
                    prefill_at = now
                    metrics = event.get("metrics")
                    batch_size = int(event.get("batch_size", 0))
                elif kind == "token":
                    token_ids = [int(token) for token in event.get("token_ids") or []]
                    require(token_ids, f"{label}[{row}] empty token event")
                    chunks.append((now, token_ids))
                elif kind == "error":
                    raise TestFailure(f"{label}[{row}] strategy error: {event.get('error')}")
                elif kind == "end":
                    require(not event.get("cancelled"), f"{label}[{row}] was cancelled")
                    ended_at = now
                    break
            require(isinstance(metrics, dict), f"{label}[{row}] missing prefill metrics")
            require(prefill_at is not None, f"{label}[{row}] missing prefill timestamp")
            require(ended_at is not None, f"{label}[{row}] stream ended without end event")
            sink[row] = RowResult(batch_size, metrics, chunks, prefill_at, ended_at)
    except BaseException as exc:
        sink[row] = exc


def request_case(label: str, rows: list[list[int]]) -> list[RowResult]:
    sink: list[RowResult | BaseException | None] = [None] * len(rows)
    if len(rows) == 1:
        run_row(label, 0, rows[0], None, sink)
    else:
        barrier = threading.Barrier(len(rows))
        threads = [threading.Thread(target=run_row,
                     args=(label, row, ids, barrier, sink), daemon=True)
                   for row, ids in enumerate(rows)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(REQUEST_TIMEOUT_SECONDS + 60)
        require(all(not thread.is_alive() for thread in threads),
                f"{label} worker did not terminate")
    for row, result in enumerate(sink):
        if isinstance(result, BaseException):
            raise TestFailure(f"{label}[{row}] failed: {result}") from result
        require(isinstance(result, RowResult), f"{label}[{row}] produced no result")
    return [result for result in sink if isinstance(result, RowResult)]


def path_rates(result: RowResult) -> PathRates:
    previous = result.prefill_at
    produced = 0
    accepts = rejects = accept_tokens = 0
    accept_s = reject_s = 0.0
    for arrived, token_ids in result.chunks:
        elapsed = arrived - previous
        previous = arrived
        before = produced
        produced += len(token_ids)
        # The final event may be clipped by the output-token limit, so it does
        # not identify the underlying MTP path and is excluded from path rates.
        if before + len(token_ids) >= DECODE_TOKENS:
            continue
        require(1 <= len(token_ids) <= 6,
                f"unexpected MTP event width {len(token_ids)}")
        if len(token_ids) > 1:
            accepts += 1
            accept_tokens += len(token_ids)
            accept_s += elapsed
        else:
            rejects += 1
            reject_s += elapsed
    return PathRates(accepts, rejects, accept_tokens, accept_s, reject_s)


def check_case(label: str, rows: list[list[int]], expected_hit: int,
               speed_gates: bool = True) -> dict[str, Any]:
    started = time.perf_counter()
    results = request_case(label, rows)
    batch_size = len(rows)
    for row, result in enumerate(results):
        require(result.batch_size == batch_size,
                f"{label}[{row}] expected B={batch_size}, got B={result.batch_size}")
        require(result.output_tokens == DECODE_TOKENS,
                f"{label}[{row}] decoded {result.output_tokens}/{DECODE_TOKENS} tokens")
        actual_hit = int(result.metrics.get("cache_hit_tokens", -1))
        require(actual_hit == expected_hit,
                f"{label}[{row}] cache hit {actual_hit}, expected {expected_hit}")

    model_seconds = max(float(row.metrics["model_prefill_seconds"])
                        for row in results)
    tail_tokens = sum(int(row.metrics["prefill_tokens"]) for row in results)
    model_prefill_tps = tail_tokens / model_seconds if model_seconds else 0.0
    strategy_seconds = max(float(row.metrics["submit_to_prefill_seconds"])
                           for row in results)
    effective_prefill_tps = sum(len(ids) for ids in rows) / strategy_seconds
    # A fully cached request intentionally prefills only one anchor token, so
    # model-only TPS is too small/noisy to be meaningful; gate end-to-end reuse.
    gated_prefill_tps = (model_prefill_tps if tail_tokens >= 4096
                         else effective_prefill_tps)
    if speed_gates:
        require(gated_prefill_tps >= MIN_PREFILL_TPS,
                f"{label} prefill {gated_prefill_tps:.1f} < {MIN_PREFILL_TPS:.1f} tok/s")

    rates = [path_rates(row) for row in results]
    accepts = sum(rate.accepts for rate in rates)
    rejects = sum(rate.rejects for rate in rates)
    accept_tokens = sum(rate.accept_tokens for rate in rates)
    accept_s = sum(rate.accept_seconds for rate in rates)
    reject_s = sum(rate.reject_seconds for rate in rates)
    combined = PathRates(accepts, rejects, accept_tokens, accept_s, reject_s)
    if speed_gates:
        require(combined.accepts >= MIN_PATH_SAMPLES,
                f"{label} has only {combined.accepts} MTP accept samples")
        require(combined.rejects >= MIN_PATH_SAMPLES,
                f"{label} has only {combined.rejects} MTP reject samples")
        require(combined.reject_tps >= MIN_REJECT_TPS,
                f"{label} reject path {combined.reject_tps:.2f} < {MIN_REJECT_TPS:.2f} tok/s")
        require(combined.accept_tps >= MIN_ACCEPT_TPS,
                f"{label} accept path {combined.accept_tps:.2f} < {MIN_ACCEPT_TPS:.2f} tok/s")

    decode_s = max(row.decode_seconds for row in results)
    actual_decode_tps = sum(row.output_tokens for row in results) / decode_s
    summary = {
        "case": label,
        "batch": batch_size,
        "input_tokens_per_row": PREFILL_TOKENS,
        "decode_tokens_per_row": DECODE_TOKENS,
        "cache_hit_tokens": expected_hit,
        "tail_prefill_tokens": tail_tokens,
        "model_prefill_tps": round(model_prefill_tps, 2),
        "effective_prefill_tps": round(effective_prefill_tps, 2),
        "decode_tps_aggregate": round(actual_decode_tps, 2),
        "mtp_accept_rate": round(combined.accept_rate, 4),
        "mtp_accept_samples": combined.accepts,
        "mtp_reject_samples": combined.rejects,
        "mtp_accept_path_tps": round(combined.accept_tps, 2),
        "mtp_reject_path_tps": round(combined.reject_tps, 2),
        "wall_seconds": round(time.perf_counter() - started, 3),
        "output_sha256": [hashlib.sha256(
            json.dumps([token for _, chunk in row.chunks for token in chunk],
                       separators=(",", ":")).encode()).hexdigest()[:16]
            for row in results],
    }
    print("PASS " + json.dumps(summary, sort_keys=True), flush=True)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=time.time_ns() & 0x7FFFFFFF,
                        help="replayable random seed; defaults to a fresh run")
    parser.add_argument("--no-speed-gates", action="store_true",
                        help="record throughput without failing its thresholds")
    args = parser.parse_args()

    health = requests.get(f"{BASE_URL}/health", timeout=10)
    health.raise_for_status()
    require(health.json().get("status") == "ok", "Strategy engine is not ready")

    a = random_ids(args.seed)
    b = random_ids(args.seed + 1)
    c = changed_first(a)
    d = random_ids(args.seed + 3)
    e = random_ids(args.seed + 4)
    f = random_ids(args.seed + 5)
    g = random_ids(args.seed + 6)
    h = random_ids(args.seed + 7)
    i = random_ids(args.seed + 8)
    j = random_ids(args.seed + 9)
    a_partial = partial(a, b)
    b_partial = partial(b, a)
    c_partial = partial(c, b)
    d_partial = partial(d, e)
    e_partial = partial(e, d)
    f_partial = partial(f, d)
    g_partial = partial(g, h)
    h_partial = partial(h, g)
    i_partial = partial(i, g)
    j_partial = partial(j, g)

    print("strategy stress acceptance", flush=True)
    print(json.dumps({
        "seed": args.seed,
        "prefill_tokens_per_row": PREFILL_TOKENS,
        "partial_hit_tokens": PARTIAL_HIT_TOKENS,
        "decode_tokens_per_row": DECODE_TOKENS,
        "speed_gates": not args.no_speed_gates,
        "thresholds": {"prefill_tps": MIN_PREFILL_TPS,
                       "reject_path_tps": MIN_REJECT_TPS,
                       "accept_path_tps": MIN_ACCEPT_TPS},
    }, sort_keys=True), flush=True)

    speed_gates = not args.no_speed_gates
    summaries = []
    summaries.append(check_case("b1_cold", [a], 0, speed_gates))
    summaries.append(check_case("b1_exact_hit", [a], PREFILL_TOKENS, speed_gates))
    summaries.append(check_case("b1_partial_hit", [a_partial], PARTIAL_HIT_TOKENS,
                                speed_gates))
    summaries.append(check_case("b2_cold", [b, c], 0, speed_gates))
    summaries.append(check_case("b2_exact_hit", [b, c], PREFILL_TOKENS, speed_gates))
    summaries.append(check_case("b2_partial_hit", [b_partial, c_partial],
                                PARTIAL_HIT_TOKENS, speed_gates))
    summaries.append(check_case("b3_cold", [d, e, f], 0, speed_gates))
    summaries.append(check_case("b3_exact_hit", [d, e, f], PREFILL_TOKENS,
                                speed_gates))
    summaries.append(check_case("b3_partial_hit", [d_partial, e_partial, f_partial],
                                PARTIAL_HIT_TOKENS, speed_gates))
    summaries.append(check_case("b4_cold", [g, h, i, j], 0, speed_gates))
    summaries.append(check_case("b4_exact_hit", [g, h, i, j], PREFILL_TOKENS,
                                speed_gates))
    summaries.append(check_case(
        "b4_partial_hit", [g_partial, h_partial, i_partial, j_partial],
        PARTIAL_HIT_TOKENS, speed_gates))
    print(f"SUMMARY PASS passed={len(summaries)}/12", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except TestFailure as exc:
        print(f"SUMMARY FAIL: {exc}", flush=True)
        raise SystemExit(1)
