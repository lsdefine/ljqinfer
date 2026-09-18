#!/usr/bin/env python3
"""Random differential gate: native Bn batch invariance (B=2 vs two B=1 calls).

This is an operator test, not a model/E2E test.  Both sides receive identical
real weights, activations, positions, K0 metadata, page mappings and initial KV
bytes.  The reference executes B1 independently for each request; the candidate
executes the native fused Bn implementation once.  Any failure is serialized as
/tmp/native_bn_attn_failure.pt for deterministic replay.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from model import wcache
from ops.kernels import get_decode_attn, load_kernels

D = 6144
PAGE = 2048
PAGES = 4
EDGE_K0 = (0, 1, 31, 32, 63, 64, 127, 128, 511, 512,
           1023, 1024, 2046, 2047, 2048, 2049, 4094)


def _args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--cases", type=int, default=1000)
    p.add_argument("--seed", type=int, default=20260809)
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--stop-first", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--out-atol", type=float, default=0.02)
    p.add_argument("--out-rtol", type=float, default=0.01)
    p.add_argument("--kv-atol", type=float, default=0.002)
    p.add_argument("--kv-rtol", type=float, default=0.002)
    return p.parse_args()


def _weights(attn, rank: int):
    return (attn.norm[rank], attn.q_a[rank], attn.q_a_norm[rank],
            attn.q_b[rank], attn.kv_a[rank], attn.kv_a_norm[rank],
            attn.k_b[rank], attn.v_b[rank], attn.o[rank])


def _call_b1(op, x, pos, pool, table, k0, ws):
    # Reference side: the same native Bn operator invoked with B=1.  The gate
    # is batch invariance: one fused B=2 call must equal two independent B=1
    # calls over identical weights, activations and KV bytes.
    _, y = op.forward_rank_paged_batch_k0(
        x, pos, pool, [table], [k0], *ws)
    # Native outputs alias a per-device static workspace.  Preserve each result
    # before the second reference call reuses that workspace.
    return y.clone()


def _call_bn(op, x, pos, pool, tables, k0s, ws):
    b, q, d = x.shape
    _, y = op.forward_rank_paged_batch_k0(
        x.reshape(b * q, d), pos.reshape(b * q), pool,
        list(tables.unbind(0)), [k0s[i:i + 1] for i in range(b)], *ws)
    return y.clone().reshape(b, q, d)


def _stats(a: torch.Tensor, b: torch.Tensor) -> dict:
    d = (a.float() - b.float()).abs()
    flat = d.reshape(-1)
    idx = int(flat.argmax().item())
    return {
        "max_abs": float(flat[idx].item()),
        "mean_abs": float(flat.mean().item()),
        "bad_index": [int(v.item()) for v in torch.unravel_index(torch.tensor(idx), d.shape)],
        "ref_at_bad": float(a.reshape(-1)[idx].item()),
        "bn_at_bad": float(b.reshape(-1)[idx].item()),
    }


def _written_mask(tables: torch.Tensor, k0s: torch.Tensor, q: int,
                  device: torch.device) -> torch.Tensor:
    mask = torch.zeros((PAGES, PAGE), dtype=torch.bool, device=device)
    for row in range(2):
        k0 = int(k0s[row].item())
        for logical in range(k0, k0 + q):
            lp, off = divmod(logical, PAGE)
            mask[int(tables[row, lp].item()), off] = True
    return mask


def main() -> None:
    args = _args()
    if args.cases <= 0:
        raise ValueError("--cases must be positive")
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    load_kernels()
    print("[ab] loading real TP8 weights", flush=True)
    weights = wcache.load("tp8", register=False, verbose=True)
    op = get_decode_attn()
    rank = args.device
    device = torch.device(f"cuda:{rank}")
    gen = torch.Generator(device=device)
    gen.manual_seed(args.seed)

    failures = []
    maxima = {"out": 0.0, "kv": 0.0}
    started = time.perf_counter()
    for case in range(args.cases):
        q = 1 if case % 2 == 0 else 2
        layer = random.randrange(len(weights.layers))
        # Force edge coverage first, then continue with a mixed edge/uniform set.
        if case < len(EDGE_K0) ** 2:
            k00 = EDGE_K0[case % len(EDGE_K0)]
            k01 = EDGE_K0[(case // len(EDGE_K0)) % len(EDGE_K0)]
        else:
            def pick_k0():
                return random.choice(EDGE_K0) if random.random() < 0.6 else random.randrange(0, 4096 - q + 1)
            k00, k01 = pick_k0(), pick_k0()
        k00 = min(k00, 4096 - q)
        k01 = min(k01, 4096 - q)
        k0s = torch.tensor([k00, k01], dtype=torch.int32, device=device)
        pos = torch.stack((
            torch.arange(k00, k00 + q, dtype=torch.int64, device=device),
            torch.arange(k01, k01 + q, dtype=torch.int64, device=device)))

        perm = torch.randperm(PAGES, generator=gen, device=device)
        # Both are complete valid page-table permutations.  Their first two
        # logical pages (the tested 0..4095 range) are physically disjoint.
        tables = torch.stack((perm, torch.roll(perm, shifts=2))).long().contiguous()

        x = torch.randn((2, q, D), dtype=torch.float16, device=device,
                        generator=gen).contiguous()
        # Every fourth case uses identical rows to expose cross-row contamination.
        if case % 4 == 0:
            x[1].copy_(x[0])
        initial = torch.randn((PAGES, PAGE, 576), dtype=torch.float16,
                              device=device, generator=gen)
        ref_pool = initial.clone()
        bn_pool = initial.clone()
        ws = _weights(weights.layers[layer].attn, rank)

        y0 = _call_b1(op, x[0], pos[0], ref_pool, tables[0], k0s[0:1], ws)
        y1 = _call_b1(op, x[1], pos[1], ref_pool, tables[1], k0s[1:2], ws)
        y_ref = torch.stack((y0, y1))
        y_bn = _call_bn(op, x, pos, bn_pool, tables, k0s, ws)
        torch.cuda.synchronize(device)

        out_ok = torch.allclose(y_ref, y_bn, atol=args.out_atol, rtol=args.out_rtol)
        kv_ok = torch.allclose(ref_pool, bn_pool, atol=args.kv_atol, rtol=args.kv_rtol)
        mask = _written_mask(tables, k0s, q, device)
        untouched_ok = torch.equal(initial[~mask], bn_pool[~mask])
        finite_ok = bool(torch.isfinite(y_ref).all() and torch.isfinite(y_bn).all()
                         and torch.isfinite(ref_pool[mask]).all()
                         and torch.isfinite(bn_pool[mask]).all())
        out_stats = _stats(y_ref, y_bn)
        kv_stats = _stats(ref_pool[mask], bn_pool[mask])
        maxima["out"] = max(maxima["out"], out_stats["max_abs"])
        maxima["kv"] = max(maxima["kv"], kv_stats["max_abs"])

        if not (out_ok and kv_ok and untouched_ok and finite_ok):
            failure = {
                "case": case, "seed": args.seed, "layer": layer, "q": q,
                "k0": [k00, k01], "tables": tables.cpu().tolist(),
                "out_ok": bool(out_ok), "kv_ok": bool(kv_ok),
                "untouched_ok": bool(untouched_ok), "finite_ok": finite_ok,
                "out": out_stats, "kv": kv_stats,
            }
            failures.append(failure)
            torch.save({
                "meta": failure, "x": x.cpu(), "positions": pos.cpu(),
                "tables": tables.cpu(), "k0s": k0s.cpu(),
                "initial_pool": initial.cpu(), "y_ref": y_ref.cpu(),
                "y_bn": y_bn.cpu(), "ref_pool": ref_pool.cpu(),
                "bn_pool": bn_pool.cpu(),
            }, "/tmp/native_bn_attn_failure.pt")
            print("AB_FAIL " + json.dumps(failure, sort_keys=True), flush=True)
            if args.stop_first:
                raise AssertionError("native Bn differs from two B1 calls")

        if (case + 1) % 25 == 0 or case == 0:
            elapsed = time.perf_counter() - started
            print(f"AB_PROGRESS cases={case + 1}/{args.cases} "
                  f"max_out={maxima['out']:.6g} max_kv={maxima['kv']:.6g} "
                  f"failures={len(failures)} elapsed_s={elapsed:.1f}", flush=True)

    result = {
        "cases": args.cases, "seed": args.seed, "failures": len(failures),
        "max_out": maxima["out"], "max_kv": maxima["kv"],
        "elapsed_s": time.perf_counter() - started,
    }
    print("AB_PASS " + json.dumps(result, sort_keys=True), flush=True)
    if failures:
        raise AssertionError(f"{len(failures)} random differential cases failed")


if __name__ == "__main__":
    main()
