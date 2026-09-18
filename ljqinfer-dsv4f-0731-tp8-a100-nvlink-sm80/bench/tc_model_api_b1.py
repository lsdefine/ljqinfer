# -*- coding: utf-8 -*-
"""B=1 ModelExecution gate (torchrun 8 ranks).
A: fresh prefill+decode N tokens.  B: export aligned prefix -> reset -> load -> decode.
PASS if A == B token sequences.
torchrun --nproc_per_node=8 tc_model_api_b1.py
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # repo root on sys.path (this file lives in bench/)
import json, os, sys, threading, time
import torch
from model.model_api import load_execution, BLOCK_TOKENS

N_GEN = int(os.environ.get("N_GEN", "32"))
RANK = int(os.environ.get("RANK", "0"))


def log(*a):
    if RANK == 0:
        print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def run(ex, ids):
    out = []
    st = {}
    ex.generate_batch([ids], [N_GEN], [threading.Event()],
                      emit=lambda row, toks: out.extend(toks), stats=st)
    return out, st


def main():
    ex = load_execution()
    if RANK != 0:
        ex.serve_workers()
        return
    ids = json.load(open("/tmp/c1_ids.json"))
    log(f"prompt={len(ids)} width={ex.kv_format.width} caps={ex.capabilities}")

    ex.set_input(ids)
    a, sa = run(ex, ids)
    log("A", a, f"prefill={sa['prefill_seconds']:.2f}s decode={sa['decode_seconds']:.2f}s")
    st = sorted(sa["step_seconds"][1:])  # skip first step (warmup)
    if st:
        log(f"B1Q8 steps={len(st)} ms/step mean={sum(st)/len(st)*1e3:.2f} median={st[len(st)//2]*1e3:.2f} "
            f"min={st[0]*1e3:.2f} tok/step={(len(a)-1)/sa['steps']:.2f}")

    end = len(ids) // BLOCK_TOKENS * BLOCK_TOKENS
    blob = torch.empty(end, ex.kv_format.width, dtype=torch.bfloat16, device="cpu", pin_memory=True)
    ex.export_kv(0, 0, end, blob)
    log(f"exported {end} tokens ({blob.numel()*2/2**20:.1f} MiB)")
    ex.reset()

    ex.set_input(ids)
    r = ex.load_kv_span(0, 0, end, blob, pool1_tail_pages=2)
    log(f"loaded, free_pages={r.free_pages}")
    b, sb = run(ex, ids)
    log("B", b, f"prefill={sb['prefill_seconds']:.2f}s")

    ok = a == b
    log("PASS" if ok else f"FAIL first_diff={next((i for i,(x,y) in enumerate(zip(a,b)) if x!=y), None)}")
    ex.reset()
    ex.shutdown()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
