#!/usr/bin/env python3
"""MTP accept@1 on natural greedy continuation.

Base extends one token/step via prefill(cache_start) (returns hidden).
MTP prefills the shifted prompt stream once, then drafts 1 token/step.
accept@1 = P(draft == base's actual next token).
"""
import os, sys, time, traceback
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.0")
sys.path.insert(0, "/mnt/data/kw/ljqinfer")
os.chdir("/mnt/data/kw/ljqinfer")
import torch
from model.model import Engine, prefill, lm_head, mtp_forward, _rmsnorm

PROMPT = [151331, 151333, 198, 2610, 525, 264, 10950, 17847]  # gsm-ish chat stub
STEPS = 32

def main():
    t0 = time.perf_counter()
    e = Engine.load()
    print(f"LOAD_S={time.perf_counter()-t0:.2f} mtp_kv={'Y' if e.mtp_kv else 'N'}", flush=True)

    # ---- prefill TPS on ~2.8K prompt ----
    long_ids = torch.tensor(PROMPT * 350, dtype=torch.long)
    e.kv.length = 0
    t1 = time.perf_counter()
    _ = prefill(e, long_ids)
    torch.cuda.synchronize()
    pf_s = time.perf_counter() - t1
    print(f"PREFILL ntok={long_ids.numel()} t={pf_s:.3f}s tps={long_ids.numel()/pf_s:.1f}", flush=True)

    # ---- natural continuation accept@1 ----
    fnorm = e.w.final_norm
    ids = torch.tensor(PROMPT, dtype=torch.long)
    L = ids.numel()
    e.kv.length = 0
    h = prefill(e, ids)                                   # [L,D]
    y = int(lm_head(e.rt, e.w, h[-1:]).reshape(-1).argmax().item())  # t_L

    # MTP prompt prefill: tokens t_1..t_L with hidden(t_0..t_{L-1})
    hf = _rmsnorm(h, fnorm)                               # post_final variant
    mtp_toks = ids[1:].tolist() + [y]
    d, _ = mtp_forward(e, mtp_toks, 0, hf)                # drafts t_{L+1}

    total, hits = 0, 0
    cur = L
    seq = [y]
    for step in range(STEPS):
        # base: feed y at position cur -> hidden + next token (ground truth)
        h1 = prefill(e, torch.tensor([y], dtype=torch.long), cache_start=cur)  # [1,D]
        y_next = int(lm_head(e.rt, e.w, h1).reshape(-1).argmax().item())
        ok = (d == y_next)
        total += 1; hits += int(ok)
        print(f"step{step}: draft={d} truth={y_next} hit={ok}", flush=True)
        # mtp: append token y_next with hidden(y position) -> draft next
        hf1 = _rmsnorm(h1, fnorm)
        d, _ = mtp_forward(e, [y_next], cur, hf1)
        cur += 1
        y = y_next
        seq.append(y)
        if y in (151329, 151336, 151338):  # eos-ish
            break

    print("SEQ:", seq, flush=True)
    print(f"ACCEPT@1 = {hits}/{total} = {hits/max(total,1):.3f}", flush=True)
    print("MTP_SMOKE_DONE", flush=True)

if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
