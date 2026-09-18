
import os as _os
import sys as _sys
from pathlib import Path as _Path
_LJQINFER_ROOT = _Path(__file__).resolve().parents[1]
_os.chdir(_LJQINFER_ROOT)
_sys.path.insert(0, str(_LJQINFER_ROOT))
import os, time, torch
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.0")
from model.model import Engine, prefill, lm_head, advance

PROMPT = [151331, 151333, 198, 2610, 525, 264, 10950, 17847]

def main():
    t0 = time.perf_counter()
    e = Engine.load()
    print(f"LOAD_S={time.perf_counter()-t0:.1f}", flush=True)
    ids = torch.tensor(PROMPT, dtype=torch.long)
    resid = prefill(e, ids)
    t_last = int(lm_head(e.rt, e.w, resid).argmax())
    L0 = e.kv.length
    print(f"PREFILL len={L0} t_last={t_last}", flush=True)

    # --- B) Q1 logits vs Q2 row0 logits, same prefix ---
    La = advance(e, [t_last]).clone()          # Q1
    e.kv.length = L0
    Lb = advance(e, [t_last, 12345]).clone()   # Q2, junk second token
    e.kv.length = L0
    d = (La[0] - Lb[0])
    print(f"B) Q1row_vs_Q2row0 maxabs={d.abs().max().item():.6f} "
          f"argmax {int(La[0].argmax())} vs {int(Lb[0].argmax())}", flush=True)

    # --- A) row0 sensitivity to second token (causality check) ---
    L1 = advance(e, [t_last, 111]).clone(); e.kv.length = L0
    L2 = advance(e, [t_last, 22222]).clone(); e.kv.length = L0
    d2 = (L1[0] - L2[0])
    print(f"A) row0_d1_vs_d2 maxabs={d2.abs().max().item():.6f} "
          f"argmax {int(L1[0].argmax())} vs {int(L2[0].argmax())}", flush=True)
    # row1 should of course differ:
    print(f"A2) row1 argmax {int(L1[1].argmax())} vs {int(L2[1].argmax())}", flush=True)

    # --- C) pure advance Q1 throughput ---
    tok = torch.tensor([t_last], dtype=torch.long)
    for _ in range(5): advance(e, tok)
    e.kv.length = L0
    torch.cuda.synchronize(); t1 = time.perf_counter()
    N = 100
    for _ in range(N):
        advance(e, tok)
    torch.cuda.synchronize()
    ta = time.perf_counter() - t1
    e.kv.length = L0
    print(f"C) pure advance    : {N/ta:.2f} steps/s", flush=True)
    # generate loop with argmax overhead isolated:
    t1 = time.perf_counter()
    cur = tok
    for _ in range(N):
        lg = advance(e, cur)
        cur = torch.tensor([int(lg.argmax())], dtype=torch.long)
    torch.cuda.synchronize()
    tc = time.perf_counter() - t1
    print(f"C) step+argmax    : {N/tc:.2f} steps/s", flush=True)
    print("PROBE_DONE", flush=True)

if __name__ == "__main__":
    main()
