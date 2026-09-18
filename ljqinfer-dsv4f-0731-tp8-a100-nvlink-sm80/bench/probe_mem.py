import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # repo root on sys.path (this file lives in bench/)
import os, time, torch
from model.model_api import load_execution
RANK = int(os.environ.get("RANK", "0"))
def mem(tag):
    torch.cuda.synchronize()
    a = torch.cuda.memory_allocated() / 2**30; r = torch.cuda.memory_reserved() / 2**30; p = torch.cuda.max_memory_allocated() / 2**30
    print(f"[mem] {tag:28s} alloc={a:6.2f} reserved={r:6.2f} peak={p:6.2f} GiB", flush=True)
ms = int(os.environ.get("P_MAX_SEQ", "1048576")); bs = int(os.environ.get("P_BS", "1")); ch = int(os.environ.get("P_CHUNK", "12288"))
ex = load_execution(verbose=False, max_seq_len=ms, max_batch_size=bs)
if RANK != 0:
    ex.serve_workers(); raise SystemExit
mem("after load")
ex.prefill_chunk = ch
n = int(os.environ.get("P_LEN", "24576"))
ids = [100 + (i * 7) % 20000 for i in range(n)]
ex.set_input(ids)
slot = ex._slot(0)
mem("after slot+graph")
for rnd in range(int(os.environ.get("P_ROUNDS","1"))):
    if rnd: ex.set_input(ids); slot = ex._slot(0); mem(f"round{rnd} after slot")
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time(); pos = 0
    while pos < n:
        k = min(ch, n - pos); ex._forward(slot, pos, ids[pos:pos + k]); pos += k
    torch.cuda.synchronize()
    mem(f"prefill {n} chunk={ch}")
    print(f"[t] prefill {n} in {time.time()-t0:.1f}s = {n/(time.time()-t0):.0f} tok/s", flush=True)
    if os.environ.get("P_GEN"):
        import threading; out=[]
        ex.generate_batch([ids], [int(os.environ["P_GEN"])], [threading.Event()], emit=lambda row, toks: out.extend(toks))
        mem(f"round{rnd} after gen{len(out)}")
    else:
        ex._rpc("rewind", slot, pos)
    ex.reset(); mem(f"round{rnd} after reset")
ex.shutdown()
