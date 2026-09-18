
import json, urllib.request, threading, time, uuid, sys

URL = "http://127.0.0.1:8000/v1/chat/completions"
SALT = uuid.uuid4().hex[:8]
RESULT = "/tmp/batch_matrix_result.json"

def fill(tag, n):
    return (f"Session {SALT} note {tag}. Background material on distributed inference engines. " * n)

# ---- prompt bank: varied lengths, needles, one duplicate-prefix family ----
P = {}
P["short_a"] = "What is 12 multiplied by 7? Reply with the number only."
P["short_b"] = "What is the capital city of France? Reply with one word."
P["mid_alpha"] = fill("A", 22) + " Important: the secret code for Alpha is 4271. Question: what is Alpha's secret code? Reply with the 4 digits only."
P["mid_beta"]  = fill("B", 22) + " Important: the secret code for Beta is 9835. Question: what is Beta's secret code? Reply with the 4 digits only."
P["long_gamma"] = fill("G", 40) + " Important: the secret code for Gamma is 1607. Question: what is Gamma's secret code? Reply with the 4 digits only."
P["long_delta"] = fill("D", 40) + " Important: the secret code for Delta is 5312. Question: what is Delta's secret code? Reply with the 4 digits only."
P["tiny"] = "Say OK."
P["story"] = fill("S", 30) + " Summarize the passage above in exactly one short sentence."

def call(name, max_tokens, out, idx):
    body = json.dumps({"model": "ljqinfer-dsv4f-0731",
                       "messages": [{"role": "user", "content": P[name]}],
                       "max_tokens": max_tokens, "temperature": 0}).encode()
    req = urllib.request.Request(URL, data=body, headers={
        "Content-Type": "application/json", "Authorization": "Bearer devkey"})
    t0 = time.perf_counter()
    try:
        r = json.load(urllib.request.urlopen(req, timeout=180))
        m = r.get("ljqinfer", {})
        out[idx] = {"name": name, "mt": max_tokens, "ok": True,
                    "text": r["choices"][0]["message"]["content"],
                    "finish": r["choices"][0].get("finish_reason"),
                    "input": m.get("input_tokens"), "hit": m.get("cache_hit_tokens"),
                    "stored": m.get("cache_stored_blocks"),
                    "sec": round(time.perf_counter() - t0, 2)}
    except Exception as e:
        out[idx] = {"name": name, "mt": max_tokens, "ok": False,
                    "text": "ERR:" + repr(e)[:120], "sec": round(time.perf_counter() - t0, 2)}

def run_group(specs, stagger=0.0):
    """specs: list of (prompt_name, max_tokens). Fired concurrently."""
    out = [None] * len(specs)
    ths = []
    for i, (n, mt) in enumerate(specs):
        t = threading.Thread(target=call, args=(n, mt, out, i))
        ths.append(t)
    for t in ths:
        t.start()
        if stagger:
            time.sleep(stagger)
    for t in ths:
        t.join()
    return out

log = {"salt": SALT, "gold": {}, "cases": []}

# ================= PHASE 1: B=1 gold (strictly serial) =================
print("=== PHASE 1: B=1 gold (serial) ===", flush=True)
GOLD_MT = 32
for name in P:
    o = run_group([(name, GOLD_MT)])[0]
    log["gold"][name] = o
    print(f"  gold {name:11s} ok={o['ok']} input={o.get('input')} hit={o.get('hit')} "
          f"stored={o.get('stored')} {o['sec']}s text={o['text'][:48]!r}", flush=True)

def check(case_name, specs, stagger=0.0, note=""):
    res = run_group(specs, stagger=stagger)
    rows = []
    allpass = True
    for spec, o in zip(specs, res):
        name, mt = spec
        g = log["gold"][name]
        if not o["ok"]:
            verdict = "ERROR"; allpass = False
        elif mt == GOLD_MT:
            verdict = "MATCH" if o["text"] == g["text"] else "MISMATCH"
            if verdict == "MISMATCH": allpass = False
        else:
            # shorter budget: must be a prefix of gold
            verdict = "PREFIX_OK" if g["text"].startswith(o["text"]) else "PREFIX_BAD"
            if verdict == "PREFIX_BAD": allpass = False
        rows.append({**o, "verdict": verdict})
    log["cases"].append({"case": case_name, "note": note, "pass": allpass, "rows": rows})
    print(f"--- {case_name} ({note}) => {'PASS' if allpass else 'FAIL'}", flush=True)
    for r in rows:
        print(f"      {r['name']:11s} mt={r['mt']:<3} {r['verdict']:9s} input={r.get('input')} "
              f"hit={r.get('hit')} stored={r.get('stored')} {r['sec']}s text={r['text'][:40]!r}", flush=True)
    return allpass

# ================= PHASE 2: batch matrix =================
print("=== PHASE 2: batch matrix ===", flush=True)
M = GOLD_MT
check("B2_equal_len",      [("mid_alpha", M), ("mid_beta", M)],                      note="B=2 same length, cache now warm")
check("B2_mixed_len",      [("tiny", M), ("long_gamma", M)],                          note="B=2 very short + long")
check("B3_mixed",          [("short_a", M), ("mid_alpha", M), ("long_delta", M)],     note="B=3 short/mid/long")
check("B4_full",           [("mid_alpha", M), ("mid_beta", M), ("long_gamma", M), ("long_delta", M)], note="B=4 all distinct")
check("B4_identical",      [("mid_alpha", M)] * 4,                                    note="B=4 SAME prompt x4 (shared-prefix store conflict)")
check("B3_mixed_maxtok",   [("mid_alpha", 8), ("mid_beta", M), ("long_gamma", 64)],   note="B=3 different max_tokens (early finish rows)")
check("B4_stagger",        [("short_a", M), ("short_b", M), ("tiny", M), ("story", M)], stagger=0.08,
      note="B=4 staggered arrival within 0.3s window")
check("B2_repeat_hit",     [("long_gamma", M), ("long_delta", M)],                    note="B=2 replay -> cold KV hit path")
check("B4_full_again",     [("mid_alpha", M), ("mid_beta", M), ("long_gamma", M), ("long_delta", M)], note="B=4 repeat for stability")

# ================= PHASE 3: B=1 re-verify after batching =================
print("=== PHASE 3: B=1 re-verify (no batch contamination) ===", flush=True)
check("B1_after_batches",  [("mid_alpha", M)], note="single request after heavy batching")
check("B1_after_batches2", [("long_gamma", M)], note="single request after heavy batching")

npass = sum(1 for c in log["cases"] if c["pass"])
ntot = len(log["cases"])
log["summary"] = {"passed": npass, "total": ntot}
json.dump(log, open(RESULT, "w"), indent=1, ensure_ascii=False)
print(f"=== SUMMARY: {npass}/{ntot} cases PASS ===", flush=True)
