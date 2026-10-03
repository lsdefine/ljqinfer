"""Fixed-width batch step timing: everyone boards in the same grace window."""
import json, threading, time, urllib.request

PORT, KEY = 8000, "devkey"
PROMPT = "Count slowly and describe what you see around you in detail."


def one(out, i, max_tokens):
    body = {"model": "local", "stream": True, "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": PROMPT}]}
    req = urllib.request.Request(
        "http://127.0.0.1:%d/v1/chat/completions" % PORT,
        json.dumps(body).encode(),
        {"Content-Type": "application/json", "Authorization": "Bearer " + KEY})
    stats = {}
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=300) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data: "):
                continue
            body_s = line[6:]
            if body_s == "[DONE]":
                break
            ev = json.loads(body_s)
            if ev.get("ljqinfer"):
                stats = ev["ljqinfer"]
    out[i] = (time.perf_counter() - t0, stats)


def run(n, max_tokens=160):
    out = [None] * n
    ths = [threading.Thread(target=one, args=(out, i, max_tokens)) for i in range(n)]
    t0 = time.perf_counter()
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    wall = time.perf_counter() - t0
    rows = [o for o in out if o and o[1]]
    if not rows:
        print("b=%d  no stats" % n)
        return
    def avg(key):
        return sum(r[1].get(key, 0) for r in rows) / len(rows)
    steps = avg("decode_steps")
    print("b=%d  step=%6.2fms  stall=%.3fs  wall/step=%6.2fms  steps=%5.1f  "
          "mtp=%.2f tok/step  out=%5.1f tok  thr=%6.1f tok/s  wall=%.2fs"
          % (n, avg("model_step_host_ms"), avg("prefill_stall_seconds"),
             avg("wall_ms_per_step"), steps, avg("mtp_accepted_per_step"),
             avg("output_tokens") or steps * avg("decode_tokens_per_step"),
             sum(r[1].get("output_tokens", 0) for r in rows) / wall, wall))


print("== fixed-width batches (all board within the same grace window)")
run(1)
for n in (1, 2, 3, 4):
    run(n)
