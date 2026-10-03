"""Greedy engine: the same sentence must come back the same. Find where it forks."""
import json, threading, urllib.request

PORT, KEY = 8000, "devkey"
Q = "Name three colors and explain why each one is calming. Be concise."


def call(out, i, prompt, max_tokens=64):
    body = {"model": "local", "stream": True, "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}]}
    req = urllib.request.Request(
        "http://127.0.0.1:%d/v1/chat/completions" % PORT,
        json.dumps(body).encode(),
        {"Content-Type": "application/json", "Authorization": "Bearer " + KEY})
    text, stats, pieces = "", {}, []
    with urllib.request.urlopen(req, timeout=300) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data: ") or line[6:] == "[DONE]":
                continue
            ev = json.loads(line[6:])
            if ev.get("ljqinfer"):
                stats = ev["ljqinfer"]
            for ch in ev.get("choices") or []:
                piece = (ch.get("delta") or {}).get("content") or ""
                if piece:
                    pieces.append(piece)
                    text += piece
    out[i] = (text, stats, pieces)


def fork(a, b):
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i, repr(a[max(0, i - 40):i + 25]), repr(b[max(0, i - 40):i + 25])
    return (None if len(a) == len(b) else n), "", ""


print("== five solo runs of the same sentence (engine is greedy: must be identical)")
runs = []
for i in range(5):
    out = [None]
    call(out, 0, Q)
    t, st, pieces = out[0]
    runs.append(t)
    print("  solo#%d len=%3d mtp=%.3f steps=%s" % (i, len(t), st.get("mtp_accepted_per_step", -1), st.get("decode_steps")))
print("  identical across 5 solo runs: %s" % (len(set(runs)) == 1))
for i in range(1, 5):
    pos, ca, cb = fork(runs[0], runs[i])
    if pos is not None:
        print("  solo#0 vs solo#%d forks at char %d\n      A=%s\n      B=%s" % (i, pos, ca, cb))

print("== stats keys available")
out = [None]
call(out, 0, Q, 8)
print("  " + json.dumps(out[0][1], sort_keys=True))
