"""Ten solo runs of one greedy sentence: which knob moves when the answer moves?"""
import json, urllib.request

PORT, KEY = 8000, "devkey"
Q = "Name three colors and explain why each one is calming. Be concise."
FIELDS = ("cache_hit_tokens", "chunks", "prefill_tokens", "accepted_tokens",
          "decode_steps", "output_tokens", "mtp_accepted_per_step")


def call(prompt, max_tokens=64):
    body = {"model": "local", "stream": True, "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}]}
    req = urllib.request.Request(
        "http://127.0.0.1:%d/v1/chat/completions" % PORT,
        json.dumps(body).encode(),
        {"Content-Type": "application/json", "Authorization": "Bearer " + KEY})
    text, stats = "", {}
    with urllib.request.urlopen(req, timeout=300) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data: ") or line[6:] == "[DONE]":
                continue
            ev = json.loads(line[6:])
            if ev.get("ljqinfer"):
                stats = ev["ljqinfer"]
            for ch in ev.get("choices") or []:
                text += (ch.get("delta") or {}).get("content") or ""
    return text, stats


runs = []
for i in range(10):
    t, st = call(Q)
    runs.append(t)
    print("  run#%2d %s" % (i, "  ".join("%s=%s" % (f, st.get(f)) for f in FIELDS)),
          flush=True)

base = max(set(runs), key=runs.count)
print("== distinct answers: %d / 10 (majority appears %d times)"
      % (len(set(runs)), runs.count(base)))
for i, t in enumerate(runs):
    if t != base:
        n = min(len(t), len(base))
        pos = next((j for j in range(n) if t[j] != base[j]), n)
        print("  run#%d forks at char %d: %r  vs majority %r"
              % (i, pos, t[pos:pos + 30], base[pos:pos + 30]))
