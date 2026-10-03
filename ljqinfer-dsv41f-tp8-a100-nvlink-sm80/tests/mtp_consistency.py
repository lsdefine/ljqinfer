"""Same sentence, different slot, alone or in a batch: the answer must not move."""
import json, threading, time, urllib.request

PORT, KEY = 8000, "devkey"
Q = "Name three colors and explain why each one is calming. Be concise."
FILLER = ["Explain gravity in two sentences.",
          "What is the capital of Japan?",
          "Write one line about the sea."]


def call(out, i, prompt, max_tokens=64):
    body = {"model": "local", "stream": True, "max_tokens": max_tokens,
            "temperature": 0.0,
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
    out[i] = (text, stats)


def show(tag, text, st):
    print("  %-14s mtp=%.3f  steps=%3s  out=%3s tok  step=%5.2fms  text=%r"
          % (tag, st.get("mtp_accepted_per_step", -1), st.get("decode_steps"),
             st.get("output_tokens"), st.get("model_step_host_ms", 0), text[:60]))
    return text


print("== A. alone (b=1), same sentence three times")
alone = []
for i in range(3):
    out = [None]
    call(out, 0, Q)
    alone.append(show("alone#%d" % i, *out[0]))

print("== B. four identical sentences boarding together (b=4)")
out = [None] * 4
ths = [threading.Thread(target=call, args=(out, i, Q)) for i in range(4)]
for t in ths: t.start()
for t in ths: t.join()
same = [show("batch4#%d" % i, *out[i]) for i in range(4)]

print("== C. the sentence riding with three strangers (b=4)")
prompts = [Q] + FILLER
out = [None] * 4
ths = [threading.Thread(target=call, args=(out, i, prompts[i])) for i in range(4)]
for t in ths: t.start()
for t in ths: t.join()
mixed = show("mixed#0", *out[0])
for i in (1, 2, 3):
    show("filler#%d" % i, *out[i])

print("== verdict")
print("  alone self-consistent : %s" % (len(set(alone)) == 1))
print("  batch4 all equal      : %s" % (len(set(same)) == 1))
print("  batch4 == alone       : %s" % (same[0] == alone[0]))
print("  mixed  == alone       : %s" % (mixed == alone[0]))
