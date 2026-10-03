"""Does a changing prompt length pay for a SWA graph recapture every time?

Group A sends the same token length with different content; group B grows the
length by one word each time.  If the graph is the reason, A settles after the
first request and B never settles.
"""
import json
import urllib.request
import uuid

URL = "http://127.0.0.1:8000/v1/messages"
HEAD = {"content-type": "application/json", "x-api-key": "devkey"}


def ask(text):
    body = json.dumps({"model": "ljqinfer-dsv41f", "max_tokens": 4,
                       "messages": [{"role": "user", "content": text}]}).encode()
    req = urllib.request.Request(URL, body, HEAD)
    with urllib.request.urlopen(req, timeout=300) as r:
        stats = json.loads(r.read())["usage"]
    return stats


def show(tag, text):
    s = ask(text)
    print("%-14s in=%-5d hit=%-5d compute_s=%.4f prefill_s=%.4f"
          % (tag, s["input_tokens"], s.get("cache_hit_tokens", 0),
             s.get("prefill_compute_seconds", 0.0), s.get("model_prefill_seconds", 0.0)),
          flush=True)


filler = "banana " * 120
for i in range(5):
    show("same_len_%d" % i, uuid.uuid4().hex + " " + filler + " reply ok")
for i in range(4):
    show("grow_len_%d" % i, uuid.uuid4().hex + " " + filler + " word" * (i + 1) + " reply ok")
