"""Where does prefill wall time actually go?  One line per request."""
import json
import urllib.request
import uuid

KEYS = ("input_tokens", "cache_hit_tokens", "prefill_tokens",
        "cache_load_seconds", "prefill_compute_seconds",
        "finish_prefill_seconds", "model_prefill_seconds",
        "time_to_first_token_seconds")


def ask(text):
    body = json.dumps({"model": "ljqinfer-dsv41f", "max_tokens": 8,
                       "messages": [{"role": "user", "content": text}]}).encode()
    req = urllib.request.Request("http://127.0.0.1:8000/v1/messages", body,
                                 {"content-type": "application/json",
                                  "x-api-key": "devkey"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read()).get("ljqinfer") or {}


def show(tag, st):
    print("%-14s %s" % (tag, "  ".join(
        "%s=%s" % (k.replace("_seconds", "_s").replace("_tokens", "_tok"),
                   st.get(k)) for k in KEYS)))


if __name__ == "__main__":
    n = uuid.uuid4().hex
    short = "hi " + n
    mid = ("背景资料%s。" % n) + ("缓存机制的设计要点是块对齐与前缀复用。" * 60)
    long = ("背景资料%s。" % n) + ("缓存机制的设计要点是块对齐与前缀复用。" * 260)
    show("tiny_cold", ask(short))
    show("tiny_again", ask(short))
    show("mid_cold", ask(mid))
    show("long_cold", ask(long))
    show("long_warm", ask(long))
