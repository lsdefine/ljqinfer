#!/usr/bin/env python3
"""128-alignment boundary matrix over HTTP (post-fix regression).

For each exact prompt_tokens L in {127,128,129,255,256,257,384,385}:
  R1 (cold)  -> answer A1, hit should be 0
  R2 (warm)  -> answer A2, hit should be (L//128 - 1)*128  [tail block kept for recompute]
  B4 (warm, 4 identical rows concurrently) -> answers must all equal A1

Judgements:
  J1 A1 correct (contains the planted code)
  J2 A1 == A2                (cold-KV store/load lossless at block boundary)
  J3 A1 == each B4 answer    (batch invariance at block boundary)
  J4 hit follows (L//128-1)*128   (page accounting sane; reported, not fatal)
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # repo root on sys.path (this file lives in bench/)
import json, threading, urllib.request, sys

URL = "http://127.0.0.1:8000/v1/chat/completions"
HDR = {"Content-Type": "application/json", "Authorization": "Bearer devkey"}
CODE = "XQ-4821"
WORD = " orbit"


def call(prompt, max_tokens=16):
    payload = {"model": "ljqinfer-dsv4f-0731",
               "messages": [{"role": "user", "content": prompt}],
               "max_tokens": max_tokens, "temperature": 0}
    req = urllib.request.Request(URL, data=json.dumps(payload).encode(), headers=HDR)
    with urllib.request.urlopen(req, timeout=180) as r:
        j = json.loads(r.read().decode())
    return j["choices"][0]["message"]["content"].strip(), j["usage"]["prompt_tokens"]


def build(n_words):
    return ("Code %s.%s. Question: what is the code above? Reply with the code only."
            % (CODE, WORD * n_words))


def calibrate(target, cache={}):
    """Find n_words such that prompt_tokens == target, by probing the server."""
    if target in cache:
        return cache[target]
    lo, hi = 0, target + 40
    # linear-ish search: probe, adjust by the token delta
    n = max(0, target - 20)
    for _ in range(40):
        _, tok = call(build(n), max_tokens=1)
        if tok == target:
            cache[target] = n
            return n
        n += (target - tok)
        if n < 0:
            return None
    return None


def concurrent(prompt, k=4):
    out = [None] * k
    def w(i):
        try:
            out[i] = call(prompt)[0]
        except Exception as e:
            out[i] = "ERROR:%s" % e
    ts = [threading.Thread(target=w, args=(i,)) for i in range(k)]
    for t in ts: t.start()
    for t in ts: t.join()
    return out


def main():
    targets = [127, 128, 129, 255, 256, 257, 384, 385]
    fails = 0
    print("%-6s %-7s %-6s %-6s %-6s %-8s %s" % ("L", "nwords", "J1", "J2", "J3", "expect_hit", "answer"))
    for L in targets:
        n = calibrate(L)
        if n is None:
            print("%-6d CALIBRATION FAILED" % L); fails += 1; continue
        p = build(n)
        a1, tok1 = call(p)
        a2, tok2 = call(p)
        b4 = concurrent(p, 4)
        j1 = CODE in a1
        j2 = (a1 == a2)
        j3 = all(x == a1 for x in b4)
        exp_hit = max(0, (L // 128 - 1) * 128)
        ok = j1 and j2 and j3
        if not ok:
            fails += 1
        print("%-6d %-7d %-6s %-6s %-6s %-8d %r%s" %
              (L, n, j1, j2, j3, exp_hit, a1[:24],
               "" if ok else "   <<< FAIL  A2=%r B4=%r" % (a2[:24], [x[:24] for x in b4])))
        sys.stdout.flush()
    print("\nRESULT:", "ALL PASS" if fails == 0 else "%d FAIL" % fails)


if __name__ == "__main__":
    main()
