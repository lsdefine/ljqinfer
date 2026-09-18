#!/usr/bin/env python3
"""Long-context needle over HTTP: 8k / 16k tokens, needle at 25% / 50% / 90% depth.

The post-fix kernel drives indexer.proj -> page selection. Short prompts cannot
exercise it. Judgements per case:
  J1 answer correct (needle recovered at that depth)
  J2 R1 == R2 (cold-KV store/load lossless at long context)
  J3 R1 == B2 rows (batch invariance at long context)
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # repo root on sys.path (this file lives in bench/)
import json, threading, urllib.request, sys

URL = "http://127.0.0.1:8000/v1/chat/completions"
HDR = {"Content-Type": "application/json", "Authorization": "Bearer devkey"}
FILLER = ("The archive room stores quarterly maintenance records for the northern facility. "
          "Technicians log every calibration event with a timestamp and an operator initial. "
          "Routine inspections cover pressure valves, coolant lines and backup generators. ")


def call(prompt, mt=16):
    payload = {"model": "ljqinfer-dsv4f-0731",
               "messages": [{"role": "user", "content": prompt}],
               "max_tokens": mt, "temperature": 0}
    req = urllib.request.Request(URL, data=json.dumps(payload).encode(), headers=HDR)
    with urllib.request.urlopen(req, timeout=600) as r:
        j = json.loads(r.read().decode())
    return j["choices"][0]["message"]["content"].strip(), j["usage"]["prompt_tokens"]


def build(words, depth, code):
    body = FILLER * max(1, words // 30)
    cut = int(len(body) * depth)
    needle = " IMPORTANT: the vault code is %s. " % code
    return ("Facility dossier.\n" + body[:cut] + needle + body[cut:] +
            "\n\nQuestion: what is the vault code? Reply with the code only.")


def par(prompt, k):
    out = [None] * k
    def w(i):
        try: out[i] = call(prompt)[0]
        except Exception as e: out[i] = "ERR:%s" % e
    ts = [threading.Thread(target=w, args=(i,)) for i in range(k)]
    for t in ts: t.start()
    for t in ts: t.join()
    return out


def main():
    fails = 0
    for words, tag in [(6000, "~8k"), (12000, "~16k")]:
        for depth in (0.25, 0.50, 0.90):
            code = "VT-%d" % (3000 + int(depth * 100) + words // 1000)
            p = build(words, depth, code)
            a1, tok = call(p)
            a2, _ = call(p)
            b2 = par(p, 2)
            j1 = code in a1
            j2 = (a1 == a2)
            j3 = all(x == a1 for x in b2)
            ok = j1 and j2 and j3
            if not ok: fails += 1
            print("[%s] %-5s tok=%-6d depth=%-4.2f J1=%-5s J2=%-5s J3=%-5s want=%s got=%r"
                  % ("PASS" if ok else "FAIL", tag, tok, depth, j1, j2, j3, code, a1[:30]))
            sys.stdout.flush()
    print("\nRESULT:", "ALL PASS" if fails == 0 else "%d FAIL" % fails)


if __name__ == "__main__":
    main()
