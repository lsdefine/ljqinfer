#!/usr/bin/env python3
"""Post-fix online regression: length x batch x cold-KV reuse.

Judgements (all must hold):
  J1 ANSWER : each row recovers its own secret code (no cross-row contamination)
  J2 R1==R2 : identical prompt re-sent later gives byte-identical answer
              -> cold-KV store/load is lossless
  J3 B1==B4 : same prompt answered alone vs in a batch of 4 gives byte-identical
              answer -> batch invariance holds at real prompt lengths
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # repo root on sys.path (this file lives in bench/)
import json, threading, urllib.request, sys

URL = "http://127.0.0.1:8000/v1/chat/completions"
HDR = {"Content-Type": "application/json", "Authorization": "Bearer devkey"}

FILLER = ("The archive room stores quarterly maintenance records for the northern facility. "
          "Technicians log every calibration event with a timestamp and an operator initial. "
          "Routine inspections cover pressure valves, coolant lines, and backup generators. ")


def make_prompt(uid, words):
    """Unique prefix per row (uid first), needle buried in the middle."""
    head = "Session %s maintenance dossier. " % uid
    body = FILLER * max(1, words // 30)
    half = len(body) // 2
    needle = "IMPORTANT: the secret code for session %s is %s. " % (uid, code_of(uid))
    q = ("\n\nQuestion: what is the secret code for session %s? "
         "Reply with the code only, nothing else." % uid)
    return head + body[:half] + needle + body[half:] + q


def code_of(uid):
    return "ZK-%d" % (7000 + uid * 137 % 900)


def ask(uid, words, out, idx):
    payload = {"model": "ljqinfer-dsv4f-0731",
               "messages": [{"role": "user", "content": make_prompt(uid, words)}],
               "max_tokens": 24, "temperature": 0}
    req = urllib.request.Request(URL, data=json.dumps(payload).encode(), headers=HDR)
    try:
        with urllib.request.urlopen(req, timeout=180) as r:
            j = json.loads(r.read().decode())
        out[idx] = j["choices"][0]["message"]["content"].strip()
    except Exception as e:
        out[idx] = "ERROR:%s" % e


def run_batch(uids, words):
    out = [None] * len(uids)
    ts = [threading.Thread(target=ask, args=(u, words, out, i)) for i, u in enumerate(uids)]
    for t in ts: t.start()
    for t in ts: t.join()
    return out


def main():
    tiers = [("short_~80tok", 60), ("mid_~400tok_3blk", 300), ("long_~1600tok_12blk", 1200)]
    fails = []
    for name, words in tiers:
        uids = [11, 22, 33, 44]
        r1 = run_batch(uids, words)          # B=4 round 1 (cold)
        r2 = run_batch(uids, words)          # B=4 round 2 (should hit cold-KV)
        b1 = [run_batch([u], words)[0] for u in uids]   # B=1 one at a time

        for i, u in enumerate(uids):
            want = code_of(u)
            ok_ans = want in (r1[i] or "")
            ok_r12 = (r1[i] == r2[i])
            ok_b14 = (r1[i] == b1[i])
            tag = "PASS" if (ok_ans and ok_r12 and ok_b14) else "FAIL"
            if tag == "FAIL":
                fails.append((name, u, want, r1[i], r2[i], b1[i], ok_ans, ok_r12, ok_b14))
            print("  [%s] %-20s uid=%-3d want=%-8s J1=%-5s J2(R1==R2)=%-5s J3(B1==B4)=%-5s  got=%r"
                  % (tag, name, u, want, ok_ans, ok_r12, ok_b14, (r1[i] or "")[:40]))
        sys.stdout.flush()

    print("\n==== %d FAIL ====" % len(fails))
    for f in fails:
        print("  tier=%s uid=%s want=%s\n    R1=%r\n    R2=%r\n    B1=%r" % (f[0], f[1], f[2], f[3], f[4], f[5]))
    print("RESULT:", "ALL PASS" if not fails else "HAS FAILURES")


if __name__ == "__main__":
    main()
