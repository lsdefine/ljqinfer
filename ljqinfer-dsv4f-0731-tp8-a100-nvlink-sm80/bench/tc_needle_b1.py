# -*- coding: utf-8 -*-
"""Long-context needle discriminator (torchrun 8 ranks).
For each length: build chat prompt with 3 planted facts + filler, ask fact #3 (planted at ~2/3).
A: production generate_batch (prefill chunks + B1Q8 graph decode).
B: same prefill, then eager one-token forwards (prefill kernel path) greedy.
Prints both answers and whether the key appears.  NEEDLE_LENS=1500,5000 torchrun --nproc_per_node=8 tc_needle_b1.py
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # repo root on sys.path (this file lives in bench/)
import json, os, sys, threading, time, random
import torch
from tokenizers import Tokenizer
from model.model_api import load_execution
from server.encoding_dsv4 import encode_messages

RANK = int(os.environ.get("RANK", "0"))
N_GEN = 24
TK = Tokenizer.from_file("/mnt/data/kw/models/DeepSeek-V4-Flash-0731/tokenizer.json")
WORDS = ("the quick brown fox jumps over lazy dog while river flows gently through valley and mountain "
         "village people trade goods at markets under bright sky as children play along dusty road "
         "quietly reading books about history science art music and travel in distant lands").split()
FACTS = [("The secret code for the blue vault is 7391-KESTREL.", "What is the secret code for the blue vault?", "KESTREL"),
         ("My cat is named Bartholomew and he is 9 years old.", "What is my cat named and how old is he?", "Bartholomew"),
         ("The meeting with Dr. Okonkwo is scheduled for March 14 at 3pm.", "When is the meeting with Dr. Okonkwo?", "March 14")]


def log(*a):
    if RANK == 0:
        print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def filler(n_tok, seed):
    rng = random.Random(seed)
    out, n = [], 0
    while n < n_tok:
        s = " ".join(rng.choice(WORDS) for _ in range(rng.randint(8, 16))).capitalize() + "."
        out.append(s); n += len(s.split()) * 1.1
    return " ".join(out)


def build(n_tok, qi):
    seg = n_tok // 3
    doc = (FACTS[0][0] + " " + filler(seg, 1) + " " + FACTS[1][0] + " " + filler(seg, 2) + " "
           + FACTS[2][0] + " " + filler(seg, 3))
    msgs = [{"role": "system", "content": "You are a helpful assistant. Answer briefly."},
            {"role": "user", "content": "Here is a document:\n\n" + doc + "\n\nQuestion: " + FACTS[qi][1]}]
    return TK.encode(encode_messages(msgs, thinking_mode="chat"), add_special_tokens=False).ids


def run_A(ex, ids):
    out = []
    ex.set_input(ids)
    ex.generate_batch([ids], [N_GEN], [threading.Event()], emit=lambda row, toks: out.extend(toks))
    ex.reset()
    return out


def run_B(ex, ids):
    ex.set_input(ids)
    slot = ex._slot(0)
    pos, logits = 0, None
    while pos < len(ids):
        n = min(ex.prefill_chunk, len(ids) - pos)
        logits = ex._forward(slot, pos, ids[pos:pos + n]); pos += n
    out = []
    for _ in range(N_GEN):
        t = int(torch.argmax(logits).item()); out.append(t)
        if t == 1:
            break
        logits = ex._forward(slot, pos, [t]); pos += 1
    ex._rpc("rewind", slot, pos)
    ex.reset()
    return out


def main():
    ex = load_execution()
    if RANK != 0:
        ex.serve_workers(); return
    lens = [int(x) for x in os.environ.get("NEEDLE_LENS", "1500,5000").split(",")]
    qis = [int(x) for x in os.environ.get("NEEDLE_QIS", "2,1").split(",")]
    for n in lens:
        for qi in qis:
            ids = build(n, qi)
            log(f"=== N={n} prompt={len(ids)} q={FACTS[qi][1]!r} key={FACTS[qi][2]}")
            a = run_A(ex, ids); ta = TK.decode(a)
            log(f"A graph : {'OK ' if FACTS[qi][2].lower() in ta.lower() else 'BAD'} {ta!r}")
            if os.environ.get("NEEDLE_EAGER", "0") == "1":
                b = run_B(ex, ids); tb = TK.decode(b)
                log(f"B eager : {'OK ' if FACTS[qi][2].lower() in tb.lower() else 'BAD'} {tb!r}")
                fd = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None)
                log(f"A==B: {a == b} first_diff={fd}")
                if fd is not None:
                    lo, hi = max(0, fd - 3), min(len(a), fd + 6)
                    log(f"A[{lo}:{hi}]={a[lo:hi]} -> {TK.decode(a[lo:hi])!r}")
                    log(f"B[{lo}:{hi}]={b[lo:hi]} -> {TK.decode(b[lo:hi])!r}")
    ex.shutdown()


if __name__ == "__main__":
    main()
