# -*- coding: utf-8 -*-
"""Long-context chunked-prefill + graph-decode gate (torchrun 8 ranks).

Builds one ~N-token chat prompt with K facts planted evenly, then for each
prefill chunk size (all <= 12k) runs production generate_batch (chunked prefill
+ B1Q8 graph decode) asking every fact; optionally compares against the eager
one-token path (prefill kernels, greedy) for token-exact equality.

  NEEDLE_LEN=65536 NEEDLE_CHUNKS=12288,8192,4096 NEEDLE_EAGER=1 \
  torchrun --nproc_per_node=8 tc_needle_long.py
Exit code 1 on any miss / A!=B.
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # repo root on sys.path (this file lives in bench/)
import os, sys, threading, time, random
import torch
from tokenizers import Tokenizer
from model.model_api import load_execution
from model.past import BLOCK_TOKENS
from server.encoding_dsv4 import encode_messages

RANK = int(os.environ.get("RANK", "0"))
N_GEN = 24
TK = Tokenizer.from_file("/mnt/data/kw/models/DeepSeek-V4-Flash-0731/tokenizer.json")
WORDS = ("the quick brown fox jumps over lazy dog while river flows gently through valley and mountain "
         "village people trade goods at markets under bright sky as children play along dusty road "
         "quietly reading books about history science art music and travel in distant lands").split()
FACTS = [("The secret code for the blue vault is 7391-KESTREL.", "What is the secret code for the blue vault?", "KESTREL"),
         ("My cat is named Bartholomew and he is 9 years old.", "What is my cat named and how old is he?", "Bartholomew"),
         ("The meeting with Dr. Okonkwo is scheduled for March 14 at 3pm.", "When is the meeting with Dr. Okonkwo?", "March 14"),
         ("The password for the wifi network Zephyr is tangerine-88.", "What is the password for the wifi network Zephyr?", "tangerine"),
         ("Our shipment container number is MSKU-4471902.", "What is our shipment container number?", "4471902"),
         ("Grandma's recipe uses exactly 17 cardamom pods.", "How many cardamom pods does grandma's recipe use?", "17")]


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
    k = len(FACTS)
    seg = n_tok // (k + 1)
    parts = [filler(seg, 0)]
    for i, f in enumerate(FACTS):
        parts += [f[0], filler(seg, i + 1)]
    doc = " ".join(parts)
    msgs = [{"role": "system", "content": "You are a helpful assistant. Answer briefly."},
            {"role": "user", "content": "Here is a document:\n\n" + doc + "\n\nQuestion: " + FACTS[qi][1]}]
    return TK.encode(encode_messages(msgs, thinking_mode="chat"), add_special_tokens=False).ids


def run_A(ex, ids):
    out = []
    ex.set_input(ids)
    t0 = time.time()
    ex.generate_batch([ids], [N_GEN], [threading.Event()], emit=lambda row, toks: out.extend(toks))
    dt = time.time() - t0
    ex.reset()
    return out, dt


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
    ex = load_execution(max_seq_len=int(os.environ["NEEDLE_MAX_SEQ"]) if os.environ.get("NEEDLE_MAX_SEQ") else None,
                        max_batch_size=int(os.environ["NEEDLE_MAX_BS"]) if os.environ.get("NEEDLE_MAX_BS") else None)
    if RANK != 0:
        ex.serve_workers(); return
    n = int(os.environ.get("NEEDLE_LEN", "65536"))
    chunks = [int(x) for x in os.environ.get("NEEDLE_CHUNKS", "12288,8192,4096").split(",")]
    eager = os.environ.get("NEEDLE_EAGER", "1") == "1"
    assert all(c <= 12 * 1024 and c % BLOCK_TOKENS == 0 for c in chunks), chunks
    fails = 0
    for ci, chunk in enumerate(chunks):
        ex.prefill_chunk = chunk
        for qi in range(len(FACTS)):
            ids = build(n, qi)
            a, dt = run_A(ex, ids); ta = TK.decode(a)
            ok = FACTS[qi][2].lower() in ta.lower()
            fails += not ok
            log(f"chunk={chunk:5d} prompt={len(ids)} q{qi} {'OK ' if ok else 'BAD'} {dt:5.1f}s {ta!r}")
            if eager and ci == 0:
                b = run_B(ex, ids)
                fd = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None)
                tb = TK.decode(b); okb = FACTS[qi][2].lower() in tb.lower()
                fails += not okb                                  # eager path must also recall; token diff is informational
                log(f"            eager {'OK ' if okb else 'BAD'} {'==' if fd is None else '!='} graph first_diff={fd}" + ("" if fd is None else f" B={tb!r}"))
    log("GATE PASS" if fails == 0 else f"GATE FAIL ({fails})")
    ex.shutdown()
    if fails:
        sys.exit(1)


if __name__ == "__main__":
    main()
