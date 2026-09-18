# -*- coding: utf-8 -*-
"""Needle-in-haystack probe against the local service (temp=0).
Plants 3 facts early, pads with filler to ~N tokens, asks each fact. Prints answers.
NEEDLE_LENS=2000,4000,8000 python _needle_probe.py
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # repo root on sys.path (this file lives in bench/)
import os, json, time, random, urllib.request

URL = 'http://localhost:8000/v1/chat/completions'
HDR = {'Content-Type': 'application/json', 'Authorization': 'Bearer devkey'}
MODEL = 'ljqinfer-dsv4f-0731'

FACTS = [('The secret code for the blue vault is 7391-KESTREL.', 'What is the secret code for the blue vault?', '7391-KESTREL'),
         ('My cat is named Bartholomew and he is 9 years old.', 'What is my cat named and how old is he?', 'Bartholomew'),
         ('The meeting with Dr. Okonkwo is scheduled for March 14th at 3:45 PM.', 'When is the meeting with Dr. Okonkwo?', 'March 14')]

WORDS = ('the quick brown fox jumps over the lazy dog while the river flows quietly through the valley and '
         'mountains rise in the distance under a pale morning sky as travellers walk along the dusty road '
         'discussing weather crops markets and the price of grain in the nearby town').split()

def filler(n_words, seed):
    r = random.Random(seed)
    out = []
    while len(out) < n_words:
        k = r.randint(8, 18)
        out.append(' '.join(r.choice(WORDS) for _ in range(k)).capitalize() + '.')
    return ' '.join(out)

def chat(messages, max_tokens=64):
    body = json.dumps({'model': MODEL, 'messages': messages, 'temperature': 0, 'max_tokens': max_tokens}).encode()
    t = time.time()
    req = urllib.request.Request(URL, data=body, headers=HDR)
    with urllib.request.urlopen(req, timeout=600) as r:
        j = json.load(r)
    return j['choices'][0]['message']['content'], j.get('usage', {}), time.time() - t

def run(n_tok, salt):
    n_words = int(n_tok * 0.75 * 375 / 4892)  # measured: 4892 words ~ 5319 tokens at filler(375)
    # facts spread: 1st at start, 2nd at 1/3, 3rd at 2/3
    seg = n_words // 3
    doc = (f'[{salt}] ' + FACTS[0][0] + ' ' + filler(seg, salt * 10 + 1) + ' ' + FACTS[1][0] + ' '
           + filler(seg, salt * 10 + 2) + ' ' + FACTS[2][0] + ' ' + filler(seg, salt * 10 + 3))
    sys_msgs = [{'role': 'system', 'content': 'You are a careful assistant. Answer briefly using only the document.'},
                {'role': 'user', 'content': 'Here is a document, remember it:\n\n' + doc + '\n\nSay OK.'}]
    ans, usage, dt = chat(sys_msgs, 8)
    print(f'  [prime] prompt_tokens={usage.get("prompt_tokens")} {dt:.1f}s -> {ans!r}')
    hist = sys_msgs + [{'role': 'assistant', 'content': ans}]
    ok = 0
    for fact, q, key in FACTS:
        if os.environ.get('SINGLE'):   # doc + question in one request: no cold-KV cache hit
            msgs = [sys_msgs[0], {'role': 'user', 'content': 'Here is a document:\n\n' + doc + '\n\n' + q}]
        else:
            msgs = hist + [{'role': 'user', 'content': q}]
        a, usage, dt = chat(msgs, 48)
        hit = key.lower() in a.lower()
        ok += hit
        print(f'  [{"OK " if hit else "BAD"}] p={usage.get("prompt_tokens")} c={usage.get("completion_tokens")} {dt:.1f}s q={q!r} -> {a!r}')
    return ok

if __name__ == '__main__':
    lens = [int(x) for x in os.environ.get('NEEDLE_LENS', '2000,4000,8000').split(',')]
    for n in lens:
        print(f'=== N~{n} tokens ===')
        s = run(n, n)
        print(f'  score {s}/3')
