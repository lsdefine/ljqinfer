"""Concurrent step time from the server's own decode clock: probe_conc.py [B ...]"""
import json, sys, threading, time, urllib.request

def one(words, ntok, port, out, idx):
    body = json.dumps({'model': 'local', 'max_tokens': ntok,
                       'messages': [{'role': 'user', 'content': 'filler ' * words +
                                     ' Now count from 1 to 400, one number per line.'}]}).encode()
    req = urllib.request.Request(f'http://127.0.0.1:{port}/v1/chat/completions', body,
                                 {'Content-Type': 'application/json',
                                  'Authorization': 'Bearer devkey'})
    t0 = time.perf_counter()
    r = json.load(urllib.request.urlopen(req, timeout=600))
    wall = time.perf_counter() - t0
    m = r['ljqinfer']
    n = m['output_tokens']
    out[idx] = (m, n, wall, r['choices'][0]['message']['content'][:40])

def run(B, ntok=64, words=4, port=8000):
    out = [None] * B
    ths = [threading.Thread(target=one, args=(words, ntok, port, out, i)) for i in range(B)]
    t0 = time.perf_counter()
    for t in ths: t.start()
    for t in ths: t.join()
    wall = time.perf_counter() - t0
    steps = [o[0]['wall_ms_per_step'] for o in out]
    nsteps = sum(o[0]['decode_steps'] for o in out)
    toks = sum(o[1] for o in out)
    print(f'B={B} wall={wall:7.3f}s tokens={toks:5d} thr={toks/wall:7.1f}tok/s '
          f'step_mean={sum(steps)/B:7.3f}ms steps={nsteps:4d} step_each=' +
          ' '.join(f'{s:.2f}' for s in steps), flush=True)
    print('   sample:', out[0][3].replace(chr(10), '|'), flush=True)

if __name__ == '__main__':
    for b in (int(a) for a in sys.argv[1:] or ['1']):
        run(b)
