"""Audit the served API the way a client sees it: api_audit.py [port]

The engine probes measure a graph replay.  This measures what a caller gets:
whether the text is right, when the first token lands, and what happens when
several callers arrive at once.  Every number here is wall clock from the
client side -- the server's own decode clock is read too, so the two can be
compared and a gap between them attributed to queueing.
"""
import json
import sys
import threading
import time
import urllib.error
import urllib.request

PORT = 8000
KEY = 'devkey'


def call(prompt, max_tokens=48, stream=False, model='local', timeout=600,
         headers=True):
    """One request.  Returns (first_token_s, total_s, text, server_metrics)."""
    body = {'model': model, 'max_tokens': max_tokens, 'stream': stream,
            'messages': [{'role': 'user', 'content': prompt}]}
    head = {'Content-Type': 'application/json'}
    if headers:
        head['Authorization'] = 'Bearer ' + KEY
    req = urllib.request.Request('http://127.0.0.1:%d/v1/chat/completions' % PORT,
                                 json.dumps(body).encode(), head)
    t0 = time.perf_counter()
    first = None
    text = ''
    metrics = {}
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        if not stream:
            payload = json.load(resp)
            first = time.perf_counter() - t0
            text = payload['choices'][0]['message']['content']
            metrics = payload.get('ljqinfer', {})
        else:
            for raw in resp:
                line = raw.decode('utf-8', 'replace').strip()
                if not line.startswith('data:'):
                    continue
                chunk = line[5:].strip()
                if chunk == '[DONE]':
                    break
                event = json.loads(chunk)
                piece = event['choices'][0].get('delta', {}).get('content') or ''
                if piece and first is None:
                    first = time.perf_counter() - t0
                text += piece
                if event.get('ljqinfer'):
                    metrics = event['ljqinfer']
    return first, time.perf_counter() - t0, text, metrics


# ---------------------------------------------------------------- semantics
CASES = [
    ('2+2 等于几？只回答数字。', ('4',), 16),
    ('Reply with exactly the word: BANANA', ('BANANA', 'banana'), 16),
    ('用一个词回答：中国的首都是哪里？', ('北京',), 16),
    ('Count from 1 to 8, space separated, nothing else.',
     ('1 2 3 4 5 6 7 8',), 40),
    ('Write a python one-liner that sums a list named xs.', ('sum(', 'xs'), 40),
    ('法国的首都是巴黎。意大利的首都是罗马。日本的首都是东京。'
     '请问意大利的首都是哪里？只答城市名。', ('罗马',), 24),
]


def semantics():
    print('== semantics (greedy, must contain expected substring)', flush=True)
    bad = 0
    for prompt, expect, ntok in CASES:
        _, wall, text, met = call(prompt, max_tokens=ntok)
        flat = text.replace('\n', ' ').strip()
        ok = any(e in text for e in expect)
        bad += not ok
        print('  [%s] %5.2fs %-46s -> %s' % (
            'ok' if ok else 'BAD', wall, prompt[:44].replace('\n', ' '),
            flat[:70]), flush=True)
        if not ok:
            print('        expected one of %r' % (expect,), flush=True)
    print('  semantics: %d/%d ok' % (len(CASES) - bad, len(CASES)), flush=True)
    return bad


# --------------------------------------------------------------- concurrency
def concurrency(n, ntok=64, stagger=0.0, prompt=None):
    """n callers, optionally arriving `stagger` seconds apart."""
    prompt = prompt or 'Count from 1 to 200, one number per line.'
    out = [None] * n
    err = [None] * n

    def worker(i):
        try:
            time.sleep(i * stagger)
            out[i] = call(prompt, max_tokens=ntok, stream=True)
        except Exception as exc:                      # noqa: BLE001
            err[i] = '%s: %s' % (type(exc).__name__, exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
    t0 = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.perf_counter() - t0
    good = [o for o in out if o]
    if not good:
        print('  n=%-2d ALL FAILED %s' % (n, err[:2]), flush=True)
        return
    ttft = sorted(o[0] for o in good if o[0] is not None)
    toks = sum(o[3].get('output_tokens', 0) for o in good)
    # What the row actually rides: its own model step, and the seconds it sat
    # frozen while somebody else's prefill ran.
    srv = [o[3].get('model_step_host_ms', 0) for o in good]
    stalls = [o[3].get('prefill_stall_seconds', 0) for o in good]
    # Time the client waited beyond what its own decode should have cost.
    queued = [o[1] - o[3].get('decode_steps', 0)
              * o[3].get('wall_ms_per_step', 0) / 1000.0
              - o[3].get('prefill_seconds', 0) for o in good]
    print('  n=%-2d stagger=%.2fs wall=%6.2fs thr=%6.1ftok/s  '
          'ttft p50=%5.2fs max=%5.2fs  step=%5.2fms stall=%4.2fs  '
          'overhead p50=%5.2fs  ok=%d/%d' % (
              n, stagger, wall, toks / wall if wall else 0,
              ttft[len(ttft) // 2], ttft[-1], sum(srv) / len(srv),
              sum(stalls) / len(stalls),
              sorted(queued)[len(queued) // 2], len(good), n), flush=True)
    if any(err):
        print('       errors: %s' % [e for e in err if e][:3], flush=True)


# -------------------------------------------------------------------- limits
def limits():
    print('== limits and errors', flush=True)
    checks = []

    def probe(name, fn):
        try:
            checks.append((name, fn()))
        except urllib.error.HTTPError as exc:
            checks.append((name, 'HTTP %d' % exc.code))
        except Exception as exc:                      # noqa: BLE001
            checks.append((name, '%s' % type(exc).__name__))

    probe('max_tokens=1', lambda: 'text=%r' % call('Say hi', 1)[2][:20])
    probe('no auth header',
          lambda: 'accepted' if call('hi', 4, headers=False) else '?')
    probe('unknown model', lambda: 'accepted (model ignored)'
          if call('hi', 4, model='no-such-model') else '?')
    probe('empty messages', lambda: _empty())
    probe('over max_seq', lambda: 'accepted len=%d'
          % len(call('word ' * 200000, 4)[2]))
    for name, result in checks:
        print('  %-18s %s' % (name, result), flush=True)


def _empty():
    req = urllib.request.Request(
        'http://127.0.0.1:%d/v1/chat/completions' % PORT,
        json.dumps({'model': 'local', 'messages': [], 'max_tokens': 4}).encode(),
        {'Content-Type': 'application/json', 'Authorization': 'Bearer ' + KEY})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return 'accepted %d' % resp.status


def main():
    global PORT
    args = sys.argv[1:]
    if args and args[0].isdigit():
        PORT = int(args[0])
    print('audit against 127.0.0.1:%d' % PORT, flush=True)
    bad = semantics()
    print('== concurrency (all arrive together)', flush=True)
    for n in (1, 2, 4, 6, 8):
        concurrency(n)
    print('== concurrency (staggered arrival, exercises join/leave)', flush=True)
    for n, gap in ((4, 0.30), (8, 0.15)):
        concurrency(n, stagger=gap)
    limits()
    print('AUDIT semantics_bad=%d' % bad, flush=True)


if __name__ == '__main__':
    main()
