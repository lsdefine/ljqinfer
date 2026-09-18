"""Real HTTP protocol gate; NOT a model-correctness or decode-step benchmark."""
from __future__ import annotations
import argparse
import concurrent.futures
import json
from pathlib import Path
import threading
import time
import urllib.request
import urllib.error


def sse_objects(response):
    data = []
    for raw in response:
        line = raw.decode('utf-8').rstrip('\r\n')
        if not line:
            if data:
                yield '\n'.join(data)
                data.clear()
        elif line.startswith('data:'):
            data.append(line[5:].lstrip())
    if data:
        yield '\n'.join(data)


def chat(base, key, prompt, tokens, stream, barrier=None):
    body = dict(messages=[dict(role='user', content=prompt)],
                max_tokens=tokens, stream=stream, temperature=0,
                reasoning_effort='none')
    if stream:
        body['stream_options'] = {'include_usage': True}
    request = urllib.request.Request(base + '/v1/chat/completions',
        data=json.dumps(body).encode(), headers={
            'Content-Type': 'application/json', 'Authorization': 'Bearer ' + key})
    if barrier:
        barrier.wait(timeout=30)
    start = time.perf_counter()
    with urllib.request.urlopen(request, timeout=600) as response:
        assert response.status == 200
        if not stream:
            result = json.load(response)
            assert result.get('choices'), result
            assert result['choices'][0].get('finish_reason') is not None, result
            assert result['choices'][0].get('message'), result
            return {'stream': False, 'wall_seconds': time.perf_counter()-start,
                    'response': result}
        events, ids, texts, done, terminal = [], set(), [], False, 0
        first = None
        for datum in sse_objects(response):
            assert not done, 'event after [DONE]'
            if datum == '[DONE]':
                done = True
                continue
            event = json.loads(datum)
            assert 'error' not in event, event
            if event.get('id'):
                ids.add(event['id'])
            for choice in event.get('choices', []):
                delta = choice.get('delta', {})
                text = delta.get('content') or delta.get('reasoning_content') or ''
                if text:
                    assert terminal == 0, 'text after terminal finish event'
                    if first is None:
                        first = time.perf_counter()-start
                    texts.append(text)
                if choice.get('finish_reason') is not None:
                    terminal += 1
            events.append(event)
        assert done and terminal == 1 and len(ids) == 1, (done, terminal, ids)
        assert texts, 'no generated text'
        return {'stream': True, 'wall_seconds': time.perf_counter()-start,
                'ttft_seconds': first, 'text': ''.join(texts), 'events': events}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base-url', default='http://127.0.0.1:8000')
    p.add_argument('--key', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--tokens', type=int, default=64)
    a = p.parse_args()
    output = Path(a.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {'scope': 'HTTP protocol only; numeric correctness and decode step time require separate gates',
              'started_at': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'cases': []}
    def save():
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    try:
        try:
            urllib.request.urlopen(a.base_url+'/v1/models', timeout=20)
        except urllib.error.HTTPError as e:
            assert e.code == 401, e.code
        else:
            raise AssertionError('unauthenticated model endpoint unexpectedly accepted')
        report['cases'].append({'name': 'unauthorized', 'passed': True}); save()
        req = urllib.request.Request(a.base_url+'/v1/models', headers={'Authorization':'Bearer '+a.key})
        with urllib.request.urlopen(req, timeout=20) as response:
            models = json.load(response)
            assert models.get('data'), models
        report['models'] = models; save()
        for stream in [False, True]:
            result = chat(a.base_url, a.key, '简短说明为什么天空是蓝色的。', a.tokens, stream)
            report['cases'].append({'name': 'single_stream' if stream else 'single_json',
                                    'passed': True, 'result': result}); save()
        barrier = threading.Barrier(4)
        prompts = ['只输出数字一到十。', '用两句话解释递归。',
                   'Write a short Python function to add two numbers.', '简短说明植物为什么需要阳光。']
        with concurrent.futures.ThreadPoolExecutor(4) as pool:
            futures = [pool.submit(chat, a.base_url, a.key, text, a.tokens, True, barrier)
                       for text in prompts]
            for i, future in enumerate(futures):
                report['cases'].append({'name': f'concurrent_{i}', 'passed': True,
                                        'result': future.result()}); save()
        report['passed'] = True
    except Exception as e:
        report.update(passed=False, error=repr(e))
        raise
    finally:
        save()
    print(json.dumps({'passed': report['passed'], 'output': str(output)}))


if __name__ == '__main__':
    main()
