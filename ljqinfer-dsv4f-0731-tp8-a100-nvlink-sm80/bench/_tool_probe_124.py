import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # repo root on sys.path (this file lives in bench/)
import json, time, requests
base = 'http://127.0.0.1:8000'; H = {'Authorization': 'Bearer devkey'}
tools = [{'type': 'function', 'function': {'name': 'get_weather', 'description': 'Get current weather for a location',
          'parameters': {'type': 'object', 'properties': {'location': {'type': 'string'}}, 'required': ['location']}}}]
msgs = [{'role': 'user', 'content': 'What is the weather in Tokyo? Use get_weather. Do not answer from memory.'}]


def call(m, mt):
    t = time.time()
    r = requests.post(base + '/v1/chat/completions', headers=H, timeout=900,
                      json={'model': 'ljqinfer-dsv4f-0731', 'messages': m, 'tools': tools, 'max_tokens': mt, 'temperature': float(__import__('os').environ.get('TEMP','0'))})
    x = r.json()
    if r.status_code != 200:
        print('BODY', json.dumps(x)[:400])
    print('STATUS', r.status_code, 'wall %.2f' % (time.time() - t))
    print('USAGE', json.dumps(x.get('usage')))
    return x


x = call(msgs, 128); ch = x['choices'][0]
print('finish', ch['finish_reason'], 'tool_calls', json.dumps(ch['message'].get('tool_calls'))[:300])
msgs.append(ch['message'])
msgs.append({'role': 'tool', 'tool_call_id': ch['message']['tool_calls'][0]['id'], 'content': '{"temp_c":27,"condition":"light rain","humidity":80}'})
msgs.append({'role': 'user', 'content': 'Now write a detailed 400-word paragraph of travel advice for Tokyo based on that weather.'})
x = call(msgs, 512); ch = x['choices'][0]
print('finish', ch['finish_reason'])
print('TEXT', ch['message']['content'][:600])
