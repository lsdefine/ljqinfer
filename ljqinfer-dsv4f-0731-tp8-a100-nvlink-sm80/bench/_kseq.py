
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # repo root on sys.path (this file lives in bench/)
import json,collections,sys
d=json.load(open('/tmp/prof_decode_rank0.json')); ev=d['traceEvents']
tags=sorted([e for e in ev if e.get('cat')=='user_annotation'],key=lambda e:e['ts'])
kern=[e for e in ev if e.get('cat')=='kernel']
rt={e['args'].get('correlation'):e for e in ev if e.get('cat')=='cuda_runtime' and 'args' in e}
for want in sys.argv[1:]:
    ts_list=[t for t in tags if t['name']==want]
    t=ts_list[len(ts_list)//2]  # a middle instance
    names=[]
    for k in kern:
        r=rt.get(k['args'].get('correlation'))
        if r and t['ts']<=r['ts']<=t['ts']+t['dur']: names.append((r['ts'],k['name'][:70],k['dur']))
    names.sort(); print('==',want,len(names))
    for _,n,du in names: print(f'  {du:6.1f}us {n}')
