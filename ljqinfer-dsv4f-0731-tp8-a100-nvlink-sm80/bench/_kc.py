
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # repo root on sys.path (this file lives in bench/)
import json,collections
d=json.load(open('/tmp/prof_decode_rank0.json'))
ev=d['traceEvents']
tags=[e for e in ev if e.get('cat')=='user_annotation']
kern=[e for e in ev if e.get('cat')=='kernel']
# map by correlation: kernel has args.correlation; find cpu op's annotation via time containment on cpu thread
tags.sort(key=lambda e:e['ts'])
import bisect
# build map correlation->(ts on cpu) via cuda_runtime events
rt={e['args'].get('correlation'):e for e in ev if e.get('cat')=='cuda_runtime' and 'args' in e}
cnt=collections.Counter(); tm=collections.Counter()
leaf=[t for t in tags if t['name'] in ('attn.indexer','attn.o_proj','attn.compressor','moe.fused_ffn','hc.pre_norm','hc.post','attn.kv_proj','attn.q_proj','attn.write_ckv','attn.all_rows','head.logits','embed','mtp.layer','mtp.head','attn.sparse_core')]
for k in kern:
    c=k['args'].get('correlation'); r=rt.get(c)
    if not r: continue
    ts=r['ts']; found='other'
    for t in leaf:
        if t['ts']<=ts<=t['ts']+t['dur']: found=t['name']; break
    cnt[found]+=1; tm[found]+=k['dur']
N=4
for k,v in sorted(cnt.items(),key=lambda x:-x[1]): print(f"{k:18s} {v//N:5d} kern/step {tm[k]/N/1000:7.2f} ms")
