"""Live HTTP regression; run explicitly with API_REPAIR_AUDIT set."""
def main():
 """GLM53 API repair regression. Run on node09 with the live TP8 service."""
 import json,time,threading,os
 from pathlib import Path
 from concurrent.futures import ThreadPoolExecutor
 import requests
 URL=os.environ.get('API_URL','http://127.0.0.1:8000')
 H={'Authorization':'Bearer '+os.environ.get('API_KEY','devkey')}
 R=Path(os.environ['API_REPAIR_AUDIT']);R.mkdir(exist_ok=True,parents=True)
 rows=[]
 def save(row):
  rows.append(row);(R/'lifecycle_results.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2));print(row['name'],row.get('ok'),flush=True)
 def payload(text,n=128):return {'model':'glm-5.3','messages':[{'role':'user','content':text}],'max_tokens':n}
 def chat(name,b,stream=False):
  b=dict(b,stream=stream);start=time.perf_counter();content='';done=False;chunks=[]
  with requests.post(URL+'/v1/chat/completions',headers=H,json=b,stream=stream,timeout=180) as r:
   r.raise_for_status()
   if not stream:
    data=r.json();content=data['choices'][0]['message'].get('content') or '';done=True
   else:
    b=[]
    for line in r.iter_lines(chunk_size=1):
     if not line.startswith(b'data: '):continue
     raw=line[6:]
     if raw==b'[DONE]':done=True;break
     x=json.loads(raw);chunks.append(x)
     assert 'error' not in x,x
     for c in x.get('choices',[]):content+=c.get('delta',{}).get('content') or ''
    data=chunks
  (R/(name+'.json')).write_text(json.dumps(data,ensure_ascii=False))
  assert done and '</think>' not in content and '<think>' not in content,(name,content)
  return {'name':name,'ok':True,'content':content,'seconds':time.perf_counter()-start}
 models=requests.get(URL+'/v1/models',headers=H,timeout=10).json()
 assert models['object']=='list' and models['data'][0]['object']=='model' and 'created' in models['data'][0]
 save({'name':'models_contract','ok':True})
 for stream in (False,True):
  r=chat('exact_'+str(stream),payload('只回答以下字符串，不解释、不加标点：READY_731'),stream)
  assert r['content'].strip()=='READY_731',r;save(r)
 # Repeated 4->1 cohorts allocate >96 ordinary streams, cross the CUDA pool ring.
 # Unequal output caps exercise row removal and graph recapture; then cancel
 # after real content and immediately generate again, never restart in-between.
 for cycle in range(8):
  gate=threading.Barrier(4)
  def job(i):
   gate.wait();tag=f'ROW_{cycle}_{i}'
   return chat(f'cohort_{cycle}_{i}',payload(f'第一行写{tag}，然后用中文详细解释数据库事务隔离和MVCC。',[24,48,80,128][i]),True)
  with ThreadPoolExecutor(max_workers=4) as pool:out=list(pool.map(job,range(4)))
  for i,r in enumerate(out):
   assert f'ROW_{cycle}_{i}' in r['content'],r
   assert all(f'ROW_{cycle}_{j}' not in r['content'] for j in range(4) if j!=i),r
   save(r)
  save(chat('single_'+str(cycle),payload('计算9+8，只给结果。'),True))
  b=dict(payload('写一篇很长的Python并发编程教程，包含大量代码。',2048),stream=True)
  with requests.post(URL+'/v1/chat/completions',headers=H,json=b,stream=True,timeout=180) as r:
   r.raise_for_status();got=False
   for line in r.iter_lines(chunk_size=1):
    if not line.startswith(b'data: ') or line[6:]==b'[DONE]':continue
    x=json.loads(line[6:])
    if any(c.get('delta',{}).get('content') for c in x.get('choices',[])):got=True;break
   assert got
  save({'name':'cancel_'+str(cycle),'ok':True})
  r=chat('after_cancel_'+str(cycle),payload('计算9+8，只给结果。'),True)
  assert '17' in r['content'];save(r)
 assert requests.get(URL+'/health',timeout=5).json()['status']=='ok'
 save({'name':'FINAL','ok':True});print('LIFECYCLE_PASS',flush=True)


if __name__ == "__main__":
 main()
