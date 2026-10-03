import os,sys,json,time,statistics,hashlib
from pathlib import Path
from datetime import timedelta
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
import torch.distributed as dist
from tokenizers import Tokenizer
from jinja2 import Environment
from model.glm53_generate import Generator
R=Path(os.environ.get('DECODE_AUDIT_DIR','/mnt/data2/kw/glm53_int4_tp8/service_audit/decode50/integrated'));R.mkdir(exist_ok=True,parents=True)
rank=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(rank)
dist.init_process_group('nccl',timeout=timedelta(minutes=20))
tok=Tokenizer.from_file('/mnt/data2/kw/GLM-5.3-UNCENSORED-FP8/tokenizer.json')
template=Environment().from_string((Path(__file__).resolve().parents[1]/'server/chat_template.jinja').read_text())
cases=[
('python_lru','编程-Python','实现一个线程安全的Python LRU缓存类，支持get、put、容量限制，要求O(1)操作。给出完整实现、pytest测试和关键设计说明。'),
('typescript_async','编程-TypeScript','Write a production-quality TypeScript mapLimit(items, limit, fn) that preserves order and limits concurrency. Handle rejection and invalid limits. Include implementation and tests.'),
('cpp_debug','编程-C++排错','下面C++代码有什么问题？给出修复代码，解释迭代器失效以及复杂度，并提供边界测试：\nstd::vector<int> v={1,2,3,4,5,6};\nfor(auto it=v.begin();it!=v.end();++it){if(*it%2==0)v.erase(it);}'),
('sql_window','编程-SQL','PostgreSQL有orders(id,user_id,created_at,amount,status)。写SQL统计最近30天每天收入、7日移动平均、每日收入前三用户，要求补齐没有订单的日期并解释索引方案。'),
('explain_zh','中文解释','请向初学者详细解释数据库事务的四个ACID属性，分别给出网上购物中的例子，说明MVCC和锁的关系以及常见隔离级别。'),
('writing_zh','中文写作','写一篇约800字的科幻短篇：深海观测站收到一封来自十年后自己的信。需要有环境描写、人物对话和有因果铺垫的结尾，不要先列提纲。'),
('reasoning_math','数学推理','有12枚外观相同的硬币，其中一枚假币重量不同但不知道偏重还是偏轻。用天平最多称3次找出假币并判断轻重。请给出完整且可执行的决策方案并解释。'),
('english_explain','英文解释','Explain how DNS resolution and HTTPS connection establishment work when opening a website, including recursive resolvers, caching, TLS 1.3, certificate checks and HTTP/2. Use a detailed concrete example.')]
tasks=[(a,b,c,False) for a,b,c in cases]+[(a+'_think',b,c,True) for a,b,c in [cases[0],cases[2],cases[4],cases[6]]]
def encode(prompt,thinking):
 text=template.render(messages=[{'role':'user','content':prompt}],tools=[],add_generation_prompt=True,enable_thinking=thinking,reasoning_effort='high',clear_thinking=True)
 return tok.encode(text,add_special_tokens=False).ids
if rank==0:(R/'cases.json').write_text(json.dumps(tasks,ensure_ascii=False,indent=2))
from model.glm53_dflash import DFlash2
capture_original=DFlash2.capture
def capture_checked(self):
 before_k=self.pool.k.clone();before_v=self.pool.v.clone();length=self.pool.lengths[0]
 capture_original(self)
 assert torch.equal(before_k,self.pool.k) and torch.equal(before_v,self.pool.v)
 assert self.pool.lengths[0]==length
 print('CAPTURE_KV_UNCHANGED',rank,flush=True)
DFlash2.capture=capture_checked
g=Generator.load(capacity=8192,prefill_chunk_tokens=12288)
for _ in range(2):
 g.reset();g.generate(encode('Explain what a hash table is.',False),max_new_tokens=32,use_graph=True)
assert g.engine.graph is not None and g.draft.graph is not None
if rank==0:print('WARMED',flush=True)
records=[]
for ident,category,prompt,thinking in tasks:
 g.reset();ids=encode(prompt,thinking);stamps=[]
 def emitted(tokens):stamps.append((time.perf_counter(),len(tokens)))
 result=g.generate(ids,max_new_tokens=384,use_graph=True,on_tokens=emitted,profile=False)
 torch.cuda.synchronize()
 elapsed=stamps[-1][0]-stamps[0][0]
 elapsed_t=torch.tensor(elapsed,device='cuda',dtype=torch.float64);dist.all_reduce(elapsed_t,op=dist.ReduceOp.MAX);elapsed=elapsed_t.item()
 accepted=[x['accepted_drafts'] for x in result.steps];n=len(accepted)
 text=tok.decode(result.token_ids,skip_special_tokens=False)
 rec=dict(id=ident,category=category,thinking=thinking,input_tokens=len(ids),output_tokens=len(result.token_ids),steps=n,accepted_sum=sum(accepted),accepted_mean=sum(accepted)/n if n else 0,advance_theoretical=1+sum(accepted)/n if n else 0,advance_actual=(len(result.token_ids)-1)/n if n else 0,acceptance_rate=sum(accepted)/(7*n) if n else 0,histogram=[accepted.count(k) for k in range(8)],prefix_survival=[sum(a>=k for a in accepted)/n for k in range(1,8)] if n else [],decode_seconds=elapsed,decode_tps=(len(result.token_ids)-1)/elapsed if elapsed else 0,round_ms=elapsed*1000/n if n else 0,finish_reason=result.finish_reason,output_sha256=hashlib.sha256(bytes(str(result.token_ids),'utf8')).hexdigest(),graph=True,profile=False)
 records.append(rec)
 with (R/f'rank{rank}.jsonl').open('a') as f:f.write(json.dumps(rec,ensure_ascii=False)+chr(10))
 if rank==0:
  (R/f'{ident}.json').write_text(json.dumps(dict(summary=rec,prompt=prompt,output=text,token_ids=result.token_ids,steps=result.steps),ensure_ascii=False,indent=2))
  print(json.dumps(rec,ensure_ascii=False),flush=True)
for ident,category,prompt in [cases[0],cases[4]]:
 g.reset();res=g.generate(encode(prompt,False),max_new_tokens=128,use_graph=True,profile=True)
 detail={k:statistics.mean(x[k] for x in res.steps) for k in ['draft_ms','verify_ms','commit_ms']}
 if rank==0:
  (R/f'{ident}.profile.json').write_text(json.dumps(dict(mean_ms=detail,steps=res.steps),indent=2));print('PROFILE',ident,detail,flush=True)
# Repeated prompt and exact continuation exercise published prefix restoration.
cache_ids=encode(cases[0][2],False)
g.reset();first=g.generate(cache_ids,max_new_tokens=64,use_graph=True)
second=g.generate(cache_ids,max_new_tokens=64,use_graph=True)
assert g.prefill_stats['cache_hit_tokens']>0
cache_stats=g.prefill_stats.copy()
g.reset();third=g.generate(cache_ids,max_new_tokens=64,use_graph=True)
assert first.token_ids==third.token_ids, 'same cold path must reproduce'
cache_result=dict(cold=first.token_ids,hot=second.token_ids,reset=third.token_ids,
                  cold_hot_equal=first.token_ids==second.token_ids,cache=cache_stats)
# Cold/hot have different prefill shapes. Compare each to the SAME baseline path.
if os.environ.get('DECODE_CACHE_BASELINE'):
 reference=json.loads(Path(os.environ['DECODE_CACHE_BASELINE']).read_text())
 for row in reference:assert cache_result[row['run']]==row['tokens'], row['run']
(R/f'cache.rank{rank}.json').write_text(json.dumps(cache_result,indent=2))
print('CACHE_REPLAY_CHECKED',rank,cache_stats,flush=True)
(R/f'rank{rank}.done.json').write_text(json.dumps(dict(DONE=True,cases=len(records),records=records),ensure_ascii=False,indent=2))
g.close();dist.barrier();dist.destroy_process_group()
