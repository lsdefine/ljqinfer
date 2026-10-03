import os,sys,json,time,statistics,hashlib
from pathlib import Path
from datetime import timedelta
sys.path.insert(0,os.environ.get('PROBE_REPO',str(Path(__file__).resolve().parents[1])))
import torch
import torch.distributed as dist
from tokenizers import Tokenizer
from jinja2 import Environment
from model.glm53_generate import Generator
R=Path('/mnt/data2/kw/glm53_int4_tp8/service_audit/decode50/integrated');R.mkdir(exist_ok=True,parents=True)
rank=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(rank)
dist.init_process_group('nccl',timeout=timedelta(minutes=20))
tok=Tokenizer.from_file('/mnt/data2/kw/GLM-5.3-UNCENSORED-FP8/tokenizer.json')
template=Environment().from_string(Path('/mnt/data/kw/ljqinfer_glm53_tp8/server/chat_template.jinja').read_text())
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

g=Generator.load(capacity=8192,prefill_chunk_tokens=12288)
ids=encode(cases[0][2],False)
rows=[]
for run in ['cold','hot','reset']:
 if run!='hot':g.reset()
 result=g.generate(ids,max_new_tokens=64,use_graph=True)
 rows.append(dict(run=run,tokens=result.token_ids,steps=result.steps,cache=g.prefill_stats.copy()))
if rank==0:
 (Path(os.environ.get('DECODE_AUDIT_DIR','/mnt/data2/kw/glm53_int4_tp8/service_audit/decode50'))/(os.environ['PROBE_VARIANT']+'.cache.json')).write_text(json.dumps(rows,indent=2))
 print('RESULT',[(x['run'],x['cache'],x['tokens'][:8]) for x in rows],flush=True)
g.close();dist.barrier();dist.destroy_process_group()
