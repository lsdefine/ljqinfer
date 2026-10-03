import os,sys,json,statistics,argparse
from pathlib import Path
import torch
from torch.utils.cpp_extension import load
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from ops.sparse_index_decode import extension
from ops.sparse_index_fast import extension as prefill_extension
os.environ['TORCH_CUDA_ARCH_LIST']='8.0';os.environ['MAX_JOBS']='2';torch.cuda.set_device(0)
parser=argparse.ArgumentParser();parser.add_argument('--baseline-cu',required=True);parser.add_argument('--output',required=True);parser.add_argument('--phase',choices=['prefill','decode'],default='decode');args=parser.parse_args()
if args.phase=='prefill':extension=prefill_extension
base=load(name='glm53_index_radix_port',sources=[args.baseline_cu],extra_cuda_cflags=['-O3'],verbose=False);cand=extension()
def timer(fn):
 for _ in range(3):fn()
 g=torch.cuda.CUDAGraph()
 with torch.cuda.graph(g):
  for _ in range(10):fn()
 for _ in range(3):g.replay()
 samples=[]
 for _ in range(7):
  a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True);a.record();g.replay();b.record();b.synchronize();samples.append(a.elapsed_time(b)/10)
 return statistics.median(samples)
torch.manual_seed(961);records=[]
for n,k,rows in [(63,32,3),(2053,2048,3),(65542,2048,1),(65544,2048,8),(77824,2048,32),(4099,128,33),(77824,2048,256)]:
 for kind in ['normal','ties','zeros','negative']:
  scores=torch.randn(rows,n,device='cuda')
  if kind=='ties':scores=(scores*3).round()
  if kind=='zeros':scores.zero_()
  if kind=='negative':scores=-scores.abs()
  pos=torch.full((rows,),n-1,device='cuda',dtype=torch.int64)
  if rows>1:pos[0]=-1;pos[1]=min(n-1,17)
  a=torch.empty(rows,k,device='cuda',dtype=torch.int32);b=torch.empty_like(a)
  base.select_out(scores,pos,a);cand.select_out(scores,pos,b);assert torch.equal(a,b),(n,kind)
  g=torch.cuda.CUDAGraph()
  with torch.cuda.graph(g):cand.select_out(scores,pos,b)
  scores.neg_();pos[-1]=n//2;g.replay();base.select_out(scores,pos,a);assert torch.equal(a,b),(n,kind,'graph')
  rec=dict(n=n,k=k,rows=rows,kind=kind,passed=True)
  if n>60000:
   pos.fill_(n-1);rec.update(base_ms=timer(lambda:base.select_out(scores,pos,a)),candidate_ms=timer(lambda:cand.select_out(scores,pos,b)))
  records.append(rec);print(json.dumps(rec),flush=True)
Path(args.output).write_text(json.dumps(records,indent=2));print('ALL_PASS',flush=True)
