"""Synthetic operand microbench. No model projections, TP or quality claim."""
import sys,json,math,statistics,argparse
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from ops.sparse_ops import *
p=argparse.ArgumentParser();p.add_argument('--prefix',type=int,default=24576);p.add_argument('--tokens',type=int,default=8);p.add_argument('--output',required=True);p.add_argument('--tune',action='store_true');a=p.parse_args()
torch.manual_seed(481);torch.cuda.set_device(0)
t=a.tokens;n=a.prefix+t;page=64;np=math.ceil(n/page);tile=min(t,256);k=min(2048,n)
table=torch.randperm(np,device='cuda',dtype=torch.int64)
pool=torch.randn(np,page,576,device='cuda',dtype=torch.float16)*.3
kp=torch.randn(np,page,128,device='cuda',dtype=torch.float16)
qi=torch.randn(t,32,128,device='cuda',dtype=torch.float16)
w=torch.randn(t,32,device='cuda')/math.sqrt(32)
q=torch.randn(t,8,576,device='cuda',dtype=torch.float16)*.3
pos=torch.arange(a.prefix,n,device='cuda',dtype=torch.int64);ctx=torch.tensor([n],device='cuda')
sc=torch.empty(tile,n,device='cuda');val=torch.empty(tile,k,device='cuda');i64=torch.empty(tile,k,device='cuda',dtype=torch.int64)
idx=torch.empty(t,k,device='cuda',dtype=torch.int32);out=torch.empty(t,8,512,device='cuda',dtype=torch.float16)

def score(bq=4,bn=128):
    index_scores(qi[:tile],kp,w[:tile],table,pos[:tile],ctx,sc,block_q=bq,block_n=bn)
def top():topk_indices(sc,val,i64,idx[:tile])
def pipeline(bq=4,bn=128):
    for start in range(0,t,tile):
        end=min(start+tile,t);m=end-start
        index_scores(qi[start:end],kp,w[start:end],table,pos[start:end],ctx,sc[:m],block_q=bq,block_n=bn)
        topk_indices(sc[:m],val[:m],i64[:m],idx[start:end])

def bench(fn,reps=5,loops=5):
    for _ in range(3):fn()
    torch.cuda.synchronize();g=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(loops):fn()
    for _ in range(2):g.replay()
    torch.cuda.synchronize();times=[]
    for _ in range(reps):
        s=torch.cuda.Event(enable_timing=True);e=torch.cuda.Event(enable_timing=True)
        s.record();g.replay();e.record();e.synchronize();times.append(s.elapsed_time(e)/loops)
    g.reset();return statistics.median(times)


from ops.sparse_mla_cuda import forward
pipeline();idx64=idx.long();ws=make_mla_workspace(q,splits=1)
sparse_mla(q,pool,table,idx,pos,ctx,out,ws,splits=1,block_n=64);ref=out.clone()
forward(q,pool,table,idx64,pos,ctx,out);e=float((out.float()-ref.float()).norm()/ref.float().norm());print('REL',e,flush=True);assert e<.002
print('TRITON_MS',bench(lambda:sparse_mla(q,pool,table,idx,pos,ctx,out,ws,splits=1,block_n=64)),flush=True)
print('CUDA_MS',bench(lambda:forward(q,pool,table,idx64,pos,ctx,out)),flush=True)
print('ALL_PASS',flush=True)
