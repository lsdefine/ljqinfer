# -*- coding: utf-8 -*-
"""A6 micro: synthetic fp4 banks, time moe_rank_routed_fp4 by T and ids pattern.
python tc_moe_micro.py > /tmp/moe_micro.log 2>&1"""
import torch, time, sys, os
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # repo root on sys.path (this file lives in bench/)
import ops
moe_rank_routed_fp4=ops._mod().moe_rank_routed_fp4
E,K,N=256,4096,int(os.environ.get('NFF','512'))
dev='cuda:0'
REF=os.environ.get('REF','cmp'); REFP='/tmp/moe_ref.pt'
refd=torch.load(REFP) if (REF=='cmp' and os.path.exists(REFP)) else {}
g=torch.Generator(device=dev).manual_seed(0)
def bank(rows,cols):
    w=torch.randint(0,256,(E,rows,cols//2),dtype=torch.uint8,device=dev,generator=g)
    s=torch.randint(120,130,(E,rows,cols//32),dtype=torch.uint8,device=dev,generator=g)
    return w,s
w1w,w1s=bank(N,K); w3w,w3s=bank(N,K); w2w,w2s=bank(K,N)
print('bank MB/layer', (w1w.numel()*2+w2w.numel())/1e6)
def run(T,mode,iters=50):
    x=torch.randn(T,K,device=dev,generator=g).bfloat16()
    if mode=='rand': ids=torch.stack([torch.randperm(E,device=dev,generator=g)[:6] for _ in range(T)])
    elif mode=='same': ids=torch.arange(6,device=dev).repeat(T,1)
    elif mode=='disjoint': ids=torch.arange(T*6,device=dev).reshape(T,6)%E
    wts=torch.rand(T,6,device=dev,generator=g)
    for _ in range(5): y=moe_rank_routed_fp4(x,ids,wts,w1w,w1s,w3w,w3s,w2w,w2s)
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as p:
        for _ in range(iters): y=moe_rank_routed_fp4(x,ids,wts,w1w,w1s,w3w,w3s,w2w,w2s)
        torch.cuda.synchronize()
    agg={}
    for ev in p.key_averages():
        if ev.self_device_time_total>0: agg[ev.key[:40]]=ev.self_device_time_total/iters
    uniq=len(set(ids.flatten().tolist()))
    key=f'{T}_{mode}'
    if REF=='save': refd[key]=y.cpu()
    elif key in refd:
        eq=torch.equal(refd[key],y.cpu()); print('  BITEXACT' if eq else f'  ***MISMATCH maxdiff={(refd[key].float()-y.cpu().float()).abs().max().item()}')
    print(f"T={T:2d} {mode:8s} uniq={uniq:3d} "+" | ".join(f"{k.split('(')[0][-28:]}={v:7.1f}us" for k,v in sorted(agg.items(),key=lambda kv:-kv[1])[:4]),flush=True)
for T in [int(v) for v in os.environ.get('TS','8,32').split(',')]:
    for mode in os.environ.get('MODES','rand,same,disjoint').split(','):
        run(T,mode)
if REF=='save': torch.save(refd,REFP); print('saved',REFP)
