import argparse, time
import torch, torch_npu
from model.config import EngineConfig
from model.decode_graph import DecodeGraphRunner
from model.model import Engine
from model.spmd import SPMDRuntime

p=argparse.ArgumentParser()
p.add_argument('--rank',type=int,required=True)
p.add_argument('--layers',type=int,default=1)
p.add_argument('--batch',type=int,default=1)
p.add_argument('--device-base',type=int,default=4)
p.add_argument('--query-width',type=int,default=8)
a=p.parse_args()
dev=a.device_base+a.rank
root=f'/dev/shm/qtp4_graph{a.layers}_b{a.batch}_q{a.query_width}_root.bin'
rt=SPMDRuntime(a.rank,4,dev,root)
cfg=EngineConfig(max_cached_tokens=2048*a.batch,
                 max_sequence_tokens=2048*a.batch,
                 max_sequences=a.batch,
                 prefill_chunk_size=8192,mtp_backend='disabled')
e=Engine.load(a.rank,cfg,device=f'npu:{dev}',collective=rt)
max_prefix=max(32,3*a.query_width)
r=DecodeGraphRunner(e,a.batch,max_prefix,a.layers,a.query_width)
q=a.query_width

def candidates(base):
    return [[base+row*q+i for i in range(q)] for row in range(a.batch)]

def observe():
    return (r.local_logits.clone(), r.gdn_conv_pending[:, :, -1].clone(),
            r.gdn_rec_pending[:, :, -1].clone(), r.k_pending.clone(),
            r.v_pending.clone())

def snap(tokens, positions):
    r.prepare(tokens,positions); r._forward(); torch.npu.synchronize(r.device)
    return observe()

def diff(xs,ys):
    return max((float((x-y).float().abs().max().cpu()) if x.numel() else 0.0)
               for x,y in zip(xs,ys))

def scalar_diff(x,y):
    return float((x-y).float().abs().max().cpu()) if x.numel() else 0.0

tokens1=candidates(1); tokens2=candidates(101); tokens3=candidates(201)
pos0=[0]*a.batch
base=snap(tokens1,pos0)
t0=time.perf_counter(); r.capture(warm=True); capture_s=time.perf_counter()-t0
first=observe()
first_diff=diff(base,first)
expected=snap(tokens2,pos0)
t=[]
for _ in range(5):
    r.prepare(tokens2,pos0)
    t0=time.perf_counter(); r.replay(); t.append(time.perf_counter()-t0)
    if len(t)<5: r.rollback()
actual=observe()
dynamic_diff=diff(expected,actual)
changed=scalar_diff(base[0],actual[0])

# Publish only the first 3/8 candidates.  Prove accepted state is published and
# rejected slots remain byte-for-byte unchanged.
accepted=min(3,q)
k_before=r.k_in.clone(); v_before=r.v_in.clone()
if r.gdn_conv_in.numel():
    conv_candidate=torch.cat((r.gdn_conv_in, r.gdn_conv_pending[:,:,:accepted]), dim=2)[:,:,-r.gdn_conv_in.shape[2]:].clone()
else:
    conv_candidate=None
rec_candidate=r.gdn_rec_pending[:,:,accepted-1].clone() if r.gdn_rec_in.numel() else None
k_candidate=r.k_pending[:,:,:accepted].clone() if r.k_in.numel() else None
v_candidate=r.v_pending[:,:,:accepted].clone() if r.v_in.numel() else None
r.commit([accepted]*a.batch)
commit_diff=0.0
if r.gdn_conv_in.numel():
    commit_diff=max(commit_diff,scalar_diff(r.gdn_conv_in,conv_candidate),
                    scalar_diff(r.gdn_rec_in,rec_candidate))
if r.k_in.numel():
    commit_diff=max(commit_diff,
        scalar_diff(r.k_in[:,:,:accepted],k_candidate),
        scalar_diff(r.v_in[:,:,:accepted],v_candidate),
        scalar_diff(r.k_in[:,:,accepted:],k_before[:,:,accepted:]),
        scalar_diff(r.v_in[:,:,accepted:],v_before[:,:,accepted:]))

# Dynamic absolute position after a partial commit must match eager exactly.
pos_next=[accepted]*a.batch
state_expected=snap(tokens3,pos_next)
r.prepare(tokens3,pos_next); r.replay()
state_actual=observe()
state_diff=diff(state_expected,state_actual)

# Rollback must not mutate any committed input state.
committed=(r.gdn_conv_in.clone(),r.gdn_rec_in.clone(),r.k_in.clone(),r.v_in.clone())
r.rollback()
rollback_diff=diff(committed,(r.gdn_conv_in,r.gdn_rec_in,r.k_in,r.v_in))

# Re-run once to leave valid output and check each rank chooses the same last-Q
# token after reconstructing the TP-sharded vocabulary.
r.prepare(tokens3,pos_next); r.replay()
last_local=r.local_logits[:,-1].contiguous()
full=rt.all_gather(last_local); torch.npu.synchronize(r.device)
full=full.reshape(4,a.batch,-1).permute(1,0,2).reshape(a.batch,-1)
tokens=full.float().argmax(-1).to(torch.float32)
tg=rt.all_gather(tokens); torch.npu.synchronize(r.device)
tg=tg.reshape(4,a.batch)
rank_token_diff=float((tg-tg[0:1]).abs().max().cpu())
row_state_diff=(float((state_actual[1][:,0]-state_actual[1][:,1]).float().abs().max().cpu())
                if a.batch>1 and state_actual[1].shape[0] else 1.0)
finite=all(bool(torch.isfinite(x).all().cpu()) for x in actual)
print(f'rank={a.rank} layers={a.layers} B={a.batch} Q={q} finite={finite} '
      f'first_diff={first_diff} dynamic_diff={dynamic_diff} state_diff={state_diff} '
      f'commit_diff={commit_diff} rollback_diff={rollback_diff} changed={changed} '
      f'row_state_diff={row_state_diff} tokens={[int(x) for x in tokens.cpu().tolist()]} '
      f'rank_token_diff={rank_token_diff} capture_s={capture_s:.4f} '
      f'replay_ms={[round(x*1000,3) for x in t]}',flush=True)
if (not finite or first_diff!=0 or dynamic_diff!=0 or state_diff!=0 or
        commit_diff!=0 or rollback_diff!=0 or changed==0 or row_state_diff==0 or
        rank_token_diff!=0):
    raise RuntimeError('B1Q8 decode graph correctness gate failed')
r.rollback()
rt.destroy()
