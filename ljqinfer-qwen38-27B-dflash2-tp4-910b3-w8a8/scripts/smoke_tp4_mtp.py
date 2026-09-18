import argparse, torch, torch_npu
from model.config import EngineConfig
from model.model import Engine
from model.mtp import MTPContext
from model.spmd import SPMDRuntime

p=argparse.ArgumentParser(); p.add_argument('--rank',type=int,required=True)
p.add_argument('--device-base',type=int,default=4); a=p.parse_args()
dev=a.device_base+a.rank; root='/dev/shm/qtp4_mtp_root.bin'
rt=SPMDRuntime(a.rank,4,dev,root)
cfg=EngineConfig(max_cached_tokens=4096,max_sequences=2,prefill_chunk_size=1024,
                 mtp_backend='qwen_native')
e=Engine.load(a.rank,cfg,device=f'npu:{dev}',collective=rt,load_mtp=True)
ids=torch.tensor([1,2],device=e.device); pos=torch.tensor([0,0],device=e.device)
sids=torch.tensor([0,1],device=e.device)
h=torch.arange(2*5120,device=e.device,dtype=torch.float32).reshape(2,5120)
h=((h%97)-48).to(torch.bfloat16)/64
main_lengths=e.cache.lengths.clone(); main_conv=e.cache.hot_gdn_conv.clone()
ctx=MTPContext(ids,pos,h,sids)
d1=e.mtp.draft(ctx,1); torch.npu.synchronize(e.device)
log1=d1.logits.clone(); tok1=d1.token_ids.clone(); e.mtp.rollback([0,0])
d2=e.mtp.draft(ctx,1); torch.npu.synchronize(e.device)
rollback_diff=float((log1-d2.logits).float().abs().max().cpu())
e.mtp.commit([1,0])
len0=e.mtp._k[0].shape[0]; absent1=(1 not in e.mtp._k)
ctx2=MTPContext(torch.tensor([3,4],device=e.device),torch.tensor([1,0],device=e.device),
                h.flip(0),sids)
d3=e.mtp.draft(ctx2,1); torch.npu.synchronize(e.device)
e.mtp.rollback([0,0])
finite=bool(torch.isfinite(d3.logits).all().cpu())
main_unchanged=(bool(torch.equal(main_lengths,e.cache.lengths)) and
                bool(torch.equal(main_conv,e.cache.hot_gdn_conv)))
tg=rt.all_gather(d3.token_ids.to(torch.float32)); torch.npu.synchronize(e.device)
rank_diff=float((tg-tg[0:1]).abs().max().cpu())
print(f'rank={a.rank} finite={finite} rollback_diff={rollback_diff} '
      f'commit_len0={len0} rejected_absent1={absent1} main_unchanged={main_unchanged} '
      f'tokens={d3.token_ids.cpu().tolist()} rank_diff={rank_diff}',flush=True)
if not finite or rollback_diff!=0 or len0!=1 or not absent1 or not main_unchanged or rank_diff!=0:
    raise RuntimeError('native MTP transaction gate failed')
rt.destroy()
