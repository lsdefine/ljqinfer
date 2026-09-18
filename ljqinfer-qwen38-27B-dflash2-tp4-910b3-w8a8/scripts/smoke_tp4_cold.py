import argparse, torch, torch_npu
from model.config import EngineConfig, CONFIG
from model.model import Engine
from model.cold import store_prefix, restore_prefix
from model.spmd import SPMDRuntime
from strategy.cold_kv_cache import PrefixColdCache

p=argparse.ArgumentParser(); p.add_argument('--rank',type=int,required=True)
p.add_argument('--device-base',type=int,default=4); a=p.parse_args()
dev=a.device_base+a.rank; rt=SPMDRuntime(a.rank,4,dev,'/dev/shm/qtp4_cold_root.bin')
cfg=EngineConfig(max_cached_tokens=4096,max_sequences=2,prefill_chunk_size=1024,
                 mtp_backend='disabled')
e=Engine.load(a.rank,cfg,device=f'npu:{dev}',collective=rt); c=e.cache
ids=list(range(2048)); sid0,sid1=0,1
# Deterministic physical state spanning two independently restorable 1024 blocks.
base=torch.arange(1024*CONFIG.local_kv_heads*CONFIG.head_dim,
                  device=e.device,dtype=torch.float32).reshape(
                  1024,CONFIG.local_kv_heads,CONFIG.head_dim)
for slot in range(c.k.shape[0]):
    for block in range(2):
        k=((base%251)+(slot+1)*0.125+block).to(torch.bfloat16)
        v=((base%127)-(slot+1)*0.25-block).to(torch.bfloat16)
        c.write_kv(slot,sid0,block*1024,k,v)
for x,mod,scale in ((c.hot_gdn_conv[:,sid0],113,0.01),
                    (c.hot_gdn_recurrent[:,sid0],97,0.001)):
    q=torch.arange(x.numel(),device=e.device,dtype=torch.float32).reshape(x.shape)
    x.copy_(((q%mod)*scale).to(x.dtype))
c.lengths[sid0]=1024; c.checkpoint_gdn(sid0,1024)
c.hot_gdn_conv[:,sid0].add_(0.5); c.hot_gdn_recurrent[:,sid0].add_(0.25)
c.lengths[sid0]=2048; c.checkpoint_gdn(sid0,2048)
source=(c.export_cold_block(sid0,0,1024),
        c.export_cold_block(sid0,1024,2048))
cold=PrefixColdCache(1024); stored=store_prefix(e,cold,ids,sid0)
match=cold.begin(ids+[999]); hit=match.token_count
# Baseline suffix mutates HBM; the cold record must remain independently owned.
tok=torch.tensor([777],dtype=torch.int64,device=e.device)
h0=e.forward_tokens(tok,torch.tensor([sid0],device=e.device)); l0=e.local_logits(h0)
t0=rt.all_gather(l0); torch.npu.synchronize(e.device)
e.release_sequence(sid0)
restored=restore_prefix(e,cold,ids+[999],sid1)
roundtrip=(c.export_cold_block(sid1,0,1024),
           c.export_cold_block(sid1,1024,2048))
def md(a,b):
    if torch.is_tensor(a): return float((a-b).float().abs().max())
    if isinstance(a,(tuple,list)): return max((md(x,y) for x,y in zip(a,b)),default=0.0)
    return float(abs(a-b))
state_diff=md(source,roundtrip)
h1=e.forward_tokens(tok,torch.tensor([sid1],device=e.device)); l1=e.local_logits(h1)
t1=rt.all_gather(l1); torch.npu.synchronize(e.device)
suffix_diff=float((t0-t1).float().abs().max().cpu())
rank_token=rt.all_gather(t1.reshape(4,1,-1).permute(1,0,2).reshape(1,-1)
                         .float().argmax(-1).to(torch.float32)); rt.synchronize()
rank_diff=float((rank_token-rank_token[0:1]).abs().max().cpu())
finite=bool(torch.isfinite(t1).all().cpu())
print(f'rank={a.rank} stored={stored} entries={cold.entry_count} hit={hit} '
      f'restored={restored} state_diff={state_diff} suffix_diff={suffix_diff} '
      f'finite={finite} rank_diff={rank_diff}',flush=True)
if stored!=2 or cold.entry_count!=2 or hit!=2048 or restored!=2048 or \
   state_diff!=0 or suffix_diff!=0 or not finite or rank_diff!=0:
    raise RuntimeError('1024 cold DRAM gate failed')
rt.destroy()
