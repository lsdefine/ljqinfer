"""torchrun --standalone --nproc-per-node=8 tests/check_sampling_tp8.py"""
import os,sys
from pathlib import Path
from datetime import timedelta
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
import torch.distributed as dist
from model.sampling import pick_tokens
rank=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(rank)
dist.init_process_group('nccl',timeout=timedelta(seconds=90))
for temps in [[0],[1],[0,1],[1,0,1,0],[1,1,1,1]]:
    torch.manual_seed(100+rank)
    x=torch.randn(len(temps),8,154880,device='cuda')
    got=pick_tokens(x,temps)
    all_ids=[torch.empty_like(got) for _ in range(8)]
    dist.all_gather(all_ids,got)
    assert all(torch.equal(got,v) for v in all_ids)
    if rank==0:
        for i,t in enumerate(temps):
            if t==0: assert torch.equal(got[i],x[i].argmax(-1))
    if rank==0:print('TP8_PASS',temps,flush=True)
dist.destroy_process_group()
