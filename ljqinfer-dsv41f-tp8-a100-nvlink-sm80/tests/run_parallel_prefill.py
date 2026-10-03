"""Run with torchrun --standalone --nproc-per-node=8; full-size random weights."""
import argparse
from datetime import timedelta
import json
import os
from pathlib import Path
import torch
import torch.distributed as dist
import torch.nn.functional as F
from released_random import build, released_hasher, RandomHostTable
from parallel_random import RankRandom
from model.prefill_build import build_prefill, PrefillParallel
from model.model_api import ModelExecution
from model.past import SlotPool


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',required=True)
    parser.add_argument('--length',type=int,default=3)
    parser.add_argument('--trace',action='store_true')
    args=parser.parse_args()
    rank=int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(rank)
    dist.init_process_group('nccl',timeout=timedelta(minutes=15))
    parallel=PrefillParallel()
    assert dist.get_world_size()==8
    device=f'cuda:{rank}'
    weights=RankRandom(device,rank)
    w=weights.full;c=w.c
    def embed(ids):
        return torch.stack([w.random('embed.'+str(int(i)),(c['dim'],)) for i in ids]).bfloat16()
    def head(x):
        # Same deterministic global matrix as the single-rank reference, sliced
        # by vocabulary rank. Only storage generation differs, not projection.
        n=c['vocab_size']//8; lo=rank*n; hi=lo+n
        pieces=[]
        for i in range(lo//2048*2048,hi,2048):
            matrix=w.random('head.'+str(i),(min(2048,c['vocab_size']-i),c['dim'])).bfloat16()
            pieces.append(F.linear(x.float(),matrix[max(0,lo-i):min(len(matrix),hi-i)].float()))
        return parallel.logits(torch.cat(pieces,-1))
    tables={i:RandomHostTable(i,c['engram_num_embeddings'][j]) for j,i in enumerate(c['engram_layer_ids'])}
    model=build_prefill(c,weights,released_hasher(),tables,device=device,length=256,
                        parallel=parallel,embed=embed,head=head)
    if args.trace and rank==0:
        from prefill_trace import attach, compare
        trace=attach(model)
    past=SlotPool(1,256,page_tokens=16,device=device).configure_default()
    slot=past.alloc(); execution=ModelExecution(model,past)
    tokens=tuple((i*71+7)%c['vocab_size'] for i in range(args.length))
    execution.prefill_chunk(slot,tokens)
    output=execution.finish_prefill(slot)
    assert output.logits.shape==(1,c['vocab_size']) and torch.isfinite(output.logits).all()
    copies=[torch.empty_like(output.logits) for _ in range(8)]
    dist.all_gather(copies,output.logits)
    assert all(torch.equal(output.logits,x) for x in copies)
    cold=[past.export_cold(slot,p,p+1) for p in range(args.length)]
    past.release(slot); slot=past.alloc();past.import_cold(slot,cold)
    execution.replay_prefix(slot,tokens)
    replay=execution.finish_prefill(slot)
    assert torch.isfinite(replay.logits).all()
    if args.length<=128:
        torch.testing.assert_close(replay.logits,output.logits,rtol=0,atol=0)
    execution.prefill_chunk(slot,(190,),history_tokens=tokens[-3:])
    assert torch.isfinite(execution.finish_prefill(slot).logits).all()
    assert past.pos[slot]==args.length+1
    past.release(slot)
    result={'rank':rank,'world':8,'length':args.length,'complete':True,
            'owned_experts_used':len(weights.experts),'dtypes':sorted(w.dtype_hits)}
    assert weights.experts and all(rank*48<=e<(rank+1)*48 for e in weights.experts)
    if rank==0:
        reference,_=build(device=device,mixed=True,length=256)
        if args.trace: ref_trace=attach(reference)
        pool=SlotPool(1,256,page_tokens=16,device=device).configure_default()
        sl=pool.alloc();ex=ModelExecution(reference,pool)
        ex.prefill_chunk(sl,tokens); expected=ex.finish_prefill(sl).logits
        err=(output.logits-expected).float()
        result.update(max_abs=err.abs().max().item(),rmse=err.square().mean().sqrt().item(),
                      cosine=F.cosine_similarity(output.logits,expected).item(),
                      argmax_equal=bool(output.logits.argmax()==expected.argmax()))
        if args.trace:
            trace_path=Path(args.output); trace_path.mkdir(parents=True,exist_ok=True)
            (trace_path/'trace.json').write_text(json.dumps(compare(trace,ref_trace),indent=2))
        assert result['cosine']>0.99, result
        pool.release(sl)
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    (out/f'rank{rank}.json').write_text(json.dumps(result,indent=2))
    dist.barrier()
    print(json.dumps(result),flush=True)
    dist.destroy_process_group()


if __name__=='__main__': main()
