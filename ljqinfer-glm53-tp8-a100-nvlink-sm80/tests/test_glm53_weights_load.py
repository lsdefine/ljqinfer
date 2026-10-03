"""Real TP8 cache/H2D audit; run after python -m model.wcache build.
python tests/test_glm53_weights_load.py --output /path/to/audit
This validates loading, not full-model inference.
"""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from model import weights as W
from model.wcache import Cache

def worker(rank,out):
    torch.set_num_threads(2);torch.cuda.set_device(rank)
    c=Cache();original=W.build_layer;og=W.build_globals
    def forbidden(*a,**kw):raise AssertionError('warm load attempted source conversion')
    W.build_layer=W.build_globals=forbidden
    start=time.perf_counter();w=W.load_tp8(rank=rank)
    torch.cuda.synchronize();elapsed=time.perf_counter()-start
    assert len(w.layers)==78 and w.rank==rank
    assert w.embed.shape==w.lm_head.shape==(19360,6144)
    assert (w.vocab_start,w.vocab_end)==(rank*19360,(rank+1)*19360)
    assert w.final_norm.shape==(6144,)
    # Independent source vocabulary rows: offset/transposition guard.
    source=og(W.SOURCE,rank)
    assert torch.equal(w.embed.cpu(),source['embed'])
    assert torch.equal(w.lm_head.cpu(),source['lm_head'])
    assert torch.equal(w.final_norm.cpu(),source['final_norm'])
    full_count=0
    for i,l in enumerate(w.layers):
        assert l.idx==i and l.ffn_norm.shape==(6144,)
        expected=c.cfg['indexer_types'][i]=='full'
        assert (l.indexer is not None)==expected
        full_count+=int(expected)
        actual={'ffn_norm':l.ffn_norm,**{'attn.'+k:v for k,v in l.attn.items()},
                **{'indexer.'+k:v for k,v in (l.indexer or {}).items()},
                **{('dense.' if i<3 else 'moe.')+k:v for k,v in vars(l.ffn).items() if torch.is_tensor(v)}}
        ref=c.tensors(rank,i,device='cpu')
        assert set(actual)==set(ref)
        for k,v in actual.items():
            assert v.shape==ref[k].shape and v.dtype==ref[k].dtype
            # Full transfer equality, one tensor at a time bounds host RAM.
            assert torch.equal(v.cpu(),ref[k]),(i,k)
        del ref,actual
    result=dict(rank=rank,layers=78,full_indexers=full_count,hot_load_seconds=elapsed,
                allocated_bytes=torch.cuda.memory_allocated(),all_tensor_h2d_exact=True,
                source_globals_exact=True,cold_builder_calls=0,status='PASS')
    (out/f'rank{rank}.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result),flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--rank',type=int)
    a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    if a.rank is not None:worker(a.rank,out);return
    logs=[];children=[]
    for rank in range(8):
        log=(out/f'rank{rank}.log').open('w');logs.append(log)
        children.append(subprocess.Popen([sys.executable,str(Path(__file__).resolve()),
                         '--output',str(out),'--rank',str(rank)],stdout=log,stderr=subprocess.STDOUT))
    codes=[c.wait() for c in children]
    for log in logs:log.close()
    (out/'status.json').write_text(json.dumps(dict(returncodes=codes,status='PASS' if not any(codes) else 'FAIL')))
    if any(codes):raise SystemExit(1)

if __name__=='__main__':main()
