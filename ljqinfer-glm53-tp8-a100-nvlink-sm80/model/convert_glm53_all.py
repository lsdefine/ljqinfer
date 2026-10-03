"""Resumable TP8 G64 routed-INT4 cache; non-routed weights depend on source."""
import argparse,fcntl,hashlib,json,os,subprocess,sys,time
from pathlib import Path

def atomic(path,data):
    tmp=path.with_suffix(path.suffix+'.tmp')
    with tmp.open('w') as f:
        json.dump(data,f,indent=2);f.flush();os.fsync(f.fileno())
    tmp.replace(path)

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()

def worker(root,out,rank):
    from model.glm53_layer_weights import convert_rank
    manifest=json.loads((out/'manifest.json').read_text())
    started=time.time();completed=[]
    for layer in manifest['layers']:
        target=out/f'layer{layer}_g64_rank{rank}.safetensors'
        receipt=target.with_suffix('.json')
        if target.exists() and receipt.exists():
            old=json.loads(receipt.read_text())
            assert old['manifest_sha256']==sha(out/'manifest.json')
            assert old['bytes']==target.stat().st_size and old['sha256']==sha(target)
            completed.append(layer);continue
        t=time.time()
        atomic(out/f'rank{rank}.status.json',dict(state='converting',layer=layer,completed=completed,pid=os.getpid(),updated=time.time()))
        convert_rank(root,out,layer,rank,64,include_fp8_down=False,verify=True)
        atomic(receipt,dict(layer=layer,rank=rank,bytes=target.stat().st_size,sha256=sha(target),manifest_sha256=sha(out/'manifest.json'),seconds=time.time()-t,verified_roundtrip=True))
        completed.append(layer)
        atomic(out/f'rank{rank}.status.json',dict(state='running',completed=completed,last_seconds=time.time()-t,elapsed=time.time()-started,pid=os.getpid(),updated=time.time()))
        print(json.dumps(dict(rank=rank,layer=layer,seconds=time.time()-t,completed=len(completed))),flush=True)
    atomic(out/f'rank{rank}.status.json',dict(state='complete',completed=completed,elapsed=time.time()-started,pid=os.getpid(),updated=time.time()))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('source',type=Path);ap.add_argument('output',type=Path);ap.add_argument('--worker',type=int)
    args=ap.parse_args();root=args.source.resolve();out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    if args.worker is not None:return worker(root,out,args.worker)
    lock=(out/'conversion.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    cfg=json.loads((root/'config.json').read_text())
    assert (cfg['num_hidden_layers'],cfg['first_k_dense_replace'],cfg['n_routed_experts'],cfg['hidden_size'],cfg['moe_intermediate_size'])==(78,3,256,6144,2048)
    index=root/'model.safetensors.index.json';idx=json.loads(index.read_text())
    files=sorted(set(idx['weight_map'].values()))
    source_stat={n:dict(bytes=(root/n).stat().st_size,mtime_ns=(root/n).stat().st_mtime_ns) for n in files}
    repo=Path(__file__).resolve().parents[1]
    manifest=dict(format='glm53_tp8_routed_int4_cache_v1',source=str(root),source_index_sha256=sha(index),source_config_sha256=sha(root/'config.json'),source_files=source_stat,layers=list(range(3,78)),tp=8,group=64,method='weight-only MSE clipping search + four LS refinements',native_mtp_excluded=True,non_routed='load from original source; shared expert and router included in rank caches',include_fp8_down=False,code_sha256={n:sha(repo/n) for n in ['model/int4_quantize.py','model/glm53_layer_weights.py','model/convert_glm53_all.py']})
    mp=out/'manifest.json'
    if mp.exists():assert json.loads(mp.read_text())==manifest,'Source or converter changed; do not mix caches'
    else:atomic(mp,manifest)
    atomic(out/'conversion.status.json',dict(state='running',pid=os.getpid(),started=time.time(),expected_shards=600))
    children=[];logs=[]
    for rank in range(8):
        log=(out/f'rank{rank}.log').open('a');logs.append(log)
        env=os.environ.copy();env['OMP_NUM_THREADS']='2';env['PYTHONUNBUFFERED']='1'
        child=subprocess.Popen([sys.executable,'-m','model.convert_glm53_all',str(root),str(out),'--worker',str(rank)],cwd=repo,env=env,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
        children.append(child)
    codes=[c.wait() for c in children]
    for log in logs:log.close()
    good=all(c==0 for c in codes)
    atomic(out/'conversion.status.json',dict(state='complete' if good else 'failed',returncodes=codes,finished=time.time(),expected_shards=600))
    if not good:raise SystemExit(1)

if __name__=='__main__':main()
