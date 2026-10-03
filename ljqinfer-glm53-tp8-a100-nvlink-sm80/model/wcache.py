"""GLM53 final-layout tmpfs cache. First load builds; later loads only transfer.

python -m model.wcache build --wait 3600
No daemon is needed: tmpfs survives process exit, but not reboot. A fresh load
reconstructs missing/invalid entries. No GGUF/CP/native-MTP compatibility path.
"""
from pathlib import Path
import argparse,fcntl,hashlib,json,os,shutil,subprocess,sys,time
import torch
from safetensors import safe_open,SafetensorError
from safetensors.torch import save_file,load_file
from model import weights as W

BASE='/dev/shm/ljqinfer_glm53'
SCHEMA='glm53-tp8-g64-final-v1'


def _json(path,data):
    tmp=path.with_suffix(path.suffix+'.tmp')
    with tmp.open('w') as f:
        json.dump(data,f,sort_keys=True,indent=2);f.flush();os.fsync(f.fileno())
    tmp.replace(path)


def _sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8<<20),b''):h.update(block)
    return h.hexdigest()


def _stat(path):
    s=path.stat();return [str(path.resolve()),s.st_size,s.st_mtime_ns]


class Cache:
    def __init__(self,source=W.SOURCE,int4=W.INT4,cache_dir=None,wait=0):
        self.source=Path(source).resolve();self.int4=Path(int4).resolve()
        self.cfg=W.check_config(source);self.wait=wait
        index=self.source/'model.safetensors.index.json'
        files=sorted(set(json.loads(index.read_text())['weight_map'].values()))
        manifest=self.int4/'manifest.json'
        if not manifest.exists():raise FileNotFoundError('INT4 conversion manifest is required: '+str(manifest))
        identity=dict(schema=SCHEMA,source=str(self.source),int4=str(self.int4),
                      config=_sha(self.source/'config.json'),index=_sha(index),
                      shards=[_stat(self.source/f) for f in files],
                      conversion=_sha(manifest),
                      code={n:_sha(Path(__file__).parent/n) for n in ['weights.py','glm53_layer_weights.py']})
        self.fingerprint=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
        self.root=Path(cache_dir or BASE)/self.fingerprint
        self.root.mkdir(parents=True,exist_ok=True)
        with (self.root/'manifest.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            if not (self.root/'manifest.json').exists():_json(self.root/'manifest.json',identity)

    def _dependency(self,rank,layer):
        if layer is None or layer<3:return self.fingerprint
        p=self.int4/f'layer{layer}_g64_rank{rank}.safetensors';receipt=p.with_suffix('.json')
        deadline=time.monotonic()+self.wait
        while not receipt.exists():
            status=self.int4/'conversion.status.json'
            if status.exists() and json.loads(status.read_text()).get('state')=='failed':
                raise RuntimeError('INT4 conversion failed; inspect rank logs')
            if time.monotonic()>=deadline:raise FileNotFoundError('INT4 shard not ready: '+str(receipt))
            time.sleep(2)
        r=json.loads(receipt.read_text())
        if not p.exists() or p.stat().st_size!=r['bytes'] or not r['verified_roundtrip']:
            raise ValueError('invalid INT4 receipt: '+str(receipt))
        return hashlib.sha256(json.dumps([self.fingerprint,_stat(p),r],sort_keys=True).encode()).hexdigest()

    def ensure(self,rank,layer=None,build_device=None):
        if not 0<=rank<8 or (layer is not None and not 0<=layer<78):raise ValueError((rank,layer))
        name=f'rank{rank}_'+('globals' if layer is None else f'layer{layer:02d}')
        path=self.root/(name+'.safetensors');receipt=path.with_suffix('.json')
        with (self.root/(name+'.lock')).open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            dependency=self._dependency(rank,layer)
            if path.exists() and receipt.exists():
                try:
                    r=json.loads(receipt.read_text())
                    if r['dependency']==dependency and r['stat']==_stat(path):
                        with safe_open(str(path),framework='pt') as f:
                            if f.metadata().get('dependency')==dependency and sorted(f.keys())==r['keys']:
                                return path
                except (ValueError,KeyError,OSError,SafetensorError):pass
            t0=time.perf_counter()
            # Reserve ample per-builder slack before materializing. Final model fits
            # in 800 GiB tmpfs; avoid accidental giant allocation on a full host.
            avail=int(next(l.split()[1] for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:')))*1024
            if shutil.disk_usage(self.root).free<2*2**30 or avail<8*2**30:
                raise RuntimeError('insufficient tmpfs/RAM headroom to build cache shard')
            if layer is not None and layer>=3:
                source=self.int4/f'layer{layer}_g64_rank{rank}.safetensors'
                if _sha(source)!=json.loads(source.with_suffix('.json').read_text())['sha256']:
                    raise ValueError('INT4 checksum mismatch: '+str(source))
            data=W.build_globals(self.source,rank) if layer is None else W.build_layer(
                self.source,self.int4,layer,rank,build_device or f'cuda:{rank}')
            for k,v in data.items():
                if v.device.type!='cpu' or not v.is_contiguous():raise ValueError(k)
                if v.is_floating_point() and not torch.isfinite(v).all():raise ValueError('nonfinite '+k)
            tmp=path.with_suffix('.tmp')
            try:
                save_file(data,str(tmp),metadata={'schema':SCHEMA,'dependency':dependency})
                with safe_open(str(tmp),framework='pt') as f:
                    if sorted(f.keys())!=sorted(data):raise ValueError('cache key mismatch')
                    for k,v in data.items():
                        if not torch.equal(f.get_tensor(k),v):raise ValueError('cache roundtrip: '+k)
                with tmp.open('rb') as f:os.fsync(f.fileno())
                tmp.replace(path)
                _json(receipt,dict(dependency=dependency,stat=_stat(path),keys=sorted(data),
                                   bytes=path.stat().st_size,build_seconds=time.perf_counter()-t0))
            finally:
                tmp.unlink(missing_ok=True)
            print(json.dumps(dict(cache='built',rank=rank,layer=layer,bytes=path.stat().st_size,
                                  seconds=time.perf_counter()-t0)),flush=True)
            return path

    def tensors(self,rank,layer=None,device=None):
        device=torch.device(device or f'cuda:{rank}')
        path=self.ensure(rank,layer,build_device=device)
        # mmap-backed CPU tensors; bounded pageable H2D, never pin a full model.
        return load_file(str(path),device=str(device))


def build(*,source=W.SOURCE,int4=W.INT4,rank,cache_dir=None,wait=0):
    torch.set_num_threads(2)
    c=Cache(source,int4,cache_dir,wait)
    start=time.perf_counter();paths=[c.ensure(rank)]
    for layer in range(78):paths.append(c.ensure(rank,layer))
    result=dict(state='complete',rank=rank,files=len(paths),bytes=sum(p.stat().st_size for p in paths),seconds=time.perf_counter()-start)
    _json(c.root/f'rank{rank}.complete.json',result)
    print(json.dumps(result),flush=True)
    return result


def load(*,source=W.SOURCE,int4=W.INT4,rank,device=None,cache_dir=None):
    torch.set_num_threads(2)
    c=Cache(source,int4,cache_dir);device=device or f'cuda:{rank}'
    g=c.tensors(rank,device=device)
    layers=[W.unpack_layer(c.tensors(rank,i,device),i) for i in range(78)]
    return W.Weights(rank,g['embed'],g['final_norm'],g['lm_head'],layers,rank*19360,(rank+1)*19360)


def main():
    a=argparse.ArgumentParser();a.add_argument('command',choices=['build'])
    a.add_argument('--source',default=W.SOURCE);a.add_argument('--int4',default=W.INT4)
    a.add_argument('--cache-dir',default=BASE);a.add_argument('--rank',type=int);a.add_argument('--wait',type=float,default=0)
    args=a.parse_args()
    if args.rank is not None:
        build(source=args.source,int4=args.int4,rank=args.rank,cache_dir=args.cache_dir,wait=args.wait);return
    base=Path(args.cache_dir);base.mkdir(parents=True,exist_ok=True)
    with (base/'build.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        logs=[];children=[]
        for rank in range(8):
            log=(base/f'build.rank{rank}.log').open('a');logs.append(log)
            children.append(subprocess.Popen([sys.executable,'-m','model.wcache','build','--rank',str(rank),
                '--source',args.source,'--int4',args.int4,'--cache-dir',args.cache_dir,'--wait',str(args.wait)],stdout=log,stderr=subprocess.STDOUT))
        _json(base/'build.status.json',dict(state='running',pid=os.getpid(),workers=[c.pid for c in children]))
        codes=[c.wait() for c in children]
        for log in logs:log.close()
        _json(base/'build.status.json',dict(state='complete' if all(c==0 for c in codes) else 'failed',returncodes=codes))
        if any(codes):raise SystemExit(1)

if __name__=='__main__':main()
