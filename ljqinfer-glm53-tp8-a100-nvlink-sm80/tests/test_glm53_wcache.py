"""Small CPU regressions for automatic shm snapshots; no model download needed."""
import json,time
from concurrent.futures import ThreadPoolExecutor
import torch
import pytest
from model.wcache import Cache
from model import weights as W

@pytest.fixture
def cache(tmp_path,monkeypatch):
    c=Cache.__new__(Cache)
    c.root=tmp_path;c.source=tmp_path;c.int4=tmp_path
    c.fingerprint='test';c.wait=0
    calls=[]
    def build(source,rank):
        calls.append(rank);time.sleep(.02)
        return {'x':torch.arange(16,dtype=torch.float32).reshape(4,4)}
    monkeypatch.setattr(W,'build_globals',build)
    return c,calls

def test_first_load_then_hit(cache):
    c,calls=cache
    a=c.tensors(0,device='cpu');b=c.tensors(0,device='cpu')
    assert calls==[0]
    assert torch.equal(a['x'],b['x'])
    assert not list(c.root.glob('*.tmp'))

@pytest.mark.parametrize('damage',['missing','receipt','truncated','dependency'])
def test_invalid_cache_rebuilds(cache,damage):
    c,calls=cache;p=c.ensure(0);r=p.with_suffix('.json')
    if damage=='missing':p.unlink()
    elif damage=='receipt':r.write_text('{broken')
    elif damage=='truncated':p.write_bytes(b'not a tensor file')
    else:c.fingerprint='changed'
    c.ensure(0)
    assert calls==[0,0]
    assert torch.equal(c.tensors(0,device='cpu')['x'],torch.arange(16).float().reshape(4,4))

def test_concurrent_build_once(cache):
    c,calls=cache
    with ThreadPoolExecutor(max_workers=4) as pool:
        paths=list(pool.map(lambda _:c.ensure(0),range(4)))
    assert len(set(paths))==1 and calls==[0]

@pytest.mark.parametrize('rank,layer',[(-1,None),(8,None),(0,-1),(0,78)])
def test_bounds(cache,rank,layer):
    with pytest.raises(ValueError):cache[0].ensure(rank,layer)

def test_nonfinite_never_published(cache,monkeypatch):
    c,_=cache
    monkeypatch.setattr(W,'build_globals',lambda *a:{'bad':torch.tensor([float('nan')])})
    with pytest.raises(ValueError,match='nonfinite'):c.ensure(0)
    assert not list(c.root.glob('*.safetensors'))
