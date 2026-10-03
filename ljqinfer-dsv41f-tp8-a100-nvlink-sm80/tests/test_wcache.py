import pytest
import torch
from model.wcache import WeightCache, identity


def test_hit_never_rebuilds_and_bytes_survive(tmp_path):
    cache = WeightCache(tmp_path)
    key = identity(source='immutable-sha', unit='layer0', rank=3)
    tensors = {'fp4': torch.arange(128, dtype=torch.int8),
               'scale': torch.arange(128, dtype=torch.uint8),
               'fp8': torch.randn(256).to(torch.float8_e4m3fn)}
    first = cache.get_or_build(key, lambda: tensors)
    second = WeightCache(tmp_path).get_or_build(key, lambda: pytest.fail('rebuilt on hit'))
    for name in tensors:
        assert torch.equal(first[name].view(torch.uint8), tensors[name].view(torch.uint8))
        assert torch.equal(second[name].view(torch.uint8), tensors[name].view(torch.uint8))
    for change in ({'rank': 4}, {'source': 'other-sha'}, {'pack_abi': 'swizzle-v2'}, {'unit': 'layer1'}):
        args = dict(source='immutable-sha', unit='layer0', rank=3)
        args.update(change)
        assert cache.load(identity(**args)) is None


def test_failure_does_not_publish(tmp_path, monkeypatch):
    import model.wcache as mod
    cache = WeightCache(tmp_path)
    key = identity(source='a', unit='b', rank=0)
    def fail(*args, **kwargs):
        raise OSError('injected write failure')
    with monkeypatch.context() as ctx:
        ctx.setattr(mod, 'save_file', fail)
        with pytest.raises(OSError): cache.get_or_build(key, lambda: {'x': torch.ones(2)})
    assert cache.load(key) is None
    assert not list(tmp_path.glob('.building-*'))
    assert cache.get_or_build(key, lambda: {'x': torch.ones(2)})['x'].sum() == 2


def test_bad_builder_and_corrupt_cache(tmp_path):
    cache = WeightCache(tmp_path)
    key = identity(source='a', unit='b', rank='host')
    with pytest.raises(ValueError): cache.get_or_build(key, lambda: {})
    with pytest.raises(ValueError): cache.get_or_build(key, lambda: {'x': torch.ones(3, 4).T})
    cache.path(key).write_bytes(b'broken')
    with pytest.raises(Exception): cache.get_or_build(key, lambda: pytest.fail('silently rebuilding corrupt cache'))
