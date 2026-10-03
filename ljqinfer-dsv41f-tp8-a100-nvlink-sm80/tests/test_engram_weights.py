import pytest
import torch
from safetensors.torch import save_file
from model.engram_weights import HostEngram
from model.weights import shard


def test_mapped_host_table_rank_gather_and_projection(tmp_path):
    torch.manual_seed(7)
    w = torch.randn(97, 256).to(torch.float8_e4m3fn)
    s = torch.randint(125, 129, (97, 8), dtype=torch.uint8)
    path = tmp_path / 'host.safetensors'
    save_file({'e.weight': w, 'e.scale': s}, path)
    table = HostEngram.open(path, 'e')
    ids = torch.randint(0, 97, (2, 6, 24))
    a, b = table.gather(ids)
    assert torch.equal(a.view(torch.uint8), w.view(torch.uint8)[ids])
    assert torch.equal(b, s[ids])
    def unpack(v, scale):
        return v.float() * (scale.float() - 127).exp2().repeat_interleave(32, -1)
    x = unpack(a, b).flatten(-2)
    proj = torch.randn(32, 24 * 256)
    expected = x @ proj.T
    partials = []
    for rank in range(8):
        v, scale = table.gather(ids, rank=rank)
        assert v.shape == (2, 6, 3, 256)
        p = shard('layers.1.engram.wkv.weight', proj, rank)
        partials.append(unpack(v, scale).flatten(-2) @ p.T)
    torch.testing.assert_close(torch.stack(partials).sum(0), expected, atol=1e-3, rtol=1e-4)
    assert table.weight.device.type == 'cpu'
    with pytest.raises(ValueError): table.gather(torch.tensor([-1]))
    with pytest.raises(ValueError): table.gather(torch.tensor([97]))
    with pytest.raises(ValueError): table.gather(ids, rank=8)
    with pytest.raises(ValueError): table.gather(ids.float())
    with pytest.raises(ValueError): HostEngram(w, s.float())
