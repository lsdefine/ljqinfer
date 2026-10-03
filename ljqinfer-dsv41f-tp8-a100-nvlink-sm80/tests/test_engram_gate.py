"""Fused engram gate: bf16-only contract and strict per-token independence."""
import pytest
import torch

from ops.prefill import engram_gate


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')

H, D, T = 2, 32, 137


def inputs(dtype=torch.bfloat16, rows=T):
    torch.manual_seed(0)
    g = torch.Generator(device='cuda').manual_seed(0)
    kw = dict(device='cuda', dtype=dtype, generator=g)
    return (torch.randn(rows, H, D, **kw), torch.randn(rows, (H + 1) * D, **kw),
            torch.randn(H, D, **kw), torch.randn(H, D, **kw))


def test_rows_are_independent():
    """Every row must only depend on itself: chunking cannot change a result."""
    gate = engram_gate.extension().engram_gate
    x, kv, q, k = inputs()
    whole = gate(x.clone(), kv.clone(), q, k, 1e-6)
    for n in (1, 2, 3, 17, 18, 127, 128, 130, T):
        part = gate(x[:n].clone().contiguous(), kv[:n].clone().contiguous(), q, k, 1e-6)
        assert torch.equal(part, whole[:n]), f'row {n} depends on chunk length'
    pieces = [gate(x[a:b].clone().contiguous(), kv[a:b].clone().contiguous(), q, k, 1e-6)
              for a, b in [(0, 1), (1, 18), (18, 127), (127, 130), (130, T)]]
    assert torch.equal(torch.cat(pieces), whole)


@pytest.mark.parametrize('dtype', [torch.float32, torch.float16])
def test_rejects_non_bf16(dtype):
    """Wrong dtype used to be reinterpreted as bf16, silently writing half the output."""
    gate = engram_gate.extension().engram_gate
    x, kv, q, k = inputs(dtype, rows=4)
    with pytest.raises(RuntimeError):
        gate(x, kv, q, k, 1e-6)


def test_reference_matches_kernel_bf16():
    """The fp32 oracle backing the diagnostic chain must track the fused kernel."""
    x, kv, q, k = inputs()
    fused = engram_gate.extension().engram_gate(x, kv, q, k, 1e-6)
    oracle = engram_gate.reference(x, kv, q, k, 1e-6)
    assert torch.allclose(fused.float(), oracle.float(), atol=2e-2, rtol=2e-2)


def test_float32_routes_to_oracle():
    """Dispatcher keeps fp32 usable even though the kernel rejects it."""
    x, kv, q, k = inputs()
    out = engram_gate.engram_gate(x.float(), kv.float(), q.float(), k.float(), 1e-6)
    assert out.dtype is torch.float32
    assert torch.allclose(out, engram_gate.reference(x, kv, q, k, 1e-6).float(), atol=3e-2, rtol=3e-2)
