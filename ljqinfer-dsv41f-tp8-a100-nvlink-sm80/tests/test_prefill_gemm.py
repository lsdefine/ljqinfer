"""Canonical bytes/scales and FP8-activation GEMM, independent FP64 oracle."""
import pytest
import torch
from ops.prefill.gemm import packed_linear, PrefillLinear


@pytest.mark.parametrize('kind', ['fp4','fp8'])
@pytest.mark.parametrize('n,k', [(35,64),(288,96)])
def test_packed_linear(kind,n,k):
    g=torch.Generator().manual_seed(127)
    x=torch.randn(2,3,k,generator=g).bfloat16()
    if kind=='fp4':
        # All signed nibble values, including negative zero.
        codes=(torch.arange(n*k).reshape(n,k)%16).to(torch.uint8)
        w=(codes[:,0::2] | codes[:,1::2]<<4).view(torch.int8)
        lut=torch.tensor([0.,.5,1.,1.5,2.,3.,4.,6.,-0.,-.5,-1.,-1.5,-2.,-3.,-4.,-6.])
        v=lut[codes.long()].double()
        scale=torch.randint(122,130,(n,k//32),generator=g,dtype=torch.uint8)
        ws=(2.**(scale.double()-127)).repeat_interleave(32,1)
    else:
        w=torch.randn(n,k,generator=g).to(torch.float8_e4m3fn)
        v=w.double()
        scale=torch.randint(122,130,((n+31)//32,k//32),generator=g,dtype=torch.uint8)
        ws=(2.**(scale.double()-127)).repeat_interleave(32,0).repeat_interleave(32,1)[:n]
    a=x.double().reshape(2,3,k//32,32)
    asc=2.**torch.ceil(torch.log2(a.abs().amax(-1,keepdim=True).clamp_min(1e-4)/448))
    aq=(a/asc).to(torch.float8_e4m3fn).double()*asc
    expected=(aq.flatten(-2)@(v*ws).T).bfloat16()
    actual=packed_linear(x,w,scale,output_tile=32)
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    bound=PrefillLinear({'test.weight':w,'test.scale':scale})
    torch.testing.assert_close(bound('test',x),actual,rtol=0,atol=0)


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_grouped_linear_oracle_and_tp(device):
    from ops.prefill.gemm import grouped_linear
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    # Dyadic inputs make FP32 accumulation exact, independently of reduction order.
    gen = torch.Generator(device=device).manual_seed(147)
    n, k = (1024, 4096) if device == 'cuda' else (35, 64)
    x = (torch.randint(-4, 5, (3,8,k), generator=gen, device=device).float()/8).bfloat16()
    w = (torch.randint(-4, 5, (8,n,k), generator=gen, device=device).float()/128).bfloat16()
    result = grouped_linear(x, w)
    for group in range(8):
        expected = (x[:,group].double() @ w[group].double().T).bfloat16()
        torch.testing.assert_close(result[:,group], expected, rtol=0, atol=0)
        partial = grouped_linear(x[:,group:group+1].contiguous(), w[group:group+1])
        torch.testing.assert_close(result[:,group:group+1], partial, rtol=0, atol=0)
    with pytest.raises(ValueError): grouped_linear(x, w[:1])
    with pytest.raises(TypeError): grouped_linear(x.float(), w)


def test_packed_linear_rejects_wrong_abi():
    x=torch.ones(2,32,dtype=torch.bfloat16)
    w=torch.zeros(32,16,dtype=torch.int8)
    s=torch.full((32,1),127,dtype=torch.uint8)
    with pytest.raises(TypeError): packed_linear(x.float(),w,s)
    with pytest.raises(TypeError): packed_linear(x,w,s.float())
    with pytest.raises(ValueError): packed_linear(x,w,s[:1])
    with pytest.raises(ValueError): packed_linear(x,w,s,output_tile=31)
    with pytest.raises(ValueError): packed_linear(x[:,:16],w,s)
