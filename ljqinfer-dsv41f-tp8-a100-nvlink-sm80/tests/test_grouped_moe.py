"""CUDA port: independent unpack oracle, ragged GEMM and routed MoE parity."""
import pytest
import torch
from ops.prefill.grouped_moe import extension, unpack_fp4, GroupedBankRouted
from routed_reference import BankRouted

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


def test_unpack_and_ragged_gemm():
    w=torch.arange(256,device='cuda',dtype=torch.int16).to(torch.int8).reshape(2,8,16)
    s=torch.arange(123,139,device='cuda',dtype=torch.uint8).reshape(2,8,1)
    u=w.to(torch.int16)&255
    code=torch.stack((u&15,u>>4),-1).flatten(-2)
    table=torch.tensor([0,.5,1,1.5,2,3,4,6],device='cuda',dtype=torch.float64)
    oracle=(table[(code&7).long()]*torch.where(code&8 != 0,-1.,1.)*(s.double()-127).exp2().repeat_interleave(32,-1)).bfloat16()
    torch.testing.assert_close(unpack_fp4(w,s),oracle,rtol=0,atol=0)
    gen=torch.Generator(device='cuda').manual_seed(733)
    sizes=torch.tensor([1,17,0,33],device='cuda',dtype=torch.int64)
    x=(torch.randint(-8,9,(51,64),device='cuda',generator=gen).float()/16).bfloat16()
    weights=(torch.randint(-8,9,(4,32,64),device='cuda',generator=gen).float()/16).bfloat16()
    actual=extension().grouped_gemm_sm80(x,weights,sizes,0)
    expected=torch.cat([torch.nn.functional.linear(a.double(),b.double()).bfloat16() for a,b in zip(x.split(sizes.tolist()),weights)])
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)


@pytest.mark.parametrize('length',[1,129,1024])
def test_routed_moe(length):
    # Actual released dimensions, 48 local of 384 total experts, six routes/token.
    from parallel_random import RankRandom
    class Parallel:
        rank=0
        def sum(self,x): pass
    w=RankRandom('cuda',0);c=w.full.c
    gen=torch.Generator(device='cuda').manual_seed(817)
    x=torch.randn(length,c['dim'],device='cuda',generator=gen).bfloat16()
    ids=torch.rand(length,c['n_routed_experts'],device='cuda',generator=gen).topk(6,-1).indices
    if length==1: ids[0]=torch.arange(6,device='cuda')
    p=torch.rand(length,6,device='cuda',generator=gen);p=p/p.sum(-1,keepdim=True)
    reference=BankRouted('layers.0.ffn',c,w,Parallel())(x,ids,p)
    op=GroupedBankRouted('layers.0.ffn',c,w,Parallel())
    actual=op(x,ids,p)
    delta=(actual-reference).float()
    rel=(delta.norm()/reference.norm().clamp_min(1e-20)).item()
    print({'length':length,'max_abs':delta.abs().max().item(),'relative_l2':rel,'gemm_calls':op.gemm_calls})
    assert op.gemm_calls>0 and torch.isfinite(actual).all()
    assert rel < .005
    torch.testing.assert_close(actual,reference,rtol=.01,atol=2e-4)
    if length==1:
        empty=op(x,torch.full_like(ids,383),p)
        assert torch.count_nonzero(empty)==0


@pytest.mark.parametrize('expert',[0,7,31])
def test_projection_fp64_oracle(expert):
    """Full dimensions; compare arithmetic to FP64, not the old backend."""
    from parallel_random import RankRandom
    from ops.prefill.gemm import activation_fp8
    w=RankRandom('cuda',0)
    gen=torch.Generator(device='cuda').manual_seed(817+expert)
    base='layers.0.ffn.local_experts.'
    w13,s13=w[base+'w13.weight'][expert],w[base+'w13.scale'][expert]
    projections=[(w13[0],s13[0]),(w13[1],s13[1]),
                 (w[base+'w2.weight'][expert],w[base+'w2.scale'][expert])]
    for packed,scale in projections:
        k=packed.shape[-1]*2
        x=torch.randn(129,k,device='cuda',generator=gen).bfloat16()
        a=activation_fp8(x)
        torch.testing.assert_close(a,a.bfloat16().float(),rtol=0,atol=0)
        b=unpack_fp4(packed[None],scale[None])
        ref=torch.nn.functional.linear(a.double(),b[0].double())
        y=extension().grouped_gemm_sm80(a.bfloat16(),b,
                     torch.tensor([129],device='cuda'),0).double()
        norm=ref.norm().clamp_min(1e-30)
        rel=((y-ref).norm()/norm).item()
        gap=((y-ref.bfloat16().double()).norm()/norm).item()
        print(dict(expert=expert,k=k,relative_l2=rel,rounding_gap=gap))
        assert torch.isfinite(y).all()
        # BF16 RNE bound for normal values, plus FP32 accumulation budget.
        assert rel < 2**-8 + 1e-6
        # FP32 rounding can cross a BF16 midpoint: use a forward-error bound.
        u=2**-24
        gamma=k*u/(1-k*u)
        accumulation_bound=gamma*(a.double().abs()@b[0].double().abs().T)
        rounded=ref.bfloat16()
        up=torch.nextafter(rounded,torch.full_like(rounded,float('inf'))).double()
        down=torch.nextafter(rounded,torch.full_like(rounded,-float('inf'))).double()
        half_ulp=torch.maximum(up-rounded.double(),rounded.double()-down)/2
        assert ((y-ref).abs() <= half_ulp+accumulation_bound).all()
