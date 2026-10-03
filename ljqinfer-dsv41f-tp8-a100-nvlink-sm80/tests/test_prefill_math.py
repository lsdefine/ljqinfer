"""Independent scalar/formula oracles, separate from chunk self-consistency."""
import math
import pytest
import torch
from ops.prefill import attention as a, residual as r
from ops.prefill.quant import fp4_roundtrip


@pytest.mark.parametrize('e4m3,block', [(True,16),(False,32)])
def test_fp4_scalar_oracle(e4m3,block):
    torch.manual_seed(301)
    x=torch.cat((torch.randn(9,32),torch.zeros(1,32),
                 torch.tensor([[.25,.75,1.25,1.75,2.5,3.5,5.,6.]*4])))
    levels=[0.,.5,1.,1.5,2.,3.,4.,6.]
    expected=torch.empty_like(x)
    for row in range(len(x)):
        for lo in range(0,32,block):
            values=x[row,lo:lo+block].tolist()
            peak=max(abs(v) for v in values)
            if e4m3:
                scale=torch.tensor(max(peak,6*2**-9)/6).to(torch.float8_e4m3fn).float().item()
            else:
                scale=2**math.ceil(math.log2(max(peak,6*2**-126)/6))
            for j,v in enumerate(values):
                code=min(range(8),key=lambda i:(abs(abs(v)/scale-levels[i]),i%2))
                expected[row,lo+j]=math.copysign(levels[code]*scale,v)
    torch.testing.assert_close(fp4_roundtrip(x,block=block,e4m3_scale=e4m3),expected,rtol=0,atol=0)


def test_mhc_and_engram_formula_oracle():
    torch.manual_seed(19)
    t,h,d=3,4,8
    x=torch.randn(t,h,d); fn=torch.randn(h*h+2*h,h*d)*.1
    scale=torch.tensor([.7,.8,.9]);base=torch.randn(h*h+2*h)*.1
    pre,post,comb=r.mixes(x,fn,scale,base,norm_eps=1e-6,hc_eps=1e-6,iters=20)
    for j in range(t):
        flat=x[j].flatten()
        z=(fn@flat)/torch.sqrt(flat.square().mean()+1e-6)
        p=torch.sigmoid(z[:h]*scale[0]+base[:h])+1e-6
        q=2*torch.sigmoid(z[h:2*h]*scale[1]+base[h:2*h])
        matrix=(z[2*h:]*scale[2]+base[2*h:]).reshape(h,h).softmax(-1)+1e-6
        matrix=matrix/(matrix.sum(0,keepdim=True)+1e-6)
        for _ in range(19):
            matrix=matrix/(matrix.sum(1,keepdim=True)+1e-6)
            matrix=matrix/(matrix.sum(0,keepdim=True)+1e-6)
        torch.testing.assert_close(pre[j],p)
        torch.testing.assert_close(post[j],q)
        torch.testing.assert_close(comb[j],matrix)
    y=torch.randn(t,d)
    expected=torch.stack([sum(comb[:,i,j,None]*x[:,i] for i in range(h))+post[:,j,None]*y for j in range(h)],1)
    torch.testing.assert_close(r.expand(y,x,post,comb),expected)
    kv=torch.randn_like(x);value=torch.randn(t,d);qw=torch.randn(h,d);kw=torch.randn(h,d)
    q=x/torch.sqrt(x.square().mean(-1,keepdim=True)+1e-6)*qw
    k=kv/torch.sqrt(kv.square().mean(-1,keepdim=True)+1e-6)*kw
    z=(q*k).sum(-1)/math.sqrt(d)
    expected=x+torch.sigmoid(z.sign()*z.abs().clamp_min(1e-6).sqrt())[...,None]*value[:,None,:]
    torch.testing.assert_close(r.engram_gate(x,torch.cat((kv.flatten(-2),value),-1),qw,kw),expected)


def test_index_ties_and_tp_sum_before_selection():
    torch.manual_seed(9)
    q=torch.randn(7,4,32);w=torch.randn(7,4);k=torch.randn(13,32);p=torch.arange(7)+20
    expected=a.select(q,w,k,p,1,4)
    def reduce(partial):
        other=torch.einsum('thd,kd->thk',q[:,2:],k).relu()
        partial.add_((other*w[:,2:,None]).sum(1)*32**-.5*4**-.5)
    actual=a.select(q[:,:2],w[:,:2],k,p,1,4,total_heads=4,reduce_scores=reduce)
    torch.testing.assert_close(actual,expected)
    # Exact zero ties must not change when the future key suffix or tiling changes.
    zero=torch.zeros(7,4)
    for tile in [1,4,17]:
        ids=a.select(q,zero,k,p,1,4,key_tile=tile)
        torch.testing.assert_close(ids,torch.arange(4).expand(7,-1))
