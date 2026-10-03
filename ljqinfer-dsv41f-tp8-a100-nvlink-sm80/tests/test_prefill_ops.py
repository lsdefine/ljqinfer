import pytest
import torch
from ops.prefill import attention as a, residual as r, candidates as c


@pytest.mark.parametrize('ratio', [1, 2])
def test_compress_arbitrary_chunks(ratio):
    torch.manual_seed(12)
    v, s = torch.randn(137, 32), torch.randn(137, 32)
    cv, cs = torch.zeros(2, 32), torch.zeros(2, 32)
    full, _ = a.compress(v, s, 0, ratio, cv, cs)
    cv2, cs2 = torch.zeros_like(cv), torch.zeros_like(cs)
    chunks = [1, 4, 126, 3, 3]
    start, parts = 0, []
    for n in chunks:
        x, origin = a.compress(v[start:start+n], s[start:start+n], start, ratio, cv2, cs2)
        assert origin == start//ratio
        parts.append(x); start += n
    torch.testing.assert_close(torch.cat(parts), full)
    torch.testing.assert_close(cv2, cv); torch.testing.assert_close(cs2, cs)
    if ratio == 2:
        oracle = torch.stack([(v[i:i+2]*s[i:i+2].softmax(0)).sum(0) for i in range(0,136,2)])
        torch.testing.assert_close(full, oracle)


@pytest.mark.parametrize('ratio', [1, 2])
def test_sparse_and_selection_dense_oracle(ratio):
    torch.manual_seed(81)
    t, d, h = 17, 32, 4
    pos = torch.arange(125, 125+t)
    q, local, keys = torch.randn(t,h,d), torch.randn(141,d), torch.randn(142//ratio,d)
    sink, weights = torch.randn(h), torch.randn(t,h)
    idx = a.select(q, weights, keys, pos, ratio, 7, query_tile=3, key_tile=11)
    scores = torch.einsum('thd,kd->thk', q, keys).relu().mul(weights[...,None]).sum(1)*d**-.5*h**-.5
    scores.masked_fill_(torch.arange(len(keys))[None] >= (pos[:,None]+1)//ratio, -torch.inf)
    # ReLU can create exact zero ties; compare selected scores, not arbitrary IDs.
    torch.testing.assert_close(scores.gather(-1, idx).sort(-1).values, scores.topk(7).values.sort(-1).values)
    got = a.sparse(q,local,1,keys,idx,pos,sink,window=128,ratio=ratio,scale=d**-.5,query_tile=3)
    oracle=[]
    for j,p in enumerate(pos.tolist()):
        rows=torch.cat((local[max(1,p-127)-1:p],keys[idx[j]]))
        log=q[j]@rows.T*d**-.5
        prob=torch.cat((log,sink[:,None]),-1).softmax(-1)[:,:-1]
        oracle.append(prob@rows)
    torch.testing.assert_close(got,torch.stack(oracle),atol=2e-6,rtol=2e-5)


def test_candidate_blocks_dense_oracle():
    torch.manual_seed(99)
    q,w,k=torch.randn(31,4,32),torch.randn(31,4),torch.randn(55,32)
    pos=torch.arange(4,35)*2
    out=c.build(q,w,k,pos,2,block_size=4,top_blocks=2,query_tile=3,block_tile=3)
    scores=torch.einsum('thd,kd->thk',q,k).relu().mul(w[...,None]).sum(1)
    for i,p in enumerate(pos.tolist()):
        n=(p+1)//2; last=(n-1)//4
        bs=torch.tensor([scores[i,j*4:(j+1)*4].max() for j in range(last)])
        old=bs.topk(min(1,last)).indices.tolist()
        expected={row for b in old+[last] for row in range(b*4,min(b*4+4,n))}
        assert set(out[i][out[i]>=0].tolist())==expected


def test_rope_and_residual_orientation():
    torch.manual_seed(6)
    x=torch.randn(9,4,32);f=torch.polar(torch.ones(9,8),torch.randn(9,8))
    torch.testing.assert_close(a.rope(a.rope(x,f),f,inverse=True),x)
    pre,post,comb=r.mixes(x,torch.randn(24,128),torch.randn(3),torch.randn(24))
    y=torch.randn(9,32)
    expected=post[...,None]*y[:,None]+torch.einsum('tij,tid->tjd',comb,x)
    torch.testing.assert_close(r.expand(y,x,post,comb),expected)
    torch.testing.assert_close(comb.sum(-2),torch.ones(9,4),atol=2e-6,rtol=0)


def test_empty_history_and_tp_contract():
    q,w,k=torch.randn(1,2,32),torch.randn(1,2),torch.empty(0,32)
    assert (a.select(q,w,k,torch.tensor([0]),2,4)==-1).all()
    with pytest.raises(ValueError): a.select(q,w,k,torch.tensor([0]),2,4,total_heads=4)
    local=torch.randn(1,32)
    out=a.sparse(q,local,0,k,torch.full((1,4),-1),torch.tensor([0]),torch.zeros(2),window=128,ratio=2,scale=32**-.5)
    assert torch.isfinite(out).all()
