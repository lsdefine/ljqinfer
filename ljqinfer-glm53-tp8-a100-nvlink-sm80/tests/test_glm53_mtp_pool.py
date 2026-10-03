import pytest
import torch
from model.glm53_mtp_pool import MTPKVPool,VERIFY_Q,TARGET_LAYERS

def pool(**kw):
    return MTPKVPool(max_tokens=12,max_sequence_tokens=24,max_sequences=2,
        page_size=4,layers=2,head_dim=3,device='cpu',**kw)

def tensors(n,base=0):
    k=[(torch.arange(n*3).reshape(n,1,3)+base+100*l).bfloat16() for l in range(2)]
    return k,[x+20 for x in k]

def test_geometry():
    assert VERIFY_Q==8 and TARGET_LAYERS==(5,19,33,47,61,75)
    p=MTPKVPool(max_tokens=64,max_sequence_tokens=64,device='cpu')
    assert p.k.shape==(6,1,64,1,128)

def test_append_page_boundary_and_read():
    p=pool();k,v=tensors(3);p.append(0,0,k,v);a,b=tensors(4,40);p.append(0,p.lengths[0],a,b)
    for l in range(2):
        kk,vv=p.read_layer(l,0,0,7)
        assert torch.equal(kk,torch.cat([k[l],a[l]]))
        assert torch.equal(vv,torch.cat([v[l],b[l]]))
    assert p.lengths==[7,0] and p.resident_pages==2

def test_slots_do_not_alias():
    p=pool();k,v=tensors(4);p.append(0,0,k,v);a,b=tensors(4,80);p.append(1,0,a,b)
    assert torch.equal(p.read_layer(0,0,0,4)[0],k[0])
    assert torch.equal(p.read_layer(0,1,0,4)[0],a[0])
    assert p.host_page_table[0][0]!=p.host_page_table[1][0]

def test_oom_is_atomic():
    p=pool();k,v=tensors(8);p.append(0,0,k,v)
    before=[t[:] for t in p.host_page_table];free=p.free_pages[:]
    with pytest.raises(MemoryError):p.append(1,0,k,v)
    assert p.host_page_table==before and p.free_pages==free and p.lengths==[8,0]

def test_truncate_rewrite_release():
    p=pool();k,v=tensors(9);p.append(0,0,k,v);p.truncate(0,3)
    assert p.resident_pages==1
    a,b=tensors(5,90);p.append(0,p.lengths[0],a,b)
    assert torch.equal(p.read_layer(0,0,0,8)[0],torch.cat([k[0][:3],a[0]]))
    p.release(0);assert p.resident_pages==0 and p.lengths==[0,0]
    with pytest.raises(ValueError):p.read_layer(0,0,0,1)
    p.append(1,0,a,b);assert torch.equal(p.read_layer(0,1,0,5)[0],a[0])

def test_shape_rejected_before_allocate():
    p=pool();k,v=tensors(2);k[1]=torch.empty((2,2,3),dtype=torch.bfloat16)
    with pytest.raises(ValueError):p.append(0,0,k,v)
    assert p.resident_pages==0

@pytest.mark.parametrize('sid',[-1,2])
def test_bad_slot(sid):
    with pytest.raises(IndexError):pool().release(sid)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable')
def test_cuda_lifecycle():
    p=MTPKVPool(max_tokens=12,max_sequence_tokens=24,max_sequences=2,
        page_size=4,layers=2,head_dim=3,device='cuda:0')
    k,v=tensors(7);k=[x.cuda() for x in k];v=[x.cuda() for x in v]
    p.append(0,0,k,v)
    for l in range(2):
        a,b=p.read_layer(l,0,0,7)
        assert torch.equal(a,k[l]) and torch.equal(b,v[l])
    p.truncate(0,3)
    a,b=tensors(5,90);a=[x.cuda() for x in a];b=[x.cuda() for x in b]
    p.append(0,3,a,b)
    assert torch.equal(p.read_layer(0,0,0,8)[0],torch.cat([k[0][:3],a[0]]))
    p.release(0);p.append(1,0,a,b)
    assert p.lengths==[0,5]
    assert torch.equal(p.read_layer(0,1,0,5)[0],a[0])
    p.release(1);assert p.resident_pages==0
