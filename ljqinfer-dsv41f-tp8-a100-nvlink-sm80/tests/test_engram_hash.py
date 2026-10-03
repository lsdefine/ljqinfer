"""Released tokenizer/hash parity and host row wiring; no model weight download."""
import importlib.util
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import pytest
import torch
from model.engram import EngramRows
from model.engram_weights import HostEngram
from model.prefill_config import released_config
from released_random import released_hasher


@pytest.fixture(scope='module')
def oracle():
    root=Path(os.environ.get('DSV41_REFERENCE',
        '/mnt/data/kw/models/DeepSeek-V4.1-Flash/inference'))
    if not (root/'engram.py').is_file():
        pytest.skip('official Engram reference not installed')
    spec=importlib.util.spec_from_file_location('released_engram_reference',root/'engram.py')
    module=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=module
    spec.loader.exec_module(module)
    from tokenizers import Tokenizer
    class Adapter:
        backend_tokenizer=Tokenizer.from_file(str(root.parent/'tokenizer.json'))
        def __len__(self): return self.backend_tokenizer.get_vocab_size()
    c=released_config()
    args=SimpleNamespace(**c,max_batch_size=1,max_seq_len=256)
    return module.NgramHashState(args,module.EngramLayout.from_args(args),Adapter())


@pytest.mark.parametrize('masked',[False,True])
def test_official_hash_full_chunks_and_replay(oracle,masked):
    hasher=released_hasher()
    tokens=tuple(range(7,144))
    mask=torch.ones(len(tokens),dtype=torch.bool)
    if masked: mask[[0,2,63,127]]=False
    expected=oracle(torch.tensor([tokens]),0,mask[None])[0]
    actual=hasher(tokens,start=0,token_mask=mask)
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    boundaries=[0,1,3,7,64,128,len(tokens)]
    for a,b in zip(boundaries,boundaries[1:]):
        h=max(0,a-3)
        part=hasher(tokens[a:b],start=a,history_tokens=tokens[h:a],
                    token_mask=mask[a:b],history_mask=mask[h:a])
        torch.testing.assert_close(part,expected[a:b],rtol=0,atol=0)
    a=9
    replay=hasher(tokens[a:],start=a,history_tokens=tokens[a-3:a],
                  token_mask=mask[a:],history_mask=mask[a-3:a])
    torch.testing.assert_close(replay,expected[a:],rtol=0,atol=0)


def test_hash_rejects_missing_history_and_bad_ids(oracle):
    hasher=released_hasher()
    with pytest.raises(ValueError): hasher((9,),start=3)
    with pytest.raises(ValueError): hasher((-1,),start=0)
    with pytest.raises(ValueError): hasher((len(hasher.token_map),),start=0)
    with pytest.raises(ValueError): hasher((9,),start=0,token_mask=[True,False])


def test_host_rows_decode_e8m0_and_rank_slice():
    class Hash:
        layout=SimpleNamespace(layer_ids=(0,))
        def __call__(self,tokens,**kw):
            return torch.arange(24).remainder(3).expand(len(tokens),1,24)
    values=(torch.arange(3*256).reshape(3,256).remainder(15).float()/8).to(torch.float8_e4m3fn)
    scales=torch.tensor([[125]*8,[127]*8,[129]*8],dtype=torch.uint8)
    table=HostEngram(values,scales)
    ids=Hash()((7,8))[:,0]
    expected=(values.float().unflatten(-1,(8,32))*
              (scales.float()-127).exp2()[...,None]).flatten(-2)[ids].bfloat16()
    torch.testing.assert_close(EngramRows(Hash(),0,table)(0,0,(7,8)),expected,rtol=0,atol=0)
    for rank in range(8):
        actual=EngramRows(Hash(),0,table,rank=rank)(0,0,(7,8))
        torch.testing.assert_close(actual,expected[:,rank*3:(rank+1)*3],rtol=0,atol=0)
    assert table.weight.device.type=='cpu'
