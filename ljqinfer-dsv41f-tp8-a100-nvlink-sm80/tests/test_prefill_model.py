"""Micro-tensor component regression only; NOT released-model validation."""
import pytest
import torch
import torch.nn.functional as F
from model.past import SlotPool, default_layer_views
from model.prefill_layer import DenseLinear, PrefillAttention
from model.prefill_block import DenseRouted, PrefillMoE, PrefillBlock, PrefillEngram
from model.prefill import PrefillModel, PrefillOutput
from model.prefill_attention import PrefillScratch
from ops.prefill import residual as r


class MicroRows:
    """Host Engram table stand-in: rows keyed by token, already dequantized."""
    def __init__(self, table, device):
        self.table, self.device = table, device

    def gather_host(self, slot, start, tokens, history_tokens=()):
        return self.table[torch.as_tensor(tokens, device=self.device)]

    def dequantize(self, rows):
        return rows


class MicroComponentChain:
    """Test-only all-block oracle; never used as a model entry point."""
    def __init__(self, c, blocks, embed, head, norm, target_layers):
        self.c, self.blocks, self.embed, self.head, self.norm = c, blocks, embed, head, norm
        self.targets = target_layers

    def forward(self, tokens, *, past, slot, start, history_tokens=(), scratch=None):
        x = self.embed(tokens)
        h = x[:, None, :].expand(-1, self.c['hc_mult'], -1).clone()
        pre = torch.zeros(h.shape[:2], device=h.device, dtype=torch.float32)
        pre[:, 0] = 1.
        scratch, targets = PrefillScratch() if scratch is None else scratch, []
        for block in self.blocks:
            h = block.prepare(h, slot=slot, start=start, tokens=tokens)
            if block.layer in self.targets:
                targets.append(h.mean(-2))
            h, pre = block(h, pre, past=past, slot=slot, start=start, scratch=scratch)
        y = r.rms(r.collapse(h, pre), self.norm, self.c['norm_eps'])
        return PrefillOutput(self.head(y[-1:]), torch.cat(targets, -1))


def build(device='cpu', layers=40):
    torch.manual_seed(33)
    c = dict(n_layers=layers, dim=32, n_heads=2, head_dim=32, q_lora_rank=16,
        o_groups=2, o_lora_rank=8, index_n_heads=2, index_head_dim=32,
        index_topk=4, norm_eps=1e-6, hc_mult=2, hc_eps=1e-6, hc_sinkhorn_iters=20,
        n_activated_experts=2, gate_temp=1., route_scale=1., norm_topk_prob=True,
        score_func='sqrtsoftplus', swiglu_limit=7., candidate_source_layer=20,
        candidate_block_size=4, candidate_topk_blocks=2, engram_layer_ids=(0,1))
    w = {}
    def weight(name, *shape, ones=False, gain=.03):
        w[name] = (torch.ones(shape) if ones else torch.randn(shape)*gain).to(device)
    views = default_layer_views()
    for i in range(layers):
        p = f'layers.{i}'
        for part in ['attn','ffn']:
            weight(p+'.hc_'+part+'_fn',8,64)
            weight(p+'.hc_'+part+'_scale',3,ones=True)
            weight(p+'.hc_'+part+'_base',8)
            weight(p+'.'+part+'_norm.weight',32,ones=True)
        a = p+'.attn'
        for name,n,k in [('wq_a',16,32),('wq_b',64,16),('wkv',32,32),('wo_b',32,16)]:
            weight(a+'.'+name+'.weight',n,k)
        weight(a+'.q_norm.weight',16,ones=True)
        weight(a+'.kv_norm.weight',32,ones=True)
        weight(a+'.wo_a.weight',16,32)
        weight(a+'.attn_sink',2)
        if views[i].mode == 'full':
            for name in ['wkv','wgate']:
                weight(a+'.compressor.'+name+'.weight',32,32)
            weight(a+'.compressor.norm.weight',32,ones=True)
            weight(a+'.indexer.wk.weight',32,32)
            weight(a+'.indexer.k_norm.weight',32,ones=True)
        if views[i].mode in ('full','reindex'):
            weight(a+'.indexer.wq_b.weight',64,16)
            weight(a+'.indexer.weights_proj.weight',2,32)
        weight(p+'.ffn.gate.weight',3,32)
        weight(p+'.ffn.gate.bias',3)
        for expert in ['shared_experts']+[f'experts.{j}' for j in range(3)]:
            for name,n,k in [('w1',24,32),('w3',24,32),('w2',32,24)]:
                weight(p+'.ffn.'+expert+'.'+name+'.weight',n,k)
        if i in c['engram_layer_ids']:
            weight(p+'.engram.wkv.weight',96,16)
            weight(p+'.engram.q_weight',2,32,ones=True)
            weight(p+'.engram.k_weight',2,32,ones=True)
    lin = DenseLinear(w)
    freqs = torch.polar(torch.ones(256,8,device=device),
                        torch.arange(256,device=device)[:,None]*torch.arange(1,9,device=device)[None,:]/100)
    table = torch.randn(53,2,8,device=device)*.05
    blocks = []
    for i in range(layers):
        attn = PrefillAttention(i,c,w,lin,freqs)
        routed = DenseRouted(f'layers.{i}.ffn',3,lin,7.)
        moe = PrefillMoE(i,c,w,lin,routed)
        # The micro chain runs in float32; the fused engram gate is a bf16-only
        # CUDA kernel, so engram equivalence is covered by test_engram_gate.py.
        eng = None
        blocks.append(PrefillBlock(i,c,w,attn,moe,engram=eng))
    embedding = torch.randn(53,32,device=device)
    head = torch.randn(53,32,device=device)*.1
    model = MicroComponentChain(c,blocks,lambda ids: embedding[torch.as_tensor(ids,device=device)],lambda x:F.linear(x,head),
                        torch.ones(32,device=device),target_layers=(0,layers-1))
    return model


def pool(device):
    return SlotPool(2,256,page_tokens=16,device=device).configure_default(32,32,
        window_dtype=torch.float32,ckv_dtype=torch.float32,index_dtype=torch.float32)


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_40_layers_chunks_and_slots(device):
    if device == 'cuda' and not torch.cuda.is_available(): pytest.skip('CUDA unavailable')
    model=build(device)
    tokens=torch.arange(137,device=device)%53
    whole, chunks=pool(device),pool(device)
    s=whole.alloc(); whole.ensure(s,137)
    expected=model.forward(tokens,past=whole,slot=s,start=0)
    assert whole.pos[s]==0  # only owner commits
    s=chunks.alloc(); other=chunks.alloc()
    pieces=[]
    for start,end in [(0,1),(1,18),(18,127),(127,130),(130,137)]:
        chunks.ensure(s,end)
        got=model.forward(tokens[start:end],past=chunks,slot=s,start=start)
        chunks.set_pos(s,end)
        pieces.append(got.main_hidden)
        if start==1:
            chunks.ensure(other,3)
            model.forward(tokens[:3].flip(0),past=chunks,slot=other,start=0)
            chunks.set_pos(other,3)
    torch.testing.assert_close(got.logits,expected.logits,atol=3e-4,rtol=3e-4)
    torch.testing.assert_close(torch.cat(pieces),expected.main_hidden,atol=3e-4,rtol=3e-4)
    for layer in range(40):
        torch.testing.assert_close(chunks.windows[layer].read(s,9,137),whole.windows[layer].read(0,9,137),atol=2e-3,rtol=2e-3)
    for layer in whole.sources:
        torch.testing.assert_close(chunks.sources[layer].ckv(s,137),whole.sources[layer].ckv(0,137),atol=2e-3,rtol=2e-3)
    with pytest.raises(ValueError): model.forward(tokens[:1],past=chunks,slot=s,start=0)


def test_reject_unbound_packed_and_missing_engram():
    lin=DenseLinear({'a.weight':torch.zeros(2,2,dtype=torch.int8)})
    with pytest.raises(TypeError): lin('a',torch.ones(1,2))
    model=build(layers=3)
    model.blocks[0].engram=None
    with pytest.raises(ValueError):
        PrefillModel(model.c,model.blocks,model.embed,model.head,model.norm)


def test_prefill_imports_never_reach_decode():
    import ast
    from pathlib import Path
    root=Path(__file__).resolve().parents[1]
    files=list((root/'ops/prefill').glob('*.py'))+list((root/'model').glob('prefill*.py'))
    assert len(files)>=9
    for path in files:
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node,ast.ImportFrom):
                assert 'decode' not in (node.module or ''),path
            elif isinstance(node,ast.Import):
                assert all('decode' not in x.name for x in node.names),path


def test_micro_configuration_is_not_a_released_model():
    model = build(layers=3)
    with pytest.raises(ValueError, match='released V4.1 config'):
        PrefillModel(model.c, model.blocks, model.embed, model.head, model.norm)
