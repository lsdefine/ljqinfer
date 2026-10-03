"""Full released shapes; deterministic random weights materialized only on use.
No low-rank stand-ins, expert-count reduction or skipped GEMMs.
Explicit FP32 diagnostic and released mixed-precision execution.
"""
import hashlib
from functools import lru_cache
from pathlib import Path
from model.engram import EngramHash, EngramRows
import math
import torch
import torch.nn.functional as F
from model.prefill_config import released_config, rotary_frequencies
from model.prefill import PrefillModel
from model.prefill_layer import DenseLinear, PrefillAttention
from model.prefill_block import PrefillBlock, PrefillEngram, PrefillMoE, DenseRouted
from model.past import default_layer_views


class RandomWeights:
    def __init__(self, device):
        self.c, self.device = released_config(), device
        self.accesses = []

    def random(self, name, shape):
        gen = torch.Generator(device=self.device)
        gen.manual_seed(int.from_bytes(hashlib.sha256(name.encode()).digest()[:8], 'little') % (2**63-1))
        return torch.randn(shape, device=self.device, generator=gen) * (.2/math.sqrt(shape[-1]))

    def __getitem__(self, name):
        c = self.c
        d, h, hd, qr, o, rank = (c[k] for k in ['dim','hc_mult','head_dim','q_lora_rank','o_groups','o_lora_rank'])
        self.accesses.append(name)
        if '.hc_' in name:
            n = h*(h+2)
            if name.endswith('_fn'): return self.random(name, (n,h*d))
            if name.endswith('_base'): return self.random(name, (n,))
            if name.endswith('_scale'): return torch.ones(3,device=self.device)
        if name.endswith(('q_weight','k_weight')):
            return torch.ones(h,d,device=self.device)
        if name.endswith('norm.weight'):
            n = qr if '.q_norm.' in name else c['index_head_dim'] if '.k_norm.' in name else hd if any(x in name for x in ['.kv_norm.','.compressor.norm.']) else d
            return torch.ones(n,device=self.device)
        if name.endswith('.attn_sink'): return self.random(name, (c['n_heads'],))
        if name.endswith('.gate.bias'): return self.random(name, (c['n_routed_experts'],))
        if name.endswith('.gate.weight'): return self.random(name, (c['n_routed_experts'],d))
        suffix = name.split('.')[-2]
        if '.engram.' in name: shape = (d*(h+1),(c['engram_max_ngram_size']-1)*c['engram_n_heads']*c['engram_head_dim'])
        elif '.ffn.' in name:
            shape = (d,c['moe_inter_dim']) if suffix=='w2' else (c['moe_inter_dim'],d)
        elif '.compressor.' in name: shape = (hd,d)
        elif '.indexer.' in name:
            shape = {'wk':(c['index_head_dim'],hd),'wq_b':(c['index_n_heads']*c['index_head_dim'],qr),'weights_proj':(c['index_n_heads'],d)}[suffix]
        else:
            shape = {'wq_a':(qr,d),'wq_b':(c['n_heads']*hd,qr),'wkv':(hd,d),'wo_a':(o*rank,c['n_heads']*hd//o),'wo_b':(d,o*rank)}[suffix]
        return self.random(name, shape)


class MixedWeights(RandomWeights):
    """Released weight dtypes, full shapes, one transient packed matrix at a time."""
    def __init__(self, device):
        super().__init__(device)
        self.pending_scale = None
        self.dtype_hits = set()

    def __getitem__(self, name):
        if name.endswith('.scale'):
            if self.pending_scale is None or self.pending_scale[0] != name:
                raise RuntimeError('scale must follow its weight request')
            _, scale = self.pending_scale
            self.pending_scale = None
            return scale
        x = super().__getitem__(name)
        if '.hc_' in name or name.endswith(('.gate.bias','.attn_sink')):
            return x
        if '.compressor.' in name and name.endswith('.weight') and not '.norm.' in name:
            layer = int(name.split('.')[1])
            return x if self.c['compress_ratios'][layer] > 1 else x.bfloat16()
        dense = (not name.endswith('.weight') or name.endswith('norm.weight') or
                 name.endswith(('.gate.weight','.wo_a.weight','.indexer.wk.weight',
                                '.indexer.weights_proj.weight')))
        if dense:
            self.dtype_hits.add('bf16')
            return x.bfloat16()
        n,k = x.shape
        if '.experts.' in name:
            z = x.unflatten(-1,(-1,32))
            scale = (z.abs().amax(-1).clamp_min(1e-4)/6).log2().ceil().exp2()
            u = z/scale[...,None]
            mid = torch.tensor([.25,.75,1.25,1.75,2.5,3.5,5.],device=x.device)
            code = torch.bucketize(u.abs().contiguous(),mid)
            tie = (code<7) & (u.abs()==mid[code.clamp(max=6)]) & (code%2==1)
            code = ((code+tie.long()) | (u.signbit().long()<<3)).flatten(-2).byte()
            weight = (code[:,::2] | (code[:,1::2]<<4)).view(torch.int8)
            self.dtype_hits.add('fp4')
        else:
            assert n%32==0 and k%32==0
            z = x.reshape(n//32,32,k//32,32)
            scale = (z.abs().amax((1,3)).clamp_min(1e-4)/448).log2().ceil().exp2()
            weight = (z/scale[:,None,:,None]).to(torch.float8_e4m3fn).reshape(n,k)
            self.dtype_hits.add('fp8')
        self.pending_scale = (name[:-6]+'scale',(scale.log2()+127).byte())
        return weight


@lru_cache(maxsize=1)
def released_hasher():
    from tokenizers import Tokenizer
    class Adapter:
        backend_tokenizer = Tokenizer.from_file(str(Path(
            '/mnt/data/kw/models/DeepSeek-V4.1-Flash/tokenizer.json')))
        def __len__(self): return self.backend_tokenizer.get_vocab_size()
    return EngramHash(released_config(), Adapter())


class RandomHostTable:
    """Lazy full-domain random weights, keyed by actual (layer, hash row).

    Materialize only requested CPU rows. Repeated/colliding hashes share weights.
    Returned ABI is exactly HostEngram's FP8 / E8M0; never a GPU table.
    """
    def __init__(self, layer, num_rows):
        self.layer, self.num_rows = layer, num_rows

    def gather(self, ids, *, rank=None):
        if rank is not None:
            if not 0 <= rank < 8: raise ValueError('rank outside 0..7')
            ids = ids[...,rank*3:(rank+1)*3]
        if ids.dtype != torch.int64 or ids.device.type != 'cpu':
            raise ValueError('CPU int64 hashes required')
        if ids.numel() and (ids.min()<0 or ids.max()>=self.num_rows):
            raise ValueError('hash out of range')
        unique, inverse = ids.unique(return_inverse=True)
        rows=[]
        for row in unique.tolist():
            gen=torch.Generator().manual_seed(self.layer*1000000007+row)
            rows.append(torch.randn(256,generator=gen)*.2)
        x=torch.stack(rows).unflatten(-1,(8,32))
        scale=(x.abs().amax(-1).clamp_min(1e-4)/448).log2().ceil().exp2()
        v=(x/scale[...,None]).to(torch.float8_e4m3fn).flatten(-2)
        sc=(scale.log2()+127).to(torch.uint8)
        return (v.view(torch.uint8)[inverse].view(torch.float8_e4m3fn).reshape(*ids.shape,256),
                sc[inverse].reshape(*ids.shape,8))


def build(device='cuda', length=256, *, mixed=False):
    from ops.prefill.gemm import PrefillLinear
    w = MixedWeights(device) if mixed else RandomWeights(device)
    c = w.c
    lin = PrefillLinear(w) if mixed else DenseLinear(w)
    fs = {b:rotary_frequencies(c, 2 if b else 0, length, device=device) for b in [False,True]}
    def embed(ids):
        x = torch.stack([w.random('embed.'+str(int(i)),(c['dim'],)) for i in ids])
        return x.bfloat16() if mixed else x
    # Tiled vocabulary projection changes storage, not vocabulary or matrix shape.
    def head(x):
        out=[]
        for i in range(0,c['vocab_size'],2048):
            weight=w.random('head.'+str(i),(min(2048,c['vocab_size']-i),c['dim']))
            if mixed: weight=weight.bfloat16().float()
            out.append(F.linear(x.float(),weight))
        return torch.cat(out,-1)
    hasher = released_hasher()
    if mixed:
        from model.prefill_build import build_prefill
        tables={i:RandomHostTable(i,c['engram_num_embeddings'][j])
                for j,i in enumerate(c['engram_layer_ids'])}
        return build_prefill(c,w,hasher,tables,device=device,length=length,
                             embed=embed,head=head),w
    blocks=[]
    for i in range(40):
        attn=PrefillAttention(i,c,w,lin,fs[bool(c['compress_ratios'][i])])
        routed=DenseRouted(f'layers.{i}.ffn',c['n_routed_experts'],lin,c['swiglu_limit'])
        moe=PrefillMoE(i,c,w,lin,routed)
        eng=None
        if i in c['engram_layer_ids']:
            table=RandomHostTable(i,c['engram_num_embeddings'][c['engram_layer_ids'].index(i)])
            eng=PrefillEngram(i,w,lin,EngramRows(hasher,i,table),eps=c['norm_eps'])
        blocks.append(PrefillBlock(i,c,w,attn,moe,engram=eng))
    model=PrefillModel(c,blocks,embed,head,torch.ones(c['dim'],device=device), target_layers=c['dspark_target_layer_ids'])
    return model,w
