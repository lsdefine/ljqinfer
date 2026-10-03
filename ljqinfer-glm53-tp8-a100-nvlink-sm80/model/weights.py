"""GLM53 rank-local TP8 final-layout weights. No GGUF, CP delta or native MTP.

load_tp8 uses an automatically populated shm cache. A rank owns vocabulary rows
[r*19360:(r+1)*19360], dense intermediate columns and 8 attention heads.
Embedding requires masked local lookup + all-reduce; logits require all-gather.
This is the weight contract, not an implementation of the whole-model engine.
"""
from dataclasses import dataclass
from pathlib import Path
import json
import torch
from safetensors import safe_open
from safetensors.torch import load_file
from model.glm53_layer_weights import Source, load_attention, load_indexer
from ops.moe_layer import MoELayerWeights as MoE

TP, D, N_LAYER, VOCAB = 8, 6144, 78, 154880
DENSE_LAYERS = frozenset({0, 1, 2})
SPECIAL = frozenset()  # all routed experts use the same INT4 ABI
SOURCE = '/mnt/data2/kw/GLM-5.3-UNCENSORED-FP8'
INT4 = '/mnt/data2/kw/glm53_int4_tp8'

Attn = dict[str, torch.Tensor]

@dataclass
class DenseFFN:
    gu: torch.Tensor
    down: torch.Tensor

@dataclass
class Layer:
    idx: int
    attn: dict
    indexer: dict | None
    ffn_norm: torch.Tensor
    ffn: DenseFFN | MoE

@dataclass
class Weights:
    rank: int
    embed: torch.Tensor
    final_norm: torch.Tensor
    lm_head: torch.Tensor
    layers: list[Layer]
    vocab_start: int
    vocab_end: int


def check_config(source):
    cfg = json.loads((Path(source)/'config.json').read_text())
    expected = dict(num_hidden_layers=78, hidden_size=6144, vocab_size=154880,
                    first_k_dense_replace=3, intermediate_size=12288,
                    moe_intermediate_size=2048, n_routed_experts=256,
                    num_attention_heads=64, q_lora_rank=2048, kv_lora_rank=512)
    for key,value in expected.items():
        if cfg.get(key) != value: raise ValueError((key,cfg.get(key),value))
    return cfg


def build_globals(source, rank):
    check_config(source)
    index=json.loads((Path(source)/'model.safetensors.index.json').read_text())['weight_map']
    result={}
    for target,name in [('embed','model.embed_tokens.weight'),
                        ('final_norm','model.norm.weight'),('lm_head','lm_head.weight')]:
        with safe_open(str(Path(source)/index[name]),framework='pt') as f:
            t=f.get_tensor(name) if target=='final_norm' else f.get_slice(name)[rank*19360:(rank+1)*19360,:]
            result[target]=t.contiguous().clone()
    return result


def build_layer(source, int4, layer, rank, device):
    """Cold path only; source dequant/layout is never executed on a cache hit."""
    a=load_attention(source,layer,rank,device)
    result={'attn.'+k:v.detach().cpu().contiguous() for k,v in a.items()}
    del a
    indexer=load_indexer(source,layer,rank,device)
    if indexer is not None:
        result.update({'indexer.'+k:v.detach().cpu().contiguous() for k,v in indexer.items()})
    del indexer
    if layer<3:
        s=Source(source,layer);sl=slice(rank*1536,(rank+1)*1536)
        result['ffn_norm']=s.get('post_attention_layernorm.weight').half()
        g=s.matrix('mlp.gate_proj',device,rows=sl)
        u=s.matrix('mlp.up_proj',device,rows=sl)
        result['dense.gu']=torch.cat((g,u)).bfloat16().cpu().contiguous()
        del g,u
        result['dense.down']=s.matrix('mlp.down_proj',device,cols=sl).bfloat16().cpu().contiguous()
    else:
        path=Path(int4)/f'layer{layer}_g64_rank{rank}.safetensors'
        data=load_file(str(path),device='cpu')
        result['ffn_norm']=data['ffn_norm']
        names=dict(router='router',bias='bias',gu='gu',gu_scale='gu_scale',
                   down='down_int4',down_scale='down_int4_scale',
                   shared_gu='shared_gu',shared_down='shared_down')
        result.update({'moe.'+k:data[v] for k,v in names.items()})
    return result


def unpack_layer(data, layer):
    def part(prefix): return {k[len(prefix):]:v for k,v in data.items() if k.startswith(prefix)}
    return Layer(layer,part('attn.'),part('indexer.') or None,data['ffn_norm'],
                 DenseFFN(**part('dense.')) if layer<3 else MoE(**part('moe.')))


def load_tp8(model_path=SOURCE, int4_path=INT4, *, rank, device=None, cache_dir=None):
    """One process/rank; first call materializes shm, subsequent calls only H2D."""
    from model.wcache import load
    return load(source=model_path,int4=int4_path,rank=rank,device=device,cache_dir=cache_dir)
