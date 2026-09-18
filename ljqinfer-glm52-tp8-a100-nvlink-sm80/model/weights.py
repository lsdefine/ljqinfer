#!/usr/bin/env python3
"""TP8 weight loader for GLM-5.2 (this exact GGUF only).

Single process, 8 GPUs. Every weight is loaded as *raw quantized bytes* (the
kernels in the .so dequant on the fly) except norms / router / eh_proj which are
tiny and kept as fp16/fp32. There is deliberately NO qtype/split/config metadata
in the returned structure: the split rule for every tensor in this model is
uniform ("shard some dim into 8, or replicate"), and which dequant kernel to use
is decided in model.py from the layer index (it is hard-wired to this model).

    w = load_tp8(model_path)                 # -> Weights
    w.embed[r]                               # Q6_K byte shard on cuda:r
    w.layers[i].attn.q_b[r]                  # raw Q8_0 byte shard on cuda:r
    w.layers[i].ffn                          # DenseFFN | MoE
    w.final_norm                            # fp16 vec on cuda:0
    w.lm_head[r]                             # Q6_K byte shard (None if load_head=False)
    w.mtp                                    # MTP | None (layer 78 speculative head)
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch
from gguf import GGUFReader, quants

# ---------------------------------------------------------------- geometry ----
TP = 8
DEVICES = tuple(torch.device(f'cuda:{i}') for i in range(TP))
D = 6144
N_LAYER = 78                         # transformer blocks 0..77
MTP_LAYER = 78                       # speculative (next-token) block
VOCAB = 154880
EMB_BLOCK = 256                      # Q6_K super-block
EMB_BYTES = 210                      # Q6_K bytes per super-block
DENSE_LAYERS = frozenset({0, 1, 2})  # blocks with a plain FFN
SPECIAL = frozenset({8, 75, 76, 77, 78})  # MoE blocks with qK experts (see model.py _CFG)

Shard = list  # length-TP list of torch.Tensor, element r lives on cuda:r


# ------------------------------------------------------------ gguf access -----
class TensorSource:
    """Thin read-only view over the shard .gguf files: source[name] -> tensor."""
    def __init__(self, path):
        p = Path(path)
        files = sorted(p.glob('*.gguf')) if p.is_dir() else [p]
        if not files:
            raise FileNotFoundError(path)
        self.readers = [GGUFReader(str(x), 'r') for x in files]
        self.by = {}
        for r in self.readers:
            for t in r.tensors:
                if t.name in self.by:
                    raise ValueError(f'duplicate tensor: {t.name}')
                self.by[t.name] = t

    def __getitem__(self, name):
        return self.by[name]

# ------------------------------------------------------------- primitives -----
def _np(t):
    """Raw quantized bytes of a gguf tensor as a numpy array (logical shape)."""
    return np.asarray(t.data)


def replicate(src, name, *, fp16=False, fp32=False):
    """Full tensor copied to every device (norms / router / bias)."""
    a = _np(src[name])
    x = torch.from_numpy(np.array(a, copy=True))
    if fp16:
        x = x.to(torch.float16)
    elif fp32:
        x = x.to(torch.float32)
    return [x.to(d).contiguous() for d in DEVICES]


def shard(src, name, dim):
    """Tensor split into TP even chunks along `dim`; chunk r -> cuda:r.

    Works for every weight in this model: rows (q_b/gate/up/*_shexp gate/up),
    byte columns (o/down/down_shexp), expert dims (gate_exps/up_exps/down_exps),
    heads (k_b/v_b) and Q6_K super-block columns (embed/head) all reduce to
    "shape[dim] // 8".
    """
    a = _np(src[name])
    n = a.shape[dim] // TP
    if a.shape[dim] % TP:
        raise ValueError(f'{name}: dim {dim}={a.shape[dim]} not divisible by {TP}')
    sl = [slice(None)] * a.ndim
    out = []
    for r, d in enumerate(DEVICES):
        sl[dim] = slice(r * n, (r + 1) * n)
        out.append(torch.from_numpy(a[tuple(sl)].copy()).to(d))
    return out


# --------------------------------------------------------------- structs ------
@dataclass
class Attn:
    norm: Shard          # fp16, replicated
    q_a: Shard           # Q8_0 bytes, replicated
    q_a_norm: Shard      # fp16, replicated
    q_b: Shard           # Q8_0 bytes, row-shard (dim0)
    kv_a: Shard          # Q8_0 bytes, replicated
    kv_a_norm: Shard     # fp16, replicated
    k_b: Shard           # Q8_0 bytes, head-shard (dim0)
    v_b: Shard           # Q8_0 bytes, head-shard (dim0)
    o: Shard             # Q8_0 bytes, byte-col-shard (dim1) -> allreduce


@dataclass
class DenseFFN:
    norm: Shard          # fp16, replicated
    gate: Shard          # Q8_0 bytes, row-shard (dim0)
    up: Shard            # Q8_0 bytes, row-shard (dim0)
    down: Shard          # Q8_0 bytes, byte-col-shard (dim1) -> allreduce


@dataclass
class MoE:
    norm: Shard          # fp16, replicated
    router: Shard        # fp16, replicated  (ffn_gate_inp)
    bias: Shard          # fp32, replicated  (exp_probs_b.bias)
    gate_exps: Shard     # expert bytes, shard expert-inter dim (dim1)
    up_exps: Shard       # expert bytes, shard expert-inter dim (dim1)
    down_exps: Shard     # expert bytes, shard byte dim (dim2) -> allreduce
    gate_shexp: Shard    # Q8_0 bytes, row-shard (dim0)
    up_shexp: Shard      # Q8_0 bytes, row-shard (dim0)
    down_shexp: Shard    # Q8_0 bytes, byte-col-shard (dim1) -> allreduce


@dataclass
class Layer:
    idx: int
    attn: Attn
    ffn: object          # DenseFFN | MoE


@dataclass
class MTP:
    enorm: torch.Tensor              # fp16 vec, cuda:0
    hnorm: torch.Tensor              # fp16 vec, cuda:0
    shared_head_norm: torch.Tensor   # fp16 vec, cuda:0
    eh_proj: torch.Tensor            # fp16 (D, 2D), cuda:0
    block: Layer                     # blk.78 (attn + special MoE)


@dataclass
class Weights:
    embed: Shard                     # Q6_K byte shard (token_embd)
    layers: list                     # list[Layer], len N_LAYER
    final_norm: torch.Tensor         # fp16 vec, cuda:0 (output_norm)
    lm_head: Shard                   # Q6_K byte shard (output) or None
    mtp: MTP                         # or None


# ---------------------------------------------------------------- loaders -----
def _attn(src, layer) -> Attn:
    p = f'blk.{layer}.'
    return Attn(
        norm=replicate(src, p + 'attn_norm.weight', fp16=True),
        q_a=replicate(src, p + 'attn_q_a.weight'),
        q_a_norm=replicate(src, p + 'attn_q_a_norm.weight', fp16=True),
        q_b=shard(src, p + 'attn_q_b.weight', 0),
        kv_a=replicate(src, p + 'attn_kv_a_mqa.weight'),
        kv_a_norm=replicate(src, p + 'attn_kv_a_norm.weight', fp16=True),
        k_b=shard(src, p + 'attn_k_b.weight', 0),
        v_b=shard(src, p + 'attn_v_b.weight', 0),
        o=shard(src, p + 'attn_output.weight', 1),
    )


def _dense(src, layer) -> DenseFFN:
    p = f'blk.{layer}.'
    return DenseFFN(
        norm=replicate(src, p + 'ffn_norm.weight', fp16=True),
        gate=shard(src, p + 'ffn_gate.weight', 0),
        up=shard(src, p + 'ffn_up.weight', 0),
        down=shard(src, p + 'ffn_down.weight', 1),
    )


def _moe(src, layer) -> MoE:
    p = f'blk.{layer}.'
    return MoE(
        norm=replicate(src, p + 'ffn_norm.weight', fp16=True),
        router=replicate(src, p + 'ffn_gate_inp.weight', fp16=True),
        bias=replicate(src, p + 'exp_probs_b.bias', fp32=True),
        gate_exps=shard(src, p + 'ffn_gate_exps.weight', 1),
        up_exps=shard(src, p + 'ffn_up_exps.weight', 1),
        down_exps=shard(src, p + 'ffn_down_exps.weight', 2),
        gate_shexp=shard(src, p + 'ffn_gate_shexp.weight', 0),
        up_shexp=shard(src, p + 'ffn_up_shexp.weight', 0),
        down_shexp=shard(src, p + 'ffn_down_shexp.weight', 1),
    )


def _layer(src, layer) -> Layer:
    ffn = _dense(src, layer) if layer in DENSE_LAYERS else _moe(src, layer)
    return Layer(idx=layer, attn=_attn(src, layer), ffn=ffn)


def _embed_shard(src, name) -> Shard:
    """Q6_K token_embd/output: reshape flat bytes to (V, D/256, 210), shard dim1."""
    t = src[name]
    if int(t.tensor_type) != 14 or tuple(map(int, t.shape)) != (D, VOCAB):
        raise ValueError(f'bad {name}: type={t.tensor_type} shape={t.shape}')
    raw = _np(t).reshape(VOCAB, D // EMB_BLOCK, EMB_BYTES)
    n = (D // EMB_BLOCK) // TP
    return [torch.from_numpy(raw[:, r * n:(r + 1) * n].copy()).to(d)
            for r, d in enumerate(DEVICES)]


def _vec0(src, name) -> torch.Tensor:
    """F32 norm vector -> fp16 on cuda:0."""
    t = src[name]
    if int(t.tensor_type) != 0:
        raise ValueError(f'{name} type={t.tensor_type}')
    return torch.from_numpy(np.array(t.data, copy=True)).to(DEVICES[0], dtype=torch.float16)


def _mtp(src) -> MTP:
    p = f'blk.{MTP_LAYER}.nextn.'
    t = src[p + 'eh_proj.weight']
    if int(t.tensor_type) != 8:
        raise ValueError(f'eh_proj type={t.tensor_type}')
    w = np.asarray(quants.dequantize(t.data, t.tensor_type), dtype=np.float32)
    if w.shape == (2 * D, D):
        w = w.T
    if w.shape != (D, 2 * D):
        raise ValueError(f'eh_proj dequant shape {w.shape}, expected {(D, 2 * D)}')
    return MTP(
        enorm=_vec0(src, p + 'enorm.weight'),
        hnorm=_vec0(src, p + 'hnorm.weight'),
        shared_head_norm=_vec0(src, p + 'shared_head_norm.weight'),
        eh_proj=torch.from_numpy(w).to(DEVICES[0], dtype=torch.float16).contiguous(),
        block=_layer(src, MTP_LAYER),
    )


def load_tp8(model_path, *, load_head=True, load_mtp=True, progress=False) -> Weights:
    """Load all weights of the GLM-5.2 GGUF at `model_path` sharded over 8 GPUs."""
    import time
    src = TensorSource(model_path)
    t0 = time.perf_counter()
    layers = []
    for i in range(N_LAYER):
        layers.append(_layer(src, i))
        if progress:
            print(f'LOAD blk.{i} t={time.perf_counter() - t0:.1f}s', flush=True)
    w = Weights(
        embed=_embed_shard(src, 'token_embd.weight'),
        layers=layers,
        final_norm=_vec0(src, 'output_norm.weight'),
        lm_head=_embed_shard(src, 'output.weight') if load_head else None,
        mtp=_mtp(src) if load_mtp else None,
    )
    if progress:
        print(f'LOAD done t={time.perf_counter() - t0:.1f}s', flush=True)
    return w


if __name__ == '__main__':
    import sys
    mp = sys.argv[1] if len(sys.argv) > 1 else \
        '/mnt/data/kw/models/huihui-ai/Huihui-GLM-5.2-abliterated-GGUF/UD-Q3_K_M'
    w = load_tp8(mp, progress=True)
    print('layers', len(w.layers))
    print('embed', [tuple(x.shape) for x in w.embed[:1]], w.embed[0].dtype)
    a = w.layers[3].attn
    print('attn.q_b', [tuple(x.shape) for x in a.q_b[:1]])
    print('ffn type', type(w.layers[3].ffn).__name__)
    print('mtp.eh_proj', tuple(w.mtp.eh_proj.shape))
