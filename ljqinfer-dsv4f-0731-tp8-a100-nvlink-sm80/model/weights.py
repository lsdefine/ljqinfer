"""DeepSeek-V4-Flash TP8 weight structures and startup loaders.

This module follows the reference engine contract:

* :class:`WeightCache` is a host-side startup source backed by tmpfs.
* :func:`load_tp8` builds a small structured :class:`Weights` tree whose tensor
  leaves live on the eight GPUs.
* Runtime code consumes that tree directly; it must never call ``tensor()`` or
  perform H2D copies on the hot path.

The routed experts are packed into contiguous per-layer banks while loading.
That matches the reference MoE layout and gives fused kernels one stable tensor
per weight/scale bank instead of 256 Python/CUDA tensor objects.
"""
from __future__ import annotations

import json
import mmap
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch

from model.config import CACHE_DIR, CACHE_FORMAT, N_EXPERT, N_LAYER, N_MTP, TP

TORCH_DTYPES = {
    "BOOL": torch.bool, "U8": torch.uint8, "I8": torch.int8,
    "I16": torch.int16, "F16": torch.float16, "BF16": torch.bfloat16,
    "I32": torch.int32, "F32": torch.float32, "I64": torch.int64,
    "F64": torch.float64, "F8_E4M3": torch.float8_e4m3fn,
    "F8_E8M0": torch.float8_e8m0fnu,
}

Shard = list[torch.Tensor]


@dataclass(frozen=True)
class WeightInfo:
    name: str
    dtype: str
    shape: tuple[int, ...]
    layout: str
    split_dim: int | None


class WeightCache:
    """Read-only host views over the existing shared-memory source snapshot."""

    def __init__(self, cache_dir: Path = CACHE_DIR):
        self.cache_dir = Path(cache_dir)
        path = self.cache_dir / "manifest.json"
        self.manifest = json.loads(path.read_text())
        if self.manifest.get("format") != CACHE_FORMAT:
            raise RuntimeError(
                f"cache format {self.manifest.get('format')!r} != {CACHE_FORMAT!r}")
        self._file = (self.cache_dir / "weights.blob").open("rb")
        self._mmap = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)

    def close(self) -> None:
        self._mmap.close()
        self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def names(self):
        return self.manifest["entries"].keys()

    def info(self, name: str) -> WeightInfo:
        e = self.manifest["entries"][name]
        return WeightInfo(name, e["dtype"], tuple(e["shape"]),
                          e["layout"], e.get("split_dim"))

    def tensor(self, name: str, rank: int | None = None) -> torch.Tensor:
        e = self.manifest["entries"][name]
        if e["layout"] == "replica":
            if rank is not None and not 0 <= rank < TP:
                raise ValueError(f"rank must be 0..{TP-1}")
            offset, nbytes, shape = e["offset"], e["nbytes"], e["shape"]
        else:
            if rank is None:
                raise ValueError(f"TP tensor {name} requires rank")
            if not 0 <= rank < TP:
                raise ValueError(f"rank must be 0..{TP-1}")
            s = e["shards"][rank]
            offset, nbytes, shape = s["offset"], s["nbytes"], s["shape"]
        view = memoryview(self._mmap)[offset:offset + nbytes]
        return torch.frombuffer(view, dtype=TORCH_DTYPES[e["dtype"]]).reshape(shape)


# ------------------------------------------------------------ final tree -----
@dataclass
class QuantPair:
    weight: Shard
    scale: Shard


@dataclass
class Compressor:
    ape: Shard
    norm: Shard
    wgate: Shard
    wkv: Shard


@dataclass
class Indexer:
    compressor: Compressor
    wq_b: QuantPair
    weights_proj: Shard


@dataclass
class Attention:
    q_norm: Shard
    kv_norm: Shard
    wq_a: QuantPair
    wq_b: QuantPair
    wkv: QuantPair
    wo_a: QuantPair
    wo_b: QuantPair
    attn_sink: Shard
    compressor: Optional[Compressor]
    indexer: Optional[Indexer]


@dataclass
class ExpertBank:
    w1: QuantPair
    w2: QuantPair
    w3: QuantPair


@dataclass
class Router:
    weight: Shard
    bias: Optional[Shard]
    tid2eid: Optional[Shard]


@dataclass
class MoE:
    gate: Router
    experts: ExpertBank
    shared_experts: ExpertBank


@dataclass
class Layer:
    idx: int
    attn_norm: Shard
    attn: Attention
    ffn_norm: Shard
    ffn: MoE
    hc_attn_fn: Shard
    hc_attn_scale: Shard
    hc_attn_base: Shard
    hc_ffn_fn: Shard
    hc_ffn_scale: Shard
    hc_ffn_base: Shard


@dataclass
class MTP:
    idx: int
    block: Layer
    main_norm: Optional[Shard] = None
    main_proj: Optional[QuantPair] = None
    confidence_proj: Optional[Shard] = None
    markov_w1: Optional[Shard] = None
    markov_w2: Optional[Shard] = None
    norm: Optional[Shard] = None
    hc_head_fn: Optional[Shard] = None
    hc_head_scale: Optional[Shard] = None
    hc_head_base: Optional[Shard] = None


@dataclass
class Weights:
    embed: Shard
    layers: list[Layer]
    norm: Shard
    head: Shard
    hc_head_fn: Shard
    hc_head_scale: Shard
    hc_head_base: Shard
    mtp: list[MTP]


# --------------------------------------------------------------- loaders -----
def _has(src: WeightCache, name: str) -> bool:
    return name in src.manifest["entries"]


def _copy_rank(src: WeightCache, name: str, rank: int) -> torch.Tensor:
    info = src.manifest["entries"][name]
    host = src.tensor(name, rank if info["layout"] == "tp" else None)
    with torch.cuda.device(rank):
        return host.to(device=f"cuda:{rank}", non_blocking=False).contiguous()


def _tensor(src: WeightCache, name: str) -> Shard:
    with ThreadPoolExecutor(max_workers=TP) as pool:
        return list(pool.map(lambda rank: _copy_rank(src, name, rank), range(TP)))


def _optional_tensor(src: WeightCache, name: str) -> Optional[Shard]:
    return _tensor(src, name) if _has(src, name) else None


def _quant(src: WeightCache, prefix: str) -> QuantPair:
    return QuantPair(_tensor(src, prefix + ".weight"),
                     _tensor(src, prefix + ".scale"))


def _optional_quant(src: WeightCache, prefix: str) -> Optional[QuantPair]:
    return _quant(src, prefix) if _has(src, prefix + ".weight") else None


def _expert_component(src: WeightCache, prefix: str, component: str,
                      suffix: str) -> Shard:
    """Pack 256 source shards into one contiguous expert bank per rank."""
    names = [f"{prefix}.experts.{eid}.{component}.{suffix}"
             for eid in range(N_EXPERT)]

    def load_rank(rank: int) -> torch.Tensor:
        host = torch.stack([src.tensor(name, rank) for name in names], dim=0)
        with torch.cuda.device(rank):
            return host.to(device=f"cuda:{rank}", non_blocking=False).contiguous()

    with ThreadPoolExecutor(max_workers=TP) as pool:
        return list(pool.map(load_rank, range(TP)))


def _expert_bank(src: WeightCache, prefix: str) -> ExpertBank:
    def pair(component: str) -> QuantPair:
        return QuantPair(
            _expert_component(src, prefix, component, "weight"),
            _expert_component(src, prefix, component, "scale"),
        )
    return ExpertBank(w1=pair("w1"), w2=pair("w2"), w3=pair("w3"))


def _shared_bank(src: WeightCache, prefix: str) -> ExpertBank:
    return ExpertBank(
        w1=_quant(src, prefix + ".shared_experts.w1"),
        w2=_quant(src, prefix + ".shared_experts.w2"),
        w3=_quant(src, prefix + ".shared_experts.w3"),
    )


def _compressor(src: WeightCache, prefix: str) -> Compressor:
    return Compressor(
        ape=_tensor(src, prefix + ".ape"),
        norm=_tensor(src, prefix + ".norm.weight"),
        wgate=_tensor(src, prefix + ".wgate.weight"),
        wkv=_tensor(src, prefix + ".wkv.weight"),
    )


def _attention(src: WeightCache, prefix: str) -> Attention:
    cp = prefix + ".compressor"
    ip = prefix + ".indexer"
    compressor = _compressor(src, cp) if _has(src, cp + ".ape") else None
    indexer = None
    if _has(src, ip + ".weights_proj.weight"):
        indexer = Indexer(
            compressor=_compressor(src, ip + ".compressor"),
            wq_b=_quant(src, ip + ".wq_b"),
            weights_proj=_tensor(src, ip + ".weights_proj.weight"),
        )
    return Attention(
        q_norm=_tensor(src, prefix + ".q_norm.weight"),
        kv_norm=_tensor(src, prefix + ".kv_norm.weight"),
        wq_a=_quant(src, prefix + ".wq_a"),
        wq_b=_quant(src, prefix + ".wq_b"),
        wkv=_quant(src, prefix + ".wkv"),
        wo_a=_quant(src, prefix + ".wo_a"),
        wo_b=_quant(src, prefix + ".wo_b"),
        attn_sink=_tensor(src, prefix + ".attn_sink"),
        compressor=compressor,
        indexer=indexer,
    )


def _layer(src: WeightCache, prefix: str, idx: int) -> Layer:
    fp = prefix + ".ffn"
    return Layer(
        idx=idx,
        attn_norm=_tensor(src, prefix + ".attn_norm.weight"),
        attn=_attention(src, prefix + ".attn"),
        ffn_norm=_tensor(src, prefix + ".ffn_norm.weight"),
        ffn=MoE(
            gate=Router(
                weight=_tensor(src, fp + ".gate.weight"),
                bias=_optional_tensor(src, fp + ".gate.bias"),
                tid2eid=_optional_tensor(src, fp + ".gate.tid2eid"),
            ),
            experts=_expert_bank(src, fp),
            shared_experts=_shared_bank(src, fp),
        ),
        hc_attn_fn=_tensor(src, prefix + ".hc_attn_fn"),
        hc_attn_scale=_tensor(src, prefix + ".hc_attn_scale"),
        hc_attn_base=_tensor(src, prefix + ".hc_attn_base"),
        hc_ffn_fn=_tensor(src, prefix + ".hc_ffn_fn"),
        hc_ffn_scale=_tensor(src, prefix + ".hc_ffn_scale"),
        hc_ffn_base=_tensor(src, prefix + ".hc_ffn_base"),
    )


def load_layer_tp8(src: WeightCache, layer: int) -> Layer:
    if not 0 <= layer < N_LAYER:
        raise ValueError(f"layer must be 0..{N_LAYER - 1}")
    return _layer(src, f"layers.{layer}", layer)


def _mtp(src: WeightCache, index: int) -> MTP:
    prefix = f"mtp.{index}"
    return MTP(
        idx=index,
        block=_layer(src, prefix, N_LAYER + index),
        main_norm=_optional_tensor(src, prefix + ".main_norm.weight"),
        main_proj=_optional_quant(src, prefix + ".main_proj"),
        confidence_proj=_optional_tensor(src, prefix + ".confidence_head.proj.weight"),
        markov_w1=_optional_tensor(src, prefix + ".markov_head.markov_w1.weight"),
        markov_w2=_optional_tensor(src, prefix + ".markov_head.markov_w2.weight"),
        norm=_optional_tensor(src, prefix + ".norm.weight"),
        hc_head_fn=_optional_tensor(src, prefix + ".hc_head_fn"),
        hc_head_scale=_optional_tensor(src, prefix + ".hc_head_scale"),
        hc_head_base=_optional_tensor(src, prefix + ".hc_head_base"),
    )


def load_tp8(cache_dir: Path = CACHE_DIR, *, load_mtp: bool = True,
             progress: bool = False) -> Weights:
    """Build the complete final device-layout tree on all eight GPUs."""
    started = time.perf_counter()
    src = WeightCache(cache_dir)
    try:
        layers = []
        for index in range(N_LAYER):
            layers.append(load_layer_tp8(src, index))
            if progress:
                print(f"LOAD layers.{index} t={time.perf_counter() - started:.1f}s",
                      flush=True)
        mtp = []
        if load_mtp:
            for index in range(N_MTP):
                mtp.append(_mtp(src, index))
                if progress:
                    print(f"LOAD mtp.{index} t={time.perf_counter() - started:.1f}s",
                          flush=True)
        result = Weights(
            embed=_tensor(src, "embed.weight"),
            layers=layers,
            norm=_tensor(src, "norm.weight"),
            head=_tensor(src, "head.weight"),
            hc_head_fn=_tensor(src, "hc_head_fn"),
            hc_head_scale=_tensor(src, "hc_head_scale"),
            hc_head_base=_tensor(src, "hc_head_base"),
            mtp=mtp,
        )
    finally:
        src.close()
    if progress:
        print(f"LOAD done t={time.perf_counter() - started:.1f}s", flush=True)
    return result
