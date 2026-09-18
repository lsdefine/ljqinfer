"""Safetensors loader and TP4 sharding for Qwen3.8-27B-W8A8.

Checkpoint tensors are kept in their native output-major layout. QuantLinear
owns the transpose needed by the NPU dynamic W8A8 matmul. Loading is lazy per
layer so correctness probes do not need to allocate the complete model.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json
import torch
from .config import CONFIG, MODEL_DIR

@dataclass
class QuantLinear:
    weight: Any
    scale: Any
    offset: Any | None = None
    shard_axis: str = "replicated"
    # NPU quant_matmul consumes KxN.  Materialize it once at load time rather
    # than transpose+contiguous on every eager/graph invocation.
    native_weight: Any | None = None

@dataclass
class DenseLayerWeights:
    gate: QuantLinear | None
    up: QuantLinear | None
    down: QuantLinear
    gate_up: QuantLinear | None = None

@dataclass
class FullAttentionWeights:
    q: QuantLinear | None
    k: QuantLinear | None
    v: QuantLinear | None
    o: QuantLinear
    q_norm: Any
    k_norm: Any
    qkv: QuantLinear | None = None

@dataclass
class GDNWeights:
    qkv: QuantLinear | None
    z: QuantLinear | None
    ab: Any
    conv: Any
    A_log: Any
    dt_bias: Any
    norm: Any
    out: Any
    qkvz: QuantLinear | None = None
    # Custom decode kernel consumes KxC contiguous weights; materialize once.
    conv_kc: Any | None = None

@dataclass
class LayerWeights:
    input_norm: Any
    post_norm: Any
    attention: FullAttentionWeights | GDNWeights
    mlp: DenseLayerWeights

@dataclass
class MTPWeights:
    pre_fc_norm_embedding: Any
    pre_fc_norm_hidden: Any
    fc: Any
    layer: LayerWeights
    norm: Any

class SafeTensorStore:
    def __init__(self, model_dir: Path | str = MODEL_DIR, device: str = "cpu"):
        self.model_dir = Path(model_dir)
        self.device = device
        index = json.loads((self.model_dir / "quant_model_weights.safetensors.index.json").read_text())
        self.weight_map: dict[str, str] = index["weight_map"]
        self._handles: dict[str, Any] = {}

    def _handle(self, filename: str):
        if filename not in self._handles:
            from safetensors import safe_open
            self._handles[filename] = safe_open(str(self.model_dir / filename), framework="pt", device=self.device)
        return self._handles[filename]

    def get(self, name: str):
        try:
            filename = self.weight_map[name]
        except KeyError as exc:
            raise KeyError(f"missing checkpoint tensor {name}") from exc
        return self._handle(filename).get_tensor(name)

    def has(self, name: str) -> bool:
        return name in self.weight_map

class Weights:
    def __init__(self, rank: int, model_dir: Path | str = MODEL_DIR, device: str = "cpu"):
        if not 0 <= rank < CONFIG.tp:
            raise ValueError("rank must be 0..3")
        self.rank = rank
        self.store = SafeTensorStore(model_dir, device=device)
        self.device = device
        self._layers: dict[int, LayerWeights] = {}
        self._embedding = None
        self._final_norm = None
        self._lm_head = None
        self._mtp = None

    @classmethod
    def mock(cls, rank: int = 0):
        obj = object.__new__(cls)
        obj.rank, obj.store, obj.device = rank, None, "mock"
        obj._layers, obj._embedding, obj._final_norm, obj._lm_head = {}, None, None, None
        obj._mtp = None
        return obj

    @property
    def loaded(self) -> bool:
        return self.store is not None

    def _to(self, x):
        return x if self.device == "cpu" else x.to(self.device, non_blocking=False)

    def _native_quant_weight(self, weight):
        """Materialize the KxN W8A8 weight once in the NPU QMM native format."""
        if self.device == "cpu":
            return None
        import torch_npu
        native = weight.transpose(0, 1).contiguous()
        return torch_npu.npu_format_cast(native, 29)  # FRACTAL_NZ

    def _slice(self, x, dim: int, parts: int | None = None):
        parts = parts or CONFIG.tp
        if x.shape[dim] % parts:
            raise ValueError(f"cannot TP-shard shape={tuple(x.shape)} dim={dim} parts={parts}")
        # ``chunk(...).contiguous()`` may return the original contiguous view
        # unchanged (notably [N, 1] scales and dim-0 weight shards), retaining
        # the full checkpoint storage. Older NPU QMM reads the physical storage
        # extent for scale metadata, so each TP shard must own compact storage.
        return x.chunk(parts, dim=dim)[self.rank].clone().contiguous()

    def tensor(self, name: str, *, shard_dim: int | None = None):
        x = self.store.get(name)
        if shard_dim is not None:
            x = self._slice(x, shard_dim)
        return self._to(x)

    def quant_linear(self, name: str, *, shard_dim: int | None = None) -> QuantLinear:
        w = self.tensor(name + ".weight", shard_dim=shard_dim)
        # Per-output-channel scales/offsets follow output sharding only.
        scale_dim = 0 if shard_dim == 0 else None
        s = self.tensor(name + ".weight_scale", shard_dim=scale_dim).reshape(-1)
        offset_name = name + ".weight_offset"
        # This checkpoint stores 440 explicit but identically-zero offsets.
        # Older torch-npu rejects offset with BF16 output, so normalize an exact
        # zero offset to its mathematically equivalent absent representation.
        o = None
        if self.store.has(offset_name):
            raw_offset = self.store.get(offset_name)
            if bool(raw_offset.count_nonzero()):
                o = self.tensor(offset_name, shard_dim=scale_dim).reshape(-1)
        axis = "column" if shard_dim == 0 else "row" if shard_dim == 1 else "replicated"
        native = self._native_quant_weight(w)
        return QuantLinear(w, s, o, axis, native)

    def segmented_tensor(self, name: str, sizes: tuple[int, ...]):
        """Shard each packed output segment independently, then concatenate.

        GDN qkv is checkpointed as [all-Q, all-K, all-V]. A plain four-way
        chunk would mix those regions and is therefore numerically invalid.
        """
        import torch
        x = self.store.get(name)
        if sum(sizes) != x.shape[0]:
            raise ValueError(f"packed sizes {sizes} do not match {name} {tuple(x.shape)}")
        pieces = [part.chunk(CONFIG.tp, dim=0)[self.rank].contiguous()
                  for part in torch.split(x, sizes, dim=0)]
        return self._to(torch.cat(pieces, dim=0))

    def segmented_quant_linear(self, name: str, sizes: tuple[int, ...]) -> QuantLinear:
        w = self.segmented_tensor(name + ".weight", sizes)
        s = self.segmented_tensor(name + ".weight_scale", sizes).reshape(-1)
        offset_name = name + ".weight_offset"
        o = None
        if self.store.has(offset_name):
            raw_offset = self.store.get(offset_name)
            if bool(raw_offset.count_nonzero()):
                o = self.segmented_tensor(offset_name, sizes).reshape(-1)
        native = self._native_quant_weight(w)
        return QuantLinear(w, s, o, "column", native)

    def pack_quant_linears(self, *parts: QuantLinear) -> QuantLinear:
        """Concatenate same-input output projections into one native QMM."""
        import torch
        if not parts:
            raise ValueError("at least one projection is required")
        if len({p.weight.shape[1] for p in parts}) != 1:
            raise ValueError("packed projections must share input width")
        weight = torch.cat([p.weight for p in parts], dim=0).contiguous()
        scale = torch.cat([p.scale for p in parts], dim=0).contiguous()
        offset = None
        if any(p.offset is not None for p in parts):
            offset = torch.cat([
                p.offset if p.offset is not None else torch.zeros_like(p.scale)
                for p in parts], dim=0).contiguous()
        native = self._native_quant_weight(weight)
        return QuantLinear(weight, scale, offset, "column", native)

    @property
    def embedding(self):
        if self._embedding is None:
            self._embedding = self.tensor("model.language_model.embed_tokens.weight")
        return self._embedding

    @property
    def final_norm(self):
        if self._final_norm is None:
            self._final_norm = self.tensor("model.language_model.norm.weight")
        return self._final_norm

    @property
    def lm_head(self):
        """Output-vocabulary shard [vocab/TP, hidden]."""
        if self._lm_head is None:
            self._lm_head = self.tensor("lm_head.weight", shard_dim=0)
        return self._lm_head

    @property
    def vocab_start(self) -> int:
        return self.rank * (CONFIG.vocab_size // CONFIG.tp)

    @property
    def vocab_end(self) -> int:
        return (self.rank + 1) * (CONFIG.vocab_size // CONFIG.tp)

    def layer(self, idx: int) -> LayerWeights:
        if idx in self._layers:
            return self._layers[idx]
        p = f"model.language_model.layers.{idx}"
        gate_proj = self.quant_linear(p + ".mlp.gate_proj", shard_dim=0)
        up_proj = self.quant_linear(p + ".mlp.up_proj", shard_dim=0)
        mlp = DenseLayerWeights(
            None, None, self.quant_linear(p + ".mlp.down_proj", shard_dim=1),
            self.pack_quant_linears(gate_proj, up_proj),
        )
        if idx in CONFIG.full_attention_layers:
            # q_proj is per-head [q, gate] packed; a contiguous head shard keeps
            # each local q paired with its output gate.
            q_proj = self.quant_linear(p + ".self_attn.q_proj", shard_dim=0)
            k_proj = self.quant_linear(p + ".self_attn.k_proj", shard_dim=0)
            v_proj = self.quant_linear(p + ".self_attn.v_proj", shard_dim=0)
            attn = FullAttentionWeights(
                None, None, None,
                self.quant_linear(p + ".self_attn.o_proj", shard_dim=1),
                self.tensor(p + ".self_attn.q_norm.weight"),
                self.tensor(p + ".self_attn.k_norm.weight"),
                self.pack_quant_linears(q_proj, k_proj, v_proj),
            )
        else:
            # qkv/z are column parallel; a/b and BF16 output projection are
            # explicitly sliced to the same local GDN head ranges.
            packed = (CONFIG.linear_num_key_heads * CONFIG.linear_key_head_dim,
                      CONFIG.linear_num_key_heads * CONFIG.linear_key_head_dim,
                      CONFIG.linear_num_value_heads * CONFIG.linear_value_head_dim)
            qkv_proj = self.segmented_quant_linear(
                p + ".linear_attn.in_proj_qkv", packed)
            z_proj = self.quant_linear(p + ".linear_attn.in_proj_z", shard_dim=0)
            ab = torch.cat((
                self.tensor(p + ".linear_attn.in_proj_a.weight", shard_dim=0),
                self.tensor(p + ".linear_attn.in_proj_b.weight", shard_dim=0),
            ), dim=0)
            attn = GDNWeights(
                None, None, ab,
                self.segmented_tensor(p + ".linear_attn.conv1d.weight", packed),
                self.tensor(p + ".linear_attn.A_log", shard_dim=0),
                self.tensor(p + ".linear_attn.dt_bias", shard_dim=0),
                self.tensor(p + ".linear_attn.norm.weight"),
                self.tensor(p + ".linear_attn.out_proj.weight", shard_dim=1),
                self.pack_quant_linears(qkv_proj, z_proj),
            )
            attn.conv_kc = attn.conv.reshape(attn.conv.shape[0], -1).transpose(0, 1).contiguous()
        out = LayerWeights(
            self.tensor(p + ".input_layernorm.weight"),
            self.tensor(p + ".post_attention_layernorm.weight"), attn, mlp)
        self._layers[idx] = out
        return out

    @property
    def mtp(self) -> MTPWeights:
        if self._mtp is not None:
            return self._mtp
        p = "mtp.layers.0"
        q_proj = self.quant_linear(p + ".self_attn.q_proj", shard_dim=0)
        k_proj = self.quant_linear(p + ".self_attn.k_proj", shard_dim=0)
        v_proj = self.quant_linear(p + ".self_attn.v_proj", shard_dim=0)
        attn = FullAttentionWeights(
            None, None, None,
            self.quant_linear(p + ".self_attn.o_proj", shard_dim=1),
            self.tensor(p + ".self_attn.q_norm.weight"),
            self.tensor(p + ".self_attn.k_norm.weight"),
            self.pack_quant_linears(q_proj, k_proj, v_proj),
        )
        gate_proj = self.quant_linear(p + ".mlp.gate_proj", shard_dim=0)
        up_proj = self.quant_linear(p + ".mlp.up_proj", shard_dim=0)
        mlp = DenseLayerWeights(
            None, None, self.quant_linear(p + ".mlp.down_proj", shard_dim=1),
            self.pack_quant_linears(gate_proj, up_proj),
        )
        layer = LayerWeights(
            self.tensor(p + ".input_layernorm.weight"),
            self.tensor(p + ".post_attention_layernorm.weight"), attn, mlp)
        # MTP fc is BF16 [H,2H]. Column-shard its output and all-gather the
        # resulting hidden shards in the plugin; this avoids replicated GEMM.
        fc = self.tensor("mtp.fc.weight", shard_dim=0)
        self._mtp = MTPWeights(
            self.tensor("mtp.pre_fc_norm_embedding.weight"),
            self.tensor("mtp.pre_fc_norm_hidden.weight"), fc, layer,
            self.tensor("mtp.norm.weight"))
        return self._mtp

    def clear_layer(self, idx: int) -> None:
        self._layers.pop(idx, None)
