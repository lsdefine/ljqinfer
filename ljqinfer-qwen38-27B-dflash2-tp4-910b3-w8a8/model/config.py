"""Validated Qwen3.5-27B (directory name Qwen3.8) TP4 geometry.

Only immutable model ABI belongs here. Runtime knobs live in EngineConfig so a
replacement MTP backend does not need to import model internals.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json

MODEL_DIR = Path("/data/models/Qwen3.8-27B-W8A8")
TP = 4
WORLD_SIZE = 4
DEFAULT_DEVICES = (4, 5, 6, 7)
KV_PAGE_SIZE = 2048
COLD_CHECKPOINT_INTERVAL = 1024
# Physical HBM KV capacity shared by all active sequences.  This is deliberately
# independent from the model's per-sequence position limit so batching can pack
# several contexts into one global page pool.
DEFAULT_MAX_CACHED_TOKENS = 800_000
DEFAULT_MAX_SEQUENCE_TOKENS = 262_144

@dataclass(frozen=True)
class QwenTP4Config:
    hidden_size: int = 5120
    vocab_size: int = 248320
    num_hidden_layers: int = 64
    num_attention_heads: int = 24
    num_key_value_heads: int = 4
    head_dim: int = 256
    intermediate_size: int = 17408
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10_000_000.0
    partial_rotary_factor: float = 0.25
    max_position_embeddings: int = 262144
    full_attention_interval: int = 4
    linear_num_key_heads: int = 16
    linear_num_value_heads: int = 48
    linear_key_head_dim: int = 128
    linear_value_head_dim: int = 128
    linear_conv_kernel_dim: int = 4
    mtp_num_hidden_layers: int = 1
    dtype: str = "bfloat16"
    gdn_ssm_dtype: str = "bfloat16"
    tp: int = TP

    @property
    def full_attention_layers(self) -> tuple[int, ...]:
        return tuple(range(3, self.num_hidden_layers, 4))

    @property
    def linear_attention_layers(self) -> tuple[int, ...]:
        full = set(self.full_attention_layers)
        return tuple(i for i in range(self.num_hidden_layers) if i not in full)

    @property
    def local_q_heads(self) -> int:
        return self.num_attention_heads // self.tp

    @property
    def local_kv_heads(self) -> int:
        return self.num_key_value_heads // self.tp

    @property
    def local_gdn_k_heads(self) -> int:
        return self.linear_num_key_heads // self.tp

    @property
    def local_gdn_v_heads(self) -> int:
        return self.linear_num_value_heads // self.tp

    @property
    def rotary_dim(self) -> int:
        return int(self.head_dim * self.partial_rotary_factor)

    @property
    def gdn_conv_dim(self) -> int:
        return (self.linear_num_key_heads * self.linear_key_head_dim * 2
                + self.linear_num_value_heads * self.linear_value_head_dim)

    @property
    def gdn_conv_state_shape(self) -> tuple[int, int]:
        return (self.linear_conv_kernel_dim - 1, self.gdn_conv_dim // self.tp)

    @property
    def gdn_recurrent_state_shape(self) -> tuple[int, int, int]:
        return (self.local_gdn_v_heads,
                self.linear_value_head_dim, self.linear_key_head_dim)

@dataclass(frozen=True)
class EngineConfig:
    model_dir: Path = MODEL_DIR
    devices: tuple[int, ...] = DEFAULT_DEVICES
    # Global physical page-pool capacity shared across sequences.
    max_cached_tokens: int = DEFAULT_MAX_CACHED_TOKENS
    # Independent logical/context limit for any one sequence.
    max_sequence_tokens: int = DEFAULT_MAX_SEQUENCE_TOKENS
    max_sequences: int = 8
    kv_page_size: int = KV_PAGE_SIZE
    cold_checkpoint_interval: int = COLD_CHECKPOINT_INTERVAL
    prefill_chunk_size: int = 12288
    decode_buckets: tuple[int, ...] = (1, 2, 4, 8)
    mtp_backend: str = "qwen_native"

    def __post_init__(self) -> None:
        if len(self.devices) != TP:
            raise ValueError(f"TP4 requires four devices, got {self.devices}")
        if self.max_cached_tokens <= 0 or self.max_sequence_tokens <= 0:
            raise ValueError("KV pool and sequence capacities must be positive")
        if self.max_sequence_tokens > self.max_cached_tokens:
            raise ValueError("one sequence cannot exceed the global KV pool")
        if self.max_sequence_tokens > CONFIG.max_position_embeddings:
            raise ValueError("sequence capacity exceeds model position limit")
        if self.cold_checkpoint_interval <= 0:
            raise ValueError("cold_checkpoint_interval must be positive")
        if self.prefill_chunk_size < 8192:
            raise ValueError("prefill chunks must be at least 8192 tokens")
        if self.max_sequences > max(self.decode_buckets):
            raise ValueError("largest decode bucket must cover max_sequences")

CONFIG = QwenTP4Config()

def validate_model_config(path: Path | str = MODEL_DIR / "config.json") -> dict:
    data = json.loads(Path(path).read_text())
    t = data["text_config"]
    checks = {
        "hidden_size": CONFIG.hidden_size,
        "vocab_size": CONFIG.vocab_size,
        "num_hidden_layers": CONFIG.num_hidden_layers,
        "num_attention_heads": CONFIG.num_attention_heads,
        "num_key_value_heads": CONFIG.num_key_value_heads,
        "head_dim": CONFIG.head_dim,
        "intermediate_size": CONFIG.intermediate_size,
        "linear_num_key_heads": CONFIG.linear_num_key_heads,
        "linear_num_value_heads": CONFIG.linear_num_value_heads,
        "linear_key_head_dim": CONFIG.linear_key_head_dim,
        "linear_value_head_dim": CONFIG.linear_value_head_dim,
        "linear_conv_kernel_dim": CONFIG.linear_conv_kernel_dim,
        "mtp_num_hidden_layers": CONFIG.mtp_num_hidden_layers,
    }
    bad = {k: (t.get(k), v) for k, v in checks.items() if t.get(k) != v}
    expected = ["full_attention" if i % 4 == 3 else "linear_attention"
                for i in range(CONFIG.num_hidden_layers)]
    if t.get("layer_types") != expected:
        bad["layer_types"] = (t.get("layer_types"), expected)
    if bad:
        raise ValueError(f"model config mismatch: {bad}")
    return data
