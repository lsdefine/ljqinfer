"""model 层门面: Engine + 权重专家视图 + 对 strategy/_impl 的 19 符号契约 re-export。

性能基线 (勿删): True-paged full-78-layer baseline, 8K append, 3-hot-run median TPS: 0+8K=2199.571; 8K+8K=2014.867; 32K+8K=1467.128; 48K+8K=1210.479.
"""

from __future__ import annotations

from typing import Optional

import time
import torch

from ops.kernels import K, load_kernels

from dataclasses import dataclass, field

from model.weights import MoE, Weights

from model.config import (CACHE_DIM, DEFAULT_PREFILL_CHUNK_TOKENS, EOS_DEFAULT,
                          EXECUTION_LEN, KV_PAGE_SIZE, N_LAYER, SPECIAL_CFG, TP)
from model.runtime import TPRuntime, KVCache, _nccl_init_all
from model.blocks import _rmsnorm
from model.prefill import lm_head, prefill, prefill_sequence_chunked
from model.decode_backend import (advance, base_hidden, ensure_mtp_graphs,
                          mtp_forward, retract)
from model.batch_decode import generate_mtp_batch

# Stable orchestration facade consumed by ModelExecution; ``prefill`` and
# ``Engine`` are also retained for diagnostics and weight-cache tooling.
__all__ = (
    "Engine", "KVCache", "KV_PAGE_SIZE", "DEFAULT_PREFILL_CHUNK_TOKENS",
    "N_LAYER", "TP", "CACHE_DIM", "EOS_DEFAULT", "_rmsnorm", "prefill",
    "prefill_sequence_chunked", "lm_head", "advance",
    "retract", "base_hidden", "mtp_forward",
    "ensure_mtp_graphs", "generate_mtp_batch",
)


def _moe_experts(x, gate_b, up_b, down_b, expert_ids, expert_w):
    """Ordinary routed experts (prefill), rank-local IQ3 gate/up + IQ4 down shards.

    dispatch_meta turns the per-token top-8 selection (expert_ids [T,8] int64,
    expert_w [T,8] fp32, already grouped-topk normalised * routed_scaling) into
    an expert-sorted CSR: tok [n] int64 (source rows), weight [n,1] fp32, offsets
    [257] int64 (CPU). moe_rank_forward_v10_scatter then runs the 3 variable-M
    grouped GEMMs (silu(gate)*up -> down) over the packed expert shards and does
    the fused weighted scatter back to [T, D] (rank-local partial, pre all-reduce)."""
    ei = expert_ids.to(torch.int64).contiguous()
    ew = expert_w.to(torch.float32).contiguous()
    tok, weight, offsets = K.iq.dispatch_meta(ei, ew)
    routed = K.iq.moe_rank_forward_v10_scatter(
        gate_b, up_b, down_b, x.contiguous(), tok, weight, offsets)
    return routed.to(dtype=x.dtype)


def _special_moe_experts(x, w: MoE, layer: int, expert_ids, expert_w):
    """Special routed experts (prefill), K/IQ-quant per SPECIAL_CFG[layer].

    Same dispatch_meta CSR as moe_experts. Body: special_moe_rank_forward_v5
    (gather once + batch-dequant active down + multi-stream gate/up).
    Signature (cpp): (x, tok, ww, off_cpu, gp, up, dp, gqt, dqt) -> [T,D]
    rank-local partial (pre all-reduce). Rank shard selected by x.device.index
    (single-process TP map: rank == cuda id).
    """
    gqt, dqt = SPECIAL_CFG[layer]
    r = x.device.index
    ei = expert_ids.to(torch.int64).contiguous()
    ew = expert_w.to(torch.float32).contiguous()
    tok, weight, offsets = K.iq.dispatch_meta(ei, ew)
    # The special IQ MMQ kernels consume float* activations, while the
    # surrounding Q8/residual path is fp16.  Keep that boundary local.
    x32 = x.to(torch.float32).contiguous()
    routed = K.special_ext.special_moe_rank_forward_v5(
        x32, tok, weight, offsets,
        w.gate_exps[r], w.up_exps[r], w.down_exps[r],
        int(gqt), int(dqt))
    return routed.to(dtype=x.dtype)


@dataclass
class Engine:
    rt: TPRuntime
    w: Weights
    kv: KVCache
    graphs: dict = field(default_factory=dict)   # Q -> B1 compatibility registry
    base_graphs: dict = field(default_factory=dict)  # (B,Q) -> resident Base graph
    mtp_kv: Optional[KVCache] = None             # 1-layer MLA cache for blk.78 draft
    mtp_graphs: dict = field(default_factory=dict)   # T -> captured MTP draft graph
    lm_head_fp16: List[torch.Tensor] = field(default_factory=list)
    graph_residency_frozen: bool = False  # startup-built graphs are immutable while serving
    # Base/MTP prefill share this chunk budget, keeping activation residency fixed.
    prefill_chunk_tokens: int = 12 * 1024

    @classmethod
    def load(cls, devices=None,
             prefill_chunk_tokens: int = DEFAULT_PREFILL_CHUNK_TOKENS) -> "Engine":
        devices = devices or list(range(TP))
        started = time.perf_counter()
        last = started

        def stage(name: str) -> None:
            nonlocal last
            now = time.perf_counter()
            print(f"[startup] Engine.load {name} step={now-last:.3f}s total={now-started:.3f}s", flush=True)
            last = now

        print(f"[startup] Engine.load begin devices={devices} max_len={EXECUTION_LEN}", flush=True)
        rt = TPRuntime(devices)
        stage("runtime_ready")
        load_kernels()
        stage("kernels_ready")
        rt.comms = _nccl_init_all(devices)
        stage("nccl_ready")
        # Rebuild the exact TP8 weight tree from the verified tmpfs snapshot.
        # Missing/stale snapshots fail loudly instead of silently reverting to slow GGUF load.
        from model import wcache
        w = wcache.load("tp8")
        stage("weights_ready")
        kv = KVCache.alloc(rt)
        stage("base_kv_ready")
        mtp_kv = KVCache.alloc(rt, n_layers=1) if w.mtp else None
        stage("mtp_kv_ready")
        chunk = int(prefill_chunk_tokens)
        if chunk <= 0:
            raise ValueError("prefill_chunk_tokens must be positive")
        engine = cls(rt=rt, w=w, kv=kv, mtp_kv=mtp_kv, prefill_chunk_tokens=chunk)
        from model.decode_backend import prebuild_decode_residency
        prebuild_decode_residency(engine)
        stage("graphs_ready")
        stage("ready")
        return engine
