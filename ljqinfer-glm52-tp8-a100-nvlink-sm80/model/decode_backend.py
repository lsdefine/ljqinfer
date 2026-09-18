"""Single-sequence decode: graph state, residency, capture, replay, and orchestration."""

from __future__ import annotations

import gc
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional

import torch

from ops.kernels import K, get_decode_attn, get_peer_ar
from ops.decode_attn import decode_attn

from model.runtime import KVCache
from model.blocks import (
    _rmsnorm, decode_ffn_rank, decode_moe_rank_dualstream,
    decode_special_moe_rank_dualstream, _capture_one,
)
from model.prefill import lm_head, embed_tokens, block_forward

from model.config import *  # noqa: F401,F403 — 本机固定常量, 全部写死

if TYPE_CHECKING:
    from model.model import Engine

# Graph state and static workspaces
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class DecodeShape:
    """Fixed decode ABI shape: B sequences, Q query rows, T=B*Q packed rows."""

    B: int
    Q: int

    def __post_init__(self) -> None:
        if self.B < 1 or self.Q < 1:
            raise ValueError(f"decode shape requires positive B/Q, got B={self.B} Q={self.Q}")

    @property
    def T(self) -> int:
        return self.B * self.Q


@dataclass
class DecodeGraph:
    """Per-Q capture. copy_inputs -> replay -> read .logits (rank0)."""

    shape: DecodeShape
    ids_cpu: torch.Tensor                          # int32 [Q] host mirror
    pos: List[torch.Tensor]                        # int64 [Q] per rank
    pos_offset: List[torch.Tensor]                 # int64 [Q] static 0..Q-1 per rank
    k0: List[torch.Tensor]                         # int32 [1] per rank
    xs: List[torch.Tensor]                         # fp16 [Q,D] residual
    logits: Optional[torch.Tensor] = None          # fp32 [Q,VOCAB] rank0
    lm_rows: Optional[torch.Tensor] = None         # fp16 [Q,D] rank0 final-norm rows
    streams: List[torch.cuda.Stream] = field(default_factory=list)
    moe_aux_streams: List[torch.cuda.Stream] = field(default_factory=list)
    moe_fork_events: List[dict] = field(default_factory=list)
    moe_done_events: List[dict] = field(default_factory=list)
    _engine_ref: object = None
    _batch_page_tables: Optional[List[torch.Tensor]] = field(
        default=None, repr=False)
    batch_page_tables: List[List[torch.Tensor]] = field(
        default_factory=list, repr=False)
    # static extras for graph path (allocated at capture)
    ids_dev: List[torch.Tensor] = field(default_factory=list)   # int32 [Q] per rank
    partial: List[torch.Tensor] = field(default_factory=list)   # fp16 [Q,D] per rank

    @property
    def Q(self) -> int:
        return self.shape.Q

    @property
    def T(self) -> int:
        return self.shape.T

    def copy_inputs(self, token: torch.Tensor, length: int) -> None:
        ids = torch.as_tensor(token, dtype=torch.int32).reshape(-1).contiguous()
        if int(ids.numel()) != self.Q:
            raise ValueError(f"graph Q={self.Q} got token width {ids.numel()}")
        self.ids_cpu = ids.cpu() if ids.device.type != "cpu" else ids
        base = int(length)
        Q = self.Q
        for r in range(TP):
            if self.ids_dev:
                self.ids_dev[r].copy_(self.ids_cpu, non_blocking=True)
            if self.k0:
                self.k0[r].fill_(base)
            if self.pos:
                p = self.pos[r]
                # Q=1: fill_ avoids per-step arange host/device tax (accept4e)
                if Q == 1:
                    p.fill_(base)
                else:
                    torch.add(self.pos_offset[r], base, out=p)

    def copy_batch_inputs(self, state, tokens: torch.Tensor) -> None:
        """Bind B independent request rows to one fixed-shape resident graph."""
        B, Q = self.shape.B, self.Q
        if state.batch_size != B:
            raise ValueError(f"the resident DecodeGraph accepts only B={B}")
        ids = torch.as_tensor(tokens, dtype=torch.int32).reshape(B, Q).contiguous()
        lengths = [int(n) for n in state.lengths]
        if len(lengths) != B:
            raise ValueError("batch length metadata does not match graph B")

        # Preserve the proven B1 pointer-rebind path. B2 owns graph-static
        # page-table tensors, one independent table per sequence and rank.
        if B == 1:
            owner = self._engine_ref.kv
            cache = state.caches[0]
            if len(owner.page_table) != len(cache.page_table):
                raise RuntimeError("request page-table rank mismatch")
            canonical = getattr(owner, "_unified_graph_page_tables", None)
            if canonical is None:
                canonical = [table.clone() for table in owner.page_table]
                owner._unified_graph_page_tables = canonical
            for dst, src in zip(owner.page_table, cache.page_table):
                if src.numel() > dst.numel():
                    raise ValueError("request page table exceeds owner table")
                dst[:src.numel()].copy_(src, non_blocking=True)
            self._batch_page_tables = canonical
            self.copy_inputs(ids.reshape(-1), lengths[0])
            return

        if not getattr(self, "batch_page_tables", None):
            raise RuntimeError("batched graph page-table residency is missing")
        self.ids_cpu = ids.reshape(-1).cpu()
        lengths_cpu = torch.tensor(lengths, dtype=torch.int32)
        for r in range(TP):
            self.ids_dev[r].copy_(self.ids_cpu, non_blocking=True)
            self.k0[r].copy_(lengths_cpu, non_blocking=True)
            for b, base in enumerate(lengths):
                lo, hi = b * Q, (b + 1) * Q
                torch.add(self.pos_offset[r][lo:hi], base,
                          out=self.pos[r][lo:hi])
                src = state.caches[b].page_table[r]
                dst = self.batch_page_tables[r][b]
                if src.numel() > dst.numel():
                    raise ValueError("request page table exceeds graph table")
                dst[:src.numel()].copy_(src, non_blocking=True)

    def replay(self) -> torch.Tensor:
        """Replay the resident graph and restore any request page-table binding."""
        try:
            _mono_replay(self._engine_ref, self)
            return self.logits
        finally:
            canonical, self._batch_page_tables = self._batch_page_tables, None
            if canonical is not None:
                for dst, src in zip(self._engine_ref.kv.page_table, canonical):
                    dst.copy_(src, non_blocking=True)

    def release(self) -> None:
        """Reset owned CUDA graphs and return the recyclable peer-AR handle."""
        engine = self._engine_ref
        if engine is None:
            return
        if getattr(engine, "graph_residency_frozen", False):
            raise RuntimeError("cannot release decode graph after residency freeze")
        for st in self.streams + self.moe_aux_streams:
            st.synchronize()
        _reset_cudagraphs(self.__dict__)
        pext = get_peer_ar()
        if getattr(self, "_ar_hid", None) is not None:
            if hasattr(pext, "peer_ar_unregister"):
                pext.peer_ar_unregister(self._ar_hid)
            self._ar_hid = None
        if getattr(self, "_am_hid", None) is not None:
            pext.peer_argmax_unregister(self._am_hid)
            self._am_hid = None
        if getattr(self, "_latent_hid", None) is not None:
            pext.latent_gather_unregister(self._latent_hid)
            self._latent_hid = None
        if getattr(self, "_mtp_gather_hid", None) is not None:
            pext.peer_gather_bf16_unregister(self._mtp_gather_hid)
            self._mtp_gather_hid = None
        if getattr(self, "_mtp_bcast_hid", None) is not None:
            pext.peer_bcast_fp16_unregister(self._mtp_bcast_hid)
            self._mtp_bcast_hid = None
        # Base and MTP graphs share this object but live in separate registries.
        # Identity checks prevent an MTP Q=1/2 cleanup from removing its base peer.
        for registry in (engine.graphs, engine.mtp_graphs):
            if registry.get(self.Q) is self:
                registry.pop(self.Q)
        base_graphs = getattr(engine, "base_graphs", {})
        if base_graphs.get((self.shape.B, self.Q)) is self:
            base_graphs.pop((self.shape.B, self.Q))
        self._engine_ref = None


@dataclass
class _FfnWorkspace:
    """Graph-static FFN/LM workspace container (per-rank buffer lists).

    Pure data holder: buffers are allocated once by _ensure_static_buffers and
    referenced inside CUDA-graph capture; no replay/eager logic lives here.
    """

    Q: int
    pos: List[torch.Tensor]                        # int64 [Q] per rank
    k0: List[torch.Tensor]                         # int32 [1] per rank
    xs: List[torch.Tensor]                         # fp16 [Q,D] residual
    logits: Optional[torch.Tensor] = None          # fp32 [VOCAB] rank0
    streams: List[torch.cuda.Stream] = field(default_factory=list)
    ids_dev: List[torch.Tensor] = field(default_factory=list)   # int32 [Q] per rank
    partial: List[torch.Tensor] = field(default_factory=list)   # fp16 [Q,D] per rank


def replay_decode(graph, state, tokens: torch.Tensor):
    """Replay the resident B1 graph through its declared query width."""
    shape = graph.shape
    B, Q = shape.B, shape.Q
    graph.copy_batch_inputs(state, tokens)
    logits = graph.replay().reshape(B, Q, VOCAB)
    if graph.lm_rows is None:
        raise RuntimeError("decode graph did not produce base hidden states")
    hidden = graph.lm_rows.reshape(B, Q, D).contiguous()
    return logits, hidden












def _ensure_static_buffers(engine, g: DecodeGraph) -> None:
    """Allocate graph-static buffers once (ids_dev / partial / logits / gather)."""
    rt = engine.rt
    T = g.T
    local = D // TP
    if not g.ids_dev:
        for r in range(TP):
            dev = rt.devices[r]
            with torch.cuda.device(dev):
                g.ids_dev.append(torch.zeros(T, device=dev, dtype=torch.int32))
                g.partial.append(torch.zeros(T, D, device=dev, dtype=torch.float16))
    if not getattr(g, "embed_parts", None):
        g.embed_parts = []
        g.gather_buf = []
        for r in range(TP):
            dev = rt.devices[r]
            with torch.cuda.device(dev):
                g.embed_parts.append(torch.zeros(T, local, device=dev, dtype=torch.bfloat16))
                g.gather_buf.append(
                    torch.empty((TP, T, local), dtype=torch.bfloat16, device=dev)
                )
    if g.logits is None:
        with torch.cuda.device(rt.devices[0]):
            g.logits = torch.zeros(T, 154880, device=rt.devices[0], dtype=torch.float32)
    # graph-static FFN / lm workspaces (allocate once, never inside capture)
    if not getattr(g, "ws_h_norm", None):
        g.ws_h_norm = []
        g.ws_shared_h = []
        g.ws_shared_y = []
        g.ws_ei = []
        g.ws_ew = []
        g.ws_x32 = []
        g.ws_routed_h = []
        g.ws_routed = []
        g.ws_logits_r = []
        g.ws_xf = []
        g.ws_lm_in = []
        g.ws_lm_part = []
        g.ws_tok = []  # special LR: arange(Q).repeat_interleave(TOPK) for index_select
        local = D // TP
        for r in range(TP):
            dev = rt.devices[r]
            with torch.cuda.device(dev):
                g.ws_h_norm.append(torch.zeros(T, D, device=dev, dtype=torch.float16))
                g.ws_shared_h.append(torch.zeros(T, LOCAL_SHARED_FF, device=dev, dtype=torch.float16))
                g.ws_shared_y.append(torch.zeros(T, D, device=dev, dtype=torch.float32))
                g.ws_ei.append(torch.zeros(T, TOPK, device=dev, dtype=torch.int64))
                g.ws_ew.append(torch.zeros(T, TOPK, device=dev, dtype=torch.float32))
                g.ws_x32.append(torch.zeros(T, D, device=dev, dtype=torch.float32))
                g.ws_routed_h.append(torch.zeros(T * TOPK, LOCAL_ROUTED_FF, device=dev, dtype=torch.float32))
                g.ws_routed.append(torch.zeros(T, D, device=dev, dtype=torch.float16))
                g.ws_logits_r.append(torch.zeros(T, N_EXPERT, device=dev, dtype=torch.float32))
                g.ws_xf.append(torch.zeros(T, D, device=dev, dtype=torch.float32))
                g.ws_lm_in.append(torch.zeros(T, local, device=dev, dtype=torch.float16))
                g.ws_lm_part.append(torch.zeros(T, 154880, device=dev, dtype=torch.float32))
                g.ws_tok.append(
                    torch.arange(T, device=dev, dtype=torch.long).repeat_interleave(TOPK).contiguous()
                )
        with torch.cuda.device(rt.devices[0]):
            g.lm_rows = torch.zeros(T, D, device=rt.devices[0], dtype=torch.float16)
            g.ws_mtp_h = torch.zeros(T, D, device=rt.devices[0], dtype=torch.float16)
    if not getattr(g, "ws_attn_xn", None):
        g.ws_attn_xn = []
        g.ws_attn_q_local = []
        g.ws_attn_kv_local = []
        g.ws_attn_q = []
        g.ws_attn_kv = []
        for r in range(TP):
            dev = rt.devices[r]
            with torch.cuda.device(dev):
                g.ws_attn_xn.append(torch.empty(T, D, device=dev, dtype=torch.float16))
                g.ws_attn_q_local.append(torch.empty(T, 256, device=dev, dtype=torch.float16))
                g.ws_attn_kv_local.append(torch.empty(T, 72, device=dev, dtype=torch.float16))
                g.ws_attn_q.append(torch.empty(T, 2048, device=dev, dtype=torch.float16))
                g.ws_attn_kv.append(torch.empty(T, 576, device=dev, dtype=torch.float16))
    if not g.streams:
        g.streams = list(rt.streams)

# Base-decode graph body
# -----------------------------------------------------------------------------

def _run_ranks(rt, fn) -> None:
    """Submit fn(r) on each rank's capture stream from THIS thread.

    CUDA graph multi-device capture only records work launched onto the
    capturing stream from the capturing context. ``TPRuntime.run`` uses
    worker threads — illegal under capture ("operation not permitted when
    stream is capturing"). Main-thread sequential launch matches accept4e.
    """
    for r in range(TP):
        with torch.cuda.device(rt.devices[r]), torch.cuda.stream(rt.streams[r]):
            fn(r)


def _rank_embed_lookup(engine, g, r: int) -> None:
    w = engine.w
    y = K.q6.lookup(w.embed[r], g.ids_dev[r])
    g.embed_parts[r].copy_(y)


def _rank_pack_gather(g, r: int) -> None:
    gathered = g.gather_buf[r].permute(1, 0, 2)
    g.xs[r].copy_(gathered.reshape_as(g.xs[r]))


def _rank_attn_project(engine, g, r: int, a) -> None:
    """Produce this rank's q_a/kv_a row shards without entering the collective."""
    orch = get_decode_attn()
    x = g.xs[r].reshape(g.T, D)
    xn = g.ws_attn_xn[r]
    orch.rms_norm_half_out(x, a.norm[r], xn)
    K.q8.forward_out(xn, a.q_a[r][r * 256:(r + 1) * 256], D,
                     g.ws_attn_q_local[r])
    K.q8.forward_out(xn, a.kv_a[r][r * 72:(r + 1) * 72], D,
                     g.ws_attn_kv_local[r])


def _rank_attn_gather(g, r: int) -> None:
    """Enter the graph-safe TP8 latent collective on this rank's stream."""
    g._peer_ext.latent_gather_run(g._latent_hid, r)


def _rank_attn_projected(engine, g, r: int, layer: int, a) -> None:
    """Run the unchanged Nova attention suffix after full latents are visible."""
    orch = get_decode_attn()
    # Private native ABI uses flattened positions and one tensor per sequence;
    # the public decode_attn/v1 boundary remains unchanged for every other caller.
    y = orch.forward_rank_paged_batch_k0_projected(
        g.pos[r].reshape(-1),
        engine.kv.workspace[layer][r],
        ([engine.kv.page_table[r]] if g.shape.B == 1
         else g.batch_page_tables[r]),
        ([g.k0[r]] if g.shape.B == 1
         else [g.k0[r][b:b + 1] for b in range(g.shape.B)]),
        g.ws_attn_q[r],
        a.q_a_norm[r],
        a.q_b[r],
        g.ws_attn_kv[r],
        a.kv_a_norm[r],
        a.k_b[r],
        a.v_b[r],
        a.o[r],
    )[1]
    g.partial[r].copy_(y.reshape_as(g.partial[r]))

def _rank_add_partial(g, r: int) -> None:
    g.xs[r].add_(g.partial[r])

def _rank_ffn(g, r: int, layer: int, blk) -> None:
    # Commit attention residual and preserve the exact FP16 sum while copying
    # it to the FP32 workspace used by the unchanged RMSNorm reduction chain.
    ws = g
    # Fused: xs += partial (fp16, exact same order) and h_norm = rmsnorm(xs)*w
    # in one kernel. ws_xf[r] is no longer written here; it is scratch reused
    # by the routed-expert output later in the layer.
    K.residual_add_copy.fused_add_rmsnorm(
        g.xs[r], g.partial[r], blk.ffn.norm[r], ws.ws_h_norm[r])
    if layer >= 3 and layer not in SPECIAL_LAYERS:
        y2 = decode_moe_rank_dualstream(
            K, blk.ffn, ws.ws_h_norm[r], r, ws,
            g.streams[r], g.moe_aux_streams[r],
            g.moe_fork_events[r][layer], g.moe_done_events[r][layer],
        )
    else:
        y2 = decode_ffn_rank(K, blk.ffn, ws.ws_h_norm[r], layer, r, ws=ws)
    if y2.data_ptr() != g.partial[r].data_ptr():
        g.partial[r].copy_(y2.reshape_as(g.partial[r]))

def _final_norm0(engine, g: DecodeGraph) -> None:
    w = engine.w
    resid = g.xs[0].reshape(-1, D)
    g.ws_mtp_h.copy_(resid[-g.T:])          # Base residuals [T,D] (pre final_norm)
    xf = g.ws_xf[0][:g.T]                   # packed B*Q verify 需全部 Q 行 logits
    xf.copy_(resid[-g.T:])
    inv = torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + 1e-5)
    xf.mul_(inv)
    xf.mul_(w.final_norm)
    g.lm_rows.copy_(xf)

def _rank_lm_slice(g, r: int) -> None:
    local = D // TP
    src = g.lm_rows[:, r * local:(r + 1) * local]
    g.ws_lm_in[r].copy_(src, non_blocking=True)

def _rank_lm_matvec(engine, g, r: int) -> None:
    K.q6.gemm_fp16_out(engine.lm_head_fp16[r], g.ws_lm_in[r],
                       g.ws_lm_part[r])

def _static_decode_body(engine, g: DecodeGraph) -> None:
    """One decode step into static buffers. Does NOT bump kv.length."""
    rt, w = engine.rt, engine.w
    Q = g.Q
    _ensure_static_buffers(engine, g)

    # q6.lookup may allocate a small output; copy into static embed_parts
    # so all_gather always sees stable addresses (graph-safe).
    _run_ranks(rt, lambda r: _rank_embed_lookup(engine, g, r))
    torch.cuda.nccl.all_gather(
        g.embed_parts, g.gather_buf, streams=rt.streams, comms=rt.comms
    )

    # copy_ does bf16->fp16 cast; no intermediate .to() alloc (graph-safe)
    _run_ranks(
        rt, lambda r: g.xs[r].copy_(
            g.gather_buf[r].permute(1, 0, 2).reshape(g.T, D)))

    for blk in w.layers:
        layer = blk.idx
        a = blk.attn

        # A spin-wait P2P collective must be submitted on every rank before
        # any rank queues its dependent Nova suffix.  Keeping all three phases
        # in one per-rank callback deadlocks warm-up at rank0.
        _run_ranks(rt, lambda r: _rank_attn_project(engine, g, r, a))
        _run_ranks(rt, lambda r: _rank_attn_gather(g, r))
        _run_ranks(rt, lambda r: _rank_attn_projected(engine, g, r, layer, a))
        rt.all_reduce(g.partial)

        _run_ranks(rt, lambda r: _rank_ffn(g, r, layer, blk))
        rt.all_reduce(g.partial)

        _run_ranks(rt, lambda r: _rank_add_partial(g, r))

    # Graph-safe lm_head: no default-stream wait_stream / Event (illegal under capture).
    # final_norm only on rank0; slice D2D into ws_lm_in; matvec; NCCL reduce into logits.
    _run_ranks(rt, lambda r: _final_norm0(engine, g) if r == 0 else None)
    # Peer D2D slice must run in each destination device/stream context.
    _run_ranks(rt, lambda r: _rank_lm_slice(g, r))
    _run_ranks(rt, lambda r: _rank_lm_matvec(engine, g, r))
    torch.cuda.nccl.reduce(
        g.ws_lm_part, root=0, op=torch.cuda.nccl.SUM,
        streams=rt.streams, comms=rt.comms,
    )

    def _store_logits(r: int) -> None:
        if r == 0:
            g.logits.copy_(g.ws_lm_part[0])

    _run_ranks(rt, _store_logits)

# Graph residency lifecycle
# -----------------------------------------------------------------------------

def _trim_cuda_cache(devices, reason: str) -> None:
    """Return dead eager allocations before a CUDA-graph private pool is made.

    Prefill and graph warm-up use the default caching allocator, while graph
    capture allocates from private pools which cannot reuse those cached blocks.
    A long prefill can otherwise leave many GiB reserved-but-unused and make a
    tiny graph allocation OOM despite there being enough reclaimable memory.
    """
    gc.collect()
    devices = tuple(int(dev) for dev in devices)
    for dev in devices:
        torch.cuda.synchronize(dev)
    before = [torch.cuda.memory_reserved(dev) for dev in devices]
    # empty_cache is process-wide, so sample every rank before invoking it once;
    # otherwise the first loop iteration receives all ranks' reported release.
    torch.cuda.empty_cache()
    after = [torch.cuda.memory_reserved(dev) for dev in devices]
    released = [(start - end) / (1 << 30)
                for start, end in zip(before, after)]
    detail = ",".join(f"{x:.2f}" for x in released)
    print(f"[decode_path] CUDA cache trim ({reason}) released GiB/rank=[{detail}]", flush=True)


def _trim_cuda_cache_best_effort(devices, reason: str) -> None:
    """Trim during exception cleanup without replacing the original failure."""
    try:
        _trim_cuda_cache(devices, reason)
    except Exception as exc:
        print("[decode_path] CUDA cache trim warning "
              f"({reason}): {type(exc).__name__}: {exc}", flush=True)


def _reset_cudagraphs(items) -> int:
    """Best-effort destroy all CUDAGraph objects in a nested container."""
    stack = [items]
    seen = set()
    count = 0
    while stack:
        item = stack.pop()
        if isinstance(item, dict):
            stack.extend(item.values())
        elif isinstance(item, (list, tuple)):
            stack.extend(item)
        elif isinstance(item, torch.cuda.CUDAGraph) and id(item) not in seen:
            seen.add(id(item))
            try:
                item.reset()
            except Exception:
                pass
            count += 1
    return count


def release_decode_residency(engine, reason: str = "startup rollback") -> None:
    """Release graph residency only while startup construction is still mutable."""
    if engine.graph_residency_frozen:
        raise RuntimeError("decode graph residency is frozen for the serving lifetime")
    if not (engine.base_graphs or engine.mtp_graphs or engine.lm_head_fp16):
        return
    devices = engine.rt.devices
    for stream in engine.rt.streams:
        stream.synchronize()

    # base_graphs is authoritative and includes the B1 compatibility object.
    for graph in list(engine.base_graphs.values()):
        graph.release()
    engine.base_graphs.clear()
    engine.graphs.clear()

    # MTP graphs share DecodeGraph.release; identity-based registry removal keeps
    # same-width base graphs intact during exceptional startup rollback.
    for graph in list(engine.mtp_graphs.values()):
        graph.release()
    engine.mtp_graphs.clear()

    for chain in list(getattr(engine, "mtp_chain_graphs", {}).values()):
        for body in (chain.gA, chain.gB):
            if body is not None:
                body.release()
        chain.mg = [None] * TP
    if getattr(engine, "mtp_chain_graphs", None):
        engine.mtp_chain_graphs.clear()

    engine.lm_head_fp16.clear()
    K.q8.clear_weight_cache()
    gc.collect()
    _trim_cuda_cache(devices, reason)


def _build_lm_head_residency(engine) -> None:
    # Fail-closed: materialize every shard before any CUDA graph capture.
    if engine.lm_head_fp16:
        raise RuntimeError("LM-head residency must be empty before startup build")
    for r, packed in enumerate(engine.w.lm_head):
        if packed is None:
            raise RuntimeError(f"LM-head shard {r} is absent")
        dev = engine.rt.devices[r]
        with torch.cuda.device(dev), torch.cuda.stream(engine.rt.streams[r]):
            fp16 = torch.empty(
                (packed.size(0), packed.size(1) * 256),
                device=dev, dtype=torch.float16)
            K.q6.dequant_fp16_out(packed, fp16)
            engine.lm_head_fp16.append(fp16)
            # Warm the extension-private cuBLAS handle/workspace eagerly:
            # cublasCreate/cudaMalloc are illegal inside graph capture.
            warm_x = torch.zeros((1, fp16.size(1)), device=dev,
                                 dtype=torch.float16)
            warm_out = torch.empty((1, 8), device=dev, dtype=torch.float32)
            K.q6.gemm_fp16_out(fp16[:8], warm_x, warm_out)
    for stream in engine.rt.streams:
        stream.synchronize()
    if len(engine.lm_head_fp16) != TP:
        raise RuntimeError("incomplete LM-head residency")


def prebuild_decode_residency(engine) -> None:
    """Build every supported graph slot once, then freeze serving residency."""
    if engine.graph_residency_frozen:
        return
    if engine.base_graphs or engine.graphs or engine.mtp_graphs:
        raise RuntimeError("graph residency must be empty before startup prebuild")
    from model.routed_down_cache import build_routed_down_cache
    build_routed_down_cache(engine)
    # This is the sole allocator trim in the process lifetime.  Calling
    # empty_cache between captures can invalidate assumptions made by an already
    # captured private graph pool, so every later capture must retain it.
    _trim_cuda_cache(engine.rt.devices, "before startup graph prebuild")
    engine._graph_prebuild_in_progress = True
    try:
        _build_lm_head_residency(engine)
        for b in range(1, 5):
            capture_decode_graph(engine, Q_MAX, B=b)
        if engine.w.mtp is not None:
            for q in range(1, Q_MAX + 1):
                engine.mtp_graphs[q] = capture_mtp_graph(engine, q)
            engine.mtp_chain_graphs = {}
            for b in range(1, 5):
                engine.mtp_chain_graphs[b] = capture_mtp_chain_graph(engine, b)
    except Exception:
        release_decode_residency(engine, "startup graph prebuild rollback")
        raise
    finally:
        engine._graph_prebuild_in_progress = False
    engine.kv.length = 0
    if engine.mtp_kv is not None:
        engine.mtp_kv.length = 0
    engine.graph_residency_frozen = True
    print(
        f"[startup] graph residency frozen slots=base[B1Q{Q_MAX}..B4Q{Q_MAX}],"
        f"mtp[Q1..Q{Q_MAX}],chain[B1..B4]", flush=True)

# Base-decode capture and replay
# -----------------------------------------------------------------------------

def _base_graph_pool(engine, stage: str, rank: int):
    """Share one private CUDA pool across mutually-exclusive batch shapes."""
    pools = getattr(engine, "_base_graph_pool_handles", None)
    if pools is None:
        pools = engine._base_graph_pool_handles = {}
    key = (stage, rank)
    pool = pools.get(key)
    if pool is None:
        with torch.cuda.device(engine.rt.devices[rank]):
            pool = torch.cuda.graph_pool_handle()
        pools[key] = pool
    return pool


def _try_cuda_graph_capture(engine, g: DecodeGraph) -> None:
    """Capture the required base-decode shell and one MONO graph per rank."""
    rt = engine.rt
    w = engine.w
    Q = g.Q
    _ensure_static_buffers(engine, g)

    length = int(engine.kv.length)
    g.ids_cpu.zero_()
    if Q >= 1:
        g.ids_cpu[0] = 1
    for r in range(TP):
        g.ids_dev[r].copy_(g.ids_cpu)
        g.k0[r].fill_(length)
        if Q == 1:
            g.pos[r].fill_(length)
        else:
            torch.add(g.pos_offset[r], length, out=g.pos[r])

    print(f"[decode_path] warm static body Q={Q} k0={length}", flush=True)
    for _ in range(2):
        _static_decode_body(engine, g)
    for st in rt.streams:
        st.synchronize()

    g.g_embed = [None] * TP
    g.g_pack = [None] * TP
    g.g_mono = [None] * TP
    g.g_final = [None] * TP
    g.g_lm_mm = [None] * TP
    g.g_store = [None] * TP
    g.streams = list(rt.streams)

    if not getattr(engine, "_graph_prebuild_in_progress", False):
        _trim_cuda_cache(rt.devices, f"base Q={Q} before MONO capture")

    pext = None
    try:
        # Shell graphs remain separate because embedding all-gather and LM reduce
        # are cross-rank collectives outside the layer MONO graph.
        for r in range(TP):
            st = rt.streams[r]
            with torch.cuda.device(rt.devices[r]):
                g.g_embed[r] = _capture_one(
                    st, lambda r=r: _rank_embed_lookup(engine, g, r), keep_graph=True,
                    pool=_base_graph_pool(engine, "embed", r))
                g.g_pack[r] = _capture_one(
                    st,
                    lambda r=r: g.xs[r].copy_(
                        g.gather_buf[r].permute(1, 0, 2).reshape(g.T, D)),
                    keep_graph=True,
                    pool=_base_graph_pool(engine, "pack", r),
                )
                if r == 0:
                    def _final_body():
                        _rank_add_partial(g, 0)
                        _final_norm0(engine, g)
                    g.g_final[0] = _capture_one(
                        st, _final_body, keep_graph=True,
                        pool=_base_graph_pool(engine, "final", r))
                g.g_lm_mm[r] = _capture_one(
                    st, lambda r=r: _rank_lm_matvec(engine, g, r), keep_graph=True,
                    pool=_base_graph_pool(engine, "lm_mm", r))
                if r == 0:
                    g.g_store[0] = _capture_one(
                        st, lambda: g.logits.copy_(g.ws_lm_part[0]), keep_graph=True,
                        pool=_base_graph_pool(engine, "store", r))
        for st in rt.streams:
            st.synchronize()

        pext = get_peer_ar()
        g._ar_hid = pext.peer_ar_register(list(g.partial))
        # All ranks must launch before synchronization: peer AR is collective.
        for r in range(TP):
            with torch.cuda.device(rt.devices[r]), torch.cuda.stream(rt.streams[r]):
                pext.peer_ar_run(g._ar_hid, r)
        for st in rt.streams:
            st.synchronize()

        for r in range(TP):
            with torch.cuda.device(rt.devices[r]):
                def _mono_body(r=r):
                    for li, blk in enumerate(w.layers):
                        if li > 0:
                            _rank_add_partial(g, r)
                        _rank_attn_project(engine, g, r, blk.attn)
                        _rank_attn_gather(g, r)
                        _rank_attn_projected(engine, g, r, blk.idx, blk.attn)
                        pext.peer_ar_run(g._ar_hid, r)
                        _rank_ffn(g, r, blk.idx, blk)
                        pext.peer_ar_run(g._ar_hid, r)
                g.g_mono[r] = _capture_one(
                    rt.streams[r], _mono_body, keep_graph=True,
                    pool=_base_graph_pool(engine, "mono", r))

        print(
            f"[decode_path] MONO capture Q={Q} OK: 1 graph/rank, peer AR in-graph",
            flush=True,
        )
    except Exception as e:
        graph_fields = ("g_embed", "g_pack", "g_mono", "g_final", "g_lm_mm", "g_store")
        reset = _reset_cudagraphs([getattr(g, name, None) for name in graph_fields])
        for name in graph_fields:
            setattr(g, name, [])
        if getattr(g, "_ar_hid", None) is not None and pext is not None:
            try:
                pext.peer_ar_unregister(g._ar_hid)
            finally:
                g._ar_hid = None
        gc.collect()
        _trim_cuda_cache_best_effort(
            rt.devices, f"base Q={Q} failed MONO cleanup ({reset} graphs)")
        raise RuntimeError(
            f"MONO capture required for decode Q={Q}; startup aborted"
        ) from e


def capture_decode_graph(engine, Q: int, B: int = 1) -> DecodeGraph:
    """Build one fixed resident Base graph for shape ``[B,Q]``."""
    if Q < 1 or Q > Q_MAX or B < 1:
        raise ValueError(
            f"capture requires B>=1 and Q=1..{Q_MAX}, got B={B} Q={Q}")
    T = B * Q
    g = DecodeGraph(
        shape=DecodeShape(B, Q),
        ids_cpu=torch.zeros(T, dtype=torch.int32),
        pos=[],
        pos_offset=[],
        k0=[],
        xs=[],
        logits=None,
        streams=[],
        _engine_ref=engine,
    )
    rt = engine.rt
    for r in range(TP):
        dev = rt.devices[r]
        with torch.cuda.device(dev):
            g.pos.append(torch.zeros(T, device=dev, dtype=torch.int64))
            g.pos_offset.append(
                torch.arange(Q, device=dev, dtype=torch.int64).repeat(B))
            g.k0.append(torch.zeros(B, device=dev, dtype=torch.int32))
            g.xs.append(torch.zeros(T, D, device=dev, dtype=torch.float16))
    if B > 1:
        g.batch_page_tables = [
            [engine.kv.page_table[r].clone() for _ in range(B)]
            for r in range(TP)
        ]
    g.streams = list(rt.streams)
    ordinary_layers = tuple(
        li for li in range(3, N_LAYER) if li not in SPECIAL_LAYERS)
    for r in range(TP):
        dev = rt.devices[r]
        with torch.cuda.device(dev):
            g.moe_aux_streams.append(torch.cuda.Stream(device=dev))
            g.moe_fork_events.append(
                {li: torch.cuda.Event() for li in ordinary_layers})
            g.moe_done_events.append(
                {li: torch.cuda.Event() for li in ordinary_layers})
    _ensure_static_buffers(engine, g)
    g._peer_ext = get_peer_ar()
    g._latent_hid = g._peer_ext.latent_gather_register(
        list(g.ws_attn_q_local), list(g.ws_attn_kv_local),
        list(g.ws_attn_q), list(g.ws_attn_kv))

    try:
        _try_cuda_graph_capture(engine, g)   # 失败即抛: 本引擎不留eager兜底
    except Exception:
        if g._latent_hid is not None:
            g._peer_ext.latent_gather_unregister(g._latent_hid)
            g._latent_hid = None
        raise
    assert all(g.g_embed) and all(g.g_pack) and all(g.g_mono)
    assert all(g.g_lm_mm) and g.g_final[0] is not None and g.g_store[0] is not None
    # cuBLAS nodes in the graph hold raw pointers to dequantized Q8 weights.
    # Keep those cache entries alive across later prefill cache clears.
    K.q8.pin_weight_cache()

    if B == 1:
        engine.graphs[Q] = g
    engine.base_graphs[(B, Q)] = g
    return g


def _mono_replay(engine, g: DecodeGraph) -> None:
    """Replay the sole base-decode path: shell collectives around layer MONO."""
    rt = engine.rt
    streams, comms, devices = rt.streams, rt.comms, rt.devices

    for r in range(TP):
        streams[r].wait_stream(torch.cuda.default_stream(devices[r]))
        torch.cuda.set_stream(streams[r])

    for graph in g.g_embed:
        graph.replay()
    torch.cuda.nccl.all_gather(
        g.embed_parts, g.gather_buf, streams=streams, comms=comms)
    for graph in g.g_pack:
        graph.replay()
    for graph in g.g_mono:
        graph.replay()

    g.g_final[0].replay()
    # rank0 final norm feeds one LM-head slice on every rank.  Peer D2D copies
    # remain eager because the legacy cross-device dependency is not graph-safe.
    for r in range(TP):
        with torch.cuda.device(devices[r]), torch.cuda.stream(streams[r]):
            _rank_lm_slice(g, r)
    for graph in g.g_lm_mm:
        graph.replay()
    torch.cuda.nccl.reduce(
        g.ws_lm_part, root=0, op=torch.cuda.nccl.SUM,
        streams=streams, comms=comms)
    g.g_store[0].replay()

    for r in range(TP):
        default = torch.cuda.default_stream(devices[r])
        torch.cuda.set_stream(default)
        default.wait_stream(streams[r])

# MTP capture and replay
# -----------------------------------------------------------------------------

def _mtp_extras(engine, g: DecodeGraph) -> None:
    """Allocate graph-static MTP prep/output buffers and weight mirrors."""
    rt = engine.rt
    m = engine.w.mtp
    dev0 = rt.devices[0]
    with torch.cuda.device(dev0):
        g.hid_in = torch.zeros(g.shape.T, D, device=dev0, dtype=torch.float16)
        g.m_ef = torch.zeros(g.shape.T, D, device=dev0, dtype=torch.float32)
        g.m_hf = torch.zeros(g.shape.T, D, device=dev0, dtype=torch.float32)
        g.m_cat = torch.zeros(g.shape.T, 2 * D, device=dev0, dtype=torch.float16)
        g.m_logits = torch.zeros(g.ws_lm_part[0].shape[1], device=dev0, dtype=torch.float32)
        g.m_ehw = m.eh_proj.detach().to(dev0, torch.float16).t().contiguous()  # [2D,D]
        g.m_enorm = m.enorm.detach().to(dev0, torch.float32)
        g.m_hnorm = m.hnorm.detach().to(dev0, torch.float32)
    g.m_shnorm = [
        m.shared_head_norm.detach().to(dev, torch.float32)
        for dev in rt.devices
    ]
    g.m_lm_views = [g.ws_lm_part[r][:1] for r in range(TP)]


def _mtp_prep0(g: DecodeGraph) -> None:
    """rank0: rmsnorm(emb)*enorm ++ rmsnorm(hid)*hnorm -> eh_proj -> xs[0]。"""
    eb = g.gather_buf[0].permute(1, 0, 2).reshape(-1, D)
    ef, hf = g.m_ef, g.m_hf
    ef.copy_(eb)
    inv = torch.rsqrt(ef.pow(2).mean(-1, keepdim=True) + 1e-5)
    ef.mul_(inv).mul_(g.m_enorm)
    hf.copy_(g.hid_in)
    inv2 = torch.rsqrt(hf.pow(2).mean(-1, keepdim=True) + 1e-5)
    hf.mul_(inv2).mul_(g.m_hnorm)
    g.m_cat[:, :D].copy_(ef)
    g.m_cat[:, D:].copy_(hf)
    torch.matmul(g.m_cat, g.m_ehw, out=g.xs[0])


def _mtp_attn(engine, g: DecodeGraph, r: int) -> None:
    m = engine.w.mtp
    y = decode_attn(
        get_decode_attn(),
        x=g.xs[r].view(g.shape.B, g.Q, D),
        positions=g.pos[r].view(g.shape.B, g.Q),
        kv_workspace=engine.mtp_kv.workspace[0][r],
        page_tables=(g.mtp_pt[r] if getattr(g, "mtp_pt", None) is not None
                     else engine.mtp_kv.page_table[r].view(1, -1)),
        context_lengths=g.k0[r].view(g.shape.B),
        attn_norm=m.block.attn.norm[r],
        q_a=m.block.attn.q_a[r],
        q_a_norm=m.block.attn.q_a_norm[r],
        q_b=m.block.attn.q_b[r],
        kv_a=m.block.attn.kv_a[r],
        kv_a_norm=m.block.attn.kv_a_norm[r],
        k_b=m.block.attn.k_b[r],
        v_b=m.block.attn.v_b[r],
        attn_out=m.block.attn.o[r],
    )
    g.partial[r].copy_(y.reshape_as(g.partial[r]))


def _mtp_ffn(engine, g: DecodeGraph, r: int) -> None:
    """折 attn 残差 + norm + FFN(layer78 special LR fp16 路径)。"""
    m = engine.w.mtp
    # Fused: xs += partial and h_norm = rmsnorm(xs)*w in one kernel
    # (same glue as _rank_ffn; ws_xf no longer written here, scratch only).
    K.residual_add_copy.fused_add_rmsnorm(
        g.xs[r], g.partial[r], m.block.ffn.norm[r], g.ws_h_norm[r])
    y2 = decode_special_moe_rank_dualstream(
        K, m.block.ffn, g.ws_h_norm[r], MTP_LAYER, r, g,
        g.streams[r], g.moe_aux_streams[r],
        g.moe_fork_events[r][MTP_LAYER], g.moe_done_events[r][MTP_LAYER],
    )
    if y2.data_ptr() != g.partial[r].data_ptr():
        g.partial[r].copy_(y2.reshape_as(g.partial[r]))


def _mtp_final_lm(engine, g: DecodeGraph, r: int) -> None:
    """Fold FFN residual, normalize final rows, then run the local LM shard."""
    g.xs[r].add_(g.partial[r])
    rows = getattr(g, "row_idx", None)
    B = g.shape.B if rows is not None else 1
    xf = g.ws_xf[r][:B]
    if rows is None:
        xf.copy_(g.xs[r][g.Q - 1:])
    else:
        xf.copy_(g.xs[r].index_select(0, rows[r]))
    inv = torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + 1e-5)
    xf.mul_(inv).mul_(g.m_shnorm[r])
    local = D // TP
    g.ws_lm_in[r][:B].copy_(xf[:, r * local:(r + 1) * local])
    K.q6.gemm_fp16_out(engine.lm_head_fp16[r], g.ws_lm_in[r][:B],
                       g.ws_lm_part[r][:B])


def _static_mtp_body(engine, g: DecodeGraph) -> None:
    """整步 eager 版(warmup 用), 与 mtp_replay 相位一一对应。"""
    rt = engine.rt
    _run_ranks(rt, lambda r: _rank_embed_lookup(engine, g, r))
    torch.cuda.nccl.all_gather(g.embed_parts, g.gather_buf, streams=rt.streams, comms=rt.comms)
    _run_ranks(rt, lambda r: _mtp_prep0(g) if r == 0 else None)
    for r in range(1, TP):
        rt.streams[r].wait_stream(rt.streams[0])
    _run_ranks(rt, lambda r: g.xs[r].copy_(g.xs[0], non_blocking=True) if r > 0 else None)
    _run_ranks(rt, lambda r: _mtp_attn(engine, g, r))
    rt.all_reduce(g.partial)
    _run_ranks(rt, lambda r: _mtp_ffn(engine, g, r))
    rt.all_reduce(g.partial)
    _run_ranks(rt, lambda r: _mtp_final_lm(engine, g, r))
    torch.cuda.nccl.reduce(g.m_lm_views, root=0, op=torch.cuda.nccl.SUM,
                           streams=rt.streams, comms=rt.comms)
    _run_ranks(rt, lambda r: g.m_logits.copy_(g.ws_lm_part[0][0]) if r == 0 else None)


def _mtp_fused_rank(engine, g: DecodeGraph, pext, r: int) -> None:
    """One graph-capturable rank-local MTP body, including peer synchronization."""
    _rank_embed_lookup(engine, g, r)
    pext.peer_gather_bf16_run(g._mtp_gather_hid, r)
    if r == 0:
        _mtp_prep0(g)
    pext.peer_bcast_fp16_run(g._mtp_bcast_hid, r)
    _mtp_attn(engine, g, r)
    pext.peer_ar_run(g._ar_hid, r)
    _mtp_ffn(engine, g, r)
    pext.peer_ar_run(g._ar_hid, r)
    _mtp_final_lm(engine, g, r)


def capture_mtp_graph(engine, Q: int) -> DecodeGraph:
    """Build a fused MTP draft graph for one sequence of width Q."""
    assert 1 <= Q <= Q_MAX, f"mtp graph Q must be 1..{Q_MAX}, got {Q}"
    assert engine.w.mtp is not None and engine.mtp_kv is not None
    rt = engine.rt
    g = DecodeGraph(
        shape=DecodeShape(1, Q),
        ids_cpu=torch.zeros(Q, dtype=torch.int32),
        pos=[], pos_offset=[], k0=[], xs=[], logits=None, streams=[],
        _engine_ref=engine,
    )
    for r in range(TP):
        dev = rt.devices[r]
        with torch.cuda.device(dev):
            g.pos.append(torch.zeros(Q, device=dev, dtype=torch.int64))
            g.pos_offset.append(torch.arange(Q, device=dev, dtype=torch.int64))
            g.k0.append(torch.zeros(1, device=dev, dtype=torch.int32))
            g.xs.append(torch.zeros(Q, D, device=dev, dtype=torch.float16))
    g.streams = list(rt.streams)
    for r in range(TP):
        dev = rt.devices[r]
        with torch.cuda.device(dev):
            g.moe_aux_streams.append(torch.cuda.Stream(device=dev))
            g.moe_fork_events.append({MTP_LAYER: torch.cuda.Event()})
            g.moe_done_events.append({MTP_LAYER: torch.cuda.Event()})
    _ensure_static_buffers(engine, g)
    _mtp_extras(engine, g)
    get_decode_attn()
    for r in range(TP):
        g.pos[r].copy_(g.pos_offset[r])
    print(f"[decode_path] warm mtp body Q={Q}", flush=True)
    for _ in range(2):
        _static_mtp_body(engine, g)
    for st in rt.streams:
        st.synchronize()
    if not getattr(engine, "_graph_prebuild_in_progress", False):
        _trim_cuda_cache(rt.devices, f"MTP Q={Q} before capture")

    g.mg_fused = [None] * TP
    pext = get_peer_ar()
    try:
        g._ar_hid = pext.peer_ar_register(list(g.partial))
        g._mtp_gather_hid = pext.peer_gather_bf16_register(
            list(g.embed_parts), list(g.gather_buf))
        g._mtp_bcast_hid = pext.peer_bcast_fp16_register(list(g.xs), 0)

        # Warm every operation and peer sequence before graph capture.
        for _ in range(2):
            _run_ranks(rt, lambda r: _mtp_fused_rank(engine, g, pext, r))
            for st in rt.streams:
                st.synchronize()

        for r in range(TP):
            with torch.cuda.device(rt.devices[r]):
                g.mg_fused[r] = _capture_one(
                    rt.streams[r],
                    lambda r=r: _mtp_fused_rank(engine, g, pext, r),
                    keep_graph=True,
                )
        with torch.cuda.device(rt.devices[0]):
            g.mg_store0 = _capture_one(
                rt.streams[0], lambda: g.m_logits.copy_(g.ws_lm_part[0][0]))
    except Exception:
        reset = _reset_cudagraphs([
            getattr(g, "mg_fused", None), getattr(g, "mg_store0", None)])
        g.mg_fused = []
        g.mg_store0 = None
        for attr, unregister in (
            ("_mtp_bcast_hid", "peer_bcast_fp16_unregister"),
            ("_mtp_gather_hid", "peer_gather_bf16_unregister"),
            ("_ar_hid", "peer_ar_unregister"),
        ):
            hid = getattr(g, attr, None)
            if hid is not None:
                try:
                    getattr(pext, unregister)(hid)
                finally:
                    setattr(g, attr, None)
        gc.collect()
        _trim_cuda_cache_best_effort(
            rt.devices, f"MTP Q={Q} failed capture cleanup ({reset} graphs)")
        raise
    K.q8.pin_weight_cache()
    print(f"[decode_path] fused MTP graph captured Q={Q}", flush=True)
    return g


def mtp_replay(engine, g: DecodeGraph) -> torch.Tensor:
    """Replay one fused MTP draft step and return rank-0 fp32 logits."""
    rt = engine.rt
    rt_streams, comms, devices = rt.streams, rt.comms, rt.devices
    for r in range(TP):
        rt_streams[r].wait_stream(torch.cuda.default_stream(devices[r]))
        torch.cuda.set_stream(rt_streams[r])
    for graph in g.mg_fused:
        graph.replay()
    torch.cuda.nccl.reduce(g.m_lm_views, root=0, op=torch.cuda.nccl.SUM,
                           streams=rt_streams, comms=comms)
    g.mg_store0.replay()
    for r in range(TP):
        default = torch.cuda.default_stream(devices[r])
        torch.cuda.set_stream(default)
        default.wait_stream(rt_streams[r])
    return g.m_logits


# Public orchestration API
# -----------------------------------------------------------------------------

def advance(engine: Engine, tokens) -> torch.Tensor:
    """Advance the base model by Q=1/2 committed or candidate tokens.

    Returns rank-0 logits [Q,V] and advances the base KV length by Q.
    """
    token = torch.as_tensor(tokens, dtype=torch.long).reshape(-1)
    engine.kv.validate_page_tables()
    q = token.numel()
    graph = engine.graphs.get(q)
    if graph is None:
        raise RuntimeError(f"missing startup-built base decode graph Q={q}")
    graph.copy_inputs(token, engine.kv.length)
    graph.replay()
    engine.kv.length += q
    return graph.logits.reshape(q, -1)


def retract(engine: Engine, n: int = 1) -> None:
    """Drop the last n (rejected) KV slots — bookkeeping only, slots get
    overwritten by the next advance."""
    engine.kv.length -= n


def base_hidden(engine: Engine, k: int) -> torch.Tensor:
    """Post-final-norm hidden [k,D] fp16 of the last advance(k) rows —
    the draft input contract of mtp_forward."""
    k = int(k)
    if k < 1:
        raise ValueError("k must be >= 1")
    graph = engine.graphs.get(k)
    if graph is None:
        raise RuntimeError(f"no main-model graph for Q={k}")
    residual = graph.xs[0][-k:]
    return _rmsnorm(residual, engine.w.final_norm).to(torch.float16).contiguous()


def mtp_forward(engine: Engine, tokens, start: int,
                hidden: torch.Tensor, *, cache: Optional[KVCache] = None,
                return_hidden: bool = False) -> tuple:
    """MTP nextn (blk.78) over T tokens (vLLM Glm4Moe formula), T>=1.

    tokens: [T] ints = t_{start+1} .. t_{start+T} (shifted-by-one stream).
    hidden: [T,D] rank0 fp16 = base residuals of t_{start} .. t_{start+T-1},
            already final-normed by caller (post_final variant).
    Fills the selected MTP cache slots [start, start+T).  By default returns
    ``(draft_token, logits)``.  With ``return_hidden=True`` it additionally
    returns the final raw MTP-block row used to recurse the same head.
    """
    if engine.w.mtp is None or engine.mtp_kv is None:
        raise RuntimeError("MTP head not loaded")
    mtp_cache = engine.mtp_kv if cache is None else cache
    mtp_cache.validate_page_tables()
    rt, w, m = engine.rt, engine.w, engine.w.mtp
    ids = torch.as_tensor(tokens, dtype=torch.int32).reshape(-1)
    T = ids.numel()

    # Batched rows share one width-specialized graph serially.  Its attention
    # node captured the owner page-table *addresses*; replacing their contents
    # retargets it to this request-local physical page run without recapture.
    # Streams are synchronized at mtp_replay's public boundary, so the next row
    # cannot race the table swap.
    g = engine.mtp_graphs.get(int(T))
    if g is not None and cache is None:
        g.copy_inputs(ids, start)
        g.hid_in.copy_(hidden, non_blocking=True)
        logits = mtp_replay(engine, g)
        mtp_cache.length = start + int(T)
        result = (int(logits.argmax().item()), logits)
        return result + (g.xs[0][-1:],) if return_hidden else result

    if g is not None:
        owner = engine.mtp_kv
        if len(owner.page_table) != len(mtp_cache.page_table):
            raise RuntimeError("MTP cache page-table rank mismatch")
        canonical = getattr(owner, "_batched_graph_page_tables", None)
        if canonical is None:
            canonical = [table.clone() for table in owner.page_table]
            owner._batched_graph_page_tables = canonical
        for dst, src in zip(owner.page_table, mtp_cache.page_table):
            if src.numel() > dst.numel():
                raise ValueError("request MTP page table exceeds owner table")
            dst[:src.numel()].copy_(src, non_blocking=True)
        try:
            g.copy_inputs(ids, start)
            g.hid_in.copy_(hidden, non_blocking=True)
            logits = mtp_replay(engine, g)
            mtp_cache.length = start + int(T)
            result = (int(logits.argmax().item()), logits)
            return result + (g.xs[0][-1:],) if return_hidden else result
        finally:
            # mtp_replay returns only after all rank side streams have joined
            # their default streams, so restoring now cannot race attention.
            for dst, src in zip(owner.page_table, canonical):
                dst.copy_(src, non_blocking=True)

    dev0 = rt.devices[0]

    # --- rank0: embed tokens, norm both halves, eh_proj -> [T,D] --------------
    emb = embed_tokens(rt, w, ids)
    e = _rmsnorm(emb[0], m.enorm)                      # [T,D] fp16 on dev0
    h = hidden.to(dev0, dtype=torch.float16)
    if h.dim() == 1:
        h = h.unsqueeze(0)
    h = _rmsnorm(h, m.hnorm)                           # [T,D] fp16 on dev0
    x0 = torch.nn.functional.linear(
        torch.cat([e, h], dim=-1).float(), m.eh_proj.float()
    ).to(torch.float16).contiguous()                   # [T,D] fp16 on dev0

    # --- distribute to 8 ranks and run blk.78 (attn+ffn) on mtp_kv ------------
    x: List[torch.Tensor] = [x0.to(rt.devices[r]).contiguous() for r in range(TP)]
    pos = torch.arange(start, start + T, dtype=torch.int64)
    positions: List[Optional[torch.Tensor]] = [
        pos.to(rt.devices[r]) for r in range(TP)]
    # The MTP block is logical layer 78, but its standalone cache has one slot.
    mtp_cache.length = start
    block_forward(rt, m.block, x, mtp_cache, positions, kv_layer=0)
    mtp_cache.length = start + T

    # --- shared_head_norm -> lm_head (reuse base lm_head weights) -------------
    hn = _rmsnorm(x[0][-1:], m.shared_head_norm)       # [1,D] rank0
    logits = lm_head(rt, w, hn)                        # [VOCAB] fp32 rank0
    tok = int(logits.reshape(-1).argmax().item())
    result = (tok, logits)
    return result + (x[0][-1:],) if return_hidden else result




CHAIN_COUNT = Q_MAX - 1


@dataclass
class MtpChainGraph:
    """Single-replay fused MTP draft chain for B sequences (Q_MAX-1 drafts)."""
    B: int
    gA: DecodeGraph = None            # step0 body, shape (B, Q_MAX)
    gB: DecodeGraph = None            # recurrent body, shape (B, 1)
    mg: List[object] = field(default_factory=list)   # per-rank CUDA graphs
    pt: List[torch.Tensor] = field(default_factory=list)      # [B,P] int64/rank
    c_base: List[torch.Tensor] = field(default_factory=list)  # [B] int32/rank
    c_tok32: List[torch.Tensor] = field(default_factory=list) # [B] int32/rank
    c_drafts: torch.Tensor = None     # dev0 [CHAIN_COUNT, B] int32
    ids_cpu: torch.Tensor = None      # pinned [B*Q_MAX] int32
    pos_cpu: torch.Tensor = None      # pinned [B*Q_MAX] int64
    k0_cpu: torch.Tensor = None       # pinned [B] int32
    base_cpu: torch.Tensor = None     # pinned [B] int32
    row_cpu: torch.Tensor = None      # pinned [B] int64



def _chain_lm_tail(engine, gc: "MtpChainGraph", g: DecodeGraph, pext,
                   r: int, s: int) -> None:
    """AR the [B,V] lm partial, argmax on rank0, bcast tokens to all ranks."""
    pext.peer_argmax_run(g._am_hid, r)
    if r == 0:
        gc.c_drafts[s].copy_(gc.c_tok32[0])


def _chain_fused_rank(engine, gc: "MtpChainGraph", pext, r: int) -> None:
    """Full CHAIN_COUNT-step MTP draft chain for rank r in one launch stream."""
    gA, gB = gc.gA, gc.gB
    B = gc.B
    _mtp_fused_rank(engine, gA, pext, r)
    _chain_lm_tail(engine, gc, gA, pext, r, 0)
    if r == 0:
        torch.index_select(gA.xs[0], 0, gA.row_idx[0], out=gB.hid_in)
    gB.k0[r].copy_(gc.c_base[r])
    gB.pos[r].copy_(gc.c_base[r])
    for s in range(1, CHAIN_COUNT):
        _mtp_fused_rank(engine, gB, pext, r)
        _chain_lm_tail(engine, gc, gB, pext, r, s)
        if s < CHAIN_COUNT - 1:
            if r == 0:
                gB.hid_in.copy_(gB.xs[0])
            gB.k0[r].add_(1)
            gB.pos[r].add_(1)


def _build_chain_body(engine, gc: "MtpChainGraph", B: int, Q: int,
                      final_rows: bool) -> DecodeGraph:
    """Allocate a graph-static MTP body (no capture) of shape (B, Q)."""
    rt = engine.rt
    T = B * Q
    g = DecodeGraph(
        shape=DecodeShape(B, Q),
        ids_cpu=torch.zeros(T, dtype=torch.int32),
        pos=[],
        pos_offset=[],
        k0=[],
        xs=[],
        logits=None,
        streams=[],
        _engine_ref=engine,
    )
    g.streams = list(rt.streams)
    for r in range(TP):
        dev = rt.devices[r]
        with torch.cuda.device(dev):
            g.moe_aux_streams.append(torch.cuda.Stream(device=dev))
            g.moe_fork_events.append({MTP_LAYER: torch.cuda.Event()})
            g.moe_done_events.append({MTP_LAYER: torch.cuda.Event()})
    _ensure_static_buffers(engine, g)
    g.row_idx = []
    for r, dev in enumerate(rt.devices):
        with torch.cuda.device(dev):
            g.pos.append(torch.zeros(T, device=dev, dtype=torch.long))
            g.pos_offset.append(
                torch.arange(Q, device=dev, dtype=torch.int64).repeat(B))
            g.xs.append(torch.zeros(T, D, device=dev, dtype=torch.float16))
            g.k0.append(torch.zeros(B, device=dev, dtype=torch.int32))
            if final_rows:
                g.row_idx.append(torch.zeros(B, device=dev, dtype=torch.long))
            else:
                g.row_idx.append(torch.arange(B, device=dev, dtype=torch.long))
    _mtp_extras(engine, g)
    g.mtp_pt = gc.pt
    return g


def capture_mtp_chain_graph(engine, B: int) -> "MtpChainGraph":
    """Build the fused single-replay MTP draft chain graph for batch B."""
    assert engine.w.mtp is not None and engine.mtp_kv is not None
    rt = engine.rt
    gc = MtpChainGraph(B=B)
    P = engine.mtp_kv.page_table[0].numel()
    for r, dev in enumerate(rt.devices):
        with torch.cuda.device(dev):
            gc.pt.append(torch.zeros(B, P, device=dev, dtype=torch.int64))
            gc.c_base.append(torch.zeros(B, device=dev, dtype=torch.int32))
            gc.c_tok32.append(torch.zeros(B, device=dev, dtype=torch.int32))
    gc.gA = _build_chain_body(engine, gc, B, Q_MAX, final_rows=True)
    gc.gB = _build_chain_body(engine, gc, B, 1, final_rows=False)
    gc.gB.ids_dev = gc.c_tok32
    with torch.cuda.device(rt.devices[0]):
        gc.c_drafts = torch.zeros(CHAIN_COUNT, B, device=rt.devices[0],
                                  dtype=torch.int32)
    gc.ids_cpu = torch.zeros(B * Q_MAX, dtype=torch.int32, pin_memory=True)
    gc.pos_cpu = torch.zeros(B * Q_MAX, dtype=torch.long, pin_memory=True)
    gc.k0_cpu = torch.zeros(B, dtype=torch.int32, pin_memory=True)
    gc.base_cpu = torch.zeros(B, dtype=torch.int32, pin_memory=True)
    gc.row_cpu = torch.zeros(B, dtype=torch.long, pin_memory=True)

    # Warm every kernel shape eagerly before any capture.
    for _ in range(2):
        _static_mtp_body(engine, gc.gA)
        _static_mtp_body(engine, gc.gB)
    for st in rt.streams:
        st.synchronize()
    if not getattr(engine, "_graph_prebuild_in_progress", False):
        _trim_cuda_cache(rt.devices, f"MTP chain B={B} before capture")

    gc.mg = [None] * TP
    pext = get_peer_ar()
    try:
        for g in (gc.gA, gc.gB):
            g._ar_hid = pext.peer_ar_register(list(g.partial))
            g._mtp_gather_hid = pext.peer_gather_bf16_register(
                list(g.embed_parts), list(g.gather_buf))
            g._mtp_bcast_hid = pext.peer_bcast_fp16_register(list(g.xs), 0)
            g._am_hid = pext.peer_argmax_register(
                list(g.ws_lm_part), list(gc.c_tok32), B)

        # Warm the fused peer sequence, then capture one graph per rank.
        # NOTE: must warm via parallel workers (rt.run): the full chain per rank
        # overflows the CUDA launch queue, so sequential main-thread launch
        # deadlocks on the peer spin kernels. Capture only records, so the
        # main-thread _run_ranks below stays legal.
        for _ in range(2):
            rt.run(lambda r: _chain_fused_rank(engine, gc, pext, r))
            for st in rt.streams:
                st.synchronize()

        def _capture(r: int) -> None:
            dev = rt.devices[r]
            with torch.cuda.device(dev):
                mg = torch.cuda.CUDAGraph()
                with torch.cuda.stream(rt.streams[r]):
                    with torch.cuda.graph(
                            mg, stream=rt.streams[r],
                            pool=_base_graph_pool(engine, "chain", r)):
                        _chain_fused_rank(engine, gc, pext, r)
                gc.mg[r] = mg
        _run_ranks(rt, _capture)
        for st in rt.streams:
            st.synchronize()
    except Exception:
        gc.mg = [None] * TP
        raise
    return gc


def mtp_chain_replay(engine, gc: "MtpChainGraph") -> None:
    """Replay the fused chain graphs (one per rank) on the decode streams."""
    rt = engine.rt
    for r, dev in enumerate(rt.devices):
        with torch.cuda.device(dev):
            rt.streams[r].wait_stream(torch.cuda.default_stream(dev))
    for r, dev in enumerate(rt.devices):
        with torch.cuda.device(dev):
            with torch.cuda.stream(rt.streams[r]):
                gc.mg[r].replay()
    for r, dev in enumerate(rt.devices):
        with torch.cuda.device(dev):
            torch.cuda.default_stream(dev).wait_stream(rt.streams[r])


def mtp_draft_chain_batched(engine, tokens_list, starts, hiddens,
                            caches) -> List[List[int]]:
    """Draft CHAIN_COUNT tokens per sequence with one fused graph replay.

    tokens_list[b]: accepted tokens (1..Q_MAX) priming sequence b.
    starts[b]: MTP KV start position for sequence b.
    hiddens[b]: [len(tokens_list[b]), D] fp16 residuals on device 0.
    caches[b]: per-sequence MTP KVCache (page tables copied, never mutated
    beyond length accounting).
    """
    B = len(tokens_list)
    gc = getattr(engine, "mtp_chain_graphs", {}).get(B)
    if gc is None:
        raise RuntimeError(f"missing startup-built MTP chain graph B={B}")
    rt = engine.rt
    gA = gc.gA
    gc.ids_cpu.zero_()
    for b, toks in enumerate(tokens_list):
        n = len(toks)
        assert 1 <= n <= Q_MAX, f"chain prime width must be 1..{Q_MAX}, got {n}"
        gc.ids_cpu[b * Q_MAX:b * Q_MAX + n] = torch.as_tensor(
            toks, dtype=torch.int32)
        gc.row_cpu[b] = b * Q_MAX + n - 1
        gc.k0_cpu[b] = int(starts[b])
        gc.base_cpu[b] = int(starts[b]) + n
        torch.arange(int(starts[b]), int(starts[b]) + Q_MAX,
                     out=gc.pos_cpu[b * Q_MAX:(b + 1) * Q_MAX])
    for r, dev in enumerate(rt.devices):
        with torch.cuda.device(dev):
            gA.ids_dev[r].copy_(gc.ids_cpu, non_blocking=True)
            gA.pos[r].copy_(gc.pos_cpu, non_blocking=True)
            gA.k0[r].copy_(gc.k0_cpu, non_blocking=True)
            gA.row_idx[r].copy_(gc.row_cpu, non_blocking=True)
            gc.c_base[r].copy_(gc.base_cpu, non_blocking=True)
            for b, cache in enumerate(caches):
                src = cache.page_table[r]
                gc.pt[r][b, :src.numel()].copy_(src, non_blocking=True)
    with torch.cuda.device(rt.devices[0]):
        for b, (toks, hid) in enumerate(zip(tokens_list, hiddens)):
            n = len(toks)
            gA.hid_in[b * Q_MAX:b * Q_MAX + n].copy_(hid, non_blocking=True)
    mtp_chain_replay(engine, gc)
    drafts = gc.c_drafts.cpu()
    for b, (toks, cache) in enumerate(zip(tokens_list, caches)):
        cache.length = int(starts[b]) + len(toks) + CHAIN_COUNT - 1
    return [[int(drafts[s, b]) for s in range(CHAIN_COUNT)]
            for b in range(B)]


def ensure_mtp_graphs(engine: Engine) -> None:
    """Assert that startup built both immutable MTP graph slots."""
    missing = [q for q in range(1, Q_MAX + 1) if q not in engine.mtp_graphs]
    if missing:
        raise RuntimeError(f"missing startup-built MTP graphs: {missing}")
