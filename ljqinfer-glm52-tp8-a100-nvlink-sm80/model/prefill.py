"""prefill 前向: embed → 79层 block_forward 主循环 → final_norm → lm_head。单序列 + 批两套。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional, Sequence, Tuple, Union

import torch

from ops.kernels import K

from model.weights import Attn, DenseFFN, MoE, Layer, Weights
from model.runtime import TPRuntime, KVCache
from model.blocks import _rmsnorm

from model.config import *  # noqa: F401,F403 — 本机固定常量, 全部写死

if TYPE_CHECKING:
    from model.model import Engine


@dataclass
class BatchPrefillState:
    """Ephemeral ownership record for one all-at-once multi-sequence request.

    The model layer does not allocate/recycle pages dynamically.  Construction
    assigns each sequence one disjoint, whole-page slice of the engine's
    already-allocated physical pool.  ``release`` invalidates the complete
    request at once; the next request may overwrite every assigned page.
    """
    caches: List[KVCache]
    offsets: List[int]
    lengths: List[int]
    capacities: List[int]
    page_indices: List[List[int]]
    # Optional request-local one-layer MTP views, bound when the prompt-shifted
    # MTP slots are filled during (chunked) prefill.  Ownership transfers to the
    # decode state together with the base caches.
    mtp_caches: List[KVCache] = field(default_factory=list)
    released: bool = False

    def split(self, flat: torch.Tensor) -> List[torch.Tensor]:
        if self.released:
            raise RuntimeError("batch prefill state has been released")
        if flat.shape[0] != self.offsets[-1]:
            raise ValueError("flat tensor length does not match this batch")
        return [flat[self.offsets[i]:self.offsets[i + 1]]
                for i in range(len(self.lengths))]

    def release(self) -> None:
        if self.released:
            return
        for cache in self.caches:
            cache.length = 0
        for cache in self.mtp_caches:
            cache.length = 0
        self.mtp_caches.clear()
        self.released = True

    def decode_state(self, graph) -> "BatchDecodeState":
        """Freeze this prefill result and bind its resident decode graph."""
        from model.batch_decode import BatchDecodeState
        if self.released:
            raise RuntimeError("cannot decode a released prefill state")
        lengths = [cache.length for cache in self.caches]
        if not self.caches or self.caches[0].page_table is None:
            raise RuntimeError("batch decode requires paged KV page tables")
        k0s = [[torch.tensor([n], dtype=torch.int32,
                             device=self.caches[0].page_table[r].device)
                 for n in lengths] for r in range(TP)]
        # Decode kernels require graph-stable full-pool tables.  Preserve each
        # row's actual logical->physical mapping (including cold-loaded pages);
        # the unused tail is never read because k0/capacity bound the prefix.
        tables: List[List[torch.Tensor]] = [[] for _ in range(TP)]
        for r in range(TP):
            pool_pages = self.caches[0].pool[0][r].shape[0]
            device = self.caches[0].page_table[r].device
            for pages in self.page_indices:
                table = torch.zeros(pool_pages, dtype=torch.int64, device=device)
                table[:len(pages)] = torch.tensor(
                    pages, dtype=torch.int64, device=device)
                tables[r].append(table)
        mtp_caches = list(self.mtp_caches)
        state = BatchDecodeState(caches=self.caches,
                                 lengths=lengths,
                                 capacities=self.capacities,
                                 graph=graph,
                                 page_indices=[list(pages)
                                               for pages in self.page_indices],
                                 k0s=k0s,
                                 tables=tables,
                                 mtp_caches=mtp_caches)
        # Ownership of the shared paged-KV slots (and any request-local MTP
        # views bound during chunked prefill) moves to the decode state.
        # A later prefill-wrapper release must not invalidate live decode KV.
        self.mtp_caches = []
        self.released = True
        return state


def attn_forward(rt: TPRuntime, a: Attn, x: List[torch.Tensor],
                 kv: KVCache, layer: int, positions) -> None:
    """MLA attention, writes result back into x (list of 8 rank-local [T,D]).
    Traffic: exactly ONE all-reduce on the o_proj output."""
    out: List[torch.Tensor] = [None] * TP  # type: ignore
    cache_start = kv.length

    def _rank(r: int):
        # KV storage is a preallocated invariant.  Never replace it with a
        # tensor returned by an allocating fast path: captured decode graphs
        # retain the original data_ptr.  The cached inplace ABI is used for
        # every prefill shape, including a cold full-capacity prefill.
        pool = kv.workspace[layer][r]
        ptr = pool.data_ptr()
        _, partial = K.prefill_attn.forward_rank_paged_inplace_tc(
            x[r], positions[r], pool, kv.page_table[r], cache_start,
            a.norm[r], a.q_a[r], a.q_a_norm[r], a.q_b[r],
            a.kv_a[r], a.kv_a_norm[r], a.k_b[r], a.v_b[r], a.o[r])
        if pool.data_ptr() != ptr:
            raise RuntimeError("prefill KV pool data_ptr changed")
        out[r] = partial                                         # [T,D] pre all-reduce

    rt.run(_rank)
    rt.all_reduce(out)                                            # === ALL-REDUCE #1 ===
    for r in range(TP):
        x[r] = out[r]


def _grouped_topk(probs: torch.Tensor, bias: torch.Tensor):
    """DeepSeek-V3 grouped top-k router selection.
    probs=sigmoid(logits) [T,N_EXPERT] fp32; bias [N_EXPERT] fp32.
    Selection score sel=probs+bias; group=N_GROUP of GROUP_SIZE; group score=
    sum of top-2 sel in group; keep GROUP_TOPK groups; take TOPK experts by sel
    among their candidates. Weight = probs[expert] (NO bias), then *ROUTED_SCALING/sum.
    Returns (expert_ids[T,TOPK] int64, expert_w[T,TOPK] fp32)."""
    T = probs.shape[0]
    if T < 128:
        # CUDA cheaply reduces 256 experts to the exact winning set. Re-run the
        # final torch.topk so equal expert scores retain torch's ordering too.
        candidate_eid = torch.empty((T, TOPK), dtype=torch.int64, device=probs.device)
        scratch_w = torch.empty((T, TOPK), dtype=torch.float32, device=probs.device)
        K.route.decode_moe_route_probs_q13_cuda(probs, bias, candidate_eid, scratch_w)
        candidate_mask = torch.zeros_like(probs, dtype=torch.bool)
        candidate_mask.scatter_(1, candidate_eid, True)
        sel = probs + bias
        sel_masked = torch.where(candidate_mask, sel, sel.new_full((), float('-inf')))
    else:
        # The specialized kernel is decode-only; preserve the original route
        # for large prefill batches.
        sel = probs + bias                                         # [T,256]
        selg = sel.view(T, N_GROUP, GROUP_SIZE)                    # [T,8,32]
        gscore = selg.topk(2, dim=-1).values.sum(-1)              # [T,8]
        keep = gscore.topk(GROUP_TOPK, dim=-1).indices           # [T,4]
        gmask = torch.zeros(T, N_GROUP, dtype=torch.bool, device=probs.device)
        gmask.scatter_(1, keep, True)
        cand = gmask.unsqueeze(-1).expand(T, N_GROUP, GROUP_SIZE).reshape(T, N_EXPERT)
        sel_masked = torch.where(cand, sel, sel.new_full((), float('-inf')))
    eid = sel_masked.topk(TOPK, dim=-1).indices                  # [T,8] expert ids
    w = probs.gather(1, eid)                                      # weight from probs
    s = w.sum(-1, keepdim=True)
    inv = torch.where(s > 6.103515625e-5, ROUTED_SCALING / s, s.new_zeros(()))
    return eid.to(torch.int64), (w * inv).to(torch.float32)


def ffn_forward(rt: TPRuntime, layer_w, x: List[torch.Tensor], layer: int) -> None:
    """Dense FFN (layers 0-2) or MoE (rest); special layers dispatch by idx.
    Traffic: exactly ONE all-reduce on the combined output."""
    from model.model import _moe_experts, _special_moe_experts
    out: List[torch.Tensor] = [None] * TP  # type: ignore
    is_special = layer in SPECIAL_LAYERS

    def _rank(r: int):
        h = x[r]
        if isinstance(layer_w, DenseFFN):
            g = K.q8_matmul(h, layer_w.gate[r], LOCAL_FF)
            u = K.q8_matmul(h, layer_w.up[r], LOCAL_FF)
            act = torch.nn.functional.silu(g) * u                 # placeholder for fused silu-mul
            out[r] = K.q8_matmul(act, layer_w.down[r], D)         # [T,D] partial
            return
        # MoE
        moe: MoE = layer_w
        # shared expert (always on), rank-local
        sg = K.q8_matmul(h, moe.gate_shexp[r], LOCAL_SHARED_FF)
        su = K.q8_matmul(h, moe.up_shexp[r], LOCAL_SHARED_FF)
        shared = K.q8_matmul(torch.nn.functional.silu(sg) * su, moe.down_shexp[r], D)
        # routing is replicated (same on every rank); compute once per rank cheaply
        scores = _router_scores(h, moe)                           # [T, N_EXPERT] probs
        eid, ew = _grouped_topk(scores, moe.bias[r])
        if is_special:
            routed = _special_moe_experts(h, moe, layer, eid, ew)
        else:
            routed = _moe_experts(h, moe.gate_exps[r], moe.up_exps[r], moe.down_exps[r], eid, ew)
        out[r] = routed + shared                                  # [T,D] partial

    rt.run(_rank)
    rt.all_reduce(out)                                            # === ALL-REDUCE #2 ===
    for r in range(TP):
        x[r] = out[r]


def _router_scores(h: torch.Tensor, moe: MoE) -> torch.Tensor:
    """Router gate: probs = sigmoid(h @ router^T) [T, N_EXPERT] fp32.
    router is fp16 replicated (ffn_gate_inp.weight); compute in fp32. Bias is
    NOT added here (selection adds it in _grouped_topk; weights use raw probs)."""
    r = h.device.index
    logits = torch.matmul(h.to(torch.float32), moe.router[r].to(torch.float32).t())
    return torch.sigmoid(logits)


def block_forward(rt: TPRuntime, blk: Layer, x: List[torch.Tensor],
                  kv: KVCache, positions, *,
                  kv_layer: Optional[int] = None) -> None:
    """One transformer block: pre-norm attn (+residual), pre-norm ffn (+residual).

    Residual clone/add and FFN pre-norm must run on the same per-rank CUDA
    streams as attn/ffn/NCCL.  Host-side default-stream ops race with the
    asynchronous all-reduces and produce post-allreduce rank divergence.
    """
    # attn sub-block: the fused forward_rank kernel applies Attn.norm internally,
    # so pass the RAW residual here (no pre-norm on the caller side).
    res: List[Optional[torch.Tensor]] = [None] * TP

    def _save_attn(r: int) -> None:
        res[r] = x[r].clone()

    rt.run(_save_attn)
    attn_forward(rt, blk.attn, x, kv,
                 blk.idx if kv_layer is None else kv_layer,
                 positions)

    # The attention all-reduce has completed before this run.  Fuse the
    # residual add with the following FFN residual save/norm: both operations
    # are per-rank and ordered on the same CUDA stream, so this removes one
    # host barrier per layer without changing the numerical order.
    def _add_attn_save_norm(r: int) -> None:
        x[r] = res[r] + x[r]
        res[r] = x[r].clone()
        x[r] = _rmsnorm(x[r], blk.ffn.norm[r])

    rt.run(_add_attn_save_norm)
    ffn_forward(rt, blk.ffn, x, blk.idx)

    def _add_ffn(r: int) -> None:
        x[r] = res[r] + x[r]

    rt.run(_add_ffn)


def embed_tokens(rt: TPRuntime, w: Weights, input_ids: torch.Tensor) -> List[torch.Tensor]:
    """Q6_K row lookup + all-gather -> replicated [T,D] fp16 on every rank."""
    ids = torch.as_tensor(input_ids, dtype=torch.int32, device="cpu").reshape(-1)
    if not ids.numel() or ids.numel() != input_ids.shape[-1]:
        raise ValueError("prefill expects one non-empty token sequence")
    if int(ids.min()) < 0 or int(ids.max()) >= VOCAB:
        raise IndexError("token id outside vocabulary")

    T, local = ids.numel(), D // TP
    parts: List[torch.Tensor] = [None] * TP  # type: ignore
    rt.run(lambda r: parts.__setitem__(
        r, K.q6.lookup(w.embed[r], ids.to(rt.devices[r]).contiguous())))

    gathered = [torch.empty((TP, T, local), dtype=torch.bfloat16,
                            device=rt.devices[r]) for r in range(TP)]
    torch.cuda.nccl.all_gather(parts, gathered, streams=rt.streams, comms=rt.comms)

    out: List[torch.Tensor] = [None] * TP  # type: ignore
    rt.run(lambda r: out.__setitem__(
        r, gathered[r].permute(1, 0, 2).to(torch.float16).reshape(T, D)))
    return out


def lm_head(rt: TPRuntime, w: Weights, hidden: torch.Tensor) -> torch.Tensor:
    """Explicit post-prefill projection: hidden [..., D] -> fp32 [VOCAB]."""
    if w.lm_head is None:
        raise RuntimeError("output.weight was not loaded")
    if hidden.shape[-1] != D or hidden.device != torch.device(rt.devices[0]):
        raise ValueError("lm_head expects hidden [...,6144] on rank 0")

    # final_norm is intentionally resident only on rank 0.  Produce one normalized
    # row there, then send its eight contiguous hidden slices to their weight ranks.
    with torch.cuda.device(rt.devices[0]):
        rt.streams[0].wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(rt.streams[0]):
            h = _rmsnorm(hidden.reshape(-1, D)[-1:], w.final_norm)[0]
            h = h.to(torch.float16).contiguous()
            norm_ready = torch.cuda.Event()
            norm_ready.record()

    local = D // TP
    partial: List[torch.Tensor] = [None] * TP  # type: ignore

    def _rank(r: int):
        if r:
            rt.streams[r].wait_event(norm_ready)
        x_r = h[r * local:(r + 1) * local].to(
            device=w.lm_head[r].device, non_blocking=True).contiguous()
        partial[r] = K.q6.matvec(w.lm_head[r], x_r)

    rt.run(_rank)
    torch.cuda.nccl.reduce(
        partial, root=0, op=torch.cuda.nccl.SUM,
        streams=rt.streams, comms=rt.comms)
    with torch.cuda.device(rt.devices[0]):
        torch.cuda.current_stream().wait_stream(rt.streams[0])
    return partial[0]


def prefill(engine: Engine, input_ids: torch.Tensor, *, cache_start: int = 0,
            cache: Optional[KVCache] = None) -> torch.Tensor:
    """Run embedding + 78 transformer blocks and return hidden [T, D] on rank 0.

    ``cache_start=0`` is ordinary prefill.  A positive value appends this block
    after an already-valid MLA cache prefix; positions and causal masking begin
    there.  Final norm and LM head remain outside the timed prefill engine.
    ``cache`` selects the target KV (engine.kv by default, or a request-local
    paged view for batched rows).
    """
    rt, w = engine.rt, engine.w
    kv = engine.kv if cache is None else cache
    kv.validate_page_tables()
    T = input_ids.shape[-1]
    end = cache_start + T
    if cache_start < 0 or end > kv.max_len:
        raise ValueError(
            f"prefill cache range [{cache_start}, {end}) exceeds KV capacity {kv.max_len}")
    if cache_start > kv.length:
        raise ValueError(
            f"cache_start {cache_start} exceeds valid KV length {kv.length}")
    kv.length = cache_start
    positions: List[Optional[torch.Tensor]] = [None] * TP

    def _make_positions(r: int) -> None:
        positions[r] = torch.arange(cache_start, end, dtype=torch.int64,
                                    device=f"cuda:{rt.devices[r]}")

    rt.run(_make_positions)
    x = embed_tokens(rt, w, input_ids)
    for blk in w.layers:
        block_forward(rt, blk, x, kv, positions)
        # Bound cuBLAS Q8 dequantized-weight lifetime to one completed layer.
        K.q8.clear_weight_cache()
    # Public prefill completion contract: returned hidden/KV must be safe for
    # callers on any CUDA stream, not merely ordered on the rank-side streams.
    for stream in rt.streams:
        stream.synchronize()
    kv.length = end
    return x[0]


def _prefill_chunk_tokens(engine: Engine, chunk_tokens: Optional[int] = None) -> int:
    """Resolve the per-row base/MTP prefill chunk budget."""
    if chunk_tokens is not None:
        n = int(chunk_tokens)
    else:
        n = int(getattr(engine, "prefill_chunk_tokens", DEFAULT_PREFILL_CHUNK_TOKENS))
    if n <= 0:
        raise ValueError("prefill chunk tokens must be positive")
    return n


def prefill_sequence_chunked(
        engine: Engine,
        input_ids: torch.Tensor,
        *,
        cache_start: int = 0,
        cache: Optional[KVCache] = None,
        mtp_cache: Optional[KVCache] = None,
        chunk_tokens: Optional[int] = None) -> torch.Tensor:
    """Chunked base prefill (+ optional interior MTP) for one sequence.

    ``input_ids`` is the uncached suffix beginning at absolute ``cache_start``.
    Base KV is written through ``cache`` (default ``engine.kv``).  When
    ``mtp_cache`` is provided and the MTP head is loaded, prompt-shifted MTP
    slots ``[cache_start, cache_start + T - 1)`` are filled; the caller must
    still prime the final slot with the first generated token.  Returns the
    residual of the last base chunk ``[C, D]`` on rank 0 (only the last row is
    required by ``lm_head`` / final MTP prime).
    """
    from model.decode_backend import mtp_forward
    ids = torch.as_tensor(input_ids, dtype=torch.long).reshape(-1)
    T = int(ids.numel())
    if T <= 0:
        raise ValueError("prefill_sequence_chunked expects a non-empty suffix")
    if int(ids.min()) < 0 or int(ids.max()) >= VOCAB:
        raise IndexError("token id outside vocabulary")

    chunk = _prefill_chunk_tokens(engine, chunk_tokens)
    base_cache = engine.kv if cache is None else cache
    # Preserve the proven all-at-once path exactly when no split is needed.
    # Besides avoiding wrapper overhead for short prompts, this prevents any
    # graph/cache lifetime change from being introduced by the chunk scheduler.
    if T <= chunk and mtp_cache is None:
        return prefill(engine, ids, cache_start=cache_start, cache=cache)
    do_mtp = (
        mtp_cache is not None
        and engine.w.mtp is not None
        and engine.mtp_kv is not None
    )
    if do_mtp:
        # Match base prefix validity: private cache may already hold [0, cache_start).
        if cache_start > mtp_cache.length:
            raise ValueError(
                f"MTP cache_start {cache_start} exceeds valid length {mtp_cache.length}")
        mtp_cache.length = cache_start

    last_resid: Optional[torch.Tensor] = None
    pending_hidden: Optional[torch.Tensor] = None  # final-normed 1 x D
    pending_abs: Optional[int] = None

    for rel in range(0, T, chunk):
        abs_start = cache_start + rel
        end = min(rel + chunk, T)
        piece = ids[rel:end]
        resid = prefill(
            engine, piece, cache_start=abs_start, cache=base_cache)
        # Cross-chunk MTP boundary: previous last residual + first token of this piece.
        if do_mtp and pending_hidden is not None:
            tok = piece[:1].to(dtype=torch.long)
            mtp_forward(
                engine, tok, int(pending_abs), pending_hidden, cache=mtp_cache)
            pending_hidden = None
            pending_abs = None
        # Interior MTP for this chunk: positions abs_start .. abs_start+C-2.
        c_len = int(piece.numel())
        if do_mtp and c_len > 1:
            hidden = _rmsnorm(resid[:-1], engine.w.final_norm).to(torch.float16)
            mtp_forward(
                engine, piece[1:].to(dtype=torch.long), abs_start, hidden,
                cache=mtp_cache)
        if end < T:
            # Defer the chunk-final residual until the next chunk's first token.
            pending_hidden = _rmsnorm(resid[-1:], engine.w.final_norm).to(
                torch.float16)
            pending_abs = abs_start + c_len - 1
        last_resid = resid
        del resid

    if last_resid is None:
        raise RuntimeError("chunked prefill produced no residual")
    if pending_hidden is not None:
        # Should only happen if the suffix is empty after a pending boundary.
        raise RuntimeError("chunked MTP left an unresolved boundary residual")
    return last_resid


def prefill_batch_chunked(engine: Engine,
                          input_ids: Union[torch.Tensor, Sequence[torch.Tensor],
                                           Sequence[Sequence[int]]], *,
                          max_lengths: Optional[Sequence[int]] = None,
                          chunk_tokens: Optional[int] = None,
                          page_indices: Optional[Sequence[Sequence[int]]] = None,
                          loaded_lengths: Optional[Sequence[int]] = None
                          ) -> Tuple[List[torch.Tensor], BatchPrefillState]:
    """Memory-bounded batch prefill: rows run serially, each in base chunks.

    Batching exists for decode; prefill peak memory is what kills the process,
    so this path never flattens two prompts into one launch.  Row ``i`` is
    prefilled alone in ``chunk_tokens``-token pieces (default
    ``engine.prefill_chunk_tokens``) into its own disjoint page run, and the
    interior MTP stream is filled in the same pass.  All rows then enter the
    shared batched decode state exactly as the all-at-once path does.

    Returns ``(last_chunk_residual_per_sequence, state)``.  Only the final row
    of each residual is meaningful (``lm_head`` / final MTP prime); earlier rows
    of the last chunk are returned as-is to avoid an extra copy.
    """
    from model.batch_decode import _batch_request_caches, _alloc_mtp_caches
    from model.decode_backend import ensure_mtp_graphs
    if isinstance(input_ids, torch.Tensor):
        if input_ids.dim() == 1:
            sequences = [input_ids]
        elif input_ids.dim() == 2:
            sequences = [input_ids[i] for i in range(input_ids.shape[0])]
        else:
            raise ValueError("prefill_batch_chunked input must be [T] or [B,T]")
    else:
        sequences = [torch.as_tensor(seq, dtype=torch.long).reshape(-1)
                     for seq in input_ids]
    if not sequences or any(seq.numel() == 0 for seq in sequences):
        raise ValueError("prefill_batch_chunked expects non-empty sequences")

    lengths = [int(seq.numel()) for seq in sequences]
    loaded = ([0] * len(lengths) if loaded_lengths is None
              else [int(n) for n in loaded_lengths])
    if len(loaded) != len(lengths) or any(n < 0 or n > length
                                          for n, length in zip(loaded, lengths)):
        raise ValueError("loaded_lengths must cover each row within its prompt")
    state = _batch_request_caches(
        engine, lengths, max_lengths, page_indices=page_indices)
    want_mtp = engine.w.mtp is not None and engine.mtp_kv is not None
    if want_mtp:
        # Warm capture writes owner MTP slots 0..1, so it must happen before any
        # request-local MTP KV exists; chunked prefill writes real slots below.
        ensure_mtp_graphs(engine)
    try:
        if want_mtp:
            state.mtp_caches.extend(_alloc_mtp_caches(
                engine, state.capacities, page_indices=state.page_indices))
        residuals: List[torch.Tensor] = []
        for i, seq in enumerate(sequences):
            # Recompute the last cached token so prefill always produces the
            # boundary residual/logits and refreshes the final shifted-MTP slot.
            cache_start = max(0, loaded[i] - 1)
            state.caches[i].length = cache_start
            mtp_cache = state.mtp_caches[i] if want_mtp else None
            if mtp_cache is not None:
                mtp_cache.length = cache_start
            resid = prefill_sequence_chunked(
                engine, seq[cache_start:], cache_start=cache_start,
                cache=state.caches[i], mtp_cache=mtp_cache,
                chunk_tokens=chunk_tokens)
            if int(state.caches[i].length) != lengths[i]:
                raise RuntimeError(
                    "chunked batch prefill left an inconsistent KV length")
            residuals.append(resid)
        return residuals, state
    except Exception:
        state.release()
        raise
