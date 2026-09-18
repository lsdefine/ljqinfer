"""Qwen3.8-27B DFlash2 sidecar for the standalone TP4 engine.

The target and the BF16 drafter both use TP4.  Hidden states stay replicated,
while QKV/gate-up are column-parallel, O/down/input-FC are row-parallel, and
each rank owns only its local attention heads and DFlash KV pages.  Target
auxiliary hidden states are projected into the draft context cache, while
speculative query KV is throw-away.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from pathlib import Path
from typing import Sequence
import json
import math

import torch
import torch_npu
from safetensors import safe_open

from ops.kernels import K


DFLASH_DIR = Path("/data/models/Qwen3.8-27B-DFlash2")
TARGET_LAYERS = (5, 19, 33, 47, 61)
BLOCK_SIZE = 8
NUM_DRAFT = BLOCK_SIZE - 1
MASK_TOKEN_ID = 248070
TOP_K = 16
HIDDEN = 5120
INTERMEDIATE = 17408
N_HEADS = 32
N_KV_HEADS = 8
TP_WORLD = 4
LOCAL_HEADS = N_HEADS // TP_WORLD
LOCAL_KV_HEADS = N_KV_HEADS // TP_WORLD
LOCAL_INTERMEDIATE = INTERMEDIATE // TP_WORLD
HEAD_DIM = 128
WINDOW = 2048
ROPE_THETA = 10_000_000.0
EPS = 1e-6


def _rms(x, weight):
    """DFlash checkpoints store the effective RMS weight directly."""
    return torch_npu.npu_rms_norm(x, weight, epsilon=EPS)[0]


def _add_rms(x, residual, weight):
    """Mirror the target decode graph's fused residual-add + RMSNorm path."""
    normed, _rstd, summed = torch_npu.npu_add_rms_norm(
        x, residual, weight, epsilon=EPS)
    return normed, summed


def _grouped_conv(hidden, delta, base, side, taps=2, group_size=16,
                  out=None):
    """Fixed-B1Q8 grouped convolution loaded from the repository kernel wrapper."""
    if taps != 2 or group_size != 16:
        raise ValueError("DFlash grouped convolution requires tap2/group16")
    return K.dflash_grouped_conv_b1q8(hidden, delta, base, side, out=out)


@dataclass
class DraftLayer:
    input_norm: torch.Tensor
    post_norm: torch.Tensor
    qkv: torch.Tensor
    o: torch.Tensor
    q_norm: torch.Tensor
    k_norm: torch.Tensor
    gate_up: torch.Tensor
    down: torch.Tensor
    attn_base: torch.Tensor
    attn_kernel: torch.Tensor
    mlp_base: torch.Tensor
    mlp_kernel: torch.Tensor


class DFlashKVPool:
    """Shared paged DFlash KV pool with one logical page table per sequence."""

    def __init__(self, *, layers: int, max_tokens: int, page_size: int,
                 device, dtype=torch.bfloat16, kv_heads: int = N_KV_HEADS,
                 max_sequence_tokens: int | None = None,
                 max_sequences: int = 1):
        self.layers = int(layers)
        self.max_tokens = int(max_tokens)
        self.max_sequence_tokens = int(
            self.max_tokens if max_sequence_tokens is None else max_sequence_tokens)
        self.max_sequences = int(max_sequences)
        self.page_size = int(page_size)
        self.kv_heads = int(kv_heads)
        if (self.layers <= 0 or self.max_tokens <= 0 or
                self.max_sequence_tokens <= 0 or self.max_sequences <= 0 or
                self.page_size <= 0 or self.kv_heads <= 0):
            raise ValueError("invalid DFlash KV pool geometry")
        if self.max_sequence_tokens > self.max_tokens:
            raise ValueError("DFlash sequence capacity exceeds global pool")
        self.num_pages = ceil(self.max_tokens / self.page_size)
        self.logical_pages = ceil(self.max_sequence_tokens / self.page_size)
        shape = (self.layers, self.num_pages, self.page_size,
                 self.kv_heads, HEAD_DIM)
        self.k = torch.empty(shape, dtype=dtype, device=device)
        self.v = torch.empty(shape, dtype=dtype, device=device)
        self.page_table = torch.full(
            (self.max_sequences, self.logical_pages), -1,
            dtype=torch.int64, device=device)
        self.host_page_table = [
            [-1] * self.logical_pages for _ in range(self.max_sequences)]
        self.free_pages = list(range(self.num_pages - 1, -1, -1))
        export_shape = (12, self.layers, self.page_size, self.kv_heads, HEAD_DIM)
        self.cold_export_k_stage = torch.empty(
            export_shape, dtype=dtype, device=device)
        self.cold_export_v_stage = torch.empty(
            export_shape, dtype=dtype, device=device)

    def _check_sequence(self, sequence_id: int) -> int:
        sid = int(sequence_id)
        if not 0 <= sid < self.max_sequences:
            raise IndexError(f"DFlash sequence_id {sid} out of range")
        return sid

    def _physical_page(self, sequence_id: int, logical: int,
                       allocate: bool) -> int:
        sid, logical = self._check_sequence(sequence_id), int(logical)
        if not 0 <= logical < self.logical_pages:
            raise IndexError(
                f"DFlash logical page {logical} exceeds per-sequence capacity")
        physical = int(self.host_page_table[sid][logical])
        if physical < 0 and allocate:
            if not self.free_pages:
                raise MemoryError("DFlash KV page pool exhausted")
            physical = int(self.free_pages.pop())
            self.host_page_table[sid][logical] = physical
            self.page_table[sid, logical] = physical
        return physical

    def write(self, start: int, keys: Sequence[torch.Tensor],
              values: Sequence[torch.Tensor], sequence_id: int = 0) -> None:
        if len(keys) != self.layers or len(values) != self.layers:
            raise ValueError("DFlash layer count mismatch")
        total = int(keys[0].shape[0]) if keys else 0
        expected_tail = (self.kv_heads, HEAD_DIM)
        for key, value in zip(keys, values):
            if key.shape != value.shape or tuple(key.shape[1:]) != expected_tail:
                raise ValueError("DFlash K/V must be equal [T,H,D] tensors")
            if int(key.shape[0]) != total:
                raise ValueError("DFlash K/V token count mismatch")
        start = int(start)
        if start < 0 or start + total > self.max_sequence_tokens:
            raise ValueError("DFlash KV write exceeds per-sequence capacity")
        done = 0
        while done < total:
            position = start + done
            logical, offset = divmod(position, self.page_size)
            physical = self._physical_page(sequence_id, logical, True)
            take = min(total - done, self.page_size - offset)
            for layer, (key, value) in enumerate(zip(keys, values)):
                self.k[layer, physical, offset:offset + take].copy_(
                    key[done:done + take])
                self.v[layer, physical, offset:offset + take].copy_(
                    value[done:done + take])
            done += take

    def write_layer(self, layer: int, start: int, key: torch.Tensor,
                    value: torch.Tensor, sequence_id: int = 0) -> None:
        layer, start = int(layer), int(start)
        if not 0 <= layer < self.layers:
            raise IndexError("DFlash layer out of range")
        if key.shape != value.shape or tuple(key.shape[1:]) != (
                self.kv_heads, HEAD_DIM):
            raise ValueError("DFlash layer K/V must be equal [T,H,D] tensors")
        total = int(key.shape[0])
        if start < 0 or start + total > self.max_sequence_tokens:
            raise ValueError("DFlash KV write exceeds per-sequence capacity")
        done = 0
        while done < total:
            position = start + done
            logical, offset = divmod(position, self.page_size)
            physical = self._physical_page(sequence_id, logical, True)
            take = min(total - done, self.page_size - offset)
            self.k[layer, physical, offset:offset + take].copy_(
                key[done:done + take])
            self.v[layer, physical, offset:offset + take].copy_(
                value[done:done + take])
            done += take

    def read_layer(self, layer: int, start: int, end: int,
                   sequence_id: int = 0):
        layer, start, end = int(layer), int(start), int(end)
        if not 0 <= layer < self.layers or start < 0 or end < start:
            raise ValueError("invalid DFlash KV read")
        if end > self.max_sequence_tokens:
            raise ValueError("DFlash KV read exceeds per-sequence capacity")
        keys, values = [], []
        position = start
        while position < end:
            logical, offset = divmod(position, self.page_size)
            physical = self._physical_page(sequence_id, logical, False)
            if physical < 0:
                raise RuntimeError(f"missing DFlash KV logical page {logical}")
            take = min(end - position, self.page_size - offset)
            keys.append(self.k[layer, physical, offset:offset + take])
            values.append(self.v[layer, physical, offset:offset + take])
            position += take
        if not keys:
            shape = (0, self.kv_heads, HEAD_DIM)
            return (torch.empty(shape, dtype=self.k.dtype, device=self.k.device),
                    torch.empty(shape, dtype=self.v.dtype, device=self.v.device))
        return torch.cat(keys, 0), torch.cat(values, 0)

    def export_blocks(self, start: int, end: int, sequence_id: int = 0):
        start, end = int(start), int(end)
        if (start < 0 or end <= start or start % self.page_size or
                end % self.page_size):
            raise ValueError("DFlash cold export must use complete pages")
        count = (end - start) // self.page_size
        if count > self.cold_export_k_stage.shape[0]:
            raise ValueError("DFlash cold export exceeds staging capacity")
        for block, logical in enumerate(range(
                start // self.page_size, end // self.page_size)):
            physical = self._physical_page(sequence_id, logical, False)
            if physical < 0:
                raise RuntimeError(f"missing DFlash KV logical page {logical}")
            self.cold_export_k_stage[block].copy_(self.k[:, physical])
            self.cold_export_v_stage[block].copy_(self.v[:, physical])
        keys_host = self.cold_export_k_stage[:count].detach().to("cpu")
        values_host = self.cold_export_v_stage[:count].detach().to("cpu")
        return tuple(
            (tuple(keys_host[block, layer] for layer in range(self.layers)),
             tuple(values_host[block, layer] for layer in range(self.layers)))
            for block in range(count))

    def export_block(self, start: int, end: int, sequence_id: int = 0):
        records = self.export_blocks(start, end, sequence_id)
        if len(records) != 1:
            raise ValueError("single DFlash export requires exactly one page")
        return records[0]

    def import_block(self, start: int, end: int, keys, values,
                     sequence_id: int = 0) -> None:
        """Queue one cold page directly into its final sequence-local mapping."""
        start, end = int(start), int(end)
        if (end <= start or start % self.page_size or
                end - start != self.page_size):
            raise ValueError("invalid DFlash cold block interval")
        if len(keys) != self.layers or len(values) != self.layers:
            raise ValueError("DFlash cold block layer count mismatch")
        physical = self._physical_page(
            sequence_id, start // self.page_size, True)
        for layer, (key, value) in enumerate(zip(keys, values)):
            self.k[layer, physical].copy_(key, non_blocking=True)
            self.v[layer, physical].copy_(value, non_blocking=True)

    def reserve_sequence_pages(self, sequence_id: int, capacity: int) -> tuple[int, ...]:
        """Lease DFlash pages through a row's declared decode capacity."""
        sid, capacity = self._check_sequence(sequence_id), int(capacity)
        if capacity < 0 or capacity > self.max_sequence_tokens:
            raise ValueError(
                f"DFlash capacity {capacity} outside [0,{self.max_sequence_tokens}]")
        count = (capacity + self.page_size - 1) // self.page_size
        return tuple(self._physical_page(sid, logical, True)
                     for logical in range(count))

    def page_indices(self, sequence_id: int) -> tuple[int, ...]:
        sid = self._check_sequence(sequence_id)
        return tuple(page for page in self.host_page_table[sid] if page >= 0)

    @property
    def resident_pages(self) -> int:
        return self.num_pages - len(self.free_pages)

    def release_sequence(self, sequence_id: int) -> None:
        sid = self._check_sequence(sequence_id)
        owned = [page for page in self.host_page_table[sid] if page >= 0]
        self.free_pages.extend(owned)
        self.host_page_table[sid] = [-1] * self.logical_pages
        self.page_table[sid].fill_(-1)

    def reset(self, sequence_id: int | None = None) -> None:
        if sequence_id is None:
            for sid in range(self.max_sequences):
                self.release_sequence(sid)
        else:
            self.release_sequence(sequence_id)


class _DFlashBatchGraph:
    """Fixed-B, fixed-Q DFlash graph with independent paged context per row."""

    def __init__(self, owner: "DFlash2Sidecar", batch_size: int):
        self.owner = owner
        self.batch_size = int(batch_size)
        if self.batch_size not in (2, 3, 4):
            raise ValueError("batched DFlash graph requires B=2, 3, or 4")
        self.device = owner.device
        self.graph = None
        self.anchor = torch.zeros(
            (self.batch_size,), dtype=torch.long, device=self.device)
        self.context_len = torch.zeros_like(self.anchor)
        self.page_table = torch.full(
            (self.batch_size, owner.kv_pool.logical_pages), -1,
            dtype=torch.int64, device=self.device)
        self.output = torch.empty(
            (self.batch_size, NUM_DRAFT), dtype=torch.long, device=self.device)
        self.candidate = torch.empty(
            (self.batch_size, NUM_DRAFT, TOP_K), dtype=torch.long,
            device=self.device)
        self.unary = torch.empty(
            (self.batch_size, NUM_DRAFT, TOP_K), dtype=owner.fc.dtype,
            device=self.device)
        self._context_index = torch.arange(
            WINDOW - BLOCK_SIZE, dtype=torch.long, device=self.device)
        self._mask_ids = torch.full(
            (self.batch_size, NUM_DRAFT), MASK_TOKEN_ID, dtype=torch.long,
            device=self.device)
        self._position_offsets = torch.arange(
            BLOCK_SIZE, dtype=torch.long, device=self.device)

    def prepare(self, anchors, sequence_ids) -> None:
        if len(anchors) != self.batch_size or len(sequence_ids) != self.batch_size:
            raise ValueError(f"fixed DFlash graph requires B={self.batch_size}")
        self.anchor.copy_(torch.as_tensor(
            anchors, dtype=torch.long, device=self.device))
        lengths = []
        for row, sequence_id in enumerate(sequence_ids):
            sid = self.owner.kv_pool._check_sequence(sequence_id)
            lengths.append(int(self.owner.context_lengths[sid]))
            self.page_table[row].copy_(self.owner.kv_pool.page_table[sid])
        self.context_len.copy_(torch.as_tensor(
            lengths, dtype=torch.long, device=self.device))

    def _context_plan(self):
        count = torch.clamp(
            self.context_len, min=0, max=WINDOW - BLOCK_SIZE)
        padding = WINDOW - BLOCK_SIZE - count
        first = self.context_len - count
        absolute = (first[:, None] + self._context_index[None, :] -
                    padding[:, None])
        valid = self._context_index[None, :] >= padding[:, None]
        safe = torch.clamp(absolute, min=0)
        logical = torch.div(
            safe, self.owner.kv_pool.page_size, rounding_mode="floor")
        offset = torch.remainder(safe, self.owner.kv_pool.page_size)
        physical = torch.clamp(
            self.page_table.gather(1, logical), min=0)
        return physical * self.owner.kv_pool.page_size + offset, valid

    def _context(self, layer_idx, flat, valid):
        b = self.batch_size
        c = WINDOW - BLOCK_SIZE
        shape = (-1, LOCAL_KV_HEADS, HEAD_DIM)
        key = self.owner.kv_pool.k[layer_idx].view(shape).index_select(
            0, flat.reshape(-1)).view(b, c, LOCAL_KV_HEADS, HEAD_DIM)
        value = self.owner.kv_pool.v[layer_idx].view(shape).index_select(
            0, flat.reshape(-1)).view(b, c, LOCAL_KV_HEADS, HEAD_DIM)
        # Invalid context slots are already excluded by the FIAS mask. Their
        # safe gather indices remain in-bounds, so zero-filling only adds two
        # redundant graph operators per layer.
        return key, value

    def _grouped_conv(self, hidden, delta, base, side):
        # Keep the numerically validated B1Q8 AICore kernel and its independent
        # launches, but write each row directly into one contiguous result.
        output = torch.empty_like(hidden)
        for batch_row in range(self.batch_size):
            lo, hi = batch_row * BLOCK_SIZE, (batch_row + 1) * BLOCK_SIZE
            _grouped_conv(hidden[lo:hi], delta[lo:hi], base, side,
                          out=output[lo:hi])
        return output

    def _row_linear(self, hidden, weight):
        # BF16 GEMM is shape-sensitive: one M=B*8 call differs from B calls of
        # M=8 even with identical rows. Keep B1Q8 arithmetic for each sequence.
        rows = []
        for batch_row in range(self.batch_size):
            lo, hi = batch_row * BLOCK_SIZE, (batch_row + 1) * BLOCK_SIZE
            rows.append(self.owner._linear(hidden[lo:hi], weight))
        return torch.cat(rows, dim=0)

    def _row_all_reduce(self, tensor):
        gathered = self.owner.engine.collective.all_gather(tensor)
        torch.add(gathered[0], gathered[1], out=tensor)
        tensor.add_(gathered[3])
        tensor.add_(gathered[2])
        return tensor
    def _mlp(self, hidden, layer):
        packed = self._row_linear(hidden, layer.gate_up)
        activated = torch_npu.npu_swiglu(packed, dim=-1)
        partial = self._row_linear(activated, layer.down)
        return self._row_all_reduce(partial)

    def _attention(self, hidden, positions, layer, layer_idx, context_plan):
        b = self.batch_size
        q_width = LOCAL_HEADS * HEAD_DIM
        kv_width = LOCAL_KV_HEADS * HEAD_DIM
        qkv = self._row_linear(hidden, layer.qkv)
        q, k, v = torch.split(
            qkv, (q_width, kv_width, kv_width), dim=-1)
        q = _rms(q.view(-1, LOCAL_HEADS, HEAD_DIM), layer.q_norm)
        k = _rms(k.view(-1, LOCAL_KV_HEADS, HEAD_DIM), layer.k_norm)
        v = v.view(b, BLOCK_SIZE, LOCAL_KV_HEADS, HEAD_DIM)
        q, k = K.rope(q, k, positions.reshape(-1), HEAD_DIM, ROPE_THETA)
        q = q.view(b, BLOCK_SIZE, LOCAL_HEADS, HEAD_DIM)
        k = k.view(b, BLOCK_SIZE, LOCAL_KV_HEADS, HEAD_DIM)
        context_flat, context_valid = context_plan
        ck, cv = self._context(layer_idx, context_flat, context_valid)
        kk = torch.cat((ck, k), dim=1)
        vv = torch.cat((cv, v), dim=1)
        valid = torch.cat((context_valid, torch.ones(
            (b, BLOCK_SIZE), dtype=torch.bool, device=self.device)), dim=1)
        mask = (~valid).view(b, 1, 1, WINDOW).expand(
            b, 1, BLOCK_SIZE, WINDOW)
        out, _ = torch_npu.npu_fused_infer_attention_score(
            q, kk, vv, atten_mask=mask, num_heads=LOCAL_HEADS,
            num_key_value_heads=LOCAL_KV_HEADS,
            scale=HEAD_DIM ** -0.5, input_layout="BSND", sparse_mode=0,
            inner_precise=0)
        partial = self._row_linear(
            out.reshape(b * BLOCK_SIZE, -1), layer.o)
        return self._row_all_reduce(partial)

    def _layer(self, hidden, residual, positions, layer, layer_idx,
               context_plan):
        if residual is None:
            residual = hidden
            h = _rms(hidden, layer.input_norm)
        else:
            h, residual = _add_rms(hidden, residual, layer.input_norm)
        coeff = self._row_linear(h, layer.attn_kernel).reshape(
            -1, 2, 2, HIDDEN // 16)
        h = self._grouped_conv(h, coeff[:, 0], layer.attn_base, 0)
        h = self._attention(h, positions, layer, layer_idx, context_plan)
        h = self._grouped_conv(h, coeff[:, 1], layer.attn_base, 1)
        h, residual = _add_rms(h, residual, layer.post_norm)
        coeff = self._row_linear(h, layer.mlp_kernel).reshape(
            -1, 2, 2, HIDDEN // 16)
        h = self._grouped_conv(h, coeff[:, 0], layer.mlp_base, 0)
        h = self._mlp(h, layer)
        h = self._grouped_conv(h, coeff[:, 1], layer.mlp_base, 1)
        return h, residual

    def _select_path(self, candidate, unary, hidden):
        b = self.batch_size
        hp = self.owner._linear(
            hidden.reshape(b * NUM_DRAFT, HIDDEN),
            self.owner.selector_proj).view(b, NUM_DRAFT, -1)
        successors = self.owner.succ[candidate]
        pred_ids = torch.cat((
            self.anchor[:, None, None].expand(-1, 1, TOP_K),
            candidate[:, :-1]), dim=1)
        predecessors = self.owner.pred[pred_ids]
        scores = unary[:, :, None, :] + torch.einsum(
            "blpr,blcr,blr->blpc", predecessors, successors, hp)
        previous = torch.zeros((b,), dtype=torch.long, device=self.device)
        batch_index = torch.arange(b, dtype=torch.long, device=self.device)
        output = []
        for step in range(NUM_DRAFT):
            row = scores[batch_index, step, previous]
            previous = row.argmax(-1)
            output.append(candidate[batch_index, step, previous])
        return torch.stack(output, dim=1)

    def _forward(self):
        b = self.batch_size
        ids = torch.cat((self.anchor[:, None], self._mask_ids), dim=1)
        positions = self.context_len[:, None] + self._position_offsets[None, :]
        hidden = self.owner.engine.weights.embedding.index_select(
            0, ids.reshape(-1))
        context_plan = self._context_plan()
        residual = None
        for layer_idx, layer in enumerate(self.owner.layers):
            hidden, residual = self._layer(
                hidden, residual, positions, layer, layer_idx, context_plan)
        hidden, _ = _add_rms(hidden, residual, self.owner.final_norm)
        sample_hidden = hidden.view(b, BLOCK_SIZE, HIDDEN)[:, 1:]
        flat_hidden = sample_hidden.reshape(b * NUM_DRAFT, HIDDEN)
        candidate, unary = self.owner._full_topk(flat_hidden)
        candidate = candidate.view(b, NUM_DRAFT, TOP_K)
        unary = unary.view(b, NUM_DRAFT, TOP_K)
        self.candidate.copy_(candidate)
        self.unary.copy_(unary)
        self.output.copy_(self._select_path(candidate, unary, sample_hidden))

    def capture(self, pool=None, warm=True) -> None:
        if self.graph is not None:
            return
        if warm:
            self._forward()
            self.owner._device_sync()
        graph = torch_npu.npu.NPUGraph()
        try:
            with torch_npu.npu.graph(graph, pool=pool):
                self._forward()
        except Exception:
            try:
                graph.reset()
            except Exception:
                pass
            raise
        self.owner._device_sync()
        # The first replay materializes captured outputs before any host read.
        graph.replay()
        self.owner._device_sync()
        self.graph = graph

    def replay(self) -> None:
        if self.graph is None:
            raise RuntimeError("batched DFlash graph is not captured")
        self.graph.replay()


class DFlash2Sidecar:
    """Single-sequence greedy DFlash2 sidecar.

    ``append_context`` is called with accepted target-token auxiliary features.
    ``draft`` consumes an anchor token and emits seven coherent candidates.
    """
    name = "dflash2"
    block_size = BLOCK_SIZE
    target_layer_ids = TARGET_LAYERS

    def __init__(self, engine, model_dir: str | Path = DFLASH_DIR):
        self.engine = engine
        self.device = engine.device
        self.model_dir = Path(model_dir)
        cfg = json.loads((self.model_dir / "config.json").read_text())
        dcfg = cfg["dflash_config"]
        if tuple(dcfg["target_layer_ids"]) != TARGET_LAYERS:
            raise ValueError("DFlash2 target_layer_ids mismatch")
        self.rank = int(engine.rank)
        collective = engine.collective
        self.world = int(getattr(collective, "world", 1))
        if self.world != TP_WORLD or not 0 <= self.rank < self.world:
            raise ValueError(
                f"DFlash2 TP requires world={TP_WORLD}, got rank={self.rank} "
                f"world={self.world}")
        self.store = safe_open(str(self.model_dir / "model.safetensors"),
                               framework="pt", device="cpu")
        feature_width = HIDDEN // self.world
        self.feature_start = self.rank * feature_width
        self.feature_end = self.feature_start + feature_width
        fc_full = self.store.get_tensor("fc.weight")
        fc_parts = [
            fc_full[:, i * HIDDEN + self.feature_start:
                    i * HIDDEN + self.feature_end]
            for i in range(len(TARGET_LAYERS))
        ]
        self.fc = torch.cat(fc_parts, dim=1).contiguous().to(
            self.device, non_blocking=False)
        self.hidden_norm = self._get("hidden_norm.weight")
        self.final_norm = self._get("norm.weight")
        self.layers = [self._load_layer(i) for i in range(5)]
        self.pred = self._get("candidate_selector.predecessor_codebook")
        self.succ = self._get("candidate_selector.successor_codebook")
        self.selector_proj = self._get("candidate_selector.hidden_projection.weight")
        max_tokens = int(engine.engine_config.max_cached_tokens)
        self.kv_pool = DFlashKVPool(
            layers=len(self.layers), max_tokens=max_tokens, page_size=1024,
            max_sequence_tokens=int(engine.engine_config.max_sequence_tokens),
            max_sequences=int(engine.engine_config.max_sequences),
            device=self.device, kv_heads=LOCAL_KV_HEADS)
        self.context_lengths = [0] * int(engine.engine_config.max_sequences)
        self.active_sequence_id = 0
        # Draft is fixed B1Q8.  Keep dynamic scalars and outputs at stable
        # addresses so one NPUGraph serves every context length.
        self.graph = None
        # B1 stays on the original graph. B2-B4 are captured lazily because all
        # ranks enter draft_batch in the same stable scheduler order.
        self._batch_graphs = {}
        self._graph_pool = None
        self.graph_anchor = torch.zeros((1,), dtype=torch.long, device=self.device)
        self.graph_context_len = torch.zeros((1,), dtype=torch.long, device=self.device)
        self.graph_page_table = torch.full(
            (self.kv_pool.logical_pages,), -1, dtype=torch.int64,
            device=self.device)
        self.graph_output = torch.empty((NUM_DRAFT,), dtype=torch.long,
                                        device=self.device)
        self.graph_candidate = torch.empty((NUM_DRAFT, TOP_K), dtype=torch.long,
                                           device=self.device)
        self.graph_unary = torch.empty((NUM_DRAFT, TOP_K), dtype=self.fc.dtype,
                                       device=self.device)
        self._graph_context_index = torch.arange(
            WINDOW - BLOCK_SIZE, dtype=torch.long, device=self.device)
        self._graph_mask_ids = torch.full(
            (NUM_DRAFT,), MASK_TOKEN_ID, dtype=torch.long, device=self.device)
        # Decode append is fixed at at most B1Q8 rows. Stable graph inputs and
        # outputs remove the eager launch chain without coupling to verify graphs.
        self.append_graph = None
        self.append_feature_in = [
            torch.zeros((BLOCK_SIZE, HIDDEN), dtype=self.fc.dtype,
                        device=self.device)
            for _ in TARGET_LAYERS]
        self.append_context_start = torch.zeros(
            (1,), dtype=torch.long, device=self.device)
        self.append_key_out = [
            torch.empty((BLOCK_SIZE, LOCAL_KV_HEADS, HEAD_DIM),
                        dtype=self.fc.dtype, device=self.device)
            for _ in self.layers]
        self.append_value_out = [torch.empty_like(key)
                                 for key in self.append_key_out]
        self._append_offsets = torch.arange(
            BLOCK_SIZE, dtype=torch.long, device=self.device)

    def _get(self, name):
        return self.store.get_tensor(name).to(self.device, non_blocking=False)

    def _get_rows(self, name, start, end):
        weight = self.store.get_tensor(name)[start:end].contiguous()
        return weight.to(self.device, non_blocking=False)

    def _get_cols(self, name, start, end):
        weight = self.store.get_tensor(name)[:, start:end].contiguous()
        return weight.to(self.device, non_blocking=False)

    def _get_cat_rows(self, specs):
        # Slice and merge on CPU so device memory never holds full projections.
        weights = tuple(
            self.store.get_tensor(name)[start:end]
            for name, start, end in specs)
        return torch.cat(weights, dim=0).contiguous().to(
            self.device, non_blocking=False)

    def _all_reduce(self, tensor):
        return self.engine.all_reduce(tensor)

    def _load_layer(self, i):
        p = f"layers.{i}."
        q0 = self.rank * LOCAL_HEADS * HEAD_DIM
        q1 = q0 + LOCAL_HEADS * HEAD_DIM
        kv0 = self.rank * LOCAL_KV_HEADS * HEAD_DIM
        kv1 = kv0 + LOCAL_KV_HEADS * HEAD_DIM
        mlp0 = self.rank * LOCAL_INTERMEDIATE
        mlp1 = mlp0 + LOCAL_INTERMEDIATE
        return DraftLayer(
            self._get(p + "input_layernorm.weight"),
            self._get(p + "post_attention_layernorm.weight"),
            self._get_cat_rows((
                (p + "self_attn.q_proj.weight", q0, q1),
                (p + "self_attn.k_proj.weight", kv0, kv1),
                (p + "self_attn.v_proj.weight", kv0, kv1),
            )),
            self._get_cols(p + "self_attn.o_proj.weight", q0, q1),
            self._get(p + "self_attn.q_norm.weight"),
            self._get(p + "self_attn.k_norm.weight"),
            self._get_cat_rows((
                (p + "mlp.gate_proj.weight", mlp0, mlp1),
                (p + "mlp.up_proj.weight", mlp0, mlp1),
            )),
            self._get_cols(p + "mlp.down_proj.weight", mlp0, mlp1),
            self._get(p + "attention_conv.base_kernel"),
            self._get(p + "attention_conv.kernel_projection.weight"),
            self._get(p + "mlp_conv.base_kernel"),
            self._get(p + "mlp_conv.kernel_projection.weight"),
        )

    @staticmethod
    def _linear(x, w):
        # Use the same BF16 linear entry point as the target decode graph.
        return K.bf16_linear(x, w)

    def _mlp(self, hidden, layer):
        # Exact BF16 analogue of K.swiglu_mlp: packed column projection,
        # native NPU SwiGLU, row projection, then the target collective.
        packed = K.bf16_linear(hidden, layer.gate_up)
        activated = torch_npu.npu_swiglu(packed, dim=-1)
        return self._all_reduce(K.bf16_linear(activated, layer.down))

    def combine_features(self, features: Sequence[torch.Tensor]):
        if len(features) != len(TARGET_LAYERS):
            raise ValueError(f"need {len(TARGET_LAYERS)} target features")
        local = torch.cat(tuple(
            feature[:, self.feature_start:self.feature_end]
            for feature in features), dim=-1)
        if local.shape[-1] != self.fc.shape[-1]:
            raise ValueError(
                f"local feature width {local.shape[-1]} != {self.fc.shape[-1]}")
        return self._all_reduce(self._linear(local, self.fc))

    def _project_kv(self, hidden, positions, layer: DraftLayer):
        h = _rms(hidden, self.hidden_norm)
        q_width = LOCAL_HEADS * HEAD_DIM
        kv_width = LOCAL_KV_HEADS * HEAD_DIM
        kv = self._linear(h, layer.qkv[q_width:])
        k, v = kv.split(kv_width, dim=-1)
        k = k.view(-1, LOCAL_KV_HEADS, HEAD_DIM)
        v = v.view(-1, LOCAL_KV_HEADS, HEAD_DIM)
        k = _rms(k, layer.k_norm)
        dummy_q = torch.zeros((k.shape[0], 1, HEAD_DIM), dtype=k.dtype,
                              device=k.device)
        _, k = K.rope(dummy_q, k, positions, HEAD_DIM, ROPE_THETA)
        return k, v

    def append_context(self, features: Sequence[torch.Tensor], positions: torch.Tensor,
                       sequence_id: int = 0):
        """Commit arbitrary prefill rows for one sequence."""
        sid = self.kv_pool._check_sequence(sequence_id)
        h = self.combine_features(features)
        positions = positions.to(self.device, dtype=torch.long)
        if h.shape[0] != positions.numel():
            raise ValueError("feature/position row mismatch")
        start = int(self.context_lengths[sid])
        expected = torch.arange(start, start + int(h.shape[0]), dtype=torch.long,
                                device=self.device)
        if not torch.equal(positions, expected):
            raise ValueError("DFlash context positions must be contiguous")
        for layer_index, layer in enumerate(self.layers):
            key, value = self._project_kv(h, positions, layer)
            self.kv_pool.write_layer(
                layer_index, start, key, value, sequence_id=sid)
            del key, value
        self.context_lengths[sid] = start + int(h.shape[0])

    def _append_graph_forward(self):
        h = self.combine_features(self.append_feature_in)
        positions = self.append_context_start[0] + self._append_offsets
        for i, layer in enumerate(self.layers):
            key, value = self._project_kv(h, positions, layer)
            self.append_key_out[i].copy_(key)
            self.append_value_out[i].copy_(value)

    def capture_append(self, warm=True):
        """Capture the fixed B1Q8 target-feature to draft-KV projection."""
        if self.append_graph is not None:
            return
        if warm:
            self._append_graph_forward()
            self._device_sync()
        graph = torch_npu.npu.NPUGraph()
        try:
            with torch_npu.npu.graph(graph):
                self._append_graph_forward()
        except Exception:
            try:
                graph.reset()
            except Exception:
                pass
            raise
        self._device_sync()
        graph.replay()
        self._device_sync()
        self.append_graph = graph

    def append_context_graph(self, features: Sequence[torch.Tensor],
                             sequence_id: int = 0):
        """Project one accepted decode prefix with a fixed-width replay."""
        if self.append_graph is None:
            raise RuntimeError("DFlash append graph is not captured")
        if len(features) != len(TARGET_LAYERS):
            raise ValueError(f"need {len(TARGET_LAYERS)} target features")
        rows = int(features[0].shape[0]) if features else 0
        if not 0 < rows <= BLOCK_SIZE:
            raise ValueError(f"decode append rows must be in [1,{BLOCK_SIZE}]")
        for dst, src in zip(self.append_feature_in, features):
            if int(src.shape[0]) != rows or tuple(src.shape[1:]) != (HIDDEN,):
                raise ValueError("decode append feature shape mismatch")
            dst[:rows].copy_(src)
        sid = self.kv_pool._check_sequence(sequence_id)
        start = int(self.context_lengths[sid])
        self.append_context_start.fill_(start)
        self.append_graph.replay()
        self.kv_pool.write(
            start,
            [key[:rows] for key in self.append_key_out],
            [value[:rows] for value in self.append_value_out],
            sequence_id=sid)
        self.context_lengths[sid] = start + rows

    def _context(self, i, sequence_id: int = 0):
        sid = self.kv_pool._check_sequence(sequence_id)
        end = int(self.context_lengths[sid])
        start = max(0, end - WINDOW)
        return self.kv_pool.read_layer(i, start, end, sequence_id=sid)

    def export_cold_block(self, start: int, end: int,
                          sequence_id: int = 0):
        return self.kv_pool.export_block(start, end, sequence_id=sequence_id)

    def import_cold_block(self, start: int, end: int, keys, values,
                          sequence_id: int = 0) -> None:
        sid = self.kv_pool._check_sequence(sequence_id)
        self.kv_pool.import_block(
            start, end, keys, values, sequence_id=sid)
        self.context_lengths[sid] = max(self.context_lengths[sid], int(end))

    def _attention(self, hidden, positions, layer, layer_idx,
                   sequence_id: int = 0):
        q_width = LOCAL_HEADS * HEAD_DIM
        kv_width = LOCAL_KV_HEADS * HEAD_DIM
        qkv = self._linear(hidden, layer.qkv)
        q, k, v = torch.split(qkv, (q_width, kv_width, kv_width), dim=-1)
        q = q.view(-1, LOCAL_HEADS, HEAD_DIM)
        k = k.view(-1, LOCAL_KV_HEADS, HEAD_DIM)
        v = v.view(-1, LOCAL_KV_HEADS, HEAD_DIM)
        q = _rms(q, layer.q_norm)
        k = _rms(k, layer.k_norm)
        q, k = K.rope(q, k, positions, HEAD_DIM, ROPE_THETA)
        ck, cv = self._context(layer_idx, sequence_id)
        kk, vv = torch.cat((ck, k), 0), torch.cat((cv, v), 0)
        if kk.shape[0] > WINDOW:
            kk, vv = kk[-WINDOW:], vv[-WINDOW:]
        out, _ = torch_npu.npu_fused_infer_attention_score(
            q.unsqueeze(0), kk.unsqueeze(0), vv.unsqueeze(0),
            num_heads=LOCAL_HEADS, num_key_value_heads=LOCAL_KV_HEADS,
            scale=HEAD_DIM ** -0.5, input_layout="BSND", sparse_mode=0,
            inner_precise=0)
        partial = self._linear(out.reshape(hidden.shape[0], -1), layer.o)
        return self._all_reduce(partial)

    def _graph_context_plan(self):
        """Map the fixed context window to flattened physical KV rows once."""
        count = torch.clamp(self.graph_context_len[0], min=0,
                            max=WINDOW - BLOCK_SIZE)
        padding = WINDOW - BLOCK_SIZE - count
        first = self.graph_context_len[0] - count
        valid = self._graph_context_index >= padding
        absolute = first + self._graph_context_index - padding
        safe = torch.clamp(absolute, min=0)
        logical = torch.div(safe, self.kv_pool.page_size,
                            rounding_mode="floor")
        offset = torch.remainder(safe, self.kv_pool.page_size)
        physical = torch.clamp(self.graph_page_table[logical], min=0)
        return physical * self.kv_pool.page_size + offset, valid

    def _graph_context(self, layer_idx, flat, valid):
        """Gather one layer's local-head context from physical KV rows."""
        shape = (-1, LOCAL_KV_HEADS, HEAD_DIM)
        key = self.kv_pool.k[layer_idx].view(shape).index_select(0, flat)
        value = self.kv_pool.v[layer_idx].view(shape).index_select(0, flat)
        # Invalid context slots are already excluded by the FIAS mask. Their
        # safe gather indices remain in-bounds, so zero-filling only adds two
        # redundant graph operators per layer.
        return key, value

    def _graph_attention(self, hidden, positions, layer, layer_idx,
                         context_plan):
        q_width = LOCAL_HEADS * HEAD_DIM
        kv_width = LOCAL_KV_HEADS * HEAD_DIM
        qkv = self._linear(hidden, layer.qkv)
        q, k, v = torch.split(qkv, (q_width, kv_width, kv_width), dim=-1)
        q = q.view(-1, LOCAL_HEADS, HEAD_DIM)
        k = k.view(-1, LOCAL_KV_HEADS, HEAD_DIM)
        v = v.view(-1, LOCAL_KV_HEADS, HEAD_DIM)
        q = _rms(q, layer.q_norm)
        k = _rms(k, layer.k_norm)
        q, k = K.rope(q, k, positions, HEAD_DIM, ROPE_THETA)
        context_flat, context_valid = context_plan
        ck, cv = self._graph_context(layer_idx, context_flat, context_valid)
        kk = torch.cat((ck, k), 0)
        vv = torch.cat((cv, v), 0)
        valid = torch.cat((context_valid, torch.ones(
            (BLOCK_SIZE,), dtype=torch.bool, device=self.device)))
        mask = (~valid).view(1, 1, 1, WINDOW).expand(
            1, 1, BLOCK_SIZE, WINDOW)
        out, _ = torch_npu.npu_fused_infer_attention_score(
            q.unsqueeze(0), kk.unsqueeze(0), vv.unsqueeze(0),
            atten_mask=mask, num_heads=LOCAL_HEADS,
            num_key_value_heads=LOCAL_KV_HEADS,
            scale=HEAD_DIM ** -0.5, input_layout="BSND", sparse_mode=0,
            inner_precise=0)
        partial = self._linear(out.reshape(hidden.shape[0], -1), layer.o)
        return self._all_reduce(partial)

    def _graph_layer(self, hidden, residual, positions, layer, layer_idx,
                     context_plan):
        if residual is None:
            residual = hidden
            h = _rms(hidden, layer.input_norm)
        else:
            h, residual = _add_rms(hidden, residual, layer.input_norm)
        coeff = self._linear(h, layer.attn_kernel).reshape(
            -1, 2, 2, HIDDEN // 16)
        h = _grouped_conv(h, coeff[:, 0], layer.attn_base, 0)
        h = self._graph_attention(
            h, positions, layer, layer_idx, context_plan)
        h = _grouped_conv(h, coeff[:, 1], layer.attn_base, 1)
        h, residual = _add_rms(h, residual, layer.post_norm)
        coeff = self._linear(h, layer.mlp_kernel).reshape(
            -1, 2, 2, HIDDEN // 16)
        h = _grouped_conv(h, coeff[:, 0], layer.mlp_base, 0)
        h = self._mlp(h, layer)
        h = _grouped_conv(h, coeff[:, 1], layer.mlp_base, 1)
        return h, residual

    def _layer(self, hidden, residual, positions, layer, layer_idx,
               sequence_id: int = 0):
        if residual is None:
            residual = hidden
            h = _rms(hidden, layer.input_norm)
        else:
            h, residual = _add_rms(hidden, residual, layer.input_norm)
        coeff = self._linear(h, layer.attn_kernel).reshape(-1, 2, 2, HIDDEN // 16)
        h = _grouped_conv(h, coeff[:, 0], layer.attn_base, 0)
        h = self._attention(h, positions, layer, layer_idx, sequence_id)
        h = _grouped_conv(h, coeff[:, 1], layer.attn_base, 1)
        h, residual = _add_rms(h, residual, layer.post_norm)
        coeff = self._linear(h, layer.mlp_kernel).reshape(-1, 2, 2, HIDDEN // 16)
        h = _grouped_conv(h, coeff[:, 0], layer.mlp_base, 0)
        h = self._mlp(h, layer)
        h = _grouped_conv(h, coeff[:, 1], layer.mlp_base, 1)
        return h, residual

    def _full_topk(self, hidden):
        local = self.engine.local_logits(hidden)
        vals, idx = torch.topk(local, TOP_K, dim=-1)
        idx = idx + self.engine.weights.vocab_start
        world = self.engine.collective.world
        n = hidden.shape[0]
        gathered_v = self.engine.collective.all_gather(vals.contiguous())
        # The standalone HCCL ABI concatenates ranks along dim 0.  Token IDs
        # travel as fp32 because that is supported by every deployed HCCL build.
        gathered_i = self.engine.collective.all_gather(idx.float().contiguous()).long()
        vals = gathered_v.reshape(world, n, TOP_K).permute(1, 0, 2).reshape(n, -1)
        idx = gathered_i.reshape(world, n, TOP_K).permute(1, 0, 2).reshape(n, -1)
        vals, which = torch.topk(vals, TOP_K, dim=-1)
        return idx.gather(-1, which), vals

    def _select_path(self, candidate, unary, hidden, anchor):
        # scores[step, previous_candidate, current_candidate]
        hp = self._linear(hidden, self.selector_proj)
        successors = self.succ[candidate]
        pred_ids = torch.cat((torch.full_like(candidate[:1], int(anchor)), candidate[:-1]), 0)
        predecessors = self.pred[pred_ids]
        scores = unary[:, None, :] + torch.einsum("lpr,lcr,lr->lpc",
                                                  predecessors, successors, hp)
        out, previous = [], 0
        for step in range(NUM_DRAFT):
            previous = int(scores[step, previous].argmax().item())
            out.append(int(candidate[step, previous].item()))
        return out

    @torch.inference_mode()
    def draft_eager(self, anchor_token: int, sequence_id: int = 0):
        sid = self.kv_pool._check_sequence(sequence_id)
        self.active_sequence_id = sid
        ids = torch.tensor([anchor_token] + [MASK_TOKEN_ID] * NUM_DRAFT,
                           dtype=torch.long, device=self.device)
        ctx = int(self.context_lengths[sid])
        positions = torch.arange(ctx, ctx + BLOCK_SIZE,
                                 dtype=torch.long, device=self.device)
        hidden = self.engine.weights.embedding.index_select(0, ids)
        residual = None
        for i, layer in enumerate(self.layers):
            hidden, residual = self._layer(
                hidden, residual, positions, layer, i, sid)
        hidden, _ = _add_rms(hidden, residual, self.final_norm)
        sample_hidden = hidden[1:]
        candidate, unary = self._full_topk(sample_hidden)
        path = self._select_path(candidate, unary, sample_hidden, anchor_token)
        return path, candidate, unary

    def _graph_select_path(self, candidate, unary, hidden):
        hp = self._linear(hidden, self.selector_proj)
        successors = self.succ[candidate]
        anchor = self.graph_anchor.view(1, 1).expand_as(candidate[:1])
        pred_ids = torch.cat((anchor, candidate[:-1]), 0)
        predecessors = self.pred[pred_ids]
        scores = unary[:, None, :] + torch.einsum(
            "lpr,lcr,lr->lpc", predecessors, successors, hp)
        previous = torch.zeros((), dtype=torch.long, device=self.device)
        output = []
        for step in range(NUM_DRAFT):
            row = scores[step].index_select(0, previous.reshape(1))[0]
            previous = row.argmax(-1)
            output.append(candidate[step].index_select(
                0, previous.reshape(1))[0])
        return torch.stack(output)

    def _graph_forward(self):
        ids = torch.cat((self.graph_anchor, self._graph_mask_ids))
        positions = self.graph_context_len[0] + torch.arange(
            BLOCK_SIZE, dtype=torch.long, device=self.device)
        hidden = self.engine.weights.embedding.index_select(0, ids)
        context_plan = self._graph_context_plan()
        residual = None
        for i, layer in enumerate(self.layers):
            hidden, residual = self._graph_layer(
                hidden, residual, positions, layer, i, context_plan)
        hidden, _ = _add_rms(hidden, residual, self.final_norm)
        sample_hidden = hidden[1:]
        candidate, unary = self._full_topk(sample_hidden)
        self.graph_candidate.copy_(candidate)
        self.graph_unary.copy_(unary)
        self.graph_output.copy_(self._graph_select_path(
            candidate, unary, sample_hidden))

    def _device_sync(self):
        torch.npu.synchronize(self.device)

    def capture(self, pool=None, warm=True):
        if self.graph is not None:
            return
        self._graph_pool = pool
        if warm:
            self._graph_forward()
            self._device_sync()
        graph = torch_npu.npu.NPUGraph()
        try:
            with torch_npu.npu.graph(graph, pool=pool):
                self._graph_forward()
        except Exception:
            try:
                graph.reset()
            except Exception:
                pass
            raise
        self._device_sync()
        graph.replay()
        self._device_sync()
        self.graph = graph

    @torch.inference_mode()
    def draft(self, anchor_token: int, sequence_id: int = 0):
        sid = self.kv_pool._check_sequence(sequence_id)
        if self.graph is None:
            return self.draft_eager(anchor_token, sequence_id=sid)
        self.active_sequence_id = sid
        self.graph_anchor.fill_(int(anchor_token))
        self.graph_context_len.fill_(int(self.context_lengths[sid]))
        self.graph_page_table.copy_(self.kv_pool.page_table[sid])
        self.graph.replay()
        self._device_sync()
        return (self.graph_output.cpu().tolist(), self.graph_candidate,
                self.graph_unary)

    def capture_batch(self, batch_size: int, pool=None, warm=True):
        """Capture one fixed B2-B4 draft graph outside request latency."""
        batch = int(batch_size)
        if batch not in (2, 3, 4):
            raise ValueError("batched DFlash graph size must be in [2,4]")
        graph = self._batch_graphs.get(batch)
        if graph is None:
            graph = _DFlashBatchGraph(self, batch)
            graph.capture(pool=self._graph_pool if pool is None else pool,
                          warm=warm)
            self._batch_graphs[batch] = graph
        return graph

    @torch.inference_mode()
    def draft_batch(self, anchor_tokens, sequence_ids, active_rows=None):
        """Draft active rows while preserving the scheduler's stable row order."""
        anchors = [int(token) for token in anchor_tokens]
        sids = [int(sid) for sid in sequence_ids]
        if not anchors or len(anchors) != len(sids):
            raise ValueError("anchor_tokens and sequence_ids must match")
        if len(anchors) > 4:
            raise ValueError("DFlash decode batch size must be in [1,4]")
        if len(set(sids)) != len(sids):
            raise ValueError("sequence_ids must be unique")
        active = ([True] * len(sids) if active_rows is None else
                  [bool(value) for value in active_rows])
        if len(active) != len(sids):
            raise ValueError("active_rows must match sequence_ids")
        if len(anchors) == 1:
            return (self.draft(anchors[0], sequence_id=sids[0])
                    if active[0] else None,)

        batch = len(anchors)
        graph = self.capture_batch(batch)
        graph.prepare(anchors, sids)
        graph.replay()
        self._device_sync()
        paths = graph.output.cpu().tolist()
        return tuple(
            None if not active[row] else
            (paths[row], graph.candidate[row], graph.unary[row])
            for row in range(batch))

    @torch.inference_mode()
    def draft_batch_device(self, anchor_tokens, sequence_ids, active_rows=None):
        """Replay draft and expose resident anchors/paths without synchronizing."""
        anchors = [int(token) for token in anchor_tokens]
        sids = [int(sid) for sid in sequence_ids]
        if not anchors or len(anchors) != len(sids):
            raise ValueError("anchor_tokens and sequence_ids must match")
        if len(anchors) > 4:
            raise ValueError("DFlash decode batch size must be in [1,4]")
        if len(set(sids)) != len(sids):
            raise ValueError("sequence_ids must be unique")
        active = ([True] * len(sids) if active_rows is None else
                  [bool(value) for value in active_rows])
        if len(active) != len(sids):
            raise ValueError("active_rows must match sequence_ids")
        if len(anchors) == 1:
            if not active[0]:
                raise ValueError("single-row device draft requires an active row")
            if self.graph is None:
                raise RuntimeError("single-row device draft requires a captured graph")
            sid = self.kv_pool._check_sequence(sids[0])
            self.active_sequence_id = sid
            self.graph_anchor.fill_(anchors[0])
            self.graph_context_len.fill_(int(self.context_lengths[sid]))
            self.graph_page_table.copy_(self.kv_pool.page_table[sid])
            self.graph.replay()
            return self.graph_anchor, self.graph_output.view(1, NUM_DRAFT)

        graph = self.capture_batch(len(anchors))
        graph.prepare(anchors, sids)
        graph.replay()
        return graph.anchor, graph.output

    def append_context_graph_batch(self, feature_rows, sequence_ids) -> None:
        """Commit accepted target features to each independent draft row."""
        rows = tuple(feature_rows)
        sids = [int(sid) for sid in sequence_ids]
        if len(rows) != len(sids):
            raise ValueError("feature_rows and sequence_ids must match")
        for features, sid in zip(rows, sids):
            if features is not None:
                self.append_context_graph(features, sequence_id=sid)

    @property
    def context_len(self) -> int:
        """Compatibility view of the selected sequence length."""
        return int(self.context_lengths[self.active_sequence_id])

    @context_len.setter
    def context_len(self, value: int) -> None:
        self.context_lengths[self.active_sequence_id] = int(value)

    def reset(self, sequence_id: int | None = None):
        if sequence_id is None:
            self.kv_pool.reset()
            self.context_lengths[:] = [0] * len(self.context_lengths)
            self.active_sequence_id = 0
            return
        sid = self.kv_pool._check_sequence(sequence_id)
        self.kv_pool.reset(sid)
        self.context_lengths[sid] = 0
        if self.active_sequence_id == sid:
            self.active_sequence_id = 0
