"""Stable TP4 Q8+DFlash2 execution API consumed by the strategy layer."""
from __future__ import annotations

from dataclasses import dataclass, field
import gc
from typing import Callable, Optional, Sequence
import os
import threading
import time
import torch

from .config import CONFIG, EngineConfig
from .decode_graph import DecodeGraphRunner
from .dflash2 import DFlash2Sidecar, WINDOW as DFLASH_WINDOW
from .model import Engine
from .spmd import SPMDRuntime

EOS_TOKEN_IDS = {248044, 248046}
VERIFY_WIDTH = 8
MIN_VERIFY_CAPACITY = 512
RESIDENT_VERIFY_CAPACITIES = (512, 1024)


@dataclass(frozen=True)
class BoardingRequest:
    """One FIFO request restored by the strategy for safe-point boarding."""

    input_ids: tuple[int, ...]
    max_new_tokens: int
    cancel_event: threading.Event


@dataclass(frozen=True)
class BatchPrefillRow:
    """One resident row produced by serial batched prefill."""
    sequence_id: int
    input_length: int
    restored_length: int
    capacity: int
    last_hidden: torch.Tensor
    target_page_table: tuple[int, ...]
    dflash_page_table: tuple[int, ...]

    @property
    def target_resident_pages(self) -> tuple[int, ...]:
        return tuple(page for page in self.target_page_table if page >= 0)

    @property
    def dflash_resident_pages(self) -> tuple[int, ...]:
        return tuple(page for page in self.dflash_page_table if page >= 0)


@dataclass
class BatchPrefillState:
    """Successful all-row prefill transaction and decode handoff owner.

    Target KV/GDN and DFlash KV remain resident in their shared physical pools.
    The state owns every listed sequence until :meth:`release`; failed batch
    construction never returns a state and rolls every begun row back.
    """
    owner: "ModelExecution"
    rows: tuple[BatchPrefillRow, ...]
    released: bool = False

    def _live(self) -> None:
        if self.released:
            raise RuntimeError("batch prefill state has been released")

    @property
    def sequence_ids(self) -> tuple[int, ...]:
        return tuple(row.sequence_id for row in self.rows)

    @property
    def lengths(self) -> tuple[int, ...]:
        return tuple(row.input_length for row in self.rows)

    @property
    def restored_lengths(self) -> tuple[int, ...]:
        return tuple(row.restored_length for row in self.rows)

    @property
    def capacities(self) -> tuple[int, ...]:
        return tuple(row.capacity for row in self.rows)

    def export_prefix_records(self, row: int, start: int, end: int):
        self._live()
        return self.owner._export_prefix_records(
            start, end, self.rows[int(row)].sequence_id)

    def decode_handoff(self) -> dict:
        """Return resident views/metadata needed by a future B-dimensional graph.

        Page-table tensors and GDN tensors are views into graph-stable owner
        allocations; consumers must copy into their persistent graph inputs,
        never rebind a captured tensor to these Python views.
        """
        self._live()
        cache = self.owner.engine.cache
        draft = self.owner.drafter.kv_pool
        sids = self.sequence_ids
        return {
            "sequence_ids": sids,
            "lengths": self.lengths,
            "capacities": self.capacities,
            "last_hidden": torch.cat(
                tuple(row.last_hidden for row in self.rows), dim=0),
            "target_page_tables": tuple(cache.page_table[sid] for sid in sids),
            "dflash_page_tables": tuple(draft.page_table[sid] for sid in sids),
            "gdn_conv": tuple(cache.hot_gdn_conv[:, sid] for sid in sids),
            "gdn_recurrent": tuple(
                cache.hot_gdn_recurrent[:, sid] for sid in sids),
        }

    def merge(self, other: "BatchPrefillState") -> "BatchPrefillState":
        """Transfer two disjoint live leases into one epoch owner.

        No target or DFlash page is returned here.  The retired input states are
        made inert and the merged state becomes the sole owner of every SID, so
        dynamic alighting cannot accidentally recycle pages before epoch end.
        """
        self._live()
        other._live()
        if other.owner is not self.owner:
            raise ValueError("cannot merge batch states from different executions")
        overlap = set(self.sequence_ids).intersection(other.sequence_ids)
        if overlap:
            raise ValueError(f"cannot merge overlapping sequence ids: {sorted(overlap)}")
        merged = BatchPrefillState(self.owner, self.rows + other.rows)
        for sid in merged.sequence_ids:
            current = self.owner._batch_sequence_owners.get(sid)
            if current is not self and current is not other:
                raise RuntimeError(f"sequence slot {sid} changed owner during merge")
        for sid in merged.sequence_ids:
            self.owner._batch_sequence_owners[sid] = merged
        self.released = True
        other.released = True
        return merged

    def release(self) -> None:
        if self.released:
            return
        # Mark first so accidental re-entry cannot double-return physical pages.
        self.released = True
        for sid in self.sequence_ids:
            if self.owner._batch_sequence_owners.get(sid) is not self:
                continue
            self.owner.drafter.reset(sid)
            self.owner.engine.release_sequence(sid)
            del self.owner._batch_sequence_owners[sid]


@dataclass
class ModelExecution:
    engine: Engine
    rt: SPMDRuntime
    verify: DecodeGraphRunner
    drafter: DFlash2Sidecar
    _batch_sequence_owners: dict[int, BatchPrefillState] = field(
        default_factory=dict, init=False, repr=False)

    @classmethod
    def startup(cls, rank: Optional[int] = None) -> "ModelExecution":
        rank = int(os.environ.get("LJQ_SPMD_RANK", "0") if rank is None else rank)
        cfg = EngineConfig()
        device_index = int(cfg.devices[rank])
        rt = SPMDRuntime(rank=rank, world=CONFIG.tp, device_index=device_index)
        rt.communicator()
        engine = Engine.load(rank, cfg, device=str(rt.device), collective=rt)
        rt.barrier()

        # DFlash weights are replicated.  Keep the two common short-context
        # capacities resident; larger power-of-two buckets remain epoch-local so
        # the 262K context contract does not imply a rectangular B4 arena.
        drafter = DFlash2Sidecar(engine)
        rt.barrier()
        resident_capacities = cls._resident_verify_capacities(
            cfg.max_sequence_tokens)
        initial_capacity = resident_capacities[0]
        verify_buckets = {}
        for capacity in resident_capacities:
            for batch in (1, 2, 3, 4):
                verify_buckets[(batch, capacity)] = DecodeGraphRunner(
                    engine, batch_size=batch, max_prefix=capacity,
                    num_layers=CONFIG.num_hidden_layers,
                    query_width=VERIFY_WIDTH)
        verify = verify_buckets[(1, initial_capacity)]
        # Warm every resident geometry before the first shared graph pool is
        # occupied; otherwise a later capacity can silently capture a slow path.
        for runner in verify_buckets.values():
            runner.warm()
            rt.barrier()
        verify.capture(warm=False)
        rt.barrier()
        # Draft and verify alternate, so all resident B/capacity graphs safely
        # share one pool while persistent inputs/outputs stay independently owned.
        graph_pool = verify.graph.pool()
        for key, runner in verify_buckets.items():
            if key == (1, initial_capacity):
                continue
            runner.capture(pool=graph_pool, warm=False)
            rt.barrier()
        drafter.capture(pool=graph_pool, warm=True)
        rt.barrier()
        for batch in (2, 3, 4):
            drafter.capture_batch(batch, pool=graph_pool, warm=True)
            rt.barrier()
        # Append contains a TP collective and owns stable inputs/outputs. Capture
        # independently after all shared-pool lazy warmup has completed.
        drafter.capture_append(warm=True)
        rt.barrier()
        runtime = cls(engine, rt, verify, drafter)
        runtime._verify_buckets = verify_buckets
        # Fixed prefill inputs cap request-time HBM at one target chunk.  The
        # Python prompt is materialized only on CPU and copied into these views.
        width = int(cfg.prefill_chunk_size)
        runtime._prefill_token_in = torch.empty(
            (width,), dtype=torch.int64, device=rt.device)
        runtime._prefill_position_in = torch.empty_like(runtime._prefill_token_in)
        runtime._prefill_position_base = torch.arange(
            width, dtype=torch.int64, device=rt.device)
        runtime._prefill_sequence_in = torch.zeros_like(runtime._prefill_token_in)
        runtime._prefill_last_hidden = torch.empty(
            (1, CONFIG.hidden_size), dtype=torch.bfloat16, device=rt.device)
        return runtime

    @staticmethod
    def _resident_verify_capacities(context_capacity: int) -> tuple[int, ...]:
        limit = max(1, int(context_capacity))
        capacities = tuple(
            capacity for capacity in RESIDENT_VERIFY_CAPACITIES
            if capacity <= limit)
        return capacities or (limit,)

    @property
    def device(self):
        return self.rt.device

    def _trim_dynamic_epoch_allocator(self) -> None:
        """Return allocations released by transient graphs to the NPU driver."""
        self.rt.synchronize()
        gc.collect()
        torch.npu.empty_cache()
        self.rt.synchronize()

    @property
    def context_capacity(self) -> int:
        configured = getattr(
            self.engine.engine_config, "max_sequence_tokens",
            self.engine.engine_config.max_cached_tokens)
        return min(int(configured), int(CONFIG.max_position_embeddings))

    def _verify_capacity(self, token_count: int) -> int:
        required = max(1, int(token_count))
        if required > self.context_capacity:
            raise ValueError(
                f"engine context requires prompt+max_new <= {self.context_capacity}, "
                f"got {required}")
        capacity = min(MIN_VERIFY_CAPACITY, self.context_capacity)
        while capacity < required:
            capacity = min(capacity * 2, self.context_capacity)
        return capacity

    def _select_verify(
            self, token_count: int, batch_size: int = 1, *,
            epoch_cache: Optional[
                dict[tuple[int, int], DecodeGraphRunner]] = None
            ) -> DecodeGraphRunner:
        batch = int(batch_size)
        if batch not in (1, 2, 3, 4):
            raise ValueError("decode batch_size must be in [1,4]")
        capacity = self._verify_capacity(token_count)
        key = (batch, capacity)
        verify = self._verify_buckets.get(key)
        if verify is None and epoch_cache is not None:
            verify = epoch_cache.get(key)
        if verify is None:
            verify = DecodeGraphRunner(
                self.engine, batch_size=batch, max_prefix=capacity,
                num_layers=CONFIG.num_hidden_layers, query_width=VERIFY_WIDTH)
            verify.capture(warm=True)
            self.rt.barrier()
            if epoch_cache is None:
                self._verify_buckets[key] = verify
            else:
                epoch_cache[key] = verify
        self.verify = verify
        return verify

    def _global_argmax_rows(self, local_logits) -> list[int]:
        local = local_logits.contiguous()
        gathered = self.rt.all_gather(local)
        self.rt.synchronize()
        rows = int(local.shape[0])
        full = (gathered.reshape(self.rt.world, rows, -1)
                .permute(1, 0, 2).reshape(rows, -1))
        return [int(x) for x in full.float().argmax(-1).cpu().tolist()]

    def _broadcast_boarding_request(
            self, local: Optional[BoardingRequest]) -> Optional[BoardingRequest]:
        """Broadcast one rank-0 safe-point admission without control-plane races."""
        if self.rt.rank != 0 and local is not None:
            raise ValueError("only rank 0 may originate a boarding request")
        # Raw HCCL bindings support floating dtypes only.  These control values
        # are all well below float32's exact-integer limit (2**24).
        meta = torch.zeros(4, dtype=torch.float32, device=self.rt.device)
        if local is not None:
            meta.copy_(torch.tensor(
                [1, len(local.input_ids), int(local.max_new_tokens),
                 int(local.cancel_event.is_set())],
                dtype=torch.float32, device=self.rt.device))
        gathered = self.rt.all_gather(meta.contiguous())
        self.rt.synchronize()
        has_request, length, limit, cancelled = (
            int(x) for x in gathered[0].reshape(-1).cpu().tolist())
        if not has_request:
            return None
        if length <= 0 or limit < 0:
            raise RuntimeError("rank-0 boarding metadata is invalid")
        tokens = torch.zeros(length, dtype=torch.float32, device=self.rt.device)
        if local is not None:
            tokens.copy_(torch.tensor(
                local.input_ids, dtype=torch.float32, device=self.rt.device))
        gathered_tokens = self.rt.all_gather(tokens.contiguous())
        self.rt.synchronize()
        input_ids = tuple(
            int(x) for x in gathered_tokens[0].reshape(-1).cpu().tolist())
        event = local.cancel_event if local is not None else threading.Event()
        if cancelled:
            event.set()
        return BoardingRequest(input_ids, limit, event)

    def _boarding_slot(self, state: BatchPrefillState, prompt_length: int,
                       max_new_tokens: int) -> Optional[int]:
        """Return a free SID only when both physical pools cover the epoch."""
        capacity = int(prompt_length) + int(max_new_tokens)
        if prompt_length <= 0 or max_new_tokens < 0 or capacity > self.context_capacity:
            return None
        used = set(state.sequence_ids)
        max_sequences = int(self.engine.engine_config.max_sequences)
        sid = next((slot for slot in range(max_sequences) if slot not in used), None)
        if sid is None:
            return None
        capacities = list(state.capacities) + [capacity]
        target_page_size = int(self.engine.cache.spec.page_size)
        draft_page_size = int(self.drafter.kv_pool.page_size)
        target_pages = sum((item + target_page_size - 1) // target_page_size
                           for item in capacities)
        draft_pages = sum((item + draft_page_size - 1) // draft_page_size
                          for item in capacities)
        if target_pages > int(self.engine.cache.spec.num_pages):
            return None
        if draft_pages > int(self.drafter.kv_pool.num_pages):
            return None
        return sid

    def _broadcast_int_rows(self, local_rows: Optional[Sequence[int]], *,
                            capacity: int) -> list[int]:
        """Broadcast one rank-0 row/id list so followers stay in lockstep."""
        if capacity <= 0:
            raise ValueError("broadcast capacity must be positive")
        if self.rt.rank != 0 and local_rows is not None:
            raise ValueError("only rank 0 may originate broadcast rows")
        payload = torch.full((capacity + 1,), -1, dtype=torch.float32,
                             device=self.rt.device)
        if local_rows is not None:
            rows = [int(row) for row in local_rows]
            if len(rows) > capacity:
                raise ValueError("broadcast rows exceed capacity")
            payload[0] = len(rows)
            if rows:
                payload[1:1 + len(rows)] = torch.tensor(
                    rows, dtype=torch.float32, device=self.rt.device)
        gathered = self.rt.all_gather(payload.contiguous())
        self.rt.synchronize()
        values = [int(x) for x in gathered[0].reshape(-1).cpu().tolist()]
        count = values[0]
        if count < 0 or count > capacity:
            raise RuntimeError("rank-0 row broadcast metadata is invalid")
        return values[1:1 + count]

    def _prefill_with_aux(self, input_ids: Sequence[int], sequence_id: int = 0):
        ids = torch.as_tensor(input_ids, dtype=torch.int64, device=self.device).reshape(-1)
        if not ids.numel():
            raise ValueError("input_ids must not be empty")
        total = int(ids.numel())
        chunk = int(self.engine.engine_config.prefill_chunk_size)
        hidden_out = torch.empty((total, CONFIG.hidden_size), dtype=torch.bfloat16,
                                 device=self.device)
        aux_out = [torch.empty_like(hidden_out) for _ in self.verify.aux_layer_ids]
        for lo in range(0, total, chunk):
            hi = min(total, lo + chunk)
            hidden, aux = self.engine.forward_tokens(
                ids[lo:hi], host_sequence_id=int(sequence_id), return_aux=True,
                aux_layer_ids=self.verify.aux_layer_ids,
                collect_gdn_checkpoints=True)
            if len(aux) != len(aux_out):
                raise RuntimeError("target auxiliary feature count mismatch")
            hidden_out[lo:hi].copy_(hidden)
            for slot, feature in enumerate(aux):
                aux_out[slot][lo:hi].copy_(feature)
            self.engine.cache.flush_gdn_checkpoints()
        self.engine.cache.wait_gdn_checkpoints()
        return hidden_out, aux_out

    def _boundary_hidden_rows(self, sequence_id: int):
        pool = self.engine.cache.cold_boundary_hidden_host
        if pool is None:
            raise RuntimeError("cold boundary hidden pool is not allocated")
        # Compatibility with small unit fakes created before the pool gained a
        # sequence dimension.  Production allocation is always three-dimensional.
        return pool if pool.dim() == 2 else pool[int(sequence_id)]

    def _stream_prefill(self, input_ids: Sequence[int], absolute_start: int,
                        sequence_id: int = 0) -> torch.Tensor:
        """Prefill fixed-size chunks and consume every auxiliary row immediately.

        Only one target chunk and one DFlash projection layer are transient on
        device.  Prompt tokens remain on CPU; boundary hidden rows are copied
        directly into the preallocated pinned-host pool.
        """
        ids_host = torch.as_tensor(input_ids, dtype=torch.int64,
                                   device="cpu").reshape(-1)
        total = int(ids_host.numel())
        if not total:
            raise ValueError("input_ids must not be empty")
        chunk = int(self.engine.engine_config.prefill_chunk_size)
        block = int(self.engine.cache.spec.checkpoint_interval)
        boundary_pool = self._boundary_hidden_rows(sequence_id)
        last_hidden = self._prefill_last_hidden
        for lo in range(0, total, chunk):
            width = min(chunk, total - lo)
            absolute = int(absolute_start) + lo
            token_view = self._prefill_token_in[:width]
            position_view = self._prefill_position_in[:width]
            sequence_view = self._prefill_sequence_in[:width]
            token_view.copy_(ids_host[lo:lo + width], non_blocking=False)
            torch.add(self._prefill_position_base[:width], absolute,
                      out=position_view)
            sequence_view.fill_(int(sequence_id))
            hidden, aux = self.engine.forward_tokens(
                token_view, sequence_ids=sequence_view,
                positions=position_view, host_sequence_id=int(sequence_id),
                return_aux=True, aux_layer_ids=self.verify.aux_layer_ids,
                collect_gdn_checkpoints=True)
            if len(aux) != len(self.verify.aux_layer_ids):
                raise RuntimeError("target auxiliary feature count mismatch")
            self.drafter.append_context(
                aux, position_view, sequence_id=sequence_id)
            end = absolute + width
            first_boundary = ((absolute // block) + 1) * block
            for boundary in range(first_boundary, end + 1, block):
                boundary_pool[boundary // block - 1].copy_(
                    hidden[boundary - absolute - 1], non_blocking=True)
            last_hidden.copy_(hidden[-1:])
            self.engine.cache.flush_gdn_checkpoints()
            del hidden, aux
        self.engine.cache.wait_gdn_checkpoints()
        return last_hidden

    def _load_verify_rows(self, state: BatchPrefillState,
                          graph: DecodeGraphRunner, start_row: int = 0) -> None:
        """Load freshly-prefilled paged rows into consecutive graph rows."""
        state._live()
        start = int(start_row)
        if start < 0 or start + len(state.rows) > graph.batch_size:
            raise ValueError("prefill rows do not fit destination verify graph")
        cache = self.engine.cache
        for offset, row in enumerate(state.rows):
            graph_row = start + offset
            length = row.input_length
            if length > graph.max_prefix:
                raise ValueError("verify graph does not cover prefill row")
            graph.gdn_conv_in[:, graph_row].copy_(
                cache.hot_gdn_conv[:, row.sequence_id])
            graph.gdn_rec_in[:, graph_row].copy_(
                cache.hot_gdn_recurrent[:, row.sequence_id])
            for slot in range(len(CONFIG.full_attention_layers)):
                k, v = cache.read_kv(slot, row.sequence_id, length)
                graph.k_in[slot, graph_row, :length].copy_(k)
                graph.v_in[slot, graph_row, :length].copy_(v)

    @staticmethod
    def _copy_verify_rows(source: DecodeGraphRunner,
                          destination: DecodeGraphRunner,
                          source_rows: Sequence[int]) -> None:
        """Copy committed graph rows, including safe in-place compaction."""
        rows = [int(row) for row in source_rows]
        if len(set(rows)) != len(rows):
            raise ValueError("source verify rows must be unique")
        if any(row < 0 or row >= source.batch_size for row in rows):
            raise ValueError("source verify row outside graph batch")
        if len(rows) > destination.batch_size:
            raise ValueError("destination verify graph is too small")
        if destination.max_prefix < source.max_prefix:
            raise ValueError("destination verify graph cannot shrink prefix capacity")
        if destination is source and rows != sorted(rows):
            raise ValueError("in-place verify compaction must preserve row order")
        for output_row, source_row in enumerate(rows):
            if destination is source and output_row == source_row:
                continue
            destination.gdn_conv_in[:, output_row].copy_(
                source.gdn_conv_in[:, source_row])
            destination.gdn_rec_in[:, output_row].copy_(
                source.gdn_rec_in[:, source_row])
            destination.k_in[:, output_row, :source.max_prefix].copy_(
                source.k_in[:, source_row, :source.max_prefix])
            destination.v_in[:, output_row, :source.max_prefix].copy_(
                source.v_in[:, source_row, :source.max_prefix])

    def _rebind_verify_rows(
            self, source: DecodeGraphRunner, source_rows: Sequence[int],
            boarded: Optional[BatchPrefillState] = None,
            required_capacity: Optional[int] = None, *,
            epoch_cache: Optional[
                dict[tuple[int, int], DecodeGraphRunner]] = None
            ) -> DecodeGraphRunner:
        """Compact survivors and append freshly-prefilled rows at a safe point."""
        if source._pending:
            raise RuntimeError("cannot rebind verify rows with pending candidates")
        kept = [int(row) for row in source_rows]
        boarded_count = 0 if boarded is None else len(boarded.rows)
        batch = len(kept) + boarded_count
        if batch not in (1, 2, 3, 4):
            raise ValueError("rebound decode batch size must be in [1,4]")
        required = max(source.max_prefix, int(required_capacity or 0))
        if boarded is not None:
            boarded._live()
            required = max(required, *(row.capacity for row in boarded.rows))
        graph = self._select_verify(
            required, batch_size=batch, epoch_cache=epoch_cache)
        if graph is not source:
            graph.reset()
        self._copy_verify_rows(source, graph, kept)
        if graph is source:
            for tensor in (graph.gdn_conv_in, graph.gdn_rec_in,
                           graph.k_in, graph.v_in):
                tensor[:, len(kept):].zero_()
        if boarded is not None:
            self._load_verify_rows(boarded, graph, start_row=len(kept))
        self.rt.synchronize()
        self.verify = graph
        return graph

    def _prime_verify_batch(self, state: BatchPrefillState,
                            verify: Optional[DecodeGraphRunner] = None
                            ) -> DecodeGraphRunner:
        """Copy resident prefill rows into one stable B-dimensional graph."""
        state._live()
        rows = state.rows
        batch = len(rows)
        if batch not in (1, 2, 3, 4):
            raise ValueError("decode batch size must be in [1,4]")
        required = max(row.capacity for row in rows)
        graph = (self._select_verify(required, batch_size=batch)
                 if verify is None else verify)
        if graph.batch_size != batch or graph.max_prefix < required:
            raise ValueError("verify graph does not cover batch prefill state")
        graph.reset()
        self._load_verify_rows(state, graph)
        self.rt.synchronize()
        self.verify = graph
        return graph

    def _prime_verify(self, prompt_len: int, sequence_id: int = 0) -> None:
        """Compatibility loader for the original single-sequence API."""
        self.verify.reset()
        cache = self.engine.cache
        self.verify.gdn_conv_in[:, 0].copy_(cache.hot_gdn_conv[:, sequence_id])
        self.verify.gdn_rec_in[:, 0].copy_(
            cache.hot_gdn_recurrent[:, sequence_id])
        for slot in range(len(CONFIG.full_attention_layers)):
            k, v = cache.read_kv(slot, sequence_id, prompt_len)
            self.verify.k_in[slot, 0, :prompt_len].copy_(k)
            self.verify.v_in[slot, 0, :prompt_len].copy_(v)
        self.rt.synchronize()

    def cold_record_template(self):
        """CPU pool ABI only: meta tensors allocate no model/device storage."""
        cache = self.engine.cache
        draft = self.drafter.kv_pool
        block = int(cache.spec.checkpoint_interval)
        def meta(tensor):
            return torch.empty(tuple(tensor.shape), dtype=tensor.dtype, device="meta")
        def layers(stage):
            return tuple(meta(stage[0, i]) for i in range(stage.shape[1]))
        return {
            "schema": 1, "start": 0, "end": block,
            "target": (0, block, layers(cache.cold_export_k_stage),
                       layers(cache.cold_export_v_stage),
                       meta(cache.gdn_checkpoint_conv_host[0, 0]),
                       meta(cache.gdn_checkpoint_recurrent_host[0, 0])),
            "dflash": (layers(draft.cold_export_k_stage),
                       layers(draft.cold_export_v_stage)),
            "boundary_hidden": meta(self._boundary_hidden_rows(0)[0]),
        }

    def _export_prefix_records(self, start: int, end: int,
                               sequence_id: int = 0):
        """Export completed cold blocks through fixed-capacity staging pools."""
        start, end = int(start), int(end)
        block = int(self.engine.cache.spec.checkpoint_interval)
        capacity = int(self.engine.cache.cold_export_k_stage.shape[0])
        if start < 0 or end <= start or start % block or end % block:
            raise ValueError("cold batch export requires aligned intervals")
        records = []
        for batch_start in range(start, end, capacity * block):
            batch_end = min(end, batch_start + capacity * block)
            target = self.engine.cache.export_cold_blocks(
                sequence_id, batch_start, batch_end)
            dflash = self.drafter.kv_pool.export_blocks(
                batch_start, batch_end, sequence_id=sequence_id)
            count = len(target)
            if len(dflash) != count:
                raise RuntimeError("target and DFlash cold batch sizes differ")
            hidden_pool = self._boundary_hidden_rows(sequence_id)
            first_boundary = batch_start // block
            hidden_host = hidden_pool[first_boundary:first_boundary + count]
            for index, target_record in enumerate(target):
                record_start = batch_start + index * block
                record_end = record_start + block
                records.append({
                    "schema": 1,
                    "start": record_start,
                    "end": record_end,
                    "target": target_record,
                    "dflash": dflash[index],
                    "boundary_hidden": hidden_host[index],
                })
        return tuple(records)

    def _export_prefix_record(self, start: int, end: int,
                              sequence_id: int = 0):
        """Compatibility wrapper for callers exporting one cold block."""
        records = self._export_prefix_records(start, end, sequence_id)
        if len(records) != 1:
            raise ValueError("single cold export requires exactly one interval")
        return records[0]

    def _restore_prefix_records(self, records: Sequence[dict],
                                sequence_id: int = 0):
        records = tuple(records)
        if not records:
            return 0, None
        expected = 0
        for record in records:
            if int(record.get("schema", -1)) != 1:
                raise ValueError("unsupported Qwen+DFlash cold record schema")
            start, end = int(record["start"]), int(record["end"])
            if start != expected or end <= start:
                raise ValueError("non-contiguous Qwen+DFlash cold records")
            expected = end
        restored = self.engine.cache.import_cold_blocks(
            sequence_id, tuple(record["target"] for record in records))
        if restored != expected:
            raise RuntimeError("target cold restore length mismatch")
        window_start = max(0, expected - DFLASH_WINDOW)
        for record in records:
            start, end = int(record["start"]), int(record["end"])
            if end > window_start:
                self.drafter.import_cold_block(
                    start, end, record["dflash"][0], record["dflash"][1],
                    sequence_id=sequence_id)
        self.drafter.context_lengths[int(sequence_id)] = expected
        self.drafter.active_sequence_id = int(sequence_id)
        boundary_hidden = records[-1]["boundary_hidden"].to(self.device)
        return expected, boundary_hidden

    @torch.inference_mode()
    def prefill_batch(self, input_ids: Sequence[Sequence[int]], *,
                      sequence_ids: Optional[Sequence[int]] = None,
                      max_lengths: Optional[Sequence[int]] = None,
                      restored_records: Optional[
                          Sequence[Sequence[dict]]] = None
                      ) -> BatchPrefillState:
        """Serial-row, fixed-chunk prefill into shared target/DFlash page pools.

        Rows are intentionally not fused: each row uses the production 12K
        chunk path.  A restored row resumes at exactly ``R`` because DFlash2
        cold records contain the complete boundary state (there is no shifted
        MTP ``R-1`` recomputation).  The returned state is the sole owner of all
        resident rows and is ready to be copied into stable batched-decode graph
        inputs.
        """
        sequences = [tuple(int(token) for token in row) for row in input_ids]
        if not sequences or any(not row for row in sequences):
            raise ValueError("prefill_batch expects non-empty sequences")
        batch = len(sequences)
        max_sequences = int(self.engine.engine_config.max_sequences)
        if batch > max_sequences:
            raise ValueError(
                f"batch size {batch} exceeds max_sequences {max_sequences}")

        sids = (list(range(batch)) if sequence_ids is None else
                [int(sid) for sid in sequence_ids])
        if len(sids) != batch or len(set(sids)) != batch:
            raise ValueError("sequence_ids must be unique and match batch size")
        if any(sid < 0 or sid >= max_sequences for sid in sids):
            raise ValueError("sequence_id outside configured sequence slots")
        busy = [sid for sid in sids if sid in self._batch_sequence_owners]
        if busy:
            raise RuntimeError(
                f"sequence slots already owned by a live batch state: {busy}")

        lengths = [len(row) for row in sequences]
        capacities = (list(lengths) if max_lengths is None else
                      [int(length) for length in max_lengths])
        if len(capacities) != batch:
            raise ValueError("max_lengths must match batch size")
        if any(capacity < length or capacity > self.context_capacity
               for length, capacity in zip(lengths, capacities)):
            raise ValueError(
                "each max length must cover its prompt within context capacity")

        records = ([()] * batch if restored_records is None else
                   [tuple(row) for row in restored_records])
        if len(records) != batch:
            raise ValueError("restored_records must match batch size")

        # Capacity planning includes pages currently owned by the selected rows,
        # because the transaction clears those rows before rebuilding them.
        cache = self.engine.cache
        target_owned = sum(
            sum(page >= 0 for page in cache.host_page_table[sid])
            for sid in sids)
        target_available = len(cache.free_pages) + target_owned
        target_required = sum(
            (capacity + cache.spec.page_size - 1) // cache.spec.page_size
            for capacity in capacities)
        draft_pool = self.drafter.kv_pool
        draft_owned = sum(len(draft_pool.page_indices(sid)) for sid in sids)
        draft_available = len(draft_pool.free_pages) + draft_owned
        draft_required = sum(
            (capacity + draft_pool.page_size - 1) // draft_pool.page_size
            for capacity in capacities)
        if target_required > target_available:
            raise MemoryError(
                f"batch target KV needs {target_required} pages, "
                f"only {target_available} are available")
        if draft_required > draft_available:
            raise MemoryError(
                f"batch DFlash KV needs {draft_required} pages, "
                f"only {draft_available} are available")

        # Begin the transaction by invalidating every selected row.  No state is
        # published until all rows have completed and synchronized successfully.
        for sid in sids:
            self.drafter.reset(sid)
            self.engine.release_sequence(sid)
        completed: list[BatchPrefillRow] = []
        try:
            # Lease every row's full decode capacity before computing row 0.
            # This makes success an actual resource guarantee rather than a
            # best-effort capacity estimate that later rows could consume.
            for sid, capacity in zip(sids, capacities):
                cache.reserve_sequence_pages(sid, capacity)
                draft_pool.reserve_sequence_pages(sid, capacity)
            restored_rows = []
            for ids, sid, cold in zip(sequences, sids, records):
                restored, boundary_hidden = self._restore_prefix_records(
                    cold, sid)
                if restored > len(ids):
                    raise ValueError("restored prefix exceeds input length")
                restored_rows.append((restored, boundary_hidden))
            # Cold imports queue non-blocking copies into final target/DFlash
            # pages.  Make every row resident before any suffix reads its state.
            if any(restored for restored, _ in restored_rows):
                self.rt.synchronize()

            for ids, sid, capacity, (restored, boundary_hidden) in zip(
                    sequences, sids, capacities, restored_rows):
                suffix = ids[restored:]  # DFlash2 resumes at exactly R.
                if suffix:
                    last_hidden = self._stream_prefill(
                        suffix, absolute_start=restored,
                        sequence_id=sid).clone()
                else:
                    if boundary_hidden is None:
                        raise RuntimeError(
                            "exact cold hit is missing boundary hidden")
                    last_hidden = boundary_hidden.reshape(1, -1).clone()
                committed = int(cache.lengths[sid].item())
                draft_length = int(self.drafter.context_lengths[sid])
                if committed != len(ids) or draft_length != len(ids):
                    raise RuntimeError(
                        f"inconsistent prefill row sid={sid}: "
                        f"target={committed} draft={draft_length} "
                        f"expected={len(ids)}")
                completed.append(BatchPrefillRow(
                    sequence_id=sid, input_length=len(ids),
                    restored_length=restored, capacity=capacity,
                    last_hidden=last_hidden,
                    target_page_table=tuple(cache.host_page_table[sid]),
                    dflash_page_table=tuple(
                        draft_pool.host_page_table[sid])))
            self.rt.synchronize()
            state = BatchPrefillState(self, tuple(completed))
            for sid in sids:
                self._batch_sequence_owners[sid] = state
            return state
        except Exception:
            for sid in sids:
                self.drafter.reset(sid)
                self.engine.release_sequence(sid)
            raise

    # Ref-compatible descriptive name; both entry points have identical
    # ownership and serial chunking semantics.
    def prefill_batch_chunked(self, input_ids: Sequence[Sequence[int]], **kwargs
                              ) -> BatchPrefillState:
        return self.prefill_batch(input_ids, **kwargs)

    @torch.inference_mode()
    def decode_dflash_batch(self, state: BatchPrefillState, max_new_tokens,
                            on_tokens: Optional[Callable[[int, Sequence[int]], None]] = None):
        """Decode 1-4 resident rows with one fused BxQ target verify per round."""
        if state.owner is not self:
            raise ValueError("batch prefill state belongs to another execution")
        state._live()
        batch = len(state.rows)
        if batch not in (1, 2, 3, 4):
            raise ValueError("decode batch size must be in [1,4]")
        if isinstance(max_new_tokens, int):
            limits = [int(max_new_tokens)] * batch
        else:
            limits = [int(x) for x in max_new_tokens]
        if len(limits) != batch:
            raise ValueError("max_new_tokens must have one value per batch row")
        if any(limit < 0 for limit in limits):
            raise ValueError("max_new_tokens must be non-negative")
        for row, limit in zip(state.rows, limits):
            if row.input_length + limit > row.capacity:
                raise ValueError(
                    f"row sid={row.sequence_id} capacity {row.capacity} does not "
                    f"cover prompt+decode {row.input_length + limit}")

        graph = None
        started = time.perf_counter()
        try:
            handoff = state.decode_handoff()
            graph = self._select_verify(
                max(row.capacity for row in state.rows), batch_size=batch)
            self._prime_verify_batch(state, graph)
            first = self._global_argmax_rows(
                self.engine.local_logits(handoff["last_hidden"]))
            outputs = [[token] if limits[row] else []
                       for row, token in enumerate(first)]
            for row_index, token in enumerate(first):
                if limits[row_index] and on_tokens is not None:
                    on_tokens(row_index, [token])
            current = list(first)
            committed = list(state.lengths)
            rounds = [0] * batch
            proposed = [0] * batch
            accepted = [0] * batch
            draft_s = verify_s = 0.0

            def active(row_index: int) -> bool:
                return (len(outputs[row_index]) < limits[row_index] and
                        current[row_index] not in EOS_TOKEN_IDS)

            while any(active(row) for row in range(batch)):
                active_rows = [active(row) for row in range(batch)]
                t = time.perf_counter()
                device_draft = getattr(self.drafter, "draft_batch_device", None)
                device_paths = None
                if callable(device_draft) and hasattr(graph, "prepare_draft"):
                    device_anchors, device_paths = device_draft(
                        current, state.sequence_ids, active_rows=active_rows)
                    drafted = None
                else:
                    drafted = self.drafter.draft_batch(
                        current, state.sequence_ids, active_rows=active_rows)
                draft_s += time.perf_counter() - t
                positions = [list(range(committed[row],
                                        committed[row] + VERIFY_WIDTH))
                             for row in range(batch)]
                if device_paths is not None:
                    graph.prepare_draft(current, device_anchors, device_paths,
                                        positions, active_rows=active_rows)
                    paths = None
                else:
                    paths = [list(item[0]) if item is not None else
                             [current[row]] * (VERIFY_WIDTH - 1)
                             for row, item in enumerate(drafted)]
                    queries = [[current[row]] + paths[row] for row in range(batch)]
                    graph.prepare(queries, positions)
                t = time.perf_counter()
                graph.replay()
                flat_target = self._global_argmax_rows(
                    graph.local_logits.reshape(-1, graph.local_logits.shape[-1]))
                verify_s += time.perf_counter() - t
                targets = [flat_target[row * VERIFY_WIDTH:(row + 1) * VERIFY_WIDTH]
                           for row in range(batch)]
                if device_paths is not None:
                    resident_paths = device_paths.cpu().tolist()
                    paths = [list(resident_paths[row]) if active_rows[row] else
                             [current[row]] * (VERIFY_WIDTH - 1)
                             for row in range(batch)]

                commit_counts = [0] * batch
                returned_rows = [[] for _ in range(batch)]
                feature_rows = [None] * batch
                for row in range(batch):
                    if not active(row):
                        continue
                    matched = 0
                    for draft_token, target_token in zip(
                            paths[row], targets[row][:-1]):
                        if draft_token != target_token:
                            break
                        matched += 1
                    remaining = limits[row] - len(outputs[row])
                    returned = targets[row][:min(matched + 1, remaining)]
                    eos_at = next((index for index, token in enumerate(returned)
                                   if token in EOS_TOKEN_IDS), None)
                    if eos_at is not None:
                        returned = returned[:eos_at + 1]
                    count = len(returned)
                    commit_counts[row] = count
                    returned_rows[row] = returned
                    if count:
                        feature_rows[row] = [
                            graph.aux_hidden[slot, row, :count]
                            for slot in range(len(graph.aux_layer_ids))]
                        rounds[row] += 1
                        proposed[row] += len(paths[row])
                        accepted[row] += min(matched, max(0, count - 1))

                if not any(commit_counts):
                    graph.rollback()
                    break
                graph.commit(commit_counts)
                self.drafter.append_context_graph_batch(
                    feature_rows, state.sequence_ids)
                for row, returned in enumerate(returned_rows):
                    if not returned:
                        continue
                    outputs[row].extend(returned)
                    current[row] = returned[-1]
                    committed[row] += len(returned)
                    if on_tokens is not None:
                        on_tokens(row, list(returned))

            self.rt.synchronize()
            elapsed = time.perf_counter() - started
            rows = []
            for index, prefill_row in enumerate(state.rows):
                rows.append({
                    "sequence_id": prefill_row.sequence_id,
                    "token_ids": outputs[index],
                    "cache_hit_tokens": prefill_row.restored_length,
                    "prefill_tokens": (prefill_row.input_length -
                                       prefill_row.restored_length),
                    "rounds": rounds[index],
                    "proposed_draft_tokens": proposed[index],
                    "accepted_draft_tokens": accepted[index],
                    "acceptance_rate": (accepted[index] / proposed[index]
                                        if proposed[index] else 0.0),
                    "decode_generated_tokens": max(0, len(outputs[index]) - 1),
                })
            return {
                "rows": rows,
                "batch_size": batch,
                "decode_seconds": elapsed,
                "draft_seconds": draft_s,
                "verify_seconds": verify_s,
                "decode_tps": (sum(max(0, len(row) - 1) for row in outputs) /
                               elapsed if elapsed > 0 else 0.0),
                "backend": "dflash2_batched_q8",
                "verify_width": VERIFY_WIDTH,
                "verify_capacity": graph.max_prefix,
            }
        finally:
            if graph is not None and graph._pending:
                graph.rollback()
            if graph is not None:
                graph.reset()
            state.release()

    @torch.inference_mode()
    def decode_dflash_batch_dynamic(
            self, state: BatchPrefillState, max_new_tokens, *,
            cancel_events: Optional[Sequence[threading.Event]] = None,
            on_tokens: Optional[Callable[[int, Sequence[int]], None]] = None,
            select_active_rows: Optional[
                Callable[[Sequence[int], Sequence[bool]], Sequence[int]]] = None,
            board_request: Optional[
                Callable[[int], Optional[BoardingRequest]]] = None,
            on_boarded: Optional[Callable[[int, int], None]] = None,
            boarding_interval_steps: int = 128):
        """Decode with ref-compatible safe-point alighting and boarding.

        Active verify/draft views stay in ``B∈[1,4]``.  Finished rows leave the
        active view without releasing pages; newly admitted rows merge into the
        same epoch lease.  Physical pages return only when the epoch state is
        released at the end of this call.
        """
        if state.owner is not self:
            raise ValueError("batch prefill state belongs to another execution")
        state._live()
        batch = len(state.rows)
        if batch not in (1, 2, 3, 4):
            raise ValueError("decode batch size must be in [1,4]")
        if isinstance(max_new_tokens, int):
            limits = [int(max_new_tokens)] * batch
        else:
            limits = [int(x) for x in max_new_tokens]
        if len(limits) != batch:
            raise ValueError("max_new_tokens must have one value per batch row")
        if any(limit < 0 for limit in limits):
            raise ValueError("max_new_tokens must be non-negative")
        for row, limit in zip(state.rows, limits):
            if row.input_length + limit > row.capacity:
                raise ValueError(
                    f"row sid={row.sequence_id} capacity {row.capacity} does not "
                    f"cover prompt+decode {row.input_length + limit}")
        if cancel_events is None:
            cancels = [threading.Event() for _ in range(batch)]
        else:
            cancels = list(cancel_events)
            if len(cancels) != batch:
                raise ValueError("cancel_events must have one value per batch row")
        if boarding_interval_steps <= 0:
            raise ValueError("boarding_interval_steps must be positive")

        graph = None
        epoch = state
        epoch_verify_cache: dict[
            tuple[int, int], DecodeGraphRunner] = {}
        touched_verify: list[DecodeGraphRunner] = []

        def track_verify(runner: DecodeGraphRunner) -> DecodeGraphRunner:
            if all(existing is not runner for existing in touched_verify):
                touched_verify.append(runner)
            return runner

        started = time.perf_counter()
        draft_s = verify_s = 0.0
        steps = 0
        try:
            handoff = epoch.decode_handoff()
            graph = track_verify(self._select_verify(
                max(row.capacity for row in epoch.rows), batch_size=batch,
                epoch_cache=epoch_verify_cache))
            self._prime_verify_batch(epoch, graph)
            first = self._global_argmax_rows(
                self.engine.local_logits(handoff["last_hidden"]))
            outputs: list[list[int]] = [
                [token] if limits[row] else [] for row, token in enumerate(first)]
            current = list(first)
            committed = list(epoch.lengths)
            rounds = [0] * batch
            proposed = [0] * batch
            accepted = [0] * batch
            done = [limits[row] == 0 or cancels[row].is_set() or
                    len(outputs[row]) >= limits[row] or
                    (bool(outputs[row]) and outputs[row][-1] in EOS_TOKEN_IDS)
                    for row in range(batch)]
            active_rows = list(range(batch))
            for row_index, token in enumerate(first):
                if limits[row_index] and on_tokens is not None:
                    on_tokens(row_index, [token])

            boarding_closed = False

            def policy_selection(*, honor_cancels: bool = False) -> list[int]:
                if self.rt.rank == 0:
                    if honor_cancels:
                        for row, event in enumerate(cancels):
                            if event.is_set():
                                done[row] = True
                    if select_active_rows is None:
                        local_rows = [row for row in active_rows if not done[row]]
                    else:
                        local_rows = [int(row) for row in select_active_rows(
                            tuple(active_rows), tuple(done))]
                        if len(set(local_rows)) != len(local_rows):
                            raise ValueError(
                                "active-row policy returned duplicate rows")
                        active_set = set(active_rows)
                        if any(row not in active_set for row in local_rows):
                            raise ValueError(
                                "active-row policy introduced an inactive row")
                        live_rows = {row for row in active_rows if not done[row]}
                        if not live_rows.issubset(local_rows):
                            raise ValueError(
                                "active-row policy dropped a live row")
                else:
                    local_rows = None
                selected_rows = self._broadcast_int_rows(
                    local_rows, capacity=max(len(outputs), 1))
                active_set = set(active_rows)
                if len(set(selected_rows)) != len(selected_rows):
                    raise ValueError("active-row policy returned duplicate rows")
                if any(row not in active_set for row in selected_rows):
                    raise ValueError("active-row policy introduced an inactive row")
                selected_set = set(selected_rows)
                for row in active_rows:
                    if row not in selected_set:
                        done[row] = True
                return selected_rows

            def compact_active(selected_rows: list[int]) -> None:
                nonlocal graph, active_rows
                if selected_rows == active_rows:
                    return
                if not selected_rows:
                    active_rows = []
                    return
                index = {row: position for position, row in enumerate(active_rows)}
                selected_indices = [index[row] for row in selected_rows]
                graph = track_verify(self._rebind_verify_rows(
                    graph, selected_indices,
                    epoch_cache=epoch_verify_cache))
                active_rows = selected_rows

            def board_one() -> bool:
                nonlocal graph, epoch, active_rows, boarding_closed
                nonlocal outputs, current, committed, rounds, proposed, accepted
                nonlocal done, limits, cancels
                if (boarding_closed or
                        (self.rt.rank == 0 and board_request is None) or
                        len(active_rows) >= 4):
                    return False
                used = set(epoch.sequence_ids)
                max_sequences = int(self.engine.engine_config.max_sequences)
                if not any(slot not in used for slot in range(max_sequences)):
                    boarding_closed = True
                    return False
                original_i = len(outputs)
                local = (board_request(original_i) if self.rt.rank == 0 else None)
                request = self._broadcast_boarding_request(local)
                if request is None:
                    return False
                if not request.input_ids:
                    raise ValueError("boarded request input_ids must not be empty")
                if request.max_new_tokens < 0:
                    raise ValueError("boarded max_new_tokens must be non-negative")
                sid = self._boarding_slot(
                    epoch, len(request.input_ids), request.max_new_tokens)
                if sid is None:
                    # Head-of-line cannot fit this epoch; finish resident rows.
                    boarding_closed = True
                    return False
                capacity = len(request.input_ids) + int(request.max_new_tokens)
                try:
                    boarded = self.prefill_batch(
                        [request.input_ids], sequence_ids=[sid],
                        max_lengths=[capacity])
                except MemoryError:
                    boarding_closed = True
                    return False
                kept = list(range(len(active_rows)))
                graph = track_verify(self._rebind_verify_rows(
                    graph, kept, boarded=boarded, required_capacity=capacity,
                    epoch_cache=epoch_verify_cache))
                epoch = epoch.merge(boarded)
                first_token = self._global_argmax_rows(
                    self.engine.local_logits(epoch.rows[-1].last_hidden))[0]
                limits.append(int(request.max_new_tokens))
                cancels.append(request.cancel_event)
                outputs.append([first_token] if limits[original_i] else [])
                current.append(first_token)
                committed.append(epoch.rows[-1].input_length)
                rounds.append(0)
                proposed.append(0)
                accepted.append(0)
                finished = (
                    limits[original_i] == 0 or cancels[original_i].is_set() or
                    len(outputs[original_i]) >= limits[original_i] or
                    (bool(outputs[original_i]) and
                     outputs[original_i][-1] in EOS_TOKEN_IDS))
                done.append(finished)
                active_rows.append(original_i)
                # Publish ownership/prefill before the first token so strategy
                # queues preserve the ref-compatible prefill -> token order.
                if on_boarded is not None:
                    on_boarded(original_i, len(active_rows))
                if limits[original_i] and on_tokens is not None:
                    on_tokens(original_i, [first_token])
                return True

            if active_rows:
                compact_active(policy_selection(honor_cancels=True))

            # Opportunistically fill the epoch before the first MTP round.
            # This is deliberately non-blocking: B=1 sees only one empty FIFO
            # probe, while already queued requests start batched immediately.
            if active_rows:
                while len(active_rows) < 4 and board_one():
                    pass
                # Cancellation racing with initial boarding is safe to apply
                # before the first replay.
                compact_active(policy_selection(honor_cancels=True))

            while active_rows:
                active_mask = [not done[row] for row in active_rows]
                if not any(active_mask):
                    break
                # epoch.rows is admission-ordered and aligns with original ids.
                sids = [epoch.rows[row].sequence_id for row in active_rows]
                curr = [current[row] for row in active_rows]
                t = time.perf_counter()
                device_draft = getattr(self.drafter, "draft_batch_device", None)
                device_paths = None
                if callable(device_draft) and hasattr(graph, "prepare_draft"):
                    device_anchors, device_paths = device_draft(
                        curr, sids, active_rows=active_mask)
                    drafted = None
                else:
                    drafted = self.drafter.draft_batch(
                        curr, sids, active_rows=active_mask)
                draft_s += time.perf_counter() - t
                positions = [
                    list(range(committed[row], committed[row] + VERIFY_WIDTH))
                    for row in active_rows]
                if device_paths is not None:
                    graph.prepare_draft(curr, device_anchors, device_paths,
                                        positions, active_rows=active_mask)
                    paths = None
                else:
                    paths = [list(item[0]) if item is not None else
                             [curr[index]] * (VERIFY_WIDTH - 1)
                             for index, item in enumerate(drafted)]
                    queries = [[curr[index]] + paths[index]
                               for index in range(len(active_rows))]
                    graph.prepare(queries, positions)
                t = time.perf_counter()
                graph.replay()
                flat_target = self._global_argmax_rows(
                    graph.local_logits.reshape(-1, graph.local_logits.shape[-1]))
                verify_s += time.perf_counter() - t
                width = VERIFY_WIDTH
                targets = [flat_target[index * width:(index + 1) * width]
                           for index in range(len(active_rows))]
                if device_paths is not None:
                    resident_paths = device_paths.cpu().tolist()
                    paths = [list(resident_paths[index]) if active_mask[index] else
                             [curr[index]] * (VERIFY_WIDTH - 1)
                             for index in range(len(active_rows))]

                commit_counts = [0] * len(active_rows)
                returned_rows = [[] for _ in range(len(active_rows))]
                feature_rows = [None] * len(active_rows)
                for index, row in enumerate(active_rows):
                    if done[row]:
                        continue
                    matched = 0
                    for draft_token, target_token in zip(
                            paths[index], targets[index][:-1]):
                        if draft_token != target_token:
                            break
                        matched += 1
                    remaining = limits[row] - len(outputs[row])
                    returned = targets[index][:min(matched + 1, remaining)]
                    eos_at = next((pos for pos, token in enumerate(returned)
                                   if token in EOS_TOKEN_IDS), None)
                    if eos_at is not None:
                        returned = returned[:eos_at + 1]
                    count = len(returned)
                    commit_counts[index] = count
                    returned_rows[index] = returned
                    if count:
                        feature_rows[index] = [
                            graph.aux_hidden[slot, index, :count]
                            for slot in range(len(graph.aux_layer_ids))]
                        rounds[row] += 1
                        proposed[row] += len(paths[index])
                        accepted[row] += min(matched, max(0, count - 1))

                if not any(commit_counts):
                    graph.rollback()
                    break
                graph.commit(commit_counts)
                self.drafter.append_context_graph_batch(feature_rows, sids)
                for index, row in enumerate(active_rows):
                    returned = returned_rows[index]
                    if not returned:
                        continue
                    outputs[row].extend(returned)
                    current[row] = returned[-1]
                    committed[row] += len(returned)
                    if on_tokens is not None:
                        on_tokens(row, list(returned))
                    if (len(outputs[row]) >= limits[row] or
                            current[row] in EOS_TOKEN_IDS):
                        done[row] = True

                steps += 1
                cancel_safe_point = steps % boarding_interval_steps == 0
                if select_active_rows is None and not cancel_safe_point:
                    # Token limits and EOS results are deterministic on every TP
                    # rank, so ordinary rounds need no rank-0 control collective.
                    # Cancellation and boarding remain synchronized at their
                    # fixed safe point below.
                    selected_rows = [row for row in active_rows if not done[row]]
                else:
                    selected_rows = policy_selection(
                        honor_cancels=cancel_safe_point)
                if selected_rows != active_rows:
                    compact_active(selected_rows)
                if not active_rows:
                    break
                # Followers must still enter board_one() so the boarding
                # request broadcast stays in lockstep with rank 0.
                if steps % boarding_interval_steps == 0:
                    while len(active_rows) < 4 and board_one():
                        pass
                    # Apply cancellations racing with boarding before replay.
                    compact_active(policy_selection(honor_cancels=True))

            self.rt.synchronize()
            elapsed = time.perf_counter() - started
            rows = []
            for index, prefill_row in enumerate(epoch.rows):
                rows.append({
                    "sequence_id": prefill_row.sequence_id,
                    "token_ids": outputs[index],
                    "cache_hit_tokens": prefill_row.restored_length,
                    "prefill_tokens": (prefill_row.input_length -
                                       prefill_row.restored_length),
                    "rounds": rounds[index],
                    "proposed_draft_tokens": proposed[index],
                    "accepted_draft_tokens": accepted[index],
                    "acceptance_rate": (accepted[index] / proposed[index]
                                        if proposed[index] else 0.0),
                    "decode_generated_tokens": max(0, len(outputs[index]) - 1),
                })
            return {
                "rows": rows,
                "batch_size": len(epoch.rows),
                "active_batch_size": len(active_rows),
                "decode_seconds": elapsed,
                "draft_seconds": draft_s,
                "verify_seconds": verify_s,
                "decode_tps": (sum(max(0, len(row) - 1) for row in outputs) /
                               elapsed if elapsed > 0 else 0.0),
                "backend": "dflash2_batched_q8_dynamic",
                "verify_width": VERIFY_WIDTH,
                "verify_capacity": graph.max_prefix if graph is not None else 0,
                "steps": steps,
            }
        finally:
            cleanup_errors: list[BaseException] = []
            epoch_sids = tuple(row.sequence_id for row in epoch.rows)
            transient_ids = {id(runner)
                             for runner in epoch_verify_cache.values()}

            # A failure in one cleanup stage must never skip later stages.
            for runner in reversed(touched_verify):
                try:
                    if runner._pending:
                        runner.rollback()
                except BaseException as exc:
                    cleanup_errors.append(exc)
            try:
                epoch.release()
            except BaseException as exc:
                cleanup_errors.append(exc)

            for runner in reversed(touched_verify):
                try:
                    if id(runner) in transient_ids:
                        runner.close()
                    else:
                        runner.reset()
                except BaseException as exc:
                    cleanup_errors.append(exc)
            epoch_verify_cache.clear()
            startup_capacity = min(
                MIN_VERIFY_CAPACITY, self.context_capacity)
            self.verify = self._verify_buckets[(1, startup_capacity)]

            for sid in epoch_sids:
                if sid in self._batch_sequence_owners:
                    cleanup_errors.append(RuntimeError(
                        f"dirty epoch: sid={sid} still has a batch owner"))
                try:
                    target_pages = self.engine.cache.page_indices(sid)
                    if target_pages:
                        cleanup_errors.append(RuntimeError(
                            f"dirty epoch: sid={sid} retains target pages "
                            f"{target_pages}"))
                except BaseException as exc:
                    cleanup_errors.append(exc)
                try:
                    draft_pages = self.drafter.kv_pool.page_indices(sid)
                    if draft_pages:
                        cleanup_errors.append(RuntimeError(
                            f"dirty epoch: sid={sid} retains draft pages "
                            f"{draft_pages}"))
                except BaseException as exc:
                    cleanup_errors.append(exc)
            if transient_ids and not cleanup_errors:
                try:
                    self._trim_dynamic_epoch_allocator()
                except BaseException as exc:
                    cleanup_errors.append(exc)
            if cleanup_errors:
                raise RuntimeError(
                    "dynamic decode epoch cleanup failed: "
                    + "; ".join(str(exc) for exc in cleanup_errors)
                ) from cleanup_errors[0]

    @torch.inference_mode()
    def generate_dflash_batch(self, input_ids: Sequence[Sequence[int]],
                              max_new_tokens, *,
                              sequence_ids: Optional[Sequence[int]] = None,
                              restored_records: Optional[
                                  Sequence[Sequence[dict]]] = None,
                              on_tokens: Optional[
                                  Callable[[int, Sequence[int]], None]] = None):
        """Prefill rows serially, then decode them through fused B1-4 verify."""
        sequences = [tuple(int(token) for token in row) for row in input_ids]
        batch = len(sequences)
        if batch not in (1, 2, 3, 4):
            raise ValueError("decode batch size must be in [1,4]")
        if any(not row for row in sequences):
            raise ValueError("each input row must not be empty")
        if isinstance(max_new_tokens, int):
            limits = [int(max_new_tokens)] * batch
        else:
            limits = [int(x) for x in max_new_tokens]
        if len(limits) != batch:
            raise ValueError("max_new_tokens must have one value per batch row")
        if any(limit < 0 for limit in limits):
            raise ValueError("max_new_tokens must be non-negative")
        capacities = [len(row) + limit
                      for row, limit in zip(sequences, limits)]
        prefill_started = time.perf_counter()
        state = self.prefill_batch(
            sequences, sequence_ids=sequence_ids, max_lengths=capacities,
            restored_records=restored_records)
        prefill_s = time.perf_counter() - prefill_started
        result = self.decode_dflash_batch(
            state, limits, on_tokens=on_tokens)
        result["prefill_seconds"] = prefill_s
        return result

    @torch.inference_mode()
    def generate_dflash(self, input_ids: Sequence[int], max_new_tokens: int,
                        sequence_id: int = 0, *,
                        restored_records: Sequence[dict] = (),
                        on_prefill_ready: Optional[Callable[[Callable[[int, int], object]], None]] = None,
                        on_tokens: Optional[Callable[[Sequence[int]], None]] = None) -> dict:
        ids = tuple(int(x) for x in input_ids)
        n_new = int(max_new_tokens)
        if not ids:
            raise ValueError("input_ids must not be empty")
        if n_new < 0:
            raise ValueError("max_new_tokens must be non-negative")
        sequence_id = int(sequence_id)
        if sequence_id in self._batch_sequence_owners:
            raise RuntimeError(
                f"sequence slot {sequence_id} is owned by a live batch state")
        total_tokens = len(ids) + n_new
        if total_tokens > self.context_capacity:
            raise ValueError(
                f"engine context requires prompt+max_new <= {self.context_capacity}, "
                f"got {len(ids)}+{n_new}")
        self._select_verify(total_tokens)
        self.verify.reset()
        self.drafter.reset(sequence_id)
        try:
            restore_t0 = time.perf_counter()
            restored, boundary_hidden = self._restore_prefix_records(
                restored_records, sequence_id)
            self.rt.synchronize()
            restored_dflash_pages = self.drafter.kv_pool.resident_pages
            restore_s = time.perf_counter() - restore_t0
            if restored > len(ids):
                raise ValueError("restored prefix exceeds input length")

            t0 = time.perf_counter()
            suffix = ids[restored:]
            if suffix:
                last_hidden = self._stream_prefill(
                    suffix, absolute_start=restored,
                    sequence_id=sequence_id)
            else:
                if boundary_hidden is None:
                    raise RuntimeError("exact cold hit is missing boundary hidden")
                last_hidden = boundary_hidden.reshape(1, -1)
            self.rt.synchronize()
            if on_prefill_ready is not None:
                on_prefill_ready(lambda start, end: self._export_prefix_records(
                    start, end, sequence_id))
            prefill_s = time.perf_counter() - t0
            if n_new == 0:
                return {
                    "token_ids": [], "cache_hit_tokens": restored,
                    "prefill_tokens": len(suffix),
                    "dflash_loaded_pages": restored_dflash_pages,
                    "dflash_resident_pages": self.drafter.kv_pool.resident_pages,
                    "cache_load_seconds": restore_s,
                    "prefill_seconds": prefill_s, "decode_seconds": 0.0,
                    "draft_seconds": 0.0, "verify_seconds": 0.0,
                    "engine_timing_source": "npu_event",
                    "engine_profiled_rounds": 0, "engine_phase_ms": [],
                    "draft_npu_seconds": 0.0, "verify_npu_seconds": 0.0,
                    "commit_npu_seconds": 0.0, "append_npu_seconds": 0.0,
                    "rounds": 0, "proposed_draft_tokens": 0,
                    "accepted_draft_tokens": 0, "acceptance_rate": 0.0,
                    "mean_accepted_per_round": 0.0,
                    "decode_generated_tokens": 0, "decode_tps": 0.0,
                    "backend": "dflash2_q8", "verify_width": VERIFY_WIDTH,
                    "verify_capacity": self.verify.max_prefix,
                }
            first = self._global_argmax_rows(
                self.engine.local_logits(last_hidden))[0]
            self._prime_verify(len(ids), sequence_id)
            self.rt.synchronize()

            out = [first]
            if on_tokens is not None:
                on_tokens([first])
            current = first
            committed_len = len(ids)
            rounds = proposed = accepted = 0
            draft_s = verify_s = 0.0
            phase_limit = max(0, int(os.environ.get(
                "LJQ_ENGINE_PHASE_TIMING_ROUNDS", "0")))
            phase_event_rows = []

            def phase_event():
                event = torch.npu.Event(enable_timing=True)
                event.record()
                return event

            decode_t0 = time.perf_counter()

            while len(out) < n_new and current not in EOS_TOKEN_IDS:
                profile_phase = len(phase_event_rows) < phase_limit
                phase_row = {}
                if profile_phase:
                    phase_row["draft_start"] = phase_event()
                t = time.perf_counter()
                path, _, _ = self.drafter.draft(
                    current, sequence_id=sequence_id)
                if profile_phase:
                    phase_row["draft_end"] = phase_event()
                self.rt.synchronize()
                draft_s += time.perf_counter() - t

                query = [current] + path
                positions_host = list(range(committed_len,
                                            committed_len + VERIFY_WIDTH))
                if profile_phase:
                    phase_row["verify_start"] = phase_event()
                self.verify.prepare([query], [positions_host])
                t = time.perf_counter()
                self.verify.replay()
                target = self._global_argmax_rows(self.verify.local_logits[0])
                if profile_phase:
                    phase_row["verify_end"] = phase_event()
                verify_s += time.perf_counter() - t

                matched = 0
                for draft_token, target_token in zip(path, target[:-1]):
                    if draft_token != target_token:
                        break
                    matched += 1
                commit_count = matched + 1
                remaining = n_new - len(out)
                returned = target[:min(commit_count, remaining)]
                eos_at = next((i for i, tok in enumerate(returned)
                               if tok in EOS_TOKEN_IDS), None)
                if eos_at is not None:
                    returned = returned[:eos_at + 1]
                commit_count = len(returned)
                if commit_count <= 0:
                    self.verify.rollback()
                    if profile_phase:
                        phase_event_rows.append(phase_row)
                    break

                if profile_phase:
                    phase_row["commit_start"] = phase_event()
                self.verify.commit(commit_count)
                if profile_phase:
                    phase_row["commit_end"] = phase_event()
                feature_rows = [
                    self.verify.aux_hidden[slot, 0, :commit_count]
                    for slot in range(len(self.verify.aux_layer_ids))]
                if profile_phase:
                    phase_row["append_start"] = phase_event()
                self.drafter.append_context_graph(
                    feature_rows, sequence_id=sequence_id)
                if profile_phase:
                    phase_row["append_end"] = phase_event()
                    phase_event_rows.append(phase_row)

                out.extend(returned)
                if on_tokens is not None:
                    on_tokens(list(returned))
                current = returned[-1]
                committed_len += commit_count
                rounds += 1
                proposed += len(path)
                accepted += min(matched, max(0, commit_count - 1))

            self.rt.synchronize()
            decode_s = time.perf_counter() - decode_t0
            phase_rows_ms = []
            for events in phase_event_rows:
                row = {}
                for phase in ("draft", "verify", "commit", "append"):
                    start = events.get(phase + "_start")
                    end = events.get(phase + "_end")
                    if start is not None and end is not None:
                        row[phase] = float(start.elapsed_time(end))
                phase_rows_ms.append(row)
            phase_totals_ms = {
                phase: sum(row.get(phase, 0.0) for row in phase_rows_ms)
                for phase in ("draft", "verify", "commit", "append")
            }
            profiled_rounds = len(phase_rows_ms)
            return {
                "token_ids": out,
                "cache_hit_tokens": restored,
                "prefill_tokens": len(suffix),
                "dflash_loaded_pages": restored_dflash_pages,
                "dflash_resident_pages": self.drafter.kv_pool.resident_pages,
                "cache_load_seconds": restore_s,
                "prefill_seconds": prefill_s,
                "decode_seconds": decode_s,
                "draft_seconds": draft_s,
                "verify_seconds": verify_s,
                "engine_timing_source": "npu_event",
                "engine_profiled_rounds": profiled_rounds,
                "engine_phase_ms": phase_rows_ms,
                "draft_npu_seconds": phase_totals_ms["draft"] / 1000.0,
                "verify_npu_seconds": phase_totals_ms["verify"] / 1000.0,
                "commit_npu_seconds": phase_totals_ms["commit"] / 1000.0,
                "append_npu_seconds": phase_totals_ms["append"] / 1000.0,
                "rounds": rounds,
                "proposed_draft_tokens": proposed,
                "accepted_draft_tokens": accepted,
                "acceptance_rate": accepted / proposed if proposed else 0.0,
                "mean_accepted_per_round": accepted / rounds if rounds else 0.0,
                "decode_generated_tokens": max(0, len(out) - 1),
                "decode_tps": ((len(out) - 1) / decode_s
                               if decode_s > 0 and len(out) > 1 else 0.0),
                "backend": "dflash2_q8",
                "verify_width": VERIFY_WIDTH,
                "verify_capacity": self.verify.max_prefix,
            }
        finally:
            if self.verify._pending:
                self.verify.rollback()
            self.verify.reset()
            self.drafter.reset(sequence_id)
            self.engine.release_sequence(sequence_id)
