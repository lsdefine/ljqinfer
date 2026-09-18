"""Hybrid hot-cache and 1024-token GDN checkpoint contracts."""
from __future__ import annotations
from dataclasses import dataclass, field
from math import ceil
from typing import Any
from .config import (CONFIG, KV_PAGE_SIZE, COLD_CHECKPOINT_INTERVAL,
                     DEFAULT_MAX_CACHED_TOKENS, DEFAULT_MAX_SEQUENCE_TOKENS)

DTYPE_BYTES = {"bfloat16": 2, "float32": 4, "int64": 8, "bool": 1}

@dataclass(frozen=True)
class TensorSpec:
    name: str
    shape: tuple[int, ...]
    dtype: str
    @property
    def numel(self) -> int:
        out = 1
        for x in self.shape: out *= int(x)
        return out
    @property
    def nbytes(self) -> int: return self.numel * DTYPE_BYTES[self.dtype]

@dataclass(frozen=True)
class HybridCacheSpec:
    # Global physical KV capacity shared by all sequence page tables.
    max_tokens: int
    # Logical token limit represented by each sequence page table.
    max_sequence_tokens: int
    max_sequences: int
    page_size: int
    checkpoint_interval: int
    k: TensorSpec
    v: TensorSpec
    hot_gdn_conv: TensorSpec
    hot_gdn_recurrent: TensorSpec
    cold_gdn_conv: TensorSpec
    cold_gdn_recurrent: TensorSpec
    page_table: TensorSpec
    checkpoint_table: TensorSpec
    transient_checkpoint_slots: int = 12

    @property
    def num_pages(self) -> int: return ceil(self.max_tokens / self.page_size)
    # Initial scaffold compatibility; these now explicitly mean hot state.
    @property
    def gdn_conv(self) -> TensorSpec: return self.hot_gdn_conv
    @property
    def gdn_recurrent(self) -> TensorSpec: return self.hot_gdn_recurrent
    @property
    def checkpoints_per_sequence(self) -> int:
        return ceil(self.max_sequence_tokens / self.checkpoint_interval)
    @property
    def hot_bytes_per_rank(self) -> int:
        return sum(x.nbytes for x in (self.k, self.v, self.hot_gdn_conv,
                                      self.hot_gdn_recurrent, self.page_table))
    @property
    def cold_bytes_per_rank(self) -> int:
        return self.cold_gdn_conv.nbytes + self.cold_gdn_recurrent.nbytes
    @property
    def bytes_per_rank(self) -> int:
        return self.hot_bytes_per_rank + self.cold_bytes_per_rank + self.checkpoint_table.nbytes

# Compatibility alias used by initial tests.
HybridCache = HybridCacheSpec

def allocate_mock_cache(max_tokens: int = DEFAULT_MAX_CACHED_TOKENS,
                        max_sequences: int = 4,
                        page_size: int = KV_PAGE_SIZE,
                        checkpoint_interval: int = COLD_CHECKPOINT_INTERVAL,
                        prefill_chunk_size: int = 12288,
                        max_sequence_tokens: int | None = None) -> HybridCacheSpec:
    max_sequence_tokens = int(
        min(max_tokens, DEFAULT_MAX_SEQUENCE_TOKENS)
        if max_sequence_tokens is None else max_sequence_tokens)
    if min(max_tokens, max_sequence_tokens, max_sequences, page_size,
           checkpoint_interval) <= 0:
        raise ValueError("positive sizes required")
    if max_sequence_tokens > max_tokens:
        raise ValueError("sequence capacity exceeds global KV pool")
    physical_pages = ceil(max_tokens / page_size)
    logical_pages = ceil(max_sequence_tokens / page_size)
    checkpoints = ceil(max_sequence_tokens / checkpoint_interval)
    fa, la = len(CONFIG.full_attention_layers), len(CONFIG.linear_attention_layers)
    kv = (fa, physical_pages, page_size, CONFIG.local_kv_heads, CONFIG.head_dim)
    return HybridCacheSpec(
        max_tokens=max_tokens,
        max_sequence_tokens=max_sequence_tokens,
        max_sequences=max_sequences,
        page_size=page_size,
        checkpoint_interval=checkpoint_interval,
        k=TensorSpec("full_k", kv, "bfloat16"),
        v=TensorSpec("full_v", kv, "bfloat16"),
        hot_gdn_conv=TensorSpec("hot_gdn_conv", (la, max_sequences, *CONFIG.gdn_conv_state_shape), "bfloat16"),
        hot_gdn_recurrent=TensorSpec("hot_gdn_recurrent", (la, max_sequences, *CONFIG.gdn_recurrent_state_shape), CONFIG.gdn_ssm_dtype),
        cold_gdn_conv=TensorSpec("cold_gdn_conv", (max_sequences, checkpoints, la, *CONFIG.gdn_conv_state_shape), "bfloat16"),
        cold_gdn_recurrent=TensorSpec("cold_gdn_recurrent", (max_sequences, checkpoints, la, *CONFIG.gdn_recurrent_state_shape), CONFIG.gdn_ssm_dtype),
        page_table=TensorSpec("page_table", (max_sequences, logical_pages), "int64"),
        checkpoint_table=TensorSpec("checkpoint_table", (max_sequences, checkpoints), "int64"),
        transient_checkpoint_slots=max(1, ceil(prefill_chunk_size / checkpoint_interval)),
    )

@dataclass
class RuntimeCache:
    spec: HybridCacheSpec
    k: Any
    v: Any
    hot_gdn_conv: Any
    hot_gdn_recurrent: Any
    page_table: Any
    lengths: Any
    free_pages: list[int] = field(default_factory=list)
    # Host ownership avoids a device synchronization for every page lookup.
    host_page_table: list[list[int]] | None = None
    # FIA consumes int32 128-token subpage IDs. These are preallocated once.
    fia_page_table: Any | None = None
    fia_subpage_lut: Any | None = None
    # Sparse CPU snapshots avoid allocating the enormous theoretical cold pool.
    # key=(sequence_id, absolute_token_count), value=(conv, recurrent).
    cold_gdn: dict[tuple[int, int], tuple[Any, Any]] = field(default_factory=dict)
    gdn_checkpoint_conv_pool: Any | None = None
    gdn_checkpoint_recurrent_pool: Any | None = None
    gdn_checkpoint_conv_host: Any | None = None
    gdn_checkpoint_recurrent_host: Any | None = None
    gdn_checkpoint_meta: list[list[tuple[int, int]]] = field(
        default_factory=lambda: [[], []])
    gdn_checkpoint_pending: list[bool] = field(default_factory=lambda: [False, False])
    gdn_checkpoint_done: list[Any | None] = field(default_factory=lambda: [None, None])
    gdn_copy_stream: Any | None = None
    gdn_active_bank: int = 0
    cold_export_k_stage: Any | None = None
    cold_export_v_stage: Any | None = None
    cold_export_hidden_stage: Any | None = None
    # One BF16 final-hidden row per logical cold boundary.  Keep these on
    # pinned host memory so long-prefill metadata never scales HBM usage.
    cold_boundary_hidden_host: Any | None = None

    @classmethod
    def allocate(cls, spec: HybridCacheSpec, device: str):
        import torch
        dtypes = {"bfloat16": torch.bfloat16, "float32": torch.float32, "int64": torch.int64}
        def z(t: TensorSpec): return torch.zeros(t.shape, dtype=dtypes[t.dtype], device=device)
        table = z(spec.page_table)
        table.fill_(-1)
        slots = spec.transient_checkpoint_slots
        linear_layers = len(CONFIG.linear_attention_layers)
        conv_shape = (2, slots, linear_layers, *CONFIG.gdn_conv_state_shape)
        rec_shape = (2, slots, linear_layers, *CONFIG.gdn_recurrent_state_shape)
        conv_pool = torch.empty(conv_shape, dtype=torch.bfloat16, device=device)
        rec_pool = torch.empty(rec_shape, dtype=torch.bfloat16, device=device)
        is_npu = torch.device(device).type == "npu"
        conv_host = torch.empty(conv_shape, dtype=torch.bfloat16, device="cpu",
                                pin_memory=is_npu)
        rec_host = torch.empty(rec_shape, dtype=torch.bfloat16, device="cpu",
                               pin_memory=is_npu)
        cache = cls(spec, z(spec.k), z(spec.v), z(spec.hot_gdn_conv),
                    z(spec.hot_gdn_recurrent), table,
                    torch.zeros(spec.max_sequences, dtype=torch.int64, device=device),
                    list(range(spec.num_pages - 1, -1, -1)))
        if spec.page_size % 128:
            raise ValueError("FIA paged attention requires page_size divisible by 128")
        subpages = spec.page_size // 128
        logical_pages = spec.page_table.shape[1]
        cache.host_page_table = [
            [-1] * logical_pages for _ in range(spec.max_sequences)]
        cache.fia_page_table = torch.full(
            (spec.max_sequences, logical_pages * subpages), -1,
            dtype=torch.int32, device=device)
        cache.fia_subpage_lut = torch.arange(
            spec.num_pages * subpages, dtype=torch.int32, device=device
        ).reshape(spec.num_pages, subpages)
        cache.gdn_checkpoint_conv_pool = conv_pool
        cache.gdn_checkpoint_recurrent_pool = rec_pool
        cache.gdn_checkpoint_conv_host = conv_host
        cache.gdn_checkpoint_recurrent_host = rec_host
        export_shape = (slots, spec.k.shape[0], spec.checkpoint_interval,
                        *spec.k.shape[3:])
        cache.cold_export_k_stage = torch.empty(
            export_shape, dtype=dtypes[spec.k.dtype], device=device)
        cache.cold_export_v_stage = torch.empty(
            export_shape, dtype=dtypes[spec.v.dtype], device=device)
        cache.cold_export_hidden_stage = torch.empty(
            (slots, CONFIG.hidden_size), dtype=torch.bfloat16, device=device)
        # Boundary residuals are request state just like KV/GDN.  Keep one
        # independent host row per sequence so serial batched prefill cannot
        # overwrite an earlier row before its cold records are exported.
        cache.cold_boundary_hidden_host = torch.empty(
            (spec.max_sequences, spec.checkpoints_per_sequence,
             CONFIG.hidden_size),
            dtype=torch.bfloat16, device="cpu", pin_memory=is_npu)
        if is_npu:
            cache.gdn_copy_stream = torch.npu.Stream(device=device)
            cache.gdn_checkpoint_done = [torch.npu.Event(), torch.npu.Event()]
        return cache

    def _finalize_gdn_bank(self, bank: int) -> None:
        if not self.gdn_checkpoint_pending[bank]:
            return
        event = self.gdn_checkpoint_done[bank]
        if event is not None:
            event.synchronize()
        count = len(self.gdn_checkpoint_meta[bank])
        # Own the transient bank with two contiguous host clones, then publish
        # immutable views. Per-slot clone() was ~10x slower on Kunpeng DRAM.
        conv_batch = self.gdn_checkpoint_conv_host[bank, :count].clone()
        recurrent_batch = self.gdn_checkpoint_recurrent_host[bank, :count].clone()
        for slot, key in enumerate(self.gdn_checkpoint_meta[bank]):
            self.cold_gdn[key] = (conv_batch[slot], recurrent_batch[slot])
        self.gdn_checkpoint_meta[bank] = []
        self.gdn_checkpoint_pending[bank] = False

    def begin_gdn_checkpoints(self, sequence_id: int, start: int,
                              token_count: int) -> tuple[int, ...]:
        sid, start, token_count = self._check_sequence(sequence_id), int(start), int(token_count)
        end = start + token_count
        interval = self.spec.checkpoint_interval
        first = ((start // interval) + 1) * interval
        absolute = tuple(range(first, end + 1, interval))
        if len(absolute) > self.spec.transient_checkpoint_slots:
            raise ValueError("prefill chunk exceeds transient GDN checkpoint pool")
        bank = self.gdn_active_bank
        self._finalize_gdn_bank(bank)
        self.gdn_checkpoint_meta[bank] = [(sid, point) for point in absolute]
        return tuple(point - start for point in absolute)

    def flush_gdn_checkpoints(self) -> None:
        bank = self.gdn_active_bank
        count = len(self.gdn_checkpoint_meta[bank])
        if not count:
            return
        if self.gdn_copy_stream is None:
            self.gdn_checkpoint_conv_host[bank, :count].copy_(
                self.gdn_checkpoint_conv_pool[bank, :count])
            self.gdn_checkpoint_recurrent_host[bank, :count].copy_(
                self.gdn_checkpoint_recurrent_pool[bank, :count])
            self.gdn_checkpoint_pending[bank] = True
            self._finalize_gdn_bank(bank)
        else:
            import torch
            ready = torch.npu.Event()
            ready.record(torch.npu.current_stream(self.hot_gdn_conv.device))
            self.gdn_copy_stream.wait_event(ready)
            with torch.npu.stream(self.gdn_copy_stream):
                self.gdn_checkpoint_conv_host[bank, :count].copy_(
                    self.gdn_checkpoint_conv_pool[bank, :count], non_blocking=True)
                self.gdn_checkpoint_recurrent_host[bank, :count].copy_(
                    self.gdn_checkpoint_recurrent_pool[bank, :count], non_blocking=True)
                self.gdn_checkpoint_done[bank].record(self.gdn_copy_stream)
            self.gdn_checkpoint_pending[bank] = True
        self.gdn_active_bank = 1 - bank

    def wait_gdn_checkpoints(self) -> None:
        self._finalize_gdn_bank(0)
        self._finalize_gdn_bank(1)

    def _check_sequence(self, sequence_id: int) -> int:
        sid = int(sequence_id)
        if not 0 <= sid < self.spec.max_sequences:
            raise IndexError(f"sequence_id {sid} outside [0,{self.spec.max_sequences})")
        return sid

    def _physical_page(self, sequence_id: int, logical_page: int, allocate: bool) -> int:
        sid = self._check_sequence(sequence_id)
        if not 0 <= logical_page < self.page_table.shape[1]:
            raise IndexError(f"logical page {logical_page} exceeds sequence capacity")
        physical = (self.host_page_table[sid][logical_page]
                    if self.host_page_table is not None else
                    int(self.page_table[sid, logical_page].item()))
        if physical < 0 and allocate:
            if not self.free_pages:
                raise MemoryError("KV page pool exhausted")
            physical = self.free_pages.pop()
            if self.host_page_table is not None:
                self.host_page_table[sid][logical_page] = physical
            self.page_table[sid, logical_page] = physical
            subpages = self.spec.page_size // 128
            lo = logical_page * subpages
            self.fia_page_table[sid, lo:lo + subpages].copy_(
                self.fia_subpage_lut[physical])
        return physical

    def reserve_sequence_pages(self, sequence_id: int, capacity: int) -> tuple[int, ...]:
        """Lease every logical page needed through ``capacity`` tokens.

        Prefill may commit fewer tokens than the requested decode capacity.  A
        successful batch state nevertheless guarantees that future decode can
        append through its declared capacity without competing for free pages.
        """
        sid, capacity = self._check_sequence(sequence_id), int(capacity)
        limit = int(self.page_table.shape[1]) * int(self.spec.page_size)
        if capacity < 0 or capacity > limit:
            raise ValueError(f"capacity {capacity} outside [0,{limit}]")
        count = (capacity + self.spec.page_size - 1) // self.spec.page_size
        return tuple(self._physical_page(sid, logical, True)
                     for logical in range(count))

    def page_indices(self, sequence_id: int) -> tuple[int, ...]:
        """Return the physical KV pages currently owned by one sequence."""
        sid = self._check_sequence(sequence_id)
        if self.host_page_table is not None:
            return tuple(int(page) for page in self.host_page_table[sid]
                         if page >= 0)
        pages = self.page_table[sid]
        return tuple(int(page) for page in
                     pages[pages >= 0].detach().cpu().tolist())

    def write_kv(self, layer_slot: int, sequence_id: int, start: int, key, value,
                 *, non_blocking: bool = False) -> None:
        """Write [T,H,D] local full-attention KV into the paged HBM pool."""
        if key.shape != value.shape or key.ndim != 3:
            raise ValueError("key/value must have equal [T,H,D] shape")
        start, total = int(start), int(key.shape[0])
        if start < 0 or start + total > self.spec.max_sequence_tokens:
            raise ValueError("KV write exceeds per-sequence token capacity")
        done = 0
        while done < total:
            pos = start + done
            logical, offset = divmod(pos, self.spec.page_size)
            physical = self._physical_page(sequence_id, logical, True)
            take = min(total - done, self.spec.page_size - offset)
            self.k[layer_slot, physical, offset:offset + take].copy_(
                key[done:done + take], non_blocking=non_blocking)
            self.v[layer_slot, physical, offset:offset + take].copy_(
                value[done:done + take], non_blocking=non_blocking)
            done += take

    def read_kv_range(self, layer_slot: int, sequence_id: int,
                      start: int, end: int):
        """Materialize one logical KV range as contiguous [T,H,D] tensors."""
        import torch
        start, end = int(start), int(end)
        if start < 0 or end < start or end > self.spec.max_sequence_tokens:
            raise ValueError("invalid per-sequence KV read range")
        if end == start:
            shape = (0, CONFIG.local_kv_heads, CONFIG.head_dim)
            return (torch.empty(shape, dtype=self.k.dtype, device=self.k.device),
                    torch.empty(shape, dtype=self.v.dtype, device=self.v.device))
        keys, values, position = [], [], start
        while position < end:
            logical, offset = divmod(position, self.spec.page_size)
            physical = self._physical_page(sequence_id, logical, False)
            if physical < 0:
                raise RuntimeError(f"missing KV page sid={sequence_id} logical={logical}")
            take = min(end - position, self.spec.page_size - offset)
            keys.append(self.k[layer_slot, physical, offset:offset + take])
            values.append(self.v[layer_slot, physical, offset:offset + take])
            position += take
        return torch.cat(keys, dim=0), torch.cat(values, dim=0)

    def read_kv(self, layer_slot: int, sequence_id: int, length: int):
        """Materialize a sequence prefix as contiguous [T,H,D] tensors."""
        return self.read_kv_range(layer_slot, sequence_id, 0, int(length))

    def paged_attention_view(self, layer_slot: int, sequence_id: int,
                             token_count: int):
        """Return zero-copy FIA subpage views and a preallocated block table."""
        blocks = ceil(int(token_count) / 128)
        key = self.k[layer_slot].permute(0, 2, 1, 3).reshape(
            -1, CONFIG.local_kv_heads, 128, CONFIG.head_dim)
        value = self.v[layer_slot].permute(0, 2, 1, 3).reshape_as(key)
        table = self.fia_page_table[sequence_id:sequence_id + 1, :blocks]
        return key, value, table

    def layer_context(self, layer_idx: int, sequence_ids, positions,
                      update_kv: bool = True, host_sids: tuple[int, ...] | None = None,
                      gdn_absolute_start: int = 0,
                      gdn_checkpoint_offsets: tuple[int, ...] = ()):
        """Build the block-layer view without exposing pool layout details."""
        from .blocks import LayerContext
        full = layer_idx in CONFIG.full_attention_layers
        if full:
            slot = CONFIG.full_attention_layers.index(layer_idx)
            unique_sids = (host_sids if host_sids is not None else
                           tuple(dict.fromkeys(int(x) for x in sequence_ids.detach().cpu().tolist())))
            old_lengths = {sid: int(self.lengths[sid].item()) for sid in unique_sids}
            return LayerContext(sequence_ids=sequence_ids, positions=positions,
                                host_sids=host_sids, kv_cache=self,
                                kv_layer_slot=slot, kv_old_lengths=old_lengths,
                                update_kv=update_kv)
        slot = CONFIG.linear_attention_layers.index(layer_idx)
        bank = self.gdn_active_bank
        return LayerContext(sequence_ids=sequence_ids, positions=positions,
                            host_sids=host_sids,
                            gdn_conv=self.hot_gdn_conv[slot],
                            gdn_recurrent=self.hot_gdn_recurrent[slot],
                            update_kv=update_kv,
                            gdn_absolute_start=gdn_absolute_start,
                            gdn_checkpoint_offsets=gdn_checkpoint_offsets,
                            gdn_checkpoint_conv=self.gdn_checkpoint_conv_pool[bank, :, slot],
                            gdn_checkpoint_recurrent=self.gdn_checkpoint_recurrent_pool[bank, :, slot])

    def commit_full_attention(self, layer_idx: int, context, old_lengths: dict[int, int]) -> None:
        """Paged full-attention writes each layer directly into the KV pool."""
        return None

    def export_cold_blocks(self, sequence_id: int, start: int, end: int):
        """Export up to one prefill chunk with two batched D2H transfers.

        Cold blocks may be smaller than physical KV pages.  Each aligned block
        is copied into a reusable NPU staging pool, then the populated K and V
        regions cross to CPU once each.  CPU records own views of those fresh
        batch allocations, while the NPU staging pool is immediately reusable.
        """
        sid, start, end = self._check_sequence(sequence_id), int(start), int(end)
        interval, page_size = (int(self.spec.checkpoint_interval),
                               int(self.spec.page_size))
        if (start < 0 or end <= start or start % interval or end % interval or
                interval > page_size or page_size % interval):
            raise ValueError("cold batch export requires aligned page subranges")
        if end > int(self.lengths[sid].item()):
            raise ValueError("cold export exceeds committed sequence length")
        count = (end - start) // interval
        capacity = int(self.cold_export_k_stage.shape[0])
        if count > capacity:
            raise ValueError(f"cold export has {count} blocks, staging capacity is {capacity}")
        for block, block_start in enumerate(range(start, end, interval)):
            logical, offset = divmod(block_start, page_size)
            physical = self._physical_page(sid, logical, False)
            if physical < 0:
                raise RuntimeError("cold export references a missing KV page")
            self.cold_export_k_stage[block].copy_(
                self.k[:, physical, offset:offset + interval])
            self.cold_export_v_stage[block].copy_(
                self.v[:, physical, offset:offset + interval])
        keys_host = self.cold_export_k_stage[:count].detach().to("cpu")
        values_host = self.cold_export_v_stage[:count].detach().to("cpu")
        records = []
        committed = int(self.lengths[sid].item())
        for block, block_start in enumerate(range(start, end, interval)):
            block_end = block_start + interval
            state = self.cold_gdn.get((sid, block_end))
            if state is None:
                if block_end != committed:
                    raise RuntimeError(
                        f"missing GDN checkpoint sid={sid} end={block_end}")
                conv = self.hot_gdn_conv[:, sid].detach().to("cpu")
                recurrent = self.hot_gdn_recurrent[:, sid].detach().to("cpu")
            else:
                conv, recurrent = state
            keys = tuple(keys_host[block, slot] for slot in range(self.k.shape[0]))
            values = tuple(values_host[block, slot] for slot in range(self.v.shape[0]))
            records.append((block_start, block_end, keys, values, conv, recurrent))
        return tuple(records)

    def export_cold_block(self, sequence_id: int, start: int, end: int):
        """Compatibility wrapper for one immutable CPU cold block."""
        records = self.export_cold_blocks(sequence_id, start, end)
        if len(records) != 1:
            raise ValueError("single cold export requires exactly one interval")
        return records[0]

    def import_cold_blocks(self, sequence_id: int, records) -> int:
        """Queue CPU cold records directly into final paged HBM storage.

        Records already own contiguous per-layer CPU tensors.  Avoid the old
        CPU->temporary-device->pool double copy, and avoid cloning every GDN
        boundary again during each hit.  The caller synchronizes once after
        target, DFlash and boundary-hidden copies have all been queued.
        """
        sid = self._check_sequence(sequence_id)
        records = tuple(records)
        if not records:
            return 0
        expected = int(records[0][0])
        for record in records:
            start, end, keys, values, conv, recurrent = record
            start, end = int(start), int(end)
            if (start != expected or end <= start or
                    end - start != self.spec.checkpoint_interval):
                raise ValueError("invalid or non-contiguous cold block interval")
            if len(keys) != self.k.shape[0] or len(values) != self.v.shape[0]:
                raise ValueError("cold block full-attention layer count mismatch")
            for slot, (key, value) in enumerate(zip(keys, values)):
                self.write_kv(slot, sid, start, key, value, non_blocking=True)
            # Cold-cache records are immutable owners.  Retaining their CPU
            # tensors preserves every reusable boundary without cloning about
            # 78 MiB per checkpoint on every cache hit.
            self.cold_gdn[(sid, end)] = (conv, recurrent)
            expected = end
        conv, recurrent = records[-1][4], records[-1][5]
        self.hot_gdn_conv[:, sid].copy_(conv, non_blocking=True)
        self.hot_gdn_recurrent[:, sid].copy_(recurrent, non_blocking=True)
        self.lengths[sid] = expected
        return expected

    def import_cold_block(self, sequence_id: int, record, *, final: bool = False) -> int:
        """Compatibility wrapper for callers restoring one block at a time."""
        sid = self._check_sequence(sequence_id)
        start, end, keys, values, conv, recurrent = record
        start, end = int(start), int(end)
        if end <= start or end - start != self.spec.checkpoint_interval:
            raise ValueError("invalid cold block interval")
        if len(keys) != self.k.shape[0] or len(values) != self.v.shape[0]:
            raise ValueError("cold block full-attention layer count mismatch")
        for slot, (key, value) in enumerate(zip(keys, values)):
            self.write_kv(slot, sid, start, key, value, non_blocking=True)
        self.cold_gdn[(sid, end)] = (conv, recurrent)
        if final:
            self.hot_gdn_conv[:, sid].copy_(conv, non_blocking=True)
            self.hot_gdn_recurrent[:, sid].copy_(recurrent, non_blocking=True)
            self.lengths[sid] = end
        return end

    def checkpoint_gdn(self, sequence_id: int, token_count: int, force: bool = False) -> bool:
        """Store all GDN states in CPU DRAM at an aligned 1024-token boundary."""
        sid, token_count = self._check_sequence(sequence_id), int(token_count)
        if not force and token_count % self.spec.checkpoint_interval:
            return False
        conv = self.hot_gdn_conv[:, sid].detach().to("cpu").clone()
        recurrent = self.hot_gdn_recurrent[:, sid].detach().to("cpu").clone()
        self.cold_gdn[(sid, token_count)] = (conv, recurrent)
        return True

    def restore_gdn(self, sequence_id: int, token_count: int) -> int:
        """Restore the nearest checkpoint not newer than token_count."""
        sid, token_count = self._check_sequence(sequence_id), int(token_count)
        candidates = [n for s, n in self.cold_gdn if s == sid and n <= token_count]
        if not candidates:
            self.hot_gdn_conv[:, sid].zero_()
            self.hot_gdn_recurrent[:, sid].zero_()
            return 0
        restored = max(candidates)
        conv, recurrent = self.cold_gdn[(sid, restored)]
        self.hot_gdn_conv[:, sid].copy_(conv.to(self.hot_gdn_conv.device))
        self.hot_gdn_recurrent[:, sid].copy_(recurrent.to(self.hot_gdn_recurrent.device))
        return restored

    def release_sequence(self, sequence_id: int) -> None:
        sid = self._check_sequence(sequence_id)
        pages = self.page_table[sid]
        if self.host_page_table is None:
            owned = pages[pages >= 0].detach().cpu().tolist()
        else:
            owned = [page for page in self.host_page_table[sid] if page >= 0]
            self.host_page_table[sid] = [-1] * self.page_table.shape[1]
        self.free_pages.extend(int(page) for page in owned)
        pages.fill_(-1)
        if self.fia_page_table is not None:
            self.fia_page_table[sid].fill_(-1)
        self.lengths[sid] = 0
        self.hot_gdn_conv[:, sid].zero_()
        self.hot_gdn_recurrent[:, sid].zero_()
        for key in [key for key in self.cold_gdn if key[0] == sid]:
            del self.cold_gdn[key]

    # Compatibility properties.
    @property
    def gdn_conv(self): return self.hot_gdn_conv
    @property
    def gdn_recurrent(self): return self.hot_gdn_recurrent
