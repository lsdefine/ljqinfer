# -*- coding: utf-8 -*-
"""Canonical per-slot past for DSV4F (paged, multi-sequence).

Semantic contract. For slot `s` with pos == T tokens committed, one layer's
past is fully described by:

    main_kv  : ring [n_slots, RING, 512]   row = t % RING, only the last RING
                                          tokens are kept (window attention
                                          never looks further back)
    main_ckv : paged, [T // r, 512]        closed compressed windows (r in {4,128})
    idx_ckv  : paged, [T // 4, 128]        closed indexer-compressed hist (r=4)
    res_x    : [n_slots, 2r, 4096]         raw x of the prev closed window and the
                                          open window (row = t % 2r), r in {4,128}.
                                          The compressor is a pure function of it.
    pos[s]   : host int + pos_dev[s] int32 mirror (device authority for kernels)

Paging: ONE page table shared by every compressed pool of every layer.
    PAGE_TOKENS tokens per physical page (128-aligned), page rows =
    PAGE_TOKENS // r. Slot c of a compressed pool lives at
        pg  = table[s, (c * r) // PAGE_TOKENS]
        row = pg * (PAGE_TOKENS // r) + c % (PAGE_TOKENS // r)
    Pages are allocated from a global free-list, so variable-length sequences
    share the budget. 128-token blocks are the semantic granularity: chunk
    boundaries, cold-KV boundaries and `start_pos % 128 == 0`.

Cold KV: export/import at T0 % 128 == 0: ring tail (last 128 tokens) + closed
compressed rows + res_x (last 2r raw x). Import = memcpy. No derived state.
"""
from dataclasses import dataclass, field
from typing import Tuple, Callable, Dict, List, Optional

import torch

RATIO_INDEXED = 4
RATIO_COMPRESSED = 128
BLOCK_TOKENS = 128
# main_kv ring per slot: RING*512*2B*43 layers ~= 720MB/slot, replicated per rank.
# Must cover the largest chunk (<=12k) so a whole chunk can still be cold-exported.
RING_TOKENS = 16384
PAGE_TOKENS = 2048


class PageTable:
    """[n_slots, max_pages] int32 on device + host mirror; global free-list."""

    def __init__(self, n_slots: int, max_seq: int, n_pages: int,
                 page_tokens: int = PAGE_TOKENS, device=None):
        assert page_tokens % BLOCK_TOKENS == 0
        self.page_tokens = page_tokens
        self.n_slots = n_slots
        self.max_pages = (max_seq + page_tokens - 1) // page_tokens
        self.n_pages = n_pages
        self.table = torch.full((n_slots, self.max_pages), -1, dtype=torch.int32, device=device)
        self.host: List[List[int]] = [[] for _ in range(n_slots)]
        self.free: List[int] = list(range(n_pages - 1, -1, -1))

    def to_(self, device):
        if self.table.device != torch.device(device):
            self.table = self.table.to(device)
        return self

    def n_alloc(self, slot: int) -> int:
        return len(self.host[slot])

    def ensure(self, slot: int, pos_after: int) -> None:
        """Make pages for tokens [0, pos_after) resident for `slot`."""
        need = (pos_after + self.page_tokens - 1) // self.page_tokens
        assert need <= self.max_pages, "seq exceeds max_seq"
        cur = self.host[slot]
        new = []
        while len(cur) < need:
            assert self.free, "page pool exhausted"
            pg = self.free.pop()
            cur.append(pg)
            new.append(pg)
        if new:
            k = len(cur)
            self.table[slot, k - len(new):k] = torch.tensor(new, dtype=torch.int32, device=self.table.device)

    def release(self, slot: int) -> None:
        self.free.extend(reversed(self.host[slot]))
        self.host[slot] = []
        self.table[slot].fill_(-1)

    def rows(self, slot: int, ratio: int, c0: int, c1: int) -> torch.Tensor:
        """Physical row indices (device, int64) of compressed slots [c0, c1)."""
        rpp = self.page_tokens // ratio
        c = torch.arange(c0, c1, device=self.table.device)
        pg = self.table[slot, (c // rpp)].long()
        return pg * rpp + c % rpp


class PagedPool:
    """[n_pages, page_tokens // ratio, dim] physical storage."""

    def __init__(self, pt: PageTable, ratio: int, dim: int, device=None, dtype=None):
        self.pt = pt
        self.ratio = ratio
        self.rpp = pt.page_tokens // ratio
        self.data = torch.zeros(pt.n_pages, self.rpp, dim, device=device, dtype=dtype)

    def to_(self, device):
        if self.data.device != torch.device(device):
            self.data = self.data.to(device)
        return self

    @property
    def flat(self) -> torch.Tensor:  # [n_pages * rpp, dim]
        return self.data.view(-1, self.data.size(-1))

    @property
    def max_rows(self) -> int:  # static per-slot row capacity (graph-safe: max_seq // ratio)
        return self.pt.max_pages * self.rpp

    def read(self, slot: int, c0: int, c1: int) -> torch.Tensor:
        """Materialized [c1 - c0, dim] (torch reference path only)."""
        if c1 <= c0:
            return self.flat[:0]
        return self.flat.index_select(0, self.pt.rows(slot, self.ratio, c0, c1))

    def write(self, slot: int, c0: int, val: torch.Tensor) -> None:
        """val: [m, dim] closed slots [c0, c0 + m)."""
        m = val.size(0)
        if m:
            self.flat.index_copy_(0, self.pt.rows(slot, self.ratio, c0, c0 + m), val.to(self.data.dtype))


class LayerPast:
    """Base layer past (r=0): ring of the last RING tokens of latent KV."""
    ratio = 0

    def __init__(self, pt: PageTable, kv_dim: int = 512, ring: int = RING_TOKENS,
                 device=None, dtype=None):
        self.pt = pt
        self.n_slots = pt.n_slots
        self.ring = ring
        self.main_kv = torch.zeros(self.n_slots, ring, kv_dim, device=device, dtype=dtype)

    def to_(self, device) -> "LayerPast":
        if self.main_kv.device != torch.device(device):
            self.main_kv = self.main_kv.to(device)
        self.pt.to_(device)
        return self

    @property
    def device(self):
        return self.main_kv.device

    # -- main kv ---------------------------------------------------------
    def kv_rows(self, t0: int, t1: int) -> torch.Tensor:
        """Ring row indices for absolute tokens [t0, t1) (t1 - t0 <= ring)."""
        assert t1 - t0 <= self.ring
        return torch.arange(t0, t1, device=self.device) % self.ring

    def kv(self, slot: int, t0: int, t1: int) -> torch.Tensor:
        """Materialized [t1 - t0, kv_dim] of absolute tokens [t0, t1)."""
        if t1 <= t0:
            return self.main_kv[slot, :0]
        return self.main_kv[slot].index_select(0, self.kv_rows(t0, t1))

    def write_kv(self, slot: int, pos: int, kv: torch.Tensor) -> None:
        """kv: [n, kv_dim] for absolute positions [pos, pos + n)."""
        n = kv.size(0)
        if n:
            self.main_kv[slot].index_copy_(0, self.kv_rows(pos, pos + n), kv.to(self.main_kv.dtype))

    # -- cold kv ---------------------------------------------------------
    def export_cold(self, slot: int, t0: int, t1: int) -> Dict[str, torch.Tensor]:
        """Segment [t0, t1) (128-aligned, t1 - t0 <= ring), typically one chunk."""
        assert t0 % BLOCK_TOKENS == 0 and t1 % BLOCK_TOKENS == 0, "cold boundary must be 128-aligned"
        assert 0 <= t0 < t1 and t1 - t0 <= self.ring, "segment must fit in ring"
        return {"main_kv": self.kv(slot, t0, t1).contiguous()}

    def import_cold(self, slot: int, t0: int, t1: int, blob: Dict[str, torch.Tensor]) -> None:
        t = blob.get("main_kv")
        if t is not None:
            assert t.size(0) == t1 - t0
            self.write_kv(slot, t0, t)

    def export_tail(self, slot: int, pos: int) -> Dict[str, torch.Tensor]:
        """Open-tail state at pos (only meaningful at the final segment end)."""
        return {}

    def import_tail(self, slot: int, pos: int, blob: Dict[str, torch.Tensor]) -> None:
        pass


class CompressedPast(LayerPast):
    """r=128 layer: + paged closed compressed windows + res_x ring (raw x of
    the last 2r tokens, row = t % 2r) so the compressor is a pure function."""
    ratio = RATIO_COMPRESSED

    def __init__(self, pt: PageTable, kv_dim: int = 512, dim: int = 4096,
                 ring: int = RING_TOKENS, device=None, dtype=None):
        super().__init__(pt, kv_dim, ring, device, dtype)
        self.ckv_pool = PagedPool(pt, self.ratio, kv_dim, device, dtype)
        self.res_x = torch.zeros(self.n_slots, 4 * self.ratio, dim, device=device, dtype=dtype)
        # derived per-token compressor projections (broken carry idea): name ->
        # [proj_fn, kv ring, score ring] fp32 [n_slots, 4r, coff*d]; filled in
        # lockstep with res_x so decode only projects its Q new rows.
        self.derived: Dict[str, list] = {}

    def add_derived(self, name: str, fn, width: int) -> None:
        z = lambda: torch.zeros(self.n_slots, 4 * self.ratio, width, device=self.res_x.device, dtype=torch.float32)
        self.derived[name] = [fn, z(), z()]

    def to_(self, device) -> "CompressedPast":
        super().to_(device)
        self.ckv_pool.to_(device)
        self.res_x = self.res_x.to(device)
        for v in self.derived.values():
            v[1], v[2] = v[1].to(device), v[2].to(device)
        return self

    def ensure(self, slot: int, pos_after: int) -> None:
        self.pt.ensure(slot, pos_after)

    # -- raw-x residue (ring of 2r) ----------------------------------------
    def res_len(self, pos: int) -> int:
        """Rows needed to resume at pos: prev closed window + open tail."""
        r = self.ratio
        return min(pos, r + pos % r)

    def write_res(self, slot: int, pos: int, x: torch.Tensor) -> None:
        """x: [n, dim] raw x of absolute tokens [pos, pos + n), n <= 2r."""
        n = x.size(0)
        assert n <= 4 * self.ratio
        tok = torch.arange(pos, pos + n, device=self.res_x.device)
        self.write_res_t(slot, tok, x)

    def derive(self, x: torch.Tensor):
        """All derived projections of x [n, dim] (any n): {name: (kv, sc)}.  A5 of
        amdahl_b_report: the batched decode calls this ONCE on the flat B*Q row
        stream and hands per-row slices to write_res_t(derived=...).  The projection
        (sgemm_skinny2_f32) is batch-invariant, so row m is bitwise identical for
        every n -> B == 1 and B > 1 write the same bits.  Covers both the 1-ring
        (compressed) and 2-ring (indexed: ckv + ickv) layers."""
        xf = x.float()
        return {k: v[0](xf) for k, v in self.derived.items()}

    def write_res_t(self, slot: int, tok: torch.Tensor, x: torch.Tensor,
                    rows: torch.Tensor = None, derived=None) -> None:
        """Graph-safe: tok [n] device absolute positions of the rows of x.
        `rows` may be passed in pre-computed (tok % 4r) to skip the remainder launch.
        `derived` = {name: (kv, sc)} pre-computed by derive() (batched decode);
        None -> projected here (per call)."""
        if rows is None:
            rows = tok % (4 * self.ratio)
        if derived is None and self.derived:
            derived = self.derive(x)
        if len(self.derived) == 1 and x.dtype == self.res_x.dtype and x.is_cuda:
            # fused path: one launch instead of three index_copy_
            (name, (_, kvr, scr)), = self.derived.items()
            kv, sc = derived[name]
            from ops import scatter3_rows
            scatter3_rows(self.res_x[slot], kvr[slot], scr[slot], rows,
                          x.contiguous(), kv.contiguous(), sc.contiguous())
            return
        self.res_x[slot].index_copy_(0, rows, x.to(self.res_x.dtype))
        for name, (_, kvr, scr) in self.derived.items():
            kv, sc = derived[name]
            kvr[slot].index_copy_(0, rows, kv)
            scr[slot].index_copy_(0, rows, sc)

    def res(self, slot: int, t0: int, t1: int) -> torch.Tensor:
        """Materialized [t1 - t0, dim] raw x of absolute tokens [t0, t1)."""
        assert 0 <= t1 - t0 <= 4 * self.ratio
        rows = torch.arange(t0, t1, device=self.res_x.device) % (4 * self.ratio)
        return self.res_x[slot, rows]

    def ckv(self, slot: int, pos: int) -> torch.Tensor:
        return self.ckv_pool.read(slot, 0, pos // self.ratio)

    def write_ckv(self, slot: int, nc: int, ckv: torch.Tensor) -> None:
        self.ckv_pool.write(slot, nc, ckv)

    def export_cold(self, slot: int, t0: int, t1: int) -> Dict[str, torch.Tensor]:
        blob = super().export_cold(slot, t0, t1)
        r = self.ratio
        blob["main_ckv"] = self.ckv_pool.read(slot, t0 // r, t1 // r).contiguous()
        return blob

    def import_cold(self, slot: int, t0: int, t1: int, blob: Dict[str, torch.Tensor]) -> None:
        super().import_cold(slot, t0, t1, blob)
        t = blob.get("main_ckv")
        if t is not None:
            r = self.ratio
            assert t.size(0) == (t1 - t0) // r
            self.ensure(slot, t1)
            self.write_ckv(slot, t0 // r, t)

    def export_tail(self, slot: int, pos: int) -> Dict[str, torch.Tensor]:
        n = self.res_len(pos)
        return {"res_x": self.res(slot, pos - n, pos).contiguous()}

    def import_tail(self, slot: int, pos: int, blob: Dict[str, torch.Tensor]) -> None:
        t = blob["res_x"]
        assert t.size(0) == self.res_len(pos)
        self.write_res(slot, pos - t.size(0), t)


class IndexedPast(CompressedPast):
    """r=4 layer: + paged indexer history (same res_x ring, r=4)."""
    ratio = RATIO_INDEXED

    def __init__(self, pt: PageTable, kv_dim: int = 512, dim: int = 4096,
                 idx_dim: int = 128, ring: int = RING_TOKENS, device=None, dtype=None):
        super().__init__(pt, kv_dim, dim, ring, device, dtype)
        self.idx_pool = PagedPool(pt, self.ratio, idx_dim, device, dtype)

    def to_(self, device) -> "IndexedPast":
        super().to_(device)
        self.idx_pool.to_(device)
        return self

    def ickv(self, slot: int, pos: int) -> torch.Tensor:
        return self.idx_pool.read(slot, 0, pos // self.ratio)

    def write_ickv(self, slot: int, nc: int, ickv: torch.Tensor) -> None:
        self.idx_pool.write(slot, nc, ickv)

    def export_cold(self, slot: int, t0: int, t1: int) -> Dict[str, torch.Tensor]:
        blob = super().export_cold(slot, t0, t1)
        r = self.ratio
        blob["idx_ckv"] = self.idx_pool.read(slot, t0 // r, t1 // r).contiguous()
        return blob

    def import_cold(self, slot: int, t0: int, t1: int, blob: Dict[str, torch.Tensor]) -> None:
        super().import_cold(slot, t0, t1, blob)
        ti = blob.get("idx_ckv")
        if ti is not None:
            r = self.ratio
            assert ti.size(0) == (t1 - t0) // r
            self.write_ickv(slot, t0 // r, ti)



@dataclass
class SlotPool:
    """Owns n_slots sequences: per-slot pos (host + device mirror), the shared
    page table and every layer's past. One request <-> one slot."""
    n_slots: int
    max_seq: int
    pool_tokens: int = 0            # compressed-pool budget in tokens, all slots
    ring: int = RING_TOKENS
    page_tokens: int = PAGE_TOKENS
    device: object = None
    pos: List[int] = field(default_factory=list)
    pos_dev: torch.Tensor = None
    pt: PageTable = None
    layers: Dict[int, LayerPast] = field(default_factory=dict)
    free_slots: List[int] = field(default_factory=list)

    def __post_init__(self):
        if not self.pool_tokens:
            self.pool_tokens = self.n_slots * self.max_seq
        n_pages = (self.pool_tokens + self.page_tokens - 1) // self.page_tokens
        self.pt = PageTable(self.n_slots, self.max_seq, n_pages, self.page_tokens, self.device)
        self.pos = [0] * self.n_slots
        self.pos_dev = torch.zeros(self.n_slots, dtype=torch.int32, device=self.device)
        self.free_slots = list(range(self.n_slots - 1, -1, -1))

    def to_(self, device):
        self.device = device
        self.pt.to_(device)
        self.pos_dev = self.pos_dev.to(device)
        for p in self.layers.values():
            p.to_(device)
        return self

    def add_layer(self, layer_id: int, ratio: int, **kw) -> LayerPast:
        cls = {0: LayerPast, RATIO_INDEXED: IndexedPast,
               RATIO_COMPRESSED: CompressedPast}[ratio]
        lp = cls(self.pt, ring=self.ring, device=self.device, **kw)
        self.layers[layer_id] = lp
        return lp

    # -- slot lifecycle --------------------------------------------------
    def alloc(self) -> int:
        assert self.free_slots, "no free slot"
        s = self.free_slots.pop()
        self.pos[s] = 0
        self.pos_dev[s] = 0
        return s

    def release(self, slot: int) -> None:
        self.pt.release(slot)
        self.pos[slot] = 0
        self.pos_dev[slot] = 0
        self.free_slots.append(slot)

    def ensure(self, slot: int, pos_after: int) -> None:
        assert pos_after <= self.max_seq, "past overflow"
        self.pt.ensure(slot, pos_after)

    def advance(self, slot: int, n: int) -> None:
        self.pos[slot] += n
        assert self.pos[slot] <= self.max_seq, "past overflow"
        self.pos_dev[slot] = self.pos[slot]

    # -- cold kv ---------------------------------------------------------
    def export_cold(self, slot: int, t0: int, t1: Optional[int] = None) -> Dict:
        """Export segment [t0, t1) of every layer (one chunk). If t1 == pos the
        open-tail state is attached so the sequence can be resumed exactly."""
        pos = self.pos[slot]
        t1 = pos if t1 is None else t1
        assert t1 <= pos
        seg = {"t0": t0, "t1": t1,
               "layers": {l: p.export_cold(slot, t0, t1) for l, p in self.layers.items()}}
        if t1 == pos:
            seg["tail"] = {l: p.export_tail(slot, pos) for l, p in self.layers.items()}
        return seg

    def import_cold(self, slot: int, segs: List[Dict]) -> int:
        """Import contiguous segments (ascending, first t0 == pos[slot]); the last
        one must carry `tail` (res_x). Returns new pos."""
        pos = self.pos[slot]
        for seg in segs:
            t0, t1 = seg["t0"], seg["t1"]
            assert t0 == pos, f"segment gap: {t0} != {pos}"
            self.ensure(slot, t1)
            for l, b in seg["layers"].items():
                self.layers[l].import_cold(slot, t0, t1, b)
            pos = t1
        tail = segs[-1].get("tail")
        if tail is not None:
            for l, b in tail.items():
                self.layers[l].import_tail(slot, pos, b)
        self.pos[slot] = pos
        self.pos_dev[slot] = pos
        return pos
