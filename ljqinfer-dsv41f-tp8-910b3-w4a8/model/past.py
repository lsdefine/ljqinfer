# -*- coding: utf-8 -*-
"""Canonical past-state contract for DeepSeek-V4.1-Flash.

V4.1 CSA2 separates per-layer sliding-window state from four shared history
sources.  Live execution state is therefore:

* one 128-token latent-KV ring for every backbone/DSpark attention layer;
* four paged source histories owned by layers 2, 8, 14 and 20;
* compressor carry only for the ratio-2 sources (no overlapping raw-x halo).

Cold storage contains only the paged source histories. Cold import leaves
replay_pending set until compute rebuilds the hot rings and carry.
Candidate masks and top-k indices are forward-local workspaces, not past.
Uncommitted speculative rows may remain in rings/pools: absolute ``pos`` is the
visibility authority.  A reference transaction restores overwritten ring history and compressor carry
when a speculative step commits a prefix. This module is not a GPU hot path.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Tuple

import torch

from .paging import PAGE_TOKENS, PageTable, PagedPool

WINDOW_TOKENS = 128
KV_DIM = 512
INDEX_DIM = 128
KV_SOURCE_LAYERS = (2, 8, 14, 20)
KV_SOURCE_RATIOS = {2: 2, 8: 2, 14: 2, 20: 1}
INDEX_SOURCE_LAYERS = (2, 8, 14, 20, 24, 28, 32, 36)
N_BACKBONE_LAYERS = 40
N_DSPARK_LAYERS = 3
N_ATTENTION_LAYERS = N_BACKBONE_LAYERS + N_DSPARK_LAYERS


@dataclass(frozen=True)
class LayerView:
    """Static routing metadata; tensors remain owned by Window/SourcePast."""

    layer_id: int
    mode: str                 # swa | full | reindex | reuse | dspark
    kv_source_layer: Optional[int]
    index_source_layer: Optional[int]
    ratio: int


def default_layer_views() -> Dict[int, LayerView]:
    views: Dict[int, LayerView] = {}
    for layer in range(N_ATTENTION_LAYERS):
        if layer >= N_BACKBONE_LAYERS:
            views[layer] = LayerView(layer, "dspark", None, None, 0)
            continue
        if layer < 2:
            views[layer] = LayerView(layer, "swa", None, None, 0)
            continue
        kv_source = max(x for x in KV_SOURCE_LAYERS if x <= layer)
        ratio = KV_SOURCE_RATIOS[kv_source]
        index_source = max(x for x in INDEX_SOURCE_LAYERS if x <= layer)
        if layer in KV_SOURCE_LAYERS:
            mode = "full"
        elif layer in INDEX_SOURCE_LAYERS:
            mode = "reindex"
        else:
            mode = "reuse"
        views[layer] = LayerView(layer, mode, kv_source, index_source, ratio)
    return views


DECODE_BAND = 16  # max staged source rows + verify window rows per step


class WindowPast:
    """Per-layer fixed sliding-window latent KV ring."""

    def __init__(self, n_slots: int, kv_dim: int = KV_DIM,
                 ring: int = WINDOW_TOKENS + DECODE_BAND, device=None, dtype=None):
        # Verify windows write their rows into the ring inside the graph, before
        # acceptance.  Writing t <= DECODE_BAND rows at pos..pos+t-1 overwrites
        # ring slots pos-ring..pos-ring+t-1; those must be older than the oldest
        # row any later query can still see (pos-WINDOW_TOKENS+1), hence the band.
        assert ring >= WINDOW_TOKENS + DECODE_BAND
        self.n_slots = n_slots
        self.ring = ring
        # Rows [0, DECODE_BAND) are a decode scratch band. A verify window
        # writes its staged source rows and provisional KV there, so attention
        # addresses history and window rows by id alone -- no per-layer read()
        # of the ring and no cat. Ring rows keep their own slice above it.
        self.pad = DECODE_BAND
        # ``ring`` is the storage modulus; ``window`` is the SWA span a query
        # may see.  ring - window rows of slack absorb in-graph verify writes.
        self.window = WINDOW_TOKENS
        # A row the round did not accept is steered to the head of the decode
        # scratch band by a negative position: drafter windows never publish
        # there and no query addresses it, so the row evaporates instead of
        # becoming history the next draft would attend.
        self.void = 0
        self.main_kv = torch.zeros(n_slots, self.pad + ring, kv_dim, device=device, dtype=dtype)

    @property
    def device(self):
        return self.main_kv.device

    def to_(self, device) -> "WindowPast":
        self.main_kv = self.main_kv.to(device)
        return self

    def reset_slot(self, slot: int) -> None:
        """Hand the ring back empty, the way a source hands its rows back.

        A slot leaving the bus must take nothing with it: the next request to
        borrow this slot attends its own history only, so its first draft is
        the same draft it would get on a freshly started engine.
        """
        self.main_kv[slot].zero_()

    def rows(self, t0: int, t1: int) -> torch.Tensor:
        assert 0 <= t1 - t0 <= self.ring
        return self.pad + torch.arange(t0, t1, device=self.device) % self.ring

    def gather(self, slot: int, pos_t: torch.Tensor) -> torch.Tensor:
        """Read ring rows for absolute positions held on device.

        read() slices with host ints, so a captured graph would keep serving
        the window its capture saw.  Here the row ids are derived on device
        from a live cursor; positions before the history begins fold onto the
        oldest row for the caller's validity mask to drop.
        """
        rows = self.pad + pos_t.clamp_min(0) % self.ring
        if pos_t.dim() > 1:
            # Batched read: pos_t [B,W] against slot [B] held on device, so a
            # graph serves each request's live cursor without a python loop.
            return self.main_kv[slot.reshape(-1, 1), rows]
        return self.main_kv[slot].index_select(0, rows)

    def scatter(self, slot: int, pos_t: torch.Tensor, value: torch.Tensor) -> None:
        """write() with the row positions held on device (graph-safe).

        Only for a span no wider than the ring; the chunk-tail clipping that
        write() does needs the host length, which a captured caller never has.
        """
        v = value.to(device=self.device, dtype=self.main_kv.dtype)
        rows = torch.where(pos_t < 0, torch.full_like(pos_t, self.void),
                           self.pad + pos_t % self.ring)
        if pos_t.dim() > 1:
            # Batched write: row blocks for slot [B] land in one index_put_.
            assert tuple(v.shape[:2]) == tuple(pos_t.shape) and pos_t.size(1) <= self.ring
            self.main_kv[slot.reshape(-1, 1), rows] = v
            return
        assert value.size(0) == pos_t.numel() <= self.ring
        self.main_kv[slot].index_copy_(0, rows, v)

    def read(self, slot: int, t0: int, t1: int) -> torch.Tensor:
        if t1 <= t0:
            return self.main_kv[slot, :0]
        return self.main_kv[slot].index_select(0, self.rows(t0, t1))

    def write(self, slot: int, pos: int, value: torch.Tensor) -> None:
        n = value.size(0)
        # Persist only the final window after a chunk. Attention inside that
        # chunk must consume forward-local KV before this commit, not the ring.
        if n > self.ring:
            pos += n - self.ring
            value = value[-self.ring:]
            n = self.ring
        if n:
            self.main_kv[slot].index_copy_(
                0, self.rows(pos, pos + n),
                value.to(device=self.device, dtype=self.main_kv.dtype))

    def export_tail(self, slot: int, pos: int) -> Dict[str, torch.Tensor]:
        n = min(pos, WINDOW_TOKENS)
        return {"window_kv": self.read(slot, pos - n, pos).contiguous()}

    def import_tail(self, slot: int, pos: int,
                    blob: Mapping[str, torch.Tensor]) -> None:
        value = blob.get("window_kv")
        if value is not None:
            assert value.size(0) == min(pos, self.window)
            self.write(slot, pos - value.size(0), value)


class SourcePast:
    """Paged main/index KV owned by one CSA2 Full layer.

    Main and index histories have the same compression ratio in V4.1.  For
    ratio 2, ``kv_state`` and ``score_state`` exactly match the reference
    Compressor's fp32 [slot, ratio, 512] carry.  Ratio 1 has no carry.
    """

    def __init__(self, pt: PageTable, source_layer: int, ratio: int,
                 kv_dim: int = KV_DIM, index_dim: int = INDEX_DIM,
                 device=None, ckv_dtype=None, index_dtype=None):
        assert source_layer in KV_SOURCE_LAYERS
        assert ratio in (1, 2)
        assert pt.page_tokens % ratio == 0
        self.pt = pt
        self.source_layer = source_layer
        self.ratio = ratio
        self.n_slots = pt.n_slots
        self.ckv_pool = PagedPool(pt, ratio, kv_dim, device, ckv_dtype)
        # One reserved page above the table: decode selection stages the
        # uncommitted rows there, so a reader can see them in page form
        # while canonical history stays untouched.
        self.index_pool = PagedPool(pt, ratio, index_dim, device, index_dtype,
                                    reserve=1)
        if ratio > 1:
            # Canonical committed residuals: absolute position % (2*ratio).
            # Verify reads this ring without publishing rejected rows;
            # commit installs only the accepted projected prefix.
            shape = (pt.n_slots, 2 * ratio, kv_dim)
            self.kv_state = torch.zeros(shape, device=device, dtype=torch.float32)
            self.score_state = torch.full(
                shape, -torch.inf, device=device, dtype=torch.float32)
            # Residual writes land through this staging pair, taken once
            # here: a served round is arithmetic over space the engine
            # already owns, not a request for new tensors.  It spans every
            # slot's ring, so one publish covers the whole batch aboard.
            self._wkv = torch.zeros(shape[0] * shape[1], kv_dim,
                                    device=device, dtype=torch.float32)
            self._wsc = torch.zeros(shape[0] * shape[1], kv_dim,
                                    device=device, dtype=torch.float32)
        else:
            self.kv_state = None
            self.score_state = None
        self._sbuf = None

    def to_(self, device) -> "SourcePast":
        self.pt.to_(device)
        self.ckv_pool.to_(device)
        self.index_pool.to_(device)
        if self.kv_state is not None:
            self.kv_state = self.kv_state.to(device)
            self.score_state = self.score_state.to(device)
            self._wkv = self._wkv.to(device)
            self._wsc = self._wsc.to(device)
            self._sbuf = None
        return self

    def reset_slot(self, slot: int) -> None:
        if self.kv_state is not None:
            self.kv_state[slot].zero_()
            self.score_state[slot].fill_(-torch.inf)

    def write_res(self, slot: int, kv: torch.Tensor, score: torch.Tensor,
                  rows: torch.Tensor) -> None:
        """Write projected rows into the residual ring at ``rows``.

        ``rows`` is (absolute position % 2*ratio) and may be a device tensor so
        a captured graph addresses the live position.
        """
        if self.ratio == 1:
            return
        n = kv.size(0)
        if n > self._wkv.size(0):
            self.kv_state[slot].index_copy_(0, rows, kv.float())
            self.score_state[slot].index_copy_(0, rows, score.float())
            return
        kvb, scb = self._wkv[:n], self._wsc[:n]
        kvb.copy_(kv)
        scb.copy_(score)
        self.kv_state[slot].index_copy_(0, rows, kvb)
        self.score_state[slot].index_copy_(0, rows, scb)

    def write_res_batch(self, dst: torch.Tensor, src: torch.Tensor,
                        values: torch.Tensor, scores: torch.Tensor) -> None:
        """Publish the accepted rows of every request in one pass.

        ``dst`` indexes the flattened per-slot rings and ``src`` the packed
        verify window, so the whole batch travels as two index vectors and
        the launch count stops growing with the rows aboard.  A batch whose
        source rows already sit in one run arrives with ``src`` as that run's
        first row: reading it needs no gather at all.
        """
        n = dst.numel()
        if self.ratio == 1 or not n:
            return
        buf = self._sbuf
        if buf is None or buf.dtype != values.dtype:
            buf = torch.empty_like(self._wkv, dtype=values.dtype)
            self._sbuf = buf
        for state, packed, stage in ((self.kv_state, values, self._wkv),
                                     (self.score_state, scores, self._wsc)):
            rows = (packed.narrow(0, src, n) if isinstance(src, int)
                    else torch.index_select(packed, 0, src, out=buf[:n]))
            stage[:n].copy_(rows)
            state.view(-1, state.size(-1)).index_copy_(0, dst, stage[:n])

    def fold(self, slot, start: int, values, scores, pos0=None, *, write=True,
             rows_per_request=None):
        """Fold projected rows against the canonical committed residual ring.

        Lead rows are fetched by absolute position.  Prefill passes ``write=True``
        and publishes all projected rows immediately.  Speculative verify passes
        ``write=False``: it may use the committed lead to form staged compressed
        rows, but only commit may publish the accepted projected prefix.
        """
        from ops.prefill.attention import compress_groups
        r = self.ratio
        if r == 1:
            return values, start
        ring = 2 * r
        dev = values.device
        if pos0 is not None and not write:
            # Fixed capacity, live alignment: a single graph supports both
            # cursor phases. Incomplete trailing groups are visibility-masked
            # by attention and never published by commit.
            from ops.decode.fold_fused import fold_gather
            from model.decode_attention import _requests, slot_rows
            v, sc = fold_gather(values, scores, self.kv_state,
                                self.score_state, pos0,
                                slot_rows(_requests(slot), dev), r,
                                rows_per_request=rows_per_request)
            # The live-aligned branch derives RoPE from pos0 per request, so
            # there is no single scalar origin to hand back; batched starts
            # are a tuple and must not be folded into one integer.
            return compress_groups(v, sc, r).to(values.dtype), None
        lead = start % r
        if lead:
            if pos0 is None:
                idx = torch.arange(start - lead, start, device=dev) % ring
            else:
                idx = (pos0 - lead + torch.arange(lead, device=dev)) % ring
            pv, ps = self.read_res(slot, idx)
            v = torch.cat((pv, values.float()), 0)
            sc = torch.cat((ps, scores.float()), 0)
        else:
            v, sc = values.float(), scores.float()
        n = values.shape[0]
        beg = max(0, n - ring)
        if pos0 is None:
            rows = torch.arange(start + beg, start + n, device=dev) % ring
        else:
            rows = (pos0 + beg + torch.arange(n - beg, device=dev)) % ring
        if write:
            self.write_res(slot, values[beg:], scores[beg:], rows)
        m = v.shape[0] // r * r
        if m == 0:
            return values[:0], start // r
        return compress_groups(v[:m], sc[:m], r).to(values.dtype), start // r

    def read_res(self, slot: int, rows: torch.Tensor):
        """Gather ring rows for whole groups, by absolute position."""
        return (self.kv_state[slot].index_select(0, rows),
                self.score_state[slot].index_select(0, rows))

    def ckv(self, slot: int, pos: int) -> torch.Tensor:
        return self.ckv_pool.read(slot, 0, pos // self.ratio)

    def index_k(self, slot: int, pos: int) -> torch.Tensor:
        return self.index_pool.read(slot, 0, pos // self.ratio)

    def write_ckv(self, slot: int, row: int, value: torch.Tensor) -> None:
        self.ckv_pool.write(slot, row, value)

    def write_index_k(self, slot: int, row: int, value: torch.Tensor) -> None:
        self.index_pool.write(slot, row, value)

    def commit_rows(self, slots, start_rows, counts, ck, ik):
        # Publish the accepted compressor rows of a batch into the pools.
        off = 0
        for slot, row, n in zip(slots, start_rows, counts):
            n = int(n)
            if n:
                self.ckv_pool.write(int(slot), int(row), ck[off:off + n])
                self.index_pool.write(int(slot), int(row), ik[off:off + n])
            off += n

    def export_cold(self, slot: int, t0: int,
                    t1: int) -> Dict[str, torch.Tensor]:
        r = self.ratio
        assert 0 <= t0 < t1
        return {
            "main_ckv": self.ckv_pool.read(slot, t0 // r, t1 // r).contiguous(),
            "index_k": self.index_pool.read(slot, t0 // r, t1 // r).contiguous(),
        }

    def import_cold(self, slot: int, t0: int, t1: int,
                    blob: Mapping[str, torch.Tensor]) -> None:
        r = self.ratio
        n = t1 // r - t0 // r
        main = blob.get("main_ckv")
        index = blob.get("index_k")
        if main is not None:
            assert main.size(0) == n
            self.ckv_pool.write(slot, t0 // r, main)
        if index is not None:
            assert index.size(0) == n
            self.index_pool.write(slot, t0 // r, index)

    def export_tail(self, slot: int) -> Dict[str, torch.Tensor]:
        if self.ratio == 1:
            return {}
        return {
            "kv_state": self.kv_state[slot].contiguous().clone(),
            "score_state": self.score_state[slot].contiguous().clone(),
        }

    def import_tail(self, slot: int,
                    blob: Mapping[str, torch.Tensor]) -> None:
        if self.ratio == 1:
            return
        kv = blob.get("kv_state")
        score = blob.get("score_state")
        if kv is not None:
            assert kv.shape == self.kv_state[slot].shape
            self.kv_state[slot].copy_(kv)
        if score is not None:
            assert score.shape == self.score_state[slot].shape
            self.score_state[slot].copy_(score)

    def replay_partial(self, slot: int, start_pos: int, accepted: int,
                       kv: torch.Tensor, score: torch.Tensor) -> None:
        """Install accepted projections in the canonical absolute-position ring.

        Used by bounded cold replay and the snapshot reference transaction.
        Only the newest ring-sized suffix is needed; indices must be unique.
        """
        assert accepted >= 0
        if self.ratio == 1 or accepted == 0:
            return
        assert kv.shape == score.shape and kv.size(0) >= accepted
        ring = 2 * self.ratio
        beg = max(0, accepted - ring)
        rows = (start_pos + torch.arange(beg, accepted, device=kv.device)) % ring
        self.write_res(slot, kv[beg:accepted], score[beg:accepted], rows)


@dataclass
class VerifySnapshot:
    slot: int
    pos: int
    partial: Dict[int, Tuple[torch.Tensor, torch.Tensor]]
    windows: Dict[int, torch.Tensor]


@dataclass
class SlotPool:
    """Model-wide V4.1 past: four shared sources plus thin per-layer rings."""

    n_slots: int
    max_seq: int
    pool_tokens: int = 0
    ring: int = WINDOW_TOKENS + DECODE_BAND
    window: int = WINDOW_TOKENS
    page_tokens: int = PAGE_TOKENS
    device: object = None
    pos: List[int] = field(default_factory=list)
    pos_dev: Optional[torch.Tensor] = None
    pt: Optional[PageTable] = None
    sources: Dict[int, SourcePast] = field(default_factory=dict)
    windows: Dict[int, WindowPast] = field(default_factory=dict)
    views: Dict[int, LayerView] = field(default_factory=default_layer_views)
    free_slots: List[int] = field(default_factory=list)
    replay_pending: set = field(default_factory=set)
    # Hot-only CED encoder tail; never exported to the cold cache.
    prefill_tails: dict = field(default_factory=dict)
    # slot -> tuple of (absolute_token_offset, aligned_image_features)
    image_spans: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.pool_tokens:
            # A partial page is still a whole page, so the default budget rounds
            # up per slot: summing tokens first leaves every slot one short page
            # of holding max_seq at the same time.
            self.pool_tokens = self.page_tokens * self.n_slots * (
                (self.max_seq + self.page_tokens - 1) // self.page_tokens)
        n_pages = (self.pool_tokens + self.page_tokens - 1) // self.page_tokens
        self.pt = PageTable(self.n_slots, self.max_seq, n_pages,
                            self.page_tokens, self.device)
        self.pos = [0] * self.n_slots
        self.pos_dev = torch.zeros(
            self.n_slots, dtype=torch.int32, device=self.device)
        self.free_slots = list(range(self.n_slots - 1, -1, -1))

    @property
    def row_cap(self) -> int:
        """Longest a single row can grow: its share of the paged pool."""
        return self.pt.row_cap

    def configure_default(self, kv_dim: int = KV_DIM,
                          index_dim: int = INDEX_DIM,
                          window_dtype=None, ckv_dtype=None,
                          index_dtype=None) -> "SlotPool":
        assert not self.sources and not self.windows
        for layer, ratio in KV_SOURCE_RATIOS.items():
            self.sources[layer] = SourcePast(
                self.pt, layer, ratio, kv_dim, index_dim, self.device,
                ckv_dtype, index_dtype)
        for layer in range(N_ATTENTION_LAYERS):
            self.windows[layer] = WindowPast(
                self.n_slots, kv_dim, self.ring, self.device, window_dtype)
        return self

    def to_(self, device) -> "SlotPool":
        self.device = device
        self.pt.to_(device)
        self.pos_dev = self.pos_dev.to(device)
        for tail in self.prefill_tails.values():
            for key, value in tail.items():
                if isinstance(value, torch.Tensor):
                    tail[key] = value.to(device)
        for source in self.sources.values():
            source.to_(device)
        for window in self.windows.values():
            window.to_(device)
        return self

    def alloc(self) -> int:
        assert self.free_slots, "no free sequence slots"
        slot = self.free_slots.pop()
        assert self.pos[slot] == 0
        return slot

    def release(self, slot: int) -> None:
        self.prefill_tails.pop(slot, None)
        self.image_spans.pop(slot, None)
        self.replay_pending.discard(slot)
        self.pt.release(slot)
        self.pos[slot] = 0
        self.pos_dev[slot] = 0
        for source in self.sources.values():
            source.reset_slot(slot)
        for window in self.windows.values():
            window.reset_slot(slot)
        if slot not in self.free_slots:
            self.free_slots.append(slot)

    def ensure(self, slot: int, pos_after: int) -> None:
        self.pt.ensure(slot, pos_after)

    def set_pos(self, slot: int, pos: int) -> None:
        assert 0 <= pos <= self.max_seq
        self.pos[slot] = pos
        if self.pos_dev.device.type == 'npu':
            # Queue on the caller stream without a pageable H2D barrier.
            self.pos_dev[slot:slot + 1].fill_(pos)
            return
        # ``pos_dev[slot] = int`` builds a CPU tensor and does a blocking H2D
        # every commit; stage the value in a pinned ring instead.
        ring = self.__dict__.get('_pos_ring')
        if ring is None or ring[0].dtype != self.pos_dev.dtype:
            host = torch.zeros(16, dtype=self.pos_dev.dtype,
                               pin_memory=self.pos_dev.is_cuda)
            ring = self.__dict__['_pos_ring'] = [host, 0]
        host, i = ring[0], ring[1]
        ring[1] = (i + 1) % host.numel()
        host[i] = pos
        self.pos_dev[slot:slot + 1].copy_(host[i:i + 1], non_blocking=True)

    def advance(self, slot: int, n: int) -> None:
        self.set_pos(slot, self.pos[slot] + n)

    def source_for(self, layer: int) -> Optional[SourcePast]:
        source = self.views[layer].kv_source_layer
        return None if source is None else self.sources[source]

    def snapshot_verify(self, slot: int) -> VerifySnapshot:
        if slot in self.replay_pending:
            raise RuntimeError("cold Past requires bounded replay before verify")
        partial = {}
        for layer, source in self.sources.items():
            if source.ratio > 1:
                partial[layer] = (
                    source.kv_state[slot].clone(),
                    source.score_state[slot].clone(),
                )
        return VerifySnapshot(slot, self.pos[slot], partial, {
            layer: window.main_kv[slot].clone()
            for layer, window in self.windows.items()
        })

    def commit_verify_prefix(
        self, snapshot: VerifySnapshot, accepted: int,
        projected: Mapping[int, Tuple[torch.Tensor, torch.Tensor]],
    ) -> int:
        """Commit accepted rows after a fixed Q verify and repair ratio-2 carry.

        Rejected paged rows remain hidden by position. Ring overwrites MUST be
        restored, since rejected rows may clobber visible old history. This
        reference uses snapshots; production should stage pending rows.
        ``projected`` contains
        the verify workspace's pre-RoPE compressor (wkv, wgate) rows.
        """
        assert snapshot.slot not in self.free_slots
        assert snapshot.pos == self.pos[snapshot.slot]
        assert accepted >= 0
        slot = snapshot.slot
        assert snapshot.pos + accepted <= self.max_seq
        for window in self.windows.values():
            assert accepted <= window.ring
        for layer in snapshot.partial:
            if accepted:
                kv, score = projected[layer]
                assert kv.shape == score.shape and kv.size(0) >= accepted
        for layer, old_ring in snapshot.windows.items():
            window = self.windows[layer]
            kept = window.read(slot, snapshot.pos, snapshot.pos + accepted).clone()
            window.main_kv[slot].copy_(old_ring)
            window.write(slot, snapshot.pos, kept)
        for layer, (old_kv, old_score) in snapshot.partial.items():
            source = self.sources[layer]
            source.kv_state[slot].copy_(old_kv)
            source.score_state[slot].copy_(old_score)
            if accepted:
                kv, score = projected[layer]
                assert kv.size(0) >= accepted
                source.replay_partial(slot, snapshot.pos, accepted, kv, score)
        self.set_pos(slot, snapshot.pos + accepted)
        return self.pos[slot]

    def export_cold(self, slot: int, t0: int,
                    t1: Optional[int] = None) -> Dict:
        if t1 is None:
            t1 = self.pos[slot]
        assert t1 <= self.pos[slot]
        segment = {
            "version": "41-global-v1",
            "t0": t0,
            "t1": t1,
            "sources": {
                layer: source.export_cold(slot, t0, t1)
                for layer, source in self.sources.items()
            },
        }
        return segment

    def import_cold(self, slot: int, segments: List[Dict]) -> int:
        assert segments
        expected = self.pos[slot]
        for segment in segments:
            assert segment.get("version") == "41-global-v1"
            t0, t1 = segment["t0"], segment["t1"]
            assert t0 == expected and t1 > t0
            self.ensure(slot, t1)
            for layer, blob in segment["sources"].items():
                self.sources[int(layer)].import_cold(slot, t0, t1, blob)
            expected = t1
        self.mark_cold(slot, expected)
        return expected

    def mark_cold(self, slot: int, end: int) -> None:
        """Global history is valid, but windows/carry require bounded replay."""
        self.prefill_tails.pop(slot, None)
        self.image_spans.pop(slot, None)
        for source in self.sources.values():
            source.reset_slot(slot)
        for window in self.windows.values():
            window.main_kv[slot].zero_()
        self.set_pos(slot, end)
        if end:
            self.replay_pending.add(slot)
