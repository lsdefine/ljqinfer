# -*- coding: utf-8 -*-
"""Model execution facade for strategy/service layers (B=1, TP8 torchrun).

Rank 0 owns the strategy; other ranks spin in `serve_workers()` and replay
every model-touching command via broadcast_object_list.  Cold-KV blocks are
packed byte-exact: [128, width] bf16 per block, containing every layer's
export_cold tensors plus the tail state at the block end so that any block
prefix can be resumed exactly.
"""
from __future__ import annotations

import os
import threading
import itertools
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

import torch
import datetime as _dt
import torch.distributed as dist

from model.past import BLOCK_TOKENS, RING_TOKENS

_OVERLAP_RATIO = 4   # only ratio==4 compressors read the previous closed window

EOS_DEFAULT = 1
DEFAULT_PREFILL_CHUNK_TOKENS = 12 * 1024
_STOP = "__stop__"
# Control-plane process group (gloo).  Workers idle-block on it, so it must
# not be NCCL: the NCCL watchdog aborts the rank after 600s of "stuck"
# collective.  Compute stays on the default NCCL group.
_CTRL_PG = None


def _prefill_chunk_size(pos: int, total: int, limit: int) -> int:
    """Keep the last 128..255 tokens in one final chunk.

    A cold-KV hit restores whole 128-blocks and recomputes only the tail, so a
    cold miss must use the same final forward shape to stay bit-comparable.
    """
    remaining = total - pos
    assert remaining > 0 and pos % BLOCK_TOKENS == 0
    final_chunk = BLOCK_TOKENS + total % BLOCK_TOKENS
    if remaining <= final_chunk:
        return remaining
    return min(limit, remaining - final_chunk)


@dataclass(frozen=True)
class KVFormat:
    block_tokens: int = BLOCK_TOKENS
    width: int = 0
    dtype: torch.dtype = torch.bfloat16
    layout: str = "block_flat_4pool"


@dataclass(frozen=True)
class ModelCapabilities:
    max_batch_size: int
    kv_page_size: int
    kv_pool_pages: int


@dataclass(frozen=True)
class KVLoadResult:
    free_pages: int


@dataclass
class BoardingRequest:
    """One FIFO request restored by the strategy for safe-point boarding."""
    input_ids: Sequence[int]
    max_new_tokens: int
    cancel_event: threading.Event
    temperature: float = 0.0
    payload: dict = field(default_factory=dict)


KV_FORMAT = KVFormat()


def _flat_keys(seg: Dict) -> List[tuple]:
    """Deterministic packing order: (group, layer, key)."""
    keys = []
    for l in sorted(seg["layers"]):
        for k in sorted(seg["layers"][l]):
            keys.append(("layers", l, k))
    for l in sorted(seg["tail"]):
        for k in sorted(seg["tail"][l]):
            keys.append(("tail", l, k))
    return keys


def _as_u8(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().view(-1).view(torch.uint8)


SPEC_Q = 8  # B1Q8: verify width per step_g


class ModelExecution:
    def __init__(self, model, rank: int, world: int,
                 prefill_chunk: int = DEFAULT_PREFILL_CHUNK_TOKENS):
        self.model = model
        self.pool = model.pool
        self.rank = rank
        self.world = world
        self.dev = torch.device(f"cuda:{rank}")
        self.prefill_chunk = prefill_chunk - prefill_chunk % BLOCK_TOKENS
        self._row_slot: Dict[int, int] = {}
        self._ctrl_buf = torch.zeros(18, dtype=torch.int64)  # gloo control-plane fast path (CPU)
        self._row_input: Dict[int, tuple] = {}
        self._mh: Dict[int, torch.Tensor] = {}      # slot -> main_hidden of last prefilled token [1,1,.]
        self._qin: Dict[int, torch.Tensor] = {}     # slot -> [1, Q] spec input (step_g state)
        self._pos_t: Dict[int, torch.Tensor] = {}   # slot -> [1] int64 device pos (step_g state)
        self._temp_t: Dict[int, torch.Tensor] = {}  # slot -> [1] float32 device temperature (0 = greedy)
        self._graph: Dict[int, tuple] = {}          # slot -> (CUDAGraph, g_out, n_out); static in = _qin/_pos_t
        # slot-tuple -> (CUDAGraph, qin, pos_t, temp_t, g_out, n_out).  Keyed by the exact
        # tuple, not by B: step_g_batch bakes slots[b] into the graph (MTP KV is slot-bound).
        self._bgraph: Dict[tuple, tuple] = {}
        self._spec: Optional[List[tuple]] = None  # [(group,layer,key,shape,dtype,nbytes)]
        self._kv_v2_plan = None
        self._width = self._probe_width()
        # strategy reads the module-level KV_FORMAT; fill probed width in place.
        object.__setattr__(KV_FORMAT, "width", self._width)
        self.kv_format = KV_FORMAT
        self.capabilities = ModelCapabilities(
            max_batch_size=int(self.pool.n_slots),
            kv_page_size=self.pool.pt.page_tokens,
            kv_pool_pages=self.pool.pt.n_pages)

    @classmethod
    def startup(cls, devices=None, prefill_chunk_tokens: Optional[int] = None,
                **_ignored) -> "ModelExecution":
        """strategy.startup entry. Must run under torchrun (8 ranks).
        Non-zero ranks never return: they serve broadcast commands then exit,
        so only rank 0 continues into strategy/uvicorn."""
        ex = load_execution()
        if prefill_chunk_tokens is not None:
            ex.prefill_chunk = int(prefill_chunk_tokens) - int(prefill_chunk_tokens) % BLOCK_TOKENS
        if ex.rank != 0:
            ex.serve_workers()
            os._exit(0)
        return ex

    # ------------------------------------------------------------ rpc
    def _rpc(self, cmd: str, *args):
        if self.world > 1 and self.rank == 0:
            if cmd == "step":                       # fast path: fixed-size i64 header
                b = self._ctrl_buf
                b[0] = 1
                b[1] = args[0]
                b[2] = args[1]
                dist.broadcast(b, src=0, group=_CTRL_PG)
            elif cmd == "step_batch":                # fast path: [2, B, slots..., pos...]
                b = self._ctrl_buf
                slots, positions = args[0], args[1]
                nb = len(slots)
                b[0] = 2
                b[1] = nb
                for i in range(nb):
                    b[2 + i] = int(slots[i])
                    b[2 + nb + i] = int(positions[i])
                dist.broadcast(b, src=0, group=_CTRL_PG)
            else:
                self._ctrl_buf[0] = 0
                dist.broadcast(self._ctrl_buf, src=0, group=_CTRL_PG)
                dist.broadcast_object_list([(cmd, args)], src=0, group=_CTRL_PG)
        return getattr(self, "_do_" + cmd)(*args)

    def serve_workers(self):
        """Non-zero ranks: replay commands until stop."""
        assert self.rank != 0
        b = self._ctrl_buf
        while True:
            try:
                dist.broadcast(b, src=0, group=_CTRL_PG)
            except RuntimeError as exc:  # belt and braces: never die on idle
                if "Timed out" not in str(exc):
                    raise
                print("[worker r%d] control-plane recv timed out, retrying"
                      % self.rank, flush=True)
                continue
            if int(b[0]) == 1:                       # fast path: step(slot, pos)
                try:
                    self._do_step(int(b[1]), int(b[2]))
                except Exception:
                    import traceback
                    print("[worker r%d] step failed, staying alive"
                          % self.rank, flush=True)
                    traceback.print_exc()
                continue
            if int(b[0]) == 2:                       # fast path: step_batch(slots, pos)
                try:
                    nb = int(b[1])
                    sl = tuple(int(b[2 + i]) for i in range(nb))
                    ps = tuple(int(b[2 + nb + i]) for i in range(nb))
                    self._do_step_batch(sl, ps)
                except Exception:
                    import traceback
                    print("[worker r%d] step_batch failed, staying alive"
                          % self.rank, flush=True)
                    traceback.print_exc()
                continue
            obj = [None]
            try:
                dist.broadcast_object_list(obj, src=0, group=_CTRL_PG)
            except RuntimeError as exc:
                if "Timed out" not in str(exc):
                    raise
                print("[worker r%d] control-plane obj recv timed out, retrying"
                      % self.rank, flush=True)
                continue
            cmd, args = obj[0]
            if cmd == _STOP:
                return
            try:
                getattr(self, "_do_" + cmd)(*args)
            except Exception:
                import traceback
                print("[worker r%d] command %r failed, staying alive"
                      % (self.rank, cmd), flush=True)
                traceback.print_exc()

    def shutdown(self):
        if self.world > 1 and self.rank == 0:
            self._ctrl_buf[0] = 0
            dist.broadcast(self._ctrl_buf, src=0, group=_CTRL_PG)
            dist.broadcast_object_list([(_STOP, ())], src=0, group=_CTRL_PG)

    # ------------------------------------------------------------ slots
    def _slot(self, row: int) -> int:
        if row not in self._row_slot:
            self._row_slot[row] = self._rpc("alloc")
        return self._row_slot[row]

    def _do_alloc(self) -> int:
        slot = self.pool.alloc()
        if slot not in self._graph:
            self._capture(slot)
        return slot

    def _peer_ar_rows_nmax(self) -> int:
        from ops import peer_ar_rows
        return peer_ar_rows.nmax_for_pool(self.pool, int(self.pool.n_slots) * int(SPEC_Q))

    def _capture(self, slot: int, pos0: int = 256):
        """Capture step_g for `slot` as a CUDA graph over static _qin/_pos_t. Runs on all ranks
        in lockstep (NCCL inside). Warmup/capture writes garbage into the slot's kv/res_x at
        pos0..pos0+3Q; harmless: the slot is fresh and prefill overwrites everything it reads."""
        with torch.no_grad(), torch.device(self.dev):
            qin = torch.zeros(1, SPEC_Q, dtype=torch.int64, device=self.dev)
            pos_t = torch.full((1,), pos0, dtype=torch.int64, device=self.dev)
            temp_t = torch.zeros(1, dtype=torch.float32, device=self.dev)
            self._qin[slot], self._pos_t[slot], self._temp_t[slot] = qin, pos_t, temp_t
            self.pool.ensure(slot, pos0 + 4 * SPEC_Q + 8)
            if self.world > 1:
                from ops import peer_ar, peer_ar_rows
                peer_ar.prewarm()   # register IPC buffers before capture
                peer_ar_rows.prewarm(self._peer_ar_rows_nmax())
            s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(2):
                    self.model.step_g(qin, pos_t, slot, temp_t)
            torch.cuda.current_stream().wait_stream(s)
            torch.cuda.synchronize(self.dev)
            if self.world > 1:
                dist.barrier()
            graph = torch.cuda.CUDAGraph()
            # thread_local: the cold-kv pinned allocator thread may cudaHostAlloc
            # during capture; global mode would poison the capture.
            with torch.cuda.graph(graph, capture_error_mode="thread_local"):
                g_out, n_out, _ = self.model.step_g(qin, pos_t, slot, temp_t)
            torch.cuda.synchronize(self.dev)
            if self.world > 1:
                dist.barrier()
            self._graph[slot] = (graph, g_out, n_out)

    # --------------------------------------------------- batched decode step
    def _do_capture_batch(self, slots, pos0: int = 256):
        # Capture step_g_batch for this exact slot tuple over private static buffers.
        # Collective (NCCL inside): must run on every rank, so reach it via _rpc.
        slots = tuple(int(x) for x in slots)
        B = len(slots)
        with torch.no_grad(), torch.device(self.dev):
            qin = torch.zeros(B, SPEC_Q, dtype=torch.int64, device=self.dev)
            pos_t = torch.full((B,), pos0, dtype=torch.int64, device=self.dev)
            temp_t = torch.zeros(B, dtype=torch.float32, device=self.dev)
            for sl in slots:
                self.pool.ensure(sl, pos0 + 4 * SPEC_Q + 8)
            if self.world > 1:
                from ops import peer_ar, peer_ar_rows
                peer_ar.prewarm()
                peer_ar_rows.prewarm(self._peer_ar_rows_nmax())
            st = torch.cuda.Stream(); st.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(st):
                for _ in range(2):
                    self.model.step_g_batch(qin, pos_t, list(slots), temp_t)
            torch.cuda.current_stream().wait_stream(st)
            torch.cuda.synchronize(self.dev)
            if self.world > 1:
                dist.barrier()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, capture_error_mode="thread_local"):
                g_out, n_out, _ = self.model.step_g_batch(qin, pos_t, list(slots), temp_t)
            torch.cuda.synchronize(self.dev)
            if self.world > 1:
                dist.barrier()
            self._bgraph[slots] = (graph, qin, pos_t, temp_t, g_out, n_out)
        return None

    def _do_prepare_batch(self, slots):
        # Ensure the graph for this slot tuple exists, then bind per-slot step_g state
        # (_qin/_pos_t/_temp_t, written by spec_init) into row b of the batch buffers.
        slots = tuple(int(x) for x in slots)
        if slots not in self._bgraph:
            # Should never happen after warmup_batch_graphs (all sorted combos are
            # pre-captured); capturing here runs a warmup step over live KV.
            if self.rank == 0:
                print(f"[model] WARNING: on-the-fly batch graph capture for slots={slots}", flush=True)
            self._do_capture_batch(slots)
        _, qin, pos_t, temp_t, _, _ = self._bgraph[slots]
        for b, sl in enumerate(slots):
            qin[b].copy_(self._qin[sl][0])
            pos_t[b].copy_(self._pos_t[sl][0])
            temp_t[b].copy_(self._temp_t[sl][0])
        return None

    def _do_unbind_batch(self, slots):
        # Write batch state back to the per-slot buffers, so a later B1 replay or a
        # different slot tuple (a row got off / boarded) resumes exactly where we left.
        slots = tuple(int(x) for x in slots)
        ent = self._bgraph.get(slots)
        if ent is None:
            return None
        _, qin, pos_t, temp_t, _, _ = ent
        for b, sl in enumerate(slots):
            self._qin[sl][0].copy_(qin[b])
            self._pos_t[sl][0].copy_(pos_t[b])
            self._temp_t[sl][0].copy_(temp_t[b])
        return None

    def _do_step_batch(self, slots, positions) -> list:
        # One batched verify->accept->draft step (graph replay). Returns per-row token lists.
        slots = tuple(int(x) for x in slots)
        with torch.no_grad(), torch.device(self.dev):
            for sl, p in zip(slots, positions):
                self.pool.ensure(sl, int(p) + SPEC_Q)
            graph, qin, pos_t, temp_t, g, n_new = self._bgraph[slots]
            graph.replay()
            ns = n_new.tolist()
            gl = g.tolist()
            return [gl[b][:int(ns[b])] for b in range(len(slots))]

    def _do_release(self, slot: int):
        self.pool.release(slot)

    def _release_row(self, row: int):
        slot = self._row_slot.pop(row, None)
        if slot is not None:
            self._rpc("release", slot)

    @property
    def free_pages(self) -> int:
        return len(self.pool.pt.free)

    def required_pages(self, prompt_len: int, max_new: int) -> int:
        pt = self.pool.pt.page_tokens
        return (prompt_len + max_new + pt - 1) // pt

    def reset(self):
        for row in list(self._row_slot):
            self._release_row(row)
        self._row_input.clear()

    def set_input(self, input_ids: Sequence[int], row: int = 0):
        self._release_row(row)
        self._row_input[row] = tuple(int(t) for t in input_ids)
        self._slot(row)

    # ------------------------------------------------------------ cold kv
    def _segment(self, slot: int, t0: int, t1: int) -> Dict:
        seg = {"t0": t0, "t1": t1,
               "layers": {l: p.export_cold(slot, t0, t1) for l, p in self.pool.layers.items()},
               "tail": {l: (p.export_tail(slot, t1)
                            if getattr(p, "ratio", 0) == _OVERLAP_RATIO else {})
                        for l, p in self.pool.layers.items()}}
        return seg

    def _probe_width(self) -> int:
        slot = self.pool.alloc()
        self.pool.ensure(slot, BLOCK_TOKENS)
        self.pool.pos[slot] = BLOCK_TOKENS
        self.pool.pos_dev[slot] = BLOCK_TOKENS
        seg = self._segment(slot, 0, BLOCK_TOKENS)
        spec = []
        for g, l, k in _flat_keys(seg):
            t = seg[g][l][k]
            spec.append((g, l, k, tuple(t.shape), t.dtype, t.numel() * t.element_size()))
        self.pool.release(slot)
        def _kind(item):
            g, _, k = item[:3]
            if g == "tail":
                return 2
            return 0 if k == "main_kv" else 1
        spec.sort(key=lambda item: (_kind(item), item[1], item[2]))
        self._spec = spec
        self._main_nbytes = sum(s[-1] for s in spec if _kind(s) == 0)
        self._aux_nbytes = sum(s[-1] for s in spec if _kind(s) == 1)
        self._tail_nbytes = sum(s[-1] for s in spec if _kind(s) == 2)
        nbytes = self._main_nbytes + self._aux_nbytes + self._tail_nbytes
        assert nbytes % (2 * BLOCK_TOKENS) == 0, nbytes
        return nbytes // 2 // BLOCK_TOKENS

    @property
    def kv_block_sections(self) -> tuple[int, int, int]:
        """BF16 element counts for one cold block: main, aux, tail."""
        return (self._main_nbytes // 2,
                self._aux_nbytes // 2,
                self._tail_nbytes // 2)

    @property
    def kv_v2_fields(self) -> tuple[tuple, ...]:
        """Field-major shape/dtype layout for one canonical 128-token block."""
        return tuple(((g, l, k), shape, dtype)
                     for g, l, k, shape, dtype, _ in self._spec)

    def begin_kv_v2_capture(self, plan) -> None:
        if self.rank != 0:
            return
        if self._kv_v2_plan is not None:
            raise RuntimeError("cold KV V2 capture already active")
        self._kv_v2_plan = plan
        for layer in self.model.layers:
            layer.attn._cold_kv_v2_tail = self._capture_kv_v2_tail

    def _capture_kv_v2_tail(self, layer_id: int, start: int,
                            x: torch.Tensor, ratio: int) -> None:
        plan = self._kv_v2_plan
        if plan is None:
            return
        if ratio != _OVERLAP_RATIO:
            return          # non-overlap compressor: the previous closed
                            # window is never read at a 128-aligned resume
                            # point, so no carry needs to be stored.
        blocks = x.size(0) // BLOCK_TOKENS
        if blocks <= 0:
            return
        xb = x[:blocks * BLOCK_TOKENS].view(blocks, BLOCK_TOKENS, *x.shape[1:])
        tail = xb if ratio == BLOCK_TOKENS else xb[:, -ratio:]
        plan.write(("tail", layer_id, "res_x"), start // BLOCK_TOKENS, tail)

    def _capture_kv_v2_chunk(self, slot: int, start: int, end: int) -> None:
        plan = self._kv_v2_plan
        stop = end - end % BLOCK_TOKENS
        if plan is None or stop <= start:
            return
        blocks = (stop - start) // BLOCK_TOKENS
        for layer_id, past in self.pool.layers.items():
            for key, tensor in past.export_cold(slot, start, stop).items():
                spec = next(s for s in self._spec
                            if s[0] == "layers" and s[1] == layer_id and s[2] == key)
                shape = spec[3]
                plan.write(("layers", layer_id, key), start // BLOCK_TOKENS,
                           tensor.view(blocks, *shape))

    def finish_kv_v2_capture(self, commit: bool = True):
        plan, self._kv_v2_plan = self._kv_v2_plan, None
        for layer in self.model.layers:
            if hasattr(layer.attn, "_cold_kv_v2_tail"):
                del layer.attn._cold_kv_v2_tail
        if plan is None:
            return None
        if not commit:
            return plan.abort()
        try:
            torch.cuda.synchronize(self.dev)
            return plan.commit()
        except BaseException:
            plan.abort()
            raise

    def _pack_block(self, slot: int, t0: int, dst: torch.Tensor):
        seg = self._segment(slot, t0, t0 + BLOCK_TOKENS)
        parts = [_as_u8(seg[g][l][k]) for g, l, k, *_ in self._spec]
        buf = torch.cat(parts).view(torch.bfloat16).view(BLOCK_TOKENS, self._width)
        dst.copy_(buf, non_blocking=False)

    def _unpack_block(self, flat: torch.Tensor, t0: int, want_tail: bool) -> Dict:
        u8 = flat.contiguous().view(-1).view(torch.uint8)
        seg = {"t0": t0, "t1": t0 + BLOCK_TOKENS, "layers": {}, "tail": {}}
        off = 0
        for g, l, k, shape, dtype, nb in self._spec:
            t = u8[off:off + nb].view(dtype).view(shape)
            off += nb
            if g == "layers" or want_tail:
                seg[g].setdefault(l, {})[k] = t
        if not want_tail:
            seg.pop("tail")
        return seg

    def _unpack_span(self, flat: torch.Tensor, start: int, n: int) -> List[Dict]:
        """Merge n consecutive blocks into ring-sized segments: one strided copy
        per (layer, key) instead of one index_copy per block per layer."""
        u8 = flat.contiguous().view(n, -1).view(torch.uint8)  # [n, width_bytes]
        per = RING_TOKENS // BLOCK_TOKENS
        segs = []
        for b0 in range(0, n, per):
            b1 = min(n, b0 + per)
            m = b1 - b0
            last = (b1 == n)
            seg = {"t0": start + b0 * BLOCK_TOKENS, "t1": start + b1 * BLOCK_TOKENS, "layers": {}, "tail": {}}
            off = 0
            for g, l, k, shape, dtype, nb in self._spec:
                if g == "layers":
                    t = u8[b0:b1, off:off + nb].contiguous().view(dtype).view(m * shape[0], *shape[1:])
                    seg["layers"].setdefault(l, {})[k] = t
                elif last:
                    seg["tail"].setdefault(l, {})[k] = u8[b1 - 1, off:off + nb].view(dtype).view(shape)
                off += nb
            if not last:
                seg.pop("tail")
            segs.append(seg)
        return segs

    def export_kv(self, row: int, start: int, end: int, destination: torch.Tensor):
        """destination: [end-start, width] (pinned cpu ok); rank0 only."""
        assert start % BLOCK_TOKENS == 0 and (end - start) % BLOCK_TOKENS == 0
        slot = self._row_slot[row]
        assert end <= self.pool.pos[slot]
        for i, t0 in enumerate(range(start, end, BLOCK_TOKENS)):
            self._pack_block(slot, t0, destination[i * BLOCK_TOKENS:(i + 1) * BLOCK_TOKENS])

    def _unpack_compact(self, packet: torch.Tensor, start: int, n: int,
                        include_main: bool, include_tail: bool) -> List[Dict]:
        main_b, aux_b, tail_b = self._main_nbytes, self._aux_nbytes, self._tail_nbytes
        row_b = (main_b if include_main else 0) + aux_b
        u8 = packet.contiguous().view(-1).view(torch.uint8)
        rows = u8[:n * row_b].view(n, row_b)
        seg = {"t0": start, "t1": start + n * BLOCK_TOKENS,
               "layers": {}, "tail": {}}
        off = 0
        for g, l, k, shape, dtype, nb in self._spec:
            if g == "tail":
                off += nb
                continue
            is_main = (k == "main_kv")
            if is_main and not include_main:
                off += nb
                continue
            local = off if include_main else off - main_b
            t = rows[:, local:local + nb].contiguous().view(dtype)
            seg["layers"].setdefault(l, {})[k] = t.view(
                n * shape[0], *shape[1:])
            off += nb
        if include_tail:
            tail = u8[n * row_b:n * row_b + tail_b]
            toff = 0
            for g, l, k, shape, dtype, nb in self._spec:
                if g == "tail":
                    seg["tail"].setdefault(l, {})[k] = tail[
                        toff:toff + nb].view(dtype).view(shape)
                    toff += nb
        else:
            seg.pop("tail")
        return [seg]

    @staticmethod
    def _kv_v2_group_of(spec: tuple) -> str:
        g, _, k = spec[:3]
        return "tail" if g == "tail" else ("main" if k == "main_kv" else "aux")

    def load_kv_v2_group(self, row: int, start: int,
                         source: torch.Tensor, group: str) -> KVLoadResult:
        """Import contiguous Format-V2 rows without CPU gather staging."""
        slot = self._slot(row)
        assert group in ("main", "aux", "tail")
        assert source.device.type == "cpu" and source.dtype == torch.uint8
        assert source.ndim == 2 and source.is_contiguous()
        n = int(source.size(0))
        assert start % BLOCK_TOKENS == 0 and n > 0
        if group == "aux":
            assert start == self.pool.pos[slot], (start, self.pool.pos[slot])
        row_bytes = sum(s[-1] for s in self._spec if self._kv_v2_group_of(s) == group)
        assert source.size(1) == row_bytes, (source.size(1), row_bytes, group)
        self._rpc("import_v2", slot, start, n, group, row_bytes)
        self._do_import_v2_data(slot, start, source, group)
        return KVLoadResult(free_pages=self.free_pages)

    def _do_import_v2(self, slot: int, start: int, n: int,
                      group: str, row_bytes: int) -> None:
        if self.rank != 0:
            packet = torch.empty((n, row_bytes), dtype=torch.uint8, device=self.dev)
            self._do_import_v2_data(slot, start, packet, group)

    def _do_import_v2_data(self, slot: int, start: int,
                           packet: torch.Tensor, group: str) -> None:
        import time as _t
        t0 = _t.perf_counter()
        packet = packet.to(self.dev, non_blocking=False)
        torch.cuda.synchronize(self.dev); t1 = _t.perf_counter()
        if self.world > 1:
            dist.broadcast(packet, src=0)
        torch.cuda.synchronize(self.dev); t2 = _t.perf_counter()
        n, row_bytes = packet.shape
        end = start + n * BLOCK_TOKENS
        fields: Dict[int, Dict[str, torch.Tensor]] = {}
        offset = 0
        for spec in self._spec:
            if self._kv_v2_group_of(spec) != group:
                continue
            g, layer, key, shape, dtype, nbytes = spec
            part = packet[:, offset:offset + nbytes]
            fields.setdefault(layer, {})[key] = part.view(dtype).view(n, *shape).flatten(0, 1)
            offset += nbytes
        assert offset == row_bytes
        if group == "aux":
            self.pool.ensure(slot, end)
            for layer, blob in fields.items():
                self.pool.layers[layer].import_cold(slot, start, end, blob)
            self.pool.pos[slot] = end
            self.pool.pos_dev[slot] = end
        elif group == "main":
            for layer, blob in fields.items():
                self.pool.layers[layer].import_cold(slot, start, end, blob)
        else:
            assert n == 1, "tail restore is one 128-token boundary"
            for layer, blob in fields.items():
                self.pool.layers[layer].import_tail(slot, end, blob)
        torch.cuda.synchronize(self.dev); t3 = _t.perf_counter()
        if self.rank == 0:
            print(f"[kvload-v2] group={group} blocks={n} "
                  f"mib={packet.numel()/2**20:.2f} h2d={t1-t0:.3f}s "
                  f"bcast={t2-t1:.3f}s import={t3-t2:.3f}s", flush=True)

    def load_kv_span(self, row: int, start: int, end: int, flat: torch.Tensor,
                     pool1_tail_pages: int = 2,
                     include_main: Optional[bool] = None,
                     include_tail: Optional[bool] = None) -> KVLoadResult:
        slot = self._slot(row)
        assert start == self.pool.pos[slot] and start % BLOCK_TOKENS == 0
        assert (end - start) % BLOCK_TOKENS == 0
        if (include_main is None) != (include_tail is None):
            raise ValueError("include_main and include_tail must be provided together")
        legacy = include_main is None
        if legacy:
            assert flat.ndim == 2 and flat.shape == (end - start, self._width)
            include_main = include_tail = True
        assert include_main is not None and include_tail is not None
        if not legacy:
            n = (end - start) // BLOCK_TOKENS
            row_n = ((self._main_nbytes if include_main else 0)
                     + self._aux_nbytes) // 2
            expected = n * row_n + (self._tail_nbytes // 2 if include_tail else 0)
            assert flat.numel() == expected, (flat.numel(), expected)
        if end > start:
            self._rpc("import", slot, start, end, flat.numel(),
                      bool(include_main), bool(include_tail), legacy)
            self._do_import_data(slot, start, end, flat,
                                 bool(include_main), bool(include_tail), legacy)
        return KVLoadResult(free_pages=self.free_pages)

    def _do_import(self, slot: int, start: int, end: int, numel: int,
                   include_main: bool, include_tail: bool, legacy: bool):
        if self.rank != 0:
            flat = torch.empty(numel, dtype=torch.bfloat16, device=self.dev)
            self._do_import_data(slot, start, end, flat,
                                 include_main, include_tail, legacy)

    def _do_import_data(self, slot: int, start: int, end: int, flat: torch.Tensor,
                        include_main: bool, include_tail: bool, legacy: bool):
        import time as _t
        t0 = _t.perf_counter()
        flat = flat.to(self.dev, non_blocking=False)
        torch.cuda.synchronize(self.dev); t1 = _t.perf_counter()
        if self.world > 1:
            dist.broadcast(flat, src=0)
        torch.cuda.synchronize(self.dev); t2 = _t.perf_counter()
        n = (end - start) // BLOCK_TOKENS
        segs = (self._unpack_span(flat.view(end - start, self._width), start, n)
                if legacy else
                self._unpack_compact(flat, start, n, include_main, include_tail))
        t3 = _t.perf_counter()
        self.pool.import_cold(slot, segs)
        torch.cuda.synchronize(self.dev); t4 = _t.perf_counter()
        if self.rank == 0:
            mode = "full" if legacy else ("main+aux" if include_main else "aux")
            print(f"[kvload] blocks={n} mode={mode} tail={int(include_tail)} "
                  f"mib={flat.numel()*flat.element_size()/2**20:.2f} "
                  f"h2d={t1-t0:.3f}s bcast={t2-t1:.3f}s "
                  f"unpack={t3-t2:.3f}s import={t4-t3:.3f}s", flush=True)

    # ------------------------------------------------------------ forward
    def _do_forward(self, slot: int, start_pos: int, ids: tuple):
        # default device is thread-local and strategy drives us from its own thread;
        # scope it so strategy's pinned CPU allocs on the same thread stay on cpu.
        toks = torch.tensor([list(ids)], dtype=torch.int64, device=self.dev)
        with torch.no_grad(), torch.device(self.dev):
            out_ids, logits, mh = self.model(toks, start_pos=start_pos, full_logits=False, slot=slot)
            self.model.write_draft_kv(mh, start_pos, slot)
            self._mh[slot] = mh[:, -1:]
        lg = logits[0] if logits.dim() == 3 else logits
        return lg[-1].float()

    def _do_spec_init(self, slot: int, tok: int, pos: int, temp: float = 0.0):
        """tok = first generated token at position pos (not yet in kv). Drafts from it, sets step_g state."""
        with torch.no_grad(), torch.device(self.dev):
            self._temp_t[slot].fill_(max(temp, 0.0))
            first = torch.tensor([tok], dtype=torch.int64, device=self.dev)
            d0, _, _ = self.model.forward_spec(first, self._mh[slot], pos - 1, slot=slot)   # [1, block+1], d0[0,0]==tok
            qin = self._qin[slot]
            n = min(d0.size(1), SPEC_Q)
            qin[0, :n] = d0[0, :n]; qin[0, n:] = d0[0, n - 1]
            self._pos_t[slot].fill_(pos)

    def _do_step(self, slot: int, pos: int) -> list:
        """One B1Q8 verify->accept->draft step (CUDA graph replay). Returns new tokens (>=1)."""
        with torch.no_grad(), torch.device(self.dev):
            self.pool.ensure(slot, pos + SPEC_Q)
            graph, g, n_new = self._graph[slot]
            graph.replay()
            n = int(n_new.item())
            return g[:n].tolist()

    def _forward(self, slot, start_pos, ids) -> torch.Tensor:
        return self._rpc("forward", slot, start_pos, tuple(ids))

    @staticmethod
    def _pick(logits: torch.Tensor, temperature: float) -> int:
        if temperature <= 0:
            return int(torch.argmax(logits).item())
        probs = torch.softmax(logits / temperature, dim=-1)
        return int(torch.multinomial(probs, 1).item())

    # ------------------------------------------------------------ generate
    def warmup_batch_graphs(self, max_b=None) -> float:
        """Capture every batch graph up-front, while all slots are still empty.

        Capture runs a warmup step that writes garbage KV around pos 256 of each slot,
        so it is only safe *before* any prompt / restored prefix KV lives there.  At
        request time prepare_batch must therefore always find its graph already built
        (it then only re-binds row state, which is side-effect free).
        """
        n = int(self.pool.n_slots)
        max_b = n if max_b is None else min(int(max_b), n)
        if max_b < 2:
            return 0.0
        t0 = time.time()
        slots = [self._rpc("alloc") for _ in range(n)]
        try:
            for b in range(2, max_b + 1):
                for sub in itertools.combinations(slots, b):
                    self._rpc("prepare_batch", tuple(sub))
        finally:
            for sl in slots:
                self._rpc("release", sl)
        return time.time() - t0

    def _prefill_row(self, r, ids, limit, cancel, temp, eos, emit,
                     row_prefill_begin, row_prefill_end) -> dict:
        """Prefill one row (chunked, cold-KV store transaction), emit its first
        token and arm spec decode.  Shared by the initial batch and boarding."""
        if not ids:
            raise ValueError("batch rows must be non-empty")
        slot = self._slot(r)
        pos = self.pool.pos[slot]
        assert pos <= len(ids) and ids[:pos] == self._row_input.get(r, ())[:pos]
        # One sequence = one cold-KV store transaction.  Rows are prefilled
        # strictly one at a time, so the transaction opened here is closed
        # before the next row opens its own -- identical to the B=1 path.
        if row_prefill_begin is not None:
            row_prefill_begin(r)
        stored = False
        try:
            logits = None
            while pos < len(ids):
                n = _prefill_chunk_size(pos, len(ids), self.prefill_chunk)
                chunk_start = pos
                logits = self._forward(slot, pos, ids[pos:pos + n])
                pos += n
                self._capture_kv_v2_chunk(slot, chunk_start, pos)
            if logits is None:  # full prefix hit: re-run last token for logits
                pos -= 1
                self._rpc("rewind", slot, pos)
                logits = self._forward(slot, pos, ids[pos:pos + 1])
                pos += 1
            stored = True
        finally:
            if row_prefill_end is not None:
                row_prefill_end(r, stored)
        tok = self._pick(logits, temp)
        # First token is NOT emitted here: the strategy publishes the row's
        # "prefill" event only after this returns (on_prefill / on_boarded), and
        # the API layer requires that event to precede any "token" event.
        # Callers emit d["first_tok"] right after publishing.
        d = {"row": r, "slot": slot, "pos": pos, "temp": temp, "limit": limit,
             "cancel": cancel, "n_gen": 1, "n_steps": 0, "t_end": None,
             "first_tok": tok,
             "done": tok == eos or 1 >= limit or cancel.is_set()}
        if not d["done"]:
            self._rpc("spec_init", slot, tok, pos, temp)
        return d

    def _generate_multi(self, input_ids, max_new_tokens, cancel_events, emit,
                        temperatures, eos_token_id, on_prefill, stats,
                        row_prefill_begin=None, row_prefill_end=None,
                        board_request=None, on_boarded=None,
                        boarding_interval_steps: int = 128) -> None:
        """B>1: prefill every row, then advance the whole batch through one captured
        graph (step_g_batch).  Rows leave as they hit eos/limit/cancel; the surviving
        subset is re-bound to its own graph (unbind writes per-row spec state back to
        the slot buffers first, so the hand-off is exact)."""
        B = len(input_ids)
        eos = EOS_DEFAULT if eos_token_id is None else int(eos_token_id)
        rows = []
        t0 = time.time()
        for r in range(B):
            ids = tuple(int(t) for t in input_ids[r])
            temp = float(temperatures[r]) if temperatures else 0.0
            rows.append(self._prefill_row(r, ids, int(max_new_tokens[r]), cancel_events[r],
                                          temp, eos, emit, row_prefill_begin, row_prefill_end))
        t1 = time.time()
        n_slots = int(self.pool.n_slots)
        interval = max(1, int(boarding_interval_steps))
        if on_prefill is not None:
            on_prefill()
        for d in rows:
            emit(d["row"], [d.pop("first_tok")])

        step_seconds = []
        active = [d for d in rows if not d["done"]]
        bound = None
        while active:
            # Batch graphs are pre-captured per *sorted* slot tuple (warmup order);
            # pool.alloc is LIFO so live rows may hold slots in any order.
            order = sorted(range(len(active)), key=lambda i: active[i]["slot"])
            slots = tuple(active[i]["slot"] for i in order)
            if len(active) == 1:
                # Last row standing: fall back to the B1 graph instead of binding a
                # single-slot batch graph.  Such a graph is not pre-captured, and
                # capturing one here would run a warmup step over this slot's live KV.
                # unbind_batch already wrote _qin/_pos_t/_temp_t back -- exactly the
                # state the B1 step_g graph reads.
                if bound is not None:
                    self._rpc("unbind_batch", bound)
                    bound = None
                ts = time.perf_counter()
                outs = [self._rpc("step", active[0]["slot"], active[0]["pos"])]
            else:
                if slots != bound:
                    if bound is not None:
                        self._rpc("unbind_batch", bound)   # write state back first
                    self._rpc("prepare_batch", slots)
                    bound = slots
                ts = time.perf_counter()
                outs_sorted = self._rpc("step_batch", slots, tuple(active[i]["pos"] for i in order))
                outs = [None] * len(active)
                for k, i in enumerate(order):
                    outs[i] = outs_sorted[k]
            step_seconds.append(time.perf_counter() - ts)
            for d, new in zip(active, outs):
                d["n_steps"] += 1
                keep = len(new)
                for k, t in enumerate(new):
                    if t == eos or d["n_gen"] + k + 1 >= d["limit"]:
                        keep = k + 1
                        break
                new = new[:keep]
                d["n_gen"] += keep
                d["pos"] += keep
                emit(d["row"], new)
                if new[-1] == eos or d["n_gen"] >= d["limit"] or d["cancel"].is_set():
                    d["done"] = True
                    d["t_end"] = time.time()
            if any(d["done"] for d in active):
                active = [d for d in active if not d["done"]]
            # ---- boarding safe point: every `interval` steps, admit one queued
            # request onto a fresh tail row (no slot reuse) while active rows
            # pause.  New row is prefilled with the batch unbound, so spec state
            # of live rows is already written back to their slot buffers.
            if (board_request is not None and active
                    and len(step_seconds) % interval == 0
                    and len(rows) < n_slots):
                new_row = len(rows)
                br = board_request(new_row)
                if br is not None:
                    if bound is not None:
                        self._rpc("unbind_batch", bound)
                        bound = None
                    ids = tuple(int(t) for t in br.input_ids)
                    d = self._prefill_row(new_row, ids, int(br.max_new_tokens), br.cancel_event,
                                          float(br.temperature), eos, emit,
                                          row_prefill_begin, row_prefill_end)
                    rows.append(d)
                    if not d["done"]:
                        active.append(d)
                    if on_boarded is not None:
                        on_boarded(new_row, len(active))
                    emit(new_row, [d.pop("first_tok")])
        if bound is not None:
            self._rpc("unbind_batch", bound)
        for d in rows:
            self._rpc("rewind", d["slot"], d["pos"])
        t2 = time.time()
        if stats is not None:
            stats.update(prefill_seconds=t1 - t0, decode_seconds=t2 - t1,
                         row_decode_seconds=[(d["t_end"] or t2) - t1 for d in rows],
                         row_steps=[d["n_steps"] for d in rows],
                         steps=len(step_seconds),
                         accepts=[max(d["n_gen"] - 1 - d["n_steps"], 0) for d in rows],
                         step_seconds=step_seconds)

    def generate_batch(self, input_ids: Sequence[Sequence[int]],
                       max_new_tokens: Sequence[int],
                       cancel_events: Sequence[threading.Event],
                       emit: Callable[[int, list[int]], None],
                       temperatures: Optional[Sequence[float]] = None,
                       eos_token_id: Optional[int] = None,
                       on_prefill: Optional[Callable[[], None]] = None,
                       stats: Optional[dict] = None,
                       select_active_rows=None, board_request=None,
                       on_boarded=None, boarding_interval_steps: int = 128,
                       row_prefill_begin=None, row_prefill_end=None) -> None:
        if len(input_ids) > 1 or (len(input_ids) == 1 and board_request is not None):
            return self._generate_multi(input_ids, max_new_tokens, cancel_events, emit,
                                        temperatures, eos_token_id, on_prefill, stats,
                                        row_prefill_begin, row_prefill_end,
                                        board_request, on_boarded, boarding_interval_steps)
        if len(input_ids) != 1:
            raise ValueError(f"unsupported batch size B={len(input_ids)}")
        ids = tuple(int(t) for t in input_ids[0])
        if not ids:
            raise ValueError("batch rows must be non-empty")
        limit = int(max_new_tokens[0])
        cancel = cancel_events[0]
        temp = float(temperatures[0]) if temperatures else 0.0
        eos = EOS_DEFAULT if eos_token_id is None else int(eos_token_id)
        row = 0
        slot = self._slot(row)
        pos = self.pool.pos[slot]
        assert pos <= len(ids) and ids[:pos] == self._row_input.get(row, ())[:pos]

        t0 = time.time()
        # prefill remaining prompt, chunk starts stay 128-aligned.
        # Same per-sequence store transaction as the B>1 path.
        if row_prefill_begin is not None:
            row_prefill_begin(row)
        stored = False
        try:
            logits = None
            while pos < len(ids):
                n = _prefill_chunk_size(pos, len(ids), self.prefill_chunk)
                chunk_start = pos
                logits = self._forward(slot, pos, ids[pos:pos + n])
                pos += n
                self._capture_kv_v2_chunk(slot, chunk_start, pos)
            if logits is None:  # full prefix hit: re-run last token for logits
                pos -= 1
                self._rpc("rewind", slot, pos)
                logits = self._forward(slot, pos, ids[pos:pos + 1])
                pos += 1
            stored = True
        finally:
            if row_prefill_end is not None:
                row_prefill_end(row, stored)
        t1 = time.time()
        if on_prefill is not None:
            on_prefill()

        n_gen = 0
        n_steps = 0
        step_seconds = []
        # B1Q8 speculative decode via Transformer.step_g for any temperature (temp>0 samples the
        # target in-graph via Gumbel-max); pos = position of last emitted token
        tok = self._pick(logits, temp)
        n_gen += 1
        emit(row, [tok])
        if tok != eos and n_gen < limit and not cancel.is_set():
            self._rpc("spec_init", slot, tok, pos, temp)
            while n_gen < limit and not cancel.is_set():
                ts = time.perf_counter()
                new = self._rpc("step", slot, pos)
                step_seconds.append(time.perf_counter() - ts)
                n_steps += 1
                keep = len(new)
                for k, t in enumerate(new):
                    if t == eos or n_gen + k + 1 >= limit:
                        keep = k + 1
                        break
                new = new[:keep]
                n_gen += keep
                pos += keep
                emit(row, new)
                if new[-1] == eos:
                    break
            self._rpc("rewind", slot, pos)
        t2 = time.time()
        if stats is not None:
            # steps = step_g replays; first token comes from prefill logits, so decode tokens = n_gen - 1
            stats.update(prefill_seconds=t1 - t0, decode_seconds=t2 - t1,
                         row_decode_seconds=[t2 - t1], row_steps=[n_steps],
                         steps=n_steps, accepts=[max(n_gen - 1 - n_steps, 0)],
                         step_seconds=step_seconds)

    def _do_rewind(self, slot: int, pos: int):
        self.pool.pos[slot] = pos
        self.pool.pos_dev[slot] = pos


# ---------------------------------------------------------------- loading
def load_execution(verbose: bool = True, max_seq_len: int | None = None, max_batch_size: int | None = None) -> ModelExecution:
    """torchrun entry: builds TP model on this rank (mirrors tc_oracle_decode_dist)."""
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world > 1 and not dist.is_initialized():
        dist.init_process_group("nccl")
    global _CTRL_PG
    if world > 1 and _CTRL_PG is None:
        # This group only carries idle "wait for the next command" broadcasts;
        # gloo's default 30 min recv timeout would kill every worker rank after
        # half an hour without traffic (observed: rank2 exitcode 3).
        _CTRL_PG = dist.new_group(backend="gloo",
                                  timeout=_dt.timedelta(days=365))
    torch.cuda.set_device(rank)
    from model import wcache, arch
    from model.arch import Transformer
    from model.bind import bind, check_bound
    from model.args_dsv4 import make_args
    arch.init_distributed()
    W = wcache.load("tp8", rank=rank, verbose=(verbose and rank == 0))
    torch.set_default_dtype(torch.bfloat16)
    model = Transformer(make_args(**{k: v for k, v in {"max_seq_len": max_seq_len, "max_batch_size": max_batch_size}.items() if v is not None}))
    bind(model, W)
    bad = [n for n in check_bound(model) if not n.startswith("mtp.")]
    assert not bad, bad
    dev = f"cuda:{rank}"
    for m in model.modules():
        for k, v in list(m._buffers.items()):
            if v is not None and v.device.type == "cpu":
                m._buffers[k] = v.to(dev)
    torch.set_default_device(dev)
    if world > 1:
        dist.barrier()
    return ModelExecution(model, rank, world)
