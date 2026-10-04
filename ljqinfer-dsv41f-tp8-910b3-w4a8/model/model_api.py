"""Execution boundary: model computes; this layer reserves and commits Past.

Forward writes KV/window/carry for [start, start+len(tokens)) but MUST NOT
advance pool.pos, allocate/release slots, invoke cold storage or schedule work.
Any forward failure may leave dirty tensors: the owner must discard the slot.
All calls for a pool use one serialized execution lane and its CUDA stream.
"""
from dataclasses import dataclass
from typing import Any, Protocol
import torch


class ModelCompute(Protocol):
    def finish_prefill(self, *, past, slot: int) -> Any:
        """Prepare decoder SWA/logits from the hot encoder tail; no position commit."""
        ...

    def forward(self, tokens: tuple[int, ...], *, past, slot: int,
                start: int, history_tokens: tuple[int, ...] = ()) -> Any: ...

    def replay(self, tokens: tuple[int, ...], *, past, slot: int,
               start: int, history_tokens: tuple[int, ...]) -> None:
        """Rebuild hot-only state for [start, past.pos[slot]).

        Cached complete global KV/index rows are READ ONLY. Causal reads
        are bounded by each replay query position, not just the loaded end.
        Truncate SWA at start; restore Engram from history_tokens (up to 3).
        Rebuild incomplete compressor carry at the hit boundary, even though
        completed global rows must not be recomputed or overwritten. Handle
        encoder/decoder/DSpark hot state as required by the compute backend.
        No slot allocation, position commit or cold-cache access is allowed.
        """
        ...


class PrefillCancelled(Exception):
    pass


@dataclass(frozen=True)
class ChunkResult:
    start: int
    end: int
    output: Any


class ModelExecution:
    def __init__(self, compute: ModelCompute, past, *, prefill_chunk_tokens=12288):
        if not 1 <= prefill_chunk_tokens <= 12288:
            raise ValueError('prefill chunk size must be 1..12288')
        self.compute, self.past = compute, past
        self.prefill_chunk_tokens = prefill_chunk_tokens

    def replay_prefix(self, slot, prefix_tokens, *, cancel=None):
        """Bounded cold resume; token history is already owned by the caller."""
        p = self.past
        if slot in p.free_slots or not 0 <= slot < p.n_slots:
            raise ValueError('execution requires a live slot')
        if slot not in p.replay_pending:
            return 0
        end = p.pos[slot]
        if len(prefix_tokens) != end:
            raise ValueError('replay requires exactly the cached prefix tokens')
        if cancel is not None and cancel.is_set():
            raise PrefillCancelled()
        start = max(0, end - p.window)
        with torch.inference_mode():
            self.compute.replay(tuple(prefix_tokens[start:end]), past=p, slot=slot,
                                start=start,
                                history_tokens=tuple(prefix_tokens[max(0, start-3):start]))
        if p.pos[slot] != end:
            raise RuntimeError('replay must not commit Past position')
        if cancel is not None and cancel.is_set():
            raise PrefillCancelled()
        p.replay_pending.remove(slot)
        return end - start

    def finish_prefill(self, slot, *, cancel=None):
        """Explicit generation preparation; pure cache-prefill need not call it."""
        p = self.past
        if not 0 <= slot < p.n_slots or slot in p.free_slots or slot in p.replay_pending:
            raise ValueError('finish requires a live hot slot')
        if cancel is not None and cancel.is_set():
            raise PrefillCancelled()
        end = p.pos[slot]
        with torch.inference_mode():
            output = self.compute.finish_prefill(past=p, slot=slot)
        if p.pos[slot] != end:
            raise RuntimeError('finish must not commit Past position')
        if cancel is not None and cancel.is_set():
            raise PrefillCancelled()
        return output

    def prefill_chunk(self, slot, tokens, *, history_tokens=(), cancel=None):
        tokens = tuple(tokens)
        if slot in self.past.free_slots or not 0 <= slot < self.past.n_slots:
            raise ValueError('execution requires a live slot')
        if not 0 < len(tokens) <= self.prefill_chunk_tokens:
            raise ValueError('invalid prefill chunk length')
        if cancel is not None and cancel.is_set():
            raise PrefillCancelled()
        if slot in self.past.replay_pending:
            raise RuntimeError('cold Past requires bounded replay before prefill')
        start = self.past.pos[slot]
        end = start + len(tokens)
        if end > self.past.max_seq:
            raise ValueError('maximum sequence length exceeded')
        self.past.ensure(slot, end)
        with torch.inference_mode():
            output = self.compute.forward(tokens, past=self.past, slot=slot, start=start,
                                          history_tokens=tuple(history_tokens))
        if self.past.pos[slot] != start:
            raise RuntimeError('model must not commit Past position')
        if cancel is not None and cancel.is_set():
            raise PrefillCancelled()
        self.past.set_pos(slot, end)
        return ChunkResult(start, end, output)
