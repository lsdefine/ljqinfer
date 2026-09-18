"""Stable, replaceable MTP plugin ABI.

The base engine depends only on ``MTPBackend``. A future MTP implementation can
be selected by registry name or injected as an object without changing model,
strategy, cache, graph, or service code.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Any

from ops.kernels import K
from .blocks import LayerContext, block_forward
from .config import CONFIG

@dataclass(frozen=True)
class MTPContext:
    input_ids: Any
    positions: Any
    hidden_states: Any
    sequence_ids: Any | None = None
    cache: Any | None = None

@dataclass(frozen=True)
class MTPDraft:
    token_ids: Any
    logits: Any | None = None
    hidden_states: Any | None = None

class MTPBackend(ABC):
    name = "abstract"

    @abstractmethod
    def load(self, engine) -> None: ...

    @abstractmethod
    def draft(self, context: MTPContext, num_tokens: int) -> MTPDraft: ...

    def begin_transaction(self, sequence_ids) -> None:
        return None

    def commit(self, accepted_lengths) -> None:
        return None

    def rollback(self, accepted_lengths) -> None:
        return None

class DisabledMTP(MTPBackend):
    name = "disabled"
    def load(self, engine) -> None: self.engine = engine
    def draft(self, context: MTPContext, num_tokens: int) -> MTPDraft:
        raise RuntimeError("MTP backend is disabled")


class QwenNativeMTP(MTPBackend):
    """Checkpoint-native one-token Qwen3.5 predictor.

    The plugin owns its attention history. A draft writes only ``_pending``;
    target-model verification can therefore commit or discard it without ever
    corrupting the base engine cache.
    """

    name = "qwen_native"

    def load(self, engine) -> None:
        self.engine = engine
        self.weights = engine.weights.mtp
        self._k: dict[int, Any] = {}
        self._v: dict[int, Any] = {}
        self._pending: tuple[list[int], dict[int, Any], dict[int, Any]] | None = None
        self._transaction_sids: list[int] | None = None

    def begin_transaction(self, sequence_ids) -> None:
        import torch
        if self._pending is not None:
            raise RuntimeError("previous MTP draft has not been committed or rolled back")
        if torch.is_tensor(sequence_ids):
            sequence_ids = sequence_ids.detach().cpu().tolist()
        self._transaction_sids = [int(x) for x in sequence_ids]

    def _all_gather_last_dim(self, local):
        if self.engine.collective is None:
            raise RuntimeError("qwen_native MTP requires the TP4 collective")
        gathered = self.engine.collective.all_gather(local)
        # HCCL returns [rank, token, H/tp]. Reassemble token-major hidden rows.
        return gathered.permute(1, 0, 2).reshape(local.shape[0], -1)

    def _global_argmax(self, local_logits):
        gathered = self.engine.collective.all_gather(local_logits)
        logits = gathered.permute(1, 0, 2).reshape(local_logits.shape[0], -1)
        return logits.float().argmax(dim=-1)

    def draft(self, context: MTPContext, num_tokens: int) -> MTPDraft:
        import torch
        if num_tokens != 1:
            raise ValueError("this checkpoint has one MTP layer and supports num_tokens=1")
        ids = context.input_ids.to(self.engine.device, dtype=torch.int64).reshape(-1)
        positions = context.positions.to(self.engine.device, dtype=torch.int64).reshape(-1)
        hidden = context.hidden_states.to(self.engine.device).reshape(-1, CONFIG.hidden_size)
        if ids.numel() != hidden.shape[0] or positions.numel() != ids.numel():
            raise ValueError("MTP input_ids, positions and hidden_states must be token-aligned")
        if context.sequence_ids is None:
            sids = list(range(ids.numel()))
        elif torch.is_tensor(context.sequence_ids):
            sids = [int(x) for x in context.sequence_ids.detach().cpu().tolist()]
        else:
            sids = [int(x) for x in context.sequence_ids]
        if len(sids) != ids.numel():
            raise ValueError("MTP sequence_ids must have one entry per token")
        if self._transaction_sids is None:
            self.begin_transaction(sids)
        elif self._transaction_sids != sids:
            raise ValueError("MTP transaction sequence_ids differ from draft")

        embedding = K.embedding(ids, self.engine.weights.embedding)
        embedding = K.rms_norm(embedding, self.weights.pre_fc_norm_embedding,
                               CONFIG.rms_norm_eps)
        hidden = K.rms_norm(hidden, self.weights.pre_fc_norm_hidden,
                            CONFIG.rms_norm_eps)
        local = K.bf16_linear(torch.cat((embedding, hidden), dim=-1),
                              self.weights.fc)
        fused = self._all_gather_last_dim(local)

        # Shallow copies are sufficient: full_attention appends by torch.cat and
        # replaces dictionary values; committed tensors themselves are immutable.
        pk = {sid: self._k.get(sid) for sid in set(sids)}
        pv = {sid: self._v.get(sid) for sid in set(sids)}
        seq = torch.tensor(sids, dtype=torch.int64, device=self.engine.device)
        layer_ctx = LayerContext(positions, seq, past_k=pk, past_v=pv,
                                 update_kv=True)
        out = block_forward(fused, CONFIG.num_hidden_layers, self.weights.layer,
                            layer_ctx, self.engine.all_reduce)
        out = K.rms_norm(out, self.weights.norm, CONFIG.rms_norm_eps)
        local_logits = self.engine.local_logits(out)
        token_ids = self._global_argmax(local_logits)
        self._pending = (sids, pk, pv)
        return MTPDraft(token_ids, local_logits, out)

    @staticmethod
    def _accepted_map(sids: list[int], accepted_lengths) -> dict[int, int]:
        import torch
        if isinstance(accepted_lengths, int):
            return {sid: int(accepted_lengths) for sid in set(sids)}
        if torch.is_tensor(accepted_lengths):
            accepted_lengths = accepted_lengths.detach().cpu().tolist()
        values = [int(x) for x in accepted_lengths]
        unique = list(dict.fromkeys(sids))
        if len(values) != len(unique):
            raise ValueError("accepted_lengths must have one entry per sequence")
        return dict(zip(unique, values))

    def commit(self, accepted_lengths) -> None:
        if self._pending is None:
            raise RuntimeError("no pending MTP transaction")
        sids, pk, pv = self._pending
        accepted = self._accepted_map(sids, accepted_lengths)
        for sid in set(sids):
            if accepted[sid] not in (0, 1):
                raise ValueError("one-layer MTP accepts only lengths 0 or 1")
            if accepted[sid] == 1:
                self._k[sid], self._v[sid] = pk[sid], pv[sid]
        self._pending = None
        self._transaction_sids = None

    def rollback(self, accepted_lengths) -> None:
        # Rollback may retain an accepted prefix; with one draft token this is
        # exactly the same 0/1 state transition as commit.
        self.commit(accepted_lengths)


_REGISTRY: dict[str, Callable[[], MTPBackend]] = {
    DisabledMTP.name: DisabledMTP,
    QwenNativeMTP.name: QwenNativeMTP,
}

def register_mtp(name: str, factory: Callable[[], MTPBackend]) -> None:
    if not name or name in _REGISTRY:
        raise ValueError(f"invalid or duplicate MTP backend {name!r}")
    _REGISTRY[name] = factory

def create_mtp(name: str) -> MTPBackend:
    try:
        return _REGISTRY[name]()
    except KeyError as exc:
        raise KeyError(f"unknown MTP backend {name!r}; available={sorted(_REGISTRY)}") from exc

def install_mtp(backend: MTPBackend | str) -> MTPBackend:
    return create_mtp(backend) if isinstance(backend, str) else backend
