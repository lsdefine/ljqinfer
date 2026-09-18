"""Single-owner TP4 engine facade.

One process owns one rank's resident weights, cache and collective.  Prefill and
decode both call :meth:`forward_tokens`, so replacing kernels or MTP does not
change scheduling/cache ownership.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable

from ops.kernels import K
from .blocks import block_forward
from .config import CONFIG, EngineConfig
from .runtime import RuntimeCache, allocate_mock_cache
from .weights import Weights


@dataclass
class Engine:
    rank: int
    weights: Weights
    cache: Any
    config: object = CONFIG
    engine_config: EngineConfig | None = None
    device: str = "cpu"
    collective: Any | None = None
    mtp: Any | None = None

    @classmethod
    def load_mock(cls, rank=0, max_tokens=131072, max_sequences=4):
        if not 0 <= rank < CONFIG.tp:
            raise ValueError("rank must be 0..3")
        return cls(rank, Weights.mock(rank),
                   allocate_mock_cache(max_tokens, max_sequences))

    @classmethod
    def load(cls, rank: int, engine_config: EngineConfig | None = None,
             device: str | None = None, collective: Any | None = None,
             load_mtp: bool = False):
        cfg = engine_config or EngineConfig()
        if not 0 <= rank < CONFIG.tp:
            raise ValueError("rank must be 0..3")
        device = device or f"npu:{rank}"
        weights = Weights(rank, cfg.model_dir, device=device)
        spec = allocate_mock_cache(
            max_tokens=cfg.max_cached_tokens,
            max_sequences=cfg.max_sequences,
            page_size=cfg.kv_page_size,
            checkpoint_interval=cfg.cold_checkpoint_interval,
            prefill_chunk_size=cfg.prefill_chunk_size,
            max_sequence_tokens=cfg.max_sequence_tokens)
        cache = RuntimeCache.allocate(spec, device)
        engine = cls(rank, weights, cache, CONFIG, cfg, device, collective)
        if load_mtp:
            from .mtp import install_mtp
            engine.mtp = install_mtp(cfg.mtp_backend)
            engine.mtp.load(engine)
        return engine

    def all_reduce(self, tensor):
        if self.collective is None:
            # Useful only for rank-local correctness probes. Production TP4
            # launchers must inject a collective object.
            return tensor
        result = self.collective.all_reduce(tensor)
        return tensor if result is None else result

    def _sequence_ids(self, sequence_ids, token_count: int):
        import torch
        if sequence_ids is None:
            sequence_ids = torch.zeros(token_count, dtype=torch.int64,
                                       device=self.device)
        elif not torch.is_tensor(sequence_ids):
            sequence_ids = torch.tensor(sequence_ids, dtype=torch.int64,
                                        device=self.device)
        else:
            sequence_ids = sequence_ids.to(self.device, dtype=torch.int64)
        if sequence_ids.shape != (token_count,):
            raise ValueError("sequence_ids must have one entry per token")
        return sequence_ids

    def forward_tokens(self, input_ids, sequence_ids=None, positions=None,
                       update_cache: bool = True, host_sequence_id: int | None = None,
                       return_aux: bool = False, aux_layer_ids=(5, 19, 33, 47, 61),
                       collect_gdn_checkpoints: bool = False):
        """Run real decoder layers for a packed token batch.

        Tokens of each sequence must occur in causal order.  A sequence may
        appear multiple times (prefill) or once (batched decode).
        """
        import torch
        if not torch.is_tensor(input_ids):
            input_ids = torch.tensor(input_ids, dtype=torch.int64,
                                     device=self.device)
        else:
            input_ids = input_ids.to(self.device, dtype=torch.int64)
        input_ids = input_ids.reshape(-1)
        sids = self._sequence_ids(sequence_ids, input_ids.numel())
        if host_sequence_id is not None:
            host_sequence_id = int(host_sequence_id)
            host_rows = None
            host_sids = (host_sequence_id,)
        else:
            host_rows = [int(x) for x in sids.detach().cpu().tolist()]
            host_sids = tuple(dict.fromkeys(host_rows))
        old_lengths = {sid: int(self.cache.lengths[sid].item()) for sid in host_sids}
        checkpoint_offsets = ()
        checkpoint_start = 0
        if collect_gdn_checkpoints:
            if host_sequence_id is None:
                raise ValueError("GDN checkpoint collection requires one host sequence")
            checkpoint_start = old_lengths[host_sequence_id]
            checkpoint_offsets = self.cache.begin_gdn_checkpoints(
                host_sequence_id, checkpoint_start, input_ids.numel())
        if positions is None:
            if host_sequence_id is not None:
                positions = torch.arange(old_lengths[host_sequence_id],
                                         old_lengths[host_sequence_id] + input_ids.numel(),
                                         dtype=torch.int64, device=self.device)
            else:
                seen = dict(old_lengths)
                values = []
                for sid in host_rows:
                    values.append(seen[sid]); seen[sid] += 1
                positions = torch.tensor(values, dtype=torch.int64,
                                         device=self.device)
        else:
            positions = positions.to(self.device, dtype=torch.int64)
        hidden = K.embedding(input_ids, self.weights.embedding)
        aux = [] if return_aux else None
        aux_ids = set(aux_layer_ids) if return_aux else set()
        for layer_idx in range(CONFIG.num_hidden_layers):
            ctx = self.cache.layer_context(
                layer_idx, sids, positions, update_cache, host_sids,
                checkpoint_start, checkpoint_offsets)
            hidden = block_forward(hidden, layer_idx,
                                   self.weights.layer(layer_idx), ctx,
                                   self.all_reduce)
            if layer_idx in aux_ids:
                aux.append(hidden)
            if layer_idx in CONFIG.full_attention_layers:
                self.cache.commit_full_attention(layer_idx, ctx, old_lengths)
        hidden = K.rms_norm(hidden, self.weights.final_norm,
                            CONFIG.rms_norm_eps)
        if update_cache:
            counts: dict[int, int] = {}
            if host_sequence_id is not None:
                counts[host_sequence_id] = input_ids.numel()
            else:
                for sid in host_rows:
                    counts[sid] = counts.get(sid, 0) + 1
            for sid, added in counts.items():
                self.cache.lengths[sid] = old_lengths[sid] + added
        return (hidden, aux) if return_aux else hidden

    def local_logits(self, hidden):
        """Return this rank's vocabulary shard logits."""
        import torch
        return torch.matmul(hidden, self.weights.lm_head.t())

    def release_sequence(self, sequence_id: int) -> None:
        self.cache.release_sequence(sequence_id)
