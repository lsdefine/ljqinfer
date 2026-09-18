"""Chunked prefill orchestration over the single-owner Engine."""
from __future__ import annotations


def _chunk_ends(start: int, total: int, chunk_size: int, checkpoint_interval: int):
    """Yield large model-forward chunks; cold boundaries never split a call."""
    del start, checkpoint_interval
    for lo in range(0, total, chunk_size):
        yield lo, min(total, lo + chunk_size)


def prefill(engine, input_ids, sequence_id: int = 0, chunk_size: int | None = None,
            return_all_hidden: bool = True):
    """Prefill one sequence in >=8K model calls backed by preallocated outputs.

    Cold checkpoints are captured inside GDN layers; they never split a model
    forward at the 1024-token storage interval.
    """
    import torch
    if not engine.weights.loaded:
        # Preserve the original scaffold contract used by lightweight tests.
        from .blocks import block_forward
        hidden = {"shape": (len(input_ids), engine.config.hidden_size),
                  "dtype": "bfloat16", "mock": True}
        for i in range(engine.config.num_hidden_layers):
            hidden = block_forward(hidden, i, ctx=engine.cache)
        return hidden
    ids = (input_ids if torch.is_tensor(input_ids)
           else torch.tensor(input_ids, dtype=torch.int64, device=engine.device))
    ids = ids.to(engine.device, dtype=torch.int64).reshape(-1)
    cfg = engine.engine_config
    chunk_size = int(chunk_size or cfg.prefill_chunk_size)
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    start = int(engine.cache.lengths[int(sequence_id)].item())
    if ids.numel() == 0:
        if return_all_hidden:
            return torch.empty((0, engine.config.hidden_size), dtype=torch.bfloat16,
                               device=engine.device)
        return None
    output = (torch.empty((ids.numel(), engine.config.hidden_size),
                          dtype=torch.bfloat16, device=engine.device)
              if return_all_hidden else None)
    hidden = None
    for lo, hi in _chunk_ends(start, ids.numel(), chunk_size,
                              cfg.cold_checkpoint_interval):
        hidden = engine.forward_tokens(
            ids[lo:hi], host_sequence_id=int(sequence_id),
            collect_gdn_checkpoints=True)
        if output is not None:
            output[lo:hi].copy_(hidden)
        engine.cache.flush_gdn_checkpoints()
    engine.cache.wait_gdn_checkpoints()
    return output if return_all_hidden else hidden[-1:]
