"""Batched one-token decode shared by eager and graph runners."""
from __future__ import annotations


def decode_step(engine, token_ids, sequence_ids=None, return_logits: bool = True):
    """Advance one token for every independent active sequence row."""
    import torch
    if not engine.weights.loaded:
        from .prefill import prefill
        return prefill(engine, token_ids)
    ids = (token_ids if torch.is_tensor(token_ids) else
           torch.tensor(token_ids, dtype=torch.int64, device=engine.device))
    ids = ids.to(engine.device, dtype=torch.int64).reshape(-1)
    if sequence_ids is None:
        sequence_ids = torch.arange(ids.numel(), dtype=torch.int64,
                                    device=engine.device)
    elif not torch.is_tensor(sequence_ids):
        sequence_ids = torch.tensor(sequence_ids, dtype=torch.int64,
                                    device=engine.device)
    else:
        sequence_ids = sequence_ids.to(engine.device, dtype=torch.int64)
    sequence_ids = sequence_ids.reshape(-1)
    if sequence_ids.shape != ids.shape:
        raise ValueError("sequence_ids and token_ids must have equal shape")
    host_sids = [int(x) for x in sequence_ids.detach().cpu().tolist()]
    if len(set(host_sids)) != len(host_sids):
        raise ValueError("batched decode requires one row per sequence")
    hidden = engine.forward_tokens(ids, sequence_ids)
    return engine.local_logits(hidden) if return_logits else hidden
