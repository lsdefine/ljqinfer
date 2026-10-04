"""Tensor-only CED sparse packing; never materializes the full history.

Input local contains the retained CED tail, not the encoder sliding ring.
The returned zero slots have distinct IDs for the vendor sparse attention ABI.
"""
import torch


def pack(local, pool, table, slot, selected, positions, origin, valid):
    """Pack at most T*K global rows from ratio-one paged CED history."""
    t, k = selected.shape
    length, dim = local.shape
    page_rows = pool.shape[1]
    logical = selected.long()
    limit = torch.minimum(positions + 1, valid.reshape(()))
    global_ok = (logical >= 0) & (logical < limit[:, None])
    safe = torch.where(global_ok, logical, 0)
    page_table = table.index_select(0, slot.long()).reshape(-1)
    physical_pages = page_table.index_select(0, (safe // page_rows).reshape(-1))
    physical = physical_pages.clamp_min(0) * page_rows + safe.reshape(-1) % page_rows
    count = 128 + k
    packed = local.new_empty((length + t*k + count, dim))
    packed[:length].copy_(local)
    gathered = packed[length:length + t*k]
    torch.index_select(pool.flatten(0, 1), 0, physical, out=gathered)
    gathered.masked_fill_(~global_ok.reshape(-1, 1), 0)
    packed[length + t*k:].zero_()

    offsets = torch.arange(128, device=local.device)
    absolute = positions[:, None] - 127 + offsets
    local_ids = absolute - origin.reshape(())
    local_ok = (positions[:, None] >= 0) & (absolute >= 0) & (local_ids >= 0) & (local_ids < length)
    global_ids = length + torch.arange(t*k, device=local.device).reshape(t, k)
    ids = torch.cat((local_ids, global_ids), dim=1)
    ok = torch.cat((local_ok, global_ok), dim=1)
    count = 128 + k
    zeros = length + t*k + torch.arange(count, device=local.device)
    indices = torch.where(ok, ids, zeros).to(torch.int32)[None, :, None].contiguous()
    packed = packed[None, :, None]
    missing = torch.zeros((t, 32), device=local.device, dtype=torch.int32)
    missing[:, 0] = (~ok).sum(dim=1).to(torch.int32)
    return packed, indices, missing
