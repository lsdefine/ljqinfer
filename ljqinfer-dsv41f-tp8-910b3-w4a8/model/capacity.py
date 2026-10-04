"""Physical BF16 Past budget, independent of logical sequence capacity."""
import torch
from .past import SlotPool


def past_bytes(pool_tokens, slots=4, max_seq=524288, page_tokens=2048, ring=128):
    if min(pool_tokens, slots, max_seq, page_tokens, ring) <= 0 or page_tokens % 2:
        raise ValueError('invalid pool geometry')
    pages = (pool_tokens + page_tokens - 1) // page_tokens
    rounded_tokens = pages * page_tokens
    return dict(history=rounded_tokens * 3200,
                windows=slots * 43 * ring * 512 * 2,
                carry=slots * 3 * 2 * 512 * 4 * 2,
                page_table=slots * ((max_seq + page_tokens - 1) // page_tokens) * 8,
                positions=slots * 4)


def allocate_past(device, *, pool_tokens=2097152, slots=4, max_seq=524288):
    return SlotPool(slots, max_seq, pool_tokens=pool_tokens, device=device).configure_default(
        window_dtype=torch.bfloat16, ckv_dtype=torch.bfloat16, index_dtype=torch.bfloat16)


def past_tensors(past):
    yield past.pt.table
    yield past.pos_dev
    for window in past.windows.values():
        yield window.main_kv
    for source in past.sources.values():
        yield source.ckv_pool.data
        yield source.index_pool.data
        if source.kv_state is not None:
            yield source.kv_state
            yield source.score_state


def verify_past_storage(past):
    """Touch EVERY physical history element, check endpoints, restore empty pool.

    This is storage verification, not attention-kernel or long-context validation.
    """
    for source in past.sources.values():
        for tensor in (source.ckv_pool.data, source.index_pool.data):
            tensor.fill_(0.5)
            flat = tensor.reshape(-1)
            if flat[0].item() != 0.5 or flat[-1].item() != 0.5:
                raise RuntimeError('physical KV storage verification failed')
            tensor.zero_()
    actual = sum(t.numel() * t.element_size() for t in past_tensors(past))
    expected = sum(past_bytes(past.pool_tokens, past.n_slots, past.max_seq,
                              past.page_tokens, past.ring).values())
    if actual != expected:
        raise ValueError(f'Past storage mismatch: {actual} != {expected}')
    return actual
