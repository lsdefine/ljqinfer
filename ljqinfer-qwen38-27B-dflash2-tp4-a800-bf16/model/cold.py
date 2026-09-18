"""Engine-facing orchestration for the replaceable cold-prefix backend."""
from __future__ import annotations


def store_prefix(engine, cold_cache, input_ids, sequence_id: int = 0) -> int:
    """Store every new complete 1024-token block after prefill committed it."""
    ids = [int(x) for x in input_ids]
    sid = int(sequence_id)
    committed = int(engine.cache.lengths[sid].item())
    if committed < len(ids):
        raise RuntimeError("cold store requires the prompt to be committed first")
    match = cold_cache.begin(ids)
    return cold_cache.store(
        match, ids,
        lambda start, end: engine.cache.export_cold_block(sid, start, end))


def restore_prefix(engine, cold_cache, input_ids, sequence_id: int = 0) -> int:
    """Restore before any suffix/tail allocation and return the replay cursor."""
    sid = int(sequence_id)
    if int(engine.cache.lengths[sid].item()) != 0:
        raise RuntimeError("cold restore must precede tail reservation/prefill")
    match = cold_cache.begin([int(x) for x in input_ids])
    if not match.token_count:
        return 0
    return cold_cache.restore(
        match,
        lambda record, final: engine.cache.import_cold_block(
            sid, record, final=final))
