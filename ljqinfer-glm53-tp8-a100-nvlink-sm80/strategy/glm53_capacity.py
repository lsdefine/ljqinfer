"""Startup execution check, not a replacement for bounded workspace ownership."""
import time
import torch


@torch.inference_mode()
def check_capacity(resident, max_total_tokens, seed_ids):
    """Exercise full leased contexts and existing B1..B4 graphs before ready.

    Uses ordinary prefill/cache/decode paths; never changes admission or leases.
    This detects startup capacity failures, but is not a mathematical proof
    that every runtime allocation pattern is bounded.
    """
    if not seed_ids:
        raise ValueError('capacity check requires nonempty seed tokens')
    generator = resident.generator
    captures = resident.capture_count
    records = []
    try:
        for batch in range(1, resident.max_batch + 1):
            # Derive lease geometry from the engine rather than duplicate it.
            page_size = resident.engine.kv.shape[2]
            pages = resident.engine.capacity // page_size // batch
            length = min(max_total_tokens - 16,
                         pages * page_size - 16 - resident.engine.q)
            if length < 1:
                raise ValueError('capacity too small for startup check')
            prompt = (list(seed_ids) * ((length + len(seed_ids) - 1) // len(seed_ids)))[:length]
            generator.cache.clear()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            # Different first tokens prevent later rows restoring earlier rows'
            # prefix, which would skip the cold prefill allocation peak.
            from model import weights as W
            prompts = [[(int(prompt[0]) + row) % W.VOCAB] + prompt[1:]
                       for row in range(batch)]
            metrics = []
            results = resident.generate(prompts, [16] * batch,
                                        eos_token_ids=[], use_graph=True,
                                        on_prefill=lambda row, item: metrics.append(dict(item)))
            torch.cuda.synchronize()
            if len(metrics) != batch or any(m['cache_hit_tokens'] != 0 for m in metrics):
                raise RuntimeError('startup capacity check unexpectedly reused a prefix')
            if len(results) != batch or any(len(r.token_ids) != 16 for r in results):
                raise RuntimeError('startup capacity generation incomplete')
            if resident.capture_count != captures or resident.busy or resident.used_pages or resident.row_ids:
                raise RuntimeError('startup capacity check changed resident lifecycle')
            record = dict(batch=batch, input_tokens_per_request=length,
                          seconds=time.perf_counter()-started,
                          peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                          reserved_bytes=torch.cuda.memory_reserved())
            records.append(record)
            print(f'[glm53-capacity] rank={torch.distributed.get_rank()} {record}', flush=True)
    finally:
        generator.reset()
    return records
