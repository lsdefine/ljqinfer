"""CPU-only cold-cache ownership, transaction and bounded-pool regression."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from strategy.cold_kv import ColdCache, Field
from strategy.host_arena import HostArena

def main():
    key = ('kv',)
    arena = HostArena(2048, segment_bytes=1024, pin_memory=False)
    cache = ColdCache([Field(key, (None, 4), torch.float32)], 2048,
                      namespace='test', shm=arena)
    def put(tokens, parent=None, rows=16):
        own = parent is None
        parent = parent or cache.lookup([], namespace='test')
        try:
            with cache.prepare(parent, tokens) as tx:
                tx.write(key, torch.arange(rows*4).float().view(rows,4))
                return tx.commit()
        finally:
            if own: parent.close()
    put([1,2,3])
    lease = cache.lookup([1,2], namespace='test')
    assert lease.token_count == 2
    assert cache.spans(lease)[0][1:] == (0,2)
    put([9], lease)
    branch = cache.lookup([1,2,9], namespace='test')
    assert [x[1:] for x in cache.spans(branch)] == [(0,2),(2,3)]
    before = arena.used_bytes
    with cache.prepare(branch, [10]) as tx:
        tx.write(key, torch.ones(3,4))
        tx.write(key, torch.empty(0,4))
        tx.write(key, torch.ones(2,4))
    assert arena.used_bytes == before and cache.active is None
    branch.close(); lease.close()
    for i in range(20): put([100+i])
    assert arena.used_bytes <= arena.capacity
    assert cache.used_bytes <= cache.budget_bytes
    leases = [cache.pin_endpoint(i, namespace='test') for i in list(cache.entries)]
    before = arena.used_bytes
    try:
        put([999])
        raise AssertionError('leased pool must reject allocation')
    except MemoryError:
        pass
    assert cache.active is None and arena.used_bytes == before
    for lease in leases: lease.close()
    cache.clear()
    assert not cache.entries and cache.used_bytes == arena.used_bytes == 0
    print('PASS prefix branch abort rewrite zero physical_eviction pinned_capacity clear')

if __name__ == '__main__': main()
