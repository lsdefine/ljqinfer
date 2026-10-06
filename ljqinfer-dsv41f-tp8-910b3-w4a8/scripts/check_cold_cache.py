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


def check_tp():
    import os
    import torch.distributed as dist
    from types import SimpleNamespace
    from model.cold import fields_for, restore_prefix_tp, RestoreWorkspace
    world = int(os.environ.get('WORLD_SIZE', '1'))
    rank = int(os.environ.get('RANK', '0'))
    if world > 1:
        dist.init_process_group('gloo')

    class Source:
        def __init__(self, ratio):
            self.ratio = ratio
            self.ckv_pool = SimpleNamespace(data=torch.zeros(400, 3))
            self.index_pool = SimpleNamespace(data=torch.zeros(400, 2))
        def import_cold(self, slot, start, end, blob):
            for name, paged in [('main_ckv', self.ckv_pool), ('index_k', self.index_pool)]:
                paged.data[start//self.ratio:end//self.ratio].copy_(blob[name])

    class Pool:
        def __init__(self):
            self.sources = {0: Source(2), 1: Source(4)}
            self.windows, self.n_slots, self.free_slots, self.pos = {0: True}, 1, set(), [0]
        def ensure(self, slot, hit):
            assert slot == 0 and 0 < hit < 800
        def mark_cold(self, slot, hit):
            self.pos[slot] = hit

    def values(start, end, ratio, dim):
        return (torch.arange(start//ratio, end//ratio)[:, None] * 10
                + torch.arange(dim)[None, :] + 1).float()

    for count, clipped in [(0, False), (1, False), (64, False), (65, False),
                            (129, False), (129, True)]:
        pool = Pool()
        cache = lease = None
        hit = count * 5 - int(clipped)
        if rank == 0:
            cache = ColdCache(fields_for(pool), 1<<20, namespace='tp-test')
            lease = cache.lookup([], namespace=cache.namespace)
            for i in range(count):
                start, end = i*5, (i+1)*5
                with cache.prepare(lease, range(start, end)) as tx:
                    for (_, layer, name), field in cache.fields.items():
                        tx.write(field.key, values(start, end, pool.sources[layer].ratio,
                                                   field.shape[-1]))
                    entry = tx.commit()
                lease.close()
                lease = cache.pin_endpoint(entry, namespace=cache.namespace)
            lease.close()
            lease = cache.lookup(range(hit), namespace=cache.namespace)
            assert len(cache.spans(lease)) == count
        got = restore_prefix_tp(cache, pool, 0, lease, rank=rank, world=world,
                                device='cpu', workspace=RestoreWorkspace(pool, 'cpu', tokens=4))
        assert got == pool.pos[0] == hit
        for source in pool.sources.values():
            for paged in [source.ckv_pool, source.index_pool]:
                rows = hit//source.ratio
                assert torch.equal(paged.data[:rows],
                                   values(0, hit, source.ratio, paged.data.shape[-1]))
                assert not paged.data[rows:].count_nonzero()
        if rank == 0:
            lease.close()
            cache.clear()
            print(f'PASS TP{world} spans={count} clipped={clipped}', flush=True)
    if world > 1:
        dist.destroy_process_group()

if __name__ == '__main__':
    if '--tp' in sys.argv: check_tp()
    else: main()
