"""Run with torchrun --standalone --nproc-per-node=8 tests/test_cold_single_copy.py.
No weights needed. Tests the production PrefixState and NCCL path end-to-end.
"""
import os
import sys
import time
import json
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import torch.distributed as dist
from model.glm53_cache import PrefixState
from model.glm53_mtp_pool import MTPKVPool


def main():
    rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(rank)
    torch.set_num_threads(2)
    dist.init_process_group('nccl', timeout=timedelta(seconds=180))
    device = torch.device('cuda', rank)
    n = int(os.environ.get('COLD_TEST_TOKENS', '512'))
    capacity = n + 64
    pool = MTPKVPool(max_tokens=capacity, max_sequence_tokens=capacity, device=device)
    e = SimpleNamespace(device=device, capacity=capacity,
        kv=torch.empty((78, capacity, 576), dtype=torch.bfloat16, device=device),
        index=torch.empty((21, capacity, 128), dtype=torch.bfloat16, device=device),
        mtp_kv=pool, length=0, pending=None)
    pool.reserve(0, capacity)
    from unittest.mock import patch
    with patch('model.glm53_cache.COLD_CACHE_BYTES', 5 * capacity * 119808):
        state = PrefixState(e)
    tokens = list(range(n))

    def fill():
        for i, (key, data) in enumerate(state.fields.items()):
            # Public fields identical, private fields explicitly rank-distinct.
            v = i + (rank * 16 if key[0] in ('k', 'v') else 0)
            data.fill_(v)
            # Position-dependent values detect incorrect span assembly.
            data.view(capacity, -1)[:, 0].copy_(
                torch.arange(capacity, device=device).remainder(127))

    def verify(count):
        for i, (key, data) in enumerate(state.fields.items()):
            v = i + (rank * 16 if key[0] in ('k', 'v') else 0)
            view = data[:count].reshape(count, -1)
            assert torch.all(view[:, 1:] == v).item(), (rank, key)
            assert torch.equal(view[:, 0], torch.arange(count, device=device).remainder(127).to(data.dtype)), (rank, key, 'positions')

    try:
        # Sum of stored schemas is exactly one public copy + 8 draft shards.
        row_bytes = sum(state.fields[k][0].numel() * 2 for k in state.cold.fields)
        total = torch.tensor(row_bytes, device=device)
        dist.all_reduce(total)
        assert total.item() == 119808
        for key in state.transfer.owners:
            owners = torch.tensor(int(key in state.cold.fields), device=device)
            dist.all_reduce(owners)
            assert owners.item() == 1
        fill()
        # Different local commit boundaries must not alter collective order.
        cuts = sorted(set([n // (rank + 2), n // 2, n]))
        for count in cuts:
            e.length = pool.lengths[0] = count
            state.publish(tokens[:count])
        state.drain()
        assert state.cold.used_bytes == row_bytes * n
        state.invalidate()
        for data in state.fields.values():
            data.fill_(-1)
        dist.barrier(); torch.cuda.synchronize()
        start = time.perf_counter()
        hit, source = state.restore(tokens + [n])
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        assert (hit, source) == (n, 'cold')
        verify(n)
        times = [None] * dist.get_world_size()
        dist.all_gather_object(times, elapsed)
        if rank == 0:
            print(json.dumps({'test': 'single_copy_restore', 'tokens': n,
                'unique_bytes_per_token': total.item(), 'max_seconds': max(times),
                'seconds': times}), flush=True)
        # Rank 0 alone evicts everything. All ranks must safely report a miss.
        state.invalidate()
        if rank == 0:
            state.cold.clear()
        hit, source = state.restore(tokens + [n])
        assert (hit, source) == (0, 'miss'), (rank, hit, source)
        fill()
        e.length = pool.lengths[0] = n
        state.publish(tokens); state.drain()
        state.invalidate()
        hit, source = state.restore(tokens + [n])
        torch.cuda.synchronize()
        assert hit == n
        verify(n)
        # Strict prefix/branch restores cut different backing spans safely.
        state.invalidate()
        hit, source = state.restore(tokens[:n // 3] + [-2])
        torch.cuda.synchronize()
        assert hit == n // 3 and source == 'cold'
        verify(hit)
        state.clear()
        assert state.arena.used_bytes == 0
        # Repeated shared-prefix forks must evict independently on every rank.
        # Snapshot the actual GPU payload, not a separately recomputed model.
        branches = 24
        cut = n // 4
        for branch in range(branches):
            fork = tokens[:cut] + [10000 + branch] * (n - cut)
            fill()
            for data in state.fields.values():
                data[cut:n].add_(branch)
            expected = {key: data[:n].cpu().clone() for key, data in state.fields.items()}
            e.length = pool.lengths[0] = n
            state.publish(fork); state.drain()
            state.invalidate()
            for data in state.fields.values():
                data.fill_(-1)
            hit, source = state.restore(fork + [-3])
            torch.cuda.synchronize()
            assert (hit, source) == (n, 'cold'), (rank, branch, hit, source)
            for key, data in state.fields.items():
                assert torch.equal(data[:n].cpu(), expected[key]), (rank, branch, key)
            with state.cold.lookup(tokens[:cut] + [-4], namespace=state.namespace) as lease:
                assert lease.token_count == cut
                assert len(lease.ids) == 1, (rank, branch, lease.ids)
            assert state.arena.used_bytes <= state.arena.capacity
        state.clear()
        assert state.arena.used_bytes == 0
        dist.barrier()
        result_dir = os.environ.get('COLD_TEST_RESULT_DIR')
        if result_dir:
            Path(result_dir).mkdir(parents=True, exist_ok=True)
            Path(result_dir, f'rank{rank}.json').write_text(json.dumps(
                dict(rank=rank, tokens=n, branches=branches, passed=True)))
        if rank == 0:
            print('PASS: unique public copy, rank-private draft, split spans, eviction divergence, recovery, clipped lease, branch LRU', flush=True)
    finally:
        state.close()
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
