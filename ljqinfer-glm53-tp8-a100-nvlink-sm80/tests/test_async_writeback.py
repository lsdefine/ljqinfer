"""Small lifecycle tests; --cuda additionally checks real DMA/cache bytes."""
import sys
import threading
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from strategy.async_writeback import AsyncWriteback


def test_serial_and_drain():
    writer = AsyncWriteback(2)
    gate = threading.Event()
    started = threading.Event()
    values = []
    def first():
        started.set()
        assert gate.wait(5)
        values.append(1)
    try:
        writer.submit(first)
        assert started.wait(5)
        writer.submit(lambda: values.append(2))
        assert values == []
        gate.set()
        writer.drain()
        assert values == [1, 2]
    finally:
        gate.set()
        writer.close()


def test_failure_drains_all():
    writer = AsyncWriteback(2)
    gate = threading.Event()
    finished = threading.Event()
    def fail():
        assert gate.wait(5)
        raise ValueError('injected')
    writer.submit(fail)
    writer.submit(finished.set)
    gate.set()
    try:
        writer.drain()
    except ValueError as exc:
        assert str(exc) == 'injected'
    else:
        raise AssertionError('worker error lost')
    assert finished.is_set() and not writer.pending
    writer.close()


def test_cuda():
    import torch
    from strategy.cold_kv import ColdCache, Field
    from strategy.host_arena import HostArena
    from model.glm53_cache import PrefixState
    from model.glm53_cold_transfer import ColdTransfer
    device = torch.device('cuda:0')
    source = torch.arange(128 * 16, device=device, dtype=torch.float32).reshape(128, 16)
    state = PrefixState.__new__(PrefixState)
    state.engine = SimpleNamespace(device=device, length=64, pending=None,
                                   mtp_kv=SimpleNamespace(lengths=[64]))
    state.namespace = 'async-test'
    state.resident = []
    state.fields = {'kv': source}
    state.transfer = ColdTransfer(state.fields, device)
    state.arena = HostArena(128 * 16 * 4 * 4)
    state.cold = ColdCache([Field('kv', (None, 16), torch.float32)],
                           128 * 16 * 4 * 4, namespace=state.namespace,
                           pin_memory=True, shm=state.arena)
    state.copy_stream = torch.cuda.Stream(device=device)
    state.writer = AsyncWriteback(2)
    tokens = list(range(128))
    expected = source.cpu().clone()
    gate = threading.Event()
    entered = threading.Event()
    original = state._store
    def delayed(sequence):
        entered.set()
        assert gate.wait(10)
        original(sequence)
    state._store = delayed
    try:
        state.publish(tokens[:64])
        assert entered.wait(5)
        with state.cold.lookup(tokens, namespace=state.namespace) as lease:
            assert lease.token_count == 0, 'uncommitted prefix exposed'
        # Compute can progress while publication is blocked in the worker.
        assert (source[64:] + 1).sum().item() > 0
        state.engine.length = state.engine.mtp_kv.lengths[0] = 128
        state.publish(tokens)
        gate.set()
        state.drain()
        with state.cold.lookup(tokens, namespace=state.namespace) as lease:
            assert lease.token_count == 128
            for entry, start, end in state.cold.spans(lease):
                assert torch.equal(state.cold.storage['kv'][entry][:end-start], expected[start:end])
        state.invalidate()
        source.zero_()
        hit, kind = state.restore(tokens)
        torch.cuda.current_stream().synchronize()
        assert hit == 127 and kind == 'cold'
        assert torch.equal(source[:127].cpu(), expected[:127])
        state.clear()
        assert not state.cold.entries and state.arena.used_bytes == 0
    finally:
        gate.set()
        state.close()


if __name__ == '__main__':
    test_serial_and_drain()
    test_failure_drains_all()
    if '--cuda' in sys.argv:
        test_cuda()
    print('PASS: async queue, failure drain' + (', CUDA publication/restore/reuse' if '--cuda' in sys.argv else ''))
