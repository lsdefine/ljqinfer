"""CPU protocol tests: real sockets/threads, not GPU release acceptance."""
import tempfile
import threading
from strategy.control_plane import FollowerControlServer, Rank0Coordinator


def test_shutdown_broadcasts_before_collective_and_waits_for_release():
    with tempfile.TemporaryDirectory() as directory:
        barrier = threading.Barrier(4, timeout=5)
        closed = []
        errors = []
        threads = []
        def release(rank):
            barrier.wait()
            closed.append(rank)
        def serve(rank):
            try:
                FollowerControlServer(directory, rank).serve(
                    lambda ids, n: None, shutdown=lambda: release(rank))
            except BaseException as exc:
                errors.append(exc)
        for rank in range(1, 4):
            thread = threading.Thread(target=serve, args=(rank,), daemon=True)
            thread.start()
            threads.append(thread)
        coordinator = Rank0Coordinator(directory, 4, timeout=7)
        try:
            coordinator.shutdown(lambda: release(0))
            assert sorted(closed) == [0, 1, 2, 3]
            assert not coordinator._peers
        finally:
            coordinator.close()
            for thread in threads:
                thread.join(timeout=8)
        assert all(not thread.is_alive() for thread in threads)
        assert not errors


def test_shutdown_failure_is_not_acknowledged_as_success():
    import pytest
    with tempfile.TemporaryDirectory() as directory:
        errors = []
        def fail():
            raise RuntimeError('injected release failure')
        def serve():
            try:
                FollowerControlServer(directory, 1).serve(
                    lambda ids, n: None, shutdown=fail)
            except RuntimeError as exc:
                errors.append(str(exc))
        thread = threading.Thread(target=serve, daemon=True)
        thread.start()
        coordinator = Rank0Coordinator(directory, 2, timeout=3)
        try:
            with pytest.raises(RuntimeError, match='shutdown failed'):
                coordinator.shutdown(lambda: None)
            assert not coordinator._peers
        finally:
            coordinator.close()
            thread.join(timeout=4)
        assert not thread.is_alive()
        assert errors == ['injected release failure']
