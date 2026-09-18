import tempfile
import threading
import time
import unittest

from strategy.control_plane import FollowerControlServer, Rank0Coordinator, _send


class _PartialWriter:
    def __init__(self, limit=7):
        self.limit = int(limit)
        self.payload = bytearray()
        self.flushes = 0

    def write(self, data):
        count = min(self.limit, len(data))
        self.payload.extend(data[:count])
        return count

    def flush(self):
        self.flushes += 1


class ControlPlaneTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.calls = []
        self.server = FollowerControlServer(self.tmp.name, 1)

        def execute(ids, n_new):
            self.calls.append((tuple(ids), n_new))
            return {"ok": True}

        self.thread = threading.Thread(
            target=self.server.serve, args=(execute,), daemon=True)
        self.thread.start()
        self.coordinator = Rank0Coordinator(self.tmp.name, 2, timeout=2.0)

    def tearDown(self):
        # Stop the follower before deleting the directory it binds into.
        try:
            _, fp = self.coordinator._peer(1)
            _send(fp, {"op": "shutdown"})
            import json
            self.assertEqual(json.loads(fp.readline()), {"state": "done"})
        finally:
            self.coordinator.close()
            self.thread.join(timeout=3.0)
        self.assertFalse(self.thread.is_alive(), "follower did not stop")
        self.tmp.cleanup()

    def test_send_retries_partial_writes(self):
        writer = _PartialWriter(limit=7)
        value = {"op": "generate", "input_ids": list(range(1000)),
                 "max_new_tokens": 4}
        _send(writer, value)
        text = bytes(writer.payload).decode("utf-8")
        self.assertTrue(text.endswith("\n"))
        import json
        self.assertEqual(json.loads(text), value)
        self.assertEqual(writer.flushes, 1)

    def test_ready_go_done_and_reuse(self):
        self.assertEqual(
            self.coordinator.run([1, 2], 3, lambda: {"rank0": 1}),
            {"rank0": 1})
        self.assertEqual(
            self.coordinator.run([4], 1, lambda: {"rank0": 2}),
            {"rank0": 2})
        self.assertEqual(self.calls, [((1, 2), 3), ((4,), 1)])

    def test_large_command_is_not_truncated(self):
        ids = list(range(70000))
        self.assertEqual(
            self.coordinator.run(ids, 1, lambda: {"large": True}),
            {"large": True})
        self.assertEqual(self.calls[-1], (tuple(ids), 1))

    def test_local_failure_resets_protocol_session(self):
        def fail():
            raise ValueError("rank0 failed")

        with self.assertRaisesRegex(ValueError, "rank0 failed"):
            self.coordinator.run([7], 1, fail)
        # Let the follower observe the closed first connection and accept again.
        time.sleep(0.05)
        self.assertEqual(
            self.coordinator.run([8], 2, lambda: {"recovered": True}),
            {"recovered": True})
        self.assertEqual(self.calls, [((7,), 1), ((8,), 2)])


if __name__ == "__main__":
    unittest.main()
