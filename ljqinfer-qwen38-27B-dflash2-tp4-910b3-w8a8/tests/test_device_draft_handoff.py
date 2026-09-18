import unittest

import torch

from model.decode_graph import DecodeGraphRunner


class DeviceDraftHandoffTest(unittest.TestCase):
    def _runner(self):
        runner = object.__new__(DecodeGraphRunner)
        runner.batch_size = 2
        runner.query_width = 8
        runner.max_prefix = 32
        runner.device = "cpu"
        runner._pending = False
        runner.ids = torch.full((2, 8), -1, dtype=torch.int64)
        runner.positions = torch.full((2, 8), -1, dtype=torch.int64)
        runner._positions_host = torch.full((2, 8), -1, dtype=torch.int64)
        return runner

    def test_resident_ids_are_copied_and_inactive_row_is_frozen(self):
        runner = self._runner()
        anchors = torch.tensor([10, 20], dtype=torch.int64)
        paths = torch.tensor([
            [11, 12, 13, 14, 15, 16, 17],
            [21, 22, 23, 24, 25, 26, 27],
        ], dtype=torch.int64)
        positions = [list(range(4, 12)), list(range(9, 17))]

        runner.prepare_draft(
            [10, 20], anchors, paths, positions,
            active_rows=[True, False])

        self.assertEqual(runner.ids[0].tolist(),
                         [10, 11, 12, 13, 14, 15, 16, 17])
        self.assertEqual(runner.ids[1].tolist(), [20] * 8)
        self.assertEqual(runner.positions.tolist(), positions)
        self.assertEqual(runner._positions_host.tolist(), positions)

    def test_validates_pending_state_and_resident_shapes(self):
        runner = self._runner()
        anchors = torch.tensor([10, 20], dtype=torch.int64)
        paths = torch.arange(14, dtype=torch.int64).reshape(2, 7)
        positions = [list(range(8)), list(range(8, 16))]

        runner._pending = True
        with self.assertRaisesRegex(RuntimeError, "previous verify is pending"):
            runner.prepare_draft([10, 20], anchors, paths, positions)
        runner._pending = False

        with self.assertRaisesRegex(ValueError, "one token per batch row"):
            runner.prepare_draft(
                [10, 20], torch.tensor([10]), paths, positions)
        with self.assertRaisesRegex(ValueError, r"B\*\(Q-1\) draft tokens"):
            runner.prepare_draft(
                [10, 20], anchors, torch.arange(13), positions)
        with self.assertRaisesRegex(ValueError, "active_rows must match"):
            runner.prepare_draft(
                [10, 20], anchors, paths, positions, active_rows=[True])


if __name__ == "__main__":
    unittest.main()
