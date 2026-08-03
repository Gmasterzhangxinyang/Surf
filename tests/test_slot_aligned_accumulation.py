import unittest
from types import SimpleNamespace

import numpy as np

from scripts import multiframe_pointcloud_accumulation_probe as accumulation
from scripts import slot_aligned_multiframe_accumulation_probe as slot_aligned


class SlotAlignedAccumulationTest(unittest.TestCase):
    def test_window_stride_samples_symmetric_offsets_and_includes_anchor(self) -> None:
        rows = [{"frame": str(frame)} for frame in range(100, 221)]
        args = SimpleNamespace(window_before=52, window_after=52, window_frame_stride=4)

        indices = accumulation.sample_window_indices(anchor_index=60, frame_rows=rows, args=args)
        frames = [int(rows[index]["frame"]) for index in indices]

        self.assertEqual(frames[0], 108)
        self.assertEqual(frames[-1], 212)
        self.assertEqual(frames[13], 160)
        self.assertEqual(len(frames), 27)
        self.assertEqual([b - a for a, b in zip(frames, frames[1:])], [4] * 26)

    def test_window_stride_clamps_edges_without_duplicates(self) -> None:
        rows = [{"frame": str(frame)} for frame in range(6, 46)]
        args = SimpleNamespace(window_before=52, window_after=52, window_frame_stride=4)

        indices = accumulation.sample_window_indices(anchor_index=3, frame_rows=rows, args=args)
        frames = [int(rows[index]["frame"]) for index in indices]

        self.assertEqual(frames[0], 9)
        self.assertIn(9, frames)
        self.assertEqual(len(frames), len(set(frames)))
        self.assertTrue(all(6 <= frame <= 45 for frame in frames))

    def test_pose_anchor_selects_nearest_slot_center(self) -> None:
        rows = [
            {"frame": "10", "map_x": "0.0", "map_y": "0.0", "map_yaw": "0.0"},
            {"frame": "14", "map_x": "2.0", "map_y": "0.0", "map_yaw": "0.0"},
            {"frame": "18", "map_x": "3.9", "map_y": "0.0", "map_yaw": "0.0"},
            {"frame": "22", "map_x": "7.0", "map_y": "0.0", "map_yaw": "0.0"},
        ]
        slot = {"slot_id": "slot_0001", "center_np": np.asarray([4.0, 0.0], dtype=np.float64)}

        anchor = slot_aligned.select_pose_aligned_anchor(rows, slot)

        self.assertEqual(anchor["anchor_index"], 2)
        self.assertEqual(anchor["anchor_frame"], 18)
        self.assertAlmostEqual(anchor["anchor_distance_to_slot_m"], 0.1, places=6)


if __name__ == "__main__":
    unittest.main()
