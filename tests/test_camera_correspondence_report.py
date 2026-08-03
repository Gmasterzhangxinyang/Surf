import unittest

from scripts import build_slot_aligned_camera_correspondence_report as report


class CameraCorrespondenceReportTest(unittest.TestCase):
    def test_nearest_timestamp_prefers_smallest_absolute_delta(self) -> None:
        camera_ts = {10: 1.00, 11: 1.10, 12: 1.20}

        frame, timestamp, delta = report.nearest_timestamp(camera_ts, 1.16)

        self.assertEqual(frame, 12)
        self.assertAlmostEqual(timestamp, 1.20)
        self.assertAlmostEqual(delta, 0.04)

    def test_previous_window_start_uses_earliest_of_previous_camera_frames(self) -> None:
        camera_ts = {frame: float(frame) for frame in range(100, 140)}

        selected = report.previous_window_start_timestamp(camera_ts, target=124.6, lookback_frames=28)

        self.assertEqual(selected["base_previous_frame"], 124)
        self.assertEqual(selected["selected_camera_frame"], 100)
        self.assertEqual(selected["selected_window_index"], 0)

    def test_previous_window_start_selects_frame_27_before_base_when_available(self) -> None:
        camera_ts = {frame: float(frame) for frame in range(1, 200)}

        selected = report.previous_window_start_timestamp(camera_ts, target=124.6, lookback_frames=28)

        self.assertEqual(selected["base_previous_frame"], 124)
        self.assertEqual(selected["selected_camera_frame"], 97)
        self.assertEqual(selected["selected_window_index"], 96)
        self.assertAlmostEqual(selected["selected_camera_delta_sec"], -27.6)


if __name__ == "__main__":
    unittest.main()
