import math
import unittest

import numpy as np

from parking_pose_correction.dataset import lidar_to_map_xyzi, synchronize_frame_rows
from parking_pose_correction.registration import evaluate_alignment, estimate_local_correction
from parking_pose_correction.se2 import (
    apply_pose_corrections,
    interpolate_corrections,
    select_keyframes,
    stabilize_dense_corrections,
    smooth_keyframe_corrections,
)
from parking_pose_correction.timestamps import nearest_timestamp_match


class PoseDriftCorrectionTest(unittest.TestCase):
    def test_lidar_projection_uses_corrected_pose(self):
        raw = np.array([[1.0, 0.0, 0.5, 7.0]], dtype=np.float32)
        pose = np.array([5.0, 4.0, math.pi / 2.0])

        projected = lidar_to_map_xyzi(raw, pose, map_scale=2.0)

        np.testing.assert_allclose(projected[0], [5.0, 6.0, 0.5, 7.0], atol=1e-6)

    def test_synchronize_frame_rows_records_real_camera_frame(self):
        rows = [
            {
                "frame": "100",
                "lidar_timestamp": "10.051",
                "image_path": "/wrong/left000100.png",
            }
        ]
        camera_frames = np.array([298, 299, 300], dtype=np.int64)
        camera_timestamps = np.array([10.00, 10.04, 10.08], dtype=np.float64)

        synchronized = synchronize_frame_rows(
            rows,
            camera_frames,
            camera_timestamps,
            image_dir="/dataset/image",
            max_delta_sec=0.04,
        )

        self.assertEqual(synchronized[0]["camera_frame"], 299)
        self.assertEqual(synchronized[0]["camera_image_path"], "/dataset/image/left000299.png")
        self.assertAlmostEqual(synchronized[0]["camera_lidar_dt_sec"], -0.011)
        self.assertEqual(synchronized[0]["camera_match_valid"], 1)

    def test_nearest_timestamp_match_uses_timestamp_not_frame_number(self):
        frames = np.array([10, 11, 12, 13], dtype=np.int64)
        timestamps = np.array([1.00, 1.03, 1.06, 1.09], dtype=np.float64)

        match = nearest_timestamp_match(frames, timestamps, 1.055)

        self.assertEqual(match.frame, 12)
        self.assertAlmostEqual(match.delta_sec, 0.005)

    def test_select_keyframes_uses_distance_turning_and_final_frame(self):
        frames = np.arange(6, dtype=np.int64)
        poses = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.8, 0.0, 0.0],
                [1.7, 0.0, 0.0],
                [2.1, 0.0, math.radians(12.0)],
                [3.0, 0.0, math.radians(12.0)],
                [3.4, 0.0, math.radians(12.0)],
            ],
            dtype=np.float64,
        )

        selected = select_keyframes(frames, poses, distance_m=1.5, yaw_deg=10.0)

        self.assertEqual(selected.tolist(), [0, 2, 3, 5])

    def test_interpolate_and_apply_pose_corrections(self):
        all_frames = np.array([0, 1, 2], dtype=np.int64)
        keyframes = np.array([0, 2], dtype=np.int64)
        corrections = np.array(
            [[0.0, 0.0, 0.0], [2.0, -1.0, math.radians(10.0)]],
            dtype=np.float64,
        )
        poses = np.array(
            [[5.0, 4.0, 0.0], [6.0, 4.0, 0.0], [7.0, 4.0, 0.0]],
            dtype=np.float64,
        )

        dense = interpolate_corrections(all_frames, keyframes, corrections)
        corrected = apply_pose_corrections(poses, dense)

        np.testing.assert_allclose(dense[1], [1.0, -0.5, math.radians(5.0)], atol=1e-8)
        np.testing.assert_allclose(corrected[1], [7.0, 3.5, math.radians(5.0)], atol=1e-8)

    def test_local_registration_recovers_small_rigid_correction(self):
        horizontal = np.column_stack([np.linspace(-4.0, 4.0, 120), np.zeros(120)])
        vertical = np.column_stack([np.zeros(80), np.linspace(0.0, 3.0, 80)])
        target = np.vstack([horizontal, vertical])
        yaw = math.radians(2.0)
        translation = np.array([0.25, -0.15])
        c, s = math.cos(-yaw), math.sin(-yaw)
        inverse_rotation = np.array([[c, -s], [s, c]])
        scan = (target - translation) @ inverse_rotation.T

        result = estimate_local_correction(
            scan,
            target,
            map_units_per_meter=1.0,
            max_translation_m=1.0,
            max_yaw_deg=5.0,
            max_correspondence_m=0.7,
            trim_fraction=0.8,
        )

        self.assertTrue(result.converged)
        np.testing.assert_allclose([result.dx_m, result.dy_m], translation, atol=0.05)
        self.assertAlmostEqual(result.dyaw_rad, yaw, delta=math.radians(0.5))
        self.assertGreater(result.improvement_ratio, 0.2)

    def test_local_registration_respects_radial_translation_limit(self):
        target = np.column_stack([np.linspace(-2.0, 2.0, 80), np.linspace(-1.0, 1.0, 80) ** 2])
        scan = target - np.array([1.0, 1.0])

        result = estimate_local_correction(
            scan,
            target,
            map_units_per_meter=1.0,
            max_translation_m=0.5,
            max_yaw_deg=5.0,
            max_correspondence_m=2.0,
            trim_fraction=0.8,
        )

        self.assertLessEqual(math.hypot(result.dx_m, result.dy_m), 0.5 + 1e-8)

    def test_alignment_metrics_measure_applied_correction(self):
        target = np.column_stack([np.linspace(-2.0, 2.0, 80), np.sin(np.linspace(-2.0, 2.0, 80))])
        offset = np.array([0.3, -0.2])
        scan = target - offset

        before = evaluate_alignment(scan, target, map_units_per_meter=1.0)
        after = evaluate_alignment(scan + offset, target, map_units_per_meter=1.0)

        self.assertGreater(before.residual_m, after.residual_m)
        self.assertLess(after.residual_m, 1e-6)

    def test_correction_graph_interpolates_between_reliable_constraints(self):
        observed = np.array(
            [[0.0, 0.0, 0.0], [np.nan, np.nan, np.nan], [2.0, -1.0, math.radians(4.0)]],
            dtype=np.float64,
        )
        accepted = np.array([True, False, True])
        confidence = np.array([1.0, 0.0, 1.0])

        smoothed = smooth_keyframe_corrections(observed, accepted, confidence)

        np.testing.assert_allclose(smoothed[0], observed[0], atol=0.05)
        np.testing.assert_allclose(smoothed[1], [1.0, -0.5, math.radians(2.0)], atol=0.08)
        np.testing.assert_allclose(smoothed[2], observed[2], atol=0.05)

    def test_correction_graph_does_not_extrapolate_beyond_observations(self):
        observed = np.full((6, 3), np.nan, dtype=np.float64)
        observed[2] = [0.6, -0.2, math.radians(3.0)]
        observed[4] = [0.9, -0.3, math.radians(4.0)]
        accepted = np.array([False, False, True, False, True, False])
        confidence = accepted.astype(np.float64)

        smoothed = smooth_keyframe_corrections(observed, accepted, confidence)

        self.assertLessEqual(float(smoothed[:, 0].max()), 0.9 + 1e-9)
        self.assertGreaterEqual(float(smoothed[:, 0].min()), -1e-9)
        self.assertLessEqual(float(np.abs(smoothed[:, 2]).max()), math.radians(4.0) + 1e-9)

    def test_correction_graph_rejects_isolated_registration_outlier(self):
        positions = np.linspace(0.0, 40.0, 21)
        observed = np.zeros((21, 3), dtype=np.float64)
        observed[:, 0] = np.linspace(0.0, 1.0, 21)
        observed[:, 1] = np.linspace(0.0, -0.4, 21)
        observed[:, 2] = np.linspace(0.0, math.radians(2.0), 21)
        observed[10] = [5.0, -5.0, math.radians(20.0)]
        accepted = np.ones(21, dtype=bool)
        confidence = np.ones(21, dtype=np.float64)

        smoothed = smooth_keyframe_corrections(
            observed,
            accepted,
            confidence,
            positions=positions,
        )

        self.assertLess(abs(float(smoothed[10, 0]) - 0.5), 0.25)
        self.assertLess(float(np.max(np.linalg.norm(np.diff(smoothed[:, :2], axis=0), axis=1))), 0.15)
        self.assertLess(float(np.max(np.abs(np.degrees(np.diff(smoothed[:, 2]))))), 1.0)

    def test_dense_stabilizer_limits_per_frame_correction_change(self):
        corrections = np.zeros((201, 3), dtype=np.float64)
        corrections[80:121, 0] = 1.0
        corrections[80:121, 1] = -0.6
        corrections[80:121, 2] = math.radians(4.0)

        stable = stabilize_dense_corrections(
            corrections,
            map_units_per_meter=1.0,
            window_frames=31,
            max_translation_step_m=0.02,
            max_yaw_step_deg=0.05,
        )

        translation_steps = np.linalg.norm(np.diff(stable[:, :2], axis=0), axis=1)
        yaw_steps = np.abs(np.degrees(np.diff(stable[:, 2])))
        self.assertLessEqual(float(translation_steps.max()), 0.02 + 1e-9)
        self.assertLessEqual(float(yaw_steps.max()), 0.05 + 1e-9)


if __name__ == "__main__":
    unittest.main()
