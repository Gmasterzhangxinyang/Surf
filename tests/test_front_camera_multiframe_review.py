import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from scripts import build_front_camera_multiframe_review as review


class FrontCameraFrameSelectionTest(unittest.TestCase):
    def test_relative_bearing_is_zero_for_slot_ahead(self) -> None:
        pose = np.array([0.0, 0.0, 0.0])

        bearing = review.relative_bearing_deg(pose, np.array([10.0, 0.0]))

        self.assertAlmostEqual(bearing, 0.0)

    def test_projection_metrics_accept_slot_ahead(self) -> None:
        polygon = np.array([[8.0, -1.0], [12.0, -1.0], [12.0, 1.0], [8.0, 1.0]])

        result = review.projection_metrics(
            polygon,
            np.array([0.0, 0.0, 0.0]),
            1.0,
            -0.8,
            1280,
            720,
        )

        self.assertGreater(result["projected_area_px"], 0.0)
        self.assertGreater(result["projection_score"], 0.0)

    def test_selection_rejects_post_anchor_side_and_small_projection(self) -> None:
        rows = [
            {"lidar_frame": 90, "bearing_deg": 10.0, "projection_score": 0.8, "projected_area_px": 2000.0},
            {"lidar_frame": 91, "bearing_deg": 50.0, "projection_score": 0.9, "projected_area_px": 2200.0},
            {"lidar_frame": 92, "bearing_deg": 5.0, "projection_score": 0.8, "projected_area_px": 20.0},
            {"lidar_frame": 101, "bearing_deg": 0.0, "projection_score": 1.0, "projected_area_px": 3000.0},
        ]

        selected = review.select_visible_assessments(rows, 100, 40.0, 0.5, 200.0, 5)

        self.assertEqual([row["lidar_frame"] for row in selected], [90])

    def test_selection_keeps_best_five_then_returns_chronological_order(self) -> None:
        rows = [
            {
                "lidar_frame": frame,
                "bearing_deg": 0.0,
                "projection_score": frame / 100.0,
                "projected_area_px": 1000.0,
            }
            for frame in range(10, 17)
        ]

        selected = review.select_visible_assessments(rows, 20, 40.0, 0.1, 100.0, 5)

        self.assertEqual([row["lidar_frame"] for row in selected], [12, 13, 14, 15, 16])

    def test_case_abstains_when_no_frame_is_visible(self) -> None:
        case = review.case_from_assessments("slot_0001", 100, [], max_frames=5)

        self.assertEqual(case["review_status"], "camera_unobservable")
        self.assertEqual(case["selected_frames"], [])
        self.assertEqual(case["automatic_label"], "camera_unobservable")

    def test_nearest_camera_uses_timestamp_not_lidar_frame_number(self) -> None:
        camera_ts = {300: 9.99, 301: 10.02, 302: 10.08}

        frame, timestamp, delta = review.nearest_camera(camera_ts, 10.0)

        self.assertEqual(frame, 300)
        self.assertAlmostEqual(timestamp, 9.99)
        self.assertAlmostEqual(delta, -0.01)

    def test_rejection_reasons_cover_all_visibility_gates(self) -> None:
        assessment = {
            "lidar_frame": 101,
            "bearing_deg": 45.0,
            "projection_score": 0.2,
            "projected_area_px": 20.0,
            "camera_image_exists": False,
        }

        reasons = review.rejection_reasons(assessment, 100, 40.0, 0.5, 200.0)

        self.assertEqual(
            reasons,
            ["after_anchor", "outside_safe_fov", "projection_score_too_low", "projected_area_too_small", "camera_image_missing"],
        )

    def test_assess_sampled_frame_uses_pose_projection_and_nearest_camera(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            points_path = root / "points.npz"
            np.savez(
                points_path,
                points_map_xyzi=np.array([[0.0, 0.0, -0.8, 1.0], [1.0, 0.0, -0.7, 1.0]]),
                ego_map_pose=np.array([0.0, 0.0, 0.0]),
            )
            image_dir = root / "image"
            image_dir.mkdir()
            Image.new("RGB", (1280, 720), "gray").save(image_dir / "left000300.png")
            frame_row = {
                "frame": "90",
                "map_x": "0.0",
                "map_y": "0.0",
                "map_yaw": "0.0",
                "lidar_timestamp": "10.0",
                "map_points_path": str(points_path),
            }
            slot = {
                "center_map": [10.0, 0.0],
                "polygon_map": [[8.0, -1.0], [12.0, -1.0], [12.0, 1.0], [8.0, 1.0]],
            }

            result = review.assess_sampled_frame(
                90,
                frame_row,
                slot,
                1.0,
                {300: 9.99, 301: 10.02},
                root,
                Path.cwd(),
                0.08,
            )

        self.assertEqual(result["camera_frame"], 300)
        self.assertTrue(result["camera_image_exists"])
        self.assertAlmostEqual(result["bearing_deg"], 0.0)
        self.assertGreater(result["projected_area_px"], 200.0)
    def test_draw_camera_slot_overlay_writes_annotated_image(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.png"
            output = root / "overlay.png"
            Image.new("RGB", (320, 180), "gray").save(source)

            review.draw_camera_slot_overlay(
                source,
                output,
                [[40.0, 140.0], [280.0, 140.0], [220.0, 80.0], [100.0, 80.0]],
                "slot_0001",
            )

            self.assertTrue(output.exists())
            self.assertNotEqual(Image.open(output).getpixel((40, 140)), (128, 128, 128))

    def test_report_contains_multiframe_images_labels_and_denominator_stats(self) -> None:
        html_text = review.render_report(
            [
                {
                    "slot_id": "slot_0001",
                    "anchor_frame": 100,
                    "review_status": "reviewable",
                    "selected_frames": [
                        {
                            "lidar_frame": 90,
                            "camera_frame": 300,
                            "bearing_deg": 10.0,
                            "distance_m": 8.0,
                            "projection_score": 0.8,
                            "projected_area_px": 1000.0,
                            "camera_lidar_dt_sec": 0.01,
                            "overlay_path": "assets/a.png",
                        }
                    ],
                    "slot_global_map_path": "assets/map.png",
                    "slot_accumulated_zoom_path": "assets/lidar.png",
                }
            ],
            {"candidate_count": 1},
        )

        self.assertIn("assets/a.png", html_text)
        self.assertIn("target_occupied", html_text)
        self.assertIn("camera_unobservable", html_text)
        self.assertIn("verifiable_count", html_text)
        self.assertIn("precision", html_text)
    def test_script_help_runs_when_invoked_by_path(self) -> None:
        completed = subprocess.run(
            [sys.executable, "scripts/build_front_camera_multiframe_review.py", "--help"],
            cwd=Path.cwd(),
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
    def test_pose_prefilter_expands_before_anchor_and_keeps_closest_front_frames(self) -> None:
        frame_rows = {
            frame: {
                "frame": str(frame),
                "map_x": str(frame / 10.0),
                "map_y": "0.0",
                "map_yaw": "0.0",
            }
            for frame in range(0, 106, 5)
        }

        selected = review.preselect_pose_visible_frame_ids(
            frame_rows,
            np.array([11.0, 0.0]),
            anchor_frame=100,
            lookback_frames=40,
            frame_stride=5,
            half_fov_deg=40.0,
            limit=3,
        )

        self.assertEqual(selected, [90, 95, 100])

if __name__ == "__main__":
    unittest.main()
