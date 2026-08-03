from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from parking_slot_part2.camera_calibration import CameraCalibration
from parking_slot_part2.camera_observability import CameraObservabilityInput
from scripts.visualize_camera_observability import (
    _project_map,
    render_camera_observability,
)


def _rectangle(cx: float, cy: float, hx: float, hy: float) -> list[list[float]]:
    return [
        [cx - hx, cy - hy],
        [cx + hx, cy - hy],
        [cx + hx, cy + hy],
        [cx - hx, cy + hy],
    ]


def _transform() -> list[list[float]]:
    return [
        [0.0, 0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0, 0.0],
        [0.0, -1.0, 0.0, 1.5],
        [0.0, 0.0, 0.0, 1.0],
    ]


def _calibration() -> dict[str, Any]:
    return {
        "schema_version": "camera-calibration/2.0",
        "camera_model": "fisheye",
        "image_width": 400,
        "image_height": 200,
        "K": [[100.0, 0.0, 200.0], [0.0, 100.0, 100.0], [0.0, 0.0, 1.0]],
        "D": [0.0, 0.0, 0.0, 0.0],
        "T_vehicle_camera": _transform(),
        "extrinsic_translation_unit": "m",
        "calibration_id": "synthetic-visualization-test-only",
        "calibration_time": "2026-01-01T00:00:00Z",
        "vehicle_frame": "vehicle",
        "camera_frame": "camera_front",
        "is_placeholder": False,
    }


def _scene(image_path: Path) -> dict[str, Any]:
    covariance = np.diag([1e-4] * 3 + [1e-6] * 3).reshape(-1).tolist()
    return {
        "target_slot_id": "S-003",
        "map_coordinate_frame": "map-test",
        "map_units_per_meter": 1.0,
        "camera_image_path": str(image_path),
        "camera_map_pose": {
            "schema_version": "camera-map-pose/1.0",
            "timestamp_ns": 1_000_000_000,
            "frame_id": "camera_front",
            "parent_frame": "map-test",
            "T_map_camera": _transform(),
            "position_xyz": [0.0, 0.0, 1.5],
            "orientation_xyzw": [-0.5, 0.5, -0.5, 0.5],
            "pose_covariance": covariance,
            "localization_status": "valid",
            "time_offset_ms": 0.0,
            "calibration_id": "synthetic-visualization-test-only",
            "translation_unit": "m",
        },
        "camera_calibration": _calibration(),
        "nominal_horizontal_fov_deg": 180.0,
        "slots": [
            {
                "slot_id": "S-003",
                "polygon_map": _rectangle(10.0, 0.0, 1.0, 1.0),
                "center_map": [10.0, 0.0],
                "state": "unknown",
            }
        ],
        "target_ground_z_m": 0.0,
        "static_obstacle_map": {
            "schema_version": "static-obstacle-map/1.0",
            "coordinate_frame": "map-test",
            "map_units_per_meter": 1.0,
            "static_obstacles": [
                {
                    "id": "side-column",
                    "type": "column",
                    "polygon_xy": _rectangle(5.0, 4.0, 0.2, 0.2),
                    "min_z": 0.0,
                    "max_z": 2.8,
                    "confidence": 1.0,
                    "source": "manual",
                }
            ],
            "mapped_regions": [
                {
                    "region_id": "local",
                    "polygon_xy": _rectangle(5.0, 0.0, 15.0, 10.0),
                    "static_obstacle_layer_complete": True,
                }
            ],
        },
        "lidar_projection_input": {
            "coordinate_frame": "map-test",
            "xy_unit": "map_unit",
            "z_unit": "m",
            "points_map_xyz": [[5.0, 0.0, 1.5], [8.0, 1.0, 1.0]],
        },
    }


class CameraObservabilityVisualizationTest(unittest.TestCase):
    def test_render_projects_lidar_rays_zones_and_reports_split_errors(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image_path = root / "camera.png"
            Image.new("RGB", (400, 200), (35, 35, 35)).save(image_path)
            scene = _scene(image_path)
            row = CameraObservabilityInput.from_mapping(scene)
            calibration = CameraCalibration.from_mapping(_calibration())
            reference_points = np.asarray(
                [
                    [2.5, 9.5, 1.5],
                    [10.0, 0.0, 1.5],
                    [2.5, -9.5, 1.5],
                ]
            )
            predicted, valid, _ = _project_map(reference_points, row, calibration)
            np.testing.assert_array_equal(valid, [True, True, True])
            offsets = np.asarray([[1.0, 0.0], [0.0, 0.0], [-2.0, 0.0]])
            scene["reprojection_references"] = [
                {
                    "id": identifier,
                    "point_map_xyz": point.tolist(),
                    "observed_pixel_uv": observed.tolist(),
                }
                for identifier, point, observed in zip(
                    ("left", "center", "right"),
                    reference_points,
                    predicted + offsets,
                )
            ]

            report = render_camera_observability(
                scene,
                output_dir=root / "rendered",
                stem="synthetic_test_only",
            )

            self.assertEqual(report["assessment"]["decision"], "use_camera")
            self.assertEqual(report["debug_ray_count"], 27)
            self.assertEqual(report["lidar_projected_count"], 2)
            split = report["reprojection_error_by_zone"]
            self.assertEqual(split["left_edge"]["count"], 1)
            self.assertEqual(split["center"]["count"], 1)
            self.assertEqual(split["right_edge"]["count"], 1)
            self.assertEqual(split["left_edge"]["mean_px"], 1.0)
            self.assertEqual(split["center"]["mean_px"], 0.0)
            self.assertEqual(split["right_edge"]["mean_px"], 2.0)
            output_image = Path(report["image_output"])
            output_report = Path(report["report_output"])
            self.assertTrue(output_image.is_file())
            self.assertTrue(output_report.is_file())
            persisted = json.loads(output_report.read_text(encoding="utf-8"))
            self.assertEqual(
                persisted["schema_version"],
                "camera-observability-visualization/1.0",
            )
            rendered = np.asarray(Image.open(output_image).convert("RGB"))
            colors = np.unique(rendered.reshape(-1, 3), axis=0)
            self.assertGreater(len(colors), 8)

    def test_missing_reprojection_references_produces_null_not_fake_error(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image_path = root / "camera.png"
            Image.new("RGB", (400, 200), (20, 20, 20)).save(image_path)

            report = render_camera_observability(
                _scene(image_path),
                output_dir=root / "rendered",
                stem="no_references",
            )

            for summary in report["reprojection_error_by_zone"].values():
                self.assertEqual(summary["count"], 0)
                self.assertIsNone(summary["mean_px"])
                self.assertIsNone(summary["rmse_px"])
                self.assertIsNone(summary["max_px"])

    def test_lidar_frame_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image_path = root / "camera.png"
            Image.new("RGB", (400, 200), (20, 20, 20)).save(image_path)
            scene = _scene(image_path)
            scene["lidar_projection_input"]["coordinate_frame"] = "wrong-map"

            with self.assertRaisesRegex(ValueError, "coordinate frames differ"):
                render_camera_observability(scene, output_dir=root / "rendered")


if __name__ == "__main__":
    unittest.main()
