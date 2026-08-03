from __future__ import annotations

import math
import unittest
from typing import Any

import numpy as np

from parking_slot_part2.camera_calibration import CameraCalibration
from parking_slot_part2.camera_observability import (
    assess_camera_observability,
    camera_observability_tool,
)
from parking_slot_part2.camera_pose import CameraMapPoseEstimate


def _rectangle(
    center_x: float,
    center_y: float,
    half_x: float,
    half_y: float,
) -> list[list[float]]:
    return [
        [center_x - half_x, center_y - half_y],
        [center_x + half_x, center_y - half_y],
        [center_x + half_x, center_y + half_y],
        [center_x - half_x, center_y + half_y],
    ]


def _optical_camera_transform() -> list[list[float]]:
    # Camera frame is OpenCV-compatible: +x right, +y down, +z forward.
    # In this synthetic scene map/vehicle +x is forward, +y is left, +z is up.
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
        "K": [
            [100.0, 0.0, 200.0],
            [0.0, 100.0, 100.0],
            [0.0, 0.0, 1.0],
        ],
        "D": [0.0, 0.0, 0.0, 0.0],
        "T_vehicle_camera": _optical_camera_transform(),
        "extrinsic_translation_unit": "m",
        "calibration_id": "synthetic-observability-test-only",
        "calibration_time": "2026-01-01T00:00:00Z",
        "vehicle_frame": "vehicle",
        "camera_frame": "camera_front",
        "is_placeholder": False,
    }


def _camera_pose(
    *,
    parent_frame: str = "map_test",
    time_offset_ms: float = 0.0,
    position_variance: float = 1e-4,
    yaw_variance: float = 1e-6,
) -> dict[str, Any]:
    covariance = np.diag(
        [position_variance] * 3 + [yaw_variance] * 3
    ).reshape(-1)
    return {
        "schema_version": "camera-map-pose/1.0",
        "timestamp_ns": 1_000_000_000,
        "frame_id": "camera_front",
        "parent_frame": parent_frame,
        "T_map_camera": _optical_camera_transform(),
        "position_xyz": [0.0, 0.0, 1.5],
        "orientation_xyzw": [-0.5, 0.5, -0.5, 0.5],
        "pose_covariance": covariance.tolist(),
        "localization_status": "valid",
        "time_offset_ms": time_offset_ms,
        "calibration_id": "synthetic-observability-test-only",
        "translation_unit": "m",
    }


def _static_map(
    obstacles: list[dict[str, Any]],
    *,
    complete: bool = True,
    frame: str = "map_test",
) -> dict[str, Any]:
    return {
        "schema_version": "static-obstacle-map/1.0",
        "coordinate_frame": frame,
        "map_units_per_meter": 1.0,
        "static_obstacles": obstacles,
        "mapped_regions": [
            {
                "region_id": "local-test-region",
                "polygon_xy": _rectangle(5.0, 0.0, 15.0, 10.0),
                "static_obstacle_layer_complete": complete,
            }
        ],
    }


def _obstacle(
    object_id: str,
    object_type: str,
    polygon: list[list[float]],
    *,
    min_z: float = 0.0,
    max_z: float = 2.8,
) -> dict[str, Any]:
    return {
        "id": object_id,
        "type": object_type,
        "polygon_xy": polygon,
        "min_z": min_z,
        "max_z": max_z,
        "confidence": 1.0,
        "source": "manual",
    }


def _scene(
    *,
    obstacles: list[dict[str, Any]] | None = None,
    complete: bool = True,
    other_slots: list[dict[str, Any]] | None = None,
    camera_pose: dict[str, Any] | None = None,
    map_frame: str = "map_test",
) -> dict[str, Any]:
    target = {
        "slot_id": "S-003",
        "polygon_map": _rectangle(10.0, 0.0, 1.0, 1.0),
        "center_map": [10.0, 0.0],
        "state": "unknown",
    }
    return {
        "target_slot_id": "S-003",
        "map_coordinate_frame": map_frame,
        "map_units_per_meter": 1.0,
        "camera_map_pose": _camera_pose() if camera_pose is None else camera_pose,
        "camera_calibration": _calibration(),
        "nominal_horizontal_fov_deg": 180.0,
        "slots": [target, *(other_slots or [])],
        "target_ground_z_m": 0.0,
        "static_obstacle_map": _static_map(
            obstacles or [],
            complete=complete,
        ),
    }


class CameraUpstreamContractTest(unittest.TestCase):
    def test_complete_region_with_no_wall_or_column_can_continue(self) -> None:
        result = camera_observability_tool(_scene())

        self.assertEqual(result["decision"], "use_camera")
        self.assertTrue(result["camera_usable"])

    def test_incomplete_local_region_is_fail_closed(self) -> None:
        result = camera_observability_tool(_scene(complete=False))

        self.assertEqual(result["decision"], "insufficient_information")
        self.assertFalse(result["camera_usable"])
        self.assertIn(
            "static_obstacle_region_incomplete",
            result["reason_codes"],
        )

    def test_static_map_frame_mismatch_is_reported_explicitly(self) -> None:
        scene = _scene()
        scene["static_obstacle_map"]["coordinate_frame"] = "wrong-map"

        result = camera_observability_tool(scene)

        self.assertEqual(result["decision"], "insufficient_information")
        self.assertIn("coordinate_frame_mismatch", result["reason_codes"])

    def test_complete_region_must_cover_the_full_target_not_only_center(self) -> None:
        scene = _scene()
        scene["static_obstacle_map"]["mapped_regions"][0]["polygon_xy"] = (
            _rectangle(5.0, 0.0, 5.2, 10.0)
        )

        result = camera_observability_tool(scene)

        self.assertEqual(result["decision"], "insufficient_information")
        self.assertIn(
            "static_obstacle_region_incomplete",
            result["reason_codes"],
        )

    def test_full_height_wall_completely_blocks_target(self) -> None:
        wall = _obstacle(
            "wall-01",
            "wall",
            _rectangle(5.0, 0.0, 0.3, 2.0),
        )
        result = camera_observability_tool(_scene(obstacles=[wall]))

        self.assertEqual(result["decision"], "do_not_use_camera")
        self.assertIn("wall-01", result["blocking_object_ids"])
        self.assertIn("static_obstacle_blocks_view", result["reason_codes"])

    def test_narrow_column_only_partially_blocks_target(self) -> None:
        column = _obstacle(
            "column-01",
            "column",
            _rectangle(5.0, 0.0, 0.2, 0.12),
        )
        result = camera_observability_tool(_scene(obstacles=[column]))

        self.assertEqual(result["decision"], "use_camera")
        self.assertGreater(result["blocked_ray_ratio"], 0.0)
        self.assertGreaterEqual(result["clear_ray_ratio"], 0.6)

    def test_low_curb_does_not_count_as_complete_occlusion(self) -> None:
        curb = _obstacle(
            "curb-01",
            "curb",
            _rectangle(5.0, 0.0, 0.3, 2.0),
            max_z=0.25,
        )
        result = camera_observability_tool(_scene(obstacles=[curb]))

        self.assertEqual(result["decision"], "use_camera")
        self.assertEqual(result["blocked_ray_ratio"], 0.0)
        self.assertNotIn("curb-01", result["blocking_object_ids"])

    def test_out_of_route_vehicle_polygon_remains_a_physical_blocker(self) -> None:
        physical_vehicle = {
            "slot_id": "S-002",
            "polygon_map": _rectangle(5.0, 0.0, 1.0, 1.5),
            "center_map": [5.0, 0.0],
            "state": "out_of_route",
            "scope_status": "out_of_route_scope",
            "occupancy_polygon_map": _rectangle(5.0, 0.0, 1.0, 1.5),
        }
        result = camera_observability_tool(
            _scene(other_slots=[physical_vehicle])
        )

        self.assertEqual(result["decision"], "do_not_use_camera")
        self.assertIn("S-002", result["blocking_object_ids"])

    def test_pose_staleness_covariance_and_frame_mismatch_fail_closed(self) -> None:
        cases = [
            (
                _camera_pose(time_offset_ms=60.0),
                "map_test",
                "camera_pose_stale",
            ),
            (
                _camera_pose(position_variance=0.36),
                "map_test",
                "camera_pose_uncertainty_too_high",
            ),
            (
                _camera_pose(parent_frame="different_map"),
                "map_test",
                "coordinate_frame_mismatch",
            ),
        ]
        for pose, frame, expected in cases:
            with self.subTest(reason=expected):
                result = camera_observability_tool(
                    _scene(camera_pose=pose, map_frame=frame)
                )
                self.assertEqual(
                    result["decision"],
                    "insufficient_information",
                )
                self.assertIn(expected, result["reason_codes"])

    def test_invalid_upstream_pose_contract_is_fail_closed_not_invalid_input(self) -> None:
        pose = _camera_pose()
        pose.update(
            {
                "T_map_camera": [],
                "position_xyz": [],
                "orientation_xyzw": [],
                "pose_covariance": [],
                "localization_status": "invalid",
                "reason_codes": ["pose_not_bracketed"],
            }
        )

        result = camera_observability_tool(_scene(camera_pose=pose))

        self.assertEqual(result["decision"], "insufficient_information")
        self.assertFalse(result["camera_usable"])
        self.assertIn("camera_pose_missing", result["reason_codes"])
        self.assertNotIn("invalid_input", result["reason_codes"])

    def test_fisheye_180_degree_left_and_right_boundaries_are_symmetric(self) -> None:
        calibration = CameraCalibration.from_mapping(_calibration())
        uv, valid = calibration.project_camera_points(
            [[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
        )

        np.testing.assert_array_equal(valid, [True, True])
        self.assertLess(uv[0, 0], calibration.K[0, 2])
        self.assertGreater(uv[1, 0], calibration.K[0, 2])
        self.assertAlmostEqual(
            calibration.K[0, 2] - uv[0, 0],
            uv[1, 0] - calibration.K[0, 2],
            places=12,
        )

    def test_debug_trace_comes_from_the_same_geometry_decision(self) -> None:
        trace: list[dict[str, Any]] = []
        result = assess_camera_observability(_scene(), debug_trace=trace)

        self.assertEqual(result.decision, "use_camera")
        self.assertEqual(len(trace), 27)
        self.assertTrue(all(item["status"] == "clear" for item in trace))
        self.assertTrue(all(item["pixel_uv"] is not None for item in trace))

    def test_full_pitch_orientation_affects_image_projection(self) -> None:
        base = np.asarray(_optical_camera_transform(), dtype=np.float64)
        angle = math.radians(80.0)
        camera_pitch = np.asarray(
            [
                [1.0, 0.0, 0.0],
                [0.0, math.cos(angle), -math.sin(angle)],
                [0.0, math.sin(angle), math.cos(angle)],
            ]
        )
        pitched = base.copy()
        pitched[:3, :3] = base[:3, :3] @ camera_pitch
        pose = CameraMapPoseEstimate(
            timestamp_ns=1_000_000_000,
            frame_id="camera_front",
            parent_frame="map_test",
            T_map_camera=pitched,
            pose_covariance=np.diag([1e-4] * 3 + [1e-6] * 3),
            localization_status="valid",
            time_offset_ms=0.0,
            calibration_id="synthetic-observability-test-only",
            translation_unit="m",
            map_units_per_meter=None,
        ).to_dict()

        result = camera_observability_tool(_scene(camera_pose=pose))

        self.assertEqual(result["decision"], "do_not_use_camera")
        self.assertIn("target_outside_fov", result["reason_codes"])


if __name__ == "__main__":
    unittest.main()
