from __future__ import annotations

from copy import deepcopy
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np

from parking_slot_hybrid_3d.local_map_visualization import render_local_map
from parking_slot_part2.camera_observability import CameraObservabilityAssessment
from parking_slot_part2.camera_observability_map import (
    build_camera_observability_map_report,
)


def _rectangle(cx: float, cy: float, hx: float = 1.0, hy: float = 1.0) -> list[list[float]]:
    return [
        [cx - hx, cy - hy],
        [cx + hx, cy - hy],
        [cx + hx, cy + hy],
        [cx - hx, cy + hy],
    ]


def _local_map(*, coordinate_frame: str | None = "map-test") -> dict[str, Any]:
    slots = [
        {
            "slot_id": "S-A",
            "polygon_map": _rectangle(10.0, -3.0),
            "center_map": [10.0, -3.0],
            "heading_deg": 90.0,
            "state": "free",
        },
        {
            "slot_id": "S-B",
            "polygon_map": _rectangle(10.0, 3.0),
            "center_map": [10.0, 3.0],
            "heading_deg": 90.0,
            "state": "unknown",
        },
    ]
    return {
        "schema_version": "part1-local-lidar-map/1.1",
        "coordinate_frame": coordinate_frame,
        "map_units_per_meter": 1.0,
        "anchor_pose": {"frame_id": 7, "map_xy": [0.0, 0.0], "map_yaw_rad": 0.0},
        "lidar_window": {"frame_count": 3},
        "lidar_coverage": {
            "nominal_radius_m": 18.0,
            "pose_centers_map": [[0.0, 0.0]],
            "polygon_map": _rectangle(7.0, 0.0, 11.0, 8.0),
        },
        "slots": slots,
        "candidate_slot_id": None,
        "candidate": None,
        "provisional_candidates": [
            {
                "display_label": "A",
                "slot_id": "S-A",
                "state": "free",
                "visualization_only": True,
            },
            {
                "display_label": "B",
                "slot_id": "S-B",
                "state": "unknown",
                "visualization_only": True,
            },
        ],
    }


def _transform() -> list[list[float]]:
    # Camera optical +z points along map +x.
    return [
        [0.0, 0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0, 0.0],
        [0.0, -1.0, 0.0, 1.5],
        [0.0, 0.0, 0.0, 1.0],
    ]


def _strict_camera_scene() -> dict[str, Any]:
    covariance = np.diag([1e-4] * 3 + [1e-6] * 3).reshape(-1).tolist()
    return {
        "map_coordinate_frame": "map-test",
        "map_units_per_meter": 1.0,
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
            "calibration_id": "synthetic-map-visual-test-only",
            "translation_unit": "m",
        },
        "camera_calibration": {
            "schema_version": "camera-calibration/2.0",
            "camera_model": "fisheye",
            "image_width": 400,
            "image_height": 200,
            "K": [[100.0, 0.0, 200.0], [0.0, 100.0, 100.0], [0.0, 0.0, 1.0]],
            "D": [0.0, 0.0, 0.0, 0.0],
            "T_vehicle_camera": _transform(),
            "extrinsic_translation_unit": "m",
            "calibration_id": "synthetic-map-visual-test-only",
            "calibration_time": "2026-01-01T00:00:00Z",
            "vehicle_frame": "vehicle",
            "camera_frame": "camera_front",
            "is_placeholder": False,
        },
        "nominal_horizontal_fov_deg": 180.0,
        "target_ground_z_m": 0.0,
        "static_obstacle_map": {
            "schema_version": "static-obstacle-map/1.0",
            "coordinate_frame": "map-test",
            "map_units_per_meter": 1.0,
            "static_obstacles": [
                {
                    "id": "side-column",
                    "type": "column",
                    "polygon_xy": _rectangle(5.0, 7.0, 0.2, 0.2),
                    "min_z": 0.0,
                    "max_z": 2.8,
                    "confidence": 1.0,
                    "source": "manual",
                }
            ],
            "mapped_regions": [
                {
                    "region_id": "local",
                    "polygon_xy": _rectangle(7.0, 0.0, 14.0, 12.0),
                    "static_obstacle_layer_complete": True,
                }
            ],
        },
    }


class Part2CameraObservabilityMapTest(unittest.TestCase):
    def test_a_and_b_each_execute_gate_once_and_preserve_exact_debug_trace(self) -> None:
        snapshot = _local_map()
        expected_traces: dict[str, list[dict[str, Any]]] = {}

        def fake_assessment(
            observation: dict[str, Any],
            config: Any = None,
            *,
            debug_trace: list[dict[str, Any]] | None = None,
        ) -> CameraObservabilityAssessment:
            del config
            self.assertIsNotNone(debug_trace)
            assert debug_trace is not None
            target_id = str(observation["target_slot_id"])
            trace = [
                {
                    "sample_map_xy": [10.0, -3.0 if target_id == "S-A" else 3.0],
                    "target_z_m": 0.2,
                    "pixel_uv": [201.25, 99.75],
                    "status": "clear",
                    "quality": 0.8123456789,
                    "object_id": None,
                    "test_sentinel": f"first-{target_id}",
                },
                {
                    "sample_map_xy": [10.0, -3.0 if target_id == "S-A" else 3.0],
                    "target_z_m": 1.4,
                    "pixel_uv": [202.5, 98.5],
                    "status": "clear",
                    "quality": 0.7123456789,
                    "object_id": None,
                    "test_sentinel": f"second-{target_id}",
                },
            ]
            expected_traces[target_id] = deepcopy(trace)
            debug_trace.extend(trace)
            return CameraObservabilityAssessment(
                decision="use_camera",
                target_slot_id=target_id,
                nominal_fov_coverage=1.0,
                reliable_fov_coverage=1.0,
                edge_quality_score=0.8,
                clear_ray_ratio=1.0,
                reason_codes=(
                    "target_inside_reliable_fov",
                    "line_of_sight_mostly_clear",
                ),
            )

        with patch(
            "parking_slot_part2.camera_observability_map.assess_camera_observability",
            side_effect=fake_assessment,
        ) as gate:
            report = build_camera_observability_map_report(
                snapshot,
                camera_scene=_strict_camera_scene(),
            )

        self.assertEqual(gate.call_count, 2)
        self.assertEqual(
            [call.args[0]["target_slot_id"] for call in gate.call_args_list],
            ["S-A", "S-B"],
        )
        for target, overlay_target in zip(
            report["targets"], report["overlay"]["targets"], strict=True
        ):
            target_id = target["target_slot_id"]
            self.assertEqual(target["debug_trace"], expected_traces[target_id])
            self.assertIs(target["debug_trace"], overlay_target["debug_trace"])

    def test_zero_or_duplicate_provisional_targets_are_rejected_without_gate_call(self) -> None:
        empty = _local_map()
        empty["provisional_candidates"] = []

        duplicate = _local_map()
        duplicate["provisional_candidates"][1]["slot_id"] = "S-A"
        duplicate["provisional_candidates"][1]["state"] = "free"

        for invalid_snapshot, expected_message in (
            (empty, "at least one local Camera display target"),
            (duplicate, "provisional Camera targets must be unique"),
        ):
            with self.subTest(expected_message=expected_message):
                with patch(
                    "parking_slot_part2.camera_observability_map."
                    "assess_camera_observability"
                ) as gate:
                    with self.assertRaisesRegex(ValueError, expected_message):
                        build_camera_observability_map_report(
                            invalid_snapshot,
                            camera_scene=_strict_camera_scene(),
                        )
                    gate.assert_not_called()

    def test_missing_real_camera_context_is_drawn_as_fail_closed(self) -> None:
        snapshot = _local_map(coordinate_frame=None)

        report = build_camera_observability_map_report(snapshot)

        self.assertTrue(report["algorithm_executed"])
        self.assertFalse(report["semantic_camera_model_called"])
        self.assertFalse(report["camera_geometry_drawn"])
        self.assertEqual(len(report["targets"]), 2)
        for target in report["targets"]:
            assessment = target["assessment"]
            self.assertEqual(assessment["decision"], "insufficient_information")
            self.assertFalse(assessment["camera_usable"])
            self.assertIn("camera_pose_missing", assessment["reason_codes"])
            self.assertEqual(target["debug_trace"], [])

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "combined_missing.png"
            render_local_map(
                snapshot,
                snapshot["slots"],
                output,
                camera_overlay=report["overlay"],
            )
            self.assertGreater(output.stat().st_size, 10_000)

    def test_camera_scene_frame_or_scale_mismatch_fails_closed(self) -> None:
        snapshot = _local_map()
        scenes = []

        wrong_frame = _strict_camera_scene()
        wrong_frame["map_coordinate_frame"] = "different-map-frame"
        scenes.append(("coordinate frame", wrong_frame))

        wrong_scale = _strict_camera_scene()
        wrong_scale["map_units_per_meter"] = 2.0
        scenes.append(("map scale", wrong_scale))

        for mismatch, scene in scenes:
            with self.subTest(mismatch=mismatch):
                report = build_camera_observability_map_report(
                    snapshot,
                    camera_scene=scene,
                )
                self.assertTrue(report["input_limitations"])
                self.assertFalse(report["camera_geometry_drawn"])
                for target in report["targets"]:
                    self.assertEqual(
                        target["assessment"]["decision"],
                        "insufficient_information",
                    )
                    self.assertFalse(target["assessment"]["camera_usable"])
                    self.assertIn(
                        "coordinate_frame_mismatch",
                        target["assessment"]["reason_codes"],
                    )

    def test_combined_render_does_not_change_pure_part1_render_or_snapshot(self) -> None:
        snapshot = _local_map()
        snapshot_before = deepcopy(snapshot)

        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            pure_before = directory / "pure_before.png"
            combined = directory / "combined.png"
            pure_after = directory / "pure_after.png"

            render_local_map(snapshot, snapshot["slots"], pure_before)
            report = build_camera_observability_map_report(
                snapshot,
                camera_scene=_strict_camera_scene(),
            )
            render_local_map(
                snapshot,
                snapshot["slots"],
                combined,
                camera_overlay=report["overlay"],
            )
            render_local_map(snapshot, snapshot["slots"], pure_after)

            self.assertEqual(snapshot, snapshot_before)
            self.assertEqual(pure_before.read_bytes(), pure_after.read_bytes())
            self.assertNotEqual(pure_before.read_bytes(), combined.read_bytes())

    def test_explicit_null_config_uses_default_and_keeps_exact_trace(self) -> None:
        snapshot = _local_map()
        scene = _strict_camera_scene()
        scene["camera_observability_config"] = None

        report = build_camera_observability_map_report(
            snapshot,
            camera_scene=scene,
        )

        self.assertEqual(
            [target["assessment"]["decision"] for target in report["targets"]],
            ["use_camera", "use_camera"],
        )
        self.assertEqual(
            [len(target["debug_trace"]) for target in report["targets"]],
            [27, 27],
        )

    def test_algorithm_blocker_id_and_polygon_are_carried_into_map_overlay(self) -> None:
        snapshot = _local_map()
        scene = _strict_camera_scene()
        scene["slots"] = [
            {
                "slot_id": "BLOCKING-SLOT",
                "object_type": "slot",
                "state": "occupied",
                "polygon_map": _rectangle(5.0, -1.5, 0.8, 0.8),
            }
        ]

        report = build_camera_observability_map_report(
            snapshot,
            camera_scene=scene,
            target_slot_ids=["S-A"],
        )

        assessment = report["targets"][0]["assessment"]
        self.assertEqual(assessment["decision"], "do_not_use_camera")
        self.assertEqual(assessment["blocking_object_ids"], ["BLOCKING-SLOT"])
        self.assertEqual(
            report["overlay"]["occlusion_objects"],
            [
                {
                    "object_id": "BLOCKING-SLOT",
                    "object_type": "slot",
                    "polygon_map": _rectangle(5.0, -1.5, 0.8, 0.8),
                }
            ],
        )
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "combined_blocked.png"
            render_local_map(
                snapshot,
                snapshot["slots"],
                output,
                camera_overlay=report["overlay"],
            )
            self.assertGreater(output.stat().st_size, 10_000)

    def test_validated_scene_draws_same_algorithm_fov_and_debug_rays(self) -> None:
        snapshot = _local_map()

        report = build_camera_observability_map_report(
            snapshot,
            camera_scene=_strict_camera_scene(),
        )

        self.assertTrue(report["camera_geometry_drawn"])
        self.assertIsNotNone(report["overlay"]["camera_pose_map_xyyaw"])
        self.assertEqual(
            [target["assessment"]["decision"] for target in report["targets"]],
            ["use_camera", "use_camera"],
        )
        self.assertEqual(
            [len(target["debug_trace"]) for target in report["targets"]],
            [27, 27],
        )

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "combined_strict.png"
            render_local_map(
                snapshot,
                snapshot["slots"],
                output,
                camera_overlay=report["overlay"],
            )
            self.assertGreater(output.stat().st_size, 10_000)
            self.assertEqual(output.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")

    def test_overlay_cannot_claim_algorithm_without_execution_marker(self) -> None:
        snapshot = _local_map()
        report = build_camera_observability_map_report(snapshot)
        overlay = dict(report["overlay"])
        overlay["algorithm_executed"] = False

        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "executed Camera gate"):
                render_local_map(
                    snapshot,
                    snapshot["slots"],
                    Path(temporary) / "invalid.png",
                    camera_overlay=overlay,
                )


if __name__ == "__main__":
    unittest.main()
