from __future__ import annotations

import math
import unittest
from typing import Any

from parking_slot_part2.camera_observability import (
    CAMERA_REASON_CODES,
    CameraObservabilityConfig,
    CameraObservabilityInput,
    LineOfSightObject,
    assess_camera_observability,
    camera_observability_tool,
)


Point = tuple[float, float]


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


def _slot(
    slot_id: str,
    polygon_map: list[list[float]],
    *,
    state: str = "unknown",
    center_map: Point | None = None,
    occupied_probability: float | None = None,
    occupancy_polygon_map: list[list[float]] | None = None,
) -> dict[str, Any]:
    if center_map is None:
        center_map = (
            sum(point[0] for point in polygon_map) / len(polygon_map),
            sum(point[1] for point in polygon_map) / len(polygon_map),
        )
    result: dict[str, Any] = {
        "slot_id": slot_id,
        "polygon_map": polygon_map,
        "center_map": list(center_map),
        "heading_deg": 0.0,
        "state": state,
    }
    if occupied_probability is not None:
        result["occupied_probability"] = occupied_probability
    if occupancy_polygon_map is not None:
        result["occupancy_polygon_map"] = occupancy_polygon_map
    return result


def _target_slot(
    *,
    polygon_map: list[list[float]] | None = None,
    center_map: Point = (10.0, 0.0),
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "slot_id": "S-003",
        "center_map": list(center_map),
        "heading_deg": 0.0,
        "state": "unknown",
    }
    if polygon_map is not None:
        result["polygon_map"] = polygon_map
    else:
        result["polygon_map"] = _rectangle(
            center_map[0], center_map[1], 1.0, 1.0
        )
    return result


def _scene(
    *,
    target: dict[str, Any] | None = None,
    other_slots: list[dict[str, Any]] | None = None,
    static_obstacles: list[dict[str, Any]] | None = None,
    camera_pose_map_xyyaw: list[float] | None = None,
) -> dict[str, Any]:
    if target is None:
        target = _target_slot()
    return {
        "target_slot_id": target["slot_id"],
        "camera_pose_map_xyyaw": (
            [0.0, 0.0, 0.0]
            if camera_pose_map_xyyaw is None
            else camera_pose_map_xyyaw
        ),
        # A non-empty identity is enough for these geometry-only tests. No
        # Camera image or semantic detection result is supplied.
        "camera_calibration": {"calibration_id": "unit-test-calibration"},
        "nominal_horizontal_fov_deg": 180.0,
        # Legacy 2D inputs are retained only to regression-test the original
        # public geometry behavior. Production callers must use the strict
        # camera_map_pose/calibration/static-map contract.
        "camera_observability_config": {
            "allow_legacy_2d_test_contract": True,
        },
        "map_units_per_meter": 1.0,
        "slots": [target, *(other_slots or [])],
        # An explicit empty array means the static-obstacle layer was loaded
        # and contains no obstacle in this test scene.
        "static_obstacles": [] if static_obstacles is None else static_obstacles,
    }


class CameraUseDecisionTest(unittest.TestCase):
    def test_legacy_2d_contract_is_fail_closed_without_explicit_test_switch(self) -> None:
        scene = _scene()
        scene.pop("camera_observability_config")

        payload = camera_observability_tool(scene)

        self.assertEqual(payload["decision"], "insufficient_information")
        self.assertFalse(payload["camera_usable"])
        self.assertIn("camera_pose_missing", payload["reason_codes"])
        self.assertIn("camera_intrinsics_missing", payload["reason_codes"])
        self.assertIn("camera_extrinsics_missing", payload["reason_codes"])
        self.assertIn(
            "static_obstacle_region_incomplete",
            payload["reason_codes"],
        )

    def test_central_clear_target_uses_camera_and_returns_only_use_decision(self) -> None:
        payload = camera_observability_tool(_scene())

        self.assertEqual(payload["decision"], "use_camera")
        self.assertIs(payload["camera_usable"], True)
        self.assertEqual(payload["target_slot_id"], "S-003")
        self.assertAlmostEqual(payload["target_bearing_deg"], 0.0, places=6)
        self.assertAlmostEqual(payload["target_distance_m"], 10.0, places=6)
        self.assertIn("target_inside_reliable_fov", payload["reason_codes"])
        self.assertIn("line_of_sight_mostly_clear", payload["reason_codes"])
        self.assertEqual(payload["blocking_object_ids"], [])
        self.assertEqual(payload["potential_occluder_ids"], [])

        expected_metrics = {
            "nominal_fov_coverage",
            "reliable_fov_coverage",
            "edge_quality_score",
            "clear_ray_ratio",
            "blocked_ray_ratio",
            "uncertain_ray_ratio",
        }
        self.assertTrue(expected_metrics.issubset(payload))
        for forbidden in (
            "observability",
            "can_confirm_free",
            "can_confirm_occupied",
            "state",
            "prediction",
        ):
            self.assertNotIn(forbidden, payload)

        parsed = CameraObservabilityInput.from_mapping(_scene())
        assessed = assess_camera_observability(
            parsed,
            CameraObservabilityConfig(allow_legacy_2d_test_contract=True),
        )
        self.assertEqual(assessed.decision, "use_camera")

    def test_target_inside_nominal_fov_but_in_unreliable_edge_is_not_used(self) -> None:
        # atan2(10, 1.4) ~= 82 degrees: nominally visible, but deliberately in
        # the conservative unreliable edge zone.
        target = _target_slot(
            polygon_map=_rectangle(1.4, 10.0, 0.2, 0.2),
            center_map=(1.4, 10.0),
        )
        payload = camera_observability_tool(_scene(target=target))

        self.assertEqual(payload["decision"], "do_not_use_camera")
        self.assertIs(payload["camera_usable"], False)
        self.assertGreater(payload["nominal_fov_coverage"], 0.0)
        self.assertIn("target_in_unreliable_edge_region", payload["reason_codes"])
        self.assertNotIn("target_outside_fov", payload["reason_codes"])

    def test_target_completely_outside_nominal_fov_is_not_used(self) -> None:
        target = _target_slot(
            polygon_map=_rectangle(-5.0, 8.66, 0.2, 0.2),
            center_map=(-5.0, 8.66),
        )
        payload = camera_observability_tool(_scene(target=target))

        self.assertEqual(payload["decision"], "do_not_use_camera")
        self.assertIs(payload["camera_usable"], False)
        self.assertEqual(payload["nominal_fov_coverage"], 0.0)
        self.assertIn("target_outside_fov", payload["reason_codes"])

    def test_nearer_occupied_slot_on_line_of_sight_blocks_target(self) -> None:
        middle_slot = _slot(
            "S-002",
            _rectangle(5.0, 0.0, 1.0, 1.5),
            state="occupied",
        )
        payload = camera_observability_tool(_scene(other_slots=[middle_slot]))

        self.assertEqual(payload["decision"], "do_not_use_camera")
        self.assertIs(payload["camera_usable"], False)
        self.assertIn("explicit_line_of_sight_blockage", payload["reason_codes"])
        self.assertIn("S-002", payload["blocking_object_ids"])
        self.assertGreater(payload["blocked_ray_ratio"], 0.5)

    def test_nearer_occupied_slot_away_from_line_of_sight_does_not_block(self) -> None:
        side_slot = _slot(
            "S-002",
            _rectangle(5.0, 5.0, 1.0, 1.0),
            state="occupied",
        )
        payload = camera_observability_tool(_scene(other_slots=[side_slot]))

        self.assertEqual(payload["decision"], "use_camera")
        self.assertIs(payload["camera_usable"], True)
        self.assertNotIn("S-002", payload["blocking_object_ids"])
        self.assertEqual(payload["blocked_ray_ratio"], 0.0)

    def test_unknown_slot_on_line_of_sight_is_an_unresolved_potential_occluder(self) -> None:
        unknown_middle_slot = _slot(
            "S-002",
            _rectangle(5.0, 0.0, 1.0, 1.5),
            state="unknown",
        )
        payload = camera_observability_tool(
            _scene(other_slots=[unknown_middle_slot])
        )

        self.assertEqual(payload["decision"], "insufficient_information")
        self.assertIs(payload["camera_usable"], False)
        self.assertIn("unresolved_potential_occluder", payload["reason_codes"])
        self.assertIn("S-002", payload["potential_occluder_ids"])
        self.assertGreater(payload["uncertain_ray_ratio"], 0.0)

    def test_partial_corner_occlusion_still_uses_camera_when_clear_ratio_passes(self) -> None:
        target = _target_slot(
            polygon_map=_rectangle(10.0, 0.0, 1.0, 2.0),
            center_map=(10.0, 0.0),
        )
        occupied_slot = _slot(
            "S-002",
            _rectangle(5.0, 1.5, 0.8, 0.8),
            state="occupied",
            # Prefer the real vehicle footprint over the broader slot polygon.
            occupancy_polygon_map=_rectangle(5.0, 0.9, 0.5, 0.15),
        )
        config = CameraObservabilityConfig(
            minimum_clear_ray_ratio=0.60,
            maximum_blocked_ray_ratio=0.50,
            allow_legacy_2d_test_contract=True,
        )
        result = assess_camera_observability(
            _scene(target=target, other_slots=[occupied_slot]),
            config=config,
        )

        self.assertEqual(result.decision, "use_camera")
        self.assertTrue(result.camera_usable)
        self.assertGreater(result.blocked_ray_ratio, 0.0)
        self.assertGreaterEqual(
            result.clear_ray_ratio,
            config.minimum_clear_ray_ratio,
        )
        self.assertLessEqual(
            result.blocked_ray_ratio,
            config.maximum_blocked_ray_ratio,
        )

    def test_wall_or_pillar_on_line_of_sight_blocks_target(self) -> None:
        for object_type in ("wall", "pillar"):
            with self.subTest(object_type=object_type):
                object_id = f"{object_type}-01"
                obstacle = {
                    "object_id": object_id,
                    "object_type": object_type,
                    "polygon_map": _rectangle(5.0, 0.0, 0.4, 1.5),
                }
                payload = camera_observability_tool(
                    _scene(static_obstacles=[obstacle])
                )

                self.assertEqual(payload["decision"], "do_not_use_camera")
                self.assertIs(payload["camera_usable"], False)
                self.assertIn(
                    "explicit_line_of_sight_blockage", payload["reason_codes"]
                )
                self.assertIn(object_id, payload["blocking_object_ids"])

    def test_relative_bearing_wraps_correctly_across_minus_180_plus_180(self) -> None:
        # Camera yaw is +179 degrees while the target lies at -179 degrees in
        # world coordinates. The normalized relative bearing is +2 degrees,
        # not -358 degrees.
        target_angle = math.radians(-179.0)
        center = (10.0 * math.cos(target_angle), 10.0 * math.sin(target_angle))
        target = _target_slot(
            polygon_map=_rectangle(center[0], center[1], 0.2, 0.2),
            center_map=center,
        )
        payload = camera_observability_tool(
            _scene(
                target=target,
                camera_pose_map_xyyaw=[0.0, 0.0, math.radians(179.0)],
            )
        )

        self.assertEqual(payload["decision"], "use_camera")
        self.assertAlmostEqual(payload["target_bearing_deg"], 2.0, delta=0.1)

    def test_missing_pose_fov_or_target_polygon_is_insufficient_information(self) -> None:
        missing_pose = _scene()
        missing_pose.pop("camera_pose_map_xyyaw")

        missing_fov = _scene()
        missing_fov.pop("nominal_horizontal_fov_deg")

        target_without_polygon = _target_slot()
        target_without_polygon.pop("polygon_map")
        missing_polygon = _scene(target=target_without_polygon)

        cases = (
            ("pose", missing_pose, "camera_pose_missing"),
            ("fov", missing_fov, "camera_calibration_missing"),
            ("target polygon", missing_polygon, "target_geometry_missing"),
        )
        for label, scene, expected_reason in cases:
            with self.subTest(missing=label):
                payload = camera_observability_tool(scene)
                self.assertEqual(payload["decision"], "insufficient_information")
                self.assertIs(payload["camera_usable"], False)
                self.assertIn(expected_reason, payload["reason_codes"])

    def test_missing_static_obstacle_layer_is_not_treated_as_known_clear(self) -> None:
        scene = _scene()
        scene.pop("static_obstacles")

        payload = camera_observability_tool(scene)

        self.assertEqual(payload["decision"], "insufficient_information")
        self.assertIs(payload["camera_usable"], False)
        self.assertIn("occlusion_information_missing", payload["reason_codes"])

    def test_real_part1_containers_join_scope_and_use_only_explicit_probability(self) -> None:
        target = _target_slot()
        middle = _slot(
            "S-002",
            _rectangle(5.0, 0.0, 1.0, 1.5),
            state="unknown",
        )
        target.pop("state")
        middle.pop("state")
        base = {
            "target_slot_id": "S-003",
            "camera_pose_map_xyyaw": [0.0, 0.0, 0.0],
            "camera_calibration": {"calibration_id": "unit-test-calibration"},
            "nominal_horizontal_fov_deg": 180.0,
            "camera_observability_config": {
                "allow_legacy_2d_test_contract": True,
            },
            "slot_database": {
                "map_units_per_meter": 1.0,
                "slots": [target, middle],
            },
            "known_slot_scope": [
                {"slot_id": "S-003", "scope_status": "in_route_scope"},
                {"slot_id": "S-002", "scope_status": "partial_route_scope"},
            ],
            "static_obstacles": [],
        }
        with_probability = {
            **base,
            "slot_decisions": {
                "decisions": [
                    {"slot_id": "S-003", "state": "unknown"},
                    {
                        "slot_id": "S-002",
                        "state": "unknown",
                        "scope_status": "partial_route_scope",
                        "occupied_probability": 0.90,
                    },
                ]
            },
        }
        strength_only = {
            **base,
            "slot_decisions": {
                "decisions": [
                    {"slot_id": "S-003", "state": "unknown"},
                    {
                        "slot_id": "S-002",
                        "state": "unknown",
                        "scope_status": "partial_route_scope",
                        "occupied_evidence": {"strength": 0.99},
                    },
                ]
            },
        }

        blocked = camera_observability_tool(with_probability)
        unresolved = camera_observability_tool(strength_only)

        self.assertEqual(blocked["decision"], "do_not_use_camera")
        self.assertIn("S-002", blocked["blocking_object_ids"])
        self.assertEqual(unresolved["decision"], "insufficient_information")
        self.assertIn("S-002", unresolved["potential_occluder_ids"])

    def test_out_of_route_scope_slot_is_not_a_vehicle_blocker(self) -> None:
        target = _target_slot()
        outside_scope = _slot(
            "S-002",
            _rectangle(5.0, 0.0, 1.0, 1.5),
            state="unknown",
        )
        outside_scope.pop("state")
        scene = _scene(target=target)
        scene["slots"] = [target, outside_scope]
        scene["known_slot_scope"] = [
            {"slot_id": "S-002", "scope_status": "out_of_route_scope"}
        ]

        payload = camera_observability_tool(scene)

        self.assertEqual(payload["decision"], "use_camera")
        self.assertNotIn("S-002", payload["blocking_object_ids"])
        self.assertNotIn("S-002", payload["potential_occluder_ids"])

    def test_first_hit_keeps_nearer_unknown_ahead_of_farther_occupied(self) -> None:
        near_unknown = _slot(
            "S-near",
            _rectangle(3.0, 0.0, 0.5, 1.5),
            state="unknown",
        )
        far_occupied = _slot(
            "S-far",
            _rectangle(6.0, 0.0, 0.5, 1.5),
            state="occupied",
        )

        payload = camera_observability_tool(
            _scene(other_slots=[far_occupied, near_unknown])
        )

        self.assertEqual(payload["decision"], "insufficient_information")
        self.assertIn("S-near", payload["potential_occluder_ids"])
        self.assertNotIn("S-far", payload["blocking_object_ids"])

    def test_camera_inside_an_occupied_polygon_fails_closed_as_blocked(self) -> None:
        containing_slot = _slot(
            "S-containing",
            _rectangle(0.0, 0.0, 1.0, 1.0),
            state="occupied",
        )

        payload = camera_observability_tool(
            _scene(other_slots=[containing_slot])
        )

        self.assertEqual(payload["decision"], "do_not_use_camera")
        self.assertIn(
            "explicit_line_of_sight_blockage",
            payload["reason_codes"],
        )
        self.assertIn("S-containing", payload["blocking_object_ids"])

    def test_calibrated_angle_quality_curve_overrides_fallback_edge_zones(self) -> None:
        target = _target_slot(
            polygon_map=_rectangle(1.4, 10.0, 0.2, 0.2),
            center_map=(1.4, 10.0),
        )
        scene = _scene(target=target)
        scene["camera_calibration"] = {
            "calibration_id": "wide-camera-calibration",
            "angle_to_quality_curve": [[0.0, 1.0], [90.0, 0.90]],
        }

        payload = camera_observability_tool(scene)

        self.assertEqual(payload["decision"], "use_camera")
        self.assertGreaterEqual(payload["reliable_fov_coverage"], 0.60)

    def test_malformed_config_and_string_clearance_fail_closed(self) -> None:
        malformed_config = _scene()
        malformed_config["camera_observability_config"] = []
        invalid_clearance = _scene(
            other_slots=[
                {
                    **_slot(
                        "S-002",
                        _rectangle(5.0, 0.0, 1.0, 1.5),
                        state="unknown",
                    ),
                    "clearance_proven": "false",
                }
            ]
        )

        config_result = camera_observability_tool(malformed_config)
        clearance_result = camera_observability_tool(invalid_clearance)

        self.assertEqual(config_result["decision"], "insufficient_information")
        self.assertIn(
            "invalid_camera_observability_config",
            config_result["reason_codes"],
        )
        self.assertEqual(clearance_result["decision"], "insufficient_information")
        self.assertIn("invalid_input", clearance_result["reason_codes"])
        self.assertTrue(set(config_result["reason_codes"]) <= CAMERA_REASON_CODES)
        self.assertTrue(set(clearance_result["reason_codes"]) <= CAMERA_REASON_CODES)


# Import compatibility is intentional: downstream callers already import this
# public geometry-object name even though the JSON tests above use mappings.
_PUBLIC_LINE_OF_SIGHT_OBJECT = LineOfSightObject


if __name__ == "__main__":
    unittest.main()
