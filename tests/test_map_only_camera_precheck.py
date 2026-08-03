from __future__ import annotations

from copy import deepcopy
import math
import unittest

from parking_slot_part2.map_only_camera_precheck import (
    MapOnlyCameraPrecheckConfig,
    evaluate_map_only_camera_candidates,
)


def _rectangle(cx: float, cy: float, hx: float = 0.5, hy: float = 1.0):
    return [
        [cx - hx, cy - hy],
        [cx + hx, cy - hy],
        [cx + hx, cy + hy],
        [cx - hx, cy + hy],
    ]


def _slot(slot_id: str, state: str, polygon, **extra):
    cx = sum(point[0] for point in polygon) / len(polygon)
    cy = sum(point[1] for point in polygon) / len(polygon)
    return {
        "slot_id": slot_id,
        "state": state,
        "polygon_map": polygon,
        "center_map": [cx, cy],
        **extra,
    }


def _local_map(slots, *, scale: float = 1.0, yaw_deg: float = 0.0, **extra):
    return {
        "schema_version": "part1-local-lidar-map/test",
        "map_units_per_meter": scale,
        "anchor_pose": {
            "frame_id": 7,
            "map_xy": [0.0, 0.0],
            "map_yaw_rad": math.radians(yaw_deg),
        },
        "slots": slots,
        **extra,
    }


class MapOnlyCameraPrecheckTest(unittest.TestCase):
    def test_clear_central_slot_is_marked_without_calling_camera(self):
        source = _local_map([_slot("target", "free", _rectangle(5.0, 0.0))])
        before = deepcopy(source)

        report = evaluate_map_only_camera_candidates(source, ["target"])
        target = report["targets"][0]

        self.assertEqual(target["status"], "marked_candidate")
        self.assertTrue(target["camera_candidate"])
        self.assertFalse(target["camera_call_requested"])
        self.assertFalse(report["semantic_camera_model_called"])
        self.assertFalse(report["camera_call_requested"])
        self.assertFalse(report["part1_slot_states_modified"])
        self.assertEqual(source, before)
        self.assertFalse(report["machine_bev"]["decision_status_encoded"])

    def test_unknown_target_is_eligible_because_gate_is_not_free_only(self):
        report = evaluate_map_only_camera_candidates(
            _local_map([_slot("target", "unknown", _rectangle(5.0, 0.0))]),
            ["target"],
        )
        self.assertEqual(report["targets"][0]["status"], "marked_candidate")
        self.assertEqual(report["targets"][0]["part1_state"], "unknown")

    def test_target_in_nominal_180_but_outside_central_80_is_not_marked(self):
        angle = math.radians(85.0)
        center = (6.0 * math.cos(angle), 6.0 * math.sin(angle))
        report = evaluate_map_only_camera_candidates(
            _local_map([_slot("edge", "free", _rectangle(*center, 0.1, 0.1))]),
            ["edge"],
        )
        target = report["targets"][0]
        self.assertEqual(target["status"], "not_marked")
        self.assertIn(
            "target_center_outside_conservative_fov", target["reason_codes"]
        )

    def test_occupied_slot_on_line_of_sight_blocks_and_returns_id(self):
        slots = [
            _slot("target", "free", _rectangle(6.0, 0.0)),
            _slot("near", "occupied", _rectangle(3.0, 0.0, 0.6, 1.2)),
        ]
        target = evaluate_map_only_camera_candidates(
            _local_map(slots), ["target"]
        )["targets"][0]
        self.assertEqual(target["status"], "not_marked")
        self.assertIn("near", target["blocking_object_ids"])
        self.assertIn("explicit_line_of_sight_blockage", target["reason_codes"])

    def test_occupied_slot_off_line_of_sight_does_not_block(self):
        slots = [
            _slot("target", "free", _rectangle(6.0, 0.0)),
            _slot("side", "occupied", _rectangle(3.0, 4.0, 0.5, 0.5)),
        ]
        target = evaluate_map_only_camera_candidates(
            _local_map(slots), ["target"]
        )["targets"][0]
        self.assertEqual(target["status"], "marked_candidate")
        self.assertNotIn("side", target["blocking_object_ids"])

    def test_unknown_slot_on_center_line_is_unresolved_potential_occluder(self):
        slots = [
            _slot("target", "free", _rectangle(6.0, 0.0)),
            _slot("maybe", "unknown", _rectangle(3.0, 0.0, 0.6, 1.2)),
        ]
        target = evaluate_map_only_camera_candidates(
            _local_map(slots), ["target"]
        )["targets"][0]
        self.assertEqual(target["status"], "not_marked")
        self.assertIn("maybe", target["potential_occluder_ids"])
        self.assertIn("unresolved_potential_occluder", target["reason_codes"])

    def test_static_wall_blocks_but_low_curb_does_not(self):
        wall = {
            "id": "wall_1",
            "polygon_map": _rectangle(3.0, 0.0, 0.15, 1.5),
            "max_z": 2.5,
            "confidence": 1.0,
        }
        curb = {
            "id": "curb_1",
            "polygon_map": _rectangle(3.0, 0.0, 0.15, 1.5),
            "max_z": 0.15,
            "confidence": 1.0,
        }
        target_slot = _slot("target", "free", _rectangle(6.0, 0.0))
        blocked = evaluate_map_only_camera_candidates(
            _local_map([target_slot], local_obstacles=[wall]), ["target"]
        )["targets"][0]
        clear = evaluate_map_only_camera_candidates(
            _local_map([target_slot], local_obstacles=[curb]), ["target"]
        )["targets"][0]
        self.assertEqual(blocked["status"], "not_marked")
        self.assertIn("wall_1", blocked["blocking_object_ids"])
        self.assertEqual(clear["status"], "marked_candidate")

    def test_angle_wrap_at_minus_180_plus_180_is_correct(self):
        world_angle = math.radians(-175.0)
        center = (6.0 * math.cos(world_angle), 6.0 * math.sin(world_angle))
        report = evaluate_map_only_camera_candidates(
            _local_map(
                [_slot("target", "free", _rectangle(*center, 0.3, 0.3))],
                yaw_deg=179.0,
            ),
            ["target"],
        )
        target = report["targets"][0]
        self.assertAlmostEqual(target["target_bearing_deg"], 6.0, places=3)
        self.assertEqual(target["status"], "marked_candidate")

    def test_metric_thresholds_are_invariant_to_map_scale(self):
        first = evaluate_map_only_camera_candidates(
            _local_map([_slot("target", "free", _rectangle(5.0, 0.0))]),
            ["target"],
        )["targets"][0]
        scaled_polygon = [[2.0 * x, 2.0 * y] for x, y in _rectangle(5.0, 0.0)]
        second = evaluate_map_only_camera_candidates(
            _local_map([_slot("target", "free", scaled_polygon)], scale=2.0),
            ["target"],
        )["targets"][0]
        self.assertEqual(first["status"], second["status"])
        self.assertAlmostEqual(first["target_distance_m"], second["target_distance_m"])
        self.assertAlmostEqual(
            first["target_angular_width_deg"], second["target_angular_width_deg"]
        )

    def test_missing_obstacle_layer_is_disclosed_not_assumed_complete(self):
        report = evaluate_map_only_camera_candidates(
            _local_map([_slot("target", "free", _rectangle(5.0, 0.0))]),
            ["target"],
        )
        self.assertTrue(
            any("local_obstacles is absent" in item for item in report["input_limitations"])
        )

    def test_config_is_strict_and_target_list_is_bounded(self):
        with self.assertRaisesRegex(ValueError, "unknown.*fields"):
            MapOnlyCameraPrecheckConfig.from_mapping({"magic_threshold": 1.0})
        local_map = _local_map(
            [
                _slot("a", "free", _rectangle(4.0, -2.0)),
                _slot("b", "free", _rectangle(4.0, 0.0)),
                _slot("c", "free", _rectangle(4.0, 2.0)),
            ]
        )
        with self.assertRaisesRegex(ValueError, "one or two"):
            evaluate_map_only_camera_candidates(local_map, ["a", "b", "c"])


if __name__ == "__main__":
    unittest.main()
