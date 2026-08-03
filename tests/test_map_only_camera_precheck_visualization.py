from __future__ import annotations

from copy import deepcopy
import tempfile
import unittest
from pathlib import Path
from typing import Any

from PIL import Image

from parking_slot_part2.map_only_camera_precheck_visualization import (
    MACHINE_BEV_COLORS,
    render_annotated_map_only_precheck,
    render_camera_gate_bev,
)


def _rectangle(cx: float, cy: float, hx: float = 1.0, hy: float = 1.0) -> list[list[float]]:
    return [
        [cx - hx, cy - hy],
        [cx + hx, cy - hy],
        [cx + hx, cy + hy],
        [cx - hx, cy + hy],
    ]


def _local_map() -> dict[str, Any]:
    return {
        "schema_version": "part1-local-lidar-map/1.1",
        "map_units_per_meter": 1.0,
        "anchor_pose": {
            "frame_id": 7,
            "map_xy": [0.0, 0.0],
            "map_yaw_rad": 0.0,
        },
        "lidar_window": {"frame_count": 3},
        "lidar_coverage": {
            "polygon_map": _rectangle(6.0, 0.0, 9.0, 7.0),
        },
        "slots": [
            {
                "slot_id": "S-A",
                "state": "free",
                "polygon_map": _rectangle(8.0, -2.5),
            },
            {
                "slot_id": "S-B",
                "state": "unknown",
                "polygon_map": _rectangle(8.0, 2.5),
            },
            {
                "slot_id": "S-BLOCK",
                "state": "occupied",
                "polygon_map": _rectangle(4.0, 2.5),
            },
        ],
        "provisional_candidates": [
            {"slot_id": "S-A", "state": "free"},
            {"slot_id": "S-B", "state": "unknown"},
        ],
    }


def _target(
    slot_id: str,
    *,
    status: str,
    y: float,
    blocker: str | None = None,
) -> dict[str, Any]:
    marked = status == "marked_candidate"
    ray_status = "blocked" if blocker else "clear"
    return {
        "display_label": "A" if slot_id == "S-A" else "B",
        "target_slot_id": slot_id,
        "camera_gate_status": status,
        "camera_candidate": marked,
        "camera_likely_observable": marked,
        "camera_call_requested": False,
        "target_bearing_deg": -17.4 if y < 0 else 17.4,
        "target_distance_m": 8.38,
        "target_angular_width_deg": 12.0,
        "robust_center_fov_coverage": 1.0,
        "clear_ray_ratio": 0.0 if blocker else 1.0,
        "blocked_ray_ratio": 1.0 if blocker else 0.0,
        "uncertain_ray_ratio": 0.0,
        "blocking_object_ids": [] if blocker is None else [blocker],
        "potential_occluder_ids": [],
        "reason_codes": [
            "line_of_sight_robustly_clear"
            if blocker is None
            else "explicit_line_of_sight_blockage"
        ],
        "rays": [
            {
                "origin_map_xy": [0.0, 0.0],
                "sample_map_xy": [8.0, y],
                "status": ray_status,
                "object_id": blocker,
            }
        ],
    }


def _report() -> dict[str, Any]:
    return {
        "schema_version": "map-only-camera-precheck/1.0",
        "mode": "conservative_map_only_gate",
        "algorithm_executed": True,
        "semantic_camera_model_called": False,
        "camera_call_requested": False,
        "proxy_pose": {
            "source": "part1_anchor_pose",
            "position_map_xy": [0.0, 0.0],
            "yaw_deg": 0.0,
            "position_uncertainty_m": 0.6,
            "yaw_uncertainty_deg": 8.0,
        },
        "config": {"reliable_half_fov_deg": 80.0},
        "targets": [
            _target("S-A", status="marked_candidate", y=-2.5),
            _target(
                "S-B",
                status="not_marked",
                y=2.5,
                blocker="S-BLOCK",
            ),
        ],
        "input_limitations": [
            "Camera proxy is Part1 anchor pose, not a calibrated optical pose."
        ],
    }


class MapOnlyCameraPrecheckVisualizationTest(unittest.TestCase):
    def test_machine_bev_uses_only_fixed_colors_and_has_no_output_status(self) -> None:
        local_map = _local_map()
        first = _report()
        second = deepcopy(first)
        second["targets"][0] = _target(
            "S-A", status="not_marked", y=-2.5, blocker="S-BLOCK"
        )
        second["targets"][1] = _target(
            "S-B", status="marked_candidate", y=2.5
        )

        with tempfile.TemporaryDirectory() as temporary:
            first_path = Path(temporary) / "first.png"
            second_path = Path(temporary) / "second.png"
            first_meta = render_camera_gate_bev(local_map, first, first_path)
            second_meta = render_camera_gate_bev(local_map, second, second_path)

            self.assertEqual(first_path.read_bytes(), second_path.read_bytes())
            image = Image.open(first_path).convert("RGB")
            color_counts = image.getcolors(
                maxcolors=image.width * image.height
            ) or []
            colors = {color for _, color in color_counts}
            self.assertTrue(colors)
            self.assertTrue(colors.issubset(set(MACHINE_BEV_COLORS.values())))
            self.assertFalse(first_meta["anti_aliased"])
            self.assertFalse(first_meta["contains_text"])
            self.assertFalse(first_meta["contains_decision_rays"])
            self.assertFalse(first_meta["contains_decision_status"])
            self.assertTrue(first_meta["input_only"])
            self.assertEqual(first_meta["width_px"], second_meta["width_px"])

    def test_annotated_figure_records_proxy_fov_rays_marks_and_not_called(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "map_only_camera_precheck.png"
            metadata = render_annotated_map_only_precheck(
                _local_map(), _report(), output
            )

            self.assertGreater(output.stat().st_size, 10_000)
            self.assertEqual(metadata["candidate_half_fov_deg"], 80.0)
            self.assertEqual(metadata["yaw_uncertainty_deg"], 8.0)
            self.assertEqual(metadata["position_uncertainty_m"], 0.6)
            self.assertEqual(metadata["rendered_ray_count"], 2)
            self.assertFalse(metadata["camera_called"])
            self.assertEqual(
                metadata["target_labels"],
                {"S-A": "MARKED", "S-B": "NOT MARKED"},
            )

    def test_renderer_rejects_any_claim_that_camera_was_called(self) -> None:
        invalid_reports = []
        semantic = _report()
        semantic["semantic_camera_model_called"] = True
        invalid_reports.append(semantic)
        requested = _report()
        requested["camera_call_requested"] = True
        invalid_reports.append(requested)

        with tempfile.TemporaryDirectory() as temporary:
            for index, report in enumerate(invalid_reports):
                with self.subTest(index=index):
                    with self.assertRaises(ValueError):
                        render_camera_gate_bev(
                            _local_map(), report, Path(temporary) / f"{index}.png"
                        )


if __name__ == "__main__":
    unittest.main()
