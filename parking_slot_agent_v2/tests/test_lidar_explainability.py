from __future__ import annotations

from types import SimpleNamespace
import unittest

from parking_slot_agent_v2.lidar_explainability import describe_lidar_render


def _pack(frame_count: int, *, valid_count: int | None = None) -> SimpleNamespace:
    valid = frame_count if valid_count is None else valid_count
    return SimpleNamespace(
        selected_frames=tuple(range(1000, 1000 + frame_count)),
        valid_frames=tuple(range(1000 + frame_count - valid, 1000 + frame_count)),
    )


def _card(
    source: str,
    *,
    core_points: int = 0,
    layers: int = 0,
    core_ray: float = 0.0,
    volume: float = 0.0,
) -> dict:
    return {
        "decision_context": {"evidence_source": source},
        "free_geometry": {
            "core_ray_coverage": core_ray,
            "observed_volume_ratio": volume,
        },
        "occupied_geometry": {
            "core_point_count": core_points,
            "supported_height_layers": layers,
            "boundary_ratio": 0.0,
        },
        "robustness": {"passing_variants": 0, "total_variants": 0},
    }


class LidarExplainabilitySemanticsTest(unittest.TestCase):
    def test_part1_11_frame_labels_do_not_claim_extended_window(self) -> None:
        semantics = describe_lidar_render(_pack(11), _card("part1_15frame"))

        self.assertIn("选择11帧", semantics["subtitle"])
        self.assertIn("有效11帧", semantics["subtitle"])
        self.assertIn("11帧证据", semantics["viewpoint_title"])
        self.assertEqual(semantics["viewpoint_labels"], ["当前证据有效视角 11"])
        self.assertNotIn("60帧", semantics["subtitle"])
        self.assertNotIn("扩展", " ".join(semantics["viewpoint_labels"]))

    def test_part1_15_frame_labels_do_not_claim_extra_45_frames(self) -> None:
        semantics = describe_lidar_render(_pack(15), _card("part1_15frame"))

        self.assertIn("选择15帧", semantics["subtitle"])
        self.assertEqual(semantics["viewpoint_labels"], ["当前证据有效视角 15"])
        self.assertNotIn("45", " ".join(semantics["viewpoint_labels"]))

    def test_extended_60_frame_labels_split_early_and_part1_tail(self) -> None:
        semantics = describe_lidar_render(
            _pack(60),
            _card("part2_extended_causal_lidar"),
        )

        self.assertIn("选择60帧", semantics["subtitle"])
        self.assertEqual(
            semantics["viewpoint_labels"],
            ["扩展有效视角 45", "Part1尾部有效视角 15"],
        )

    def test_unknown_with_core_returns_is_not_described_as_no_points(self) -> None:
        semantics = describe_lidar_render(
            _pack(11),
            _card("part1_15frame", core_points=45, layers=1, core_ray=0.037),
        )

        headline = semantics["unknown_headline"]
        self.assertIn("核心45点", headline)
        self.assertIn("1个高度层", headline)
        self.assertNotIn("无占用候选", headline)
        self.assertNotIn("完全未观测", headline)


if __name__ == "__main__":
    unittest.main()
