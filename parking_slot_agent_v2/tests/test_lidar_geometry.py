from __future__ import annotations

from types import SimpleNamespace
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from parking_slot_agent_v2.contracts import EvidenceRecord
from parking_slot_agent_v2.lidar_geometry import (
    assess_terminal_geometry,
    build_geometry_card,
    combine_triptych_and_card,
    render_geometry_card,
)


class LidarGeometryCardTest(unittest.TestCase):
    def _case(self) -> SimpleNamespace:
        decision = {
            "occupied_evidence": {
                "strength": 0.98,
                "strong": True,
                "weak": False,
                "failures": [],
                "features": {
                    "point_count": 400,
                    "core_point_count": 350,
                    "supported_frame_count": 15,
                    "temporal_support": 1.0,
                    "temporal_consistency": 0.94,
                    "supported_layer_count": 3,
                    "z95_m": 1.8,
                    "height_span_m": 1.4,
                    "extent_x_m": 4.7,
                    "extent_y_m": 1.9,
                    "core_overlap": 0.72,
                    "adjacent_overlap": 0.02,
                    "boundary_ratio": 0.2,
                    "linearity": 0.1,
                },
            },
            "free_evidence": {
                "strength": 0.2,
                "strong": False,
                "failures": ["unresolved_core_hit_evidence"],
                "unresolved_core_hit": True,
            },
            "stability": {
                "stable": True,
                "passing_variants": 7,
                "total_variants": 7,
                "pass_ratio": 1.0,
                "failures": [],
            },
        }
        evidence = EvidenceRecord(
            evidence_id="part1",
            tool_name="part1_15frame",
            round_index=0,
            status="ok",
            artifact_paths=[],
            summary="fixture",
            metadata={"decision": decision},
            modality="lidar",
        )
        return SimpleNamespace(slot_id="slot_fixture", evidence=[evidence])

    def test_card_preserves_measured_support_and_nonterminal_boundary(self) -> None:
        pack = SimpleNamespace(
            valid_frames=np.arange(1, 16, dtype=np.int32),
            points_local_xyzi=np.zeros((1234, 4), dtype=np.float32),
        )
        card = build_geometry_card(self._case(), pack)
        self.assertFalse(card["terminal_decision"])
        self.assertFalse(card["scores_calibrated"])
        self.assertEqual(
            card["occupied_geometry"]["support_level"],
            "strong_occupied_geometry_candidate",
        )
        self.assertEqual(card["occupied_geometry"]["core_point_count"], 350)
        self.assertEqual(card["pack"]["valid_frame_count"], 15)
        assessment = assess_terminal_geometry(card)
        self.assertTrue(assessment["occupied_eligible"])
        self.assertFalse(assessment["free_eligible"])

    def test_render_and_composite_are_valid_images(self) -> None:
        pack = SimpleNamespace(
            valid_frames=np.arange(1, 16, dtype=np.int32),
            points_local_xyzi=np.zeros((10, 4), dtype=np.float32),
        )
        card = build_geometry_card(self._case(), pack)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            card_path = render_geometry_card(card, root / "card.png")
            Image.new("RGB", (1536, 512), "white").save(root / "triptych.png")
            combined = combine_triptych_and_card(
                root / "triptych.png", card_path, root / "combined.png"
            )
            with Image.open(combined) as image:
                self.assertEqual(image.size, (1536, 1292))

    def test_extended_decision_supersedes_part1_reason_without_erasing_history(self) -> None:
        case = self._case()
        case.decision_reason = "part1_pose_unstable"
        case.unknown_reasons = ["pose_sensitive_terminal"]
        decision = dict(case.evidence[0].metadata["decision"])
        decision["decision_reason"] = "strong_occupied_evidence"
        decision["unknown_reasons"] = []
        decision["selected_frames"] = list(range(1, 76))
        pack = SimpleNamespace(
            valid_frames=np.arange(1, 76, dtype=np.int32),
            points_local_xyzi=np.zeros((10, 4), dtype=np.float32),
        )
        card = build_geometry_card(
            case,
            pack,
            decision_override=decision,
            decision_source="part2_extended_causal_lidar",
        )
        context = card["decision_context"]
        self.assertEqual(context["part1_decision_reason"], "part1_pose_unstable")
        self.assertEqual(context["part1_unresolved_reasons"], ["pose_sensitive_terminal"])
        self.assertEqual(context["decision_reason"], "strong_occupied_evidence")
        self.assertEqual(context["unresolved_reasons"], [])
        self.assertEqual(context["evidence_window_frames"], 75)


if __name__ == "__main__":
    unittest.main()
