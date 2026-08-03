from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from parking_slot_hybrid_3d.shadow import build_shadow_comparison


class Hybrid3DShadowComparisonTest(unittest.TestCase):
    def test_common_baseline_and_new_route_slots_are_reported_separately(self) -> None:
        baseline = [
            {"slot_id": "a", "state": "box_vehicle_core_supported", "score": "0.9", "reason": "old-a"},
            {"slot_id": "b", "state": "box_boundary_conflict", "score": "0.4", "reason": "old-b"},
        ]
        scopes = [
            {"slot_id": "a", "scope_status": "in_route_scope", "reasons": '["scope_gate_passed"]'},
            {"slot_id": "b", "scope_status": "out_of_route_scope", "reasons": '["no_ray_or_hit_reached_slot"]'},
            {"slot_id": "c", "scope_status": "in_route_scope", "reasons": '["scope_gate_passed"]'},
            {"slot_id": "d", "scope_status": "partial_route_scope", "reasons": '["insufficient_ray_frames"]'},
        ]
        decisions = [
            {
                "slot_id": "a",
                "state": "unknown",
                "decision_reason": "weak_obstacle_evidence",
                "unknown_reasons": '["boundary_dominated","weak_vehicle_evidence"]',
                "agent_observable": "True",
            },
            {
                "slot_id": "c",
                "state": "free",
                "decision_reason": "strong_free_space_evidence",
                "unknown_reasons": "[]",
                "agent_observable": "False",
            },
            {
                "slot_id": "d",
                "state": "unknown",
                "decision_reason": "partial_route_scope",
                "unknown_reasons": '["insufficient_ray_frames","partial_route_scope"]',
                "agent_observable": "True",
            },
        ]

        result = build_shadow_comparison(baseline, scopes, decisions)

        self.assertEqual([row["slot_id"] for row in result.common_rows], ["a", "b"])
        self.assertEqual(result.common_rows[0]["new_state"], "unknown")
        self.assertEqual(
            result.common_rows[0]["migration_reasons"],
            ["weak_obstacle_evidence", "boundary_dominated", "weak_vehicle_evidence"],
        )
        self.assertEqual(result.common_rows[1]["new_state"], "")
        self.assertEqual(
            result.common_rows[1]["migration_reasons"],
            ["no_ray_or_hit_reached_slot"],
        )
        self.assertEqual([row["slot_id"] for row in result.new_route_rows], ["c", "d"])
        self.assertEqual(result.summary["baseline_common_count"], 2)
        self.assertEqual(result.summary["newly_evaluated_route_count"], 2)
        self.assertEqual(result.summary["gt_status"], "unavailable")
        self.assertEqual(
            result.summary["comparison_semantics"],
            "state_transition_and_coverage_without_gt_not_accuracy",
        )

    def test_comparison_rejects_duplicate_or_missing_decision_contracts(self) -> None:
        baseline = [{"slot_id": "a", "state": "old", "score": "1", "reason": "old"}]
        scopes = [{"slot_id": "a", "scope_status": "in_route_scope", "reasons": "[]"}]
        with self.assertRaisesRegex(ValueError, "missing decision"):
            build_shadow_comparison(baseline, scopes, [])
        with self.assertRaisesRegex(ValueError, "duplicate baseline slot_id"):
            build_shadow_comparison(baseline + baseline, scopes, [])

    def test_cli_writes_canonical_transition_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            hybrid = root / "hybrid"
            hybrid.mkdir()
            baseline_path = root / "baseline.csv"
            with baseline_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["slot_id", "state", "score", "reason"])
                writer.writeheader()
                writer.writerow({"slot_id": "a", "state": "old", "score": "0.8", "reason": "old"})
            (hybrid / "known_slot_scope.csv").write_text(
                "slot_id,scope_status,reasons\n"
                'a,in_route_scope,"[""scope_gate_passed""]"\n',
                encoding="utf-8",
            )
            (hybrid / "slot_decisions.csv").write_text(
                "slot_id,state,decision_reason,unknown_reasons,agent_observable\n"
                'a,unknown,insufficient_terminal_evidence,"[""low_free_volume_coverage""]",True\n',
                encoding="utf-8",
            )
            output = root / "comparison"
            completed = subprocess.run(
                [
                    sys.executable,
                    "scripts/build_hybrid_3d_shadow_report.py",
                    "--baseline-csv",
                    str(baseline_path),
                    "--hybrid-output-dir",
                    str(hybrid),
                    "--output-dir",
                    str(output),
                ],
                cwd=Path(__file__).resolve().parents[1],
                text=True,
                capture_output=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertTrue((output / "common_baseline_transition.csv").is_file())
            self.assertTrue((output / "newly_evaluated_route_slots.csv").is_file())
            summary = json.loads((output / "shadow_summary.json").read_text())
            self.assertEqual(summary["baseline_common_count"], 1)


if __name__ == "__main__":
    unittest.main()
