from __future__ import annotations

import csv
import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from scripts.build_hybrid_3d_global_map_report import build_report, merge_slot_states


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "build_hybrid_3d_global_map_report.py"


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


class Hybrid3DGlobalMapReportTest(unittest.TestCase):
    def test_cli_refuses_unlabelled_global_part1_render(self) -> None:
        completed = subprocess.run(
            ["python3", str(SCRIPT), "--input-dir", "/tmp/not-read"],
            cwd=PROJECT_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("Refusing to render a global state map", completed.stderr)
        self.assertIn("--legacy-route-audit", completed.stderr)

    def test_report_renders_every_slot_and_embeds_both_maps(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_dir = root / "input"
            output_dir = root / "report"
            input_dir.mkdir()

            slot_rows = []
            for index in range(5):
                x = float(index * 2)
                slot_rows.append(
                    {
                        "slot_id": f"slot_{index:04d}",
                        "polygon_map": [[x, 0.0], [x + 1.0, 0.0], [x + 1.0, 1.0], [x, 1.0]],
                        "center_map": [x + 0.5, 0.5],
                    }
                )
            slot_database = root / "slot_database.json"
            slot_database.write_text(json.dumps({"slot_count": 5, "slots": slot_rows}), encoding="utf-8")

            scopes = [
                {"slot_id": "slot_0000", "scope_status": "in_route_scope"},
                {"slot_id": "slot_0001", "scope_status": "in_route_scope"},
                {"slot_id": "slot_0002", "scope_status": "in_route_scope"},
                {"slot_id": "slot_0003", "scope_status": "partial_route_scope"},
                {"slot_id": "slot_0004", "scope_status": "out_of_route_scope"},
            ]
            decisions = [
                {
                    "slot_id": "slot_0000",
                    "scope_status": "in_route_scope",
                    "state": "occupied",
                    "decision_reason": "strong_occupied_evidence",
                    "unknown_reasons": "[]",
                },
                {
                    "slot_id": "slot_0001",
                    "scope_status": "in_route_scope",
                    "state": "free",
                    "decision_reason": "strong_free_space_evidence",
                    "unknown_reasons": "[]",
                },
                {
                    "slot_id": "slot_0002",
                    "scope_status": "in_route_scope",
                    "state": "unknown",
                    "decision_reason": "weak_obstacle_evidence",
                    "unknown_reasons": '["linear_static_structure"]',
                },
                {
                    "slot_id": "slot_0003",
                    "scope_status": "partial_route_scope",
                    "state": "unknown",
                    "decision_reason": "partial_route_scope",
                    "unknown_reasons": '["partial_route_scope"]',
                },
            ]
            _write_csv(input_dir / "known_slot_scope.csv", ["slot_id", "scope_status"], scopes)
            _write_csv(
                input_dir / "slot_decisions.csv",
                ["slot_id", "scope_status", "state", "decision_reason", "unknown_reasons"],
                decisions,
            )
            (input_dir / "summary.json").write_text(
                json.dumps({"pipeline": "fixture", "gt_status": "unavailable", "map_total": 5}),
                encoding="utf-8",
            )

            result = build_report(input_dir, slot_database, output_dir, "report.html")

            self.assertEqual(result["slot_count"], 5)
            self.assertEqual(
                result["counts"],
                {"occupied": 1, "free": 1, "unknown": 1, "partial": 1, "out_of_route": 1},
            )
            self.assertGreater((output_dir / "global_map_hybrid_3d_full.png").stat().st_size, 1000)
            self.assertGreater((output_dir / "global_map_hybrid_3d_route_zoom.png").stat().st_size, 1000)
            report = (output_dir / "report.html").read_text(encoding="utf-8")
            self.assertEqual(report.count("data:image/png;base64,"), 2)
            self.assertIn("annotated slots rendered: <b>5</b>", report)
            self.assertIn("weak_obstacle_evidence", report)

    def test_merge_rejects_missing_scope_instead_of_mislabelling_it_unknown(self) -> None:
        slots = {
            "slot_a": {"slot_id": "slot_a", "polygon_map": [[0, 0], [1, 0], [1, 1]]},
            "slot_b": {"slot_id": "slot_b", "polygon_map": [[2, 0], [3, 0], [3, 1]]},
        }
        with self.assertRaisesRegex(ValueError, "scope/database slot mismatch"):
            merge_slot_states(
                slots,
                [{"slot_id": "slot_a", "scope_status": "out_of_route_scope"}],
                [],
            )


if __name__ == "__main__":
    unittest.main()
