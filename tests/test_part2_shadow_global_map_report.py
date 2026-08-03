from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

from matplotlib.axes import Axes
from PIL import Image

from parking_slot_part2.queueing import canonical_sha256
from scripts.build_part2_shadow_global_map_report import (
    ShadowOverlay,
    _draw_evaluated_before_after_map,
    build_report,
)


HASH_A = "sha256:" + "a" * 64
HASH_B = "sha256:" + "b" * 64
EVIDENCE_ID = "ev_" + "c" * 64


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "slot_id",
                "scope_status",
                "state",
                "decision_reason",
                "unknown_reasons",
                "reference_frames",
            ),
        )
        writer.writeheader()
        writer.writerows(rows)


def _resolution(
    *,
    task_id: str,
    slot_id: str,
    scope_status: str,
    state: str,
    reason_codes: list[str],
    evidence_refs: list[str],
) -> dict[str, object]:
    resolved = state in {"occupied", "free"}
    return {
        "task_id": task_id,
        "slot_id": slot_id,
        "scope_status": scope_status,
        "input_state": "unknown",
        "state": state,
        "decision_source": "part2_agent_validated" if resolved else "part2_agent_unknown",
        "model_turns": 2,
        "tool_calls_attempted": 1,
        "stop_reasons": ["final_proposal"],
        "completion_status": "resolved" if resolved else "remained_unknown",
        "group_ids": [f"group-{slot_id}"],
        "trace_span_ids": [f"trace-{slot_id}"],
        "attempt_ids": [f"attempt-{slot_id}"],
        "evidence_refs": evidence_refs,
        "reason_codes": reason_codes,
    }


class Part2ShadowGlobalMapReportTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.input_dir = self.root / "part1"
        self.input_dir.mkdir()
        self.output_dir = self.root / "shadow_report"

        slots = []
        for index in range(5):
            x = float(index * 2)
            slots.append(
                {
                    "slot_id": f"slot_{index:04d}",
                    "polygon_map": [[x, 0.0], [x + 1.0, 0.0], [x + 1.0, 1.0], [x, 1.0]],
                    "center_map": [x + 0.5, 0.5],
                }
            )
        self.slot_database = self.root / "slot_database.json"
        self.slot_database.write_text(json.dumps({"slots": slots}), encoding="utf-8")

        scopes = [
            {"slot_id": "slot_0000", "scope_status": "in_route_scope"},
            {"slot_id": "slot_0001", "scope_status": "in_route_scope"},
            {"slot_id": "slot_0002", "scope_status": "in_route_scope"},
            {"slot_id": "slot_0003", "scope_status": "partial_route_scope"},
            {"slot_id": "slot_0004", "scope_status": "out_of_route_scope"},
        ]
        with (self.input_dir / "known_slot_scope.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=("slot_id", "scope_status"))
            writer.writeheader()
            writer.writerows(scopes)

        self.base_rows: list[dict[str, object]] = [
            {
                "slot_id": "slot_0000",
                "scope_status": "in_route_scope",
                "state": "occupied",
                "decision_reason": "strong_occupied_evidence",
                "unknown_reasons": [],
                "reference_frames": [1],
                "extension": {"preserved": 0},
            },
            {
                "slot_id": "slot_0001",
                "scope_status": "in_route_scope",
                "state": "free",
                "decision_reason": "strong_free_space_evidence",
                "unknown_reasons": [],
                "reference_frames": [2],
                "extension": {"preserved": 1},
            },
            {
                "slot_id": "slot_0002",
                "scope_status": "in_route_scope",
                "state": "unknown",
                "decision_reason": "weak_obstacle_evidence",
                "unknown_reasons": ["weak_vehicle_evidence"],
                "reference_frames": [3, 4],
                "extension": {"preserved": 2},
            },
            {
                "slot_id": "slot_0003",
                "scope_status": "partial_route_scope",
                "state": "unknown",
                "decision_reason": "partial_route_scope",
                "unknown_reasons": ["partial_route_scope"],
                "reference_frames": [5],
                "extension": {"preserved": 3},
            },
        ]
        csv_rows = [
            {
                "slot_id": str(row["slot_id"]),
                "scope_status": str(row["scope_status"]),
                "state": str(row["state"]),
                "decision_reason": str(row["decision_reason"]),
                "unknown_reasons": json.dumps(row["unknown_reasons"]),
                "reference_frames": json.dumps(row["reference_frames"]),
            }
            for row in self.base_rows
        ]
        _write_csv(self.input_dir / "slot_decisions.csv", csv_rows)
        self.metadata = {
            "schema_version": "1.0",
            "pipeline": "fixture_hybrid_3d",
            "phase": "full",
        }
        (self.input_dir / "slot_decisions.json").write_text(
            json.dumps({**self.metadata, "decisions": self.base_rows}),
            encoding="utf-8",
        )
        (self.input_dir / "summary.json").write_text(
            json.dumps({**self.metadata, "gt_status": "unavailable"}),
            encoding="utf-8",
        )

        occupied_resolution = _resolution(
            task_id="task-occupied",
            slot_id="slot_0002",
            scope_status="in_route_scope",
            state="occupied",
            reason_codes=["gate_occupied"],
            evidence_refs=[EVIDENCE_ID],
        )
        unknown_resolution = _resolution(
            task_id="task-unknown",
            slot_id="slot_0003",
            scope_status="partial_route_scope",
            state="unknown",
            reason_codes=["free_terminal_support_missing", "terminal_proposal_vetoed"],
            evidence_refs=[],
        )
        final_rows = [dict(row) for row in self.base_rows]
        for row in final_rows:
            if row["slot_id"] == "slot_0002":
                row["state"] = "occupied"
                row["part2_resolution"] = occupied_resolution
            elif row["slot_id"] == "slot_0003":
                row["part2_resolution"] = unknown_resolution
        self.final_payload = {
            "schema_version": "part2-final-route-states/1.0",
            "part2_run_id": HASH_A,
            "queue_id": HASH_B,
            "base_decisions": self.metadata,
            "decisions": final_rows,
        }
        self.final_path = self.root / "final_route_states.json"
        self._write_final()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _write_final(self) -> None:
        self.final_path.write_text(json.dumps(self.final_payload), encoding="utf-8")

    def _write_media(self) -> tuple[Path, Path]:
        media_dir = self.root / "media"
        temporary_png = self.root / "triptych.png"
        Image.new("RGB", (18, 6), (90, 120, 150)).save(temporary_png, format="PNG")
        content = temporary_png.read_bytes()
        sha256 = "sha256:" + hashlib.sha256(content).hexdigest()
        digest = sha256.split(":", 1)[1]
        media_path = media_dir / "sha256" / digest[:2] / f"{digest}.png"
        media_path.parent.mkdir(parents=True)
        media_path.write_bytes(content)
        manifest = {
            "schema_version": "part2-shadow-media-manifest/1.0",
            "queue_id": HASH_B,
            "attempts": [],
            "media": [
                {
                    "evidence_id": EVIDENCE_ID,
                    "slot_id": "slot_0002",
                    "task_id": "task-occupied",
                    "encounter_id": "encounter-a",
                    "kind": "lidar_triptych_png",
                    "renderer_version": "fixture/1.0",
                    "sha256": sha256,
                    "source_sha256": HASH_A,
                    "size_bytes": len(content),
                    "width": 18,
                    "height": 6,
                }
            ],
        }
        manifest["manifest_id"] = canonical_sha256(manifest)
        manifest_path = self.root / "generation_manifest.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        return manifest_path, media_dir

    def _write_blind_records(self, *, include_unknown: bool = True) -> Path:
        records: dict[str, object] = {
            "slot_0002": {
                "state": "occupied",
                "visibility": "clear_full",
                "ownership": "target",
                "finding": "vehicle_or_occupying_object",
                "reason_codes": ["target_vehicle_shape"],
                "confidence": 0.91,
            }
        }
        if include_unknown:
            records["slot_0003"] = {
                "state": "unknown",
                "visibility": "clear_partial",
                "ownership": "shared",
                "finding": "static_structure",
                "reason_codes": ["shared_boundary_structure"],
                "confidence": 0.73,
            }
        path = self.root / "blind_records.json"
        path.write_text(json.dumps(records), encoding="utf-8")
        return path

    def _write_selected_part1(self) -> Path:
        selected_dir = self.root / "selected_part1"
        selected_dir.mkdir()
        with (selected_dir / "known_slot_scope.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=("slot_id", "scope_status"))
            writer.writeheader()
            writer.writerow(
                {"slot_id": "slot_0002", "scope_status": "in_route_scope"}
            )
        base_row = next(row for row in self.base_rows if row["slot_id"] == "slot_0002")
        _write_csv(
            selected_dir / "slot_decisions.csv",
            [
                {
                    "slot_id": str(base_row["slot_id"]),
                    "scope_status": str(base_row["scope_status"]),
                    "state": str(base_row["state"]),
                    "decision_reason": str(base_row["decision_reason"]),
                    "unknown_reasons": json.dumps(base_row["unknown_reasons"]),
                    "reference_frames": json.dumps(base_row["reference_frames"]),
                }
            ],
        )
        (selected_dir / "slot_decisions.json").write_text(
            json.dumps({**self.metadata, "decisions": [base_row]}),
            encoding="utf-8",
        )
        (selected_dir / "summary.json").write_text(
            json.dumps(
                {
                    **self.metadata,
                    "gt_status": "unavailable",
                    "map_total": 1,
                    "in_route_scope": 1,
                    "unknown": 1,
                }
            ),
            encoding="utf-8",
        )

        selected_final = next(
            row for row in self.final_payload["decisions"] if row["slot_id"] == "slot_0002"
        )
        self.final_payload["decisions"] = [selected_final]
        self._write_final()
        return selected_dir

    def _write_selected_part1_with_no_card(self) -> Path:
        selected_dir = self._write_selected_part1()
        selected_rows = [
            row
            for row in self.base_rows
            if row["slot_id"] in {"slot_0002", "slot_0003"}
        ]
        with (selected_dir / "known_slot_scope.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=("slot_id", "scope_status"))
            writer.writeheader()
            writer.writerows(
                {
                    "slot_id": str(row["slot_id"]),
                    "scope_status": str(row["scope_status"]),
                }
                for row in selected_rows
            )
        _write_csv(
            selected_dir / "slot_decisions.csv",
            [
                {
                    "slot_id": str(row["slot_id"]),
                    "scope_status": str(row["scope_status"]),
                    "state": str(row["state"]),
                    "decision_reason": str(row["decision_reason"]),
                    "unknown_reasons": json.dumps(row["unknown_reasons"]),
                    "reference_frames": json.dumps(row["reference_frames"]),
                }
                for row in selected_rows
            ],
        )
        (selected_dir / "slot_decisions.json").write_text(
            json.dumps({**self.metadata, "decisions": selected_rows}), encoding="utf-8"
        )
        (selected_dir / "summary.json").write_text(
            json.dumps(
                {
                    **self.metadata,
                    "gt_status": "unavailable",
                    "map_total": 2,
                    "in_route_scope": 1,
                    "partial_route_scope": 1,
                    "unknown": 2,
                }
            ),
            encoding="utf-8",
        )
        resolved_row = self.final_payload["decisions"][0]
        self.final_payload["decisions"] = [resolved_row, dict(selected_rows[1])]
        self._write_final()
        return selected_dir

    def test_report_overlays_changes_embeds_maps_and_media_without_mutating_sources(self) -> None:
        manifest_path, media_dir = self._write_media()
        blind_records = self._write_blind_records()
        source_bytes = {
            path: path.read_bytes()
            for path in (
                self.input_dir / "known_slot_scope.csv",
                self.input_dir / "slot_decisions.csv",
                self.input_dir / "slot_decisions.json",
                self.input_dir / "summary.json",
                self.final_path,
            )
        }

        result = build_report(
            self.input_dir,
            self.final_path,
            self.slot_database,
            self.output_dir,
            media_manifest=manifest_path,
            media_dir=media_dir,
            blind_records=blind_records,
        )

        self.assertEqual(result["slot_count"], 5)
        self.assertEqual(result["part2_task_count"], 2)
        self.assertEqual(result["evaluated_unknown_count"], 1)
        self.assertEqual(result["resolved_count"], 1)
        self.assertEqual(result["no_card_count"], 0)
        self.assertEqual(result["changed_count"], 1)
        self.assertEqual(result["unresolved_count"], 1)
        self.assertEqual(result["remaining_unknown_count"], 1)
        self.assertEqual(result["remaining_in_route_unknown_count"], 0)
        self.assertEqual(result["remaining_partial_unknown_count"], 1)
        self.assertEqual(result["vetoed_unknown_count"], 1)
        self.assertEqual(result["embedded_media_count"], 1)
        self.assertEqual(result["blind_record_count"], 2)
        self.assertEqual(result["transitions"], {"unknown->occupied": 1})
        self.assertEqual(
            result["counts"],
            {"occupied": 2, "free": 1, "unknown": 0, "partial": 1, "out_of_route": 1},
        )
        self.assertGreater((self.output_dir / "global_map_part2_shadow_full.png").stat().st_size, 1000)
        self.assertEqual(
            Path(result["before_after_map"]).name,
            "global_map_part2_evaluated_before_after.png",
        )
        self.assertGreater(Path(result["before_after_map"]).stat().st_size, 1000)
        self.assertGreater(
            (self.output_dir / "global_map_part2_shadow_route_zoom.png").stat().st_size,
            1000,
        )
        report = (self.output_dir / "global_map_part2_shadow_report.html").read_text(
            encoding="utf-8"
        )
        self.assertIn("SHADOW / NO GT", report)
        self.assertIn("unknown &rarr; occupied", report)
        self.assertIn("free_terminal_support_missing", report)
        self.assertIn("slot_0002", report)
        self.assertIn("Evaluated-unknown semantic review", report)
        self.assertIn("static_structure", report)
        self.assertIn("Blind proposal", report)
        self.assertIn("Validated final", report)
        self.assertIn("Evaluated slots: Part1 before vs Part2 after", report)
        self.assertEqual(report.count("data:image/png;base64,"), 4)
        for source, original in source_bytes.items():
            self.assertEqual(source.read_bytes(), original)

    def test_report_rejects_a_final_row_that_does_not_preserve_part1(self) -> None:
        self.final_payload["decisions"][0]["decision_reason"] = "tampered"
        self._write_final()
        with self.assertRaisesRegex(ValueError, "preserve Part1"):
            build_report(
                self.input_dir,
                self.final_path,
                self.slot_database,
                self.output_dir,
            )

    def test_report_rejects_media_content_identity_mismatch(self) -> None:
        manifest_path, media_dir = self._write_media()
        media_file = next((media_dir / "sha256").glob("*/*.png"))
        media_file.write_bytes(media_file.read_bytes() + b"tamper")
        with self.assertRaisesRegex(ValueError, "media content identity mismatch"):
            build_report(
                self.input_dir,
                self.final_path,
                self.slot_database,
                self.output_dir,
                media_manifest=manifest_path,
                media_dir=media_dir,
            )

    def test_selected_part1_overlays_only_its_resolution_on_the_complete_map(self) -> None:
        selected_dir = self._write_selected_part1()

        result = build_report(
            selected_dir,
            self.final_path,
            self.slot_database,
            self.output_dir,
            map_input_dir=self.input_dir,
        )

        self.assertEqual(result["slot_count"], 5)
        self.assertEqual(result["route_slot_count"], 4)
        self.assertEqual(result["part2_task_count"], 1)
        self.assertEqual(result["evaluated_unknown_count"], 0)
        self.assertEqual(result["resolved_count"], 1)
        self.assertEqual(result["no_card_count"], 0)
        self.assertEqual(result["changed_count"], 1)
        self.assertEqual(result["remaining_unknown_count"], 1)
        self.assertEqual(result["remaining_in_route_unknown_count"], 0)
        self.assertEqual(result["remaining_partial_unknown_count"], 1)
        self.assertEqual(
            result["counts"],
            {"occupied": 2, "free": 1, "unknown": 0, "partial": 1, "out_of_route": 1},
        )
        report = (self.output_dir / "global_map_part2_shadow_report.html").read_text(
            encoding="utf-8"
        )
        self.assertIn("Total map slots", report)
        self.assertIn("Route-scope slots", report)
        self.assertIn("Part2 evaluated", report)
        self.assertIn("Remaining decision unknown", report)
        self.assertIn("1 total = 0 in-route +", report)
        self.assertIn("1 partial", report)

    def test_strict_selected_subset_marks_resolution_unknown_and_no_card_sets(self) -> None:
        selected_dir = self._write_selected_part1_with_no_card()

        result = build_report(
            selected_dir,
            self.final_path,
            self.slot_database,
            self.output_dir,
            map_input_dir=self.input_dir,
        )

        self.assertEqual(result["resolved_count"], 1)
        self.assertEqual(result["evaluated_unknown_count"], 0)
        self.assertEqual(result["no_card_count"], 1)
        report = (self.output_dir / "global_map_part2_shadow_report.html").read_text(
            encoding="utf-8"
        )
        self.assertIn("gray dashed: selected but no Part2 resolution card (1)", report)
        self.assertIn("blue solid: Part2 resolved / changed (1)", report)
        self.assertIn("only the 1 slots with an actual Part2 resolution", report)
        self.assertIn("1 selected/no-card slots remain pale, unnumbered context", report)

    def test_before_after_numbers_only_resolution_slots_not_no_card_context(self) -> None:
        slots = {
            row["slot_id"]: row
            for row in json.loads(self.slot_database.read_text(encoding="utf-8"))["slots"]
        }
        overlay = ShadowOverlay(
            slot_id="slot_0002",
            base_state="unknown",
            final_state="occupied",
            resolution=None,  # type: ignore[arg-type]
        )
        labels: list[str] = []
        original_text = Axes.text

        def record_text(axis: Axes, x: float, y: float, label: str, *args: object, **kwargs: object):
            labels.append(label)
            return original_text(axis, x, y, label, *args, **kwargs)

        output_path = self.root / "before_after_unit.png"
        with patch.object(Axes, "text", new=record_text):
            _draw_evaluated_before_after_map(output_path, slots, [overlay])

        self.assertTrue(output_path.is_file())
        self.assertEqual(labels.count("0002"), 2)
        self.assertNotIn("0003", labels)

    def test_blind_records_require_exact_overlay_coverage_and_strict_fields(self) -> None:
        missing_path = self._write_blind_records(include_unknown=False)
        with self.assertRaisesRegex(ValueError, "cover exactly every Part2 overlay slot"):
            build_report(
                self.input_dir,
                self.final_path,
                self.slot_database,
                self.output_dir,
                blind_records=missing_path,
            )

        records = json.loads(missing_path.read_text(encoding="utf-8"))
        records["slot_0003"] = {
            "state": "unknown",
            "visibility": "clear_partial",
            "ownership": "shared",
            "finding": "unclear",
            "reason_codes": ["ambiguous_shape"],
            "confidence": 0.5,
            "model_name": "forbidden",
        }
        missing_path.write_text(json.dumps(records), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "forbidden provider field"):
            build_report(
                self.input_dir,
                self.final_path,
                self.slot_database,
                self.output_dir,
                blind_records=missing_path,
            )

    def test_cli_accepts_blind_records_and_reports_provenance_counts(self) -> None:
        blind_records = self._write_blind_records()
        completed = subprocess.run(
            [
                sys.executable,
                "scripts/build_part2_shadow_global_map_report.py",
                "--input-dir",
                str(self.input_dir),
                "--final-route-states",
                str(self.final_path),
                "--slot-database",
                str(self.slot_database),
                "--output-dir",
                str(self.output_dir),
                "--blind-records",
                str(blind_records),
            ],
            cwd=Path(__file__).resolve().parents[1],
            check=True,
            capture_output=True,
            text=True,
        )
        result = json.loads(completed.stdout)
        self.assertEqual(result["evaluated_unknown_count"], 1)
        self.assertEqual(result["resolved_count"], 1)
        self.assertEqual(result["no_card_count"], 0)
        self.assertEqual(result["blind_record_count"], 2)
        report = Path(result["report"]).read_text(encoding="utf-8")
        self.assertIn("vehicle_or_occupying_object", report)
        self.assertIn("static_structure", report)

    def test_selected_part1_must_match_the_complete_map_base(self) -> None:
        selected_dir = self._write_selected_part1()
        with (self.input_dir / "slot_decisions.csv").open(
            newline="", encoding="utf-8"
        ) as handle:
            rows = list(csv.DictReader(handle))
        for row in rows:
            if row["slot_id"] == "slot_0002":
                row["decision_reason"] = "different_full_map_reason"
        _write_csv(self.input_dir / "slot_decisions.csv", rows)

        with self.assertRaisesRegex(ValueError, "does not match full-map base"):
            build_report(
                selected_dir,
                self.final_path,
                self.slot_database,
                self.output_dir,
                map_input_dir=self.input_dir,
            )


if __name__ == "__main__":
    unittest.main()
