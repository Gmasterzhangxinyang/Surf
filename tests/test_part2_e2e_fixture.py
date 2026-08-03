from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from parking_slot_part2.grouping import build_groups
from parking_slot_part2.preflight import EvidenceCatalog
from parking_slot_part2.queueing import (
    canonical_json_bytes,
    canonical_sha256,
    load_queue,
    make_queue_id,
    make_task_id,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLI = PROJECT_ROOT / "scripts" / "run_part2_agent.py"
FIXTURE_ROOT = PROJECT_ROOT / "tests" / "fixtures" / "part2"
QUEUE_PATH = FIXTURE_ROOT / "valid_queue.json"
REPLAY_PATH = FIXTURE_ROOT / "replay_actions.json"


def _load_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise AssertionError(f"{path} must contain a JSON object")
    return payload


def _run_cli(*arguments: object) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(CLI), *(str(argument) for argument in arguments)],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def _tree_bytes(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


class StaticPart2EndToEndFixtureTest(unittest.TestCase):
    def test_render_shadow_media_writes_manifest_and_returns_summary(self) -> None:
        queue = load_queue(QUEUE_PATH)
        with tempfile.TemporaryDirectory() as directory:
            media_dir = Path(directory) / "media"
            completed = _run_cli(
                "render-shadow-media",
                "--queue",
                QUEUE_PATH,
                "--media-dir",
                media_dir,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            summary = json.loads(completed.stdout)
            manifest_path = media_dir / "generation_manifest.json"
            self.assertEqual(
                summary,
                {
                    "schema_version": "part2-shadow-media-result/1.0",
                    "queue_id": queue.queue_id,
                    "lidar_attempt_count": 1,
                    "media_count": 0,
                    "failed_count": 1,
                    "manifest": str(manifest_path),
                },
            )
            self.assertTrue(manifest_path.is_file())
            manifest = _load_json(manifest_path)
            self.assertEqual(
                manifest_path.read_bytes(),
                canonical_json_bytes(manifest) + b"\n",
            )
            manifest_without_id = dict(manifest)
            manifest_id = manifest_without_id.pop("manifest_id")
            self.assertEqual(manifest_id, canonical_sha256(manifest_without_id))
            self.assertEqual(manifest["queue_id"], queue.queue_id)
            self.assertEqual(len(manifest["attempts"]), 1)
            self.assertEqual(manifest["attempts"][0]["status"], "unavailable")
            self.assertEqual(
                manifest["attempts"][0]["error_code"],
                "unsupported_lidar_evidence_format",
            )
            self.assertFalse(manifest["attempts"][0]["media_published"])
            self.assertEqual(manifest["media"], [])

    def test_static_fixture_exercises_the_complete_deterministic_cli_contract(self) -> None:
        self.assertTrue(QUEUE_PATH.is_file(), "missing static Part2 queue fixture")
        self.assertTrue(REPLAY_PATH.is_file(), "missing static Part2 replay fixture")

        raw_queue = _load_json(QUEUE_PATH)
        raw_replay = _load_json(REPLAY_PATH)
        self.assertEqual(
            QUEUE_PATH.read_bytes(),
            canonical_json_bytes(raw_queue) + b"\n",
        )
        self.assertEqual(
            REPLAY_PATH.read_bytes(),
            canonical_json_bytes(raw_replay) + b"\n",
        )
        self.assertEqual(raw_queue["queue_id"], make_queue_id(raw_queue))
        queue = load_queue(QUEUE_PATH)
        self.assertEqual(len(queue.items), 1)
        item = queue.items[0]
        self.assertEqual(
            item.task_id,
            make_task_id(queue.producer, item.slot_id, item.encounter.encounter_id),
        )
        map_manifest_path = FIXTURE_ROOT / "artifacts" / "map_points" / "manifest.json"
        self.assertEqual(
            queue.resources["map-points-root"].manifest_hash,
            "sha256:" + hashlib.sha256(map_manifest_path.read_bytes()).hexdigest(),
        )
        slot_database_path = FIXTURE_ROOT / "artifacts" / "slot_database.json"
        self.assertEqual(
            queue.producer["slot_map_hash"],
            "sha256:" + hashlib.sha256(slot_database_path.read_bytes()).hexdigest(),
        )
        visual = item.encounter.visual_frames[0]
        self.assertNotEqual(visual.lidar_frame, visual.camera_frame)
        self.assertLessEqual(abs(visual.camera_lidar_dt_sec), 0.04)
        self.assertLessEqual(visual.camera_timestamp, item.encounter.anchor_timestamp)

        catalog = EvidenceCatalog(queue, base_dir=FIXTURE_ROOT)
        preflight = catalog.camera_preflight(item.task_id)
        self.assertEqual(preflight.status, "ready")
        self.assertEqual(
            preflight.capabilities,
            ("can_assess_free", "can_assess_occupied"),
        )
        self.assertEqual(len(preflight.frame_evidence_ids), 1)
        groups = build_groups(queue)
        self.assertEqual(len(groups), 1)
        self.assertEqual(set(raw_replay), {"schema_version", "actions"})
        self.assertEqual(raw_replay["schema_version"], "part2-replay-actions/1.0")
        self.assertEqual(set(raw_replay["actions"]), {groups[0].group_id})
        replay_actions = raw_replay["actions"][groups[0].group_id]
        self.assertEqual(replay_actions[0]["type"], "tool_request")
        self.assertEqual(replay_actions[0]["tool_name"], "inspect_rgb_frame")
        self.assertEqual(
            replay_actions[0]["arguments"],
            {"evidence_id": preflight.frame_evidence_ids[0]},
        )
        fixture_input_bytes = QUEUE_PATH.read_bytes() + REPLAY_PATH.read_bytes()
        self.assertNotIn(b'"provider"', fixture_input_bytes.lower())
        self.assertNotIn(b"depth", fixture_input_bytes.lower())

        validation = _run_cli("validate-queue", "--queue", QUEUE_PATH)
        self.assertEqual(validation.returncode, 0, validation.stderr)
        validation_summary = json.loads(validation.stdout)
        self.assertEqual(
            set(validation_summary),
            {
                "schema_version",
                "queue_id",
                "item_count",
                "resource_count",
                "group_count",
                "scope_counts",
            },
        )
        self.assertEqual(
            validation_summary["schema_version"],
            "part2-queue-validation-summary/1.0",
        )
        self.assertEqual(validation_summary["queue_id"], queue.queue_id)
        self.assertEqual(validation_summary["item_count"], 1)
        self.assertEqual(validation_summary["group_count"], 1)

        with tempfile.TemporaryDirectory() as directory:
            temporary_root = Path(directory)
            output_dirs = (temporary_root / "run-a", temporary_root / "run-b")
            run_summaries: list[dict[str, object]] = []
            for output_dir in output_dirs:
                completed = _run_cli(
                    "run",
                    "--queue",
                    QUEUE_PATH,
                    "--replay-actions",
                    REPLAY_PATH,
                    "--output-dir",
                    output_dir,
                )
                self.assertEqual(completed.returncode, 0, completed.stderr)
                run_summary = json.loads(completed.stdout)
                self.assertEqual(
                    set(run_summary),
                    {
                        "schema_version",
                        "part2_run_id",
                        "queue_id",
                        "resolution_count",
                        "state_counts",
                    },
                )
                self.assertEqual(run_summary["schema_version"], "part2-run-result/1.0")
                self.assertEqual(run_summary["queue_id"], queue.queue_id)
                self.assertEqual(run_summary["resolution_count"], len(queue.items))
                run_summaries.append(run_summary)

            first_tree = _tree_bytes(output_dirs[0])
            self.assertEqual(first_tree, _tree_bytes(output_dirs[1]))
            self.assertEqual(run_summaries[0], run_summaries[1])
            attempt_names = {
                name for name in first_tree if name.startswith("tool_artifacts/")
            }
            self.assertEqual(len(attempt_names), 1)
            self.assertEqual(
                set(first_tree),
                {
                    "part2_resolutions.json",
                    "final_route_states.json",
                    "decision_trace.jsonl",
                    "summary.json",
                    "report.html",
                    "run_manifest.json",
                    next(iter(attempt_names)),
                },
            )

            resolutions = _load_json(output_dirs[0] / "part2_resolutions.json")
            self.assertEqual(
                set(resolutions),
                {"schema_version", "part2_run_id", "queue_id", "resolutions"},
            )
            self.assertEqual(resolutions["schema_version"], "part2-resolutions/1.0")
            resolution_rows = resolutions["resolutions"]
            self.assertEqual(
                {row["task_id"] for row in resolution_rows},
                {queued.task_id for queued in queue.items},
            )
            self.assertEqual(len(resolution_rows), len(queue.items))
            resolution = resolution_rows[0]
            self.assertEqual(
                set(resolution),
                {
                    "task_id",
                    "slot_id",
                    "scope_status",
                    "input_state",
                    "state",
                    "decision_source",
                    "model_turns",
                    "tool_calls_attempted",
                    "stop_reasons",
                    "completion_status",
                    "group_ids",
                    "trace_span_ids",
                    "attempt_ids",
                    "evidence_refs",
                    "reason_codes",
                },
            )
            self.assertEqual(resolution["input_state"], "unknown")
            self.assertEqual(resolution["state"], "occupied")
            self.assertEqual(resolution["decision_source"], "part2_agent_validated")
            self.assertEqual(resolution["completion_status"], "resolved")
            self.assertEqual(resolution["tool_calls_attempted"], 1)
            self.assertEqual(resolution["evidence_refs"], [preflight.frame_evidence_ids[0]])

            base_path = FIXTURE_ROOT / "artifacts" / "slot_decisions.json"
            base = _load_json(base_path)
            base_rows = {row["slot_id"]: row for row in base["decisions"]}
            final = _load_json(output_dirs[0] / "final_route_states.json")
            self.assertEqual(
                set(final),
                {
                    "schema_version",
                    "part2_run_id",
                    "queue_id",
                    "base_decisions",
                    "decisions",
                },
            )
            self.assertEqual(final["schema_version"], "part2-final-route-states/1.0")
            final_rows = {row["slot_id"]: row for row in final["decisions"]}
            self.assertEqual(final_rows["slot-terminal"], base_rows["slot-terminal"])
            self.assertEqual(final_rows["slot-nonqueued"], base_rows["slot-nonqueued"])
            self.assertNotIn("slot-outside", final_rows)
            queued_before = base_rows[item.slot_id]
            queued_after = final_rows[item.slot_id]
            self.assertEqual(queued_before["state"], "unknown")
            self.assertEqual(queued_after["state"], "occupied")
            self.assertEqual(
                {
                    key: value
                    for key, value in queued_after.items()
                    if key not in {"state", "part2_resolution"}
                },
                {key: value for key, value in queued_before.items() if key != "state"},
            )
            self.assertEqual(queued_after["part2_resolution"], resolution)

            trace_lines = (output_dirs[0] / "decision_trace.jsonl").read_text(
                encoding="utf-8"
            ).splitlines()
            self.assertTrue(trace_lines)
            trace = [json.loads(line) for line in trace_lines]
            self.assertTrue(
                all(
                    set(event)
                    == {
                        "schema_version",
                        "trace_span_id",
                        "sequence",
                        "category",
                        "event_type",
                        "data",
                        "event_id",
                    }
                    for event in trace
                )
            )
            self.assertTrue(
                all(
                    event["schema_version"] == "part2-evidence-ledger/1.0"
                    for event in trace
                )
            )
            self.assertEqual(
                [event["sequence"] for event in trace],
                list(range(1, len(trace) + 1)),
            )
            self.assertEqual(len({event["event_id"] for event in trace}), len(trace))
            self.assertEqual(
                {event["category"] for event in trace},
                {"lifecycle", "model", "tool", "validation", "stop"},
            )
            self.assertTrue(
                {
                    "group_started",
                    "model_turn_started",
                    "model_action_received",
                    "tool_attempted",
                    "tool_result",
                    "final_proposal_accepted",
                    "group_finished",
                    "group_stopped",
                }
                <= {event["event_type"] for event in trace}
            )

            attempt_paths = sorted((output_dirs[0] / "tool_artifacts").glob("*.json"))
            self.assertEqual(len(attempt_paths), 1)
            attempt_artifact = _load_json(attempt_paths[0])
            self.assertEqual(
                set(attempt_artifact),
                {"schema_version", "part2_run_id", "group_id", "trace_span_id", "attempt"},
            )
            self.assertEqual(attempt_artifact["schema_version"], "part2-tool-attempt/1.0")
            attempt = attempt_artifact["attempt"]
            self.assertEqual(
                set(attempt),
                {
                    "attempt_id",
                    "attempt_index",
                    "model_turn",
                    "tool_name",
                    "arguments",
                    "evidence_id",
                    "disposition",
                    "executed",
                    "result_status",
                    "error_code",
                    "error_message",
                    "data",
                },
            )
            self.assertEqual(attempt["tool_name"], "inspect_rgb_frame")
            self.assertEqual(attempt["arguments"], replay_actions[0]["arguments"])
            self.assertTrue(attempt["executed"])
            self.assertEqual(attempt["result_status"], "ok")
            self.assertFalse(attempt["data"]["semantic_inference_performed"])
            self.assertEqual(
                attempt["data"]["artifacts"][0]["declared_sha256"],
                attempt["data"]["artifacts"][0]["verified_sha256"],
            )

            summary = _load_json(output_dirs[0] / "summary.json")
            self.assertEqual(
                set(summary),
                {
                    "schema_version",
                    "part2_run_id",
                    "queue_id",
                    "queue_task_count",
                    "group_count",
                    "resolution_count",
                    "state_counts",
                    "completion_counts",
                    "model_turns",
                    "tool_calls_attempted",
                    "trace_event_count",
                },
            )
            self.assertEqual(summary["schema_version"], "part2-run-summary/1.0")
            self.assertEqual(summary["queue_task_count"], 1)
            self.assertEqual(summary["resolution_count"], 1)
            self.assertEqual(
                summary["state_counts"],
                {"free": 0, "occupied": 1, "unknown": 0},
            )
            self.assertEqual(
                summary["completion_counts"],
                {"remained_unknown": 0, "resolved": 1},
            )
            self.assertEqual(summary["tool_calls_attempted"], 1)
            self.assertEqual(summary["trace_event_count"], len(trace))

            report = (output_dirs[0] / "report.html").read_text(encoding="utf-8")
            self.assertTrue(report.startswith("<!doctype html>\n"))
            self.assertIn("Part2 Unknown Agent v1", report)
            self.assertIn(item.task_id, report)
            self.assertIn(item.slot_id, report)
            self.assertIn(str(summary["part2_run_id"]), report)

            manifest = _load_json(output_dirs[0] / "run_manifest.json")
            self.assertEqual(
                set(manifest),
                {"schema_version", "part2_run_id", "inputs", "policies", "artifacts"},
            )
            self.assertEqual(manifest["schema_version"], "part2-run-manifest/1.0")
            self.assertEqual(
                manifest["inputs"],
                {
                    "queue_identity": queue.queue_id,
                    "replay_identity": canonical_sha256(raw_replay),
                },
            )
            manifest_rows = manifest["artifacts"]
            manifest_paths = [row["path"] for row in manifest_rows]
            self.assertEqual(manifest_paths, sorted(manifest_paths))
            self.assertNotIn("run_manifest.json", manifest_paths)
            self.assertEqual(set(manifest_paths), set(first_tree) - {"run_manifest.json"})
            for row in manifest_rows:
                self.assertEqual(set(row), {"path", "sha256", "size_bytes"})
                content = first_tree[row["path"]]
                self.assertEqual(row["size_bytes"], len(content))
                self.assertEqual(
                    row["sha256"],
                    "sha256:" + hashlib.sha256(content).hexdigest(),
                )

            all_output = b"\n".join(first_tree.values())
            self.assertNotIn(str(FIXTURE_ROOT).encode("utf-8"), all_output)
            self.assertNotIn(str(temporary_root).encode("utf-8"), all_output)
            self.assertNotIn(b'"provider"', all_output.lower())


if __name__ == "__main__":
    unittest.main()
