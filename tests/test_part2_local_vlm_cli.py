from __future__ import annotations

import copy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest

from parking_slot_part2.grouping import build_groups
from parking_slot_part2.preflight import EvidenceCatalog
from parking_slot_part2.queueing import (
    canonical_json_bytes,
    load_queue,
    make_queue_id,
    make_task_id,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLI = PROJECT_ROOT / "scripts" / "run_part2_agent.py"
FIXTURE_ROOT = PROJECT_ROOT / "tests" / "fixtures" / "part2"


class _VisionEndpoint(BaseHTTPRequestHandler):
    actions: list[dict[str, object]] = []
    requests: list[dict[str, object]] = []

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        size = int(self.headers.get("content-length", "0"))
        payload = json.loads(self.rfile.read(size))
        type(self).requests.append(payload)
        if not type(self).actions:
            self.send_error(500)
            return
        action = type(self).actions.pop(0)
        content = json.dumps(
            {"choices": [{"message": {"content": json.dumps(action)}}]}
        ).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def log_message(self, format: str, *args: object) -> None:
        return


def _local_queue(path: Path) -> tuple[object, str, str, str]:
    payload = json.loads((FIXTURE_ROOT / "valid_queue.json").read_text())
    for resource in payload["resources"].values():
        resource["uri"] = (FIXTURE_ROOT / resource["uri"]).resolve().as_uri()
    quality = payload["items"][0]["encounter"]["visual_frames"][0][
        "projection_quality"
    ]
    quality.update(
        {
            "image_width_px": 2,
            "image_height_px": 2,
            "polygon_uv": [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
            "adjacent_polygons_uv": {},
        }
    )
    payload["queue_id"] = make_queue_id(payload)
    path.write_bytes(canonical_json_bytes(payload) + b"\n")
    queue = load_queue(path)
    catalog = EvidenceCatalog(queue, base_dir=path.parent)
    preflight = catalog.camera_preflight(queue.items[0].task_id)
    return (
        queue,
        build_groups(queue)[0].group_id,
        queue.items[0].task_id,
        preflight.frame_evidence_ids[0],
    )


class LocalVLMCommandLineTest(unittest.TestCase):
    def test_local_endpoint_requests_rgb_tool_then_receives_annotated_image(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            queue_path = root / "queue.json"
            queue, group_id, task_id, evidence_id = _local_queue(queue_path)
            item = queue.items[0]
            _VisionEndpoint.requests = []
            _VisionEndpoint.actions = [
                {
                    "type": "tool_request",
                    "tool_name": "inspect_rgb_frame",
                    "arguments": {"evidence_id": evidence_id},
                },
                {
                    "type": "final_proposal",
                    "assessments": [
                        {
                            "task_id": task_id,
                            "slot_id": item.slot_id,
                            "proposed_state": "occupied",
                            "target_visibility": "clear_full",
                            "target_ownership": "target",
                            "semantic_finding": "vehicle_or_occupying_object",
                            "resolved_unknown_reasons": list(item.unknown_reasons),
                            "unresolved_blockers": [],
                            "evidence_refs": [evidence_id],
                            "reason_codes": ["rgb_target_vehicle"],
                        }
                    ],
                },
            ]
            server = ThreadingHTTPServer(("127.0.0.1", 0), _VisionEndpoint)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            self.addCleanup(server.server_close)
            self.addCleanup(server.shutdown)

            output_dir = root / "output"
            media_dir = root / "media"
            replay_path = root / "live_replay.json"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(CLI),
                    "run-local-vlm",
                    "--queue",
                    str(queue_path),
                    "--base-url",
                    f"http://127.0.0.1:{server.server_port}/v1",
                    "--model",
                    "mock-local-vision",
                    "--output-dir",
                    str(output_dir),
                    "--media-dir",
                    str(media_dir),
                    "--replay-output",
                    str(replay_path),
                ],
                cwd=PROJECT_ROOT,
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            result = json.loads(completed.stdout)
            self.assertEqual(result["state_counts"], {"free": 0, "occupied": 1, "unknown": 0})
            self.assertEqual(result["tool_attempt_count"], 1)
            self.assertEqual(result["media_count"], 1)
            self.assertEqual(result["media_manifest"], str(media_dir / "generation_manifest.json"))
            self.assertEqual(len(_VisionEndpoint.requests), 2)
            first_parts = _VisionEndpoint.requests[0]["messages"][1]["content"]
            second_parts = _VisionEndpoint.requests[1]["messages"][1]["content"]
            self.assertFalse(any(part["type"] == "image_url" for part in first_parts))
            images = [part for part in second_parts if part["type"] == "image_url"]
            self.assertEqual(len(images), 1)
            self.assertTrue(images[0]["image_url"]["url"].startswith("data:image/png;base64,"))
            replay = json.loads(replay_path.read_text())
            self.assertEqual(len(replay["actions"][group_id]), 2)
            media_manifest = json.loads((media_dir / "generation_manifest.json").read_text())
            self.assertEqual(media_manifest["queue_id"], queue.queue_id)
            self.assertEqual(media_manifest["media"][0]["evidence_id"], evidence_id)
            self.assertNotIn("path", json.dumps(media_manifest).lower())
            self.assertEqual(
                json.loads((output_dir / "part2_resolutions.json").read_text())[
                    "resolutions"
                ][0]["state"],
                "occupied",
            )

            replay_output_dir = root / "replay-output"
            replayed = subprocess.run(
                [
                    sys.executable,
                    str(CLI),
                    "run",
                    "--queue",
                    str(queue_path),
                    "--replay-actions",
                    str(replay_path),
                    "--output-dir",
                    str(replay_output_dir),
                ],
                cwd=PROJECT_ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(replayed.returncode, 0, replayed.stderr)
            replay_result = json.loads(replayed.stdout)
            self.assertEqual(replay_result["part2_run_id"], result["part2_run_id"])
            live_files = {
                path.relative_to(output_dir): path.read_bytes()
                for path in output_dir.rglob("*")
                if path.is_file()
            }
            replay_files = {
                path.relative_to(replay_output_dir): path.read_bytes()
                for path in replay_output_dir.rglob("*")
                if path.is_file()
            }
            self.assertEqual(replay_files, live_files)

    def test_local_model_error_refuses_standard_report_and_replay(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            queue_path = root / "queue.json"
            _local_queue(queue_path)
            payload = json.loads(queue_path.read_text())
            second = copy.deepcopy(payload["items"][0])
            second["slot_id"] = "slot-agent-second"
            second["encounter"]["encounter_id"] = "encounter-agent-2"
            second["encounter"]["visual_frames"][0][
                "visual_frame_id"
            ] = "visual-agent-second"
            second["encounter"]["visual_frames"][0]["projection_quality"][
                "encounter_id"
            ] = "encounter-agent-2"
            second["audit"]["part1_decision_id"] = "part1-decision-slot-agent-second"
            second["relationships"] = {
                "adjacent_slot_ids": [],
                "conflict_slot_ids": [],
                "shared_evidence_ids": [],
            }
            second["task_id"] = make_task_id(
                payload["producer"],
                second["slot_id"],
                second["encounter"]["encounter_id"],
            )
            payload["items"].append(second)
            payload["queue_id"] = make_queue_id(payload)
            queue_path.write_bytes(canonical_json_bytes(payload) + b"\n")
            self.assertEqual(len(build_groups(load_queue(queue_path))), 2)
            _VisionEndpoint.requests = []
            _VisionEndpoint.actions = []
            server = ThreadingHTTPServer(("127.0.0.1", 0), _VisionEndpoint)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            self.addCleanup(server.server_close)
            self.addCleanup(server.shutdown)

            output_dir = root / "output"
            replay_path = root / "failed-replay.json"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(CLI),
                    "run-local-vlm",
                    "--queue",
                    str(queue_path),
                    "--base-url",
                    f"http://127.0.0.1:{server.server_port}/v1",
                    "--model",
                    "failing-local-vision",
                    "--output-dir",
                    str(output_dir),
                    "--media-dir",
                    str(root / "media"),
                    "--replay-output",
                    str(replay_path),
                ],
                cwd=PROJECT_ROOT,
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(completed.returncode, 1)
            self.assertIn("local model failed", completed.stderr)
            self.assertEqual(len(_VisionEndpoint.requests), 1)
            self.assertFalse(replay_path.exists())
            for name in (
                "part2_resolutions.json",
                "final_route_states.json",
                "decision_trace.jsonl",
                "summary.json",
                "report.html",
                "run_manifest.json",
            ):
                self.assertFalse((output_dir / name).exists(), name)

    def test_media_publication_failure_cannot_create_a_nonreplayable_report(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            queue_path = root / "queue.json"
            queue, _, task_id, evidence_id = _local_queue(queue_path)
            item = queue.items[0]
            _VisionEndpoint.requests = []
            _VisionEndpoint.actions = [
                {
                    "type": "tool_request",
                    "tool_name": "inspect_rgb_frame",
                    "arguments": {"evidence_id": evidence_id},
                },
                {
                    "type": "final_proposal",
                    "assessments": [
                        {
                            "task_id": task_id,
                            "slot_id": item.slot_id,
                            "proposed_state": "unknown",
                            "target_visibility": "unavailable",
                            "target_ownership": "uncertain",
                            "semantic_finding": "unclear",
                            "resolved_unknown_reasons": [],
                            "unresolved_blockers": list(item.unknown_reasons),
                            "evidence_refs": [],
                            "reason_codes": ["media_cache_unavailable"],
                        }
                    ],
                },
            ]
            server = ThreadingHTTPServer(("127.0.0.1", 0), _VisionEndpoint)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            self.addCleanup(server.server_close)
            self.addCleanup(server.shutdown)

            blocked_media = root / "blocked-media"
            blocked_media.write_bytes(b"not a directory")
            output_dir = root / "output"
            replay_path = root / "nonreplayable.json"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(CLI),
                    "run-local-vlm",
                    "--queue",
                    str(queue_path),
                    "--base-url",
                    f"http://127.0.0.1:{server.server_port}/v1",
                    "--model",
                    "mock-local-vision",
                    "--output-dir",
                    str(output_dir),
                    "--media-dir",
                    str(blocked_media),
                    "--replay-output",
                    str(replay_path),
                ],
                cwd=PROJECT_ROOT,
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(completed.returncode, 1)
            self.assertIn("not exactly reproducible", completed.stderr)
            self.assertEqual(len(_VisionEndpoint.requests), 2)
            self.assertFalse(replay_path.exists())
            self.assertFalse((output_dir / "run_manifest.json").exists())

    def test_cli_rejects_non_loopback_endpoint_before_network_or_output(self) -> None:
        completed = subprocess.run(
            [
                sys.executable,
                str(CLI),
                "run-local-vlm",
                "--queue",
                str(FIXTURE_ROOT / "valid_queue.json"),
                "--base-url",
                "https://example.com/v1",
                "--model",
                "must-not-run",
            ],
            cwd=PROJECT_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 1)
        self.assertIn("loopback", completed.stderr)

    def test_cli_rejects_overlapping_private_and_standard_paths_before_network(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            queue_path = root / "queue.json"
            queue, _, _, _ = _local_queue(queue_path)
            resource_path = Path(next(iter(queue.resources.values())).uri.removeprefix("file://"))
            cases = (
                (
                    root / "standard",
                    root / "standard" / "media",
                    root / "replay.json",
                    "disjoint",
                ),
                (
                    root / "standard-2",
                    root / "private-media",
                    root / "private-media" / "generation_manifest.json",
                    "outside output_dir and media_dir",
                ),
                (
                    root / "standard-3",
                    root / "private-media-3",
                    queue_path,
                    "queue input",
                ),
                (
                    root / "standard-4",
                    root / "private-media-4",
                    resource_path,
                    "queue resource",
                ),
            )
            for output_dir, media_dir, replay_path, expected in cases:
                with self.subTest(expected=expected):
                    completed = subprocess.run(
                        [
                            sys.executable,
                            str(CLI),
                            "run-local-vlm",
                            "--queue",
                            str(queue_path),
                            "--base-url",
                            "http://127.0.0.1:1/v1",
                            "--model",
                            "must-not-connect",
                            "--output-dir",
                            str(output_dir),
                            "--media-dir",
                            str(media_dir),
                            "--replay-output",
                            str(replay_path),
                        ],
                        cwd=PROJECT_ROOT,
                        text=True,
                        capture_output=True,
                        check=False,
                    )
                    self.assertEqual(completed.returncode, 1)
                    self.assertIn(expected, completed.stderr)


if __name__ == "__main__":
    unittest.main()
