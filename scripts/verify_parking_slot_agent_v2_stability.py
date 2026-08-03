#!/usr/bin/env python3
"""连续重放 Part2 并验证跨目录决策一致性和同目录字节确定性。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--actions", type=Path, required=True)
    parser.add_argument("--original-result", type=Path, required=True)
    parser.add_argument("--replay-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def decision_projection(payload: dict) -> list[tuple]:
    return [
        (
            row["case"]["slot"]["slot_id"],
            row["case"]["final_state"],
            row["case"]["final_scores"]["free_confidence"],
            row["case"]["final_scores"]["occupied_confidence"],
            row["stop_reason"],
            row["model_turns"],
            row["tool_rounds"],
        )
        for row in payload["slot_results"]
    ]


def run_once(args: argparse.Namespace) -> str:
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "parking_slot_agent_v2",
            "run-part2",
            "--input",
            str(args.input.resolve()),
            "--output-dir",
            str(args.replay_dir.resolve()),
            "--replay-actions",
            str(args.actions.resolve()),
        ],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def main() -> None:
    args = parse_args()
    first_stdout = run_once(args)
    replay_result = args.replay_dir.resolve() / "part2_result.json"
    first_hash = sha256(replay_result)
    second_stdout = run_once(args)
    second_hash = sha256(replay_result)
    original = load(args.original_result.resolve())
    replay = load(replay_result)
    result = {
        "schema_version": "parking-slot-agent-v2-stability-verification/1.0",
        "same_directory_byte_deterministic": first_hash == second_hash,
        "first_sha256": first_hash,
        "second_sha256": second_hash,
        "cross_directory_decision_projection_identical": (
            decision_projection(original) == decision_projection(replay)
        ),
        "queue_identical": original["queue_order"] == replay["queue_order"],
        "selected_slot_identical": (
            original["selected_slot_id"] == replay["selected_slot_id"]
        ),
        "processed_case_count": len(replay["processed_case_ids"]),
        "first_stdout": first_stdout,
        "second_stdout": second_stdout,
        "note": (
            "The original slot_1250 API PermissionDeniedError is replayed as an "
            "exhausted-action RuntimeError; the fail-closed decision projection is identical."
        ),
    }
    result["ok"] = all(
        result[key]
        for key in (
            "same_directory_byte_deterministic",
            "cross_directory_decision_projection_identical",
            "queue_identical",
            "selected_slot_identical",
        )
    )
    args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.output.resolve().write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
