#!/usr/bin/env python3
"""Run the calibrated Camera-first ParkingAgent against a local Qwen VLM."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

from parking_slot_agent_v2.agent import SingleSlotAgent
from parking_slot_agent_v2.calibrated_tools import CalibratedCameraToolSuite
from parking_slot_agent_v2.contracts import Part1Output
from parking_slot_agent_v2.model import LocalVLMAdapter, ReplayModelAdapter
from parking_slot_agent_v2.pipeline import run_candidate_queue
from scripts.run_calibrated_sheet_agent import (
    calibrated_prompt,
    plan_execute_observe_trace,
    semantic,
    write,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--camera-sheets", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8010/v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    parser.add_argument("--max-output-tokens", type=int, default=1536)
    args = parser.parse_args()

    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)

    part1 = Part1Output.from_dict(
        json.loads(args.input.read_text(encoding="utf-8"))
    )
    manifest = (args.camera_sheets / "manifest.json").resolve(strict=False)
    attached: list[str] = []
    for case in part1.slot_cases:
        sheet = (args.camera_sheets / f"{case.slot_id}.jpg").resolve(strict=False)
        if not sheet.is_file():
            continue
        case.resources["calibrated_camera_sheet_path"] = str(sheet)
        if manifest.is_file():
            case.resources["calibrated_camera_manifest_path"] = str(manifest)
        depth = (args.camera_sheets / f"{case.slot_id}.depth.json").resolve(
            strict=False
        )
        if depth.is_file():
            case.resources["calibrated_camera_depth_metadata_path"] = str(depth)
        attached.append(case.slot_id)

    write(args.output_dir / "part1_input_with_calibrated_sheets.json", part1.to_dict())
    prompt = calibrated_prompt()
    (args.output_dir / "system_prompt.txt").write_text(
        prompt + "\n", encoding="utf-8"
    )
    adapter = LocalVLMAdapter(
        base_url=args.base_url,
        model=args.model,
        timeout_seconds=args.timeout_seconds,
        max_output_tokens=args.max_output_tokens,
        system_prompt=prompt,
    )

    started = time.perf_counter()
    live = run_candidate_queue(
        part1,
        SingleSlotAgent(
            adapter,
            CalibratedCameraToolSuite(map_radius_m=part1.scene.radius_m),
        ),
        args.output_dir / "live",
        stop_at_first_free=False,
        resume=False,
    )
    wall_seconds = time.perf_counter() - started
    write(args.output_dir / "local_model_calls.json", adapter.calls)
    write(args.output_dir / "local_model_invalid_calls.json", adapter.invalid_calls)
    write(
        args.output_dir / "agent_plan_execute_observe.json",
        plan_execute_observe_trace(live, adapter.calls),
    )

    replay_payload = adapter.replay_payload()
    replay_payload.setdefault("evidence_bindings", {})
    for result in live.slot_results:
        case_id = result.case.case_id
        sequence = replay_payload["actions"].setdefault(case_id, [])
        bindings = replay_payload["evidence_bindings"].setdefault(case_id, [])
        missing = max(0, result.model_turns - len(sequence))
        for _ in range(missing):
            sequence.append(
                {
                    "type": "replay_error",
                    "error_type": "ValueError",
                    "message": "recorded invalid local structured action",
                }
            )
            bindings.append([])
        while len(bindings) < len(sequence):
            bindings.append([])
    write(args.output_dir / "replay_actions.json", replay_payload)

    replay_input = Part1Output.from_dict(
        json.loads(
            (args.output_dir / "part1_input_with_calibrated_sheets.json").read_text(
                encoding="utf-8"
            )
        )
    )
    replay = run_candidate_queue(
        replay_input,
        SingleSlotAgent(
            ReplayModelAdapter.from_path(args.output_dir / "replay_actions.json"),
            CalibratedCameraToolSuite(map_radius_m=replay_input.scene.radius_m),
        ),
        args.output_dir / "replay",
        stop_at_first_free=False,
        resume=False,
    )
    live_semantic, replay_semantic = semantic(live), semantic(replay)
    verified = live_semantic == replay_semantic
    write(
        args.output_dir / "replay_verification.json",
        {
            "verified": verified,
            "live": live_semantic,
            "replay": replay_semantic,
        },
    )
    if not verified:
        raise RuntimeError("Replay mismatch")

    states = Counter(row["final_state"] for row in live_semantic["slots"])
    tool_counts = Counter(
        evidence.tool_name
        for result in live.slot_results
        for evidence in result.case.evidence
        if evidence.round_index >= 1 and evidence.tool_name != "agent_final"
    )
    call_seconds = [float(row["elapsed_seconds"]) for row in adapter.calls]
    summary = {
        "schema_version": "parking-agent-local-qwen-ablation/1.0",
        "architecture": "same bounded Plan-Execute-Observe Agent and tools",
        "prediction_blind": True,
        "gt_used": False,
        "model": args.model,
        "base_url": args.base_url,
        "processed": len(live.slot_results),
        "state_counts": dict(states),
        "tool_counts": dict(tool_counts),
        "camera_sheet_slots": attached,
        "model_calls": len(adapter.calls),
        "invalid_model_calls": len(adapter.invalid_calls),
        "wall_seconds": wall_seconds,
        "sum_model_call_seconds": sum(call_seconds),
        "mean_model_call_seconds": (
            sum(call_seconds) / len(call_seconds) if call_seconds else None
        ),
        "validation_error_count": sum(
            len(result.validation_errors) for result in live.slot_results
        ),
        "replay_verified": verified,
    }
    write(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
