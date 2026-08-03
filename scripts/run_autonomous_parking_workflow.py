#!/usr/bin/env python3
"""One-command Part1 -> front180 -> autonomous Agent -> replay workflow."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from parking_slot_agent_v2.agent import SingleSlotAgent, TERMINAL_CONFIDENCE
from parking_slot_agent_v2.contracts import Part1Output
from parking_slot_agent_v2.extended_lidar import build_extended_part2_input
from parking_slot_agent_v2.model import ReplayModelAdapter
from parking_slot_agent_v2.openai_adapter import DEFAULT_OPENAI_MODEL, OpenAIResponsesAdapter
from parking_slot_agent_v2.part1 import build_part1_output_from_directory
from parking_slot_agent_v2.pipeline import Part2RunResult, run_candidate_queue
from parking_slot_agent_v2.tools import V2ToolSuite
from scripts.build_autonomous_agent_input import build as build_front180

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXPERIMENT = ROOT / "Nature_ParkingAgent_实验报告_20260728/random_midroute_experiment"


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def _semantic_result(run: Part2RunResult) -> dict[str, Any]:
    rows = []
    for result in run.slot_results:
        case = result.case
        rows.append({
            "slot_id": case.slot_id,
            "final_state": None if case.final_state is None else case.final_state.value,
            "final_scores": None if case.final_scores is None else case.final_scores.to_dict(),
            "tool_sequence": [
                evidence.tool_name for evidence in case.evidence if evidence.round_index >= 1
            ],
            "tool_rounds": result.tool_rounds,
            "stop_reason": result.stop_reason,
        })
    return {"slots": rows, "run_stop_reason": run.stop_reason}


def _node(manifest: dict[str, Any], name: str, status: str, **details: Any) -> None:
    manifest["nodes"][name] = {"status": status, **details}
    _write_json(Path(manifest["manifest_path"]), manifest)


def run(args: argparse.Namespace) -> None:
    experiment = args.experiment_dir.resolve()
    work = experiment / "autonomous_agent"
    work.mkdir(parents=True, exist_ok=True)
    all_local = work / "part1_all_local.json"
    front180 = work / "part1_front180.json"
    input_manifest = work / "input_manifest.json"
    prompt_path = work / "system_prompt.txt"
    live_dir = work / "openai_run"
    replay_path = work / "replay_actions.json"
    replay_dir = work / "replay_verified"
    workflow_path = work / "workflow_manifest.json"
    manifest: dict[str, Any] = {
        "schema_version": "autonomous-parking-workflow/1.0",
        "manifest_path": str(workflow_path),
        "architecture": "fixed workflow skeleton; autonomous per-case model tool choice",
        "slot_specific_policy": False,
        "terminal_confidence": TERMINAL_CONFIDENCE,
        "nodes": {},
    }
    _write_json(workflow_path, manifest)

    _node(manifest, "01_adapt_part1", "running")
    adapted = build_part1_output_from_directory(
        args.part1_dir,
        args.slot_db,
        args.dataset_id,
        requested_radius_m=args.radius_m,
    )
    _write_json(all_local, adapted.to_dict())
    _node(
        manifest,
        "01_adapt_part1",
        "completed",
        frame_count=len(adapted.scene.frames),
        candidate_count=len(adapted.slot_cases),
        output=str(all_local),
    )

    _node(manifest, "02_front180_scope", "running")
    scoped = build_front180(all_local, front180, input_manifest, prompt_path)
    _node(
        manifest,
        "02_front180_scope",
        "completed",
        formula="Part1 in {Free,Unknown} and ego_forward>=0",
        case_count=len(scoped.slot_cases),
        case_ids=[case.slot_id for case in scoped.slot_cases],
        output=str(front180),
    )

    _node(manifest, "02b_extended_lidar_tool", "running")
    extended_summary = build_extended_part2_input(
        scoped,
        frames_csv=args.frames_csv,
        slot_database=args.slot_db,
        map_points_dir=args.map_points_dir,
        output_dir=work / f"extended_lidar_w{args.extended_lidar_window}",
        window_frames=args.extended_lidar_window,
        dataset_id=args.dataset_id,
    )
    _write_json(front180, scoped.to_dict())
    _node(
        manifest,
        "02b_extended_lidar_tool",
        "completed",
        window_frames=args.extended_lidar_window,
        candidate_count=extended_summary["candidate_count"],
        state_counts=extended_summary["state_counts"],
        output=extended_summary["part1_output_path"],
    )

    _node(manifest, "03_autonomous_agent_loop", "running", model=args.model)
    part1_live = Part1Output.from_dict(json.loads(front180.read_text(encoding="utf-8")))
    live_model = OpenAIResponsesAdapter(
        model=args.model,
        reasoning_effort=args.reasoning_effort,
        image_detail=args.image_detail,
        audit_dir=live_dir / "openai_audit",
        api_key_file=args.api_key_file,
    )
    live = run_candidate_queue(
        part1_live,
        SingleSlotAgent(live_model, V2ToolSuite(map_radius_m=part1_live.scene.radius_m)),
        live_dir,
        stop_at_first_free=False,
        resume=args.resume,
    )
    replay_payload = live_model.replay_payload()
    # Adapters produced before replay-error recording can omit invalid semantic
    # model turns. Fill only the missing tail so turn-limit behavior is replayed.
    for result in live.slot_results:
        case_id = result.case.case_id
        sequence = replay_payload["actions"].setdefault(case_id, [])
        bindings = replay_payload["evidence_bindings"].setdefault(case_id, [])
        missing = max(0, result.model_turns - len(sequence))
        for _ in range(missing):
            sequence.append({
                "type": "replay_error",
                "error_type": "ValueError",
                "message": "recorded invalid structured action",
            })
            bindings.append([])
    _write_json(replay_path, replay_payload)
    _node(
        manifest,
        "03_autonomous_agent_loop",
        "completed",
        processed_case_count=len(live.processed_case_ids),
        model=args.model,
        raw_call_audit=str(live_dir / "openai_audit"),
        action_replay=str(replay_path),
        result=str(live_dir / "part2_result.json"),
    )

    _node(manifest, "04_deterministic_replay", "running")
    part1_replay = Part1Output.from_dict(json.loads(front180.read_text(encoding="utf-8")))
    replay_model = ReplayModelAdapter.from_path(replay_path)
    replay = run_candidate_queue(
        part1_replay,
        SingleSlotAgent(replay_model, V2ToolSuite(map_radius_m=part1_replay.scene.radius_m)),
        replay_dir,
        stop_at_first_free=False,
        resume=False,
    )
    live_semantic = _semantic_result(live)
    replay_semantic = _semantic_result(replay)
    verified = live_semantic == replay_semantic
    verification = {
        "schema_version": "autonomous-agent-replay-verification/1.0",
        "verified": verified,
        "live": live_semantic,
        "replay": replay_semantic,
    }
    _write_json(work / "replay_verification.json", verification)
    if not verified:
        raise RuntimeError("autonomous Agent replay semantic result mismatch")
    _node(
        manifest,
        "04_deterministic_replay",
        "completed",
        verified=True,
        output=str(work / "replay_verification.json"),
    )
    _node(manifest, "05_html_report", "ready_for_report_builder")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-dir", type=Path, default=DEFAULT_EXPERIMENT)
    parser.add_argument("--part1-dir", type=Path, default=DEFAULT_EXPERIMENT / "part1_w30k5")
    parser.add_argument(
        "--slot-db",
        type=Path,
        default=ROOT / "outputs/full_icpark_allframes_vehicle_cluster/slot_database.json",
    )
    parser.add_argument("--dataset-id", default="random-midroute-frame6681-w30k5")
    parser.add_argument(
        "--frames-csv",
        type=Path,
        default=ROOT / "outputs/frame_map_dataset_pose_corrected_final/frames.csv",
    )
    parser.add_argument(
        "--map-points-dir",
        type=Path,
        default=ROOT / "outputs/frame_map_dataset_pose_corrected_final/map_points",
    )
    parser.add_argument("--extended-lidar-window", type=int, default=60)
    parser.add_argument("--radius-m", type=float, default=18.0)
    parser.add_argument("--model", default=DEFAULT_OPENAI_MODEL)
    parser.add_argument("--reasoning-effort", default="medium")
    parser.add_argument("--image-detail", default="original")
    parser.add_argument(
        "--api-key-file",
        type=Path,
        default=Path("/home/ParkingAgent/.config/parking-agent/openai_api_key"),
    )
    parser.add_argument("--resume", action="store_true")
    run(parser.parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
