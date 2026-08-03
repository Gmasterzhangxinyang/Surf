"""Command-line entry point for the isolated v2 pipeline."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Sequence

from .agent import SingleSlotAgent
from .contracts import Part1Output
from .model import LocalVLMAdapter, ReplayModelAdapter
from .openai_adapter import DEFAULT_OPENAI_MODEL, OpenAIResponsesAdapter
from .part1 import (
    DEFAULT_RADIUS_M,
    build_part1_output_from_directory,
    run_part1,
)
from .pipeline import run_candidate_queue
from .tools import V2ToolSuite


DEFAULT_FRAMES_CSV = Path("outputs/frame_map_dataset_pose_corrected_final/frames.csv")
DEFAULT_MAP_POINTS = Path("outputs/frame_map_dataset_pose_corrected_final/map_points")
DEFAULT_SLOT_DB = Path("outputs/full_icpark_allframes_vehicle_cluster/slot_database.json")


def _read_part1(path: Path) -> Part1Output:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return Part1Output.from_dict(payload)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m parking_slot_agent_v2",
        description="30 m / 15-frame parking-slot Part1 and per-slot Part2 agent.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    adapt = commands.add_parser(
        "adapt-part1",
        help="Adapt an existing raw Part1 directory into the v2 object contract.",
    )
    adapt.add_argument("--part1-dir", type=Path, required=True)
    adapt.add_argument("--slot-db", type=Path, default=DEFAULT_SLOT_DB)
    adapt.add_argument("--dataset-id", required=True)
    adapt.add_argument("--radius-m", type=float, default=DEFAULT_RADIUS_M)
    adapt.add_argument(
        "--allow-partial-coverage",
        action="store_true",
        help="Allow a wider geometry map than the source LiDAR evidence radius.",
    )
    adapt.add_argument("--output", type=Path, required=True)

    run1 = commands.add_parser(
        "run-part1",
        help="Run the upstream Hybrid3D evidence algorithm at one t0 and emit v2 Part1.",
    )
    run1.add_argument("--frames-csv", type=Path, default=DEFAULT_FRAMES_CSV)
    run1.add_argument("--slot-db", type=Path, default=DEFAULT_SLOT_DB)
    run1.add_argument("--map-points-dir", type=Path, default=DEFAULT_MAP_POINTS)
    run1.add_argument("--output-dir", type=Path, required=True)
    run1.add_argument("--dataset-id", required=True)
    run1.add_argument("--anchor-frame", type=int)
    run1.add_argument("--radius-m", type=float, default=DEFAULT_RADIUS_M)
    run1.add_argument("--overwrite", action="store_true")

    validate = commands.add_parser(
        "validate",
        help="Strictly load a v2 Part1 output and print a compact summary.",
    )
    validate.add_argument("--input", type=Path, required=True)

    run2 = commands.add_parser(
        "run-part2",
        help="Process Free then Unknown SlotCases with a replay or loopback local VLM.",
    )
    run2.add_argument("--input", type=Path, required=True)
    run2.add_argument("--output-dir", type=Path, required=True)
    model_group = run2.add_mutually_exclusive_group(required=True)
    model_group.add_argument("--replay-actions", type=Path)
    model_group.add_argument("--base-url")
    model_group.add_argument(
        "--openai-model",
        nargs="?",
        const=DEFAULT_OPENAI_MODEL,
        help=(
            "Use the official OpenAI Responses API. The optional model defaults "
            f"to {DEFAULT_OPENAI_MODEL}; the key is read only from OPENAI_API_KEY."
        ),
    )
    run2.add_argument("--model")
    run2.add_argument(
        "--openai-reasoning-effort",
        choices=("none", "low", "medium", "high", "xhigh", "max"),
        default="medium",
    )
    run2.add_argument(
        "--openai-image-detail",
        choices=("low", "high", "original", "auto"),
        default="original",
    )
    run2.add_argument(
        "--openai-api-key-file",
        type=Path,
        help=(
            "Optional 600-permission secret file used only when OPENAI_API_KEY is "
            "unset. The key is never written to run outputs."
        ),
    )
    run2.add_argument("--replay-output", type=Path)
    run2.add_argument(
        "--evaluation-exhaustive",
        action="store_true",
        help="Evaluate every candidate without operational first-Free early stop.",
    )
    run2.add_argument(
        "--resume",
        action="store_true",
        help="Resume atomically checkpointed cases in the selected output directory.",
    )
    return parser


def _summary(part1: Part1Output) -> dict[str, Any]:
    return {
        "schema_version": part1.schema_version,
        "snapshot_id": part1.scene.snapshot_id,
        "anchor_frame": part1.scene.anchor_frame_id,
        "radius_m": part1.scene.radius_m,
        "sensor_frame_count": len(part1.scene.frames),
        "nearby_map_slot_count": len(part1.scene.slots),
        "part1_observed_slot_count": sum(slot.observed for slot in part1.scene.slots),
        "free_candidates": len(part1.free_cases),
        "unknown_candidates": len(part1.unknown_cases),
        "candidate_order": [case.slot_id for case in part1.slot_cases],
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "adapt-part1":
        output = build_part1_output_from_directory(
            args.part1_dir,
            args.slot_db,
            args.dataset_id,
            requested_radius_m=args.radius_m,
            allow_partial_coverage=args.allow_partial_coverage,
        )
        _write_json(args.output, output.to_dict())
        print(json.dumps(_summary(output), ensure_ascii=False, sort_keys=True))
        return 0

    if args.command == "run-part1":
        output = run_part1(
            args.frames_csv,
            args.slot_db,
            args.map_points_dir,
            args.output_dir,
            args.dataset_id,
            anchor_frame=args.anchor_frame,
            requested_radius_m=args.radius_m,
            overwrite=args.overwrite,
        )
        print(json.dumps(_summary(output), ensure_ascii=False, sort_keys=True))
        return 0

    part1 = _read_part1(args.input)
    if args.command == "validate":
        print(json.dumps(_summary(part1), ensure_ascii=False, sort_keys=True))
        return 0

    recording_model = None
    if args.replay_actions is not None:
        model = ReplayModelAdapter.from_path(args.replay_actions)
    elif args.openai_model is not None:
        if not os.environ.get("OPENAI_API_KEY") and args.openai_api_key_file is None:
            raise SystemExit(
                "OPENAI_API_KEY or --openai-api-key-file is required for "
                "--openai-model; do not pass the key itself as a command-line argument"
            )
        recording_model = OpenAIResponsesAdapter(
            model=args.openai_model,
            reasoning_effort=args.openai_reasoning_effort,
            image_detail=args.openai_image_detail,
            audit_dir=args.output_dir / "openai_audit",
            api_key_file=args.openai_api_key_file,
        )
        model = recording_model
    else:
        if not args.model:
            raise SystemExit("--model is required with --base-url")
        recording_model = LocalVLMAdapter(base_url=args.base_url, model=args.model)
        model = recording_model
    agent = SingleSlotAgent(model, V2ToolSuite(map_radius_m=part1.scene.radius_m))
    result = run_candidate_queue(
        part1,
        agent,
        args.output_dir,
        stop_at_first_free=not args.evaluation_exhaustive,
        resume=args.resume,
    )
    if recording_model is not None and args.replay_output is not None:
        _write_json(args.replay_output, recording_model.replay_payload())
    print(
        json.dumps(
            {
                "selected_slot_id": result.selected_slot_id,
                "stop_reason": result.stop_reason,
                "processed_case_count": len(result.processed_case_ids),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


__all__ = ["build_parser", "main"]
