#!/usr/bin/env python3
"""Prepare and serve the reusable parking-slot human validation UI."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from parking_slot_validation.evidence import EvidenceSettings, build_manifest
from parking_slot_validation.models import Prediction, fingerprint_files, normalize_prediction_row
from parking_slot_validation.storage import LabelRepository
from parking_slot_validation.web import ValidationState, create_server


DEFAULT_PREDICTIONS = Path("outputs/slot_constrained_box_scoring_pose_corrected_final/slot_box_scores.csv")
DEFAULT_FRAMES = Path("outputs/frame_map_dataset_pose_corrected_final/frames.csv")
DEFAULT_SLOTS = Path("outputs/full_icpark_allframes_vehicle_cluster/slot_database.json")
DEFAULT_OUTPUT = Path("outputs/parking_slot_human_validation_pose_corrected_final")


def read_predictions(path: Path, dataset_id: str, run_id: str) -> list[Prediction]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    predictions = [normalize_prediction_row(row, run_id, dataset_id) for row in rows]
    slot_ids = [prediction.slot_id for prediction in predictions]
    if len(slot_ids) != len(set(slot_ids)):
        raise ValueError("prediction file contains duplicate slot_id values")
    return predictions


def save_predictions(path: Path, predictions: list[Prediction]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {"schema_version": 1, "predictions": [prediction.to_dict() for prediction in predictions]},
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        ),
        encoding="utf-8",
    )


def prepare(args: argparse.Namespace) -> tuple[dict, list[Prediction]]:
    dataset_id = fingerprint_files([args.frames, args.slot_database])
    run_id = args.run_id or args.predictions.parent.name
    predictions = read_predictions(args.predictions, dataset_id, run_id)
    settings = EvidenceSettings(
        search_before=args.search_before,
        search_after=args.search_after,
        frame_stride=args.frame_stride,
        pose_prefilter_limit=args.pose_prefilter_limit,
    )

    def progress(index: int, total: int, slot_id: str) -> None:
        if index == 1 or index % 10 == 0 or index == total:
            print(f"证据生成 {index}/{total}: {slot_id}", flush=True)

    manifest = build_manifest(
        predictions,
        args.frames,
        args.slot_database,
        args.output_dir,
        settings,
        rebuild=args.rebuild_evidence,
        progress=progress,
    )
    save_predictions(args.output_dir / "predictions.json", predictions)
    print(
        json.dumps(
            {
                "manifest": str(args.output_dir / "evidence_manifest.json"),
                "cases": len(manifest["cases"]),
                **manifest["summary"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return manifest, predictions


def load_for_serve(args: argparse.Namespace) -> tuple[dict, list[Prediction]]:
    manifest_path = args.output_dir / "evidence_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError("evidence manifest is missing; run prepare first")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    run_id = args.run_id or args.predictions.parent.name
    predictions = read_predictions(args.predictions, str(manifest["dataset_id"]), run_id)
    save_predictions(args.output_dir / "predictions.json", predictions)
    return manifest, predictions


def serve(args: argparse.Namespace, manifest: dict, predictions: list[Prediction]) -> None:
    repository = LabelRepository(
        args.output_dir / "human_labels.json",
        str(manifest["dataset_id"]),
        str(manifest["manifest_id"]),
        reviewer=args.reviewer,
    )
    state = ValidationState(
        manifest=manifest,
        predictions={prediction.slot_id: prediction for prediction in predictions},
        repository=repository,
        output_dir=args.output_dir,
        index_html=ROOT / "parking_slot_validation" / "static" / "index.html",
    )
    server = create_server(args.host, args.port, state)
    host, port = server.server_address[:2]
    shown_host = "127.0.0.1" if host in {"0.0.0.0", "::"} else host
    print(f"标注页面: http://{shown_host}:{port}/", flush=True)
    print(f"标签文件: {repository.path}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--run-id", default="")


def add_prepare(parser: argparse.ArgumentParser) -> None:
    add_common(parser)
    parser.add_argument("--frames", type=Path, default=DEFAULT_FRAMES)
    parser.add_argument("--slot-database", type=Path, default=DEFAULT_SLOTS)
    parser.add_argument("--search-before", type=int, default=400)
    parser.add_argument("--search-after", type=int, default=400)
    parser.add_argument("--frame-stride", type=int, default=5)
    parser.add_argument("--pose-prefilter-limit", type=int, default=30)
    parser.add_argument("--rebuild-evidence", action="store_true")


def add_serve(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--reviewer", default="local-reviewer")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parking-slot Occupied/Free human validation")
    commands = parser.add_subparsers(dest="command", required=True)
    prepare_parser = commands.add_parser("prepare", help="build immutable multiframe evidence")
    add_prepare(prepare_parser)
    serve_parser = commands.add_parser("serve", help="serve an existing evidence manifest")
    add_common(serve_parser)
    add_serve(serve_parser)
    both_parser = commands.add_parser("prepare-and-serve", help="prepare evidence and launch UI")
    add_prepare(both_parser)
    add_serve(both_parser)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "prepare":
        prepare(args)
        return 0
    if args.command == "serve":
        manifest, predictions = load_for_serve(args)
        serve(args, manifest, predictions)
        return 0
    manifest, predictions = prepare(args)
    serve(args, manifest, predictions)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
