#!/usr/bin/env python3
"""Select one reproducible, label-blind, mid-route experiment anchor."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from parking_slot_hybrid_3d.io import load_known_slots  # noqa: E402


SCHEMA_VERSION = "parkingagent-random-anchor-selection/1.0"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames-csv", required=True, type=Path)
    parser.add_argument("--slot-db", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument("--middle-fraction", type=float, default=0.60)
    parser.add_argument("--history-span", type=int, default=100)
    parser.add_argument("--nearby-radius-m", type=float, default=25.0)
    parser.add_argument("--min-nearby-slots", type=int, default=8)
    parser.add_argument(
        "--anchor-stride",
        type=int,
        default=10,
        help="Predeclared temporal thinning of eligible anchors.",
    )
    return parser


def _canonical_bytes(payload: object) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _read_frames(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("frames CSV is empty")
    return rows


def _slot_center(slot: object) -> np.ndarray:
    polygon = np.asarray(getattr(slot, "polygon_map"), dtype=np.float64)
    if polygon.shape != (4, 2):
        raise ValueError(f"invalid slot polygon shape: {polygon.shape}")
    return polygon.mean(axis=0)


def main() -> int:
    args = build_parser().parse_args()
    if not 0.0 < args.middle_fraction <= 1.0:
        raise ValueError("--middle-fraction must be in (0, 1]")
    if args.history_span < 3:
        raise ValueError("--history-span must be at least 3")
    if args.anchor_stride < 1:
        raise ValueError("--anchor-stride must be positive")

    rows = _read_frames(args.frames_csv)
    slots, map_units_per_meter = load_known_slots(args.slot_db)
    centers = np.vstack([_slot_center(slot) for slot in slots])
    radius_map = args.nearby_radius_m * map_units_per_meter

    excluded_fraction = (1.0 - args.middle_fraction) / 2.0
    first_index = max(
        args.history_span - 1,
        int(np.ceil(excluded_fraction * len(rows))),
    )
    last_index = min(
        len(rows) - 1,
        int(np.floor((1.0 - excluded_fraction) * len(rows))) - 1,
    )
    if first_index > last_index:
        raise ValueError("no indices remain after middle/history constraints")

    project_root = PROJECT_ROOT
    eligible: list[dict[str, object]] = []
    exclusion_counts: dict[str, int] = {}

    def reject(reason: str) -> None:
        exclusion_counts[reason] = exclusion_counts.get(reason, 0) + 1

    for index in range(first_index, last_index + 1, args.anchor_stride):
        row = rows[index]
        if row.get("missing_lidar") != "0":
            reject("missing_lidar")
            continue
        if row.get("missing_image") != "0" or row.get("camera_match_valid") != "1":
            reject("invalid_camera_match")
            continue
        required_paths = (
            Path(row["lidar_path"]),
            Path(row["camera_image_path"]),
            project_root / row["map_points_path"],
        )
        if not all(path.is_file() for path in required_paths):
            reject("artifact_missing")
            continue
        anchor_xy = np.array([float(row["map_x"]), float(row["map_y"])])
        nearby = int(np.count_nonzero(np.linalg.norm(centers - anchor_xy, axis=1) <= radius_map))
        if nearby < args.min_nearby_slots:
            reject("insufficient_nearby_slot_geometry")
            continue
        eligible.append(
            {
                "row_index": index,
                "frame_id": int(row["frame"]),
                "camera_frame": int(row["camera_frame"]),
                "nearby_slot_count": nearby,
                "map_xy": [float(row["map_x"]), float(row["map_y"])],
                "map_yaw_rad": float(row["map_yaw"]),
            }
        )

    if not eligible:
        raise RuntimeError("no label-blind eligible anchors")
    selected = random.Random(args.seed).choice(eligible)
    eligible_digest = "sha256:" + hashlib.sha256(_canonical_bytes(eligible)).hexdigest()
    payload = {
        "schema_version": SCHEMA_VERSION,
        "selection_policy": {
            "prediction_blind": True,
            "gt_fields_read": [],
            "part1_state_fields_read": [],
            "seed": args.seed,
            "middle_fraction": args.middle_fraction,
            "history_span": args.history_span,
            "nearby_radius_m": args.nearby_radius_m,
            "min_nearby_slots": args.min_nearby_slots,
            "anchor_stride": args.anchor_stride,
            "eligible_row_index_range_inclusive": [first_index, last_index],
        },
        "source": {
            "frames_csv": str(args.frames_csv),
            "slot_db": str(args.slot_db),
            "frame_count": len(rows),
        },
        "eligible_anchor_count": len(eligible),
        "eligible_anchor_sha256": eligible_digest,
        "exclusion_counts": dict(sorted(exclusion_counts.items())),
        "selected_anchor": selected,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(_canonical_bytes(payload) + b"\n")
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
