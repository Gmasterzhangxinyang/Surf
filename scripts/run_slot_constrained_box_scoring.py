#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from parking_slot_box_scoring.config import BoxScoringConfig
from parking_slot_box_scoring.frame_selection import build_slot_frame_selections, read_csv_rows
from parking_slot_box_scoring.geometry import compute_slot_frame
from parking_slot_box_scoring.reporting import draw_debug_slot, write_csv, write_html_report, write_json
from parking_slot_box_scoring.scoring import score_slot_points


CSV_FIELDS = [
    "slot_id",
    "state",
    "score",
    "reason",
    "anchor_frame",
    "selected_frames",
    "selected_frame_count",
    "supported_frame_count",
    "temporal_support",
    "inside_vehicle_point_count",
    "inside_low_point_count",
    "z95_above_ground",
    "height_span",
    "bev_coverage",
    "slot_core_overlap",
    "slot_polygon_overlap",
    "boundary_ratio",
    "adjacent_overlap",
    "center_x",
    "center_y",
    "yaw",
    "length",
    "width",
    "length_m",
    "width_m",
    "point_support",
    "height_support",
    "low_height_penalty",
    "adjacent_penalty",
    "boundary_penalty",
    "linearity_penalty",
    "outside_box_residual_penalty",
    "baseline_state",
    "baseline_score",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run slot-constrained vehicle box scoring experiment.")
    parser.add_argument("--frames-csv", required=True)
    parser.add_argument("--slot-db", required=True)
    parser.add_argument("--map-points-dir", required=True)
    parser.add_argument("--baseline-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--window-before", type=int, default=28)
    parser.add_argument("--window-after", type=int, default=28)
    parser.add_argument("--frame-stride", type=int, default=4)
    parser.add_argument("--frame-selection", choices=["anchor_window", "visibility_topk"], default="anchor_window")
    parser.add_argument("--top-k-visible", type=int, default=15)
    parser.add_argument("--debug-slots", default="slot_0944,slot_1069,slot_0988,slot_1075")
    return parser.parse_args()


def load_slot_database(path: str | Path) -> tuple[list[dict], float]:
    with Path(path).open() as f:
        data = json.load(f)
    slots = data["slots"] if isinstance(data, dict) and "slots" in data else data
    map_units_per_meter = float(data.get("map_units_per_meter", 1.0)) if isinstance(data, dict) else 1.0
    processed: list[dict] = []
    for slot in slots:
        item = dict(slot)
        item["polygon_np"] = np.asarray(item["polygon_map"], dtype=np.float64)
        item["core_np"] = np.asarray(item.get("core_polygon_map") or item.get("inner_polygon") or item["polygon_map"], dtype=np.float64)
        item["margin_np"] = np.asarray(item.get("margin_polygon_map") or item.get("margin_polygon") or item["polygon_map"], dtype=np.float64)
        item["center_np"] = np.asarray(item.get("center_map", item["polygon_np"].mean(axis=0)), dtype=np.float64)
        processed.append(item)
    return processed, map_units_per_meter


def frame_row_index(frame_rows: list[dict[str, str]]) -> dict[int, dict[str, str]]:
    return {int(row["frame"]): row for row in frame_rows if row.get("frame")}


def resolve_map_points_path(row: dict[str, str], map_points_dir: str | Path) -> Path:
    raw = row.get("map_points_path", "")
    if raw:
        path = Path(raw)
        if path.exists():
            return path
        candidate = ROOT / path
        if candidate.exists():
            return candidate
    frame = int(row["frame"])
    return Path(map_points_dir) / f"{frame:06d}.npz"


def load_frame_points(frame: int, row_by_frame: dict[int, dict[str, str]], map_points_dir: str | Path, cache: dict[int, np.ndarray]) -> np.ndarray:
    if frame in cache:
        return cache[frame]
    row = row_by_frame.get(frame)
    if row is None:
        points = np.empty((0, 4), dtype=np.float64)
    else:
        path = resolve_map_points_path(row, map_points_dir)
        if not path.exists():
            points = np.empty((0, 4), dtype=np.float64)
        else:
            data = np.load(path)
            key = "points_map_xyzi" if "points_map_xyzi" in data.files else data.files[0]
            points = np.asarray(data[key], dtype=np.float64)
            if points.ndim != 2 or points.shape[1] < 3:
                points = np.empty((0, 4), dtype=np.float64)
    cache[frame] = points
    return points


def crop_points_to_slot_roi(points: np.ndarray, slot: dict, map_units_per_meter: float, roi_extra_m: float) -> np.ndarray:
    if len(points) == 0:
        return points
    margin = np.asarray(slot.get("margin_np", slot["polygon_np"]), dtype=np.float64)
    extra = roi_extra_m * map_units_per_meter
    min_xy = margin.min(axis=0) - extra
    max_xy = margin.max(axis=0) + extra
    xy = points[:, :2]
    mask = (xy[:, 0] >= min_xy[0]) & (xy[:, 0] <= max_xy[0]) & (xy[:, 1] >= min_xy[1]) & (xy[:, 1] <= max_xy[1])
    return points[mask]


def row_from_score(score, selection, map_units_per_meter: float) -> dict[str, object]:
    row = score.to_dict()
    row.update(
        {
            "anchor_frame": selection.anchor_frame,
            "selected_frames": json.dumps(selection.selected_frames),
            "selected_frame_count": score.selected_frame_count,
            "length_m": score.length / max(map_units_per_meter, 1e-9),
            "width_m": score.width / max(map_units_per_meter, 1e-9),
            "baseline_state": selection.baseline_state,
            "baseline_score": selection.baseline_score,
        }
    )
    return row


def write_comparison(path: Path, rows: list[dict[str, object]]) -> None:
    fields = [
        "slot_id",
        "baseline_state",
        "baseline_score",
        "box_state",
        "box_score",
        "score_delta",
        "reason",
        "anchor_frame",
    ]
    comp_rows = []
    for row in rows:
        box_score = float(row.get("score") or 0.0)
        baseline_score = float(row.get("baseline_score") or 0.0)
        comp_rows.append(
            {
                "slot_id": row.get("slot_id"),
                "baseline_state": row.get("baseline_state"),
                "baseline_score": baseline_score,
                "box_state": row.get("state"),
                "box_score": box_score,
                "score_delta": box_score - baseline_score,
                "reason": row.get("reason"),
                "anchor_frame": row.get("anchor_frame"),
            }
        )
    write_csv(path, comp_rows, fields)


def save_readme_if_no_baseline(output_dir: Path, baseline_dir: str | Path) -> None:
    baseline_path = Path(baseline_dir) / "slot_aligned_evidence.csv"
    if baseline_path.exists():
        return
    (output_dir / "README.md").write_text(
        "Baseline per-slot table was not found at "
        f"`{baseline_path}`. The pipeline still produced box scoring outputs, "
        "but `comparison_against_dbscan.csv` has empty baseline fields.\n"
    )


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = BoxScoringConfig(
        window_before=args.window_before,
        window_after=args.window_after,
        frame_stride=args.frame_stride,
        frame_selection=args.frame_selection,
        top_k_visible=args.top_k_visible,
    )
    slots, map_units_per_meter = load_slot_database(args.slot_db)
    slots_by_id = {str(slot["slot_id"]): slot for slot in slots}
    frame_rows = read_csv_rows(args.frames_csv)
    rows_by_frame = frame_row_index(frame_rows)
    selections = build_slot_frame_selections(
        slots,
        frame_rows,
        args.baseline_dir,
        map_units_per_meter,
        args.window_before,
        args.window_after,
        args.frame_stride,
        args.frame_selection,
    )

    debug_slots = [s.strip() for s in args.debug_slots.split(",") if s.strip()]
    cache: dict[int, np.ndarray] = {}
    score_rows: list[dict[str, object]] = []
    accumulated_by_slot: dict[str, np.ndarray] = {}
    scores_by_slot = {}

    for index, selection in enumerate(selections, 1):
        slot = slots_by_id.get(selection.slot_id)
        if slot is None:
            continue
        frame_points: list[np.ndarray] = []
        for frame in selection.selected_frames:
            points = load_frame_points(frame, rows_by_frame, args.map_points_dir, cache)
            cropped = crop_points_to_slot_roi(points, slot, map_units_per_meter, config.roi_extra_m)
            if len(cropped):
                finite = np.isfinite(cropped[:, :3]).all(axis=1)
                cropped = cropped[finite]
            frame_points.append(cropped)
        accumulated = np.vstack(frame_points) if frame_points else np.empty((0, 4), dtype=np.float64)
        try:
            score = score_slot_points(slot, accumulated, frame_points, slots_by_id, map_units_per_meter, config)
        except Exception as exc:  # Keep one bad slot from killing the diagnostic run.
            frame = compute_slot_frame(slot["polygon_np"], slot["center_np"])
            from parking_slot_box_scoring.scoring import BoxScore

            score = BoxScore(
                slot_id=selection.slot_id,
                score=0.0,
                state="box_unknown_insufficient_visibility",
                reason=f"scoring failed: {exc}",
                center_x=float(frame.center[0]),
                center_y=float(frame.center[1]),
                yaw=float(frame.yaw),
                length=float(frame.length),
                width=float(frame.width),
                point_support=0.0,
                inside_vehicle_point_count=0,
                inside_low_point_count=0,
                bev_coverage=0.0,
                height_support=0.0,
                z95_above_ground=0.0,
                height_span=0.0,
                slot_core_overlap=0.0,
                slot_polygon_overlap=0.0,
                boundary_ratio=0.0,
                adjacent_overlap=0.0,
                temporal_support=0.0,
                supported_frame_count=0,
                selected_frame_count=len(frame_points),
                low_height_penalty=1.0,
                adjacent_penalty=0.0,
                boundary_penalty=0.0,
                linearity_penalty=0.0,
                outside_box_residual_penalty=0.0,
            )
        row = row_from_score(score, selection, map_units_per_meter)
        score_rows.append(row)
        scores_by_slot[selection.slot_id] = score
        if selection.slot_id in debug_slots:
            accumulated_by_slot[selection.slot_id] = accumulated
        if index % 25 == 0:
            print(f"processed {index}/{len(selections)} slots", flush=True)

    debug_images: dict[str, str] = {}
    debug_dir = output_dir / "debug"
    for slot_id in debug_slots:
        slot = slots_by_id.get(slot_id)
        score = scores_by_slot.get(slot_id)
        accumulated = accumulated_by_slot.get(slot_id)
        if slot is None or score is None or accumulated is None:
            continue
        adjacent = [slots_by_id[sid] for sid in slot.get("adjacent_slots", []) if sid in slots_by_id]
        image_path = debug_dir / f"{slot_id}_box_scoring.png"
        draw_debug_slot(image_path, slot, adjacent, score, accumulated, config)
        debug_images[slot_id] = str(image_path.relative_to(output_dir))

    state_counter = Counter(str(row["state"]) for row in score_rows)
    summary = {
        "pipeline": config.pipeline,
        "input_frames_csv": args.frames_csv,
        "input_slot_db": args.slot_db,
        "input_map_points_dir": args.map_points_dir,
        "baseline_dir": args.baseline_dir,
        "output_dir": args.output_dir,
        "config": config.to_dict(),
        "map_units_per_meter": map_units_per_meter,
        "processed_slot_count": len(score_rows),
        "selection_count": len(selections),
        "state_counts": dict(state_counter),
        "debug_slots": debug_slots,
    }

    write_json(output_dir / "summary.json", summary)
    write_csv(output_dir / "slot_box_scores.csv", score_rows, CSV_FIELDS)
    write_json(output_dir / "slot_box_scores.json", score_rows)
    write_comparison(output_dir / "comparison_against_dbscan.csv", score_rows)
    write_html_report(output_dir / "slot_box_scoring_report.html", summary, score_rows, debug_images)
    save_readme_if_no_baseline(output_dir, args.baseline_dir)
    print(json.dumps({"state_counts": dict(state_counter), "processed_slot_count": len(score_rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
