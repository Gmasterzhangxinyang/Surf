#!/usr/bin/env python3
"""Run slot-centered multi-frame accumulation using pose-selected anchor frames.

This is a diagnostic experiment. It does not update Part 1 final states.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import multiframe_pointcloud_accumulation_probe as accumulation


DEFAULT_CONSENSUS = Path("outputs/multiframe_temporal_consensus_eps025_ms8/multiframe_temporal_consensus.csv")
DEFAULT_OUTPUT = Path("outputs/slot_aligned_accumulation_stride4_w52_eps025_ms8")
DEFAULT_INCLUDE_STATES = [
    "possible_occupied_by_multiframe_accumulation",
    "multiframe_possible_occupied_unconfirmed",
    "multiframe_boundary_or_adjacent_conflict",
    "multiframe_no_stable_vehicle_evidence",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Slot-aligned multi-frame point-cloud accumulation probe")
    parser.add_argument("--frames", type=Path, default=accumulation.DEFAULT_FRAMES)
    parser.add_argument("--slot-database", type=Path, default=accumulation.DEFAULT_SLOT_DATABASE)
    parser.add_argument("--consensus-csv", type=Path, default=DEFAULT_CONSENSUS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--include-states", nargs="+", default=DEFAULT_INCLUDE_STATES)
    parser.add_argument("--window-before", type=int, default=52)
    parser.add_argument("--window-after", type=int, default=52)
    parser.add_argument("--window-frame-stride", type=int, default=4)
    parser.add_argument("--voxel-size-m", type=float, default=0.08)
    parser.add_argument("--max-range-m", type=float, default=22.0)
    parser.add_argument("--ground-quantile", type=float, default=0.08)
    parser.add_argument("--obstacle-z-min-m", type=float, default=0.30)
    parser.add_argument("--obstacle-z-max-m", type=float, default=2.50)
    parser.add_argument("--cluster-eps-m", type=float, default=0.25)
    parser.add_argument("--cluster-min-samples", type=int, default=8)
    parser.add_argument("--vehicle-min-points", type=int, default=40)
    parser.add_argument("--vehicle-min-length-m", type=float, default=2.5)
    parser.add_argument("--vehicle-max-length-m", type=float, default=6.0)
    parser.add_argument("--vehicle-min-width-m", type=float, default=1.2)
    parser.add_argument("--vehicle-max-width-m", type=float, default=2.8)
    parser.add_argument("--vehicle-min-height-span-m", type=float, default=0.50)
    parser.add_argument("--vehicle-max-height-span-m", type=float, default=2.20)
    parser.add_argument("--vehicle-like-score-threshold", type=float, default=0.65)
    parser.add_argument("--core-owned-min-overlap", type=int, default=25)
    parser.add_argument("--core-owned-top-ratio", type=float, default=1.5)
    parser.add_argument("--review-limit", type=int, default=80)
    parser.add_argument("--plot-anchor-limit", type=int, default=30)
    return parser.parse_args()


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def angular_difference_rad(a: float, b: float) -> float:
    return abs((a - b + math.pi) % (2.0 * math.pi) - math.pi)


def select_pose_aligned_anchor(
    frame_rows: list[dict[str, str]],
    slot: dict[str, Any],
    scale: float = 1.0,
) -> dict[str, Any]:
    slot_center = np.asarray(slot["center_np"], dtype=np.float64)
    best: tuple[float, float, int, dict[str, str], float] | None = None
    for index, row in enumerate(frame_rows):
        pose_xy = np.asarray([float(row["map_x"]), float(row["map_y"])], dtype=np.float64)
        delta = slot_center - pose_xy
        distance_m = float(np.linalg.norm(delta) / max(scale, 1e-9))
        yaw = float(row.get("map_yaw", 0.0))
        angle_to_slot = math.atan2(float(delta[1]), float(delta[0])) if np.linalg.norm(delta) > 1e-12 else yaw
        yaw_alignment_rad = angular_difference_rad(yaw, angle_to_slot)
        key = (distance_m, yaw_alignment_rad, int(row["frame"]))
        if best is None or key < best[:3]:
            best = (distance_m, yaw_alignment_rad, int(row["frame"]), row, index)
    if best is None:
        raise ValueError("cannot select anchor from empty frame rows")
    distance_m, yaw_alignment_rad, frame_id, row, index_float = best
    return {
        "anchor_index": int(index_float),
        "anchor_frame": int(frame_id),
        "anchor_distance_to_slot_m": float(distance_m),
        "anchor_yaw_alignment_deg": float(math.degrees(yaw_alignment_rad)),
        "anchor_pose_map": [float(row["map_x"]), float(row["map_y"]), float(row.get("map_yaw", 0.0))],
    }


def load_consensus_candidates(path: Path, include_states: set[str]) -> list[dict[str, str]]:
    rows = read_csv_rows(path)
    return [row for row in rows if row.get("consensus_state") in include_states]


def fallback_slot_row(slot_id: str, anchor: dict[str, Any], stats: dict[str, Any]) -> dict[str, Any]:
    return {
        "anchor_frame": anchor["anchor_frame"],
        "slot_id": slot_id,
        "support_window_start": stats["support_window_start"],
        "support_window_end": stats["support_window_end"],
        "support_frame_count": stats["support_frame_count"],
        "support_frame_ids": json.dumps(stats["support_frame_ids"]),
        "window_frame_stride": stats["window_frame_stride"],
        "accumulated_point_count": stats["downsampled_points"],
        "non_ground_point_count": stats["non_ground_point_count"],
        "cluster_count": stats["cluster_count"],
        "vehicle_like_cluster_count": stats["vehicle_like_cluster_count"],
        "max_vehicle_like_score": 0.0,
        "cluster_owner_slot": "",
        "cluster_top2_slot": "",
        "core_overlap_count": 0,
        "edge_overlap_count": 0,
        "margin_overlap_count": 0,
        "adjacent_overlap_ratio": 0.0,
        "boundary_ratio": 0.0,
        "cluster_length_m": 0.0,
        "cluster_width_m": 0.0,
        "cluster_height_span_m": 0.0,
        "ownership_status": "none",
        "accumulated_state": "accumulated_no_vehicle_evidence",
        "reason": "target slot is not in candidate range for pose-aligned anchor",
    }


def augment_slot_row(slot_row: dict[str, Any], candidate: dict[str, str], anchor: dict[str, Any], debug_image: str) -> dict[str, Any]:
    out = dict(slot_row)
    out.update(
        {
            "source_consensus_state": candidate.get("consensus_state", ""),
            "source_core_supported_anchor_count": candidate.get("core_supported_anchor_count", ""),
            "source_conflict_anchor_count": candidate.get("conflict_anchor_count", ""),
            "source_core_support_ratio": candidate.get("core_support_ratio", ""),
            "source_conflict_ratio": candidate.get("conflict_ratio", ""),
            "source_owner_stability": candidate.get("owner_stability", ""),
            "source_max_vehicle_like_score": candidate.get("max_vehicle_like_score", ""),
            "anchor_distance_to_slot_m": anchor["anchor_distance_to_slot_m"],
            "anchor_yaw_alignment_deg": anchor["anchor_yaw_alignment_deg"],
            "anchor_pose_map": json.dumps(anchor["anchor_pose_map"]),
            "sampled_frame_count": out.get("support_frame_count", 0),
            "sampled_frame_ids": out.get("support_frame_ids", "[]"),
            "state_by_slot_aligned_accumulation": out.get("accumulated_state", ""),
            "debug_bev_path": debug_image,
        }
    )
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "slot_id",
        "source_consensus_state",
        "anchor_frame",
        "anchor_distance_to_slot_m",
        "anchor_yaw_alignment_deg",
        "sampled_frame_count",
        "sampled_frame_ids",
        "support_window_start",
        "support_window_end",
        "window_frame_stride",
        "state_by_slot_aligned_accumulation",
        "max_vehicle_like_score",
        "cluster_owner_slot",
        "cluster_top2_slot",
        "core_overlap_count",
        "edge_overlap_count",
        "margin_overlap_count",
        "adjacent_overlap_ratio",
        "boundary_ratio",
        "cluster_length_m",
        "cluster_width_m",
        "cluster_height_span_m",
        "ownership_status",
        "source_core_supported_anchor_count",
        "source_conflict_anchor_count",
        "source_core_support_ratio",
        "source_conflict_ratio",
        "source_owner_stability",
        "source_max_vehicle_like_score",
        "debug_bev_path",
        "reason",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def esc(value: Any) -> str:
    return html.escape(str(value))


def state_label(state: str) -> str:
    return {
        "accumulated_vehicle_core_supported": "core-supported vehicle evidence",
        "accumulated_adjacent_conflict": "adjacent-slot conflict",
        "accumulated_boundary_conflict": "boundary conflict",
        "accumulated_static_like": "static-like / not reliable vehicle",
        "accumulated_no_vehicle_evidence": "no stable vehicle evidence",
    }.get(state, state)


def write_report(output_dir: Path, summary: dict[str, Any], rows: list[dict[str, Any]], args: argparse.Namespace) -> None:
    supported = [row for row in rows if row["state_by_slot_aligned_accumulation"] == "accumulated_vehicle_core_supported"]
    conflicts = [
        row
        for row in rows
        if row["state_by_slot_aligned_accumulation"] in {"accumulated_adjacent_conflict", "accumulated_boundary_conflict"}
    ]
    supported.sort(key=lambda row: (-float(row["max_vehicle_like_score"]), -int(row["core_overlap_count"])))
    conflicts.sort(key=lambda row: (-float(row["max_vehicle_like_score"]), -int(row["core_overlap_count"])))

    def table(table_rows: list[dict[str, Any]], limit: int) -> str:
        parts = [
            "<table><tr><th>slot</th><th>source</th><th>anchor</th><th>sampled</th><th>state</th><th>score</th><th>owner/top2</th><th>core/edge/margin</th><th>adjacent</th><th>boundary</th><th>anchor dist/yaw</th><th>debug</th><th>reason</th></tr>"
        ]
        for row in table_rows[:limit]:
            image = row.get("debug_bev_path", "")
            debug = f"<a href='{esc(image)}'>debug</a>" if image else ""
            parts.append(
                "<tr>"
                f"<td>{esc(row['slot_id'])}</td>"
                f"<td>{esc(row['source_consensus_state'])}</td>"
                f"<td>{row['anchor_frame']}</td>"
                f"<td>{row['sampled_frame_count']}</td>"
                f"<td>{esc(state_label(row['state_by_slot_aligned_accumulation']))}</td>"
                f"<td>{float(row['max_vehicle_like_score']):.3f}</td>"
                f"<td>{esc(row['cluster_owner_slot'])}/{esc(row['cluster_top2_slot'])}</td>"
                f"<td>{row['core_overlap_count']}/{row['edge_overlap_count']}/{row['margin_overlap_count']}</td>"
                f"<td>{float(row['adjacent_overlap_ratio']):.3f}</td>"
                f"<td>{float(row['boundary_ratio']):.3f}</td>"
                f"<td>{float(row['anchor_distance_to_slot_m']):.2f}m / {float(row['anchor_yaw_alignment_deg']):.1f}deg</td>"
                f"<td>{debug}</td>"
                f"<td>{esc(row['reason'])}</td>"
                "</tr>"
            )
        parts.append("</table>")
        return "\n".join(parts)

    html_text = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Slot-aligned Multi-frame Accumulation Probe</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 24px; color: #111827; }}
    .note {{ background: #eff6ff; border: 1px solid #bfdbfe; padding: 12px; margin-bottom: 16px; }}
    table {{ border-collapse: collapse; font-size: 12px; width: 100%; }}
    th, td {{ border: 1px solid #d1d5db; padding: 4px 6px; text-align: left; vertical-align: top; }}
    pre {{ background: #f8fafc; border: 1px solid #e2e8f0; padding: 10px; overflow: auto; }}
  </style>
</head>
<body>
  <h1>Slot-aligned Multi-frame Accumulation Probe</h1>
  <div class="note">
    Diagnostic only. This uses pose-selected slot alignment anchors with ±{args.window_before}/{args.window_after} frame windows sampled every {args.window_frame_stride} frames.
    This is slot-aligned temporal evidence, not final occupied/free decision.
  </div>
  <h2>Summary</h2>
  <pre>{esc(json.dumps(summary, indent=2, ensure_ascii=False))}</pre>
  <h2>Top slot-aligned core-supported evidence</h2>
  {table(supported, args.review_limit)}
  <h2>Top slot-aligned boundary / adjacent conflicts</h2>
  {table(conflicts, args.review_limit)}
</body>
</html>
"""
    (output_dir / "slot_aligned_review_report.html").write_text(html_text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    debug_dir = output_dir / "debug"
    accumulation.ensure_dir(output_dir)
    accumulation.ensure_dir(debug_dir)
    base_dir = Path.cwd()
    frame_rows, _ = accumulation.load_frames(args.frames)
    slots, scale = accumulation.load_slots(args.slot_database)
    slot_by_id = {str(slot["slot_id"]): slot for slot in slots}
    center_tree = cKDTree(np.vstack([slot["center_np"] for slot in slots]))
    candidates = load_consensus_candidates(args.consensus_csv, set(args.include_states))
    if not candidates:
        raise SystemExit(f"no consensus candidates selected from {args.consensus_csv}")

    process_cache: dict[int, tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]] = {}
    debug_images: dict[int, str] = {}
    all_rows: list[dict[str, Any]] = []
    all_cluster_rows: list[dict[str, Any]] = []
    frame_stats: list[dict[str, Any]] = []
    cluster_debug_path = output_dir / "slot_aligned_cluster_debug.jsonl"
    with cluster_debug_path.open("w", encoding="utf-8") as cluster_handle:
        for index, candidate in enumerate(candidates):
            slot_id = str(candidate["slot_id"])
            slot = slot_by_id.get(slot_id)
            if slot is None:
                continue
            anchor = select_pose_aligned_anchor(frame_rows, slot, scale=scale)
            anchor_index = int(anchor["anchor_index"])
            if anchor_index not in process_cache:
                process_cache[anchor_index] = accumulation.process_anchor(anchor_index, frame_rows, base_dir, slots, center_tree, scale, args)
                stats, _, cluster_rows, debug = process_cache[anchor_index]
                frame_stats.append(stats)
                for cluster in cluster_rows:
                    cluster_handle.write(json.dumps(cluster) + "\n")
                all_cluster_rows.extend(cluster_rows)
                if len(debug_images) < args.plot_anchor_limit:
                    image_name = f"anchor_{stats['anchor_frame']:06d}_slot_aligned_debug.png"
                    accumulation.draw_anchor_debug(debug_dir / image_name, int(stats["anchor_frame"]), slots, scale, debug)
                    debug_images[anchor_index] = f"debug/{image_name}"
            stats, slot_rows, _, _ = process_cache[anchor_index]
            by_slot = {str(row["slot_id"]): row for row in slot_rows}
            slot_row = by_slot.get(slot_id, fallback_slot_row(slot_id, anchor, stats))
            all_rows.append(augment_slot_row(slot_row, candidate, anchor, debug_images.get(anchor_index, "")))
            if (index + 1) % 25 == 0 or index + 1 == len(candidates):
                print(f"[progress] slots={index + 1}/{len(candidates)} unique_anchors={len(process_cache)}", flush=True)

    all_rows.sort(
        key=lambda row: (
            row["state_by_slot_aligned_accumulation"] != "accumulated_vehicle_core_supported",
            row["state_by_slot_aligned_accumulation"] not in {"accumulated_adjacent_conflict", "accumulated_boundary_conflict"},
            -float(row["max_vehicle_like_score"]),
            -int(row["core_overlap_count"]),
        )
    )
    state_counts = Counter(row["state_by_slot_aligned_accumulation"] for row in all_rows)
    source_counts = Counter(row["source_consensus_state"] for row in all_rows)
    sampled_counts = Counter(int(row["sampled_frame_count"]) for row in all_rows)
    summary = {
        "input": {
            "frames": str(args.frames),
            "slot_database": str(args.slot_database),
            "consensus_csv": str(args.consensus_csv),
            "include_states": args.include_states,
            "window_before": args.window_before,
            "window_after": args.window_after,
            "window_frame_stride": args.window_frame_stride,
            "cluster_eps_m": args.cluster_eps_m,
            "cluster_min_samples": args.cluster_min_samples,
        },
        "candidate_slot_count": len(candidates),
        "output_slot_count": len(all_rows),
        "unique_anchor_count": len(process_cache),
        "cluster_count": len(all_cluster_rows),
        "vehicle_like_cluster_count": int(sum(1 for row in all_cluster_rows if row["vehicle_cluster_risk"] == "vehicle_like")),
        "clear_core_owned_cluster_count": int(sum(1 for row in all_cluster_rows if row["ownership_status"] == "clear_core_owned")),
        "state_counts": dict(state_counts),
        "source_consensus_counts": dict(source_counts),
        "sampled_frame_count_distribution": dict(sorted(sampled_counts.items())),
        "frame_stats": frame_stats,
        "outputs": {
            "slot_aligned_evidence_csv": str(output_dir / "slot_aligned_evidence.csv"),
            "slot_aligned_evidence_json": str(output_dir / "slot_aligned_evidence.json"),
            "slot_aligned_cluster_debug_jsonl": str(cluster_debug_path),
            "slot_aligned_review_report_html": str(output_dir / "slot_aligned_review_report.html"),
        },
    }
    write_csv(output_dir / "slot_aligned_evidence.csv", all_rows)
    write_json(output_dir / "slot_aligned_evidence.json", {"slots": all_rows})
    write_json(output_dir / "slot_aligned_summary.json", summary)
    write_report(output_dir, summary, all_rows, args)
    print(f"[done] output={output_dir}")
    print(json.dumps({k: summary[k] for k in ["candidate_slot_count", "output_slot_count", "unique_anchor_count", "cluster_count", "state_counts", "sampled_frame_count_distribution"]}, indent=2))


if __name__ == "__main__":
    main()
