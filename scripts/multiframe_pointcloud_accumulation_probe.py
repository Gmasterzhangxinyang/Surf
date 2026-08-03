#!/usr/bin/env python3
"""Probe multi-frame LiDAR accumulation for parking-slot vehicle evidence.

This is a diagnostic experiment. It does not update Part 1 final states.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon as MplPolygon

    HAS_MPL = True
except Exception:
    HAS_MPL = False


DEFAULT_FRAMES = Path("outputs/frame_map_dataset/frames.csv")
DEFAULT_SLOT_DATABASE = Path("outputs/full_icpark_allframes_vehicle_cluster/slot_database.json")
DEFAULT_OUTPUT = Path("outputs/multiframe_accumulation_probe_900_1200")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multi-frame point-cloud accumulation probe")
    parser.add_argument("--frames", type=Path, default=DEFAULT_FRAMES)
    parser.add_argument("--slot-database", type=Path, default=DEFAULT_SLOT_DATABASE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--start-frame", type=int, default=900)
    parser.add_argument("--end-frame", type=int, default=1200)
    parser.add_argument("--frame-step", type=int, default=10)
    parser.add_argument("--window-before", type=int, default=10)
    parser.add_argument("--window-after", type=int, default=10)
    parser.add_argument("--window-frame-stride", type=int, default=1)
    parser.add_argument("--voxel-size-m", type=float, default=0.08)
    parser.add_argument("--max-range-m", type=float, default=22.0)
    parser.add_argument("--ground-quantile", type=float, default=0.08)
    parser.add_argument("--obstacle-z-min-m", type=float, default=0.30)
    parser.add_argument("--obstacle-z-max-m", type=float, default=2.50)
    parser.add_argument("--cluster-eps-m", type=float, default=0.35)
    parser.add_argument("--cluster-min-samples", type=int, default=12)
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
    parser.add_argument("--review-limit", type=int, default=30)
    parser.add_argument("--plot-anchor-limit", type=int, default=18)
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def map_units(value_m: float, scale: float) -> float:
    return float(value_m * scale)


def meters(value_map: float, scale: float) -> float:
    return float(value_map / scale)


def points_in_polygon(points: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    if len(points) == 0:
        return np.zeros(0, dtype=bool)
    x = points[:, 0]
    y = points[:, 1]
    inside = np.zeros(len(points), dtype=bool)
    j = len(polygon) - 1
    for i in range(len(polygon)):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        intersect = ((yi > y) != (yj > y)) & (x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi)
        inside ^= intersect
        j = i
    return inside


def bbox_mask(points_xy: np.ndarray, bmin: np.ndarray, bmax: np.ndarray) -> np.ndarray:
    return np.all((points_xy >= bmin) & (points_xy <= bmax), axis=1)


def soft_range_score(value: float, min_value: float, max_value: float) -> float:
    if value <= 0:
        return 0.0
    if min_value <= value <= max_value:
        return 1.0
    if value < min_value:
        return float(np.clip(value / max(min_value, 1e-9), 0.0, 1.0))
    return float(np.clip(max_value / max(value, 1e-9), 0.0, 1.0))


def load_frames(path: Path) -> tuple[list[dict[str, str]], dict[int, dict[str, str]]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    rows = [row for row in rows if row.get("map_points_path")]
    rows.sort(key=lambda row: int(row["frame"]))
    return rows, {int(row["frame"]): row for row in rows}


def resolve_path(path: str, base_dir: Path) -> Path:
    out = Path(path)
    return out if out.is_absolute() else base_dir / out


def load_points(row: dict[str, str], base_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    path = resolve_path(row["map_points_path"], base_dir)
    with np.load(path) as data:
        return data["points_map_xyzi"].astype(np.float64), data["ego_map_pose"].astype(np.float64)


def sample_window_indices(anchor_index: int, frame_rows: list[dict[str, str]], args: argparse.Namespace) -> list[int]:
    stride = max(1, int(getattr(args, "window_frame_stride", 1)))
    before = max(0, int(args.window_before))
    after = max(0, int(args.window_after))
    indices: list[int] = []
    for offset in range(-before, after + 1, stride):
        index = anchor_index + offset
        if 0 <= index < len(frame_rows):
            indices.append(index)
    if 0 <= anchor_index < len(frame_rows) and anchor_index not in indices:
        indices.append(anchor_index)
    return sorted(set(indices))


def load_slots(path: Path) -> tuple[list[dict[str, Any]], float]:
    data = load_json(path)
    slots: list[dict[str, Any]] = []
    for slot in data["slots"]:
        item = dict(slot)
        item["polygon_np"] = np.asarray(slot["polygon_map"], dtype=np.float64)
        item["core_np"] = np.asarray(slot["core_polygon_map"], dtype=np.float64)
        item["inner_np"] = np.asarray(slot["inner_polygon"], dtype=np.float64)
        item["margin_np"] = np.asarray(slot["margin_polygon_map"], dtype=np.float64)
        item["center_np"] = np.asarray(slot["center_map"], dtype=np.float64)
        item["margin_bbox_min"] = item["margin_np"].min(axis=0)
        item["margin_bbox_max"] = item["margin_np"].max(axis=0)
        slots.append(item)
    return slots, float(data["map_units_per_meter"])


def voxel_downsample_zmax(points: np.ndarray, voxel_size_map: float) -> np.ndarray:
    if len(points) == 0:
        return points
    keys = np.floor(points[:, :3] / max(voxel_size_map, 1e-9)).astype(np.int64)
    order = np.lexsort((-points[:, 2], keys[:, 2], keys[:, 1], keys[:, 0]))
    keys_sorted = keys[order]
    keep_sorted = np.ones(len(order), dtype=bool)
    keep_sorted[1:] = np.any(keys_sorted[1:] != keys_sorted[:-1], axis=1)
    return points[order[keep_sorted]]


def cluster_shape_metrics(cluster_xy: np.ndarray, cluster_z: np.ndarray, scale: float) -> dict[str, Any]:
    centered = cluster_xy - cluster_xy.mean(axis=0)
    if len(cluster_xy) >= 3 and np.linalg.norm(centered) > 1e-9:
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        axes = vh[:2]
    else:
        axes = np.eye(2, dtype=np.float64)
    proj = centered @ axes.T
    mins = proj.min(axis=0)
    maxs = proj.max(axis=0)
    dims_map = np.maximum(maxs - mins, 0.0)
    length_map = float(max(dims_map))
    width_map = float(min(dims_map))
    length_m = meters(length_map, scale)
    width_m = meters(width_map, scale)
    height_span_m = float(cluster_z.max() - cluster_z.min()) if len(cluster_z) else 0.0
    long_axis = axes[0] if dims_map[0] >= dims_map[1] else axes[1]
    heading_deg = float(math.degrees(math.atan2(long_axis[1], long_axis[0])))
    corners_local = np.asarray(
        [[mins[0], mins[1]], [maxs[0], mins[1]], [maxs[0], maxs[1]], [mins[0], maxs[1]]],
        dtype=np.float64,
    )
    corners_map = corners_local @ axes + cluster_xy.mean(axis=0)
    return {
        "length_m": length_m,
        "width_m": width_m,
        "height_span_m": height_span_m,
        "footprint_area_m2": float(max(length_m * width_m, 0.0)),
        "heading_deg": heading_deg,
        "obb_corners_map": corners_map.tolist(),
    }


def vehicle_like_score(point_count: int, shape: dict[str, Any], core_ratio: float, ownership_status: str, args: argparse.Namespace) -> tuple[float, str]:
    length_m = float(shape["length_m"])
    width_m = float(shape["width_m"])
    height_m = float(shape["height_span_m"])
    point_score = min(1.0, point_count / max(args.vehicle_min_points, 1))
    length_score = soft_range_score(length_m, args.vehicle_min_length_m, args.vehicle_max_length_m)
    width_score = soft_range_score(width_m, args.vehicle_min_width_m, args.vehicle_max_width_m)
    height_score = soft_range_score(height_m, args.vehicle_min_height_span_m, args.vehicle_max_height_span_m)
    ownership_score = 1.0 if ownership_status == "clear_core_owned" else 0.45 if ownership_status in {"boundary_conflict", "adjacent_slot_conflict"} else 0.25
    score = float(np.clip(0.18 * point_score + 0.23 * length_score + 0.18 * width_score + 0.20 * height_score + 0.13 * core_ratio + 0.08 * ownership_score, 0.0, 1.0))
    linearity = length_m / max(width_m, 1e-9)
    if length_m > args.vehicle_max_length_m * 1.45 or width_m > args.vehicle_max_width_m * 1.45:
        return min(score, 0.55), "too_large_merged"
    if height_m < args.vehicle_min_height_span_m * 0.55:
        return min(score, 0.60), "low_height_structure"
    if point_count < args.vehicle_min_points * 0.5:
        return min(score, 0.55), "too_small_static"
    if linearity > 6.0 and width_m < 0.65:
        return min(score, 0.55), "wall_like_linear"
    if score >= args.vehicle_like_score_threshold:
        return score, "vehicle_like"
    return score, "not_vehicle_like"


def candidate_slot_ids(slots: list[dict[str, Any]], center_tree: cKDTree, origin: np.ndarray, max_dist_map: float) -> list[int]:
    return sorted(int(i) for i in center_tree.query_ball_point(origin, max_dist_map))


def point_slot_counts(points: np.ndarray, slot: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    in_margin = points_in_polygon(points, slot["margin_np"])
    in_inner = points_in_polygon(points, slot["inner_np"])
    in_core = points_in_polygon(points, slot["core_np"])
    return in_core, in_inner & ~in_core, in_margin & ~in_inner


def analyze_cluster(
    cluster_id: int,
    points_xy: np.ndarray,
    points_z: np.ndarray,
    candidate_ids: list[int],
    slots: list[dict[str, Any]],
    scale: float,
    args: argparse.Namespace,
) -> dict[str, Any] | None:
    overlaps: list[dict[str, Any]] = []
    cmin = points_xy.min(axis=0)
    cmax = points_xy.max(axis=0)
    for slot_idx in candidate_ids:
        slot = slots[slot_idx]
        if np.any(cmax < slot["margin_bbox_min"]) or np.any(cmin > slot["margin_bbox_max"]):
            continue
        core, edge, margin = point_slot_counts(points_xy, slot)
        core_count = int(core.sum())
        edge_count = int(edge.sum())
        margin_count = int(margin.sum())
        total = core_count + edge_count + margin_count
        if total <= 0:
            continue
        overlaps.append(
            {
                "slot_idx": slot_idx,
                "slot_id": slot["slot_id"],
                "core_count": core_count,
                "edge_count": edge_count,
                "margin_count": margin_count,
                "total_count": total,
                "distance_to_center": float(np.linalg.norm(points_xy.mean(axis=0) - slot["center_np"])),
            }
        )
    if not overlaps:
        return None
    overlaps.sort(key=lambda row: (int(row["core_count"]), int(row["total_count"]), -float(row["distance_to_center"])), reverse=True)
    top1 = overlaps[0]
    top2 = overlaps[1] if len(overlaps) > 1 else None
    top1_core = int(top1["core_count"])
    top2_core = int(top2["core_count"]) if top2 else 0
    edge_margin = int(top1["edge_count"]) + int(top1["margin_count"])
    adjacent_count = sum(int(row["total_count"]) for row in overlaps[1:])
    adjacent_overlap_ratio = adjacent_count / max(adjacent_count + int(top1["total_count"]), 1)
    boundary_ratio = edge_margin / max(int(top1["total_count"]), 1)
    if (
        top1_core >= args.core_owned_min_overlap
        and top1_core >= args.core_owned_top_ratio * max(top2_core, 1)
        and top1_core >= edge_margin
        and adjacent_overlap_ratio <= 0.30
    ):
        ownership = "clear_core_owned"
    elif top2_core > 0 and top1_core < args.core_owned_top_ratio * top2_core:
        ownership = "adjacent_slot_conflict"
    elif edge_margin > top1_core or boundary_ratio > 0.45:
        ownership = "boundary_conflict"
    elif top1_core <= 0 and edge_margin > 0:
        ownership = "margin_only"
    else:
        ownership = "slot_vs_lane_possible"
    shape = cluster_shape_metrics(points_xy, points_z, scale)
    core_ratio = top1_core / max(int(top1["total_count"]), 1)
    score, risk = vehicle_like_score(len(points_xy), shape, core_ratio, ownership, args)
    if risk == "vehicle_like" and ownership == "clear_core_owned":
        state = "accumulated_vehicle_core_supported"
        reason = "vehicle-like cluster is mainly inside target slot core"
    elif risk == "vehicle_like" and adjacent_overlap_ratio > 0.30:
        state = "accumulated_adjacent_conflict"
        reason = "vehicle-like cluster overlaps adjacent slot ownership"
    elif risk == "vehicle_like" and (boundary_ratio > 0.45 or ownership in {"boundary_conflict", "margin_only"}):
        state = "accumulated_boundary_conflict"
        reason = "vehicle-like cluster is concentrated near edge or margin"
    elif risk in {"too_small_static", "wall_like_linear", "low_height_structure", "too_large_merged"}:
        state = "accumulated_static_like"
        reason = f"cluster risk is {risk}"
    else:
        state = "accumulated_no_vehicle_evidence"
        reason = "cluster does not pass vehicle-like geometry threshold"
    return {
        "cluster_id": int(cluster_id),
        "point_count": int(len(points_xy)),
        "cluster_center_map": points_xy.mean(axis=0).tolist(),
        "cluster_z_min": float(points_z.min()),
        "cluster_z_max": float(points_z.max()),
        "cluster_length_m": float(shape["length_m"]),
        "cluster_width_m": float(shape["width_m"]),
        "cluster_height_span_m": float(shape["height_span_m"]),
        "cluster_footprint_area_m2": float(shape["footprint_area_m2"]),
        "cluster_heading_deg": float(shape["heading_deg"]),
        "cluster_obb_corners_map": shape["obb_corners_map"],
        "vehicle_like_score": float(score),
        "vehicle_cluster_risk": risk,
        "cluster_owner_slot": str(top1["slot_id"]),
        "cluster_top2_slot": str(top2["slot_id"]) if top2 else "",
        "core_overlap_count": top1_core,
        "edge_overlap_count": int(top1["edge_count"]),
        "margin_overlap_count": int(top1["margin_count"]),
        "top2_core_overlap_count": top2_core,
        "adjacent_overlap_ratio": float(adjacent_overlap_ratio),
        "boundary_ratio": float(boundary_ratio),
        "ownership_status": ownership,
        "accumulated_state": state,
        "reason": reason,
    }


def accumulate_window(
    anchor_index: int,
    frame_rows: list[dict[str, str]],
    base_dir: Path,
    scale: float,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, int, int, list[int]]:
    anchor_points, anchor_pose = load_points(frame_rows[anchor_index], base_dir)
    origin = anchor_pose[:2]
    max_range_map = map_units(args.max_range_m, scale)
    parts: list[np.ndarray] = []
    frames: list[int] = []
    selected_indices = sample_window_indices(anchor_index, frame_rows, args)
    for index in selected_indices:
        row = frame_rows[index]
        points, _ = load_points(row, base_dir)
        finite = np.isfinite(points).all(axis=1)
        points = points[finite]
        dist = np.linalg.norm(points[:, :2] - origin[None, :], axis=1)
        points = points[dist <= max_range_map]
        if len(points):
            parts.append(points)
        frames.append(int(row["frame"]))
    if not parts:
        return np.empty((0, 4), dtype=np.float64), anchor_pose, int(frame_rows[selected_indices[0]]["frame"]), int(frame_rows[selected_indices[-1]]["frame"]), frames
    return np.vstack(parts), anchor_pose, int(frame_rows[selected_indices[0]]["frame"]), int(frame_rows[selected_indices[-1]]["frame"]), frames


def process_anchor(
    anchor_index: int,
    frame_rows: list[dict[str, str]],
    base_dir: Path,
    slots: list[dict[str, Any]],
    center_tree: cKDTree,
    scale: float,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    anchor_frame = int(frame_rows[anchor_index]["frame"])
    accumulated, anchor_pose, win_start, win_end, window_frames = accumulate_window(anchor_index, frame_rows, base_dir, scale, args)
    voxel_size_map = map_units(args.voxel_size_m, scale)
    downsampled = voxel_downsample_zmax(accumulated, voxel_size_map)
    z = downsampled[:, 2] if len(downsampled) else np.empty(0, dtype=np.float64)
    ground_z = float(np.quantile(z, args.ground_quantile)) if len(z) else 0.0
    obstacle_mask = (z >= ground_z + args.obstacle_z_min_m) & (z <= ground_z + args.obstacle_z_max_m) if len(z) else np.zeros(0, dtype=bool)
    obstacle = downsampled[obstacle_mask]
    candidate_ids = candidate_slot_ids(slots, center_tree, anchor_pose[:2], map_units(args.max_range_m, scale))
    cluster_rows: list[dict[str, Any]] = []
    if len(obstacle) >= args.cluster_min_samples:
        labels = DBSCAN(eps=map_units(args.cluster_eps_m, scale), min_samples=args.cluster_min_samples).fit_predict(obstacle[:, :2])
        for cluster_id in sorted(int(value) for value in np.unique(labels) if value >= 0):
            cluster = obstacle[labels == cluster_id]
            analyzed = analyze_cluster(cluster_id, cluster[:, :2], cluster[:, 2], candidate_ids, slots, scale, args)
            if analyzed is None:
                continue
            analyzed.update(
                {
                    "anchor_frame": anchor_frame,
                    "support_window_start": win_start,
                    "support_window_end": win_end,
                    "support_frame_count": len(window_frames),
                    "support_frame_ids": window_frames,
                    "window_frame_stride": int(max(1, getattr(args, "window_frame_stride", 1))),
                    "accumulated_point_count": int(len(downsampled)),
                    "non_ground_point_count": int(len(obstacle)),
                    "anchor_pose": anchor_pose.tolist(),
                }
            )
            cluster_rows.append(analyzed)
    best_by_slot: dict[str, dict[str, Any]] = {}
    for row in cluster_rows:
        slot_id = str(row["cluster_owner_slot"])
        current = best_by_slot.get(slot_id)
        key = (float(row["vehicle_like_score"]), int(row["core_overlap_count"]), int(row["point_count"]))
        if current is None or key > (float(current["vehicle_like_score"]), int(current["core_overlap_count"]), int(current["point_count"])):
            best_by_slot[slot_id] = row
    slot_rows: list[dict[str, Any]] = []
    for slot_idx in candidate_ids:
        slot = slots[slot_idx]
        cluster = best_by_slot.get(str(slot["slot_id"]))
        if cluster is None:
            slot_rows.append(
                {
                    "anchor_frame": anchor_frame,
                    "slot_id": slot["slot_id"],
                    "support_window_start": win_start,
                    "support_window_end": win_end,
                    "support_frame_count": len(window_frames),
                    "support_frame_ids": json.dumps(window_frames),
                    "window_frame_stride": int(max(1, getattr(args, "window_frame_stride", 1))),
                    "accumulated_point_count": int(len(downsampled)),
                    "non_ground_point_count": int(len(obstacle)),
                    "cluster_count": int(len(cluster_rows)),
                    "vehicle_like_cluster_count": int(sum(1 for row in cluster_rows if row["vehicle_cluster_risk"] == "vehicle_like")),
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
                    "reason": "no cluster owned by this slot in accumulated window",
                }
            )
            continue
        slot_rows.append(
            {
                "anchor_frame": anchor_frame,
                "slot_id": slot["slot_id"],
                "support_window_start": win_start,
                "support_window_end": win_end,
                "support_frame_count": len(window_frames),
                "support_frame_ids": json.dumps(window_frames),
                "window_frame_stride": int(max(1, getattr(args, "window_frame_stride", 1))),
                "accumulated_point_count": int(len(downsampled)),
                "non_ground_point_count": int(len(obstacle)),
                "cluster_count": int(len(cluster_rows)),
                "vehicle_like_cluster_count": int(sum(1 for row in cluster_rows if row["vehicle_cluster_risk"] == "vehicle_like")),
                "max_vehicle_like_score": float(cluster["vehicle_like_score"]),
                "cluster_owner_slot": cluster["cluster_owner_slot"],
                "cluster_top2_slot": cluster["cluster_top2_slot"],
                "core_overlap_count": int(cluster["core_overlap_count"]),
                "edge_overlap_count": int(cluster["edge_overlap_count"]),
                "margin_overlap_count": int(cluster["margin_overlap_count"]),
                "adjacent_overlap_ratio": float(cluster["adjacent_overlap_ratio"]),
                "boundary_ratio": float(cluster["boundary_ratio"]),
                "cluster_length_m": float(cluster["cluster_length_m"]),
                "cluster_width_m": float(cluster["cluster_width_m"]),
                "cluster_height_span_m": float(cluster["cluster_height_span_m"]),
                "ownership_status": cluster["ownership_status"],
                "accumulated_state": cluster["accumulated_state"],
                "reason": cluster["reason"],
            }
        )
    stats = {
        "anchor_frame": anchor_frame,
        "support_window_start": win_start,
        "support_window_end": win_end,
        "support_frame_count": len(window_frames),
        "support_frame_ids": window_frames,
        "window_frame_stride": int(max(1, getattr(args, "window_frame_stride", 1))),
        "raw_accumulated_points": int(len(accumulated)),
        "downsampled_points": int(len(downsampled)),
        "non_ground_point_count": int(len(obstacle)),
        "candidate_slot_count": int(len(candidate_ids)),
        "cluster_count": int(len(cluster_rows)),
        "vehicle_like_cluster_count": int(sum(1 for row in cluster_rows if row["vehicle_cluster_risk"] == "vehicle_like")),
        "clear_core_owned_cluster_count": int(sum(1 for row in cluster_rows if row["ownership_status"] == "clear_core_owned")),
        "anchor_pose": anchor_pose.tolist(),
    }
    debug = {
        "points": downsampled,
        "obstacle": obstacle,
        "anchor_pose": anchor_pose,
        "candidate_ids": candidate_ids,
        "cluster_rows": cluster_rows,
    }
    return stats, slot_rows, cluster_rows, debug


def write_csv_rows(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def draw_anchor_debug(
    path: Path,
    frame: int,
    slots: list[dict[str, Any]],
    scale: float,
    debug: dict[str, Any],
) -> None:
    if not HAS_MPL:
        return
    points = debug["points"]
    obstacle = debug["obstacle"]
    pose = debug["anchor_pose"]
    cluster_rows = debug["cluster_rows"]
    candidate_ids = set(debug["candidate_ids"])
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=160)
    all_slot_pts = np.vstack([slot["polygon_np"] for slot in slots])

    ax = axes[0]
    for idx, slot in enumerate(slots):
        if idx in candidate_ids:
            face, edge, alpha, lw = "#dbeafe", "#60a5fa", 0.28, 0.35
        else:
            face, edge, alpha, lw = "#e5e7eb", "#cbd5e1", 0.14, 0.20
        ax.add_patch(MplPolygon(slot["polygon_np"], closed=True, facecolor=face, edgecolor=edge, alpha=alpha, linewidth=lw))
    if len(points):
        sample = points[:: max(1, len(points) // 12000)]
        ax.scatter(sample[:, 0], sample[:, 1], s=0.35, color="#64748b", alpha=0.24, linewidths=0, label="accumulated points")
    ax.scatter([pose[0]], [pose[1]], s=30, color="#111827", zorder=10, label="anchor pose")
    arrow_len = 2.5 * scale
    ax.arrow(pose[0], pose[1], arrow_len * math.cos(pose[2]), arrow_len * math.sin(pose[2]), width=0.006, head_width=0.06, color="#111827", zorder=11)
    bmin = all_slot_pts.min(axis=0) - 0.45
    bmax = all_slot_pts.max(axis=0) + 0.45
    ax.set_xlim(float(bmin[0]), float(bmax[0]))
    ax.set_ylim(float(bmin[1]), float(bmax[1]))
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(f"Global map accumulation anchor {frame}")
    ax.legend(loc="upper right", fontsize=7)

    ax = axes[1]
    nearby_points = [pose[:2].reshape(1, 2)]
    if len(points):
        nearby_points.append(points[:, :2])
    for idx in candidate_ids:
        slot = slots[idx]
        ax.add_patch(MplPolygon(slot["polygon_np"], closed=True, facecolor="#dbeafe", edgecolor="#2563eb", alpha=0.30, linewidth=0.45))
    if len(obstacle):
        ax.scatter(obstacle[:, 0], obstacle[:, 1], s=0.8, color="#ef4444", alpha=0.25, linewidths=0, label="non-ground accumulated")
        nearby_points.append(obstacle[:, :2])
    for row in cluster_rows:
        corners = np.asarray(row.get("cluster_obb_corners_map", []), dtype=np.float64)
        if len(corners) >= 3:
            color = "#dc2626" if row["accumulated_state"] == "accumulated_vehicle_core_supported" else "#7c3aed"
            ax.add_patch(MplPolygon(corners, closed=True, facecolor="none", edgecolor=color, linewidth=1.3, alpha=0.95))
            center = np.asarray(row["cluster_center_map"], dtype=np.float64)
            ax.text(center[0], center[1], str(row["cluster_owner_slot"]).replace("slot_", ""), fontsize=6, color=color)
            nearby_points.append(corners)
    ax.scatter([pose[0]], [pose[1]], s=36, color="#111827", zorder=10)
    ax.arrow(pose[0], pose[1], arrow_len * math.cos(pose[2]), arrow_len * math.sin(pose[2]), width=0.006, head_width=0.06, color="#111827", zorder=11)
    ext = np.vstack(nearby_points)
    bmin = ext.min(axis=0) - 0.35
    bmax = ext.max(axis=0) + 0.35
    ax.set_xlim(float(bmin[0]), float(bmax[0]))
    ax.set_ylim(float(bmin[1]), float(bmax[1]))
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Zoom: accumulated clusters and nearby slots")
    ax.legend(loc="upper right", fontsize=7)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def esc(value: Any) -> str:
    return html.escape(str(value))


def write_report(output_dir: Path, summary: dict[str, Any], slot_rows: list[dict[str, Any]], debug_images: dict[int, str], args: argparse.Namespace) -> None:
    supported = [row for row in slot_rows if row["accumulated_state"] == "accumulated_vehicle_core_supported"]
    conflicts = [row for row in slot_rows if row["accumulated_state"] in {"accumulated_boundary_conflict", "accumulated_adjacent_conflict"}]
    supported.sort(key=lambda row: (-float(row["max_vehicle_like_score"]), -int(row["core_overlap_count"])))
    conflicts.sort(key=lambda row: (-float(row["max_vehicle_like_score"]), -int(row["core_overlap_count"])))

    def table(rows: list[dict[str, Any]], limit: int) -> str:
        parts = [
            "<table><tr><th>anchor</th><th>slot</th><th>state</th><th>score</th><th>owner/top2</th><th>core/edge/margin</th><th>adjacent</th><th>boundary</th><th>L/W/H</th><th>reason</th></tr>"
        ]
        for row in rows[:limit]:
            parts.append(
                "<tr>"
                f"<td>{row['anchor_frame']}</td>"
                f"<td>{esc(row['slot_id'])}</td>"
                f"<td>{esc(row['accumulated_state'])}</td>"
                f"<td>{float(row['max_vehicle_like_score']):.3f}</td>"
                f"<td>{esc(row['cluster_owner_slot'])}/{esc(row['cluster_top2_slot'])}</td>"
                f"<td>{row['core_overlap_count']}/{row['edge_overlap_count']}/{row['margin_overlap_count']}</td>"
                f"<td>{float(row['adjacent_overlap_ratio']):.3f}</td>"
                f"<td>{float(row['boundary_ratio']):.3f}</td>"
                f"<td>{float(row['cluster_length_m']):.2f}/{float(row['cluster_width_m']):.2f}/{float(row['cluster_height_span_m']):.2f}</td>"
                f"<td>{esc(row['reason'])}</td>"
                "</tr>"
            )
        parts.append("</table>")
        return "\n".join(parts)

    image_parts = []
    for frame, image_name in list(debug_images.items())[: args.plot_anchor_limit]:
        image_parts.append(f"<section><h3>Anchor frame {frame}</h3><img src='debug/{esc(image_name)}'></section>")

    html_text = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Multi-frame Accumulation Probe</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 24px; color: #111827; }}
    .note {{ background: #eff6ff; border: 1px solid #bfdbfe; padding: 12px; margin-bottom: 16px; }}
    table {{ border-collapse: collapse; font-size: 12px; width: 100%; }}
    th, td {{ border: 1px solid #d1d5db; padding: 4px 6px; text-align: left; vertical-align: top; }}
    img {{ max-width: 100%; border: 1px solid #d1d5db; background: white; }}
    section {{ border-top: 1px solid #d1d5db; padding: 14px 0; }}
    pre {{ background: #f8fafc; border: 1px solid #e2e8f0; padding: 10px; overflow: auto; }}
  </style>
</head>
<body>
  <h1>Multi-frame Point-cloud Accumulation Probe</h1>
  <div class="note">
    Diagnostic only. This report does not change Part 1 free/occupied states.
    Red OBB means accumulated vehicle core-supported; purple OBB means conflict or other cluster.
  </div>
  <h2>Summary</h2>
  <pre>{esc(json.dumps(summary, indent=2, ensure_ascii=False))}</pre>
  <h2>Top accumulated vehicle core-supported</h2>
  {table(supported, args.review_limit)}
  <h2>Top boundary / adjacent conflicts</h2>
  {table(conflicts, args.review_limit)}
  <h2>Debug Maps</h2>
  {''.join(image_parts)}
</body>
</html>
"""
    (output_dir / "accumulated_review_report.html").write_text(html_text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    debug_dir = output_dir / "debug"
    ensure_dir(output_dir)
    ensure_dir(debug_dir)
    base_dir = Path.cwd()
    frame_rows_all, _ = load_frames(args.frames)
    frame_rows = [row for row in frame_rows_all if args.start_frame <= int(row["frame"]) <= args.end_frame]
    if not frame_rows:
        raise SystemExit("no frames selected")
    slots, scale = load_slots(args.slot_database)
    center_tree = cKDTree(np.vstack([slot["center_np"] for slot in slots]))
    anchor_indices = list(range(0, len(frame_rows), max(1, args.frame_step)))

    all_slot_rows: list[dict[str, Any]] = []
    all_cluster_rows: list[dict[str, Any]] = []
    frame_stats: list[dict[str, Any]] = []
    debug_images: dict[int, str] = {}
    cluster_debug_path = output_dir / "accumulated_cluster_debug.jsonl"
    with cluster_debug_path.open("w", encoding="utf-8") as cluster_handle:
        for count, anchor_index in enumerate(anchor_indices):
            stats, slot_rows, cluster_rows, debug = process_anchor(anchor_index, frame_rows, base_dir, slots, center_tree, scale, args)
            frame_stats.append(stats)
            all_slot_rows.extend(slot_rows)
            all_cluster_rows.extend(cluster_rows)
            for row in cluster_rows:
                cluster_handle.write(json.dumps(row) + "\n")
            if count < args.plot_anchor_limit:
                image_name = f"frame_{stats['anchor_frame']:06d}_accumulated_debug.png"
                draw_anchor_debug(debug_dir / image_name, int(stats["anchor_frame"]), slots, scale, debug)
                debug_images[int(stats["anchor_frame"])] = image_name
            if (count + 1) % 10 == 0 or count + 1 == len(anchor_indices):
                print(f"[progress] anchors={count + 1}/{len(anchor_indices)} slots={len(all_slot_rows)} clusters={len(all_cluster_rows)}", flush=True)

    slot_fields = [
        "anchor_frame",
        "slot_id",
        "support_window_start",
        "support_window_end",
        "support_frame_count",
        "support_frame_ids",
        "window_frame_stride",
        "accumulated_point_count",
        "non_ground_point_count",
        "cluster_count",
        "vehicle_like_cluster_count",
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
        "accumulated_state",
        "reason",
    ]
    write_csv_rows(output_dir / "accumulated_slot_evidence.csv", all_slot_rows, slot_fields)
    write_json(output_dir / "accumulated_slot_evidence.json", {"slots": all_slot_rows})
    state_counts = Counter(row["accumulated_state"] for row in all_slot_rows)
    ownership_counts = Counter(row["ownership_status"] for row in all_slot_rows)
    cluster_state_counts = Counter(row["accumulated_state"] for row in all_cluster_rows)
    summary = {
        "input": {
            "frames": str(args.frames),
            "slot_database": str(args.slot_database),
            "start_frame": args.start_frame,
            "end_frame": args.end_frame,
            "frame_step": args.frame_step,
            "window_before": args.window_before,
            "window_after": args.window_after,
            "window_frame_stride": args.window_frame_stride,
            "voxel_size_m": args.voxel_size_m,
            "max_range_m": args.max_range_m,
        },
        "anchor_frame_count": len(anchor_indices),
        "slot_evidence_rows": len(all_slot_rows),
        "cluster_count": len(all_cluster_rows),
        "vehicle_like_cluster_count": int(sum(1 for row in all_cluster_rows if row["vehicle_cluster_risk"] == "vehicle_like")),
        "clear_core_owned_cluster_count": int(sum(1 for row in all_cluster_rows if row["ownership_status"] == "clear_core_owned")),
        "slot_state_counts": dict(state_counts),
        "slot_ownership_counts": dict(ownership_counts),
        "cluster_state_counts": dict(cluster_state_counts),
        "frame_stats": frame_stats,
        "outputs": {
            "accumulated_cluster_debug_jsonl": str(cluster_debug_path),
            "accumulated_slot_evidence_csv": str(output_dir / "accumulated_slot_evidence.csv"),
            "accumulated_slot_evidence_json": str(output_dir / "accumulated_slot_evidence.json"),
            "accumulated_review_report_html": str(output_dir / "accumulated_review_report.html"),
        },
    }
    write_json(output_dir / "accumulated_summary.json", summary)
    write_report(output_dir, summary, all_slot_rows, debug_images, args)
    print(f"[done] output={output_dir}")
    print(json.dumps({k: summary[k] for k in ["anchor_frame_count", "slot_evidence_rows", "cluster_count", "vehicle_like_cluster_count", "clear_core_owned_cluster_count", "slot_state_counts", "cluster_state_counts"]}, indent=2))


if __name__ == "__main__":
    main()
