#!/usr/bin/env python3
"""Pose-assisted diagnostics for Part 2 case review.

This script does not change Part 1 states. It only adds pose-derived
disambiguation signals for the cases already handed to Part 2.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


DEFAULT_PART1 = Path("outputs/part1_slot_scoring_1000_boundary_fixed")
DEFAULT_PART2 = Path("outputs/part2_case_review_pack")
DEFAULT_FRAMES = Path("outputs/frame_map_dataset/frames.csv")
DEFAULT_OUTPUT = DEFAULT_PART2 / "pose_assisted_review"
IMAGE_W = 1280
IMAGE_H = 720


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build pose-assisted case diagnostics")
    parser.add_argument("--part1-dir", type=Path, default=DEFAULT_PART1)
    parser.add_argument("--part2-pack", type=Path, default=DEFAULT_PART2)
    parser.add_argument("--frames", type=Path, default=DEFAULT_FRAMES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--ground-quantile", type=float, default=0.08)
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: object) -> None:
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def load_frames(path: Path) -> dict[int, dict[str, str]]:
    with path.open("r", newline="") as handle:
        return {int(row["frame"]): row for row in csv.DictReader(handle)}


def load_evidence(path: Path, slot_ids: set[str]) -> dict[str, list[dict[str, object]]]:
    by_slot: dict[str, list[dict[str, object]]] = {slot_id: [] for slot_id in slot_ids}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            slot_id = str(row.get("slot_id"))
            if slot_id in by_slot:
                by_slot[slot_id].append(row)
    for rows in by_slot.values():
        rows.sort(key=lambda r: int(r["frame_id"]))
    return by_slot


def yaw_to_rot(yaw: float) -> np.ndarray:
    c = math.cos(yaw)
    s = math.sin(yaw)
    return np.array([[c, -s], [s, c]], dtype=np.float64)


def map_xy_to_lidar_xy(points_map: np.ndarray, ego_pose: np.ndarray, scale: float) -> np.ndarray:
    delta = (points_map - ego_pose[:2]) / max(scale, 1e-9)
    return delta @ yaw_to_rot(-float(ego_pose[2])).T


def lidar_to_camera(points_lidar_xyz: np.ndarray) -> np.ndarray:
    x_l = points_lidar_xyz[:, 0]
    y_l = points_lidar_xyz[:, 1]
    z_l = points_lidar_xyz[:, 2]
    cam_xyz = np.column_stack([-y_l, -z_l, x_l]).astype(np.float64)
    cam_xyz += np.array([0.6, 0.0, -0.07], dtype=np.float64)
    return cam_xyz


def project_camera(points_cam_xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    fx = 527.525085
    fy = 527.525085
    cx = 636.297913
    cy = 357.787354
    z = points_cam_xyz[:, 2]
    valid = z > 1e-4
    uv = np.full((len(points_cam_xyz), 2), np.nan, dtype=np.float64)
    uv[valid, 0] = fx * points_cam_xyz[valid, 0] / z[valid] + cx
    uv[valid, 1] = fy * points_cam_xyz[valid, 1] / z[valid] + cy
    return uv, valid


def load_points_for_frame(frame_id: int, frame_rows: dict[int, dict[str, str]], repo_root: Path) -> tuple[np.ndarray, np.ndarray]:
    row = frame_rows[frame_id]
    path = Path(row["map_points_path"])
    if not path.is_absolute():
        path = repo_root / path
    data = np.load(path)
    return data["points_map_xyzi"].astype(np.float64), data["ego_map_pose"].astype(np.float64)


def polygon_to_uv(polygon_map: np.ndarray, ego_pose: np.ndarray, scale: float, z_lidar: float) -> tuple[np.ndarray, np.ndarray]:
    xy_lidar = map_xy_to_lidar_xy(polygon_map, ego_pose, scale)
    xyz_lidar = np.column_stack([xy_lidar, np.full(len(xy_lidar), z_lidar, dtype=np.float64)])
    return project_camera(lidar_to_camera(xyz_lidar))


def polygon_area(uv: np.ndarray) -> float:
    valid = np.isfinite(uv).all(axis=1)
    pts = uv[valid]
    if len(pts) < 3:
        return 0.0
    x = pts[:, 0]
    y = pts[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) * 0.5)


def projection_quality(uv_arrays: list[np.ndarray]) -> dict[str, object]:
    finite = []
    in_image_points = 0
    finite_points = 0
    for uv in uv_arrays:
        valid = np.isfinite(uv).all(axis=1)
        if valid.any():
            pts = uv[valid]
            finite.append(pts)
            finite_points += int(len(pts))
            inside = (pts[:, 0] >= 0) & (pts[:, 0] < IMAGE_W) & (pts[:, 1] >= 0) & (pts[:, 1] < IMAGE_H)
            in_image_points += int(inside.sum())
    if not finite:
        return {"score": 0.0, "in_image_points": 0, "finite_points": 0, "area": 0.0}
    pts = np.vstack(finite)
    x0 = max(0.0, float(pts[:, 0].min()))
    y0 = max(0.0, float(pts[:, 1].min()))
    x1 = min(float(IMAGE_W), float(pts[:, 0].max()))
    y1 = min(float(IMAGE_H), float(pts[:, 1].max()))
    overlap_area = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    raw_area = sum(polygon_area(uv) for uv in uv_arrays)
    point_score = in_image_points / max(finite_points, 1)
    area_score = min(1.0, overlap_area / max(raw_area, 1.0))
    return {
        "score": float(0.65 * area_score + 0.35 * point_score),
        "in_image_points": int(in_image_points),
        "finite_points": int(finite_points),
        "area": float(overlap_area),
    }


def frame_pose(row: dict[str, str]) -> np.ndarray:
    return np.array([float(row["map_x"]), float(row["map_y"]), float(row["map_yaw"])], dtype=np.float64)


def angle_wrap_deg(value: float) -> float:
    return (value + 180.0) % 360.0 - 180.0


def pose_metrics(slot: dict[str, object], rows: list[dict[str, object]], frame_rows: dict[int, dict[str, str]], scale: float) -> dict[str, object]:
    center = np.asarray(slot["center_map"], dtype=np.float64)
    local_xy = []
    frame_ids = []
    for ev in rows:
        frame_id = int(ev["frame_id"])
        if frame_id not in frame_rows:
            continue
        pose = frame_pose(frame_rows[frame_id])
        local_xy.append(map_xy_to_lidar_xy(center.reshape(1, 2), pose, scale)[0])
        frame_ids.append(frame_id)
    if not local_xy:
        return {
            "pose_frame_count": 0,
            "min_distance_m": None,
            "median_distance_m": None,
            "min_abs_bearing_deg": None,
            "median_abs_bearing_deg": None,
            "front_fov_frame_count": 0,
            "side_view_frame_count": 0,
            "pose_baseline_m": 0.0,
            "bearing_span_deg": 0.0,
            "frame_span": 0,
        }
    arr = np.asarray(local_xy, dtype=np.float64)
    dist = np.linalg.norm(arr, axis=1)
    bearing = np.degrees(np.arctan2(arr[:, 1], arr[:, 0]))
    front = (arr[:, 0] > 0.0) & (np.abs(bearing) <= 55.0)
    side = (arr[:, 0] > 0.0) & (np.abs(bearing) > 55.0) & (np.abs(bearing) <= 110.0)
    if len(arr) >= 2:
        xy = arr
        baseline = float(np.linalg.norm(xy.max(axis=0) - xy.min(axis=0)))
    else:
        baseline = 0.0
    return {
        "pose_frame_count": int(len(arr)),
        "min_distance_m": float(dist.min()),
        "median_distance_m": float(np.median(dist)),
        "min_abs_bearing_deg": float(np.abs(bearing).min()),
        "median_abs_bearing_deg": float(np.median(np.abs(bearing))),
        "front_fov_frame_count": int(front.sum()),
        "side_view_frame_count": int(side.sum()),
        "pose_baseline_m": baseline,
        "bearing_span_deg": float(np.percentile(bearing, 95) - np.percentile(bearing, 5)) if len(bearing) >= 2 else 0.0,
        "frame_span": int(max(frame_ids) - min(frame_ids)) if frame_ids else 0,
    }


def camera_metrics(
    slot: dict[str, object],
    rows: list[dict[str, object]],
    frame_rows: dict[int, dict[str, str]],
    scale: float,
    repo_root: Path,
    ground_cache: dict[int, tuple[np.ndarray, np.ndarray, float]],
    ground_quantile: float,
) -> dict[str, object]:
    best = {"score": 0.0, "frame_id": None, "in_image_points": 0, "area": 0.0}
    visible_count = 0
    core = np.asarray(slot["core_polygon_map"], dtype=np.float64)
    inner = np.asarray(slot["inner_polygon"], dtype=np.float64)
    margin = np.asarray(slot["margin_polygon_map"], dtype=np.float64)
    for ev in rows:
        frame_id = int(ev["frame_id"])
        if frame_id not in frame_rows:
            continue
        try:
            if frame_id not in ground_cache:
                points, ego_pose = load_points_for_frame(frame_id, frame_rows, repo_root)
                ground_cache[frame_id] = (points, ego_pose, float(np.quantile(points[:, 2], ground_quantile)))
            _, ego_pose, ground_z = ground_cache[frame_id]
        except FileNotFoundError:
            continue
        quality = projection_quality(
            [
                polygon_to_uv(core, ego_pose, scale, ground_z)[0],
                polygon_to_uv(inner, ego_pose, scale, ground_z)[0],
                polygon_to_uv(margin, ego_pose, scale, ground_z)[0],
            ]
        )
        if int(quality["in_image_points"]) >= 3 and float(quality["score"]) > 0.01:
            visible_count += 1
        if float(quality["score"]) > float(best["score"]):
            best = {
                "score": float(quality["score"]),
                "frame_id": frame_id,
                "in_image_points": int(quality["in_image_points"]),
                "area": float(quality["area"]),
            }
    return {
        "camera_visible_frame_count": int(visible_count),
        "best_camera_frame_id": best["frame_id"],
        "best_camera_projection_score": float(best["score"]),
        "best_camera_in_image_points": int(best["in_image_points"]),
        "best_camera_overlap_area_px": float(best["area"]),
    }


def evidence_metrics(slot_id: str, rows: list[dict[str, object]]) -> dict[str, object]:
    support = len(rows)
    ray_frames = sum(float(r.get("ray_free_ratio", 0.0)) >= 0.20 for r in rows)
    visible_frames = sum(float(r.get("visibility_score", 0.0)) >= 0.20 for r in rows)
    clear_core_rows = [r for r in rows if str(r.get("cluster_ownership_status", "")) == "clear_core_owned"]
    boundary_rows = [
        r
        for r in rows
        if str(r.get("cluster_ownership_status", "")) in {"boundary_conflict", "adjacent_slot_conflict", "margin_only"}
        or float(r.get("boundary_ratio", 0.0)) > 0.45
        or float(r.get("adjacent_overlap_ratio", 0.0)) > 0.30
    ]
    top1_counter = Counter(str(r.get("cluster_top1_slot")) for r in clear_core_rows if r.get("cluster_top1_slot"))
    top1_slot, top1_count = (top1_counter.most_common(1)[0] if top1_counter else (None, 0))
    return {
        "support_frame_count": int(support),
        "ray_support_frames": int(ray_frames),
        "visibility_support_frames": int(visible_frames),
        "clear_core_owned_frames": int(len(clear_core_rows)),
        "boundary_conflict_like_frames": int(len(boundary_rows)),
        "boundary_conflict_frame_ratio": float(len(boundary_rows) / max(support, 1)),
        "dominant_core_owner_slot": top1_slot,
        "dominant_core_owner_frames": int(top1_count),
        "dominant_core_owner_ratio": float(top1_count / max(len(clear_core_rows), 1)),
        "max_ray_free_ratio": float(max((float(r.get("ray_free_ratio", 0.0)) for r in rows), default=0.0)),
        "max_visibility_score": float(max((float(r.get("visibility_score", 0.0)) for r in rows), default=0.0)),
        "max_core_hit_cell_ratio": float(max((float(r.get("core_hit_cell_ratio", 0.0)) for r in rows), default=0.0)),
        "max_object_hit_cell_ratio": float(max((float(r.get("object_hit_cell_ratio", 0.0)) for r in rows), default=0.0)),
        "max_height_span_m": float(max((float(r.get("height_span", 0.0)) for r in rows), default=0.0)),
        "min_distance_to_slot_m": float(min((float(r.get("distance_to_slot_m", 1e9)) for r in rows), default=1e9)),
    }


def classify_pose_status(case: dict[str, object], ev: dict[str, object], pose: dict[str, object], camera: dict[str, object]) -> tuple[str, list[str]]:
    subtype = str(case.get("uncertainty_type", case.get("part1_substate", "")))
    reasons: list[str] = []
    pose_enough = int(pose["pose_frame_count"]) >= 3 and float(pose["pose_baseline_m"]) >= 1.0
    camera_visible = int(camera["camera_visible_frame_count"]) > 0 and float(camera["best_camera_projection_score"]) >= 0.05
    close_enough = float(pose["min_distance_m"] or 1e9) <= 18.0
    boundary_ratio = float(ev["boundary_conflict_frame_ratio"])
    core_stable = (
        int(ev["clear_core_owned_frames"]) >= 2
        and str(ev.get("dominant_core_owner_slot")) == str(case["slot_id"])
        and float(ev["dominant_core_owner_ratio"]) >= 0.65
        and float(ev["max_core_hit_cell_ratio"]) >= 0.035
    )
    free_pose_supported = (
        int(ev["ray_support_frames"]) >= 3
        and float(ev["max_ray_free_ratio"]) >= 0.30
        and float(ev["max_object_hit_cell_ratio"]) <= 0.08
        and pose_enough
        and close_enough
    )

    if not pose_enough:
        reasons.append("pose diversity/support is insufficient")
    if not close_enough:
        reasons.append("slot was not observed from close enough pose")
    if not camera_visible:
        reasons.append("no strong camera-visible frame found from evidence poses")
    if boundary_ratio >= 0.50:
        reasons.append("many evidence frames are boundary/margin/adjacent conflicts")

    if subtype == "possible_free_unconfirmed":
        if free_pose_supported and camera_visible and boundary_ratio < 0.45:
            return "possible_free_pose_supported_camera_check", reasons or ["multi-pose ray-free evidence is plausible but still needs visual confirmation"]
        if free_pose_supported:
            return "possible_free_pose_supported_camera_weak", reasons
        return "possible_free_still_unconfirmed", reasons

    if subtype in {"occupied_boundary_conflict", "adjacent_slot_conflict", "static_vs_vehicle", "slot_vs_lane"}:
        if core_stable and pose_enough and boundary_ratio < 0.45:
            return "possible_occupied_pose_supported_camera_check", reasons or ["core-owned obstacle evidence is stable across poses"]
        if boundary_ratio >= 0.50:
            return "boundary_still_ambiguous", reasons
        return "conflict_still_needs_camera", reasons

    if subtype == "low_visibility":
        if camera_visible:
            return "low_visibility_but_camera_observable", reasons
        return "low_visibility_pose_not_helpful", reasons

    return "still_ambiguous", reasons


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = [
        "case_id",
        "slot_id",
        "uncertainty_type",
        "pose_assisted_status",
        "priority",
        "support_frame_count",
        "ray_support_frames",
        "visibility_support_frames",
        "clear_core_owned_frames",
        "boundary_conflict_like_frames",
        "boundary_conflict_frame_ratio",
        "min_distance_m",
        "median_distance_m",
        "min_abs_bearing_deg",
        "front_fov_frame_count",
        "pose_baseline_m",
        "bearing_span_deg",
        "camera_visible_frame_count",
        "best_camera_frame_id",
        "best_camera_projection_score",
        "max_ray_free_ratio",
        "max_object_hit_cell_ratio",
        "max_core_hit_cell_ratio",
        "dominant_core_owner_slot",
        "dominant_core_owner_ratio",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_report(path: Path, summary: dict[str, object], rows: list[dict[str, object]]) -> None:
    top_rows = sorted(
        rows,
        key=lambda r: (
            {"high": 0, "medium": 1, "low": 2}.get(str(r.get("priority")), 9),
            -float(r.get("best_camera_projection_score", 0.0)),
            -float(r.get("max_ray_free_ratio", 0.0)),
        ),
    )[:40]
    table = []
    for row in top_rows:
        table.append(
            "<tr>"
            f"<td>{html.escape(str(row['case_id']))}</td>"
            f"<td>{html.escape(str(row['slot_id']))}</td>"
            f"<td>{html.escape(str(row['uncertainty_type']))}</td>"
            f"<td>{html.escape(str(row['pose_assisted_status']))}</td>"
            f"<td>{html.escape(str(row['priority']))}</td>"
            f"<td>{row.get('support_frame_count')}</td>"
            f"<td>{row.get('ray_support_frames')}</td>"
            f"<td>{float(row.get('min_distance_m') or 0):.1f}</td>"
            f"<td>{float(row.get('pose_baseline_m') or 0):.1f}</td>"
            f"<td>{row.get('camera_visible_frame_count')}</td>"
            f"<td>{float(row.get('best_camera_projection_score') or 0):.3f}</td>"
            f"<td>{float(row.get('boundary_conflict_frame_ratio') or 0):.2f}</td>"
            f"<td>{html.escape('; '.join(row.get('pose_assisted_reasons', [])))}</td>"
            "</tr>"
        )
    html_text = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Pose Assisted Case Review</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 24px; color: #111827; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
    th, td {{ border: 1px solid #d1d5db; padding: 6px; vertical-align: top; }}
    th {{ background: #f3f4f6; text-align: left; }}
    pre {{ background: #f9fafb; border: 1px solid #d1d5db; padding: 12px; }}
  </style>
</head>
<body>
  <h1>Pose Assisted Case Review</h1>
  <p>This is a diagnostic layer only. It does not change Part 1 free/occupied states.</p>
  <pre>{html.escape(json.dumps(summary, indent=2))}</pre>
  <h2>Top Cases</h2>
  <table>
    <tr>
      <th>case</th><th>slot</th><th>type</th><th>pose status</th><th>priority</th>
      <th>support</th><th>ray frames</th><th>min dist</th><th>pose baseline</th>
      <th>camera frames</th><th>camera score</th><th>boundary ratio</th><th>reasons</th>
    </tr>
    {''.join(table)}
  </table>
</body>
</html>
"""
    path.write_text(html_text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    ensure_dir(args.output_dir)
    repo_root = Path.cwd()
    slot_db = load_json(args.part1_dir / "slot_database.json")
    slots = {str(slot["slot_id"]): slot for slot in slot_db["slots"]}  # type: ignore[index]
    scale = float(slot_db["map_units_per_meter"])  # type: ignore[index]
    cases = load_json(args.part2_pack / "case_cards_expanded.json")["cases"]  # type: ignore[index]
    frame_rows = load_frames(args.frames)
    slot_ids = {str(case["slot_id"]) for case in cases}
    evidence = load_evidence(args.part1_dir / "frame_slot_evidence.jsonl", slot_ids)
    ground_cache: dict[int, tuple[np.ndarray, np.ndarray, float]] = {}

    rows: list[dict[str, object]] = []
    for case in cases:
        slot_id = str(case["slot_id"])
        slot = slots[slot_id]
        ev_rows = evidence.get(slot_id, [])
        ev = evidence_metrics(slot_id, ev_rows)
        pose = pose_metrics(slot, ev_rows, frame_rows, scale)
        cam = camera_metrics(slot, ev_rows, frame_rows, scale, repo_root, ground_cache, args.ground_quantile)
        status, reasons = classify_pose_status(case, ev, pose, cam)
        row = {
            "case_id": case["case_id"],
            "slot_id": slot_id,
            "uncertainty_type": case.get("uncertainty_type", case.get("part1_substate", "")),
            "part1_state": case.get("part1_state", ""),
            "priority": case.get("priority", ""),
            "pose_assisted_status": status,
            "pose_assisted_reasons": reasons,
            **ev,
            **pose,
            **cam,
        }
        rows.append(row)

    status_counts = Counter(str(row["pose_assisted_status"]) for row in rows)
    type_counts = Counter(str(row["uncertainty_type"]) for row in rows)
    summary = {
        "case_count": len(rows),
        "uncertainty_type_counts": dict(type_counts),
        "pose_assisted_status_counts": dict(status_counts),
        "camera_observable_cases": sum(int(row["camera_visible_frame_count"]) > 0 for row in rows),
        "pose_supported_possible_free": sum(str(row["pose_assisted_status"]).startswith("possible_free_pose_supported") for row in rows),
        "pose_supported_possible_occupied": sum(str(row["pose_assisted_status"]) == "possible_occupied_pose_supported_camera_check" for row in rows),
        "boundary_still_ambiguous": status_counts.get("boundary_still_ambiguous", 0),
        "note": "Pose-assisted statuses are diagnostics only and do not change Part 1 final state.",
    }

    write_json(args.output_dir / "pose_assisted_cases.json", {"summary": summary, "cases": rows})
    write_json(args.output_dir / "pose_assisted_summary.json", summary)
    write_csv(args.output_dir / "pose_assisted_review.csv", rows)
    write_report(args.output_dir / "pose_assisted_report.html", summary, rows)
    print(f"[done] cases={len(rows)} output={args.output_dir}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
