#!/usr/bin/env python3
"""Build camera correspondence report for slot-aligned accumulation evidence."""

from __future__ import annotations

import argparse
import ast
import bisect
import csv
import html
import json
import math
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon as MplPolygon

try:
    from PIL import Image, ImageDraw

    HAS_PIL = True
except Exception:
    HAS_PIL = False


DEFAULT_INPUT = Path("outputs/slot_aligned_accumulation_stride4_w28_eps025_ms8")
DEFAULT_FRAMES = Path("outputs/frame_map_dataset/frames.csv")
DEFAULT_SLOT_DATABASE = Path("outputs/full_icpark_allframes_vehicle_cluster/slot_database.json")
DEFAULT_DATASET_ROOT = Path("/home/ParkingAgent/dataset/dataset/dataset")


STATE_LABEL = {
    "accumulated_vehicle_core_supported": "core-supported",
    "accumulated_adjacent_conflict": "adjacent conflict",
    "accumulated_boundary_conflict": "boundary conflict",
    "accumulated_static_like": "static-like",
    "accumulated_no_vehicle_evidence": "no vehicle evidence",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build camera correspondence report for slot-aligned evidence")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--frames", type=Path, default=DEFAULT_FRAMES)
    parser.add_argument("--slot-database", type=Path, default=DEFAULT_SLOT_DATABASE)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--states", nargs="+", default=["accumulated_vehicle_core_supported"])
    parser.add_argument("--limit", type=int, default=0, help="0 means include all selected rows")
    parser.add_argument("--comparison-review", action="store_true", help="Build richer map/camera/LiDAR comparison review")
    parser.add_argument(
        "--camera-match",
        choices=["nearest", "previous-window-start"],
        default="nearest",
        help="nearest uses closest timestamp; previous-window-start uses the earliest frame in the previous camera-frame window",
    )
    parser.add_argument("--camera-lookback-frames", type=int, default=28)
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_timestamps(path: Path) -> dict[int, float]:
    out: dict[int, float] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.split()
            out[int(parts[0])] = float(parts[1])
    return out


def nearest_timestamp(timestamps: dict[int, float], target: float) -> tuple[int, float, float]:
    items = sorted(timestamps.items())
    frames = [frame for frame, _ in items]
    values = [value for _, value in items]
    idx = bisect.bisect_left(values, target)
    candidates: list[tuple[float, int, float]] = []
    for pos in [idx - 1, idx, idx + 1]:
        if 0 <= pos < len(values):
            timestamp = values[pos]
            candidates.append((abs(timestamp - target), frames[pos], timestamp))
    if not candidates:
        raise ValueError("empty timestamp table")
    _, frame, timestamp = min(candidates)
    return int(frame), float(timestamp), float(timestamp - target)


def previous_window_start_timestamp(timestamps: dict[int, float], target: float, lookback_frames: int = 28) -> dict[str, Any]:
    items = sorted(timestamps.items())
    frames = [frame for frame, _ in items]
    values = [value for _, value in items]
    if not values:
        raise ValueError("empty timestamp table")
    idx = bisect.bisect_right(values, target) - 1
    if idx < 0:
        idx = 0
    lookback = max(1, int(lookback_frames))
    selected_idx = max(0, idx - lookback + 1)
    selected_frame = int(frames[selected_idx])
    selected_timestamp = float(values[selected_idx])
    base_frame = int(frames[idx])
    base_timestamp = float(values[idx])
    return {
        "selected_camera_frame": selected_frame,
        "selected_camera_timestamp": selected_timestamp,
        "selected_camera_delta_sec": float(selected_timestamp - target),
        "selected_window_index": int(selected_idx),
        "base_previous_frame": base_frame,
        "base_previous_timestamp": base_timestamp,
        "base_previous_delta_sec": float(base_timestamp - target),
        "lookback_frames": lookback,
    }


def select_camera_timestamp(
    timestamps: dict[int, float],
    target: float,
    match_mode: str,
    lookback_frames: int,
) -> dict[str, Any]:
    if match_mode == "previous-window-start":
        selected = previous_window_start_timestamp(timestamps, target, lookback_frames)
        selected["camera_match_mode"] = match_mode
        return selected
    frame, timestamp, delta = nearest_timestamp(timestamps, target)
    return {
        "selected_camera_frame": frame,
        "selected_camera_timestamp": timestamp,
        "selected_camera_delta_sec": delta,
        "selected_window_index": None,
        "base_previous_frame": frame,
        "base_previous_timestamp": timestamp,
        "base_previous_delta_sec": delta,
        "lookback_frames": 1,
        "camera_match_mode": "nearest",
    }


def f(row: dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return default


def i(row: dict[str, str], key: str, default: int = 0) -> int:
    try:
        return int(float(row.get(key, default)))
    except (TypeError, ValueError):
        return default


def rel(path: Path, base: Path) -> str:
    try:
        return path.resolve().relative_to(base.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def parse_list(value: str) -> list[Any]:
    if not value:
        return []
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        try:
            parsed = ast.literal_eval(value)
        except Exception:
            return []
    return parsed if isinstance(parsed, list) else []


def resolve_path(raw_path: str, base_dir: Path) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else base_dir / path


def draw_global_map(path: Path, slots: dict[str, dict[str, Any]], selected: list[dict[str, Any]]) -> None:
    selected_by_slot = {str(row["slot_id"]): row for row in selected}
    all_pts = np.vstack([np.asarray(slot["polygon_map"], dtype=np.float64) for slot in slots.values()])
    fig, ax = plt.subplots(figsize=(13, 10), dpi=190)
    for slot_id, slot in slots.items():
        poly = np.asarray(slot["polygon_map"], dtype=np.float64)
        if slot_id in selected_by_slot:
            face, edge, alpha, lw, z = "#dc2626", "#7f1d1d", 0.78, 1.3, 5
        else:
            face, edge, alpha, lw, z = "#e5e7eb", "#cbd5e1", 0.12, 0.22, 1
        ax.add_patch(MplPolygon(poly, closed=True, facecolor=face, edgecolor=edge, alpha=alpha, linewidth=lw, zorder=z))
    for idx, row in enumerate(selected, start=1):
        slot = slots.get(str(row["slot_id"]))
        if not slot:
            continue
        center = np.asarray(slot["center_map"], dtype=np.float64)
        ax.text(center[0], center[1], f"{idx}\n{str(row['slot_id']).replace('slot_', '')}", fontsize=5.3, ha="center", va="center", zorder=9)
    bmin = all_pts.min(axis=0) - 0.5
    bmax = all_pts.max(axis=0) + 0.5
    ax.set_xlim(float(bmin[0]), float(bmax[0]))
    ax.set_ylim(float(bmin[1]), float(bmax[1]))
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Camera correspondence slots on global map")
    ax.set_xlabel("map x")
    ax.set_ylabel("map y")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def draw_slot_global_context(path: Path, slots: dict[str, dict[str, Any]], slot_id: str) -> None:
    all_pts = np.vstack([np.asarray(slot["polygon_map"], dtype=np.float64) for slot in slots.values()])
    fig, ax = plt.subplots(figsize=(8, 7), dpi=170)
    target = slots[slot_id]
    target_center = np.asarray(target["center_map"], dtype=np.float64)
    for sid, slot in slots.items():
        center = np.asarray(slot["center_map"], dtype=np.float64)
        dist = float(np.linalg.norm(center - target_center))
        if sid == slot_id:
            face, edge, alpha, lw, z = "#dc2626", "#7f1d1d", 0.78, 1.5, 5
        elif dist <= 1.4:
            face, edge, alpha, lw, z = "#f97316", "#9a3412", 0.42, 0.65, 4
        else:
            face, edge, alpha, lw, z = "#e5e7eb", "#cbd5e1", 0.10, 0.18, 1
        ax.add_patch(MplPolygon(np.asarray(slot["polygon_map"], dtype=np.float64), closed=True, facecolor=face, edgecolor=edge, alpha=alpha, linewidth=lw, zorder=z))
    ax.text(target_center[0], target_center[1], slot_id.replace("slot_", ""), fontsize=8, ha="center", va="center", zorder=9)
    margin = 1.6
    ax.set_xlim(float(target_center[0] - margin), float(target_center[0] + margin))
    ax.set_ylim(float(target_center[1] - margin), float(target_center[1] + margin))
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(f"{slot_id} global map context")
    ax.set_xlabel("map x")
    ax.set_ylabel("map y")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)




def load_points(frame: int, frame_rows: dict[int, dict[str, str]], base_dir: Path) -> tuple[np.ndarray, np.ndarray] | None:
    row = frame_rows.get(frame)
    if not row:
        return None
    point_path = resolve_path(row.get("map_points_path", ""), base_dir)
    if not point_path.exists():
        return None
    with np.load(point_path) as data:
        return data["points_map_xyzi"].astype(np.float64), data["ego_map_pose"].astype(np.float64)


def lidar_to_camera(points_lidar_xyz: np.ndarray) -> np.ndarray:
    x_l = points_lidar_xyz[:, 0]
    y_l = points_lidar_xyz[:, 1]
    z_l = points_lidar_xyz[:, 2]
    cam = np.column_stack([-y_l, -z_l, x_l]).astype(np.float64)
    cam += np.array([0.6, 0.0, -0.07], dtype=np.float64)
    return cam


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


def overlay_lidar_on_image(image_path: Path, lidar_path: Path, output_path: Path, title: str) -> int:
    if not HAS_PIL or not image_path.exists() or not lidar_path.exists():
        return 0
    image = Image.open(image_path).convert("RGB")
    points = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 4)
    dist = np.linalg.norm(points[:, :3], axis=1)
    mask = (points[:, 0] > 0.5) & (dist < 45.0) & (points[:, 2] > -2.0) & (points[:, 2] < 3.5)
    sub = points[mask]
    uv, valid = project_camera(lidar_to_camera(sub[:, :3]))
    inside = valid & (uv[:, 0] >= 0) & (uv[:, 0] < image.width) & (uv[:, 1] >= 0) & (uv[:, 1] < image.height)
    draw = ImageDraw.Draw(image)
    idxs = np.where(inside)[0]
    stride = max(1, len(idxs) // 12000)
    dists = dist[mask]
    for idx in idxs[::stride]:
        u = float(uv[idx, 0])
        v = float(uv[idx, 1])
        t = max(0.0, min(1.0, float(dists[idx]) / 35.0))
        color = (int(255 * (1.0 - t)), int(70 + 140 * t), int(255 * t))
        draw.ellipse((u - 1, v - 1, u + 1, v + 1), fill=color)
    draw.rectangle((0, 0, image.width, 36), fill=(255, 255, 255))
    draw.text((8, 9), title, fill=(0, 0, 0))
    image.save(output_path)
    return int(inside.sum())


def draw_slot_accumulated_zoom(
    path: Path,
    slots: dict[str, dict[str, Any]],
    slot_id: str,
    sampled_frames: list[int],
    frame_rows: dict[int, dict[str, str]],
    base_dir: Path,
) -> int:
    if slot_id not in slots:
        return 0
    slot = slots[slot_id]
    center = np.asarray(slot["center_map"], dtype=np.float64)
    nearby_polys = []
    for sid, other in slots.items():
        other_center = np.asarray(other["center_map"], dtype=np.float64)
        if sid == slot_id or float(np.linalg.norm(other_center - center)) <= 1.5:
            nearby_polys.append((sid, np.asarray(other["polygon_map"], dtype=np.float64)))
    point_chunks = []
    ego_poses = []
    for frame in sampled_frames:
        loaded = load_points(frame, frame_rows, base_dir)
        if loaded is None:
            continue
        points, ego_pose = loaded
        ego_poses.append(ego_pose)
        dist = np.linalg.norm(points[:, :2] - center[None, :], axis=1)
        points = points[dist <= 1.8]
        if len(points):
            z = points[:, 2]
            ground_z = float(np.quantile(z, 0.08))
            obstacle = points[(z >= ground_z + 0.30) & (z <= ground_z + 2.50)]
            if len(obstacle):
                point_chunks.append(obstacle)
    obstacle_points = np.vstack(point_chunks) if point_chunks else np.empty((0, 4), dtype=np.float64)
    fig, ax = plt.subplots(figsize=(8, 7), dpi=170)
    for sid, poly in nearby_polys:
        if sid == slot_id:
            face, edge, alpha, lw, zorder = "#dc2626", "#7f1d1d", 0.45, 1.5, 4
        else:
            face, edge, alpha, lw, zorder = "#dbeafe", "#60a5fa", 0.28, 0.55, 2
        ax.add_patch(MplPolygon(poly, closed=True, facecolor=face, edgecolor=edge, alpha=alpha, linewidth=lw, zorder=zorder))
    ax.add_patch(MplPolygon(np.asarray(slot["margin_polygon_map"], dtype=np.float64), closed=True, facecolor="none", edgecolor="#f59e0b", linewidth=1.1, zorder=5))
    ax.add_patch(MplPolygon(np.asarray(slot["core_polygon_map"], dtype=np.float64), closed=True, facecolor="none", edgecolor="#7f1d1d", linewidth=1.3, zorder=6))
    if len(obstacle_points):
        ax.scatter(obstacle_points[:, 0], obstacle_points[:, 1], s=2.5, color="#ef4444", alpha=0.36, linewidths=0, label="accumulated non-ground")
    if ego_poses:
        ego = np.asarray(ego_poses[len(ego_poses) // 2], dtype=np.float64)
        ax.scatter([ego[0]], [ego[1]], s=30, color="#111827", zorder=8, label="anchor ego")
        ax.arrow(ego[0], ego[1], 0.12 * math.cos(ego[2]), 0.12 * math.sin(ego[2]), width=0.004, head_width=0.04, color="#111827", zorder=9)
    ax.text(center[0], center[1], slot_id.replace("slot_", ""), fontsize=8, ha="center", va="center", zorder=10)
    margin = 1.15
    ax.set_xlim(float(center[0] - margin), float(center[0] + margin))
    ax.set_ylim(float(center[1] - margin), float(center[1] + margin))
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(f"{slot_id} accumulated slot evidence")
    ax.set_xlabel("map x")
    ax.set_ylabel("map y")
    if len(obstacle_points):
        ax.legend(loc="upper right", fontsize=7)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return int(len(obstacle_points))


def copy_camera_image(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    shutil.copyfile(src, dst)
    return True


def build_cases(
    evidence_rows: list[dict[str, str]],
    frame_rows: dict[int, dict[str, str]],
    image_timestamps: dict[int, float],
    slots: dict[str, dict[str, Any]],
    dataset_root: Path,
    output_dir: Path,
    base_dir: Path,
    comparison_review: bool = False,
    camera_match: str = "nearest",
    camera_lookback_frames: int = 28,
) -> list[dict[str, Any]]:
    assets_dir = output_dir / "assets"
    ensure_dir(assets_dir)
    cases: list[dict[str, Any]] = []
    for idx, row in enumerate(evidence_rows, start=1):
        slot_id = str(row["slot_id"])
        anchor_frame = i(row, "anchor_frame")
        frame_row = frame_rows.get(anchor_frame)
        if frame_row is None:
            continue
        lidar_ts = float(frame_row["lidar_timestamp"])
        selected_camera = select_camera_timestamp(image_timestamps, lidar_ts, camera_match, camera_lookback_frames)
        camera_frame = int(selected_camera["selected_camera_frame"])
        camera_ts = float(selected_camera["selected_camera_timestamp"])
        dt = float(selected_camera["selected_camera_delta_sec"])
        image_path = dataset_root / "image" / f"left{camera_frame:06d}.png"
        lidar_path = resolve_path(frame_row.get("lidar_path", ""), base_dir)
        prefix = f"{idx:02d}_{slot_id}_lidar_{anchor_frame:06d}_camera_{camera_frame:06d}"
        camera_dst = assets_dir / f"{prefix}.png"
        overlay_dst = assets_dir / f"{prefix}_lidar_overlay.png"
        slot_map_dst = assets_dir / f"{prefix}_global_slot.png"
        slot_zoom_dst = assets_dir / f"{prefix}_slot_accumulated_zoom.png"
        copied = copy_camera_image(image_path, camera_dst)
        projected_count = 0
        accumulated_zoom_points = 0
        if slot_id in slots:
            draw_slot_global_context(slot_map_dst, slots, slot_id)
            if comparison_review:
                sampled_frames = [int(value) for value in parse_list(row.get("sampled_frame_ids", "")) if str(value).strip()]
                accumulated_zoom_points = draw_slot_accumulated_zoom(slot_zoom_dst, slots, slot_id, sampled_frames, frame_rows, base_dir)
        if comparison_review and copied:
            projected_count = overlay_lidar_on_image(
                image_path,
                lidar_path,
                overlay_dst,
                f"{slot_id} LiDAR {anchor_frame:06d} image {camera_frame:06d} dt={dt:.3f}s",
            )
        cases.append(
            {
                "index": idx,
                "slot_id": slot_id,
                "state": row["state_by_slot_aligned_accumulation"],
                "anchor_frame": anchor_frame,
                "lidar_timestamp": lidar_ts,
                "camera_frame": camera_frame,
                "camera_timestamp": camera_ts,
                "camera_lidar_dt_sec": dt,
                "camera_match_mode": selected_camera["camera_match_mode"],
                "camera_lookback_frames": selected_camera["lookback_frames"],
                "base_previous_camera_frame": selected_camera["base_previous_frame"],
                "base_previous_camera_timestamp": selected_camera["base_previous_timestamp"],
                "base_previous_camera_dt_sec": selected_camera["base_previous_delta_sec"],
                "camera_image_source": str(image_path),
                "lidar_path": str(lidar_path),
                "camera_image_path": rel(camera_dst, output_dir) if copied else "",
                "camera_lidar_overlay_path": rel(overlay_dst, output_dir) if overlay_dst.exists() else "",
                "slot_global_map_path": rel(slot_map_dst, output_dir) if slot_map_dst.exists() else "",
                "slot_accumulated_zoom_path": rel(slot_zoom_dst, output_dir) if slot_zoom_dst.exists() else "",
                "projected_lidar_points_in_camera": projected_count,
                "accumulated_zoom_point_count": accumulated_zoom_points,
                "max_vehicle_like_score": f(row, "max_vehicle_like_score"),
                "anchor_distance_to_slot_m": f(row, "anchor_distance_to_slot_m"),
                "sampled_frame_count": i(row, "sampled_frame_count"),
                "core_overlap_count": i(row, "core_overlap_count"),
                "boundary_ratio": f(row, "boundary_ratio"),
                "adjacent_overlap_ratio": f(row, "adjacent_overlap_ratio"),
                "cluster_top2_slot": row.get("cluster_top2_slot", ""),
                "source_consensus_state": row.get("source_consensus_state", ""),
                "source_core_supported_anchor_count": i(row, "source_core_supported_anchor_count"),
                "source_conflict_anchor_count": i(row, "source_conflict_anchor_count"),
                "source_core_support_ratio": f(row, "source_core_support_ratio"),
                "source_conflict_ratio": f(row, "source_conflict_ratio"),
                "reason": row.get("reason", ""),
            }
        )
    return cases


def write_csv(path: Path, cases: list[dict[str, Any]]) -> None:
    fields = [
        "index",
        "slot_id",
        "state",
        "anchor_frame",
        "lidar_timestamp",
        "camera_frame",
        "camera_timestamp",
        "camera_lidar_dt_sec",
        "camera_match_mode",
        "camera_lookback_frames",
        "base_previous_camera_frame",
        "base_previous_camera_timestamp",
        "base_previous_camera_dt_sec",
        "camera_image_source",
        "camera_image_path",
        "camera_lidar_overlay_path",
        "slot_global_map_path",
        "slot_accumulated_zoom_path",
        "projected_lidar_points_in_camera",
        "accumulated_zoom_point_count",
        "max_vehicle_like_score",
        "anchor_distance_to_slot_m",
        "sampled_frame_count",
        "core_overlap_count",
        "boundary_ratio",
        "adjacent_overlap_ratio",
        "cluster_top2_slot",
        "source_consensus_state",
        "source_core_supported_anchor_count",
        "source_conflict_anchor_count",
        "source_core_support_ratio",
        "source_conflict_ratio",
        "reason",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for case in cases:
            writer.writerow({field: case.get(field, "") for field in fields})


def esc(value: Any) -> str:
    return html.escape(str(value))


def write_report(path: Path, global_map: Path, cases: list[dict[str, Any]], summary: dict[str, Any], output_dir: Path) -> None:
    rows = []
    for case in cases:
        rows.append(
            "<tr>"
            f"<td>{case['index']}</td>"
            f"<td>{esc(case['slot_id'])}</td>"
            f"<td>{esc(STATE_LABEL.get(case['state'], case['state']))}</td>"
            f"<td>{case['anchor_frame']}</td>"
            f"<td>left{case['camera_frame']:06d}.png</td>"
            f"<td>{case['camera_lidar_dt_sec']:.4f}s</td>"
            f"<td>{case['max_vehicle_like_score']:.3f}</td>"
            f"<td>{case['anchor_distance_to_slot_m']:.2f}m</td>"
            f"<td>{case['core_overlap_count']}</td>"
            f"<td>{case['boundary_ratio']:.3f}</td>"
            f"<td>{case['adjacent_overlap_ratio']:.3f}</td>"
            f"<td><a href='{esc(case['camera_image_path'])}'>camera</a></td>"
            f"<td><a href='{esc(case['slot_global_map_path'])}'>map</a></td>"
            "</tr>"
        )
    sections = []
    for case in cases:
        sections.append(
            "<section>"
            f"<h2>{case['index']}. {esc(case['slot_id'])}: LiDAR {case['anchor_frame']} -> left{case['camera_frame']:06d}.png</h2>"
            f"<p>dt={case['camera_lidar_dt_sec']:.4f}s | score={case['max_vehicle_like_score']:.3f} | distance={case['anchor_distance_to_slot_m']:.2f}m | core={case['core_overlap_count']} | boundary={case['boundary_ratio']:.3f} | adjacent={case['adjacent_overlap_ratio']:.3f}</p>"
            "<div class='grid'>"
            f"<div><h3>Global map target slot</h3><img src='{esc(case['slot_global_map_path'])}'></div>"
            f"<div><h3>Selected camera image</h3><img src='{esc(case['camera_image_path'])}'></div>"
            "</div>"
            "</section>"
        )
    html_text = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Slot-aligned Camera Correspondence Report</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 24px; color: #111827; }}
    .note {{ background: #eff6ff; border: 1px solid #bfdbfe; padding: 12px; margin-bottom: 16px; }}
    table {{ border-collapse: collapse; font-size: 12px; width: 100%; }}
    th, td {{ border: 1px solid #d1d5db; padding: 4px 6px; text-align: left; }}
    section {{ border-top: 1px solid #d1d5db; padding: 16px 0; }}
    .grid {{ display: grid; grid-template-columns: 1fr 1.35fr; gap: 16px; align-items: start; }}
    img {{ max-width: 100%; border: 1px solid #d1d5db; background: white; }}
    pre {{ background: #f8fafc; border: 1px solid #e2e8f0; padding: 10px; overflow: auto; }}
  </style>
</head>
<body>
  <h1>Slot-aligned Camera Correspondence Report</h1>
  <div class="note">
    Camera images are selected by the configured camera matching strategy. This report checks visual correspondence for algorithm outputs; it does not change occupied/free decisions.
  </div>
  <h2>Summary</h2>
  <pre>{esc(json.dumps(summary, indent=2, ensure_ascii=False))}</pre>
  <h2>Selected slots on global map</h2>
  <img src="{esc(rel(global_map, output_dir))}">
  <h2>Frame to camera correspondence</h2>
  <table>
    <tr><th>#</th><th>slot</th><th>state</th><th>LiDAR frame</th><th>camera image</th><th>dt</th><th>score</th><th>distance</th><th>core</th><th>boundary</th><th>adjacent</th><th>camera</th><th>map</th></tr>
    {''.join(rows)}
  </table>
  {''.join(sections)}
</body>
</html>
"""
    path.write_text(html_text, encoding="utf-8")


REVIEW_LABELS = [
    ("map_camera_match", "map-camera match"),
    ("target_occupied", "target occupied"),
    ("adjacent_occupied", "adjacent occupied"),
    ("empty", "empty"),
    ("occluded_or_not_visible", "occluded / not visible"),
    ("wrong_slot_or_bad_alignment", "wrong slot / bad alignment"),
    ("unclear", "unclear"),
]


def write_comparison_report(path: Path, global_map: Path, cases: list[dict[str, Any]], summary: dict[str, Any], output_dir: Path) -> None:
    case_payload = [
        {
            "index": case["index"],
            "slot_id": case["slot_id"],
            "state": case["state"],
            "anchor_frame": case["anchor_frame"],
            "camera_frame": case["camera_frame"],
            "camera_lidar_dt_sec": case["camera_lidar_dt_sec"],
            "camera_match_mode": case["camera_match_mode"],
            "base_previous_camera_frame": case["base_previous_camera_frame"],
            "base_previous_camera_dt_sec": case["base_previous_camera_dt_sec"],
            "max_vehicle_like_score": case["max_vehicle_like_score"],
            "core_overlap_count": case["core_overlap_count"],
            "boundary_ratio": case["boundary_ratio"],
            "adjacent_overlap_ratio": case["adjacent_overlap_ratio"],
            "source_consensus_state": case["source_consensus_state"],
            "source_core_supported_anchor_count": case["source_core_supported_anchor_count"],
            "source_conflict_anchor_count": case["source_conflict_anchor_count"],
        }
        for case in cases
    ]
    rows = []
    for case in cases:
        rows.append(
            "<tr>"
            f"<td><a href='#case-{case['index']}'>{case['index']}</a></td>"
            f"<td>{esc(case['slot_id'])}</td>"
            f"<td>{esc(STATE_LABEL.get(case['state'], case['state']))}</td>"
            f"<td>{case['anchor_frame']}</td>"
            f"<td>left{case['camera_frame']:06d}.png</td>"
            f"<td>{case['camera_lidar_dt_sec']:.4f}s</td>"
            f"<td>{case['max_vehicle_like_score']:.3f}</td>"
            f"<td>{case['core_overlap_count']}</td>"
            f"<td>{case['boundary_ratio']:.3f}</td>"
            f"<td>{case['adjacent_overlap_ratio']:.3f}</td>"
            f"<td>{case['source_core_supported_anchor_count']}/{case['source_core_supported_anchor_count'] + case['source_conflict_anchor_count']}</td>"
            "</tr>"
        )
    label_buttons = "".join(
        f"<button type='button' data-label='{esc(value)}'>{esc(text)}</button>" for value, text in REVIEW_LABELS
    )
    sections = []
    for case in cases:
        sections.append(
            f"<section class='case' id='case-{case['index']}' data-index='{case['index']}'>"
            f"<h2>{case['index']}. {esc(case['slot_id'])} | {esc(STATE_LABEL.get(case['state'], case['state']))}</h2>"
            "<div class='metrics'>"
            f"<span>LiDAR {case['anchor_frame']} -> left{case['camera_frame']:06d}.png</span>"
            f"<span>dt {case['camera_lidar_dt_sec']:.4f}s</span>"
            f"<span>score {case['max_vehicle_like_score']:.3f}</span>"
            f"<span>core {case['core_overlap_count']}</span>"
            f"<span>boundary {case['boundary_ratio']:.3f}</span>"
            f"<span>adjacent {case['adjacent_overlap_ratio']:.3f}</span>"
            f"<span>projected points {case['projected_lidar_points_in_camera']}</span>"
            "</div>"
            "<div class='visual-grid'>"
            f"<figure><figcaption>Global target slot</figcaption><img src='{esc(case['slot_global_map_path'])}'></figure>"
            f"<figure><figcaption>Accumulated slot evidence</figcaption><img src='{esc(case['slot_accumulated_zoom_path'])}'></figure>"
            f"<figure><figcaption>Selected camera</figcaption><img src='{esc(case['camera_image_path'])}'></figure>"
            f"<figure><figcaption>Camera with LiDAR overlay</figcaption><img src='{esc(case['camera_lidar_overlay_path'])}'></figure>"
            "</div>"
            f"<p class='reason'>{esc(case['reason'])}</p>"
            "<div class='review-controls'>"
            f"{label_buttons}"
            f"<input type='text' placeholder='note for {esc(case['slot_id'])}'>"
            "<span class='chosen'>unreviewed</span>"
            "</div>"
            "</section>"
        )
    html_text = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Slot Camera Comparison Review</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 24px; color: #111827; background: #ffffff; }}
    .note {{ background: #eef2ff; border: 1px solid #c7d2fe; padding: 12px; margin-bottom: 16px; }}
    table {{ border-collapse: collapse; font-size: 12px; width: 100%; margin-bottom: 20px; }}
    th, td {{ border: 1px solid #d1d5db; padding: 4px 6px; text-align: left; }}
    .case {{ border-top: 2px solid #d1d5db; padding: 18px 0; }}
    .metrics {{ display: flex; flex-wrap: wrap; gap: 8px; margin: 8px 0 12px; }}
    .metrics span {{ background: #f3f4f6; border: 1px solid #d1d5db; padding: 4px 7px; font-size: 12px; }}
    .visual-grid {{ display: grid; grid-template-columns: repeat(2, minmax(280px, 1fr)); gap: 14px; align-items: start; }}
    figure {{ margin: 0; }}
    figcaption {{ font-weight: 650; font-size: 13px; margin-bottom: 5px; }}
    img {{ max-width: 100%; border: 1px solid #d1d5db; background: white; }}
    .reason {{ color: #374151; font-size: 13px; }}
    .review-controls {{ display: flex; flex-wrap: wrap; gap: 8px; align-items: center; margin-top: 10px; }}
    button {{ border: 1px solid #9ca3af; background: #ffffff; padding: 6px 8px; cursor: pointer; }}
    button.selected {{ background: #14532d; color: #ffffff; border-color: #14532d; }}
    input {{ min-width: 280px; padding: 7px; border: 1px solid #9ca3af; }}
    .chosen {{ font-size: 12px; color: #374151; }}
    textarea {{ width: 100%; min-height: 220px; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12px; }}
    @media (max-width: 900px) {{ .visual-grid {{ grid-template-columns: 1fr; }} }}
  </style>
</head>
<body>
  <h1>Slot Camera Comparison Review</h1>
  <div class="note">
    Use this report to compare the target slot map, accumulated LiDAR evidence, selected camera image, and camera LiDAR overlay. Labels are stored in this browser page and can be exported as JSON below.
  </div>
  <h2>Summary</h2>
  <pre>{esc(json.dumps(summary, indent=2, ensure_ascii=False))}</pre>
  <h2>Selected slots on global map</h2>
  <img src="{esc(rel(global_map, output_dir))}">
  <h2>Cases</h2>
  <table>
    <tr><th>#</th><th>slot</th><th>state</th><th>LiDAR frame</th><th>camera image</th><th>dt</th><th>score</th><th>core</th><th>boundary</th><th>adjacent</th><th>core/conflict</th></tr>
    {''.join(rows)}
  </table>
  {''.join(sections)}
  <h2>Export review labels</h2>
  <button type="button" id="refresh-export">Refresh export JSON</button>
  <textarea id="export-json" spellcheck="false"></textarea>
  <script>
    const cases = {json.dumps(case_payload, ensure_ascii=False)};
    const reviews = Object.fromEntries(cases.map(item => [String(item.index), {{...item, label: "", note: ""}}]));
    function refreshExport() {{
      document.getElementById("export-json").value = JSON.stringify(Object.values(reviews), null, 2);
    }}
    document.querySelectorAll(".case").forEach(section => {{
      const index = section.dataset.index;
      const chosen = section.querySelector(".chosen");
      const note = section.querySelector("input");
      note.addEventListener("input", () => {{
        reviews[index].note = note.value;
        refreshExport();
      }});
      section.querySelectorAll("button[data-label]").forEach(button => {{
        button.addEventListener("click", () => {{
          section.querySelectorAll("button[data-label]").forEach(other => other.classList.remove("selected"));
          button.classList.add("selected");
          reviews[index].label = button.dataset.label;
          chosen.textContent = button.dataset.label;
          refreshExport();
        }});
      }});
    }});
    document.getElementById("refresh-export").addEventListener("click", refreshExport);
    refreshExport();
  </script>
</body>
</html>
"""
    path.write_text(html_text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.output_dir is not None:
        output_dir = args.output_dir
    elif args.comparison_review:
        output_dir = args.input_dir / "camera_comparison_review"
    else:
        output_dir = args.input_dir / "camera_correspondence_report"
    ensure_dir(output_dir)
    evidence_rows = read_csv(args.input_dir / "slot_aligned_evidence.csv")
    selected = [row for row in evidence_rows if row.get("state_by_slot_aligned_accumulation") in set(args.states)]
    selected.sort(key=lambda row: (-f(row, "max_vehicle_like_score"), f(row, "anchor_distance_to_slot_m"), -i(row, "core_overlap_count")))
    if args.limit > 0:
        selected = selected[: args.limit]
    frame_rows = {int(row["frame"]): row for row in read_csv(args.frames)}
    image_timestamps = load_timestamps(args.dataset_root / "image" / "timestamps.txt")
    slot_db = load_json(args.slot_database)
    slots = {str(slot["slot_id"]): slot for slot in slot_db["slots"]}
    cases = build_cases(
        selected,
        frame_rows,
        image_timestamps,
        slots,
        args.dataset_root,
        output_dir,
        Path.cwd(),
        args.comparison_review,
        args.camera_match,
        args.camera_lookback_frames,
    )
    counts = Counter(case["state"] for case in cases)
    dt_values = [abs(float(case["camera_lidar_dt_sec"])) for case in cases]
    summary = {
        "input_dir": str(args.input_dir),
        "states": args.states,
        "camera_match": args.camera_match,
        "camera_lookback_frames": args.camera_lookback_frames,
        "selected_case_count": len(cases),
        "state_counts": dict(counts),
        "max_abs_camera_lidar_dt_sec": max(dt_values) if dt_values else None,
        "mean_abs_camera_lidar_dt_sec": sum(dt_values) / len(dt_values) if dt_values else None,
    }
    global_map = output_dir / "selected_slots_global_map.png"
    draw_global_map(global_map, slots, cases)
    if args.comparison_review:
        write_csv(output_dir / "camera_comparison_review.csv", cases)
        (output_dir / "camera_comparison_review.json").write_text(json.dumps({"summary": summary, "cases": cases}, indent=2), encoding="utf-8")
        write_comparison_report(output_dir / "camera_comparison_review_report.html", global_map, cases, summary, output_dir)
        output_path = output_dir / "camera_comparison_review_report.html"
    else:
        write_csv(output_dir / "camera_correspondence.csv", cases)
        (output_dir / "camera_correspondence.json").write_text(json.dumps({"summary": summary, "cases": cases}, indent=2), encoding="utf-8")
        write_report(output_dir / "camera_correspondence_report.html", global_map, cases, summary, output_dir)
        output_path = output_dir / "camera_correspondence_report.html"
    print(json.dumps({"output": str(output_path), **summary}, indent=2))


if __name__ == "__main__":
    main()
