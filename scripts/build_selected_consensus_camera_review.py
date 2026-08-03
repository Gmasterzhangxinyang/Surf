#!/usr/bin/env python3
"""Build camera review pack for selected multiframe consensus slots."""

from __future__ import annotations

import argparse
import ast
import bisect
import csv
import html
import json
import math
import shutil
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon as MplPolygon
from PIL import Image, ImageDraw


DEFAULT_CONSENSUS = Path("outputs/multiframe_temporal_consensus_eps025_ms8")
DEFAULT_SLOT_DATABASE = Path("outputs/full_icpark_allframes_vehicle_cluster/slot_database.json")
DEFAULT_FRAMES = Path("outputs/frame_map_dataset/frames.csv")
DEFAULT_DATASET_ROOT = Path("/home/ParkingAgent/dataset/dataset/dataset")
DEFAULT_OUTPUT = DEFAULT_CONSENSUS / "selected_camera_review"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build selected slot camera review from multiframe consensus")
    parser.add_argument("--consensus-dir", type=Path, default=DEFAULT_CONSENSUS)
    parser.add_argument("--slot-database", type=Path, default=DEFAULT_SLOT_DATABASE)
    parser.add_argument("--frames", type=Path, default=DEFAULT_FRAMES)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--slot-ids", nargs="*", default=None)
    parser.add_argument("--state", default="possible_occupied_by_multiframe_accumulation")
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def parse_list(value: str) -> list[Any]:
    if not value:
        return []
    try:
        out = json.loads(value)
    except json.JSONDecodeError:
        try:
            out = ast.literal_eval(value)
        except Exception:
            return []
    return out if isinstance(out, list) else []


def f(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return default


def i(row: dict[str, Any], key: str, default: int = 0) -> int:
    try:
        return int(float(row.get(key, default)))
    except (TypeError, ValueError):
        return default


def load_frames(path: Path) -> dict[int, dict[str, str]]:
    rows = read_csv_rows(path)
    return {int(row["frame"]): row for row in rows}


def load_timestamps(path: Path) -> dict[int, float]:
    out: dict[int, float] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.split()
            out[int(parts[0])] = float(parts[1])
    return out


def nearest_timestamp(ts: dict[int, float], target: float) -> tuple[int, float, float]:
    items = sorted(ts.items())
    frames = [item[0] for item in items]
    times = [item[1] for item in items]
    idx = bisect.bisect_left(times, target)
    candidates = []
    for j in [idx - 1, idx, idx + 1]:
        if 0 <= j < len(times):
            candidates.append((abs(times[j] - target), frames[j], times[j]))
    diff_abs, frame, timestamp = min(candidates)
    return frame, timestamp, timestamp - target


def resolve_path(raw_path: str, base_dir: Path) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else base_dir / path


def load_points(frame: int, frame_rows: dict[int, dict[str, str]], base_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    row = frame_rows[frame]
    path = resolve_path(row["map_points_path"], base_dir)
    with np.load(path) as data:
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
        color = (int(255 * (1.0 - t)), int(60 + 150 * t), int(255 * t))
        draw.ellipse((u - 1, v - 1, u + 1, v + 1), fill=color)
    draw.rectangle((0, 0, image.width, 34), fill=(255, 255, 255))
    draw.text((8, 8), title, fill=(0, 0, 0))
    image.save(output_path)
    return int(inside.sum())


def select_diverse_slots(consensus_rows: list[dict[str, str]], slots: dict[str, dict[str, Any]], top_k: int, state: str) -> list[str]:
    candidates = [row for row in consensus_rows if row["consensus_state"] == state and row["slot_id"] in slots]
    candidates.sort(key=lambda row: (-i(row, "core_supported_anchor_count"), -f(row, "max_vehicle_like_score"), f(row, "conflict_ratio")))
    if len(candidates) <= top_k:
        return [row["slot_id"] for row in candidates]
    selected = [candidates[0]["slot_id"]]
    remaining = [row["slot_id"] for row in candidates[1:]]
    while remaining and len(selected) < top_k:
        best_sid = None
        best_score = -1.0
        for sid in remaining:
            center = np.asarray(slots[sid]["center_map"], dtype=np.float64)
            min_dist = min(float(np.linalg.norm(center - np.asarray(slots[other]["center_map"], dtype=np.float64))) for other in selected)
            row = next(row for row in candidates if row["slot_id"] == sid)
            quality = 0.10 * i(row, "core_supported_anchor_count") + f(row, "max_vehicle_like_score")
            score = min_dist + 0.03 * quality
            if score > best_score:
                best_score = score
                best_sid = sid
        selected.append(str(best_sid))
        remaining.remove(str(best_sid))
    return selected


def best_evidence_for_slot(slot_id: str, consensus_row: dict[str, str], input_dirs: list[Path]) -> dict[str, Any]:
    core_frames = set(int(value) for value in parse_list(consensus_row.get("core_anchor_frames", "")))
    best: dict[str, Any] | None = None
    for input_dir in input_dirs:
        evidence_path = input_dir / "accumulated_slot_evidence.csv"
        if not evidence_path.exists():
            continue
        for row in read_csv_rows(evidence_path):
            if row["slot_id"] != slot_id:
                continue
            if row["accumulated_state"] != "accumulated_vehicle_core_supported":
                continue
            frame = i(row, "anchor_frame")
            if core_frames and frame not in core_frames:
                continue
            key = (f(row, "max_vehicle_like_score"), i(row, "core_overlap_count"), -f(row, "conflict_ratio"))
            if best is None or key > best["_key"]:
                best = {**row, "source_dir": str(input_dir), "_key": key}
    if best is None:
        frames = sorted(core_frames)
        return {"anchor_frame": frames[len(frames) // 2] if frames else 0, "source_dir": "", "_key": (0.0, 0, 0.0)}
    return best


def draw_selected_global_map(path: Path, slots: dict[str, dict[str, Any]], selected: list[str], rows_by_slot: dict[str, dict[str, str]]) -> None:
    all_pts = np.vstack([np.asarray(slot["polygon_map"], dtype=np.float64) for slot in slots.values()])
    fig, ax = plt.subplots(figsize=(12, 9), dpi=190)
    for sid, slot in slots.items():
        poly = np.asarray(slot["polygon_map"], dtype=np.float64)
        if sid in selected:
            face, edge, alpha, lw, z = "#dc2626", "#7f1d1d", 0.82, 1.6, 5
        else:
            face, edge, alpha, lw, z = "#e5e7eb", "#cbd5e1", 0.13, 0.22, 1
        ax.add_patch(MplPolygon(poly, closed=True, facecolor=face, edgecolor=edge, alpha=alpha, linewidth=lw, zorder=z))
    for idx, sid in enumerate(selected, start=1):
        center = np.asarray(slots[sid]["center_map"], dtype=np.float64)
        ax.text(center[0], center[1], f"{idx}\\n{sid.replace('slot_', '')}", fontsize=6.5, ha="center", va="center", color="#111827", zorder=10)
    bmin = all_pts.min(axis=0) - 0.5
    bmax = all_pts.max(axis=0) + 0.5
    ax.set_xlim(float(bmin[0]), float(bmax[0]))
    ax.set_ylim(float(bmin[1]), float(bmax[1]))
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Selected multiframe occupied candidates for camera review")
    ax.set_xlabel("map x")
    ax.set_ylabel("map y")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def draw_slot_zoom(
    path: Path,
    slot_id: str,
    slots: dict[str, dict[str, Any]],
    points: np.ndarray,
    ego_pose: np.ndarray,
    scale: float,
) -> None:
    slot = slots[slot_id]
    center = np.asarray(slot["center_map"], dtype=np.float64)
    fig, ax = plt.subplots(figsize=(7, 6), dpi=160)
    extent_polys = [np.asarray(slot["polygon_map"], dtype=np.float64)]
    for sid, other in slots.items():
        other_center = np.asarray(other["center_map"], dtype=np.float64)
        if float(np.linalg.norm(other_center - center)) <= 1.4:
            poly = np.asarray(other["polygon_map"], dtype=np.float64)
            ax.add_patch(MplPolygon(poly, closed=True, facecolor="#dbeafe", edgecolor="#60a5fa", alpha=0.35, linewidth=0.5))
            extent_polys.append(poly)
    ax.add_patch(MplPolygon(np.asarray(slot["margin_polygon_map"], dtype=np.float64), closed=True, facecolor="none", edgecolor="#f59e0b", linewidth=1.1, label="target margin"))
    ax.add_patch(MplPolygon(np.asarray(slot["core_polygon_map"], dtype=np.float64), closed=True, facecolor="#dc2626", edgecolor="#7f1d1d", alpha=0.55, linewidth=1.4, label="target core"))
    z = points[:, 2]
    ground_z = float(np.quantile(z, 0.08)) if len(z) else 0.0
    obstacle = points[(z >= ground_z + 0.30) & (z <= ground_z + 2.50)]
    if len(obstacle):
        dist = np.linalg.norm(obstacle[:, :2] - center[None, :], axis=1)
        obstacle = obstacle[dist <= 1.7]
        ax.scatter(obstacle[:, 0], obstacle[:, 1], s=3, color="#ef4444", alpha=0.55, linewidths=0, label="non-ground map points")
    ax.scatter([ego_pose[0]], [ego_pose[1]], s=32, color="#111827", zorder=8, label="ego")
    arrow_len = 2.5 * scale
    ax.arrow(ego_pose[0], ego_pose[1], arrow_len * math.cos(ego_pose[2]), arrow_len * math.sin(ego_pose[2]), width=0.006, head_width=0.06, color="#111827", zorder=9)
    extent = np.vstack(extent_polys + [center.reshape(1, 2), ego_pose[:2].reshape(1, 2)])
    bmin = extent.min(axis=0) - 0.35
    bmax = extent.max(axis=0) + 0.35
    ax.set_xlim(float(bmin[0]), float(bmax[0]))
    ax.set_ylim(float(bmin[1]), float(bmax[1]))
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(f"{slot_id} map zoom")
    ax.legend(loc="upper right", fontsize=7)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def rel(path: Path, base: Path) -> str:
    try:
        return path.resolve().relative_to(base.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def main() -> None:
    args = parse_args()
    ensure_dir(args.output_dir)
    assets_dir = args.output_dir / "assets"
    ensure_dir(assets_dir)
    base_dir = Path.cwd()

    slot_db = load_json(args.slot_database)
    slots = {str(slot["slot_id"]): slot for slot in slot_db["slots"]}
    scale = float(slot_db["map_units_per_meter"])
    frame_rows = load_frames(args.frames)
    image_ts = load_timestamps(args.dataset_root / "image" / "timestamps.txt")
    consensus_rows = read_csv_rows(args.consensus_dir / "multiframe_temporal_consensus.csv")
    rows_by_slot = {row["slot_id"]: row for row in consensus_rows}
    source_dirs = sorted({Path(path) for row in consensus_rows for path in parse_list(row.get("source_windows", ""))})
    # source_windows stores names; use actual expected dirs from the summary for robust lookup.
    summary_path = args.consensus_dir / "multiframe_temporal_consensus_summary.json"
    if summary_path.exists():
        summary = load_json(summary_path)
        source_dirs = [Path(source["source_dir"]) for source in summary.get("sources", [])]
    selected = args.slot_ids if args.slot_ids else select_diverse_slots(consensus_rows, slots, args.top_k, args.state)
    selected = [sid for sid in selected if sid in slots and sid in rows_by_slot]
    if not selected:
        raise SystemExit("no selected slots")

    map_path = assets_dir / "selected_slots_global_map.png"
    draw_selected_global_map(map_path, slots, selected, rows_by_slot)

    cases: list[dict[str, Any]] = []
    for idx, slot_id in enumerate(selected, start=1):
        consensus = rows_by_slot[slot_id]
        best = best_evidence_for_slot(slot_id, consensus, source_dirs)
        anchor_frame = int(best.get("anchor_frame", 0))
        if anchor_frame not in frame_rows:
            continue
        frame_row = frame_rows[anchor_frame]
        lidar_timestamp = float(frame_row["lidar_timestamp"])
        nearest_image_frame, nearest_image_timestamp, image_dt = nearest_timestamp(image_ts, lidar_timestamp)
        image_path = args.dataset_root / "image" / f"left{nearest_image_frame:06d}.png"
        lidar_path = resolve_path(frame_row["lidar_path"], base_dir)
        points, ego_pose = load_points(anchor_frame, frame_rows, base_dir)
        prefix = f"{idx:02d}_{slot_id}_frame_{anchor_frame:06d}_image_{nearest_image_frame:06d}"
        raw_dst = assets_dir / f"{prefix}_camera_raw.png"
        overlay_dst = assets_dir / f"{prefix}_camera_lidar_overlay.png"
        zoom_dst = assets_dir / f"{prefix}_slot_map_zoom.png"
        shutil.copyfile(image_path, raw_dst)
        projected_count = overlay_lidar_on_image(
            image_path,
            lidar_path,
            overlay_dst,
            f"{slot_id} LiDAR {anchor_frame:06d} image {nearest_image_frame:06d} dt={image_dt:.3f}s",
        )
        draw_slot_zoom(zoom_dst, slot_id, slots, points, ego_pose, scale)
        case = {
            "index": idx,
            "slot_id": slot_id,
            "consensus": consensus,
            "best_accumulated_evidence": {k: v for k, v in best.items() if k != "_key"},
            "anchor_frame": anchor_frame,
            "lidar_timestamp": lidar_timestamp,
            "nearest_image_frame": nearest_image_frame,
            "nearest_image_timestamp": nearest_image_timestamp,
            "nearest_image_dt_sec": image_dt,
            "projected_lidar_points_in_camera": projected_count,
            "ego_map_pose": ego_pose.tolist(),
            "slot_center_map": slots[slot_id]["center_map"],
            "assets": {
                "camera_raw": rel(raw_dst, args.output_dir),
                "camera_lidar_overlay": rel(overlay_dst, args.output_dir),
                "slot_map_zoom": rel(zoom_dst, args.output_dir),
            },
        }
        cases.append(case)

    write_path = args.output_dir / "selected_consensus_camera_review.json"
    write_path.write_text(json.dumps({"selected_slots": selected, "cases": cases}, indent=2), encoding="utf-8")
    with (args.output_dir / "selected_consensus_camera_review.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "index",
            "slot_id",
            "anchor_frame",
            "nearest_image_frame",
            "nearest_image_dt_sec",
            "projected_lidar_points_in_camera",
            "core_supported_anchor_count",
            "observed_anchor_count",
            "conflict_anchor_count",
            "max_vehicle_like_score",
            "camera_raw",
            "camera_lidar_overlay",
            "slot_map_zoom",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for case in cases:
            consensus = case["consensus"]
            writer.writerow(
                {
                    "index": case["index"],
                    "slot_id": case["slot_id"],
                    "anchor_frame": case["anchor_frame"],
                    "nearest_image_frame": case["nearest_image_frame"],
                    "nearest_image_dt_sec": case["nearest_image_dt_sec"],
                    "projected_lidar_points_in_camera": case["projected_lidar_points_in_camera"],
                    "core_supported_anchor_count": consensus.get("core_supported_anchor_count", ""),
                    "observed_anchor_count": consensus.get("observed_anchor_count", ""),
                    "conflict_anchor_count": consensus.get("conflict_anchor_count", ""),
                    "max_vehicle_like_score": consensus.get("max_vehicle_like_score", ""),
                    **case["assets"],
                }
            )

    parts = []
    for case in cases:
        c = case["consensus"]
        parts.append(
            "<section>"
            f"<h2>{case['index']}. {html.escape(case['slot_id'])}</h2>"
            f"<p>anchor LiDAR frame: {case['anchor_frame']} | timestamp-nearest image: left{case['nearest_image_frame']:06d}.png | image dt: {case['nearest_image_dt_sec']:.4f}s | projected LiDAR points: {case['projected_lidar_points_in_camera']}</p>"
            f"<p>consensus: core={html.escape(str(c.get('core_supported_anchor_count')))} / observed={html.escape(str(c.get('observed_anchor_count')))} | conflict={html.escape(str(c.get('conflict_anchor_count')))} | score={float(c.get('max_vehicle_like_score', 0.0)):.3f}</p>"
            "<div class='grid'>"
            f"<div><h3>Slot Map Zoom</h3><img src='{html.escape(case['assets']['slot_map_zoom'])}'></div>"
            f"<div><h3>Timestamp-nearest Camera + LiDAR Overlay</h3><img src='{html.escape(case['assets']['camera_lidar_overlay'])}'></div>"
            f"<div><h3>Raw Camera</h3><img src='{html.escape(case['assets']['camera_raw'])}'></div>"
            "</div>"
            "</section>"
        )
    report = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Selected Consensus Slot Camera Review</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 24px; color: #111827; }}
    .note {{ background: #eff6ff; border: 1px solid #bfdbfe; padding: 12px; margin-bottom: 16px; }}
    section {{ border-top: 1px solid #d1d5db; padding: 16px 0; }}
    .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; align-items: start; }}
    img {{ max-width: 100%; border: 1px solid #d1d5db; background: white; }}
  </style>
</head>
<body>
  <h1>Selected Consensus Slot Camera Review</h1>
  <div class="note">
    Selected slots are sampled from high-confidence multiframe occupied candidates. Camera images use timestamp-nearest binding, not same-index image binding.
  </div>
  <h2>Selected Slots On Global Map</h2>
  <img src="{html.escape(rel(map_path, args.output_dir))}">
  {''.join(parts)}
</body>
</html>
"""
    (args.output_dir / "selected_consensus_camera_review_report.html").write_text(report, encoding="utf-8")
    print(args.output_dir / "selected_consensus_camera_review_report.html")
    for case in cases:
        print(case["index"], case["slot_id"], "lidar", case["anchor_frame"], "image", case["nearest_image_frame"], "dt", f"{case['nearest_image_dt_sec']:.4f}")


if __name__ == "__main__":
    main()
