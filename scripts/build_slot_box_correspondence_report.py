#!/usr/bin/env python3
"""Build camera / point cloud / global-map correspondence report for box scoring."""

from __future__ import annotations

import argparse
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
from matplotlib.patches import Patch, Polygon as MplPolygon


DEFAULT_INPUT = Path("outputs/slot_constrained_box_scoring_v1")
DEFAULT_SLOT_DATABASE = Path("outputs/full_icpark_allframes_vehicle_cluster/slot_database.json")
DEFAULT_FRAMES_CSV = Path("outputs/frame_map_dataset/frames.csv")
DEFAULT_MAP_POINTS_DIR = Path("outputs/frame_map_dataset/map_points")


STATE_STYLE = {
    "box_vehicle_core_supported": ("#dc2626", "#7f1d1d", 0.78, "box vehicle core supported"),
    "box_boundary_conflict": ("#a855f7", "#581c87", 0.68, "box boundary conflict"),
    "box_adjacent_conflict": ("#f97316", "#9a3412", 0.72, "box adjacent conflict"),
    "box_low_height_residual": ("#64748b", "#334155", 0.48, "box low-height residual"),
    "box_wall_like_or_static_suspect": ("#0f766e", "#134e4a", 0.52, "wall/static suspect"),
    "box_no_vehicle_evidence": ("#d1d5db", "#94a3b8", 0.28, "box no vehicle evidence"),
    "box_unknown_insufficient_visibility": ("#facc15", "#a16207", 0.42, "insufficient visibility"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build point cloud / photo / map correspondence report")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--slot-database", type=Path, default=DEFAULT_SLOT_DATABASE)
    parser.add_argument("--frames-csv", type=Path, default=DEFAULT_FRAMES_CSV)
    parser.add_argument("--map-points-dir", type=Path, default=DEFAULT_MAP_POINTS_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--max-cases", type=int, default=0, help="0 means include all rows")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_frames(value: str) -> list[int]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return []
    return [int(v) for v in parsed] if isinstance(parsed, list) else []


def f(row: dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return default


def resolve_path(raw: str, root: Path) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else root / path


def report_rel(path: Path, report_dir: Path) -> str:
    try:
        return path.resolve().relative_to(report_dir.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def slot_center(slot: dict[str, Any]) -> np.ndarray:
    return np.asarray(slot["center_map"], dtype=np.float64)


def box_polygon(row: dict[str, str]) -> np.ndarray:
    center = np.asarray([f(row, "center_x"), f(row, "center_y")], dtype=np.float64)
    yaw = f(row, "yaw")
    length = f(row, "length")
    width = f(row, "width")
    long_axis = np.asarray([math.cos(yaw), math.sin(yaw)], dtype=np.float64)
    short_axis = np.asarray([-math.sin(yaw), math.cos(yaw)], dtype=np.float64)
    local = np.asarray(
        [[-length / 2, -width / 2], [length / 2, -width / 2], [length / 2, width / 2], [-length / 2, width / 2]],
        dtype=np.float64,
    )
    return center + local[:, 0:1] * long_axis + local[:, 1:2] * short_axis


def add_polygon(ax: plt.Axes, polygon: Any, face: str, edge: str, alpha: float, lw: float, zorder: int, label: str | None = None) -> None:
    ax.add_patch(MplPolygon(np.asarray(polygon, dtype=np.float64), closed=True, facecolor=face, edgecolor=edge, alpha=alpha, linewidth=lw, zorder=zorder, label=label))


def draw_ego(ax: plt.Axes, pose: np.ndarray, scale: float, *, color: str, label: str, zorder: int) -> None:
    x, y, yaw = map(float, pose[:3])
    ax.scatter([x], [y], s=26, color=color, zorder=zorder, label=label)
    ax.arrow(
        x,
        y,
        2.0 * scale * math.cos(yaw),
        2.0 * scale * math.sin(yaw),
        width=0.006,
        head_width=0.045,
        head_length=0.060,
        color=color,
        length_includes_head=True,
        zorder=zorder,
    )


def load_frame_points(frame: int, frame_rows: dict[int, dict[str, str]], map_points_dir: Path, root: Path) -> tuple[np.ndarray, np.ndarray] | None:
    row = frame_rows.get(frame)
    if row is None:
        return None
    raw = row.get("map_points_path", "")
    path = resolve_path(raw, root) if raw else map_points_dir / f"{frame:06d}.npz"
    if not path.exists():
        path = map_points_dir / f"{frame:06d}.npz"
    if not path.exists():
        return None
    with np.load(path) as data:
        return np.asarray(data["points_map_xyzi"], dtype=np.float64), np.asarray(data["ego_map_pose"], dtype=np.float64)


def draw_overview_map(path: Path, slots: dict[str, dict[str, Any]], rows: list[dict[str, str]]) -> None:
    row_by_slot = {row["slot_id"]: row for row in rows}
    all_pts = np.vstack([np.asarray(slot["polygon_map"], dtype=np.float64) for slot in slots.values()])
    fig, ax = plt.subplots(figsize=(13, 10), dpi=190)
    for slot_id, slot in slots.items():
        row = row_by_slot.get(slot_id)
        state = row.get("state", "") if row else ""
        face, edge, alpha, _ = STATE_STYLE.get(state, ("#e5e7eb", "#cbd5e1", 0.10, "not selected"))
        lw = 1.0 if state == "box_vehicle_core_supported" else 0.28
        zorder = 5 if state else 1
        add_polygon(ax, slot["polygon_map"], face, edge, alpha, lw, zorder)
    for idx, row in enumerate(rows, start=1):
        if row.get("state") not in {"box_vehicle_core_supported", "box_boundary_conflict", "box_adjacent_conflict"}:
            continue
        slot = slots.get(row["slot_id"])
        if not slot:
            continue
        c = slot_center(slot)
        ax.text(c[0], c[1], str(idx), fontsize=4.5, ha="center", va="center", zorder=10)
    bmin = all_pts.min(axis=0) - 0.5
    bmax = all_pts.max(axis=0) + 0.5
    ax.set_xlim(float(bmin[0]), float(bmax[0]))
    ax.set_ylim(float(bmin[1]), float(bmax[1]))
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Slot box correspondence overview")
    ax.set_xlabel("map x")
    ax.set_ylabel("map y")
    handles = [Patch(facecolor=face, edgecolor=edge, alpha=alpha, label=label) for face, edge, alpha, label in STATE_STYLE.values()]
    handles.append(Patch(facecolor="#e5e7eb", edgecolor="#cbd5e1", alpha=0.10, label="not selected"))
    ax.legend(handles=handles, loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def draw_case_global_map(path: Path, slots: dict[str, dict[str, Any]], row: dict[str, str], ego_pose: np.ndarray | None, scale: float) -> None:
    slot_id = row["slot_id"]
    slot = slots[slot_id]
    center = slot_center(slot)
    adjacent = set(str(s) for s in slot.get("adjacent_slots", []))
    fig, ax = plt.subplots(figsize=(8, 7), dpi=170)
    for sid, other in slots.items():
        other_center = slot_center(other)
        dist = float(np.linalg.norm(other_center - center))
        if sid == slot_id:
            face, edge, alpha, lw, z = "#ef4444", "#7f1d1d", 0.80, 1.6, 7
        elif sid in adjacent or dist <= 1.3:
            face, edge, alpha, lw, z = "#bfdbfe", "#2563eb", 0.58, 0.7, 4
        else:
            continue
        add_polygon(ax, other["polygon_map"], face, edge, alpha, lw, z)
    add_polygon(ax, slot["margin_polygon_map"], "none", "#f59e0b", 0.95, 1.0, 8, "target margin")
    add_polygon(ax, slot["core_polygon_map"], "none", "#16a34a", 0.95, 1.3, 9, "target core")
    best_box = box_polygon(row)
    add_polygon(ax, best_box, "none", "#111827", 0.95, 1.2, 10, "best box")
    ax.text(center[0], center[1], slot_id.replace("slot_", ""), fontsize=8, ha="center", va="center", zorder=12)
    if ego_pose is not None:
        draw_ego(ax, ego_pose, scale, color="#111827", label="anchor ego", zorder=13)
    pts = [np.asarray(slot["margin_polygon_map"], dtype=np.float64), best_box]
    for sid in adjacent:
        if sid in slots:
            pts.append(np.asarray(slots[sid]["polygon_map"], dtype=np.float64))
    if ego_pose is not None:
        pts.append(ego_pose[:2].reshape(1, 2))
    all_pts = np.vstack(pts)
    bmin = all_pts.min(axis=0) - 0.45
    bmax = all_pts.max(axis=0) + 0.45
    ax.set_xlim(float(bmin[0]), float(bmax[0]))
    ax.set_ylim(float(bmin[1]), float(bmax[1]))
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(f"Global map context - {slot_id} frame {row.get('anchor_frame')}")
    ax.set_xlabel("map x")
    ax.set_ylabel("map y")
    ax.legend(loc="upper right", fontsize=7)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def draw_case_pointcloud_map(
    path: Path,
    slots: dict[str, dict[str, Any]],
    row: dict[str, str],
    selected_frames: list[int],
    frame_rows: dict[int, dict[str, str]],
    map_points_dir: Path,
    root: Path,
    scale: float,
) -> tuple[int, np.ndarray | None]:
    slot_id = row["slot_id"]
    slot = slots[slot_id]
    center = slot_center(slot)
    chunks_low: list[np.ndarray] = []
    chunks_vehicle: list[np.ndarray] = []
    ego_poses: list[np.ndarray] = []
    radius = 2.0
    for frame in selected_frames:
        loaded = load_frame_points(frame, frame_rows, map_points_dir, root)
        if loaded is None:
            continue
        points, ego_pose = loaded
        ego_poses.append(ego_pose)
        d = np.linalg.norm(points[:, :2] - center[None, :], axis=1)
        points = points[d <= radius]
        if len(points) == 0:
            continue
        ground_z = float(np.quantile(points[:, 2], 0.08))
        z_rel = points[:, 2] - ground_z
        low = points[(z_rel >= 0.05) & (z_rel < 0.30)]
        vehicle = points[(z_rel >= 0.30) & (z_rel <= 2.20)]
        if len(low):
            chunks_low.append(low)
        if len(vehicle):
            chunks_vehicle.append(vehicle)
    low_points = np.vstack(chunks_low) if chunks_low else np.empty((0, 4), dtype=np.float64)
    vehicle_points = np.vstack(chunks_vehicle) if chunks_vehicle else np.empty((0, 4), dtype=np.float64)
    fig, ax = plt.subplots(figsize=(8, 7), dpi=170)
    for sid, other in slots.items():
        other_center = slot_center(other)
        if sid != slot_id and float(np.linalg.norm(other_center - center)) > 1.25:
            continue
        if sid == slot_id:
            face, edge, alpha, lw, z = "#fee2e2", "#991b1b", 0.60, 1.4, 5
        else:
            face, edge, alpha, lw, z = "#dbeafe", "#60a5fa", 0.30, 0.55, 2
        add_polygon(ax, other["polygon_map"], face, edge, alpha, lw, z)
    add_polygon(ax, slot["margin_polygon_map"], "none", "#f59e0b", 0.95, 1.1, 6, "target margin")
    add_polygon(ax, slot["core_polygon_map"], "none", "#16a34a", 0.95, 1.5, 7, "target core")
    add_polygon(ax, box_polygon(row), "none", "#111827", 0.95, 1.2, 8, "best box")
    if len(low_points):
        ax.scatter(low_points[:, 0], low_points[:, 1], s=2.0, color="#f59e0b", alpha=0.25, linewidths=0, label="accum low-height")
    if len(vehicle_points):
        ax.scatter(vehicle_points[:, 0], vehicle_points[:, 1], s=2.4, color="#ef4444", alpha=0.32, linewidths=0, label="accum vehicle-height")
    if ego_poses:
        draw_ego(ax, ego_poses[0], scale, color="#64748b", label="first ego", zorder=10)
        draw_ego(ax, ego_poses[len(ego_poses) // 2], scale, color="#111827", label="anchor/mid ego", zorder=11)
        draw_ego(ax, ego_poses[-1], scale, color="#2563eb", label="last ego", zorder=10)
    ax.text(center[0], center[1], slot_id.replace("slot_", ""), fontsize=8, ha="center", va="center", zorder=12)
    pts = [np.asarray(slot["margin_polygon_map"], dtype=np.float64), box_polygon(row)]
    if len(vehicle_points):
        pts.append(vehicle_points[:, :2])
    if len(low_points):
        pts.append(low_points[:, :2])
    if ego_poses:
        pts.append(np.vstack([p[:2] for p in ego_poses]))
    all_pts = np.vstack(pts)
    bmin = all_pts.min(axis=0) - 0.45
    bmax = all_pts.max(axis=0) + 0.45
    ax.set_xlim(float(bmin[0]), float(bmax[0]))
    ax.set_ylim(float(bmin[1]), float(bmax[1]))
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(f"Accumulated map point cloud - {slot_id}")
    ax.set_xlabel("map x")
    ax.set_ylabel("map y")
    ax.legend(loc="upper right", fontsize=7)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return int(len(vehicle_points)), (ego_poses[len(ego_poses) // 2] if ego_poses else None)


def copy_camera_image(row: dict[str, str], frame_rows: dict[int, dict[str, str]], assets_dir: Path, root: Path) -> str:
    frame = int(float(row.get("anchor_frame") or 0))
    frame_row = frame_rows.get(frame)
    if frame_row is None:
        return ""
    src = resolve_path(frame_row.get("image_path", ""), root)
    if not src.exists():
        return ""
    dst = assets_dir / f"{row['slot_id']}_frame_{frame:06d}_{src.name}"
    if not dst.exists():
        shutil.copyfile(src, dst)
    return dst.name


def write_report(path: Path, rows: list[dict[str, str]], counts: dict[str, int], overview_name: str) -> None:
    def esc(value: Any) -> str:
        return html.escape(str(value))

    count_items = "".join(f"<li><b>{esc(k)}</b>: {esc(v)}</li>" for k, v in sorted(counts.items()))
    table = []
    for idx, row in enumerate(rows, start=1):
        table.append(
            "<tr>"
            f"<td>{idx}</td>"
            f"<td>{esc(row['slot_id'])}</td>"
            f"<td>{esc(row.get('state', ''))}</td>"
            f"<td>{float(row.get('score') or 0):.3f}</td>"
            f"<td>{esc(row.get('anchor_frame', ''))}</td>"
            f"<td>{esc(row.get('supported_frame_count', ''))}/{esc(row.get('selected_frame_count', ''))}</td>"
            f"<td>{float(row.get('z95_above_ground') or 0):.2f}</td>"
            f"<td>{float(row.get('height_span') or 0):.2f}</td>"
            f"<td>{float(row.get('temporal_support') or 0):.2f}</td>"
            f"<td>{esc(row.get('reason', ''))}</td>"
            f"<td><a href='{esc(row['global_map_asset'])}'><img src='{esc(row['global_map_asset'])}'></a></td>"
            f"<td><a href='{esc(row['pointcloud_asset'])}'><img src='{esc(row['pointcloud_asset'])}'></a></td>"
            f"<td><a href='{esc(row['camera_asset'])}'><img src='{esc(row['camera_asset'])}'></a></td>"
            "</tr>"
        )
    html_text = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Slot Box Correspondence Report</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 24px; color: #111827; }}
    img {{ max-width: 260px; max-height: 170px; border: 1px solid #d1d5db; background: white; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 12px; }}
    th, td {{ border: 1px solid #d1d5db; padding: 5px 6px; vertical-align: top; }}
    th {{ background: #f3f4f6; position: sticky; top: 0; z-index: 2; }}
    .overview img {{ max-width: 100%; max-height: none; }}
    .meta {{ color: #475569; max-width: 1040px; }}
  </style>
</head>
<body>
  <h1>Point Cloud / Camera / Global Map Correspondence</h1>
  <p class="meta">Each row links the v2 slot-constrained box scoring result to the anchor-frame camera image, the target slot location in global map coordinates, and the accumulated selected-frame point cloud around that slot. These are evidence states, not final occupied/free decisions.</p>
  <h2>Counts</h2>
  <ul>{count_items}</ul>
  <h2>Global Overview</h2>
  <div class="overview"><img src="{esc(overview_name)}"></div>
  <h2>Per-slot Correspondence</h2>
  <table>
    <thead>
      <tr>
        <th>#</th><th>slot</th><th>state</th><th>score</th><th>anchor</th><th>support</th><th>z95</th><th>height span</th><th>temporal</th><th>reason</th>
        <th>global map</th><th>accumulated point cloud</th><th>camera photo</th>
      </tr>
    </thead>
    <tbody>{''.join(table)}</tbody>
  </table>
</body>
</html>
"""
    path.write_text(html_text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    root = Path.cwd()
    input_dir = args.input_dir
    output_dir = args.output_dir or input_dir / "correspondence_review"
    assets_dir = output_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    slot_db = load_json(args.slot_database)
    slots = {str(slot["slot_id"]): slot for slot in slot_db["slots"]}
    scale = float(slot_db.get("map_units_per_meter", 1.0))
    frame_rows_list = read_csv(args.frames_csv)
    frame_rows = {int(row["frame"]): row for row in frame_rows_list if row.get("frame")}
    rows = read_csv(input_dir / "slot_box_scores.csv")
    rows.sort(key=lambda row: float(row.get("score") or 0.0), reverse=True)
    if args.max_cases and args.max_cases > 0:
        rows = rows[: args.max_cases]

    overview_path = assets_dir / "slot_box_correspondence_global_overview.png"
    draw_overview_map(overview_path, slots, rows)

    enriched: list[dict[str, str]] = []
    for idx, row in enumerate(rows, start=1):
        slot_id = row["slot_id"]
        selected = parse_frames(row.get("selected_frames", ""))
        pointcloud_path = assets_dir / f"{idx:03d}_{slot_id}_pointcloud_map.png"
        vehicle_count, ego_pose = draw_case_pointcloud_map(pointcloud_path, slots, row, selected, frame_rows, args.map_points_dir, root, scale)
        global_path = assets_dir / f"{idx:03d}_{slot_id}_global_map.png"
        draw_case_global_map(global_path, slots, row, ego_pose, scale)
        camera_name = copy_camera_image(row, frame_rows, assets_dir, root)
        item = dict(row)
        item["global_map_asset"] = report_rel(global_path, output_dir)
        item["pointcloud_asset"] = report_rel(pointcloud_path, output_dir)
        item["camera_asset"] = report_rel(assets_dir / camera_name, output_dir) if camera_name else ""
        item["accumulated_vehicle_height_point_count"] = str(vehicle_count)
        enriched.append(item)
        if idx % 25 == 0:
            print(f"rendered {idx}/{len(rows)} cases", flush=True)

    counts: dict[str, int] = {}
    for row in enriched:
        counts[row["state"]] = counts.get(row["state"], 0) + 1
    write_report(output_dir / "pointcloud_camera_global_map_correspondence_report.html", enriched, counts, report_rel(overview_path, output_dir))
    with (output_dir / "correspondence_index.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = list(enriched[0].keys()) if enriched else []
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(enriched)
    (output_dir / "summary.json").write_text(json.dumps({"case_count": len(enriched), "state_counts": counts}, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output_dir / "pointcloud_camera_global_map_correspondence_report.html"), "case_count": len(enriched), "state_counts": counts}, indent=2))


if __name__ == "__main__":
    main()
