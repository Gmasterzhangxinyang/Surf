#!/usr/bin/env python3
"""Add global map slot overlays to the highest-score frame review report."""

from __future__ import annotations

import argparse
import html
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon as MplPolygon


DEFAULT_PART1 = Path("outputs/full_icpark_allframes_vehicle_cluster")
DEFAULT_REVIEW = Path("outputs/highest_score_frame_review")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build global map overlays for highest-score review cases")
    parser.add_argument("--part1-dir", type=Path, default=DEFAULT_PART1)
    parser.add_argument("--review-dir", type=Path, default=DEFAULT_REVIEW)
    return parser.parse_args()


def load_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: object) -> None:
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def resolve(path: str, repo_root: Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else repo_root / p


def report_rel(path: Path, report_dir: Path) -> str:
    try:
        return path.resolve().relative_to(report_dir.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def slot_center(slot: dict[str, object]) -> np.ndarray:
    return np.asarray(slot["center_map"], dtype=np.float64)


def add_slot(
    ax: plt.Axes,
    slot: dict[str, object],
    *,
    face: str,
    edge: str,
    alpha: float,
    lw: float,
    zorder: int,
    label: str | None = None,
) -> None:
    poly = np.asarray(slot["polygon_map"], dtype=np.float64)
    ax.add_patch(MplPolygon(poly, closed=True, facecolor=face, edgecolor=edge, alpha=alpha, linewidth=lw, zorder=zorder))
    if label:
        c = slot_center(slot)
        ax.text(c[0], c[1], label, fontsize=7, ha="center", va="center", color="#111827", zorder=zorder + 1)


def add_poly_outline(ax: plt.Axes, coords: object, *, color: str, lw: float, zorder: int, label: str | None = None) -> None:
    poly = np.asarray(coords, dtype=np.float64)
    ax.add_patch(MplPolygon(poly, closed=True, facecolor="none", edgecolor=color, alpha=0.95, linewidth=lw, zorder=zorder, label=label))


def draw_ego(ax: plt.Axes, ego_pose: np.ndarray, scale: float, *, zorder: int = 20) -> None:
    x, y, yaw = map(float, ego_pose[:3])
    arrow_len = 2.0 * scale
    ax.scatter([x], [y], s=34, color="#111827", marker="o", zorder=zorder, label="ego pose")
    ax.arrow(
        x,
        y,
        arrow_len * math.cos(yaw),
        arrow_len * math.sin(yaw),
        width=0.008,
        head_width=0.05,
        head_length=0.07,
        color="#111827",
        length_includes_head=True,
        zorder=zorder,
    )


def axis_from_points(ax: plt.Axes, points: np.ndarray, pad: float) -> None:
    bmin = points.min(axis=0) - pad
    bmax = points.max(axis=0) + pad
    ax.set_xlim(float(bmin[0]), float(bmax[0]))
    ax.set_ylim(float(bmin[1]), float(bmax[1]))
    ax.set_aspect("equal", adjustable="box")


def load_npz(path: Path) -> dict[str, np.ndarray]:
    data = np.load(path)
    return {key: data[key] for key in data.files}


def draw_global_map(
    output: Path,
    slots: dict[str, dict[str, object]],
    slot: dict[str, object],
    case: dict[str, object],
    ego_pose: np.ndarray,
    scale: float,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 8), dpi=180)
    target_id = str(case["slot_id"])
    adjacent = set(str(s) for s in slot.get("adjacent_slots", []))

    for slot_id, other in slots.items():
        if slot_id == target_id:
            continue
        if slot_id in adjacent:
            add_slot(ax, other, face="#bfdbfe", edge="#2563eb", alpha=0.72, lw=0.7, zorder=3)
        else:
            add_slot(ax, other, face="#e5e7eb", edge="#cbd5e1", alpha=0.22, lw=0.25, zorder=1)

    add_slot(ax, slot, face="#ef4444", edge="#7f1d1d", alpha=0.82, lw=1.8, zorder=8, label=target_id.replace("slot_", ""))
    add_poly_outline(ax, slot["core_polygon_map"], color="#16a34a", lw=1.1, zorder=10, label="target core")
    draw_ego(ax, ego_pose, scale)

    all_pts = np.vstack([np.asarray(s["polygon_map"], dtype=np.float64) for s in slots.values()])
    axis_from_points(ax, all_pts, pad=0.5)
    ax.set_title(f"Global map slot location - {case['label']} frame {case['frame_id']} {target_id}")
    ax.set_xlabel("map x")
    ax.set_ylabel("map y")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def draw_zoom_map(
    output: Path,
    slots: dict[str, dict[str, object]],
    slot: dict[str, object],
    case: dict[str, object],
    npz: dict[str, np.ndarray],
    scale: float,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 8), dpi=180)
    target_id = str(case["slot_id"])
    adjacent_ids = [str(s) for s in slot.get("adjacent_slots", []) if str(s) in slots]

    roi = np.asarray(npz.get("roi_obstacle_map_xyzi", np.empty((0, 4))), dtype=np.float64)
    core = np.asarray(npz.get("core_obstacle_map_xyzi", np.empty((0, 4))), dtype=np.float64)
    full = np.asarray(npz.get("points_map_xyzi", np.empty((0, 4))), dtype=np.float64)
    ego_pose = np.asarray(npz["ego_map_pose"], dtype=np.float64)

    nearby_polys = [np.asarray(slot["margin_polygon_map"], dtype=np.float64)]
    for adj_id in adjacent_ids:
        nearby_polys.append(np.asarray(slots[adj_id]["polygon_map"], dtype=np.float64))

    if len(full):
        ax.scatter(full[:, 0], full[:, 1], s=0.7, color="#94a3b8", alpha=0.18, linewidths=0, zorder=1, label="all frame points")

    for adj_id in adjacent_ids:
        add_slot(ax, slots[adj_id], face="#bfdbfe", edge="#2563eb", alpha=0.60, lw=0.9, zorder=4, label=adj_id.replace("slot_", ""))

    add_poly_outline(ax, slot["margin_polygon_map"], color="#f59e0b", lw=1.1, zorder=5, label="target margin")
    add_slot(ax, slot, face="#fee2e2", edge="#991b1b", alpha=0.72, lw=1.5, zorder=6, label=target_id.replace("slot_", ""))
    add_poly_outline(ax, slot["core_polygon_map"], color="#16a34a", lw=1.7, zorder=8, label="target core")

    if len(roi):
        ax.scatter(roi[:, 0], roi[:, 1], s=8, color="#ef4444", alpha=0.55, linewidths=0, zorder=9, label="ROI obstacle points")
    if len(core):
        ax.scatter(core[:, 0], core[:, 1], s=13, color="#7c3aed", alpha=0.90, linewidths=0, zorder=10, label="core obstacle points")

    draw_ego(ax, ego_pose, scale, zorder=15)

    extents = nearby_polys[:]
    if len(roi):
        extents.append(roi[:, :2])
    if len(core):
        extents.append(core[:, :2])
    extents.append(ego_pose[:2].reshape(1, 2))
    pts = np.vstack(extents)
    axis_from_points(ax, pts, pad=0.45)
    ax.set_title(f"Map zoom with slot polygons and point cloud - frame {case['frame_id']} {target_id}")
    ax.set_xlabel("map x")
    ax.set_ylabel("map y")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def update_report(review_dir: Path, summary: dict[str, object]) -> None:
    parts = []
    for case in summary["cases"]:  # type: ignore[index]
        assets = case["assets"]
        evidence = case["evidence"]
        counts = case["counts"]
        map_global = report_rel(review_dir / assets["global_map"], review_dir)
        map_zoom = report_rel(review_dir / assets["map_zoom"], review_dir)
        camera_overlay = report_rel(resolve(assets["camera_overlay"], Path.cwd()), review_dir)
        camera_crop = report_rel(resolve(assets["camera_crop"], Path.cwd()), review_dir)
        parts.append(
            "<section>"
            f"<h2>{html.escape(str(case['label']))}: {html.escape(str(case['slot_id']))}, frame {case['frame_id']}</h2>"
            f"<p>score={float(evidence.get('max_vehicle_like_score', 0.0)):.3f} | "
            f"frame_state={html.escape(str(evidence.get('frame_state')))} | "
            f"core_obstacle_points={counts.get('slot_core_obstacle_points')} | "
            f"projected_obstacle_points_in_camera={counts.get('projected_obstacle_points_in_camera')}</p>"
            f"<p>global map target slot center={case.get('slot_center_map')} | adjacent_slots={html.escape(', '.join(case.get('adjacent_slots', [])))}</p>"
            "<div class='grid'>"
            f"<div><h3>Global Map Slot Location</h3><img src='{html.escape(map_global)}'></div>"
            f"<div><h3>Map Zoom: Slot + Point Cloud</h3><img src='{html.escape(map_zoom)}'></div>"
            f"<div><h3>Camera Overlay</h3><img src='{html.escape(camera_overlay)}'></div>"
            f"<div><h3>Camera Crop</h3><img src='{html.escape(camera_crop)}'></div>"
            "</div>"
            "</section>"
        )

    html_text = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Highest Score Frame Map Review</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 24px; color: #111827; }}
    .note {{ padding: 12px; background: #eff6ff; border: 1px solid #bfdbfe; margin-bottom: 16px; }}
    section {{ border-top: 1px solid #d1d5db; padding: 18px 0; }}
    .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }}
    img {{ max-width: 100%; border: 1px solid #d1d5db; background: white; }}
    h1, h2, h3 {{ margin-bottom: 8px; }}
  </style>
</head>
<body>
  <h1>Highest Score Frame Map Review</h1>
  <div class="note">
    This report shows the corresponding parking slot on the global map, not only a local BEV debug view.
    Global map colors: red=target slot, blue=adjacent slots, gray=other slots, green outline=target core, black arrow=ego pose.
    Map zoom colors: purple=core obstacle points, red=ROI obstacle points.
  </div>
  {''.join(parts)}
</body>
</html>
"""
    (review_dir / "highest_score_frame_report.html").write_text(html_text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    repo_root = Path.cwd()
    review_dir = args.review_dir
    slot_db = load_json(args.part1_dir / "slot_database.json")
    slots = {str(slot["slot_id"]): slot for slot in slot_db["slots"]}  # type: ignore[index]
    scale = float(slot_db["map_units_per_meter"])  # type: ignore[index]
    summary_path = review_dir / "highest_score_frame_summary.json"
    summary = load_json(summary_path)

    for case in summary["cases"]:  # type: ignore[index]
        slot_id = str(case["slot_id"])
        slot = slots[slot_id]
        assets = case["assets"]
        roi_npz_path = resolve(str(assets["roi_pointcloud_npz"]), repo_root)
        npz = load_npz(roi_npz_path)
        ego_pose = np.asarray(npz["ego_map_pose"], dtype=np.float64)
        prefix = f"{case['label']}_frame_{int(case['frame_id']):06d}_{slot_id}"
        global_name = f"{prefix}_global_map_slot.png"
        zoom_name = f"{prefix}_map_zoom_slot_points.png"
        draw_global_map(review_dir / global_name, slots, slot, case, ego_pose, scale)
        draw_zoom_map(review_dir / zoom_name, slots, slot, case, npz, scale)
        assets["global_map"] = global_name
        assets["map_zoom"] = zoom_name
        case["slot_center_map"] = [float(v) for v in slot_center(slot)]
        case["adjacent_slots"] = [str(s) for s in slot.get("adjacent_slots", [])]

    write_json(summary_path, summary)
    update_report(review_dir, summary)
    print(f"[done] map review updated: {review_dir / 'highest_score_frame_report.html'}")
    for case in summary["cases"]:  # type: ignore[index]
        print(case["label"], case["frame_id"], case["slot_id"], case["assets"]["global_map"], case["assets"]["map_zoom"])


if __name__ == "__main__":
    main()
