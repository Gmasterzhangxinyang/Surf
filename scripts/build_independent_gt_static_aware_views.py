#!/usr/bin/env python3
"""Render prediction-blind slot evidence with semantic static-map explanations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon
import numpy as np
from scipy.spatial import cKDTree

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from build_gt_neutral_lidar_views import _contact_sheets, _sample
from gltf_lidar_ndt import load_gltf_map
from parking_slot_hybrid_3d.geometry import map_xy_to_slot_m, metric_slot
from parking_slot_hybrid_3d.io import load_known_slots


def _extent(points: np.ndarray) -> tuple[float, float]:
    if len(points) < 3:
        return 0.0, 0.0
    q = np.quantile(points[:, :2], [0.05, 0.95], axis=0)
    values = np.sort(q[1] - q[0])
    return float(values[0]), float(values[1])


def _render(
    pack_path: Path,
    slot,
    static_map_xy: np.ndarray,
    output: Path,
    threshold_m: float,
) -> dict:
    with np.load(pack_path, allow_pickle=False) as archive:
        points = np.asarray(archive["points_local_xyzi"], dtype=np.float64)
        target = np.asarray(archive["polygon_local_m"], dtype=np.float64)
        core = np.asarray(archive["core_polygon_local_m"], dtype=np.float64)
        margin = np.asarray(archive["margin_polygon_local_m"], dtype=np.float64)
        selected = np.asarray(archive["selected_frames"], dtype=np.int64)
        valid = np.asarray(archive["valid_frames"], dtype=np.int64)

    extent = np.vstack((target, margin))
    x0, y0 = extent.min(axis=0) - 0.7
    x1, y1 = extent.max(axis=0) + 0.7
    local = (
        (points[:, 0] >= x0)
        & (points[:, 0] <= x1)
        & (points[:, 1] >= y0)
        & (points[:, 1] <= y1)
    )
    static_local = map_xy_to_slot_m(static_map_xy, slot)
    static_keep = (
        (static_local[:, 0] >= x0 - threshold_m)
        & (static_local[:, 0] <= x1 + threshold_m)
        & (static_local[:, 1] >= y0 - threshold_m)
        & (static_local[:, 1] <= y1 + threshold_m)
    )
    static_local = static_local[static_keep]

    obstacle = local & (points[:, 2] > 0.30) & (points[:, 2] <= 2.80)
    obstacle_indices = np.flatnonzero(obstacle)
    explained = np.zeros(len(obstacle_indices), dtype=bool)
    distances = np.full(len(obstacle_indices), np.inf, dtype=np.float64)
    if len(static_local) and len(obstacle_indices):
        distances, _ = cKDTree(static_local).query(points[obstacle_indices, :2], k=1)
        explained = distances <= threshold_m
    explained_indices = obstacle_indices[explained]
    residual_indices = obstacle_indices[~explained]
    floor_indices = np.flatnonzero(local & (points[:, 2] <= 0.30))

    floor_draw = _sample(floor_indices, 12_000)
    explained_draw = _sample(explained_indices, 18_000)
    residual_draw = _sample(residual_indices, 18_000)

    figure, axes = plt.subplots(1, 2, figsize=(13, 6.4), dpi=130)
    bev, side = axes
    if len(floor_draw):
        bev.scatter(
            points[floor_draw, 0],
            points[floor_draw, 1],
            s=2,
            c="#aeb8c4",
            alpha=0.20,
            linewidths=0,
            rasterized=True,
        )
    if len(static_local):
        static_draw = static_local[
            np.linspace(0, len(static_local) - 1, min(len(static_local), 12_000), dtype=np.int64)
        ]
        bev.scatter(
            static_draw[:, 0],
            static_draw[:, 1],
            s=5,
            marker="x",
            c="#111827",
            alpha=0.40,
            linewidths=0.6,
            rasterized=True,
        )
    if len(explained_draw):
        bev.scatter(
            points[explained_draw, 0],
            points[explained_draw, 1],
            s=3,
            c="#f59e0b",
            alpha=0.50,
            linewidths=0,
            rasterized=True,
        )
        side.scatter(
            points[explained_draw, 0],
            points[explained_draw, 2],
            s=3,
            c="#f59e0b",
            alpha=0.42,
            linewidths=0,
            rasterized=True,
        )
    if len(residual_draw):
        colors = np.clip(points[residual_draw, 2], 0.3, 2.5)
        bev.scatter(
            points[residual_draw, 0],
            points[residual_draw, 1],
            s=3,
            c=colors,
            cmap="turbo",
            vmin=0.3,
            vmax=2.5,
            alpha=0.72,
            linewidths=0,
            rasterized=True,
        )
        side.scatter(
            points[residual_draw, 0],
            points[residual_draw, 2],
            s=3,
            c=colors,
            cmap="turbo",
            vmin=0.3,
            vmax=2.5,
            alpha=0.72,
            linewidths=0,
            rasterized=True,
        )
    for axis in (bev,):
        axis.add_patch(
            Polygon(target, closed=True, fill=False, edgecolor="#246bdb", linewidth=3)
        )
        axis.add_patch(
            Polygon(core, closed=True, fill=False, edgecolor="#132238", linewidth=2)
        )
    bev.set_xlim(x0, x1)
    bev.set_ylim(y0, y1)
    bev.set_aspect("equal")
    bev.grid(color="#dbe3ed", linewidth=0.6)
    bev.set_title("BEV: static-map explained vs unexplained returns")
    bev.set_xlabel("slot local x (m)")
    bev.set_ylabel("slot local y (m)")
    bev.legend(
        handles=[
            Line2D([0], [0], marker="x", color="#111827", linestyle="", label="semantic static map"),
            Line2D([0], [0], marker="o", color="#f59e0b", linestyle="", label="static-explained LiDAR"),
            Line2D([0], [0], marker="o", color="#246bdb", linestyle="", label="unexplained residual"),
        ],
        loc="lower center",
        frameon=False,
        ncol=3,
    )
    side.axhspan(-0.1, 0.3, color="#e8eef5")
    side.axhline(0.3, color="#64748b", linestyle="--", linewidth=1.2)
    side.set_xlim(x0, x1)
    side.set_ylim(-0.1, 2.8)
    side.grid(color="#dbe3ed", linewidth=0.6)
    side.set_title("Side view: orange=static explained, color=unexplained")
    side.set_xlabel("slot local x (m)")
    side.set_ylabel("ground-normalized z (m)")

    residual_points = points[residual_indices]
    short_extent, long_extent = _extent(residual_points)
    explained_ratio = float(len(explained_indices) / max(len(obstacle_indices), 1))
    figure.suptitle(
        f"{slot.slot_id} | prediction-blind static-aware review | "
        f"valid {len(valid)}/{len(selected)} | static {explained_ratio:.1%} | "
        f"residual extent {short_extent:.2f}×{long_extent:.2f} m",
        fontsize=13,
        fontweight="bold",
    )
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)
    return {
        "slot_id": slot.slot_id,
        "image": str(output),
        "selected_frame_count": int(len(selected)),
        "valid_frame_count": int(len(valid)),
        "above_ground_point_count": int(len(obstacle_indices)),
        "static_explained_point_count": int(len(explained_indices)),
        "unexplained_point_count": int(len(residual_indices)),
        "static_explained_ratio": explained_ratio,
        "unexplained_short_extent_m": short_extent,
        "unexplained_long_extent_m": long_extent,
        "nearest_static_distance_median_m": (
            float(np.median(distances)) if len(distances) else None
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--universe", type=Path, required=True)
    parser.add_argument("--slot-db", type=Path, required=True)
    parser.add_argument("--gltf", type=Path, required=True)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--static-distance-m", type=float, default=0.35)
    parser.add_argument("--map-sample-step", type=float, default=0.012)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    args.output_dir.mkdir(parents=True)

    universe = json.loads(args.universe.read_text(encoding="utf-8"))
    ids = [str(row["slot_id"]) for row in universe["slots"]]
    slots, map_units_per_meter = load_known_slots(args.slot_db)
    by_id = {
        slot.slot_id: metric_slot(slot, map_units_per_meter)
        for slot in slots
    }
    gltf = load_gltf_map(args.gltf, args.map_sample_step)
    layers = [
        gltf.layers[name].points
        for name in ("wall", "elevator", "arrester")
        if name in gltf.layers and len(gltf.layers[name].points)
    ]
    if not layers:
        raise RuntimeError("no wall/elevator/arrester semantic layers found")
    static_map_xy = np.vstack(layers)

    rows = []
    for slot_id in ids:
        rows.append(
            _render(
                args.evidence_dir / f"{slot_id}.npz",
                by_id[slot_id],
                static_map_xy,
                args.output_dir / "slot_views" / f"{slot_id}.png",
                args.static_distance_m,
            )
        )
    sheets = _contact_sheets(rows, args.output_dir / "contact_sheets")
    manifest = {
        "schema_version": "independent-gt-static-aware-review/1.0",
        "prediction_blind": True,
        "part1_fields_used": [],
        "state_labels_used": [],
        "semantic_static_layers": ["wall", "elevator", "arrester"],
        "static_distance_m": args.static_distance_m,
        "map_sample_step": args.map_sample_step,
        "pose_uncertainty_note": "0.35m association includes current 0.15-0.21m structural residual",
        "rows": rows,
        "contact_sheets": sheets,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "slot_count": len(rows),
                "static_map_point_count": len(static_map_xy),
                "output_dir": str(args.output_dir),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
