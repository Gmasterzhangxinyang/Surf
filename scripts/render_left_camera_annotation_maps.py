#!/usr/bin/env python3
"""Replace annotation location maps with explicit LiDAR and left-camera poses."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon, Wedge

from build_camera_visible_annotation_pack import _ego_xy


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--universe", type=Path, required=True)
    parser.add_argument("--camera-manifest", type=Path, required=True)
    parser.add_argument("--frames-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--half-fov-deg", type=float, default=50.5)
    args = parser.parse_args()
    universe = json.loads(args.universe.read_text(encoding="utf-8"))
    camera = json.loads(args.camera_manifest.read_text(encoding="utf-8"))
    origin = np.asarray(camera["camera_origin_lidar_m"][:2], dtype=np.float64)
    camera_yaw = math.radians(float(camera["camera_yaw_deg"]))
    visible_ids = {
        str(row["slot_id"]) for row in camera["rows"] if int(row["view_count"]) > 0
    }
    history = set(int(value) for value in universe["history_frames"])
    with args.frames_csv.open("r", encoding="utf-8", newline="") as handle:
        frame_rows = [
            row for row in csv.DictReader(handle) if int(row["frame"]) in history
        ]
    route_map = np.asarray(
        [[float(row["map_x"]), float(row["map_y"])] for row in frame_rows],
        dtype=np.float64,
    )
    anchor = route_map[-1]
    anchor_yaw = float(frame_rows[-1]["map_yaw"])
    scale = float(universe["map_units_per_meter"])
    route = _ego_xy(route_map, anchor, anchor_yaw, scale)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for target_id in sorted(visible_ids):
        figure, axis = plt.subplots(figsize=(11, 10), dpi=150)
        points = [route]
        for slot in universe["slots"]:
            slot_id = str(slot["slot_id"])
            polygon = _ego_xy(
                np.asarray(slot["polygon_map"], dtype=np.float64),
                anchor,
                anchor_yaw,
                scale,
            )
            points.append(polygon)
            if slot_id == target_id:
                style = ("#f0abfc", "#a21caf", 3.5, 0.95, 4)
            elif slot_id in visible_ids:
                style = ("#dbeafe", "#2563eb", 1.4, 0.72, 3)
            else:
                style = ("#e5e7eb", "#9ca3af", 0.7, 0.42, 1)
            axis.add_patch(
                Polygon(
                    polygon,
                    closed=True,
                    facecolor=style[0],
                    edgecolor=style[1],
                    linewidth=style[2],
                    alpha=style[3],
                    zorder=style[4],
                )
            )
            if slot_id == target_id or slot_id in visible_ids:
                center = polygon.mean(axis=0)
                axis.text(
                    center[0],
                    center[1],
                    slot_id.replace("slot_", ""),
                    ha="center",
                    va="center",
                    fontsize=8 if slot_id == target_id else 6,
                    fontweight="bold" if slot_id == target_id else "normal",
                    zorder=5,
                )
        axis.plot(route[:, 0], route[:, 1], color="#111827", linewidth=2.0, zorder=2)
        axis.scatter([0], [0], marker="x", s=90, color="#ef4444", zorder=6)
        axis.text(0.1, -0.35, "LiDAR / map pose", color="#b91c1c", fontsize=9)
        axis.add_patch(
            Wedge(
                tuple(origin),
                10.0,
                math.degrees(camera_yaw) - args.half_fov_deg,
                math.degrees(camera_yaw) + args.half_fov_deg,
                facecolor="#22d3ee18",
                edgecolor="#0891b2",
                linestyle="--",
                linewidth=1.3,
                zorder=0,
            )
        )
        axis.scatter([origin[0]], [origin[1]], marker="o", s=75, color="#0891b2", zorder=7)
        axis.arrow(
            origin[0],
            origin[1],
            2.5 * math.cos(camera_yaw),
            2.5 * math.sin(camera_yaw),
            width=0.08,
            head_width=0.55,
            color="#0891b2",
            length_includes_head=True,
            zorder=7,
        )
        axis.text(
            origin[0] + 0.15,
            origin[1] + 0.25,
            f"Left camera (+{origin[1]:.2f}m lateral)",
            color="#0e7490",
            fontsize=9,
            fontweight="bold",
        )
        combined = np.concatenate(points)
        margin = 3.5
        axis.set_xlim(combined[:, 0].min() - margin, combined[:, 0].max() + margin)
        axis.set_ylim(combined[:, 1].min() - margin, combined[:, 1].max() + margin)
        axis.set_aspect("equal")
        axis.grid(color="#d7dee8", linewidth=0.5)
        axis.set_xlabel("LiDAR forward (m)")
        axis.set_ylabel("LiDAR left (m)")
        axis.set_title(
            f"{target_id}: magenta target; cyan = calibrated left-camera pose/FOV",
            fontsize=13,
            fontweight="bold",
        )
        figure.savefig(args.output_dir / f"{target_id}.png", bbox_inches="tight")
        plt.close(figure)
    print(
        json.dumps(
            {
                "map_count": len(visible_ids),
                "camera_origin_lidar_m": camera["camera_origin_lidar_m"],
                "camera_yaw_deg": camera["camera_yaw_deg"],
                "output_dir": str(args.output_dir),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
