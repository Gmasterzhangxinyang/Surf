#!/usr/bin/env python3
"""Render annotation maps with audited pose semantics and vehicle footprint."""

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


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--universe", type=Path, required=True)
    p.add_argument("--camera-manifest", type=Path, required=True)
    p.add_argument("--frames-csv", type=Path, required=True)
    p.add_argument("--pose-scan", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--half-fov-deg", type=float, default=50.5)
    p.add_argument("--vehicle-length-m", type=float, default=4.6)
    p.add_argument("--vehicle-width-m", type=float, default=1.9)
    a = p.parse_args()

    universe = json.loads(a.universe.read_text(encoding="utf-8"))
    camera = json.loads(a.camera_manifest.read_text(encoding="utf-8"))
    scan = json.loads(a.pose_scan.read_text(encoding="utf-8"))
    origin = np.asarray(camera["camera_origin_lidar_m"][:2], dtype=float)
    camera_yaw = math.radians(float(camera["camera_yaw_deg"]))
    visible = {str(x["slot_id"]) for x in camera["rows"] if int(x["view_count"]) > 0}
    history = set(int(x) for x in universe["history_frames"])
    with a.frames_csv.open(newline="", encoding="utf-8") as f:
        rows = [x for x in csv.DictReader(f) if int(x["frame"]) in history]
    route_map = np.asarray([[float(x["map_x"]), float(x["map_y"])] for x in rows])
    anchor = route_map[-1]
    yaw = float(rows[-1]["map_yaw"])
    scale = float(universe["map_units_per_meter"])
    route = _ego_xy(route_map, anchor, yaw, scale)
    offset = float(rows[-1].get("longitudinal_pose_offset_m", 0.0))

    half_l = a.vehicle_length_m / 2
    half_w = a.vehicle_width_m / 2
    footprint = np.asarray(
        [[-half_l, -half_w], [half_l, -half_w], [half_l, half_w], [-half_l, half_w]]
    )
    a.output_dir.mkdir(parents=True, exist_ok=True)

    for target in sorted(visible):
        fig, ax = plt.subplots(figsize=(11, 10), dpi=150)
        extents = [route, footprint]
        for slot in universe["slots"]:
            sid = str(slot["slot_id"])
            poly = _ego_xy(np.asarray(slot["polygon_map"]), anchor, yaw, scale)
            extents.append(poly)
            if sid == target:
                style = ("#f0abfc", "#a21caf", 3.5, 0.95, 5)
            elif sid in visible:
                style = ("#dbeafe", "#2563eb", 1.4, 0.72, 3)
            else:
                style = ("#e5e7eb", "#9ca3af", 0.7, 0.40, 1)
            ax.add_patch(
                Polygon(
                    poly,
                    closed=True,
                    facecolor=style[0],
                    edgecolor=style[1],
                    linewidth=style[2],
                    alpha=style[3],
                    zorder=style[4],
                )
            )
            if sid == target or sid in visible:
                center = poly.mean(axis=0)
                ax.text(
                    center[0],
                    center[1],
                    sid.replace("slot_", ""),
                    ha="center",
                    va="center",
                    fontsize=8 if sid == target else 6,
                    fontweight="bold" if sid == target else "normal",
                    zorder=6,
                )

        ax.plot(route[:, 0], route[:, 1], color="#111827", linewidth=2, zorder=2)
        ax.add_patch(
            Polygon(
                footprint,
                closed=True,
                facecolor="#fecaca55",
                edgecolor="#dc2626",
                linestyle="--",
                linewidth=2,
                zorder=7,
                label="4.6 x 1.9 m reference footprint",
            )
        )
        ax.scatter([0], [0], marker="x", s=105, color="#dc2626", zorder=9)
        ax.text(
            -2.25,
            -1.35,
            "red X = LiDAR/map pose\n(dashed body is a reference footprint)",
            color="#991b1b",
            fontsize=8,
            zorder=10,
        )
        ax.add_patch(
            Wedge(
                tuple(origin),
                10,
                math.degrees(camera_yaw) - a.half_fov_deg,
                math.degrees(camera_yaw) + a.half_fov_deg,
                facecolor="#22d3ee18",
                edgecolor="#0891b2",
                linestyle="--",
                linewidth=1.3,
                zorder=0,
            )
        )
        ax.scatter([origin[0]], [origin[1]], s=72, color="#0891b2", zorder=9)
        ax.arrow(
            origin[0],
            origin[1],
            2.5 * math.cos(camera_yaw),
            2.5 * math.sin(camera_yaw),
            width=0.08,
            head_width=0.55,
            color="#0891b2",
            length_includes_head=True,
            zorder=8,
        )
        ax.text(
            origin[0] + 0.15,
            origin[1] + 0.2,
            f"left camera ({origin[0]:+.2f}, {origin[1]:+.2f}) m",
            color="#0e7490",
            fontsize=8,
            fontweight="bold",
            zorder=10,
        )
        points = np.concatenate(extents)
        ax.set_xlim(points[:, 0].min() - 3.5, points[:, 0].max() + 3.5)
        ax.set_ylim(points[:, 1].min() - 3.5, points[:, 1].max() + 3.5)
        ax.set_aspect("equal")
        ax.grid(color="#d7dee8", linewidth=0.5)
        ax.set_xlabel("forward from corrected LiDAR pose (m)")
        ax.set_ylabel("left from corrected LiDAR pose (m)")
        ax.set_title(
            f"{target} | anchor pose shifted {offset:+.2f} m longitudinally\n"
            f"frame-only optimum {scan['best']['offset_m']:+.2f} m; "
            f"static residual {scan['current_zero']['trimmed35_rmse_m']:.3f}"
            f" → {scan['best']['trimmed35_rmse_m']:.3f} m",
            fontsize=12,
            fontweight="bold",
        )
        fig.savefig(a.output_dir / f"{target}.png", bbox_inches="tight")
        plt.close(fig)

    print(
        json.dumps(
            {
                "maps": len(visible),
                "pose_offset_m": offset,
                "camera_origin_lidar_m": camera["camera_origin_lidar_m"],
                "camera_yaw_deg": camera["camera_yaw_deg"],
                "output_dir": str(a.output_dir),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
