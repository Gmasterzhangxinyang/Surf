#!/usr/bin/env python3
"""Apply whole-map registration transform to full LiDAR-only trajectory."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_MPL = True
except Exception:
    HAS_MPL = False

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gltf_lidar_ndt import draw_gltf_map, load_gltf_map, normalize_angle, yaw_to_rot  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply LiDAR-map registration to full odometry CSV")
    parser.add_argument("--registration-summary", type=Path, required=True)
    parser.add_argument("--odom-csv", type=Path, default=Path("outputs/lidar_only_odometry_full/lidar_only_poses.csv"))
    parser.add_argument("--gltf", type=Path, default=Path("file.gltf"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--map-sample-step", type=float, default=0.06)
    return parser.parse_args()


def transform_xy(points: np.ndarray, scale: float, yaw: float, trans: np.ndarray) -> np.ndarray:
    return scale * (points @ yaw_to_rot(yaw).T) + trans


def render(path: Path, gltf_map: object, poses: np.ndarray) -> None:
    if not HAS_MPL:
        return
    fig, ax = plt.subplots(figsize=(10, 12), dpi=240)
    draw_gltf_map(ax, gltf_map)
    ax.plot(poses[:, 0], poses[:, 1], color="#dc2626", linewidth=1.4, label="full LiDAR trajectory")
    ax.scatter([poses[0, 0]], [poses[0, 1]], c="#16a34a", s=34, label="start", zorder=8)
    ax.scatter([poses[-1, 0]], [poses[-1, 1]], c="#dc2626", s=34, label="end", zorder=8)
    ax.set_title("Full LiDAR trajectory registered to GLTF map")
    ax.legend(loc="upper right", fontsize=7)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    reg = json.loads(args.registration_summary.read_text())
    scale = float(reg["scale"])
    yaw = float(reg["yaw"])
    trans = np.asarray(reg["translation"], dtype=np.float64)
    rows = list(csv.DictReader(args.odom_csv.open(newline="")))
    local = np.asarray([[float(r["x"]), float(r["y"]), float(r["yaw_rad"])] for r in rows], dtype=np.float64)
    out = np.empty_like(local)
    out[:, :2] = transform_xy(local[:, :2], scale, yaw, trans)
    out[:, 2] = [normalize_angle(float(v + yaw)) for v in local[:, 2]]

    csv_path = args.output_dir / "full_registered_trajectory.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["frame", "x", "y", "yaw"])
        writer.writeheader()
        for row, pose in zip(rows, out):
            writer.writerow({"frame": int(row["frame"]), "x": pose[0], "y": pose[1], "yaw": pose[2]})

    gltf_map = load_gltf_map(args.gltf, args.map_sample_step)
    render(args.output_dir / "full_registered_trajectory.png", gltf_map, out)
    inside = (
        (out[:, 0] >= gltf_map.bounds_min[0] - 0.5)
        & (out[:, 0] <= gltf_map.bounds_max[0] + 0.5)
        & (out[:, 1] >= gltf_map.bounds_min[1] - 0.5)
        & (out[:, 1] <= gltf_map.bounds_max[1] + 0.5)
    )
    summary = {
        "frames": len(rows),
        "source_registration": str(args.registration_summary),
        "scale": scale,
        "yaw": yaw,
        "translation": trans.tolist(),
        "bounds_min": out[:, :2].min(axis=0).tolist(),
        "bounds_max": out[:, :2].max(axis=0).tolist(),
        "inside_map_plus_margin_ratio": float(inside.mean()),
        "path_length_map_units": float(np.linalg.norm(np.diff(out[:, :2], axis=0), axis=1).sum()),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[csv] {csv_path}")
    print(f"[summary] {args.output_dir / 'summary.json'}")
    print(f"[trajectory] {args.output_dir / 'full_registered_trajectory.png'}")


if __name__ == "__main__":
    main()
