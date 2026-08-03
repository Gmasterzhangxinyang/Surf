#!/usr/bin/env python3
"""Locally refine a LiDAR-map registration seed."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gltf_lidar_ndt import load_gltf_map, normalize_angle, yaw_to_rot  # noqa: E402
from register_lidar_map_to_gltf import apply_sim2, build_local_map, cost_for, render, transform_poses  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Refine LiDAR-map registration from seed")
    parser.add_argument("--seed-summary", type=Path, required=True)
    parser.add_argument("--odom-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, default=Path("/home/ParkingAgent/dataset/dataset/dataset"))
    parser.add_argument("--gltf", type=Path, default=Path("file.gltf"))
    parser.add_argument("--frame-step", type=int, default=80)
    parser.add_argument("--max-frames", type=int, default=180)
    parser.add_argument("--max-points-per-frame", type=int, default=450)
    parser.add_argument("--sample-points", type=int, default=1800)
    parser.add_argument("--map-sample-step", type=float, default=0.05)
    parser.add_argument("--scan-voxel", type=float, default=0.20)
    parser.add_argument("--local-map-voxel", type=float, default=0.18)
    parser.add_argument("--range-min", type=float, default=2.0)
    parser.add_argument("--range-max", type=float, default=28.0)
    parser.add_argument("--z-min", type=float, default=-0.35)
    parser.add_argument("--z-max", type=float, default=1.55)
    parser.add_argument("--scale-window", type=float, default=0.018)
    parser.add_argument("--yaw-window-deg", type=float, default=18.0)
    parser.add_argument("--translation-window", type=float, default=2.0)
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, str]]:
    rows = list(csv.DictReader(path.open(newline="")))
    if not rows:
        raise ValueError(f"empty odom csv: {path}")
    return rows


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seed = json.loads(args.seed_summary.read_text())
    s0 = float(seed["scale"])
    yaw0 = float(seed["yaw"])
    t0 = np.asarray(seed["translation"], dtype=np.float64)
    rows = load_rows(args.odom_csv)
    local_map, local_poses, frames = build_local_map(rows, args)
    if len(local_map) > args.sample_points:
        local_map = local_map[:: int(np.ceil(len(local_map) / args.sample_points))][: args.sample_points]
    gltf_map = load_gltf_map(args.gltf, args.map_sample_step)
    tree = cKDTree(gltf_map.reference_points)

    s_min, s_max = max(0.001, s0 - args.scale_window), s0 + args.scale_window
    yaw_window = np.deg2rad(args.yaw_window_deg)
    tx_min, tx_max = t0[0] - args.translation_window, t0[0] + args.translation_window
    ty_min, ty_max = t0[1] - args.translation_window, t0[1] + args.translation_window

    def objective(v: np.ndarray) -> float:
        scale = float(np.clip(v[0], s_min, s_max))
        yaw = float(np.clip(v[1], yaw0 - yaw_window, yaw0 + yaw_window))
        tx = float(np.clip(v[2], tx_min, tx_max))
        ty = float(np.clip(v[3], ty_min, ty_max))
        pts = apply_sim2(local_map, scale, yaw, np.array([tx, ty]))
        cost, metrics = cost_for(pts, tree, gltf_map.bounds_min, gltf_map.bounds_max)
        outside = 1.0 - metrics["inside"]
        return cost + 0.8 * outside

    best = minimize(
        objective,
        np.array([s0, yaw0, t0[0], t0[1]], dtype=np.float64),
        method="Nelder-Mead",
        options={"maxiter": 500, "xatol": 1e-5, "fatol": 1e-5},
    )
    scale = float(np.clip(best.x[0], s_min, s_max))
    yaw = normalize_angle(float(np.clip(best.x[1], yaw0 - yaw_window, yaw0 + yaw_window)))
    trans = np.array([float(np.clip(best.x[2], tx_min, tx_max)), float(np.clip(best.x[3], ty_min, ty_max))])
    lidar_world = apply_sim2(local_map, scale, yaw, trans)
    final_cost, metrics = cost_for(lidar_world, tree, gltf_map.bounds_min, gltf_map.bounds_max)
    traj_world = transform_poses(local_poses, scale, yaw, trans)
    render(args.output_dir / "refined_lidar_map_registered.png", gltf_map, lidar_world, traj_world)
    with (args.output_dir / "refined_sampled_trajectory.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["frame", "x", "y", "yaw"])
        writer.writeheader()
        for frame, pose in zip(frames, traj_world):
            writer.writerow({"frame": int(frame), "x": pose[0], "y": pose[1], "yaw": pose[2]})
    summary = {
        "seed": str(args.seed_summary),
        "frames_used": int(len(frames)),
        "points_used": int(len(local_map)),
        "scale": scale,
        "yaw": yaw,
        "translation": trans.tolist(),
        "final_cost": final_cost,
        "metrics": metrics,
        "trajectory_bounds_min": traj_world[:, :2].min(axis=0).tolist(),
        "trajectory_bounds_max": traj_world[:, :2].max(axis=0).tolist(),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[summary] {args.output_dir / 'summary.json'}")
    print(f"[overlay] {args.output_dir / 'refined_lidar_map_registered.png'}")
    print(f"[quality] cost={final_cost:.3f} p50={metrics['p50']:.3f} p80={metrics['p80']:.3f} inlier={metrics['inlier']:.3f} inside={metrics['inside']:.3f}")


if __name__ == "__main__":
    main()
