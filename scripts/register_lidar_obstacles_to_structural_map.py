#!/usr/bin/env python3
"""Register LiDAR obstacle map to structural GLTF layers only."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gltf_lidar_ndt import load_gltf_map  # noqa: E402
from register_lidar_map_to_gltf import (  # noqa: E402
    apply_sim2,
    build_local_map,
    cost_for,
    initial_search,
    render,
    transform_poses,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Register LiDAR high-obstacle map to structural map layers")
    parser.add_argument("--dataset-root", type=Path, default=Path("/home/ParkingAgent/dataset/dataset/dataset"))
    parser.add_argument("--odom-csv", type=Path, default=Path("outputs/lidar_only_odometry_full/lidar_only_poses.csv"))
    parser.add_argument("--gltf", type=Path, default=Path("file.gltf"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/lidar_obstacle_structural_registration"))
    parser.add_argument("--layers", default="wall,arrester,elevator")
    parser.add_argument("--frame-step", type=int, default=80)
    parser.add_argument("--max-frames", type=int, default=150)
    parser.add_argument("--max-points-per-frame", type=int, default=500)
    parser.add_argument("--map-sample-step", type=float, default=0.04)
    parser.add_argument("--scan-voxel", type=float, default=0.18)
    parser.add_argument("--local-map-voxel", type=float, default=0.18)
    parser.add_argument("--range-min", type=float, default=2.0)
    parser.add_argument("--range-max", type=float, default=28.0)
    parser.add_argument("--z-min", type=float, default=0.25)
    parser.add_argument("--z-max", type=float, default=2.20)
    parser.add_argument("--sample-points", type=int, default=900)
    parser.add_argument("--scale-min", type=float, default=0.035)
    parser.add_argument("--scale-max", type=float, default=0.10)
    parser.add_argument("--scale-steps", type=int, default=8)
    parser.add_argument("--yaw-step-deg", type=float, default=10.0)
    parser.add_argument("--xy-step", type=float, default=0.55)
    parser.add_argument("--icp-iterations", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    gltf_map = load_gltf_map(args.gltf, args.map_sample_step)
    target_layers = [x.strip() for x in args.layers.split(",") if x.strip()]
    chunks = []
    for name in target_layers:
        layer = gltf_map.layers.get(name)
        if layer is not None and len(layer.points):
            chunks.append(layer.points)
    if not chunks:
        raise SystemExit(f"no target layer points for {target_layers}")
    target = np.vstack(chunks)
    local_rows = __import__("register_lidar_map_to_gltf").load_rows(args.odom_csv)
    local_map, local_poses, frames = build_local_map(local_rows, args)
    scale, yaw, trans, coarse_cost, coarse_metrics = initial_search(local_map, target, gltf_map, args)
    lidar_world = apply_sim2(local_map, scale, yaw, trans)
    final_cost, metrics = cost_for(lidar_world, cKDTree(target), gltf_map.bounds_min, gltf_map.bounds_max)
    traj_world = transform_poses(local_poses, scale, yaw, trans)
    render(args.output_dir / "structural_overlay.png", gltf_map, lidar_world, traj_world)
    np.savez_compressed(args.output_dir / "structural_registered_lidar_map.npz", lidar_map=lidar_world.astype(np.float32), trajectory=traj_world.astype(np.float32), frames=frames)
    summary = {
        "target_layers": target_layers,
        "frames_used": int(len(frames)),
        "points_used": int(len(local_map)),
        "target_points": int(len(target)),
        "scale": scale,
        "yaw": yaw,
        "translation": trans.tolist(),
        "coarse_cost": coarse_cost,
        "final_cost": final_cost,
        "metrics": metrics,
        "trajectory_bounds_min": traj_world[:, :2].min(axis=0).tolist(),
        "trajectory_bounds_max": traj_world[:, :2].max(axis=0).tolist(),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[summary] {args.output_dir / 'summary.json'}")
    print(f"[overlay] {args.output_dir / 'structural_overlay.png'}")
    print(f"[quality] cost={final_cost:.3f} p50={metrics['p50']:.3f} p80={metrics['p80']:.3f} inlier={metrics['inlier']:.3f} inside={metrics['inside']:.3f}")


if __name__ == "__main__":
    main()
