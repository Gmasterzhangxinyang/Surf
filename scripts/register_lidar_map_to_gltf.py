#!/usr/bin/env python3
"""Register accumulated LiDAR local map to GLTF semantic map as a whole.

This script does global map-to-map alignment first, instead of trying to
localize single scans. It uses only LiDAR odometry and LiDAR scans to build a
local point cloud map, then estimates a 2D Sim(2) transform to the GLTF map.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.spatial import cKDTree

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_MPL = True
except Exception:
    HAS_MPL = False

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gltf_lidar_ndt import draw_gltf_map, load_gltf_map, normalize_angle, yaw_to_rot  # noqa: E402


DATASET_ROOT = Path("/home/ParkingAgent/dataset/dataset/dataset")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Register accumulated LiDAR map to GLTF map")
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--odom-csv", type=Path, default=Path("outputs/lidar_only_odometry_full/lidar_only_poses.csv"))
    parser.add_argument("--gltf", type=Path, default=Path("file.gltf"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/lidar_map_global_registration"))
    parser.add_argument("--frame-step", type=int, default=20)
    parser.add_argument("--max-frames", type=int, default=600)
    parser.add_argument("--max-points-per-frame", type=int, default=900)
    parser.add_argument("--map-sample-step", type=float, default=0.06)
    parser.add_argument("--scan-voxel", type=float, default=0.22)
    parser.add_argument("--local-map-voxel", type=float, default=0.20)
    parser.add_argument("--range-min", type=float, default=2.0)
    parser.add_argument("--range-max", type=float, default=28.0)
    parser.add_argument("--z-min", type=float, default=-0.35)
    parser.add_argument("--z-max", type=float, default=1.55)
    parser.add_argument("--sample-points", type=int, default=3000)
    parser.add_argument("--scale-min", type=float, default=0.035)
    parser.add_argument("--scale-max", type=float, default=0.10)
    parser.add_argument("--scale-steps", type=int, default=14)
    parser.add_argument("--yaw-step-deg", type=float, default=5.0)
    parser.add_argument("--xy-step", type=float, default=0.35)
    parser.add_argument("--icp-iterations", type=int, default=40)
    return parser.parse_args()


def voxel_downsample(points: np.ndarray, voxel: float, max_points: int = 0) -> np.ndarray:
    if len(points) == 0:
        return points
    keys = np.floor(points / voxel).astype(np.int64)
    _, idx = np.unique(keys, axis=0, return_index=True)
    idx.sort()
    out = points[idx]
    if max_points > 0 and len(out) > max_points:
        stride = int(math.ceil(len(out) / max_points))
        out = out[::stride][:max_points]
    return out


def load_rows(path: Path) -> list[dict[str, str]]:
    rows = list(csv.DictReader(path.open(newline="")))
    if not rows:
        raise ValueError(f"empty odom csv: {path}")
    return rows


def transform_xy(points: np.ndarray, pose: np.ndarray) -> np.ndarray:
    return points @ yaw_to_rot(float(pose[2])).T + pose[:2]


def load_scan(path: Path, args: argparse.Namespace) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.float32)
    pts = raw.reshape(-1, 4)[:, :3].astype(np.float64)
    r = np.linalg.norm(pts[:, :2], axis=1)
    mask = np.isfinite(pts).all(axis=1)
    mask &= r >= args.range_min
    mask &= r <= args.range_max
    mask &= pts[:, 2] >= args.z_min
    mask &= pts[:, 2] <= args.z_max
    xy = pts[mask, :2]
    return voxel_downsample(xy, args.scan_voxel, args.max_points_per_frame)


def build_local_map(rows: list[dict[str, str]], args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    indices = list(range(0, len(rows), max(1, args.frame_step)))
    if len(indices) > args.max_frames:
        stride = int(math.ceil(len(indices) / args.max_frames))
        indices = indices[::stride][: args.max_frames]
    chunks: list[np.ndarray] = []
    poses: list[np.ndarray] = []
    frames: list[int] = []
    lidar_dir = args.dataset_root / "velodyne"
    for idx in indices:
        row = rows[idx]
        frame = int(row["frame"])
        pose = np.array([float(row["x"]), float(row["y"]), float(row["yaw_rad"])], dtype=np.float64)
        scan = load_scan(lidar_dir / f"{frame:06d}.bin", args)
        if len(scan) == 0:
            continue
        chunks.append(transform_xy(scan, pose))
        poses.append(pose)
        frames.append(frame)
    if not chunks:
        raise ValueError("empty local map")
    points = voxel_downsample(np.vstack(chunks), args.local_map_voxel, args.sample_points * 4)
    return points, np.asarray(poses), np.asarray(frames, dtype=np.int32)


def apply_sim2(points: np.ndarray, scale: float, yaw: float, trans: np.ndarray) -> np.ndarray:
    return scale * (points @ yaw_to_rot(float(yaw)).T) + trans


def transform_poses(poses: np.ndarray, scale: float, yaw: float, trans: np.ndarray) -> np.ndarray:
    out = np.empty_like(poses)
    out[:, :2] = apply_sim2(poses[:, :2], scale, yaw, trans)
    out[:, 2] = [normalize_angle(float(v + yaw)) for v in poses[:, 2]]
    return out


def cost_for(points: np.ndarray, tree: cKDTree, bounds_min: np.ndarray, bounds_max: np.ndarray) -> tuple[float, dict[str, float]]:
    inside = np.all((points >= bounds_min - 0.25) & (points <= bounds_max + 0.25), axis=1)
    if int(inside.sum()) < max(50, len(points) // 8):
        return 1e6, {"inside": float(inside.mean()), "p50": 999.0, "p80": 999.0, "inlier": 0.0, "coverage": 0.0}
    cand = points[inside]
    dist, _ = tree.query(cand, k=1, workers=-1)
    p50 = float(np.percentile(dist, 50))
    p80 = float(np.percentile(dist, 80))
    inlier = float((dist < 0.18).mean())
    keys = np.floor(cand / 0.22).astype(np.int64)
    coverage = float(len(np.unique(keys, axis=0)))
    spread = np.ptp(cand, axis=0)
    area = max(1e-6, float(spread[0] * spread[1]))
    cost = p50 + 0.35 * p80 + 0.65 * (1.0 - inlier) + 0.35 * (1.0 - inside.mean()) + 0.25 / math.sqrt(area) + 0.8 / math.sqrt(max(1.0, coverage))
    return float(cost), {"inside": float(inside.mean()), "p50": p50, "p80": p80, "inlier": inlier, "coverage": coverage}


def initial_search(local_map: np.ndarray, map_points: np.ndarray, gltf_map: object, args: argparse.Namespace) -> tuple[float, float, np.ndarray, float, dict[str, float]]:
    sample = local_map
    if len(sample) > args.sample_points:
        sample = sample[:: int(math.ceil(len(sample) / args.sample_points))][: args.sample_points]
    sample = sample - sample.mean(axis=0)
    tree = cKDTree(map_points)
    xs = np.arange(gltf_map.bounds_min[0], gltf_map.bounds_max[0] + 1e-9, args.xy_step)
    ys = np.arange(gltf_map.bounds_min[1], gltf_map.bounds_max[1] + 1e-9, args.xy_step)
    scales = np.linspace(args.scale_min, args.scale_max, args.scale_steps)
    yaws = np.arange(-math.pi, math.pi, math.radians(args.yaw_step_deg))
    best = (float("inf"), float(scales[0]), 0.0, (gltf_map.bounds_min + gltf_map.bounds_max) * 0.5, {})
    for scale in scales:
        scaled = sample * float(scale)
        for yaw in yaws:
            rotated = scaled @ yaw_to_rot(float(yaw)).T
            for x in xs:
                shifted_x = rotated[:, 0] + x
                x_ok = (shifted_x >= gltf_map.bounds_min[0] - 0.25) & (shifted_x <= gltf_map.bounds_max[0] + 0.25)
                if int(x_ok.sum()) < max(50, len(sample) // 8):
                    continue
                sub = rotated[x_ok]
                sx = shifted_x[x_ok]
                for y in ys:
                    pts = np.column_stack([sx, sub[:, 1] + y])
                    cost, metrics = cost_for(pts, tree, gltf_map.bounds_min, gltf_map.bounds_max)
                    if cost < best[0]:
                        best = (cost, float(scale), float(yaw), np.array([float(x), float(y)]), metrics)

    def objective(v: np.ndarray) -> float:
        scale = float(np.clip(v[0], args.scale_min, args.scale_max))
        pts = apply_sim2(sample, scale, float(v[1]), np.array([float(v[2]), float(v[3])]))
        return cost_for(pts, tree, gltf_map.bounds_min, gltf_map.bounds_max)[0]

    _, scale, yaw, trans, _ = best
    res = minimize(objective, np.array([scale, yaw, trans[0], trans[1]], dtype=np.float64), method="Nelder-Mead", options={"maxiter": 300})
    scale = float(np.clip(res.x[0], args.scale_min, args.scale_max))
    yaw = normalize_angle(float(res.x[1]))
    trans = np.array([float(res.x[2]), float(res.x[3])])
    pts = apply_sim2(sample, scale, yaw, trans)
    final_cost, metrics = cost_for(pts, tree, gltf_map.bounds_min, gltf_map.bounds_max)
    return scale, yaw, trans, final_cost, metrics


def icp_refine(local_map: np.ndarray, map_points: np.ndarray, scale: float, yaw: float, trans: np.ndarray, args: argparse.Namespace) -> tuple[float, float, np.ndarray, dict[str, float]]:
    sample = local_map
    if len(sample) > args.sample_points:
        sample = sample[:: int(math.ceil(len(sample) / args.sample_points))][: args.sample_points]
    tree = cKDTree(map_points)
    for _ in range(args.icp_iterations):
        world = apply_sim2(sample, scale, yaw, trans)
        dist, idx = tree.query(world, k=1, workers=-1)
        keep = dist < np.percentile(dist, 70)
        if int(keep.sum()) < 20:
            break
        src = sample[keep]
        tgt = map_points[idx[keep]]
        src_scaled = src * scale
        src_cent = src_scaled.mean(axis=0)
        tgt_cent = tgt.mean(axis=0)
        a = src_scaled - src_cent
        b = tgt - tgt_cent
        u, _, vt = np.linalg.svd(a.T @ b)
        r = vt.T @ u.T
        if np.linalg.det(r) < 0:
            vt[-1, :] *= -1
            r = vt.T @ u.T
        yaw_delta = math.atan2(float(r[1, 0]), float(r[0, 0]))
        yaw = normalize_angle(yaw + yaw_delta)
        trans = tgt_cent - (src_cent @ yaw_to_rot(yaw_delta).T)
    final = apply_sim2(sample, scale, yaw, trans)
    _, metrics = cost_for(final, tree, map_points.min(axis=0), map_points.max(axis=0))
    return scale, yaw, trans, metrics


def render(path: Path, gltf_map: object, lidar_world: np.ndarray, traj_world: np.ndarray) -> None:
    if not HAS_MPL:
        return
    fig, ax = plt.subplots(figsize=(10, 12), dpi=240)
    draw_gltf_map(ax, gltf_map)
    if len(lidar_world) > 250000:
        lidar_world = lidar_world[:: int(math.ceil(len(lidar_world) / 250000))]
    ax.scatter(lidar_world[:, 0], lidar_world[:, 1], s=0.08, c="#2563eb", alpha=0.35, linewidths=0, label="accumulated LiDAR map")
    ax.plot(traj_world[:, 0], traj_world[:, 1], color="#dc2626", linewidth=1.7, label="LiDAR odom trajectory")
    ax.scatter([traj_world[0, 0]], [traj_world[0, 1]], c="#16a34a", s=34, label="start", zorder=8)
    ax.scatter([traj_world[-1, 0]], [traj_world[-1, 1]], c="#dc2626", s=34, label="end", zorder=8)
    ax.set_title("Accumulated LiDAR map registered to GLTF map")
    ax.legend(loc="upper right", fontsize=7)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_rows(args.odom_csv)
    local_map, local_poses, frames = build_local_map(rows, args)
    gltf_map = load_gltf_map(args.gltf, args.map_sample_step)
    scale, yaw, trans, coarse_cost, coarse_metrics = initial_search(local_map, gltf_map.reference_points, gltf_map, args)
    # ICP refinement is intentionally conservative; use coarse result if ICP hurts metrics.
    icp_scale, icp_yaw, icp_trans, icp_metrics = icp_refine(local_map, gltf_map.reference_points, scale, yaw, trans, args)
    base_pts = apply_sim2(local_map, scale, yaw, trans)
    icp_pts = apply_sim2(local_map, icp_scale, icp_yaw, icp_trans)
    base_cost, base_metrics = cost_for(base_pts, cKDTree(gltf_map.reference_points), gltf_map.bounds_min, gltf_map.bounds_max)
    icp_cost, _ = cost_for(icp_pts, cKDTree(gltf_map.reference_points), gltf_map.bounds_min, gltf_map.bounds_max)
    if icp_cost < base_cost:
        scale, yaw, trans, metrics, final_cost = icp_scale, icp_yaw, icp_trans, icp_metrics, icp_cost
    else:
        metrics, final_cost = base_metrics, base_cost
    lidar_world = apply_sim2(local_map, scale, yaw, trans)
    traj_world = transform_poses(local_poses, scale, yaw, trans)
    render(args.output_dir / "lidar_map_registered.png", gltf_map, lidar_world, traj_world)
    np.savez_compressed(args.output_dir / "registered_lidar_map.npz", lidar_map=lidar_world.astype(np.float32), trajectory=traj_world.astype(np.float32), frames=frames)
    with (args.output_dir / "registered_trajectory.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["frame", "x", "y", "yaw"])
        writer.writeheader()
        for frame, pose in zip(frames, traj_world):
            writer.writerow({"frame": int(frame), "x": pose[0], "y": pose[1], "yaw": pose[2]})
    inside = np.all((lidar_world >= gltf_map.bounds_min - 0.5) & (lidar_world <= gltf_map.bounds_max + 0.5), axis=1)
    summary = {
        "frames_used": int(len(frames)),
        "points_used": int(len(local_map)),
        "scale": scale,
        "yaw": yaw,
        "translation": trans.tolist(),
        "coarse_cost": coarse_cost,
        "final_cost": final_cost,
        "metrics": metrics,
        "lidar_inside_map_plus_margin_ratio": float(inside.mean()),
        "trajectory_bounds_min": traj_world[:, :2].min(axis=0).tolist(),
        "trajectory_bounds_max": traj_world[:, :2].max(axis=0).tolist(),
        "note": "Whole accumulated LiDAR map registered to GLTF. Use visual overlay for acceptance; repeated parking structure remains ambiguous.",
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[summary] {args.output_dir / 'summary.json'}")
    print(f"[overlay] {args.output_dir / 'lidar_map_registered.png'}")
    print(f"[map] {args.output_dir / 'registered_lidar_map.npz'}")
    print(f"[trajectory] {args.output_dir / 'registered_trajectory.csv'}")
    print(f"[quality] cost={final_cost:.3f} p50={metrics['p50']:.3f} p80={metrics['p80']:.3f} inlier={metrics['inlier']:.3f} inside={summary['lidar_inside_map_plus_margin_ratio']:.3f}")


if __name__ == "__main__":
    main()
