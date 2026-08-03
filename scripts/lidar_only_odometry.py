#!/usr/bin/env python3
"""Pure LiDAR 2D scan-to-submap odometry for KITTI-style Velodyne bins.

This deliberately does not read dataset poses. It estimates a local trajectory
from consecutive LiDAR scans and writes enough diagnostics to judge whether the
result is usable before trying to align it to a parking-map frame.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy.spatial import cKDTree

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_MPL = True
except Exception:
    HAS_MPL = False


DATASET_ROOT = Path("/home/ParkingAgent/dataset/dataset/dataset")


@dataclass
class IcpResult:
    pose: np.ndarray
    rmse: float
    fitness: float
    inliers: int
    iterations: int
    accepted: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run pure LiDAR scan-to-submap odometry")
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/lidar_only_odometry"))
    parser.add_argument("--start-index", type=int, default=1)
    parser.add_argument("--num-frames", type=int, default=600, help="0 means all available frames from start")
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--range-min", type=float, default=2.0)
    parser.add_argument("--range-max", type=float, default=35.0)
    parser.add_argument("--z-min", type=float, default=0.05)
    parser.add_argument("--z-max", type=float, default=2.30)
    parser.add_argument("--scan-voxel", type=float, default=0.22)
    parser.add_argument("--map-voxel", type=float, default=0.18)
    parser.add_argument("--max-scan-points", type=int, default=3500)
    parser.add_argument("--submap-scans", type=int, default=25)
    parser.add_argument("--max-correspondence", type=float, default=0.90)
    parser.add_argument("--min-inliers", type=int, default=120)
    parser.add_argument("--min-fitness", type=float, default=0.12)
    parser.add_argument("--max-step-xy", type=float, default=1.20)
    parser.add_argument("--max-step-yaw-deg", type=float, default=12.0)
    parser.add_argument("--icp-iterations", type=int, default=25)
    parser.add_argument("--render-every", type=int, default=50)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def normalize_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def rot2(yaw: float) -> np.ndarray:
    c = math.cos(yaw)
    s = math.sin(yaw)
    return np.array([[c, -s], [s, c]], dtype=np.float64)


def transform_xy(points: np.ndarray, pose: np.ndarray) -> np.ndarray:
    return points @ rot2(float(pose[2])).T + pose[:2]


def compose_delta(delta_yaw: float, delta_t: np.ndarray, pose: np.ndarray) -> np.ndarray:
    out = pose.copy()
    out[:2] = rot2(delta_yaw) @ pose[:2] + delta_t
    out[2] = normalize_angle(float(pose[2] + delta_yaw))
    return out


def relative_step(prev: np.ndarray, cur: np.ndarray) -> tuple[float, float]:
    return float(np.linalg.norm(cur[:2] - prev[:2])), abs(normalize_angle(float(cur[2] - prev[2])))


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


def load_scan(path: Path, args: argparse.Namespace) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.float32)
    if raw.size % 4 != 0:
        raise ValueError(f"invalid KITTI Velodyne file: {path}")
    pts = raw.reshape(-1, 4)[:, :3].astype(np.float64)
    r = np.linalg.norm(pts[:, :2], axis=1)
    mask = np.isfinite(pts).all(axis=1)
    mask &= r >= args.range_min
    mask &= r <= args.range_max
    mask &= pts[:, 2] >= args.z_min
    mask &= pts[:, 2] <= args.z_max
    xy = pts[mask, :2]
    return voxel_downsample(xy, args.scan_voxel, args.max_scan_points)


def best_fit_delta(source_world: np.ndarray, target_world: np.ndarray) -> tuple[float, np.ndarray]:
    src_cent = source_world.mean(axis=0)
    tgt_cent = target_world.mean(axis=0)
    src = source_world - src_cent
    tgt = target_world - tgt_cent
    cov = src.T @ tgt
    u, _, vt = np.linalg.svd(cov)
    r = vt.T @ u.T
    if np.linalg.det(r) < 0:
        vt[-1, :] *= -1.0
        r = vt.T @ u.T
    yaw = math.atan2(float(r[1, 0]), float(r[0, 0]))
    t = tgt_cent - r @ src_cent
    return yaw, t


def align_scan_to_submap(scan_xy: np.ndarray, submap_xy: np.ndarray, initial_pose: np.ndarray, args: argparse.Namespace) -> IcpResult:
    pose = initial_pose.copy()
    tree = cKDTree(submap_xy)
    best_pose = pose.copy()
    best_rmse = float("inf")
    best_fitness = 0.0
    best_inliers = 0
    used_itr = 0

    for itr in range(args.icp_iterations):
        world = transform_xy(scan_xy, pose)
        dist, idx = tree.query(world, k=1, workers=-1)
        inlier_mask = dist <= args.max_correspondence
        if int(inlier_mask.sum()) < max(8, args.min_inliers // 3):
            break

        inlier_dist = dist[inlier_mask]
        keep_count = max(8, int(len(inlier_dist) * 0.82))
        local_order = np.argpartition(inlier_dist, keep_count - 1)[:keep_count]
        global_idx = np.flatnonzero(inlier_mask)[local_order]
        src = world[global_idx]
        tgt = submap_xy[idx[global_idx]]

        dyaw, dt = best_fit_delta(src, tgt)
        dyaw = float(np.clip(dyaw, -math.radians(4.0), math.radians(4.0)))
        step_norm = float(np.linalg.norm(dt))
        if step_norm > 0.45:
            dt *= 0.45 / step_norm

        pose = compose_delta(dyaw, dt, pose)
        rmse = float(np.sqrt(np.mean(inlier_dist[local_order] ** 2)))
        fitness = float(len(global_idx) / max(1, len(scan_xy)))
        if rmse < best_rmse or (fitness > best_fitness and rmse < best_rmse * 1.12):
            best_pose = pose.copy()
            best_rmse = rmse
            best_fitness = fitness
            best_inliers = int(len(global_idx))
            used_itr = itr + 1

        if abs(dyaw) < 2e-4 and float(np.linalg.norm(dt)) < 2e-3:
            break

    accepted = best_inliers >= args.min_inliers and best_fitness >= args.min_fitness and math.isfinite(best_rmse)
    return IcpResult(best_pose, best_rmse, best_fitness, best_inliers, used_itr, accepted)


def render_outputs(output_dir: Path, rows: Sequence[dict[str, object]], submaps: Sequence[np.ndarray]) -> None:
    if not HAS_MPL or not rows:
        return
    traj = np.asarray([[float(r["x"]), float(r["y"]), float(r["yaw_rad"])] for r in rows], dtype=np.float64)

    fig, ax = plt.subplots(figsize=(10, 8), dpi=180)
    ax.plot(traj[:, 0], traj[:, 1], color="#dc2626", linewidth=1.8, label="LiDAR-only odometry")
    ax.scatter([traj[0, 0]], [traj[0, 1]], c="#16a34a", s=32, label="start", zorder=3)
    ax.scatter([traj[-1, 0]], [traj[-1, 1]], c="#dc2626", s=32, label="end", zorder=3)
    step = max(1, len(traj) // 60)
    dirs = np.column_stack([np.cos(traj[::step, 2]), np.sin(traj[::step, 2])])
    ax.quiver(traj[::step, 0], traj[::step, 1], dirs[:, 0], dirs[:, 1], angles="xy", scale_units="xy", scale=4.0, width=0.002)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, color="#cbd5e1", linewidth=0.5, alpha=0.6)
    ax.set_title("Pure LiDAR scan-to-submap odometry")
    ax.set_xlabel("local x [m]")
    ax.set_ylabel("local y [m]")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "trajectory.png")
    plt.close(fig)

    if submaps:
        pts = np.vstack(submaps)
        if len(pts) > 350000:
            pts = pts[:: int(math.ceil(len(pts) / 350000))]
        fig, ax = plt.subplots(figsize=(11, 9), dpi=180)
        ax.scatter(pts[:, 0], pts[:, 1], s=0.08, c="#334155", alpha=0.28, linewidths=0)
        ax.plot(traj[:, 0], traj[:, 1], color="#ef4444", linewidth=1.4)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, color="#cbd5e1", linewidth=0.4, alpha=0.45)
        ax.set_title("Accumulated LiDAR local map")
        ax.set_xlabel("local x [m]")
        ax.set_ylabel("local y [m]")
        fig.tight_layout()
        fig.savefig(output_dir / "local_map.png")
        plt.close(fig)


def run() -> None:
    args = parse_args()
    ensure_dir(args.output_dir)
    lidar_dir = args.dataset_root / "velodyne"
    files = sorted(lidar_dir.glob("*.bin"))
    files = [p for p in files if int(p.stem) >= args.start_index]
    files = files[:: max(1, args.step)]
    if args.num_frames > 0:
        files = files[: args.num_frames]
    if not files:
        raise SystemExit(f"no lidar files found under {lidar_dir}")

    rows: list[dict[str, object]] = []
    recent_scans_world: list[np.ndarray] = []
    map_chunks_for_render: list[np.ndarray] = []
    prev_pose = np.zeros(3, dtype=np.float64)
    prev_delta = np.zeros(3, dtype=np.float64)

    for seq, path in enumerate(files):
        frame = int(path.stem)
        scan = load_scan(path, args)
        if len(scan) < 30:
            continue

        if not rows:
            pose = np.zeros(3, dtype=np.float64)
            result = IcpResult(pose, 0.0, 1.0, len(scan), 0, True)
        else:
            submap = voxel_downsample(np.vstack(recent_scans_world), args.map_voxel, 90000)
            initial = prev_pose + prev_delta
            initial[2] = normalize_angle(float(initial[2]))
            result = align_scan_to_submap(scan, submap, initial, args)
            pose = result.pose
            step_xy, step_yaw = relative_step(prev_pose, pose)
            if (not result.accepted) or step_xy > args.max_step_xy or step_yaw > math.radians(args.max_step_yaw_deg):
                pose = initial.copy()
                result = IcpResult(pose, result.rmse, result.fitness, result.inliers, result.iterations, False)

        if rows:
            prev_delta = pose - prev_pose
            prev_delta[2] = normalize_angle(float(prev_delta[2]))
        prev_pose = pose.copy()
        world_scan = transform_xy(scan, pose)
        recent_scans_world.append(world_scan)
        if len(recent_scans_world) > args.submap_scans:
            recent_scans_world.pop(0)
        if args.render_every > 0 and seq % args.render_every == 0:
            map_chunks_for_render.append(voxel_downsample(world_scan, args.map_voxel, 5000))

        row = {
            "seq": seq,
            "frame": frame,
            "x": float(pose[0]),
            "y": float(pose[1]),
            "yaw_rad": float(pose[2]),
            "rmse": float(result.rmse),
            "fitness": float(result.fitness),
            "inliers": int(result.inliers),
            "iterations": int(result.iterations),
            "accepted": bool(result.accepted),
            "scan_points": int(len(scan)),
        }
        rows.append(row)
        if not args.quiet:
            print(
                f"frame={frame:06d} x={pose[0]:.3f} y={pose[1]:.3f} yaw={pose[2]:.3f} "
                f"rmse={result.rmse:.3f} fit={result.fitness:.2f} inliers={result.inliers} accepted={result.accepted}"
            )

    csv_path = args.output_dir / "lidar_only_poses.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    accepted = np.asarray([bool(r["accepted"]) for r in rows], dtype=bool)
    fitness = np.asarray([float(r["fitness"]) for r in rows], dtype=np.float64)
    rmse = np.asarray([float(r["rmse"]) for r in rows], dtype=np.float64)
    traj = np.asarray([[float(r["x"]), float(r["y"])] for r in rows], dtype=np.float64)
    path_len = float(np.linalg.norm(np.diff(traj, axis=0), axis=1).sum()) if len(traj) > 1 else 0.0
    summary = {
        "frames_processed": len(rows),
        "start_frame": int(rows[0]["frame"]),
        "end_frame": int(rows[-1]["frame"]),
        "accepted_ratio": float(accepted.mean()),
        "mean_fitness": float(fitness.mean()),
        "median_fitness": float(np.median(fitness)),
        "mean_rmse": float(rmse[np.isfinite(rmse)].mean()) if np.isfinite(rmse).any() else None,
        "path_length_m": path_len,
        "net_displacement_m": float(np.linalg.norm(traj[-1] - traj[0])) if len(traj) > 1 else 0.0,
        "note": "Local LiDAR-only odometry; origin and yaw are arbitrary and not aligned to GLTF/map coordinates.",
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    render_outputs(args.output_dir, rows, map_chunks_for_render)

    print(f"[csv] {csv_path}")
    print(f"[summary] {args.output_dir / 'summary.json'}")
    print(f"[trajectory] {args.output_dir / 'trajectory.png'}")
    print(f"[local_map] {args.output_dir / 'local_map.png'}")


if __name__ == "__main__":
    run()
