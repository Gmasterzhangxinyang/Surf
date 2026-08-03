#!/usr/bin/env python3
"""Map + LiDAR localization pipeline.

Pipeline:
  1. use LiDAR-only odometry CSV as a motion prior
  2. build a sparse local LiDAR map from selected scans
  3. estimate a Sim(2) local-LiDAR -> GLTF-map alignment
  4. refine each scan with NDT scan-to-map
  5. smooth/refine the pose sequence with a lightweight pose-graph prior

This script does not read dataset pose files.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy.optimize import least_squares, minimize
from scipy.spatial import cKDTree

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_MPL = True
except Exception:
    HAS_MPL = False

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gltf_lidar_ndt import build_ndt_grid, draw_gltf_map, load_gltf_map, ndt_align, normalize_angle, yaw_to_rot  # noqa: E402


DATASET_ROOT = Path("/home/ParkingAgent/dataset/dataset/dataset")


@dataclass
class Sim2:
    scale: float
    yaw: float
    trans: np.ndarray
    score: float
    inlier_ratio: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Localize KITTI Velodyne scans on GLTF map using LiDAR odom + NDT")
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--gltf", type=Path, default=Path("file.gltf"))
    parser.add_argument("--odom-csv", type=Path, default=Path("outputs/lidar_only_odometry_1200/lidar_only_poses.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/lidar_map_ndt_optimized"))
    parser.add_argument("--num-frames", type=int, default=0, help="0 means all rows in odom CSV")
    parser.add_argument("--map-sample-step", type=float, default=0.06)
    parser.add_argument("--map-voxel", type=float, default=0.25)
    parser.add_argument("--scan-voxel", type=float, default=0.18)
    parser.add_argument("--range-min", type=float, default=2.0)
    parser.add_argument("--range-max", type=float, default=28.0)
    parser.add_argument("--z-min", type=float, default=-0.35)
    parser.add_argument("--z-max", type=float, default=1.55)
    parser.add_argument("--max-scan-points", type=int, default=2800)
    parser.add_argument("--local-map-frame-step", type=int, default=20)
    parser.add_argument("--local-map-max-frames", type=int, default=80)
    parser.add_argument("--local-map-max-points", type=int, default=1800)
    parser.add_argument("--coarse-sample-points", type=int, default=700)
    parser.add_argument("--scale-min", type=float, default=0.035)
    parser.add_argument("--scale-max", type=float, default=0.14)
    parser.add_argument("--scale-steps", type=int, default=13)
    parser.add_argument("--yaw-step-deg", type=float, default=10.0)
    parser.add_argument("--xy-step", type=float, default=0.65)
    parser.add_argument("--ndt-iterations", type=int, default=30)
    parser.add_argument("--refine-step", type=int, default=1, help="Refine every Nth odometry row")
    parser.add_argument("--pose-graph", action="store_true", help="Run lightweight pose-graph smoothing")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


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


def transform_xy(points: np.ndarray, pose: np.ndarray) -> np.ndarray:
    return points @ yaw_to_rot(float(pose[2])).T + pose[:2]


def load_odom_rows(path: Path, num_frames: int) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append(row)
            if num_frames > 0 and len(rows) >= num_frames:
                break
    if not rows:
        raise ValueError(f"no rows in {path}")
    return rows


def rows_to_poses(rows: Sequence[dict[str, object]]) -> np.ndarray:
    return np.asarray([[float(r["x"]), float(r["y"]), float(r["yaw_rad"])] for r in rows], dtype=np.float64)


def load_scan(path: Path, args: argparse.Namespace) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.float32)
    if raw.size % 4 != 0:
        raise ValueError(f"invalid Velodyne bin: {path}")
    pts = raw.reshape(-1, 4)[:, :3].astype(np.float64)
    r = np.linalg.norm(pts[:, :2], axis=1)
    mask = np.isfinite(pts).all(axis=1)
    mask &= r >= args.range_min
    mask &= r <= args.range_max
    mask &= pts[:, 2] >= args.z_min
    mask &= pts[:, 2] <= args.z_max
    xy = pts[mask, :2]
    return voxel_downsample(xy, args.scan_voxel, args.max_scan_points)


def build_local_lidar_map(rows: Sequence[dict[str, object]], local_poses: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    indices = list(range(0, len(rows), max(1, args.local_map_frame_step)))
    if len(indices) > args.local_map_max_frames:
        stride = int(math.ceil(len(indices) / args.local_map_max_frames))
        indices = indices[::stride][: args.local_map_max_frames]
    chunks: list[np.ndarray] = []
    lidar_dir = args.dataset_root / "velodyne"
    for idx in indices:
        frame = int(rows[idx]["frame"])
        path = lidar_dir / f"{frame:06d}.bin"
        if not path.exists():
            continue
        scan = load_scan(path, args)
        if len(scan) == 0:
            continue
        scan = voxel_downsample(scan, args.scan_voxel * 1.6, args.local_map_max_points)
        chunks.append(transform_xy(scan, local_poses[idx]))
    if not chunks:
        raise ValueError("empty local LiDAR map")
    return voxel_downsample(np.vstack(chunks), args.scan_voxel * 1.25, args.coarse_sample_points * 8)


def apply_sim2(points: np.ndarray, sim: Sim2) -> np.ndarray:
    return sim.scale * (points @ yaw_to_rot(sim.yaw).T) + sim.trans


def poses_apply_sim2(local_poses: np.ndarray, sim: Sim2) -> np.ndarray:
    out = np.empty_like(local_poses)
    out[:, :2] = apply_sim2(local_poses[:, :2], sim)
    out[:, 2] = [normalize_angle(float(v + sim.yaw)) for v in local_poses[:, 2]]
    return out


def map_match_cost(points: np.ndarray, tree: cKDTree, bounds_min: np.ndarray, bounds_max: np.ndarray) -> tuple[float, float]:
    inside = np.all((points >= bounds_min - 0.35) & (points <= bounds_max + 0.35), axis=1)
    if int(inside.sum()) < max(20, len(points) // 8):
        return 1e3, 0.0
    cand = points[inside]
    dist, _ = tree.query(cand, k=1, workers=-1)
    keep = max(15, int(len(dist) * 0.70))
    trimmed = np.partition(dist, keep - 1)[:keep]
    inlier_ratio = float((dist < 0.22).sum() / max(1, len(points)))
    hit_keys = np.floor(cand / 0.22).astype(np.int64)
    unique_hits = len(np.unique(hit_keys, axis=0))
    spread = np.ptp(cand, axis=0)
    spread_area = max(1e-6, float(spread[0] * spread[1]))
    outside_penalty = 1.0 - float(inside.mean())
    coverage_penalty = 1.20 / math.sqrt(max(1, unique_hits))
    collapse_penalty = 0.35 / math.sqrt(max(0.05, spread_area))
    score = float(trimmed.mean() + 0.45 * outside_penalty + coverage_penalty + collapse_penalty + 0.12 / math.sqrt(max(1, (dist < 0.22).sum())))
    return score, inlier_ratio


def estimate_sim2(local_map: np.ndarray, map_points: np.ndarray, bounds_min: np.ndarray, bounds_max: np.ndarray, args: argparse.Namespace) -> Sim2:
    tree = cKDTree(map_points)
    sample = local_map
    if len(sample) > args.coarse_sample_points:
        sample = sample[:: int(math.ceil(len(sample) / args.coarse_sample_points))][: args.coarse_sample_points]
    sample = sample - sample.mean(axis=0)

    xs = np.arange(bounds_min[0], bounds_max[0] + 1e-9, args.xy_step)
    ys = np.arange(bounds_min[1], bounds_max[1] + 1e-9, args.xy_step)
    scales = np.linspace(args.scale_min, args.scale_max, args.scale_steps)
    yaws = np.arange(-math.pi, math.pi, math.radians(args.yaw_step_deg))

    best = Sim2(float(scales[0]), 0.0, (bounds_min + bounds_max) * 0.5, float("inf"), 0.0)
    for scale in scales:
        scaled = sample * float(scale)
        for yaw in yaws:
            rotated = scaled @ yaw_to_rot(float(yaw)).T
            for x in xs:
                shifted_x = rotated[:, 0] + x
                x_inside = (shifted_x >= bounds_min[0] - 0.35) & (shifted_x <= bounds_max[0] + 0.35)
                if int(x_inside.sum()) < max(20, len(sample) // 8):
                    continue
                sub = rotated[x_inside]
                sx = shifted_x[x_inside]
                for y in ys:
                    pts = np.column_stack([sx, sub[:, 1] + y])
                    score, inlier_ratio = map_match_cost(pts, tree, bounds_min, bounds_max)
                    if score < best.score:
                        best = Sim2(float(scale), float(yaw), np.array([float(x), float(y)], dtype=np.float64), score, inlier_ratio)

    def objective(v: np.ndarray) -> float:
        scale = float(np.clip(v[0], args.scale_min, args.scale_max))
        sim = Sim2(scale, float(v[1]), np.array([float(v[2]), float(v[3])]), 0.0, 0.0)
        pts = apply_sim2(sample, sim)
        score, _ = map_match_cost(pts, tree, bounds_min, bounds_max)
        return score

    res = minimize(
        objective,
        np.array([best.scale, best.yaw, best.trans[0], best.trans[1]], dtype=np.float64),
        method="Nelder-Mead",
        options={"maxiter": 260, "xatol": 1e-4, "fatol": 1e-4},
    )
    refined = Sim2(float(np.clip(res.x[0], args.scale_min, args.scale_max)), normalize_angle(float(res.x[1])), np.array([float(res.x[2]), float(res.x[3])]), 0.0, 0.0)
    pts = apply_sim2(sample, refined)
    score, inlier_ratio = map_match_cost(pts, tree, bounds_min, bounds_max)
    refined.score = score
    refined.inlier_ratio = inlier_ratio
    return refined


def confidence_from(score: float, inliers: int, scan_count: int) -> float:
    if not math.isfinite(score) or scan_count <= 0:
        return 0.0
    inlier_ratio = min(1.0, inliers / max(1, min(scan_count, 900)))
    score_term = math.exp(-0.45 * max(0.0, score))
    return float(np.clip(0.62 * score_term + 0.38 * inlier_ratio, 0.0, 1.0))


def refine_scans(rows: Sequence[dict[str, object]], initial_poses: np.ndarray, sim: Sim2, ndt_grid: object, args: argparse.Namespace) -> tuple[np.ndarray, list[dict[str, object]]]:
    out = initial_poses.copy()
    diagnostics: list[dict[str, object]] = []
    lidar_dir = args.dataset_root / "velodyne"
    previous = None
    for idx, row in enumerate(rows):
        if idx % max(1, args.refine_step) != 0:
            diagnostics.append({"frame": int(row["frame"]), "score": float("nan"), "inliers": 0, "confidence": 0.0, "refined": False})
            continue
        frame = int(row["frame"])
        scan = load_scan(lidar_dir / f"{frame:06d}.bin", args) * sim.scale
        if len(scan) < 20:
            diagnostics.append({"frame": frame, "score": float("nan"), "inliers": 0, "confidence": 0.0, "refined": False})
            continue
        init = out[idx] if previous is None else 0.70 * out[idx] + 0.30 * previous
        init[2] = normalize_angle(float(init[2]))
        pose, score, inliers, iterations = ndt_align(scan, ndt_grid, init, args.ndt_iterations)
        conf = confidence_from(score, inliers, len(scan))
        if conf >= 0.30:
            out[idx] = pose
            previous = pose.copy()
            refined = True
        else:
            previous = out[idx].copy()
            refined = False
        diagnostics.append({"frame": frame, "score": score, "inliers": inliers, "confidence": conf, "iterations": iterations, "refined": refined})
    return out, diagnostics


def smooth_pose_graph(initial: np.ndarray, refined: np.ndarray, diagnostics: Sequence[dict[str, object]]) -> np.ndarray:
    n = len(refined)
    if n < 4:
        return refined
    local_delta = np.diff(initial, axis=0)
    local_delta[:, 2] = [normalize_angle(float(v)) for v in local_delta[:, 2]]
    conf = np.asarray([float(d.get("confidence", 0.0)) for d in diagnostics], dtype=np.float64)
    conf = np.clip(conf, 0.05, 1.0)
    y0 = refined.copy()
    y0[:, 2] = np.unwrap(y0[:, 2])

    def residual(v: np.ndarray) -> np.ndarray:
        p = v.reshape(n, 3)
        res: list[np.ndarray] = []
        data_weight = np.sqrt(conf)[:, None]
        data = (p - y0) * data_weight
        data[:, 2] = np.arctan2(np.sin(data[:, 2]), np.cos(data[:, 2]))
        res.append(data.reshape(-1))
        rel = np.diff(p, axis=0) - local_delta
        rel[:, 2] = np.arctan2(np.sin(rel[:, 2]), np.cos(rel[:, 2]))
        res.append((rel * np.array([2.2, 2.2, 1.4])).reshape(-1))
        accel = p[2:] - 2.0 * p[1:-1] + p[:-2]
        accel[:, 2] = np.arctan2(np.sin(accel[:, 2]), np.cos(accel[:, 2]))
        res.append((accel * np.array([0.45, 0.45, 0.25])).reshape(-1))
        return np.concatenate(res)

    solved = least_squares(residual, y0.reshape(-1), max_nfev=80, verbose=0).x.reshape(n, 3)
    solved[:, 2] = [normalize_angle(float(v)) for v in solved[:, 2]]
    return solved


def write_csv(path: Path, rows: Sequence[dict[str, object]], initial: np.ndarray, refined: np.ndarray, optimized: np.ndarray, diagnostics: Sequence[dict[str, object]]) -> None:
    fields = [
        "frame",
        "init_x",
        "init_y",
        "init_yaw",
        "ndt_x",
        "ndt_y",
        "ndt_yaw",
        "opt_x",
        "opt_y",
        "opt_yaw",
        "ndt_score",
        "ndt_inliers",
        "confidence",
        "refined",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row, ini, ref, opt, diag in zip(rows, initial, refined, optimized, diagnostics):
            writer.writerow(
                {
                    "frame": int(row["frame"]),
                    "init_x": ini[0],
                    "init_y": ini[1],
                    "init_yaw": ini[2],
                    "ndt_x": ref[0],
                    "ndt_y": ref[1],
                    "ndt_yaw": ref[2],
                    "opt_x": opt[0],
                    "opt_y": opt[1],
                    "opt_yaw": opt[2],
                    "ndt_score": diag["score"],
                    "ndt_inliers": diag["inliers"],
                    "confidence": diag["confidence"],
                    "refined": diag["refined"],
                }
            )


def render(path: Path, gltf_map: object, initial: np.ndarray, refined: np.ndarray, optimized: np.ndarray) -> None:
    if not HAS_MPL:
        return
    fig, ax = plt.subplots(figsize=(10, 12), dpi=220)
    draw_gltf_map(ax, gltf_map)
    ax.plot(initial[:, 0], initial[:, 1], color="#64748b", linewidth=1.3, alpha=0.55, label="Sim2 LiDAR odom init")
    ax.plot(refined[:, 0], refined[:, 1], color="#2563eb", linewidth=1.4, alpha=0.75, label="NDT refined")
    ax.plot(optimized[:, 0], optimized[:, 1], color="#dc2626", linewidth=2.1, label="optimized pose")
    ax.scatter([optimized[0, 0]], [optimized[0, 1]], c="#16a34a", s=34, zorder=8, label="start")
    ax.scatter([optimized[-1, 0]], [optimized[-1, 1]], c="#dc2626", s=34, zorder=8, label="end")
    ax.set_title("LiDAR odometry + NDT scan-to-map + pose optimization")
    ax.legend(loc="upper right", fontsize=7)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def run() -> None:
    args = parse_args()
    ensure_dir(args.output_dir)
    rows = load_odom_rows(args.odom_csv, args.num_frames)
    local_poses = rows_to_poses(rows)
    gltf_map = load_gltf_map(args.gltf, args.map_sample_step)
    ndt_grid = build_ndt_grid(gltf_map.reference_points, args.map_voxel)

    local_map = build_local_lidar_map(rows, local_poses, args)
    sim = estimate_sim2(local_map, gltf_map.reference_points, gltf_map.bounds_min, gltf_map.bounds_max, args)
    initial = poses_apply_sim2(local_poses, sim)
    refined, diagnostics = refine_scans(rows, initial, sim, ndt_grid, args)
    optimized = smooth_pose_graph(initial, refined, diagnostics) if args.pose_graph else refined

    csv_path = args.output_dir / "global_lidar_map_poses.csv"
    write_csv(csv_path, rows, initial, refined, optimized, diagnostics)
    render(args.output_dir / "trajectory_on_map.png", gltf_map, initial, refined, optimized)

    conf = np.asarray([float(d["confidence"]) for d in diagnostics], dtype=np.float64)
    refined_count = int(sum(bool(d["refined"]) for d in diagnostics))
    path_len = float(np.linalg.norm(np.diff(optimized[:, :2], axis=0), axis=1).sum()) if len(optimized) > 1 else 0.0
    summary = {
        "frames": len(rows),
        "odom_csv": str(args.odom_csv),
        "sim2": {
            "scale": sim.scale,
            "yaw": sim.yaw,
            "translation": sim.trans.tolist(),
            "coarse_refined_score": sim.score,
            "coarse_inlier_ratio": sim.inlier_ratio,
        },
        "ndt": {
            "refined_frames": refined_count,
            "mean_confidence": float(conf.mean()) if len(conf) else 0.0,
            "median_confidence": float(np.median(conf)) if len(conf) else 0.0,
            "frames_confidence_below_0_3": int((conf < 0.30).sum()) if len(conf) else 0,
        },
        "optimized_path_length_map_units": path_len,
        "note": "Uses LiDAR odometry and LiDAR scan-to-map matching only; does not read dataset pose ground truth.",
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    np.savez_compressed(args.output_dir / "debug_local_lidar_map.npz", local_map=local_map.astype(np.float32), initial=initial.astype(np.float32), refined=refined.astype(np.float32), optimized=optimized.astype(np.float32))

    print(f"[csv] {csv_path}")
    print(f"[summary] {args.output_dir / 'summary.json'}")
    print(f"[trajectory] {args.output_dir / 'trajectory_on_map.png'}")
    print(f"[sim2] scale={sim.scale:.6f} yaw={sim.yaw:.3f} trans=({sim.trans[0]:.3f},{sim.trans[1]:.3f}) score={sim.score:.3f}")
    print(f"[ndt] refined={refined_count}/{len(rows)} mean_conf={summary['ndt']['mean_confidence']:.3f}")


if __name__ == "__main__":
    run()
