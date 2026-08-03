#!/usr/bin/env python3
"""Track LiDAR trajectory on GLTF map using LiDAR odom relative motion + NDT.

This fixes the failure mode of applying one global Sim(2) to the entire
LiDAR-only trajectory. LiDAR-only odometry drifts over long sequences, so this
script only uses its relative motion between keyframes, then corrects each
keyframe by scan-to-map NDT in the map frame.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Sequence

import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_MPL = True
except Exception:
    HAS_MPL = False

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gltf_lidar_ndt import build_ndt_grid, coarse_search, draw_gltf_map, load_gltf_map, ndt_align, normalize_angle, yaw_to_rot  # noqa: E402
from lidar_map_ndt_optimized import load_scan  # noqa: E402


DATASET_ROOT = Path("/home/ParkingAgent/dataset/dataset/dataset")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Track full LiDAR trajectory on GLTF map")
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--gltf", type=Path, default=Path("file.gltf"))
    parser.add_argument("--odom-csv", type=Path, default=Path("outputs/lidar_only_odometry_full/lidar_only_poses.csv"))
    parser.add_argument("--seed-summary", type=Path, default=Path("outputs/lidar_map_ndt_full_keyframes/summary.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/lidar_map_tracking_full"))
    parser.add_argument("--keyframe-step", type=int, default=20)
    parser.add_argument("--map-sample-step", type=float, default=0.06)
    parser.add_argument("--map-voxel", type=float, default=0.25)
    parser.add_argument("--scan-voxel", type=float, default=0.18)
    parser.add_argument("--range-min", type=float, default=2.0)
    parser.add_argument("--range-max", type=float, default=28.0)
    parser.add_argument("--z-min", type=float, default=-0.35)
    parser.add_argument("--z-max", type=float, default=1.55)
    parser.add_argument("--max-scan-points", type=int, default=2800)
    parser.add_argument("--ndt-iterations", type=int, default=18)
    parser.add_argument("--accept-confidence", type=float, default=0.30)
    parser.add_argument("--max-pred-step", type=float, default=0.55)
    parser.add_argument("--relocalize-xy-step", type=float, default=0.45)
    parser.add_argument("--relocalize-yaw-step-deg", type=float, default=10.0)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def confidence_from(score: float, inliers: int, scan_count: int) -> float:
    if not math.isfinite(score) or scan_count <= 0:
        return 0.0
    inlier_ratio = min(1.0, inliers / max(1, min(scan_count, 900)))
    score_term = math.exp(-0.45 * max(0.0, score))
    return float(np.clip(0.62 * score_term + 0.38 * inlier_ratio, 0.0, 1.0))


def load_rows(path: Path) -> list[dict[str, str]]:
    rows = list(csv.DictReader(path.open(newline="")))
    if not rows:
        raise ValueError(f"empty odom csv: {path}")
    return rows


def rows_to_local(rows: Sequence[dict[str, str]]) -> np.ndarray:
    return np.asarray([[float(r["x"]), float(r["y"]), float(r["yaw_rad"])] for r in rows], dtype=np.float64)


def local_relative(local_a: np.ndarray, local_b: np.ndarray) -> tuple[np.ndarray, float]:
    rot_inv = yaw_to_rot(-float(local_a[2]))
    d_body = rot_inv @ (local_b[:2] - local_a[:2])
    dyaw = normalize_angle(float(local_b[2] - local_a[2]))
    return d_body, dyaw


def apply_relative(map_pose: np.ndarray, d_body: np.ndarray, dyaw: float, scale: float) -> np.ndarray:
    out = map_pose.copy()
    out[:2] = map_pose[:2] + scale * (yaw_to_rot(float(map_pose[2])) @ d_body)
    out[2] = normalize_angle(float(map_pose[2] + dyaw))
    return out


def seed_pose(local0: np.ndarray, sim: dict) -> np.ndarray:
    scale = float(sim["scale"])
    yaw = float(sim["yaw"])
    trans = np.asarray(sim["translation"], dtype=np.float64)
    xy = scale * (local0[:2] @ yaw_to_rot(yaw).T) + trans
    return np.array([xy[0], xy[1], normalize_angle(float(local0[2] + yaw))], dtype=np.float64)


def interpolate_full(local: np.ndarray, key_indices: Sequence[int], key_poses: np.ndarray, scale: float) -> np.ndarray:
    full = np.empty_like(local)
    for seg, start_idx in enumerate(key_indices):
        end_idx = key_indices[seg + 1] if seg + 1 < len(key_indices) else len(local) - 1
        base_local = local[start_idx]
        base_map = key_poses[seg]
        for idx in range(start_idx, end_idx + 1):
            d_body, dyaw = local_relative(base_local, local[idx])
            full[idx] = apply_relative(base_map, d_body, dyaw, scale)
    return full


def render(path: Path, gltf_map: object, full: np.ndarray, key_poses: np.ndarray) -> None:
    if not HAS_MPL:
        return
    fig, ax = plt.subplots(figsize=(10, 12), dpi=220)
    draw_gltf_map(ax, gltf_map)
    ax.plot(full[:, 0], full[:, 1], color="#dc2626", linewidth=1.8, label="tracked LiDAR trajectory")
    ax.scatter(key_poses[:, 0], key_poses[:, 1], s=5, c="#2563eb", alpha=0.8, linewidths=0, label="NDT keyframes")
    ax.scatter([full[0, 0]], [full[0, 1]], s=34, c="#16a34a", zorder=8, label="start")
    ax.scatter([full[-1, 0]], [full[-1, 1]], s=34, c="#dc2626", zorder=8, label="end")
    ax.set_title("Full LiDAR trajectory tracked on GLTF map")
    ax.legend(loc="upper right", fontsize=7)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_rows(args.odom_csv)
    local = rows_to_local(rows)
    seed = json.loads(args.seed_summary.read_text())["sim2"]
    scale = float(seed["scale"])

    gltf_map = load_gltf_map(args.gltf, args.map_sample_step)
    ndt_grid = build_ndt_grid(gltf_map.reference_points, args.map_voxel)
    key_indices = list(range(0, len(rows), max(1, args.keyframe_step)))
    if key_indices[-1] != len(rows) - 1:
        key_indices.append(len(rows) - 1)

    key_poses: list[np.ndarray] = []
    diagnostics: list[dict[str, object]] = []
    last_key_idx = key_indices[0]
    last_pose = seed_pose(local[last_key_idx], seed)
    lidar_dir = args.dataset_root / "velodyne"

    for seq, idx in enumerate(key_indices):
        frame = int(rows[idx]["frame"])
        if seq == 0:
            pred = last_pose.copy()
        else:
            d_body, dyaw = local_relative(local[last_key_idx], local[idx])
            pred = apply_relative(last_pose, d_body, dyaw, scale)

        scan = load_scan(lidar_dir / f"{frame:06d}.bin", args) * scale
        pose, score, inliers, iterations = ndt_align(scan, ndt_grid, pred, args.ndt_iterations)
        conf = confidence_from(score, inliers, len(scan))
        init_source = "prediction"
        if conf < args.accept_confidence:
            relocal_init, _, _ = coarse_search(
                scan,
                gltf_map.reference_points,
                gltf_map.bounds_min,
                gltf_map.bounds_max,
                args.relocalize_xy_step,
                args.relocalize_yaw_step_deg,
                0.5,
            )
            relocal_pose, relocal_score, relocal_inliers, relocal_iterations = ndt_align(scan, ndt_grid, relocal_init, args.ndt_iterations)
            relocal_conf = confidence_from(relocal_score, relocal_inliers, len(scan))
            if relocal_conf > conf:
                pose, score, inliers, iterations, conf = relocal_pose, relocal_score, relocal_inliers, relocal_iterations, relocal_conf
                init_source = "relocalize"
        if conf >= args.accept_confidence:
            accepted = True
            cur_pose = pose
        else:
            accepted = False
            cur_pose = pred

        step = float(np.linalg.norm(cur_pose[:2] - last_pose[:2])) if seq else 0.0
        if seq and step > args.max_pred_step:
            accepted = False
            cur_pose = pred

        key_poses.append(cur_pose.copy())
        diagnostics.append(
            {
                "seq": seq,
                "frame": frame,
                "pred_x": pred[0],
                "pred_y": pred[1],
                "pred_yaw": pred[2],
                "x": cur_pose[0],
                "y": cur_pose[1],
                "yaw": cur_pose[2],
                "score": score,
                "inliers": inliers,
                "confidence": conf,
                "iterations": iterations,
                "accepted": accepted,
                "init_source": init_source,
                "step": step,
            }
        )
        last_key_idx = idx
        last_pose = cur_pose.copy()
        if not args.quiet:
            print(f"key={seq:04d} frame={frame:06d} x={cur_pose[0]:.3f} y={cur_pose[1]:.3f} conf={conf:.3f} accepted={accepted}")

    key_arr = np.asarray(key_poses, dtype=np.float64)
    full = interpolate_full(local, key_indices, key_arr, scale)

    csv_path = args.output_dir / "tracked_global_poses.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["frame", "x", "y", "yaw"])
        writer.writeheader()
        for row, pose in zip(rows, full):
            writer.writerow({"frame": int(row["frame"]), "x": pose[0], "y": pose[1], "yaw": pose[2]})

    key_csv = args.output_dir / "tracked_keyframes.csv"
    with key_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(diagnostics[0].keys()))
        writer.writeheader()
        writer.writerows(diagnostics)

    render(args.output_dir / "tracked_trajectory_on_map.png", gltf_map, full, key_arr)
    conf = np.asarray([float(d["confidence"]) for d in diagnostics], dtype=np.float64)
    accepted = np.asarray([bool(d["accepted"]) for d in diagnostics], dtype=bool)
    inside = (
        (full[:, 0] >= gltf_map.bounds_min[0] - 0.5)
        & (full[:, 0] <= gltf_map.bounds_max[0] + 0.5)
        & (full[:, 1] >= gltf_map.bounds_min[1] - 0.5)
        & (full[:, 1] <= gltf_map.bounds_max[1] + 0.5)
    )
    summary = {
        "frames": len(rows),
        "keyframes": len(key_indices),
        "scale": scale,
        "seed_summary": str(args.seed_summary),
        "accepted_keyframes": int(accepted.sum()),
        "accepted_keyframe_ratio": float(accepted.mean()),
        "keyframe_confidence_mean": float(conf.mean()),
        "keyframe_confidence_median": float(np.median(conf)),
        "bounds_min": full[:, :2].min(axis=0).tolist(),
        "bounds_max": full[:, :2].max(axis=0).tolist(),
        "inside_map_plus_margin_ratio": float(inside.mean()),
        "path_length_map_units": float(np.linalg.norm(np.diff(full[:, :2], axis=0), axis=1).sum()),
        "note": "Tracking result uses LiDAR odom relative motion plus NDT scan-to-map keyframes; no dataset pose ground truth.",
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[csv] {csv_path}")
    print(f"[keyframes] {key_csv}")
    print(f"[summary] {args.output_dir / 'summary.json'}")
    print(f"[trajectory] {args.output_dir / 'tracked_trajectory_on_map.png'}")
    print(f"[quality] accepted={accepted.sum()}/{len(accepted)} inside={summary['inside_map_plus_margin_ratio']:.3f} conf_mean={summary['keyframe_confidence_mean']:.3f}")


if __name__ == "__main__":
    main()
