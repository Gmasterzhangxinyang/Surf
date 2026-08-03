#!/usr/bin/env python3
"""Build a frame-level dataset: image + LiDAR + aligned map pose + map points."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from parking_pose_correction.dataset import load_timestamp_file  # noqa: E402
from parking_pose_correction.timestamps import nearest_timestamp_match  # noqa: E402


DATASET_ROOT = Path("/home/ParkingAgent/dataset/dataset/dataset")
DEFAULT_ALIGNED_TRAJECTORY = Path("outputs/aligned_trajectory_final.csv")
DEFAULT_TRANSFORM = Path("outputs/alignment_transform.json")
DEFAULT_POSE_SOURCE = Path("outputs/azimuth_time_odometry_compatible.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create per-frame image/lidar/map-points dataset")
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--aligned-trajectory", type=Path, default=DEFAULT_ALIGNED_TRAJECTORY)
    parser.add_argument("--transform", type=Path, default=DEFAULT_TRANSFORM)
    parser.add_argument("--pose-source", type=Path, default=DEFAULT_POSE_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/frame_map_dataset"))
    parser.add_argument("--frame-step", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--start-frame", type=int, default=None)
    parser.add_argument("--end-frame", type=int, default=None)
    parser.add_argument("--range-m", type=float, default=0.0, help="0 keeps all finite LiDAR points")
    parser.add_argument("--z-min", type=float, default=None)
    parser.add_argument("--z-max", type=float, default=None)
    parser.add_argument("--compress", action="store_true", help="Use compressed npz; smaller but slower")
    parser.add_argument("--max-camera-delta-sec", type=float, default=0.04)
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def normalize_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def yaw_to_rot(yaw: float) -> np.ndarray:
    c = math.cos(yaw)
    s = math.sin(yaw)
    return np.array([[c, -s], [s, c]], dtype=np.float64)


def load_aligned_trajectory(path: Path) -> list[dict[str, float | int]]:
    rows: list[dict[str, float | int]] = []
    with path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(
                {
                    "frame": int(row["frame"]),
                    "map_x": float(row["x"]),
                    "map_y": float(row["y"]),
                    "map_yaw": normalize_angle(float(row["yaw"])),
                }
            )
    return rows


def load_pose_source(path: Path) -> dict[int, dict[str, str]]:
    if not path.exists():
        return {}
    with path.open("r", newline="") as handle:
        return {int(row["frame"]): row for row in csv.DictReader(handle)}


def select_rows(rows: list[dict[str, float | int]], args: argparse.Namespace) -> list[dict[str, float | int]]:
    selected = []
    for row in rows:
        frame = int(row["frame"])
        if args.start_frame is not None and frame < args.start_frame:
            continue
        if args.end_frame is not None and frame > args.end_frame:
            continue
        selected.append(row)
    selected = selected[:: max(1, args.frame_step)]
    if args.max_frames > 0:
        selected = selected[: args.max_frames]
    return selected


def load_lidar_xyzi(path: Path, args: argparse.Namespace) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.float32)
    if raw.size % 4 != 0:
        raise ValueError(f"invalid LiDAR bin shape: {path}")
    pts = raw.reshape(-1, 4)
    mask = np.isfinite(pts).all(axis=1)
    if args.range_m and args.range_m > 0:
        mask &= np.linalg.norm(pts[:, :2], axis=1) <= args.range_m
    if args.z_min is not None:
        mask &= pts[:, 2] >= args.z_min
    if args.z_max is not None:
        mask &= pts[:, 2] <= args.z_max
    return pts[mask]


def lidar_to_map_xyzi(raw_xyzi: np.ndarray, map_x: float, map_y: float, map_yaw: float, map_scale: float) -> np.ndarray:
    out = np.empty_like(raw_xyzi, dtype=np.float32)
    map_xy = np.array([map_x, map_y], dtype=np.float64) + map_scale * (raw_xyzi[:, :2].astype(np.float64) @ yaw_to_rot(map_yaw).T)
    out[:, 0:2] = map_xy.astype(np.float32)
    out[:, 2] = raw_xyzi[:, 2]
    out[:, 3] = raw_xyzi[:, 3]
    return out


def write_npz(path: Path, compress: bool, **arrays: np.ndarray) -> None:
    if compress:
        np.savez_compressed(path, **arrays)
    else:
        np.savez(path, **arrays)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    points_dir = output_dir / "map_points"
    ensure_dir(points_dir)

    transform = json.loads(args.transform.read_text(encoding="utf-8"))
    map_scale = float(transform["scale"])
    pose_source = load_pose_source(args.pose_source)
    trajectory = select_rows(load_aligned_trajectory(args.aligned_trajectory), args)
    camera_frames, camera_timestamps = load_timestamp_file(args.dataset_root / "image" / "timestamps.txt")

    manifest_path = output_dir / "frames.csv"
    metadata_path = output_dir / "metadata.json"
    fields = [
        "frame",
        "image_path",
        "camera_frame",
        "camera_image_path",
        "camera_timestamp",
        "camera_lidar_dt_sec",
        "camera_match_valid",
        "lidar_path",
        "map_points_path",
        "map_x",
        "map_y",
        "map_yaw",
        "lidar_timestamp",
        "odom_timestamp",
        "time_delta_sec",
        "num_points",
        "missing_image",
        "missing_lidar",
    ]

    processed = 0
    missing_lidar = 0
    missing_image = 0
    total_points = 0
    camera_delta_abs: list[float] = []
    with manifest_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for idx, row in enumerate(trajectory):
            frame = int(row["frame"])
            map_x = float(row["map_x"])
            map_y = float(row["map_y"])
            map_yaw = float(row["map_yaw"])
            source = pose_source.get(frame, {})
            if not source.get("lidar_timestamp"):
                raise ValueError(f"frame {frame} has no lidar_timestamp in {args.pose_source}")
            lidar_timestamp = float(source["lidar_timestamp"])
            camera_match = nearest_timestamp_match(camera_frames, camera_timestamps, lidar_timestamp)
            camera_delta_abs.append(abs(camera_match.delta_sec))
            image_path = args.dataset_root / "image" / f"left{camera_match.frame:06d}.png"
            lidar_path = args.dataset_root / "velodyne" / f"{frame:06d}.bin"
            map_points_path = points_dir / f"{frame:06d}.npz"
            image_missing = not image_path.exists()
            lidar_missing = not lidar_path.exists()
            if image_missing:
                missing_image += 1
            point_count = 0
            if lidar_missing:
                missing_lidar += 1
            else:
                raw_xyzi = load_lidar_xyzi(lidar_path, args)
                map_xyzi = lidar_to_map_xyzi(raw_xyzi, map_x, map_y, map_yaw, map_scale)
                pose = np.array([map_x, map_y, map_yaw], dtype=np.float32)
                write_npz(
                    map_points_path,
                    args.compress,
                    points_map_xyzi=map_xyzi,
                    ego_map_pose=pose,
                    frame=np.array([frame], dtype=np.int32),
                    map_scale=np.array([map_scale], dtype=np.float32),
                )
                point_count = int(map_xyzi.shape[0])
                total_points += point_count
                processed += 1

            writer.writerow(
                {
                    "frame": frame,
                    "image_path": str(image_path),
                    "camera_frame": camera_match.frame,
                    "camera_image_path": str(image_path),
                    "camera_timestamp": camera_match.timestamp,
                    "camera_lidar_dt_sec": camera_match.delta_sec,
                    "camera_match_valid": int(abs(camera_match.delta_sec) <= args.max_camera_delta_sec),
                    "lidar_path": str(lidar_path),
                    "map_points_path": str(map_points_path) if not lidar_missing else "",
                    "map_x": map_x,
                    "map_y": map_y,
                    "map_yaw": map_yaw,
                    "lidar_timestamp": source.get("lidar_timestamp", ""),
                    "odom_timestamp": source.get("odom_timestamp", ""),
                    "time_delta_sec": source.get("time_delta_sec", ""),
                    "num_points": point_count,
                    "missing_image": int(image_missing),
                    "missing_lidar": int(lidar_missing),
                }
            )
            if (idx + 1) % 250 == 0:
                print(f"[progress] {idx + 1}/{len(trajectory)} rows, processed={processed}, points={total_points:,}", flush=True)

    metadata = {
        "description": "Each row binds one LiDAR frame to its image, aligned map pose, and LiDAR points projected into map coordinates.",
        "dataset_root": str(args.dataset_root),
        "aligned_trajectory": str(args.aligned_trajectory),
        "transform": transform,
        "pose_source": str(args.pose_source),
        "frame_step": args.frame_step,
        "selected_rows": len(trajectory),
        "processed_lidar_frames": processed,
        "missing_lidar": missing_lidar,
        "missing_image": missing_image,
        "total_projected_points": total_points,
        "camera_sync": {
            "method": "nearest_timestamp",
            "valid_count": int(sum(delta <= args.max_camera_delta_sec for delta in camera_delta_abs)),
            "abs_median_sec": float(np.median(camera_delta_abs)),
            "abs_p95_sec": float(np.quantile(camera_delta_abs, 0.95)),
            "abs_max_sec": float(max(camera_delta_abs)),
        },
        "map_points_format": {
            "file": "npz",
            "points_map_xyzi": "[N,4] float32: x_map, y_map, z_lidar_m, intensity",
            "ego_map_pose": "[3] float32: x_map, y_map, yaw_rad",
            "frame": "[1] int32",
            "map_scale": "[1] float32",
        },
        "outputs": {
            "frames_csv": str(manifest_path),
            "map_points_dir": str(points_dir),
            "metadata_json": str(metadata_path),
        },
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
