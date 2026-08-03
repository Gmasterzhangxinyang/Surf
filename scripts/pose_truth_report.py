#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image


DATASET_ROOT = Path("/home/ParkingAgent/dataset/dataset/dataset")
MATCH_FILE = Path("/home/ParkingAgent/dataset/dataset/matched_frames.txt")


@dataclass
class MatchedFrame:
    lidar_idx: int
    image_idx: int
    pose_idx: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pose-grounded trajectory report")
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--match-file", type=Path, default=MATCH_FILE)
    parser.add_argument("--start-index", type=int, default=1)
    parser.add_argument("--num-frames", type=int, default=120)
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--accumulate-every", type=int, default=2)
    parser.add_argument("--range-max", type=float, default=45.0)
    parser.add_argument("--z-min", type=float, default=-2.2)
    parser.add_argument("--z-max", type=float, default=2.0)
    parser.add_argument("--max-points-per-frame", type=int, default=18000)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/pose_truth_report"))
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def yaw_from_pose(mat: np.ndarray) -> float:
    return math.atan2(float(mat[1, 0]), float(mat[0, 0]))


def yaw_to_rot(yaw: float) -> np.ndarray:
    c = math.cos(yaw)
    s = math.sin(yaw)
    return np.array([[c, -s], [s, c]], dtype=np.float64)


def car_footprint_xy(x: float, y: float, yaw: float, length: float = 4.6, width: float = 1.9) -> np.ndarray:
    half_l = length * 0.5
    half_w = width * 0.5
    corners = np.array(
        [[half_l, half_w], [half_l, -half_w], [-half_l, -half_w], [-half_l, half_w]],
        dtype=np.float64,
    )
    return corners @ yaw_to_rot(yaw).T + np.array([x, y], dtype=np.float64)


def load_pose_rows(path: Path) -> np.ndarray:
    poses = np.loadtxt(path, dtype=np.float64)
    if poses.ndim == 1:
        poses = poses.reshape(1, -1)
    mats = np.tile(np.eye(4, dtype=np.float64), (poses.shape[0], 1, 1))
    mats[:, :3, :4] = poses.reshape(-1, 3, 4)
    return mats


def load_matched_frames(path: Path) -> List[MatchedFrame]:
    frames: List[MatchedFrame] = []
    with path.open("r") as handle:
        for line in handle:
            if not line.strip() or line.startswith("#"):
                continue
            fields = [int(x) for x in line.split()]
            if len(fields) < 19:
                continue
            lidar_idx = fields[0]
            if lidar_idx < 0:
                continue
            image_idx = next((idx for idx in fields[1:4] if idx >= 0), -1)
            pose_idx = next((idx for idx in fields[14:19] if idx >= 0), -1)
            if image_idx < 0 or pose_idx < 0:
                continue
            frames.append(MatchedFrame(lidar_idx=lidar_idx, image_idx=image_idx, pose_idx=pose_idx))
    return frames


def load_lidar_points(path: Path, z_min: float, z_max: float, range_max: float, max_points: int) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.float32)
    pts = raw.reshape(-1, 4)[:, :3].astype(np.float64)
    mask = np.isfinite(pts).all(axis=1)
    mask &= pts[:, 2] >= z_min
    mask &= pts[:, 2] <= z_max
    mask &= np.linalg.norm(pts[:, :2], axis=1) <= range_max
    pts = pts[mask]
    if len(pts) > max_points:
        stride = int(math.ceil(len(pts) / max_points))
        pts = pts[::stride]
    return pts


def transform_points(mat: np.ndarray, points_xyz: np.ndarray) -> np.ndarray:
    return (mat[:3, :3] @ points_xyz.T).T + mat[:3, 3]


def export_image(src: Path, dst: Path) -> None:
    ensure_dir(dst.parent)
    Image.open(src).save(dst)


def render_start_end_photos(output_path: Path, start_photo: Path, end_photo: Path, start_frame: int, end_frame: int) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=180)
    pairs = [(axes[0], start_photo, f"Start frame {start_frame:06d}"), (axes[1], end_photo, f"End frame {end_frame:06d}")]
    for ax, path, title in pairs:
        ax.imshow(Image.open(path))
        ax.set_title(title)
        ax.axis("off")
    fig.tight_layout()
    ensure_dir(output_path.parent)
    fig.savefig(output_path)
    plt.close(fig)


def draw_car(ax: plt.Axes, x: float, y: float, yaw: float, label: str, color: str) -> None:
    body = car_footprint_xy(x, y, yaw)
    closed = np.vstack([body, body[0]])
    ax.fill(body[:, 0], body[:, 1], color=color, alpha=0.28, zorder=6)
    ax.plot(closed[:, 0], closed[:, 1], color=color, linewidth=1.8, zorder=7)
    heading = np.array([math.cos(yaw), math.sin(yaw)])
    ax.arrow(x, y, heading[0] * 2.6, heading[1] * 2.6, color=color, width=0.18, zorder=8)
    ax.annotate(
        f"{label}\n({x:.2f}, {y:.2f}) yaw={yaw:.2f}",
        xy=(x, y),
        xytext=(12, 12),
        textcoords="offset points",
        fontsize=8,
        bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": color, "alpha": 0.92},
        arrowprops={"arrowstyle": "->", "color": color, "lw": 1.0},
        zorder=9,
    )


def render_pose_map(output_path: Path, map_points_xy: np.ndarray, traj_xyyaw: np.ndarray, title: str) -> None:
    fig, ax = plt.subplots(figsize=(12, 12), dpi=220)
    if len(map_points_xy):
        hb = ax.hexbin(
            map_points_xy[:, 0],
            map_points_xy[:, 1],
            gridsize=280,
            mincnt=1,
            bins="log",
            cmap="Greys",
            linewidths=0,
            alpha=0.95,
            zorder=1,
        )
        fig.colorbar(hb, ax=ax, fraction=0.04, pad=0.02, label="LiDAR density")
    ax.plot(traj_xyyaw[:, 0], traj_xyyaw[:, 1], color="#d97706", linewidth=2.2, zorder=4, label="pose trajectory")
    ax.scatter([traj_xyyaw[0, 0]], [traj_xyyaw[0, 1]], c="#16a34a", s=40, zorder=5, label="start")
    ax.scatter([traj_xyyaw[-1, 0]], [traj_xyyaw[-1, 1]], c="#dc2626", s=48, zorder=5, label="end")
    draw_car(ax, float(traj_xyyaw[0, 0]), float(traj_xyyaw[0, 1]), float(traj_xyyaw[0, 2]), "START", "#16a34a")
    draw_car(ax, float(traj_xyyaw[-1, 0]), float(traj_xyyaw[-1, 1]), float(traj_xyyaw[-1, 2]), "END", "#dc2626")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.18)
    ax.legend(loc="best", framealpha=0.92)
    ax.set_xlabel("pose x [m]")
    ax.set_ylabel("pose y [m]")
    ax.set_title(title)
    fig.tight_layout()
    ensure_dir(output_path.parent)
    fig.savefig(output_path)
    plt.close(fig)


def write_csv(path: Path, frames: List[MatchedFrame], traj_xyyaw: np.ndarray) -> None:
    ensure_dir(path.parent)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["lidar_idx", "image_idx", "pose_idx", "x", "y", "yaw"])
        writer.writeheader()
        for frame, pose in zip(frames, traj_xyyaw):
            writer.writerow(
                {
                    "lidar_idx": frame.lidar_idx,
                    "image_idx": frame.image_idx,
                    "pose_idx": frame.pose_idx,
                    "x": float(pose[0]),
                    "y": float(pose[1]),
                    "yaw": float(pose[2]),
                }
            )


def main() -> None:
    args = parse_args()
    ensure_dir(args.output_dir)

    pose_rows = load_pose_rows(args.dataset_root / "pose" / "poses.txt")
    matched = load_matched_frames(args.match_file)
    selected = [f for f in matched if f.lidar_idx >= args.start_index][:: max(1, args.step)]
    selected = selected[: args.num_frames]
    if not selected:
        raise RuntimeError("no matched frames selected")

    traj = []
    accum = []
    lidar_dir = args.dataset_root / "velodyne"
    image_dir = args.dataset_root / "image"

    for i, frame in enumerate(selected):
        pose = pose_rows[frame.pose_idx]
        yaw = yaw_from_pose(pose)
        traj.append([pose[0, 3], pose[1, 3], yaw])
        if i % max(1, args.accumulate_every) == 0:
            lidar_path = lidar_dir / f"{frame.lidar_idx:06d}.bin"
            if lidar_path.exists():
                pts = load_lidar_points(lidar_path, args.z_min, args.z_max, args.range_max, args.max_points_per_frame)
                world = transform_points(pose, pts)
                accum.append(world[:, :2])

    traj_xyyaw = np.asarray(traj, dtype=np.float64)
    map_points_xy = np.vstack(accum) if accum else np.empty((0, 2), dtype=np.float64)

    start = selected[0]
    end = selected[-1]
    start_src = image_dir / f"left{start.image_idx:06d}.png"
    end_src = image_dir / f"left{end.image_idx:06d}.png"
    start_dst = args.output_dir / "start_photo.png"
    end_dst = args.output_dir / "end_photo.png"
    export_image(start_src, start_dst)
    export_image(end_src, end_dst)

    render_pose_map(args.output_dir / "trajectory_truth.png", map_points_xy, traj_xyyaw, "Pose-grounded vehicle trajectory")
    render_pose_map(args.output_dir / "current_pose_map.png", map_points_xy, traj_xyyaw[-1:, :], "Current vehicle pose")
    render_start_end_photos(args.output_dir / "start_end_photos.png", start_dst, end_dst, start.lidar_idx, end.lidar_idx)
    write_csv(args.output_dir / "pose_trajectory.csv", selected, traj_xyyaw)

    print(f"[done] wrote {len(selected)} pose frames to {args.output_dir}")
    print(f"[start] lidar={start.lidar_idx:06d} image=left{start.image_idx:06d}.png")
    print(f"[end] lidar={end.lidar_idx:06d} image=left{end.image_idx:06d}.png")


if __name__ == "__main__":
    main()
