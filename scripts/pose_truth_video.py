#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List

try:
    import cv2
    HAS_CV2 = True
except Exception:
    HAS_CV2 = False
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


DATASET_ROOT = Path("/home/ParkingAgent/dataset/dataset/dataset")
MATCH_FILE = Path("/home/ParkingAgent/dataset/dataset/matched_frames.txt")


@dataclass
class MatchedFrame:
    lidar_idx: int
    image_idx: int
    pose_idx: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pose-grounded dynamic trajectory video")
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--match-file", type=Path, default=MATCH_FILE)
    parser.add_argument("--start-index", type=int, default=1)
    parser.add_argument("--num-frames", type=int, default=2000)
    parser.add_argument("--step", type=int, default=10, help="render every Nth matched frame")
    parser.add_argument("--accumulate-every", type=int, default=20)
    parser.add_argument("--range-max", type=float, default=45.0)
    parser.add_argument("--z-min", type=float, default=-2.2)
    parser.add_argument("--z-max", type=float, default=2.0)
    parser.add_argument("--max-points-per-frame", type=int, default=12000)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--width", type=int, default=1400)
    parser.add_argument("--height", type=int, default=800)
    parser.add_argument("--frame-duration-ms", type=int, default=90)
    parser.add_argument("--output", type=Path, default=Path("outputs/pose_truth_video/trajectory_camera.gif"))
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
    corners = np.array([[half_l, half_w], [half_l, -half_w], [-half_l, -half_w], [-half_l, half_w]], dtype=np.float64)
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


def draw_car(ax: plt.Axes, x: float, y: float, yaw: float, color: str, label: str) -> None:
    body = car_footprint_xy(x, y, yaw)
    closed = np.vstack([body, body[0]])
    ax.fill(body[:, 0], body[:, 1], color=color, alpha=0.30, zorder=7)
    ax.plot(closed[:, 0], closed[:, 1], color=color, linewidth=1.8, zorder=8)
    heading = np.array([math.cos(yaw), math.sin(yaw)])
    ax.arrow(x, y, heading[0] * 2.8, heading[1] * 2.8, color=color, width=0.18, zorder=9)
    ax.annotate(
        label,
        xy=(x, y),
        xytext=(10, 10),
        textcoords="offset points",
        fontsize=8,
        bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": color, "alpha": 0.92},
        zorder=10,
    )


def fig_to_rgb(fig: plt.Figure) -> np.ndarray:
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    return rgba[:, :, :3].copy()


def main() -> None:
    args = parse_args()
    ensure_dir(args.output.parent)

    pose_rows = load_pose_rows(args.dataset_root / "pose" / "poses.txt")
    matched = load_matched_frames(args.match_file)
    selected_all = [f for f in matched if f.lidar_idx >= args.start_index]
    if args.num_frames > 0:
        selected_all = selected_all[: args.num_frames]
    render_frames = selected_all[:: max(1, args.step)]
    if not render_frames:
        raise RuntimeError("no frames selected")

    traj_all = []
    map_points = []
    lidar_dir = args.dataset_root / "velodyne"
    image_dir = args.dataset_root / "image"

    for i, frame in enumerate(selected_all):
        pose = pose_rows[frame.pose_idx]
        traj_all.append([pose[0, 3], pose[1, 3], yaw_from_pose(pose)])
        if i % max(1, args.accumulate_every) == 0:
            lidar_path = lidar_dir / f"{frame.lidar_idx:06d}.bin"
            if lidar_path.exists():
                pts = load_lidar_points(lidar_path, args.z_min, args.z_max, args.range_max, args.max_points_per_frame)
                world = transform_points(pose, pts)
                map_points.append(world[:, :2])

    traj_all = np.asarray(traj_all, dtype=np.float64)
    map_xy = np.vstack(map_points) if map_points else np.empty((0, 2), dtype=np.float64)

    if len(map_xy):
        min_xy = np.minimum(map_xy.min(axis=0), traj_all[:, :2].min(axis=0))
        max_xy = np.maximum(map_xy.max(axis=0), traj_all[:, :2].max(axis=0))
    else:
        min_xy = traj_all[:, :2].min(axis=0)
        max_xy = traj_all[:, :2].max(axis=0)
    span = max_xy - min_xy
    pad = max(4.0, float(max(span)) * 0.04)

    use_mp4 = args.output.suffix.lower() == ".mp4" and HAS_CV2
    gif_frames: List[Image.Image] = []

    if args.output.suffix.lower() == ".mp4" and not HAS_CV2:
        print("[warn] cv2 mp4 encoding unavailable on this machine, falling back to GIF output")
        args.output = args.output.with_suffix(".gif")

    writer = None
    if use_mp4:
        writer = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (args.width, args.height))
        if not writer.isOpened():
            raise RuntimeError(f"failed to open video writer for {args.output}")

    index_lookup = {frame.lidar_idx: i for i, frame in enumerate(selected_all)}

    for ordinal, frame in enumerate(render_frames, start=1):
        pose = pose_rows[frame.pose_idx]
        x, y, yaw = float(pose[0, 3]), float(pose[1, 3]), yaw_from_pose(pose)
        upto = index_lookup[frame.lidar_idx] + 1
        traj = traj_all[:upto]

        fig = plt.figure(figsize=(args.width / 100.0, args.height / 100.0), dpi=100)
        gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 1.0])
        ax_map = fig.add_subplot(gs[0, 0])
        ax_cam = fig.add_subplot(gs[0, 1])

        if len(map_xy):
            ax_map.hexbin(
                map_xy[:, 0],
                map_xy[:, 1],
                gridsize=240,
                mincnt=1,
                bins="log",
                cmap="Greys",
                linewidths=0,
                alpha=0.95,
                zorder=1,
            )
        ax_map.plot(traj[:, 0], traj[:, 1], color="#d97706", linewidth=2.3, zorder=4)
        ax_map.scatter([traj[0, 0]], [traj[0, 1]], c="#16a34a", s=40, zorder=5)
        ax_map.scatter([x], [y], c="#dc2626", s=50, zorder=6)
        draw_car(ax_map, x, y, yaw, "#dc2626", f"lidar {frame.lidar_idx:06d}\npose {frame.pose_idx:06d}\n({x:.1f}, {y:.1f})")
        ax_map.set_xlim(min_xy[0] - pad, max_xy[0] + pad)
        ax_map.set_ylim(min_xy[1] - pad, max_xy[1] + pad)
        ax_map.set_aspect("equal", adjustable="box")
        ax_map.grid(True, alpha=0.18)
        ax_map.set_title("Pose-grounded dynamic trajectory")
        ax_map.set_xlabel("pose x [m]")
        ax_map.set_ylabel("pose y [m]")

        image_path = image_dir / f"left{frame.image_idx:06d}.png"
        if image_path.exists():
            ax_cam.imshow(Image.open(image_path))
        else:
            ax_cam.text(0.5, 0.5, "image missing", ha="center", va="center", transform=ax_cam.transAxes)
        ax_cam.set_title(f"Camera left{frame.image_idx:06d}.png")
        ax_cam.axis("off")

        fig.suptitle(f"Vehicle trajectory + camera view | frame {ordinal}/{len(render_frames)}")
        fig.tight_layout()
        frame_rgb = fig_to_rgb(fig)
        plt.close(fig)
        if use_mp4 and writer is not None:
            writer.write(frame_rgb[:, :, ::-1])
        else:
            gif_frames.append(Image.fromarray(frame_rgb))

        if ordinal % 50 == 0 or ordinal == len(render_frames):
            print(f"[video] rendered {ordinal}/{len(render_frames)} frames")

    if use_mp4 and writer is not None:
        writer.release()
    else:
        if not gif_frames:
            raise RuntimeError("no gif frames rendered")
        ensure_dir(args.output.parent)
        gif_frames[0].save(
            args.output,
            save_all=True,
            append_images=gif_frames[1:],
            duration=args.frame_duration_ms,
            loop=0,
            optimize=False,
        )
    print(f"[done] wrote video to {args.output}")


if __name__ == "__main__":
    main()
