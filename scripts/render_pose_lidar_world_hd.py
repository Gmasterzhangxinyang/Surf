#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = Path("/home/ParkingAgent/dataset/dataset/dataset")
MATCH_FILE = Path("/home/ParkingAgent/dataset/dataset/matched_frames.txt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render ultra-HD dataset/world pose and LiDAR trajectory without ICpark map.")
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--match-file", type=Path, default=MATCH_FILE)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "pose_lidar_world_hd")
    parser.add_argument("--size", type=int, default=8000)
    parser.add_argument("--lidar-step", type=int, default=5)
    parser.add_argument("--max-points-per-frame", type=int, default=1800)
    parser.add_argument("--range-m", type=float, default=38.0)
    parser.add_argument("--z-min", type=float, default=-1.2)
    parser.add_argument("--z-max", type=float, default=2.5)
    return parser.parse_args()


def load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def load_poses(path: Path) -> np.ndarray:
    rows = np.loadtxt(path, dtype=np.float64)
    if rows.ndim == 1:
        rows = rows.reshape(1, -1)
    mats3 = rows.reshape(-1, 3, 4)
    mats = np.repeat(np.eye(4, dtype=np.float64)[None, :, :], len(mats3), axis=0)
    mats[:, :3, :4] = mats3
    return mats


def pose_to_xyyaw(mats: np.ndarray) -> np.ndarray:
    out = np.zeros((len(mats), 3), dtype=np.float64)
    out[:, 0] = mats[:, 0, 3]
    out[:, 1] = mats[:, 1, 3]
    out[:, 2] = np.arctan2(mats[:, 1, 0], mats[:, 0, 0])
    return out


def load_matches(path: Path) -> dict[int, int]:
    matches: dict[int, int] = {}
    if not path.exists():
        return matches
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if not line.strip() or line.startswith("#"):
                continue
            fields = [int(x) for x in line.split()]
            if len(fields) < 19:
                continue
            lidar_idx = fields[0]
            pose_candidates = [idx for idx in fields[14:19] if idx >= 0]
            if pose_candidates:
                matches[lidar_idx] = pose_candidates[0]
    return matches


def load_lidar(path: Path) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.float32)
    if raw.size % 4 != 0:
        return np.empty((0, 4), dtype=np.float32)
    return raw.reshape(-1, 4)


def transform_points(mat: np.ndarray, xyz: np.ndarray) -> np.ndarray:
    pts = np.column_stack([xyz, np.ones(len(xyz), dtype=np.float64)])
    return (mat @ pts.T).T[:, :3]


def square_bounds(points_xy: np.ndarray, pad: float) -> tuple[np.ndarray, np.ndarray]:
    mn = points_xy.min(axis=0) - pad
    mx = points_xy.max(axis=0) + pad
    center = (mn + mx) * 0.5
    half = float((mx - mn).max()) * 0.5
    return center - half, center + half


def to_px(points: np.ndarray, mn: np.ndarray, mx: np.ndarray, size: int, margin: int = 220) -> np.ndarray:
    usable = size - 2 * margin
    x = margin + (points[:, 0] - mn[0]) / (mx[0] - mn[0]) * usable
    y = size - margin - (points[:, 1] - mn[1]) / (mx[1] - mn[1]) * usable
    return np.column_stack([x, y])


def draw_grid(draw: ImageDraw.ImageDraw, mn: np.ndarray, mx: np.ndarray, size: int, step: float) -> None:
    font = load_font(34)
    x = math.floor(mn[0] / step) * step
    while x <= mx[0]:
        pts = to_px(np.array([[x, mn[1]], [x, mx[1]]], dtype=np.float64), mn, mx, size)
        width = 3 if abs((x / (step * 5)) - round(x / (step * 5))) < 1e-6 else 1
        draw.line([tuple(pts[0]), tuple(pts[1])], fill=(180, 190, 205, 70), width=width)
        if int(round(x / step)) % 5 == 0:
            draw.text((pts[0, 0] + 8, size - 175), f"{x:.0f}", fill=(71, 85, 105, 180), font=font)
        x += step
    y = math.floor(mn[1] / step) * step
    while y <= mx[1]:
        pts = to_px(np.array([[mn[0], y], [mx[0], y]], dtype=np.float64), mn, mx, size)
        width = 3 if abs((y / (step * 5)) - round(y / (step * 5))) < 1e-6 else 1
        draw.line([tuple(pts[0]), tuple(pts[1])], fill=(180, 190, 205, 70), width=width)
        if int(round(y / step)) % 5 == 0:
            draw.text((62, pts[0, 1] - 18), f"{y:.0f}", fill=(71, 85, 105, 180), font=font)
        y += step


def draw_title(draw: ImageDraw.ImageDraw, size: int, title: str, subtitle: str) -> None:
    title_font = load_font(76, bold=True)
    sub_font = load_font(42)
    draw.rounded_rectangle((50, 50, size - 50, 190), radius=24, fill=(255, 255, 255, 238), outline=(180, 190, 205, 255), width=4)
    draw.text((86, 68), title, fill=(15, 23, 42, 255), font=title_font)
    draw.text((88, 145), subtitle, fill=(71, 85, 105, 255), font=sub_font)


def draw_trajectory(draw: ImageDraw.ImageDraw, xy_yaw: np.ndarray, mn: np.ndarray, mx: np.ndarray, size: int, width: int = 13) -> None:
    px = to_px(xy_yaw[:, :2], mn, mx, size)
    draw.line([tuple(p) for p in px], fill=(220, 38, 38, 245), width=width, joint="curve")

    dot_step = max(1, len(xy_yaw) // 260)
    for i in range(0, len(xy_yaw), dot_step):
        p = px[i]
        draw.ellipse((p[0] - 8, p[1] - 8, p[0] + 8, p[1] + 8), fill=(127, 29, 29, 165))

    arrow_step = max(1, len(xy_yaw) // 95)
    arrow_len = float((mx - mn).max()) * 0.018
    for i in range(0, len(xy_yaw), arrow_step):
        x, y, yaw = xy_yaw[i]
        start = to_px(np.array([[x, y]], dtype=np.float64), mn, mx, size)[0]
        end_xy = np.array([[x + math.cos(yaw) * arrow_len, y + math.sin(yaw) * arrow_len]], dtype=np.float64)
        end = to_px(end_xy, mn, mx, size)[0]
        draw.line([tuple(start), tuple(end)], fill=(30, 64, 175, 225), width=8)
        draw.ellipse((end[0] - 8, end[1] - 8, end[0] + 8, end[1] + 8), fill=(30, 64, 175, 240))

    label_font = load_font(42, bold=True)
    label_step = max(1, len(xy_yaw) // 32)
    for i in range(0, len(xy_yaw), label_step):
        p = px[i]
        label = str(i)
        box_w = 34 + 24 * len(label)
        draw.rounded_rectangle((p[0] + 12, p[1] - 30, p[0] + box_w, p[1] + 28), radius=10, fill=(255, 255, 255, 225), outline=(185, 28, 28, 210), width=3)
        draw.text((p[0] + 23, p[1] - 27), label, fill=(127, 29, 29, 255), font=label_font)

    for idx, name, color in [(0, "START", (22, 163, 74, 255)), (len(xy_yaw) - 1, "END", (220, 38, 38, 255))]:
        p = px[idx]
        draw.ellipse((p[0] - 42, p[1] - 42, p[0] + 42, p[1] + 42), fill=color, outline=(255, 255, 255, 255), width=7)
        draw.text((p[0] + 52, p[1] - 30), name, fill=color, font=label_font)


def render_pose_only(xy_yaw: np.ndarray, output: Path, size: int) -> tuple[np.ndarray, np.ndarray]:
    mn, mx = square_bounds(xy_yaw[:, :2], pad=18.0)
    img = Image.new("RGBA", (size, size), (248, 250, 252, 255))
    draw = ImageDraw.Draw(img, "RGBA")
    draw_grid(draw, mn, mx, size, step=10.0)
    draw_trajectory(draw, xy_yaw, mn, mx, size)
    total_dist = float(np.linalg.norm(np.diff(xy_yaw[:, :2], axis=0), axis=1).sum())
    draw_title(
        draw,
        size,
        "Dataset/world true pose trajectory - ultra HD",
        f"No ICpark/OBJ map used | pose rows={len(xy_yaw)} | path length={total_dist:.1f} world-units",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    img.convert("RGB").save(output, quality=96)
    return mn, mx


def accumulate_lidar_occupancy(
    dataset_root: Path,
    poses: np.ndarray,
    matches: dict[int, int],
    mn: np.ndarray,
    mx: np.ndarray,
    size: int,
    lidar_step: int,
    max_points_per_frame: int,
    range_m: float,
    z_min: float,
    z_max: float,
) -> tuple[np.ndarray, int, int]:
    margin = 220
    usable = size - 2 * margin
    hist = np.zeros((size, size), dtype=np.uint16)
    lidar_files = sorted((dataset_root / "velodyne").glob("*.bin"))
    used_frames = 0
    used_points = 0

    for lidar_idx, path in enumerate(lidar_files):
        if lidar_idx % max(1, lidar_step) != 0:
            continue
        pose_idx = matches.get(lidar_idx, lidar_idx * 5)
        if pose_idx < 0 or pose_idx >= len(poses):
            continue
        pc = load_lidar(path)
        if len(pc) == 0:
            continue
        xyz = pc[:, :3].astype(np.float64)
        dist_xy = np.linalg.norm(xyz[:, :2], axis=1)
        mask = (
            (dist_xy <= range_m)
            & (xyz[:, 2] >= z_min)
            & (xyz[:, 2] <= z_max)
            & np.isfinite(xyz).all(axis=1)
        )
        xyz = xyz[mask]
        if len(xyz) == 0:
            continue
        if len(xyz) > max_points_per_frame:
            step = max(1, len(xyz) // max_points_per_frame)
            xyz = xyz[::step][:max_points_per_frame]
        world = transform_points(poses[pose_idx], xyz)
        x = margin + (world[:, 0] - mn[0]) / (mx[0] - mn[0]) * usable
        y = size - margin - (world[:, 1] - mn[1]) / (mx[1] - mn[1]) * usable
        ix = np.floor(x).astype(np.int32)
        iy = np.floor(y).astype(np.int32)
        ok = (ix >= 0) & (ix < size) & (iy >= 0) & (iy < size)
        ix = ix[ok]
        iy = iy[ok]
        if len(ix) == 0:
            continue
        np.add.at(hist, (iy, ix), 1)
        used_frames += 1
        used_points += int(len(ix))

    return hist, used_frames, used_points


def render_lidar_pose(
    xy_yaw: np.ndarray,
    hist: np.ndarray,
    used_frames: int,
    used_points: int,
    mn: np.ndarray,
    mx: np.ndarray,
    output: Path,
    size: int,
) -> None:
    # Compress occupancy to a visible dark-blue density layer.
    h = hist.astype(np.float32)
    if h.max() > 0:
        h = np.log1p(h)
        h /= h.max()
    bg = np.full((size, size, 3), 248, dtype=np.uint8)
    bg[:, :, 1] = 250
    bg[:, :, 2] = 252
    density = (h * 210).astype(np.uint8)
    bg[:, :, 0] = np.minimum(bg[:, :, 0], 248 - density // 2)
    bg[:, :, 1] = np.minimum(bg[:, :, 1], 250 - density // 3)
    bg[:, :, 2] = np.maximum(bg[:, :, 2], 252 - density // 8)
    img = Image.fromarray(bg, mode="RGB").convert("RGBA")
    draw = ImageDraw.Draw(img, "RGBA")
    draw_grid(draw, mn, mx, size, step=10.0)
    draw_trajectory(draw, xy_yaw, mn, mx, size, width=14)
    draw_title(
        draw,
        size,
        "Dataset/world LiDAR accumulation + true pose trajectory - ultra HD",
        f"No ICpark/OBJ map used | lidar frames={used_frames}, plotted points={used_points:,}, pose rows={len(xy_yaw)}",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    img.convert("RGB").save(output, quality=96)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    poses = load_poses(args.dataset_root / "pose" / "poses.txt")
    xy_yaw = pose_to_xyyaw(poses)
    matches = load_matches(args.match_file)

    pose_png = args.output_dir / "pose_world_trajectory_ultra_hd.png"
    lidar_pose_png = args.output_dir / "pose_lidar_world_accumulation_ultra_hd.png"
    mn, mx = render_pose_only(xy_yaw, pose_png, args.size)
    hist, used_frames, used_points = accumulate_lidar_occupancy(
        args.dataset_root,
        poses,
        matches,
        mn,
        mx,
        args.size,
        args.lidar_step,
        args.max_points_per_frame,
        args.range_m,
        args.z_min,
        args.z_max,
    )
    render_lidar_pose(xy_yaw, hist, used_frames, used_points, mn, mx, lidar_pose_png, args.size)
    summary = {
        "pose_png": str(pose_png),
        "lidar_pose_png": str(lidar_pose_png),
        "size": [args.size, args.size],
        "pose_rows": int(len(xy_yaw)),
        "lidar_frames_used": int(used_frames),
        "lidar_points_plotted": int(used_points),
        "note": "No ICpark/OBJ map is used. These images are in dataset/world visualization space.",
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
