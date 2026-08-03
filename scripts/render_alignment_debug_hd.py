#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = Path("/home/ParkingAgent/dataset/dataset/dataset")
MAP_DATA = Path("/tmp/ICpark_B1_map_B1_MapData.txt")
ALIGNMENT = ROOT / "outputs" / "pose_gltf_route_all" / "alignment.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render high-resolution map/trajectory alignment debug images.")
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--map-data", type=Path, default=MAP_DATA)
    parser.add_argument("--alignment", type=Path, default=ALIGNMENT)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "alignment_debug_hd")
    parser.add_argument("--size", type=int, default=6000)
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


def load_map(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["B1"] if "B1" in data else data


def load_poses(path: Path) -> np.ndarray:
    rows = np.loadtxt(path, dtype=np.float64)
    if rows.ndim == 1:
        rows = rows.reshape(1, -1)
    mats = rows.reshape(-1, 3, 4)
    xy_yaw = np.zeros((len(mats), 3), dtype=np.float64)
    xy_yaw[:, 0] = mats[:, 0, 3]
    xy_yaw[:, 1] = mats[:, 1, 3]
    xy_yaw[:, 2] = np.arctan2(mats[:, 1, 0], mats[:, 0, 0])
    return xy_yaw


def all_map_points(map_data: dict) -> np.ndarray:
    pts: list[list[float]] = []

    def rec(obj):
        if isinstance(obj, dict):
            if "coordinates" in obj:
                rec(obj["coordinates"])
            else:
                for value in obj.values():
                    rec(value)
        elif isinstance(obj, list):
            if len(obj) >= 2 and all(isinstance(v, (int, float)) for v in obj[:2]):
                pts.append([float(obj[0]), float(obj[1])])
            else:
                for value in obj:
                    rec(value)

    rec(map_data)
    return np.asarray(pts, dtype=np.float64)


def make_bounds(points: np.ndarray, pad_ratio: float = 0.055) -> tuple[np.ndarray, np.ndarray]:
    mn = points.min(axis=0)
    mx = points.max(axis=0)
    span = np.maximum(mx - mn, 1e-6)
    pad = max(float(span.max()) * pad_ratio, 0.1)
    return mn - pad, mx + pad


def square_bounds(mn: np.ndarray, mx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center = (mn + mx) * 0.5
    side = float((mx - mn).max())
    half = side * 0.5
    return center - half, center + half


def to_px(points: np.ndarray, mn: np.ndarray, mx: np.ndarray, size: int, margin: int = 170) -> np.ndarray:
    usable = size - 2 * margin
    x = margin + (points[:, 0] - mn[0]) / (mx[0] - mn[0]) * usable
    y = size - margin - (points[:, 1] - mn[1]) / (mx[1] - mn[1]) * usable
    return np.column_stack([x, y])


def draw_poly(draw: ImageDraw.ImageDraw, poly: Iterable, mn: np.ndarray, mx: np.ndarray, size: int, fill, outline, width: int = 1):
    arr = np.asarray(poly, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] < 2:
        return
    pts = [tuple(p) for p in to_px(arr, mn, mx, size)]
    if fill is not None and len(pts) >= 3:
        draw.polygon(pts, fill=fill)
    if outline is not None:
        closed = pts + [pts[0]] if len(pts) >= 3 else pts
        draw.line(closed, fill=outline, width=width, joint="curve")


def draw_line(draw: ImageDraw.ImageDraw, line: Iterable, mn: np.ndarray, mx: np.ndarray, size: int, fill, width: int = 1):
    arr = np.asarray(line, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] < 2:
        return
    pts = [tuple(p) for p in to_px(arr, mn, mx, size)]
    draw.line(pts, fill=fill, width=width, joint="curve")


def draw_grid(draw: ImageDraw.ImageDraw, mn: np.ndarray, mx: np.ndarray, size: int, step: float, color=(180, 190, 205, 70)):
    font = load_font(28)
    x = math.floor(mn[0] / step) * step
    while x <= mx[0]:
        pts = to_px(np.array([[x, mn[1]], [x, mx[1]]], dtype=np.float64), mn, mx, size)
        draw.line([tuple(pts[0]), tuple(pts[1])], fill=color, width=1)
        if int(round(x / step)) % 2 == 0:
            draw.text((pts[0, 0] + 6, size - 130), f"{x:.1f}", fill=(78, 88, 103, 180), font=font)
        x += step
    y = math.floor(mn[1] / step) * step
    while y <= mx[1]:
        pts = to_px(np.array([[mn[0], y], [mx[0], y]], dtype=np.float64), mn, mx, size)
        draw.line([tuple(pts[0]), tuple(pts[1])], fill=color, width=1)
        if int(round(y / step)) % 2 == 0:
            draw.text((42, pts[0, 1] - 15), f"{y:.1f}", fill=(78, 88, 103, 180), font=font)
        y += step


def draw_title(draw: ImageDraw.ImageDraw, size: int, title: str, subtitle: str):
    title_font = load_font(62, bold=True)
    sub_font = load_font(34)
    draw.rounded_rectangle((42, 42, size - 42, 160), radius=18, fill=(255, 255, 255, 235), outline=(180, 190, 205, 255), width=3)
    draw.text((70, 58), title, fill=(15, 23, 42, 255), font=title_font)
    draw.text((72, 120), subtitle, fill=(71, 85, 105, 255), font=sub_font)


def render_map(map_data: dict, output: Path, size: int) -> None:
    pts = all_map_points(map_data)
    mn, mx = square_bounds(*make_bounds(pts))
    img = Image.new("RGBA", (size, size), (248, 249, 244, 255))
    draw = ImageDraw.Draw(img, "RGBA")
    draw_grid(draw, mn, mx, size, step=0.5)

    for item in map_data.get("ground", []):
        draw_poly(draw, item.get("coordinates", []), mn, mx, size, fill=(235, 234, 222, 255), outline=(69, 78, 92, 220), width=7)

    for item in map_data.get("parkingSpace", []):
        draw_poly(draw, item.get("coordinates", []), mn, mx, size, fill=(34, 197, 94, 34), outline=(21, 128, 61, 185), width=2)

    for item in map_data.get("arrester", []):
        draw_poly(draw, item.get("coordinates", []), mn, mx, size, fill=(17, 24, 39, 170), outline=(17, 24, 39, 220), width=2)

    for item in map_data.get("wall", []):
        draw_poly(draw, item.get("coordinates", []), mn, mx, size, fill=(100, 116, 139, 130), outline=(30, 41, 59, 230), width=3)

    for item in map_data.get("laneLines", []):
        draw_line(draw, item.get("coordinates", []), mn, mx, size, fill=(245, 158, 11, 230), width=5)

    slot_font = load_font(13, bold=True)
    for idx, item in enumerate(map_data.get("parkingSpace", [])):
        center = item.get("center")
        if not center:
            continue
        px = to_px(np.asarray([center], dtype=np.float64), mn, mx, size)[0]
        name = str(item.get("name", idx))
        # Small but readable in full-resolution image.
        draw.text((px[0] - 13, px[1] - 8), name, fill=(5, 46, 22, 225), font=slot_font)

    draw_title(
        draw,
        size,
        "ICpark / OBJ semantic map - HD",
        f"slots={len(map_data.get('parkingSpace', []))}, walls={len(map_data.get('wall', []))}, laneLines={len(map_data.get('laneLines', []))}, arresters={len(map_data.get('arrester', []))}",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    img.convert("RGB").save(output, quality=95)


def draw_pose_path(
    draw: ImageDraw.ImageDraw,
    poses: np.ndarray,
    mn: np.ndarray,
    mx: np.ndarray,
    size: int,
    color=(220, 38, 38, 245),
    width: int = 8,
    label_prefix: str = "",
):
    px = to_px(poses[:, :2], mn, mx, size)
    draw.line([tuple(p) for p in px], fill=color, width=width, joint="curve")
    dot_font = load_font(30, bold=True)
    label_font = load_font(28, bold=True)

    step = max(1, len(poses) // 140)
    for i in range(0, len(poses), step):
        p = px[i]
        draw.ellipse((p[0] - 7, p[1] - 7, p[0] + 7, p[1] + 7), fill=(127, 29, 29, 190))

    label_step = max(1, len(poses) // 24)
    for i in range(0, len(poses), label_step):
        p = px[i]
        draw.rounded_rectangle((p[0] + 8, p[1] - 22, p[0] + 118, p[1] + 18), radius=7, fill=(255, 255, 255, 225), outline=(185, 28, 28, 210), width=2)
        draw.text((p[0] + 16, p[1] - 18), f"{label_prefix}{i}", fill=(127, 29, 29, 255), font=dot_font)

    for i in range(0, len(poses), max(1, len(poses) // 65)):
        x, y, yaw = poses[i]
        start = to_px(np.asarray([[x, y]], dtype=np.float64), mn, mx, size)[0]
        end_xy = np.asarray([[x + math.cos(yaw) * (mx[0] - mn[0]) * 0.018, y + math.sin(yaw) * (mx[1] - mn[1]) * 0.018]], dtype=np.float64)
        end = to_px(end_xy, mn, mx, size)[0]
        draw.line([tuple(start), tuple(end)], fill=(30, 64, 175, 210), width=5)
        draw.ellipse((end[0] - 5, end[1] - 5, end[0] + 5, end[1] + 5), fill=(30, 64, 175, 230))

    for idx, name, c in [(0, "START", (22, 163, 74, 255)), (len(poses) - 1, "END", (220, 38, 38, 255))]:
        p = px[idx]
        draw.ellipse((p[0] - 30, p[1] - 30, p[0] + 30, p[1] + 30), fill=c, outline=(255, 255, 255, 255), width=5)
        draw.text((p[0] + 38, p[1] - 22), name, fill=c, font=label_font)


def render_pose_world(poses: np.ndarray, output: Path, size: int) -> None:
    mn, mx = square_bounds(*make_bounds(poses[:, :2]))
    img = Image.new("RGBA", (size, size), (248, 250, 252, 255))
    draw = ImageDraw.Draw(img, "RGBA")
    draw_grid(draw, mn, mx, size, step=10.0)
    draw_pose_path(draw, poses, mn, mx, size, width=7)
    draw_title(
        draw,
        size,
        "Dataset pose trajectory - HD",
        f"pose rows={len(poses)}, x=[{poses[:,0].min():.2f},{poses[:,0].max():.2f}], y=[{poses[:,1].min():.2f},{poses[:,1].max():.2f}]",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    img.convert("RGB").save(output, quality=95)


def apply_alignment(poses: np.ndarray, alignment: dict) -> np.ndarray:
    scale = float(alignment["scale"])
    yaw = float(alignment["yaw"])
    tx, ty = alignment["translation"]
    c, s = math.cos(yaw), math.sin(yaw)
    xy = poses[:, :2]
    mapped = np.empty_like(poses)
    mapped[:, 0] = scale * (c * xy[:, 0] - s * xy[:, 1]) + tx
    mapped[:, 1] = scale * (s * xy[:, 0] + c * xy[:, 1]) + ty
    mapped[:, 2] = poses[:, 2] + yaw
    return mapped


def render_overlay(map_data: dict, poses: np.ndarray, alignment: dict, output: Path, size: int) -> None:
    mapped = apply_alignment(poses, alignment)
    pts = np.vstack([all_map_points(map_data), mapped[:, :2]])
    mn, mx = square_bounds(*make_bounds(pts))
    img = Image.new("RGBA", (size, size), (248, 249, 244, 255))
    draw = ImageDraw.Draw(img, "RGBA")
    draw_grid(draw, mn, mx, size, step=0.5)

    for item in map_data.get("ground", []):
        draw_poly(draw, item.get("coordinates", []), mn, mx, size, fill=(235, 234, 222, 255), outline=(69, 78, 92, 220), width=7)
    for item in map_data.get("parkingSpace", []):
        draw_poly(draw, item.get("coordinates", []), mn, mx, size, fill=(34, 197, 94, 30), outline=(21, 128, 61, 150), width=2)
    for item in map_data.get("wall", []):
        draw_poly(draw, item.get("coordinates", []), mn, mx, size, fill=(100, 116, 139, 120), outline=(30, 41, 59, 220), width=3)
    for item in map_data.get("laneLines", []):
        draw_line(draw, item.get("coordinates", []), mn, mx, size, fill=(245, 158, 11, 220), width=5)
    for item in map_data.get("arrester", []):
        draw_poly(draw, item.get("coordinates", []), mn, mx, size, fill=(17, 24, 39, 150), outline=(17, 24, 39, 210), width=2)

    draw_pose_path(draw, mapped, mn, mx, size, color=(239, 68, 68, 245), width=9, label_prefix="")
    draw_title(
        draw,
        size,
        "Current pose-to-map visual fit - HD overlay",
        f"WARNING: visual alignment only | scale={alignment['scale']:.8f}, yaw={math.degrees(float(alignment['yaw'])):.2f} deg, t={alignment['translation']}",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    img.convert("RGB").save(output, quality=95)


def main() -> None:
    args = parse_args()
    map_data = load_map(args.map_data)
    poses = load_poses(args.dataset_root / "pose" / "poses.txt")
    alignment = json.loads(args.alignment.read_text(encoding="utf-8"))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    map_png = args.output_dir / "icpark_map_detailed_hd.png"
    pose_png = args.output_dir / "pose_world_trajectory_detailed_hd.png"
    overlay_png = args.output_dir / "pose_on_icpark_visual_fit_detailed_hd.png"

    render_map(map_data, map_png, args.size)
    render_pose_world(poses, pose_png, args.size)
    render_overlay(map_data, poses, alignment, overlay_png, args.size)

    summary = {
        "map_png": str(map_png),
        "pose_png": str(pose_png),
        "overlay_png": str(overlay_png),
        "size": [args.size, args.size],
        "pose_rows": int(len(poses)),
        "slot_count": int(len(map_data.get("parkingSpace", []))),
        "warning": "overlay uses existing visual-fit alignment, not a verified world_to_map extrinsic",
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
