#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Optional

import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont


DATASET_ROOT = Path("/home/ParkingAgent/dataset/dataset/dataset")
CONFIG_PATH = DATASET_ROOT / "choose" / "slot_camera_localizer_config.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a clear OBJ slot-structure map")
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--map-obj", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--output", type=Path, default=Path("outputs/slot_structure_map/slot_structure.png"))
    parser.add_argument("--width", type=int, default=1800)
    parser.add_argument("--height", type=int, default=2200)
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def parse_obj_index(token: str, vertex_count: int) -> Optional[int]:
    head = token.split("/", 1)[0]
    if not head:
        return None
    try:
        raw = int(head)
    except ValueError:
        return None
    idx = raw - 1 if raw > 0 else vertex_count + raw
    return idx if idx >= 0 else None


def load_obj_geometry(path: Path) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray]]:
    vertices: list[tuple[float, float]] = []
    face_indices: list[list[int]] = []
    line_indices: list[list[int]] = []

    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if line.startswith("v "):
                vals = np.fromstring(line[2:], sep=" ")
                if vals.size >= 2 and np.isfinite(vals[:2]).all():
                    vertices.append((float(vals[0]), float(vals[1])))
            elif line.startswith("f "):
                idxs = [parse_obj_index(token, len(vertices)) for token in line.split()[1:]]
                idxs = [idx for idx in idxs if idx is not None]
                if len(idxs) >= 3:
                    face_indices.append(idxs)
            elif line.startswith("l "):
                idxs = [parse_obj_index(token, len(vertices)) for token in line.split()[1:]]
                idxs = [idx for idx in idxs if idx is not None]
                if len(idxs) >= 2:
                    line_indices.append(idxs)

    verts = np.asarray(vertices, dtype=np.float64)
    faces = [verts[idxs, :2] for idxs in face_indices if max(idxs) < len(verts)]
    line_segments: list[np.ndarray] = []
    for idxs in line_indices:
        if max(idxs) >= len(verts):
            continue
        pts = verts[idxs, :2]
        for i in range(len(pts) - 1):
            line_segments.append(np.vstack([pts[i], pts[i + 1]]))
    return verts[:, :2], faces, line_segments


def load_slots(path: Path) -> list[dict]:
    if not path.exists():
        return []
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        return []
    return data.get("slots", []) or []


def calculate_bounds(points: np.ndarray, width: int, height: int) -> dict:
    min_xy = points.min(axis=0)
    max_xy = points.max(axis=0)
    pad = max(0.35, float((max_xy - min_xy).max()) * 0.04)
    min_xy = min_xy - pad
    max_xy = max_xy + pad
    world_w = float(max_xy[0] - min_xy[0])
    world_h = float(max_xy[1] - min_xy[1])
    aspect = width / height
    world_aspect = world_w / world_h
    if world_aspect > aspect:
        new_h = world_w / aspect
        extra = (new_h - world_h) * 0.5
        min_xy[1] -= extra
        max_xy[1] += extra
    else:
        new_w = world_h * aspect
        extra = (new_w - world_w) * 0.5
        min_xy[0] -= extra
        max_xy[0] += extra
    return {"minX": min_xy[0], "maxX": max_xy[0], "minY": min_xy[1], "maxY": max_xy[1], "width": width, "height": height}


def world_to_px(points: np.ndarray, bounds: dict) -> np.ndarray:
    x = (points[:, 0] - bounds["minX"]) / (bounds["maxX"] - bounds["minX"]) * bounds["width"]
    y = (bounds["maxY"] - points[:, 1]) / (bounds["maxY"] - bounds["minY"]) * bounds["height"]
    return np.column_stack([x, y])


def draw_polyline(draw: ImageDraw.ImageDraw, points: np.ndarray, bounds: dict, fill: tuple[int, int, int, int], width: int) -> None:
    px = world_to_px(points, bounds)
    draw.line([tuple(p) for p in px], fill=fill, width=width, joint="curve")


def polygon_center(points: np.ndarray) -> np.ndarray:
    return points.mean(axis=0)


def main() -> None:
    args = parse_args()
    map_obj = args.map_obj or args.dataset_root / "file.obj"
    vertices, faces, line_segments = load_obj_geometry(map_obj)
    slots = load_slots(args.config)
    bounds = calculate_bounds(vertices, args.width, args.height)

    image = Image.new("RGB", (args.width, args.height), (248, 248, 242))
    draw = ImageDraw.Draw(image, "RGBA")
    font = ImageFont.load_default()

    grid_step = 1.0
    x = math.floor(bounds["minX"] / grid_step) * grid_step
    while x <= bounds["maxX"]:
        px = world_to_px(np.array([[x, bounds["minY"]]], dtype=np.float64), bounds)[0, 0]
        draw.line([(px, 0), (px, args.height)], fill=(180, 188, 199, 70), width=1)
        x += grid_step
    y = math.floor(bounds["minY"] / grid_step) * grid_step
    while y <= bounds["maxY"]:
        py = world_to_px(np.array([[bounds["minX"], y]], dtype=np.float64), bounds)[0, 1]
        draw.line([(0, py), (args.width, py)], fill=(180, 188, 199, 70), width=1)
        y += grid_step

    for face in faces[::2]:
        px = world_to_px(face, bounds)
        draw.polygon([tuple(p) for p in px], fill=(232, 234, 226, 100))

    for segment in line_segments:
        draw_polyline(draw, segment, bounds, (12, 18, 28, 210), 2)

    for slot in slots:
        poly = np.asarray(slot.get("polygon_xy", []), dtype=np.float64)
        if poly.shape != (4, 2):
            continue
        px = world_to_px(poly, bounds)
        draw.polygon([tuple(p) for p in px], fill=(245, 158, 11, 45), outline=(220, 38, 38, 235))
        closed = np.vstack([poly, poly[0]])
        draw_polyline(draw, closed, bounds, (220, 38, 38, 255), 4)
        center = world_to_px(polygon_center(poly)[None, :], bounds)[0]
        label = str(slot.get("slot_id", "slot"))
        draw.rounded_rectangle((center[0] - 24, center[1] - 10, center[0] + 24, center[1] + 10), radius=4, fill=(255, 255, 255, 230), outline=(220, 38, 38, 255))
        draw.text((center[0] - 20, center[1] - 5), label, fill=(120, 20, 20, 255), font=font)

    title = f"OBJ parking structure | vertices={len(vertices):,} faces={len(faces):,} lines={len(line_segments):,} configured_slots={len(slots)}"
    draw.rounded_rectangle((16, 16, args.width - 16, 58), radius=6, fill=(255, 255, 255, 225), outline=(185, 192, 202, 255))
    draw.text((28, 30), title, fill=(17, 24, 39, 255), font=font)

    ensure_dir(args.output.parent)
    image.save(args.output)
    print(args.output)
    print(f"vertices={len(vertices)} faces={len(faces)} lines={len(line_segments)} configured_slots={len(slots)}")


if __name__ == "__main__":
    main()
