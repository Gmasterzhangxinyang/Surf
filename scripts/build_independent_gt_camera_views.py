#!/usr/bin/env python3
"""Build per-slot camera review sheets without using Part1 predictions."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from audit_camera_extrinsic import candidates, project


def _local_xy(points_map: np.ndarray, row: dict[str, str], scale: float) -> np.ndarray:
    delta = (points_map - np.asarray([float(row["map_x"]), float(row["map_y"])])) / scale
    yaw = float(row["map_yaw"])
    rotation = np.asarray(
        [[math.cos(yaw), math.sin(yaw)], [-math.sin(yaw), math.cos(yaw)]],
        dtype=np.float64,
    )
    return delta @ rotation.T


def _ground_height(path: Path) -> float:
    points = np.fromfile(path, dtype=np.float32).reshape(-1, 4)[:, :3]
    distance = np.hypot(points[:, 0], points[:, 1])
    values = points[(distance >= 3.0) & (distance <= 20.0), 2]
    values = values[np.isfinite(values)]
    return float(np.quantile(values, 0.08)) if len(values) else -0.82


def _font(size: int) -> ImageFont.ImageFont:
    paths = [
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
        Path("/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf"),
    ]
    for path in paths:
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def _render_tile(
    row: dict[str, str],
    polygon_uv: np.ndarray,
    slot_id: str,
    *,
    tile_width: int = 640,
) -> Image.Image:
    source = Image.open(row["camera_image_path"]).convert("RGB")
    scale = tile_width / source.width
    tile_height = int(round(source.height * scale))
    source = source.resize((tile_width, tile_height), Image.Resampling.LANCZOS)
    overlay = Image.new("RGBA", source.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    points = [(float(x * scale), float(y * scale)) for x, y in polygon_uv]
    draw.polygon(points, fill=(0, 210, 255, 58), outline=(0, 230, 255, 255), width=4)
    center = tuple(np.mean(np.asarray(points), axis=0))
    draw.ellipse(
        [center[0] - 6, center[1] - 6, center[0] + 6, center[1] + 6],
        fill=(255, 70, 70, 255),
    )
    text = (
        f"{slot_id} | LiDAR {int(row['frame']):06d} | "
        f"Camera {int(float(row['camera_frame'])):06d} | "
        f"dt={float(row['camera_lidar_dt_sec']) * 1000:+.1f} ms"
    )
    draw.rectangle((0, 0, tile_width, 31), fill=(4, 14, 30, 220))
    draw.text((9, 6), text, fill=(255, 255, 255, 255), font=_font(15))
    return Image.alpha_composite(source.convert("RGBA"), overlay).convert("RGB")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames-csv", type=Path, required=True)
    parser.add_argument("--universe", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-views", type=int, default=8)
    parser.add_argument("--candidate", default="standard_camera_origin")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    args.output_dir.mkdir(parents=True)

    universe = json.loads(args.universe.read_text(encoding="utf-8"))
    scale = float(universe["map_units_per_meter"])
    history = set(int(value) for value in universe["history_frames"])
    with args.frames_csv.open("r", encoding="utf-8", newline="") as handle:
        frames = [row for row in csv.DictReader(handle) if int(row["frame"]) in history]
    convention = next(item for item in candidates() if item.name == args.candidate)

    rows = []
    for slot in universe["slots"]:
        polygon_map = np.asarray(slot["polygon_map"], dtype=np.float64)
        visible = []
        for row in frames:
            polygon_xy = _local_xy(polygon_map, row, scale)
            center = polygon_xy.mean(axis=0)
            bearing = abs(math.degrees(math.atan2(center[1], center[0])))
            if center[0] <= 1.0 or bearing >= 70.0:
                continue
            image = Image.open(row["camera_image_path"])
            ground_z = _ground_height(Path(row["lidar_path"]))
            points_lidar = np.column_stack(
                [polygon_xy, np.full(len(polygon_xy), ground_z, dtype=np.float64)]
            )
            points_camera = convention.transform(points_lidar)
            uv, valid = project(points_camera, image.width, image.height)
            if not valid.all():
                continue
            area = 0.5 * abs(
                np.dot(uv[:, 0], np.roll(uv[:, 1], 1))
                - np.dot(uv[:, 1], np.roll(uv[:, 0], 1))
            )
            visible.append(
                {
                    "row": row,
                    "uv": uv,
                    "distance_m": float(np.linalg.norm(center)),
                    "bearing_deg": float(bearing),
                    "projected_area_px2": float(area),
                }
            )
        visible.sort(key=lambda item: (-item["projected_area_px2"], item["distance_m"]))
        selected = []
        for item in visible:
            frame = int(item["row"]["frame"])
            if all(abs(frame - int(other["row"]["frame"])) >= 3 for other in selected):
                selected.append(item)
            if len(selected) >= args.max_views:
                break
        selected.sort(key=lambda item: int(item["row"]["frame"]))

        output = args.output_dir / f"{slot['slot_id']}.jpg"
        if selected:
            tiles = [_render_tile(item["row"], item["uv"], slot["slot_id"]) for item in selected]
            columns = 2
            tile_width, tile_height = tiles[0].size
            header = 52
            canvas = Image.new(
                "RGB",
                (columns * tile_width, header + math.ceil(len(tiles) / columns) * tile_height),
                "white",
            )
            draw = ImageDraw.Draw(canvas)
            draw.text(
                (12, 9),
                f"{slot['slot_id']} — prediction-blind camera identity review; state not assigned",
                fill=(12, 29, 53),
                font=_font(22),
            )
            for index, tile in enumerate(tiles):
                x = (index % columns) * tile_width
                y = header + (index // columns) * tile_height
                canvas.paste(tile, (x, y))
            canvas.save(output, quality=92, subsampling=0)
        rows.append(
            {
                "slot_id": slot["slot_id"],
                "view_count": len(selected),
                "image": str(output) if selected else None,
                "frames": [int(item["row"]["frame"]) for item in selected],
                "projected_area_px2": [
                    item["projected_area_px2"] for item in selected
                ],
            }
        )

    manifest = {
        "schema_version": "parking-slot-independent-camera-review/1.0",
        "prediction_blind": True,
        "part1_fields_used": [],
        "state_labels_used": [],
        "extrinsic_convention": args.candidate,
        "extrinsic_status": "edge-diagnostic-supported; not survey calibration",
        "ground_height": "per-frame raw LiDAR 8th percentile at 3-20m",
        "rows": rows,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "slot_count": len(rows),
                "slots_with_views": sum(row["view_count"] > 0 for row in rows),
                "total_views": sum(row["view_count"] for row in rows),
                "output_dir": str(args.output_dir),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
