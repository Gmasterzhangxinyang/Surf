#!/usr/bin/env python3
"""Filter projected Camera GT views using nearer synchronized LiDAR surfaces."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from audit_camera_extrinsic import project
from build_independent_gt_camera_views import _font, _ground_height, _local_xy, _render_tile
from calibrate_left_camera_extrinsic import _camera_points


def foreground_audit(
    row: dict[str, str],
    polygon_map: np.ndarray,
    scale: float,
    origin: np.ndarray,
    yaw_deg: float,
    *,
    depth_margin_m: float,
    pixel_margin: int,
    min_points: int,
) -> tuple[bool, dict, np.ndarray]:
    image = Image.open(row["camera_image_path"])
    polygon_xy = _local_xy(polygon_map, row, scale)
    ground_z = _ground_height(Path(row["lidar_path"]))
    target_lidar = np.column_stack(
        [polygon_xy, np.full(len(polygon_xy), ground_z, dtype=np.float64)]
    )
    target_camera = _camera_points(
        target_lidar, origin_lidar_m=origin, yaw_deg=yaw_deg
    )
    target_uv, target_valid = project(target_camera, image.width, image.height)
    if not target_valid.all():
        return True, {"reason": "target_projection_invalid"}, target_uv

    lidar = np.fromfile(row["lidar_path"], dtype=np.float32).reshape(-1, 4)[:, :3]
    lidar = lidar[np.isfinite(lidar).all(axis=1)]
    camera = _camera_points(lidar, origin_lidar_m=origin, yaw_deg=yaw_deg)
    uv, valid = project(camera, image.width, image.height)
    target_depth = float(np.median(target_camera[:, 2]))
    x1 = max(0.0, float(target_uv[:, 0].min()) - pixel_margin)
    x2 = min(float(image.width - 1), float(target_uv[:, 0].max()) + pixel_margin)
    y1 = max(0.0, float(target_uv[:, 1].min()) - pixel_margin)
    y2 = min(float(image.height - 1), float(target_uv[:, 1].max()) + pixel_margin)
    foreground = (
        valid
        & (uv[:, 0] >= x1)
        & (uv[:, 0] <= x2)
        & (uv[:, 1] >= y1)
        & (uv[:, 1] <= y2)
        & (camera[:, 2] >= 0.5)
        & (camera[:, 2] <= target_depth - depth_margin_m)
    )
    points = camera[foreground]
    count = int(len(points))
    if count:
        depth_gap = target_depth - points[:, 2]
        median_gap = float(np.median(depth_gap))
        span_x = float(np.ptp(uv[foreground, 0]))
        span_y = float(np.ptp(uv[foreground, 1]))
    else:
        median_gap = span_x = span_y = 0.0
    # A true wall is not only a handful of foreground returns; it forms a
    # spatially extended closer surface over the projected target.
    extended_surface = span_x >= 22.0 or span_y >= 22.0
    blocked = count >= min_points and extended_surface
    return blocked, {
        "reason": "foreground_occluded" if blocked else "foreground_clear",
        "foreground_point_count": count,
        "foreground_median_depth_gap_m": median_gap,
        "foreground_pixel_span": [span_x, span_y],
        "target_depth_m": target_depth,
        "depth_margin_m": depth_margin_m,
        "pixel_bbox_with_margin": [x1, y1, x2, y2],
    }, target_uv


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--source-manifest", type=Path, required=True)
    p.add_argument("--frames-csv", type=Path, required=True)
    p.add_argument("--universe", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--depth-margin-m", type=float, default=0.8)
    p.add_argument("--pixel-margin", type=int, default=18)
    p.add_argument("--min-foreground-points", type=int, default=12)
    a = p.parse_args()
    if a.output_dir.exists():
        raise FileExistsError(a.output_dir)
    a.output_dir.mkdir(parents=True)

    source = json.loads(a.source_manifest.read_text(encoding="utf-8"))
    universe = json.loads(a.universe.read_text(encoding="utf-8"))
    scale = float(universe["map_units_per_meter"])
    slots = {str(x["slot_id"]): x for x in universe["slots"]}
    with a.frames_csv.open(newline="", encoding="utf-8") as f:
        frames = {int(row["frame"]): row for row in csv.DictReader(f)}
    origin = np.asarray(source["camera_origin_lidar_m"], dtype=np.float64)
    yaw_deg = float(source["camera_yaw_deg"])

    rows = []
    rejected = []
    for source_row in source["rows"]:
        row_out = dict(source_row)
        sid = str(source_row["slot_id"])
        polygon = np.asarray(slots[sid]["polygon_map"], dtype=np.float64)
        kept = []
        lidar_rejections = []
        for frame_id in source_row.get("frames", []):
            frame = frames[int(frame_id)]
            blocked, audit, uv = foreground_audit(
                frame,
                polygon,
                scale,
                origin,
                yaw_deg,
                depth_margin_m=a.depth_margin_m,
                pixel_margin=a.pixel_margin,
                min_points=a.min_foreground_points,
            )
            item = {"lidar_frame": int(frame_id), "camera_frame": int(float(frame["camera_frame"])), **audit}
            if blocked:
                lidar_rejections.append(item)
                rejected.append({"slot_id": sid, **item})
            else:
                kept.append({"row": frame, "uv": uv, "audit": audit})

        output = a.output_dir / f"{sid}.jpg"
        if kept:
            tiles = [_render_tile(x["row"], x["uv"], sid) for x in kept]
            cols, width, height, header = 2, tiles[0].width, tiles[0].height, 68
            canvas = Image.new(
                "RGB",
                (cols * width, header + math.ceil(len(tiles) / cols) * height),
                "white",
            )
            draw = ImageDraw.Draw(canvas)
            draw.text(
                (12, 7),
                (
                    f"{sid} | static LOS + LiDAR foreground filtered | "
                    f"camera yaw {yaw_deg:+.1f}deg"
                ),
                fill=(12, 29, 53),
                font=_font(18),
            )
            for index, tile in enumerate(tiles):
                canvas.paste(tile, ((index % cols) * width, header + (index // cols) * height))
            canvas.save(output, quality=92, subsampling=0)
        row_out["view_count"] = len(kept)
        row_out["image"] = str(output) if kept else None
        row_out["frames"] = [int(x["row"]["frame"]) for x in kept]
        row_out["camera_frames"] = [int(float(x["row"]["camera_frame"])) for x in kept]
        row_out["lidar_foreground_rejections"] = lidar_rejections
        rows.append(row_out)

    result = {
        **{k: v for k, v in source.items() if k != "rows"},
        "schema_version": "parking-slot-dual-occlusion-camera-review/1.0",
        "lidar_foreground_occlusion": {
            "enabled": True,
            "depth_margin_m": a.depth_margin_m,
            "pixel_margin": a.pixel_margin,
            "min_foreground_points": a.min_foreground_points,
            "rejected_view_count": len(rejected),
            "limitation": "sparse LiDAR can miss glass or thin occluders; human Unknown remains authoritative",
        },
        "rows": rows,
    }
    (a.output_dir / "manifest.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "slots_with_views": sum(x["view_count"] > 0 for x in rows),
                "total_views": sum(x["view_count"] for x in rows),
                "lidar_foreground_rejections": len(rejected),
                "camera18529": [x for x in rejected if x["camera_frame"] == 18529],
                "output_dir": str(a.output_dir),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
