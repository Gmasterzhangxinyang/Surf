#!/usr/bin/env python3
"""Build prediction-blind Camera GT sheets with static-map LOS occlusion."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from audit_camera_extrinsic import project
from build_independent_gt_camera_views import _font, _ground_height, _local_xy, _render_tile
from calibrate_left_camera_extrinsic import _camera_points
from gltf_lidar_ndt import load_gltf_map, voxel_downsample


def camera_map_xy(row: dict[str, str], origin: np.ndarray, scale: float) -> np.ndarray:
    yaw = float(row["map_yaw"])
    c, s = math.cos(yaw), math.sin(yaw)
    forward, left = float(origin[0]), float(origin[1])
    return np.asarray(
        [
            float(row["map_x"]) + scale * (c * forward - s * left),
            float(row["map_y"]) + scale * (s * forward + c * left),
        ],
        dtype=np.float64,
    )


def ray_blocked(
    origin_map: np.ndarray,
    target_map: np.ndarray,
    wall_tree: cKDTree,
    wall_points: np.ndarray,
    scale: float,
    clearance_m: float,
    endpoint_margin_m: float,
) -> bool:
    segment = target_map - origin_map
    length_map = float(np.linalg.norm(segment))
    length_m = length_map / scale
    if length_m <= 2 * endpoint_margin_m:
        return False
    midpoint = 0.5 * (origin_map + target_map)
    indices = wall_tree.query_ball_point(
        midpoint,
        r=0.5 * length_map + clearance_m * scale,
    )
    if not indices:
        return False
    points = wall_points[np.asarray(indices, dtype=np.int64)]
    relative = points - origin_map
    t = np.clip((relative @ segment) / (length_map * length_map), 0.0, 1.0)
    closest = origin_map + t[:, None] * segment
    perpendicular_m = np.linalg.norm(points - closest, axis=1) / scale
    along_m = t * length_m
    hit = (
        (perpendicular_m <= clearance_m)
        & (along_m >= endpoint_margin_m)
        & (along_m <= length_m - endpoint_margin_m)
    )
    return bool(np.any(hit))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--frames-csv", type=Path, required=True)
    p.add_argument("--universe", type=Path, required=True)
    p.add_argument("--gltf", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--max-views", type=int, default=8)
    p.add_argument("--camera-origin-x-m", type=float, default=0.6)
    p.add_argument("--camera-origin-y-m", type=float, default=0.36)
    p.add_argument("--camera-origin-z-m", type=float, default=-0.07)
    p.add_argument("--camera-yaw-deg", type=float, default=-1.0)
    p.add_argument("--wall-clearance-m", type=float, default=0.16)
    p.add_argument("--endpoint-margin-m", type=float, default=0.65)
    p.add_argument("--min-clear-ray-fraction", type=float, default=0.40)
    a = p.parse_args()
    if a.output_dir.exists():
        raise FileExistsError(a.output_dir)
    a.output_dir.mkdir(parents=True)

    universe = json.loads(a.universe.read_text(encoding="utf-8"))
    scale = float(universe["map_units_per_meter"])
    history = set(int(x) for x in universe["history_frames"])
    with a.frames_csv.open(newline="", encoding="utf-8") as f:
        frames = [row for row in csv.DictReader(f) if int(row["frame"]) in history]
    origin = np.asarray(
        [a.camera_origin_x_m, a.camera_origin_y_m, a.camera_origin_z_m],
        dtype=np.float64,
    )

    gltf = load_gltf_map(a.gltf, sample_step=0.04 * scale)
    static_chunks = []
    for name in ("wall", "elevator"):
        if name in gltf.layers and len(gltf.layers[name].points):
            static_chunks.append(gltf.layers[name].points)
    wall_points = voxel_downsample(np.vstack(static_chunks), 0.08 * scale)
    wall_tree = cKDTree(wall_points)

    rows = []
    rejected_total = 0
    for slot in universe["slots"]:
        polygon_map = np.asarray(slot["polygon_map"], dtype=np.float64)
        samples_map = np.vstack([polygon_map.mean(axis=0), polygon_map])
        visible = []
        rejected = []
        for row in frames:
            polygon_xy = _local_xy(polygon_map, row, scale)
            center = polygon_xy.mean(axis=0)
            if center[0] <= 1.0:
                continue
            image = Image.open(row["camera_image_path"])
            ground_z = _ground_height(Path(row["lidar_path"]))
            lidar_points = np.column_stack(
                [polygon_xy, np.full(len(polygon_xy), ground_z)]
            )
            camera_points = _camera_points(
                lidar_points,
                origin_lidar_m=origin,
                yaw_deg=a.camera_yaw_deg,
            )
            uv, valid = project(camera_points, image.width, image.height)
            if not valid.all():
                continue
            camera_xy = camera_map_xy(row, origin, scale)
            blocked = [
                ray_blocked(
                    camera_xy,
                    target,
                    wall_tree,
                    wall_points,
                    scale,
                    a.wall_clearance_m,
                    a.endpoint_margin_m,
                )
                for target in samples_map
            ]
            clear_fraction = 1.0 - sum(blocked) / len(blocked)
            if clear_fraction < a.min_clear_ray_fraction:
                rejected.append(
                    {
                        "lidar_frame": int(row["frame"]),
                        "camera_frame": int(float(row["camera_frame"])),
                        "clear_ray_fraction": clear_fraction,
                        "reason": "static_occluded_wall_or_elevator",
                    }
                )
                rejected_total += 1
                continue
            area = 0.5 * abs(
                np.dot(uv[:, 0], np.roll(uv[:, 1], 1))
                - np.dot(uv[:, 1], np.roll(uv[:, 0], 1))
            )
            visible.append(
                {
                    "row": row,
                    "uv": uv,
                    "distance_m": float(np.linalg.norm(center - origin[:2])),
                    "projected_area_px2": float(area),
                    "clear_ray_fraction": clear_fraction,
                }
            )
        visible.sort(key=lambda x: (-x["projected_area_px2"], x["distance_m"]))
        selected = []
        for item in visible:
            frame = int(item["row"]["frame"])
            if all(abs(frame - int(other["row"]["frame"])) >= 3 for other in selected):
                selected.append(item)
            if len(selected) >= a.max_views:
                break
        selected.sort(key=lambda x: int(x["row"]["frame"]))

        output = a.output_dir / f"{slot['slot_id']}.jpg"
        if selected:
            tiles = [_render_tile(x["row"], x["uv"], slot["slot_id"]) for x in selected]
            cols, tile_w, tile_h, header = 2, tiles[0].width, tiles[0].height, 66
            canvas = Image.new(
                "RGB",
                (cols * tile_w, header + math.ceil(len(tiles) / cols) * tile_h),
                "white",
            )
            draw = ImageDraw.Draw(canvas)
            draw.text(
                (12, 7),
                (
                    f"{slot['slot_id']} | static-LOS filtered | camera "
                    f"({origin[0]:.2f},{origin[1]:.2f},{origin[2]:.2f})m "
                    f"yaw {a.camera_yaw_deg:+.1f}deg"
                ),
                fill=(12, 29, 53),
                font=_font(18),
            )
            for index, tile in enumerate(tiles):
                canvas.paste(
                    tile,
                    ((index % cols) * tile_w, header + (index // cols) * tile_h),
                )
            canvas.save(output, quality=92, subsampling=0)
        rows.append(
            {
                "slot_id": slot["slot_id"],
                "view_count": len(selected),
                "image": str(output) if selected else None,
                "frames": [int(x["row"]["frame"]) for x in selected],
                "camera_frames": [int(float(x["row"]["camera_frame"])) for x in selected],
                "projected_area_px2": [x["projected_area_px2"] for x in selected],
                "clear_ray_fraction": [x["clear_ray_fraction"] for x in selected],
                "static_occlusion_rejections": rejected,
            }
        )

    manifest = {
        "schema_version": "parking-slot-occlusion-aware-camera-review/1.0",
        "prediction_blind": True,
        "part1_fields_used": [],
        "state_labels_used": [],
        "camera_origin_lidar_m": origin.tolist(),
        "camera_yaw_deg": a.camera_yaw_deg,
        "static_occlusion": {
            "enabled": True,
            "layers": ["wall", "elevator"],
            "wall_clearance_m": a.wall_clearance_m,
            "endpoint_margin_m": a.endpoint_margin_m,
            "min_clear_ray_fraction": a.min_clear_ray_fraction,
            "sample_rays": "slot_center_plus_four_corners",
            "rejected_view_count": rejected_total,
            "limitation": "dynamic vehicles and unmapped obstacles are not modeled",
        },
        "rows": rows,
    }
    (a.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "slots_with_views": sum(x["view_count"] > 0 for x in rows),
                "total_views": sum(x["view_count"] for x in rows),
                "static_occlusion_rejections": rejected_total,
                "frame6184_rejections": [
                    (x["slot_id"], r["camera_frame"])
                    for x in rows
                    for r in x["static_occlusion_rejections"]
                    if r["lidar_frame"] == 6184
                ],
                "output_dir": str(a.output_dir),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
