#!/usr/bin/env python3
"""Build prediction-blind slot camera sheets with an explicit left-camera pose."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from audit_camera_extrinsic import project
from build_independent_gt_camera_views import (
    _font,
    _ground_height,
    _local_xy,
    _render_tile,
)
from calibrate_left_camera_extrinsic import _camera_points


def _annotate_depth_tile(
    tile: Image.Image,
    *,
    target_depth_near_m: float,
    target_depth_center_m: float,
    target_depth_far_m: float,
    target_left_m: float,
) -> Image.Image:
    """Add metric target-plane ownership cues without adding a state label."""
    banner_height = 48
    canvas = Image.new("RGB", (tile.width, tile.height + banner_height), (8, 20, 36))
    canvas.paste(tile, (0, banner_height))
    draw = ImageDraw.Draw(canvas)
    side = "left" if target_left_m > 0.0 else "right"
    draw.text(
        (9, 4),
        (
            "TARGET DEPTH [near / center / far] = "
            f"[{target_depth_near_m:.1f} / {target_depth_center_m:.1f} / "
            f"{target_depth_far_m:.1f}] m | {side} {abs(target_left_m):.1f} m"
        ),
        fill=(180, 245, 255),
        font=_font(15),
    )
    draw.text(
        (9, 25),
        "Ownership: use wheel/ground contact + bay near/far edges; body overlap alone is invalid.",
        fill=(255, 222, 126),
        font=_font(14),
    )
    return canvas


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames-csv", type=Path, required=True)
    parser.add_argument("--universe", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-views", type=int, default=8)
    parser.add_argument("--camera-origin-x-m", type=float, default=0.6)
    parser.add_argument("--camera-origin-y-m", type=float, required=True)
    parser.add_argument("--camera-origin-z-m", type=float, default=-0.07)
    parser.add_argument("--camera-yaw-deg", type=float, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    args.output_dir.mkdir(parents=True)

    universe = json.loads(args.universe.read_text(encoding="utf-8"))
    scale = float(universe["map_units_per_meter"])
    history = set(int(value) for value in universe["history_frames"])
    with args.frames_csv.open("r", encoding="utf-8", newline="") as handle:
        frames = [row for row in csv.DictReader(handle) if int(row["frame"]) in history]
    origin = np.asarray(
        [
            args.camera_origin_x_m,
            args.camera_origin_y_m,
            args.camera_origin_z_m,
        ],
        dtype=np.float64,
    )

    rows = []
    for slot in universe["slots"]:
        polygon_map = np.asarray(slot["polygon_map"], dtype=np.float64)
        visible = []
        for row in frames:
            polygon_xy = _local_xy(polygon_map, row, scale)
            center = polygon_xy.mean(axis=0)
            if center[0] <= 1.0:
                continue
            image = Image.open(row["camera_image_path"])
            ground_z = _ground_height(Path(row["lidar_path"]))
            points_lidar = np.column_stack(
                [polygon_xy, np.full(len(polygon_xy), ground_z, dtype=np.float64)]
            )
            points_camera = _camera_points(
                points_lidar,
                origin_lidar_m=origin,
                yaw_deg=args.camera_yaw_deg,
            )
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
                    "distance_m": float(np.linalg.norm(center - origin[:2])),
                    "projected_area_px2": float(area),
                    "target_depth_near_m": float(np.min(points_camera[:, 2])),
                    "target_depth_center_m": float(np.mean(points_camera[:, 2])),
                    "target_depth_far_m": float(np.max(points_camera[:, 2])),
                    "target_left_m": float(-np.mean(points_camera[:, 0])),
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
            tiles = [
                _annotate_depth_tile(
                    _render_tile(item["row"], item["uv"], slot["slot_id"]),
                    target_depth_near_m=item["target_depth_near_m"],
                    target_depth_center_m=item["target_depth_center_m"],
                    target_depth_far_m=item["target_depth_far_m"],
                    target_left_m=item["target_left_m"],
                )
                for item in selected
            ]
            columns = 2
            tile_width, tile_height = tiles[0].size
            header = 58
            canvas = Image.new(
                "RGB",
                (columns * tile_width, header + math.ceil(len(tiles) / columns) * tile_height),
                "white",
            )
            draw = ImageDraw.Draw(canvas)
            draw.text(
                (12, 8),
                (
                    f"{slot['slot_id']} — left camera origin "
                    f"({origin[0]:.2f},{origin[1]:.2f},{origin[2]:.2f})m, "
                    f"yaw {args.camera_yaw_deg:+.1f} deg"
                ),
                fill=(12, 29, 53),
                font=_font(19),
            )
            for index, tile in enumerate(tiles):
                canvas.paste(
                    tile,
                    ((index % columns) * tile_width, header + (index // columns) * tile_height),
                )
            canvas.save(output, quality=92, subsampling=0)
        rows.append(
            {
                "slot_id": slot["slot_id"],
                "view_count": len(selected),
                "image": str(output) if selected else None,
                "frames": [int(item["row"]["frame"]) for item in selected],
                "projected_area_px2": [item["projected_area_px2"] for item in selected],
                "depth_ownership_views": [
                    {
                        "lidar_frame": int(item["row"]["frame"]),
                        "camera_frame": int(float(item["row"]["camera_frame"])),
                        "target_depth_near_m": item["target_depth_near_m"],
                        "target_depth_center_m": item["target_depth_center_m"],
                        "target_depth_far_m": item["target_depth_far_m"],
                        "target_left_m": item["target_left_m"],
                    }
                    for item in selected
                ],
            }
        )
        metadata_path = args.output_dir / f"{slot['slot_id']}.depth.json"
        metadata_path.write_text(
            json.dumps(
                {
                    "slot_id": slot["slot_id"],
                    "camera_origin_lidar_m": origin.tolist(),
                    "camera_yaw_deg": args.camera_yaw_deg,
                    "depth_ownership_views": rows[-1]["depth_ownership_views"],
                    "occupancy_ownership_contract": {
                        "body_overlap_sufficient": False,
                        "occupied_requires": [
                            "vehicle_ground_contact_inside_target_near_far_depth",
                            "target_row_and_order_consistency",
                            "cross_frame_or_lidar_corroboration",
                        ],
                        "deeper_vehicle_overlap": (
                            "not_target_occupied; unknown_if_target_interior_occluded"
                        ),
                    },
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    manifest = {
        "schema_version": "parking-slot-depth-aware-left-camera-review/2.0",
        "prediction_blind": True,
        "part1_fields_used": [],
        "state_labels_used": [],
        "camera_origin_lidar_m": origin.tolist(),
        "camera_yaw_deg": args.camera_yaw_deg,
        "extrinsic_status": (
            "provisional multi-frame RGB/LiDAR edge and VLM consensus; "
            "not survey calibration"
        ),
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
                "camera_origin_lidar_m": origin.tolist(),
                "camera_yaw_deg": args.camera_yaw_deg,
                "output_dir": str(args.output_dir),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
