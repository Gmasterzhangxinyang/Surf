#!/usr/bin/env python3
"""Run LiDAR foreground filtering inside the projected polygon, not its bbox."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from matplotlib.path import Path as MplPath
from PIL import Image

import filter_camera_views_by_lidar_depth as base
from audit_camera_extrinsic import project
from build_independent_gt_camera_views import _ground_height, _local_xy
from calibrate_left_camera_extrinsic import _camera_points


def polygon_foreground_audit(
    row,
    polygon_map,
    scale,
    origin,
    yaw_deg,
    *,
    depth_margin_m,
    pixel_margin,
    min_points,
):
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
    # Matplotlib's positive radius expands the closed polygon by roughly the
    # requested pixel diameter. This keeps the test target-local while allowing
    # for sparse returns and small calibration errors.
    polygon = MplPath(target_uv, closed=True)
    inside = polygon.contains_points(uv, radius=float(pixel_margin))
    foreground = (
        valid
        & inside
        & (camera[:, 2] >= 0.5)
        & (camera[:, 2] <= target_depth - depth_margin_m)
    )
    count = int(np.sum(foreground))
    if count:
        depth_gap = target_depth - camera[foreground, 2]
        median_gap = float(np.median(depth_gap))
        span_x = float(np.ptp(uv[foreground, 0]))
        span_y = float(np.ptp(uv[foreground, 1]))
    else:
        median_gap = span_x = span_y = 0.0
    extended = span_x >= 16.0 or span_y >= 12.0
    blocked = count >= min_points and extended
    return blocked, {
        "reason": "foreground_occluded" if blocked else "foreground_clear",
        "foreground_point_count": count,
        "foreground_median_depth_gap_m": median_gap,
        "foreground_pixel_span": [span_x, span_y],
        "target_depth_m": target_depth,
        "depth_margin_m": depth_margin_m,
        "polygon_expansion_px": pixel_margin,
        "target_polygon_uv": target_uv.tolist(),
    }, target_uv


if __name__ == "__main__":
    base.foreground_audit = polygon_foreground_audit
    base.main()
