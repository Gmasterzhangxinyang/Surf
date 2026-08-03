#!/usr/bin/env python3
"""Foreground occlusion audit using an elevated target viewing column."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

import filter_camera_views_by_lidar_depth as base
from audit_camera_extrinsic import project
from build_independent_gt_camera_views import _ground_height, _local_xy
from calibrate_left_camera_extrinsic import _camera_points


def elevated_foreground_audit(
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

    raw = np.fromfile(row["lidar_path"], dtype=np.float32).reshape(-1, 4)[:, :3]
    raw = raw[np.isfinite(raw).all(axis=1)]
    camera = _camera_points(raw, origin_lidar_m=origin, yaw_deg=yaw_deg)
    uv, valid = project(camera, image.width, image.height)
    target_depth = float(np.median(target_camera[:, 2]))
    # The bay polygon lies on the ground and is only a thin image strip. An
    # occluding wall/car occupies the viewing column above that strip.
    x1 = max(0.0, float(target_uv[:, 0].min()) - pixel_margin)
    x2 = min(float(image.width - 1), float(target_uv[:, 0].max()) + pixel_margin)
    y1 = max(0.0, float(target_uv[:, 1].min()) - 150.0)
    y2 = min(float(image.height - 1), float(target_uv[:, 1].max()) + 8.0)
    elevated_surface = raw[:, 2] >= -0.75
    foreground = (
        valid
        & elevated_surface
        & (uv[:, 0] >= x1)
        & (uv[:, 0] <= x2)
        & (uv[:, 1] >= y1)
        & (uv[:, 1] <= y2)
        & (camera[:, 2] >= 0.5)
        & (camera[:, 2] <= target_depth - depth_margin_m)
    )
    count = int(np.sum(foreground))
    if count:
        gap = target_depth - camera[foreground, 2]
        median_gap = float(np.median(gap))
        span_x = float(np.ptp(uv[foreground, 0]))
        span_y = float(np.ptp(uv[foreground, 1]))
        z_span = float(np.ptp(raw[foreground, 2]))
    else:
        median_gap = span_x = span_y = z_span = 0.0
    extended = span_y >= 18.0 and (span_x >= 18.0 or z_span >= 0.45)
    blocked = count >= min_points and extended
    return blocked, {
        "reason": "foreground_occluded" if blocked else "foreground_clear",
        "foreground_point_count": count,
        "foreground_median_depth_gap_m": median_gap,
        "foreground_pixel_span": [span_x, span_y],
        "foreground_lidar_z_span_m": z_span,
        "target_depth_m": target_depth,
        "depth_margin_m": depth_margin_m,
        "elevated_pixel_region": [x1, y1, x2, y2],
        "lidar_z_min_m": -0.75,
    }, target_uv


if __name__ == "__main__":
    base.foreground_audit = elevated_foreground_audit
    base.main()
