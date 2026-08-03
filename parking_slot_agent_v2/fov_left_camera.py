"""Left-offset Camera FOV routing for the autonomous ParkingAgent.

This keeps the original conservative map-bearing API, but the hard horizontal
pixel gate uses the measured Camera origin in LiDAR coordinates.
"""

from __future__ import annotations

import math
from typing import Sequence

from .contracts import FovResult, FovVisibility
from .fov import (
    DEFAULT_CAMERA_CX_PX,
    DEFAULT_CAMERA_FX_PX,
    DEFAULT_CAMERA_IMAGE_WIDTH_PX,
    DEFAULT_CAMERA_PIXEL_MARGIN_PX,
    DEFAULT_HALF_FOV_DEG,
    DEFAULT_MINIMUM_ROBUST_COVERAGE,
    DEFAULT_NOMINAL_HALF_FOV_DEG,
    evaluate_map_bearing_fov,
)

DEFAULT_CAMERA_ORIGIN_FORWARD_M = 0.60
DEFAULT_CAMERA_ORIGIN_LEFT_M = 0.36
DEFAULT_CAMERA_YAW_LEFT_DEG = 0.0


def _finite(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _xy(value: Sequence[float], name: str) -> tuple[float, float]:
    if len(value) != 2:
        raise ValueError(f"{name} must contain two values")
    return _finite(value[0], name), _finite(value[1], name)


def evaluate_horizontal_projection_fov(
    *,
    ego_map_xy: Sequence[float],
    ego_yaw_rad: float,
    target_polygon_map: Sequence[Sequence[float]],
    target_center_map: Sequence[float] | None = None,
    map_units_per_meter: float,
    camera_fx_px: float = DEFAULT_CAMERA_FX_PX,
    camera_cx_px: float = DEFAULT_CAMERA_CX_PX,
    camera_width_px: float = DEFAULT_CAMERA_IMAGE_WIDTH_PX,
    camera_origin_forward_m: float = DEFAULT_CAMERA_ORIGIN_FORWARD_M,
    camera_origin_left_m: float = DEFAULT_CAMERA_ORIGIN_LEFT_M,
    camera_yaw_left_deg: float = DEFAULT_CAMERA_YAW_LEFT_DEG,
    pixel_margin_px: float = DEFAULT_CAMERA_PIXEL_MARGIN_PX,
    minimum_visible_sample_fraction: float = DEFAULT_MINIMUM_ROBUST_COVERAGE,
    camera_frame_id: int | None = None,
) -> FovResult:
    ego = _xy(ego_map_xy, "ego_map_xy")
    yaw = _finite(ego_yaw_rad, "ego_yaw_rad")
    polygon = tuple(_xy(point, "target_polygon_map") for point in target_polygon_map)
    if len(polygon) < 3:
        raise ValueError("target polygon requires at least three corners")
    center = (
        (
            sum(point[0] for point in polygon) / len(polygon),
            sum(point[1] for point in polygon) / len(polygon),
        )
        if target_center_map is None
        else _xy(target_center_map, "target_center_map")
    )
    scale = _finite(map_units_per_meter, "map_units_per_meter")
    fx = _finite(camera_fx_px, "camera_fx_px")
    cx = _finite(camera_cx_px, "camera_cx_px")
    width = _finite(camera_width_px, "camera_width_px")
    origin_forward = _finite(camera_origin_forward_m, "camera_origin_forward_m")
    origin_left = _finite(camera_origin_left_m, "camera_origin_left_m")
    camera_yaw = math.radians(_finite(camera_yaw_left_deg, "camera_yaw_left_deg"))
    margin = _finite(pixel_margin_px, "pixel_margin_px")
    minimum = _finite(
        minimum_visible_sample_fraction, "minimum_visible_sample_fraction"
    )
    if scale <= 0 or fx <= 0 or width <= 0 or not 0 < minimum <= 1:
        raise ValueError("invalid Camera projection parameters")

    cy, sy = math.cos(yaw), math.sin(yaw)
    cc, sc = math.cos(camera_yaw), math.sin(camera_yaw)
    samples = (center, *polygon)
    rows = []
    bearings = []
    inside = []
    for point in samples:
        dx = (point[0] - ego[0]) / scale
        dy = (point[1] - ego[1]) / scale
        ego_forward = cy * dx + sy * dy
        ego_left = -sy * dx + cy * dy
        relative_forward = ego_forward - origin_forward
        relative_left = ego_left - origin_left
        camera_forward = cc * relative_forward + sc * relative_left
        camera_left = -sc * relative_forward + cc * relative_left
        pixel_u = (
            None
            if camera_forward <= 1e-6
            else fx * (-camera_left) / camera_forward + cx
        )
        is_inside = pixel_u is not None and margin <= pixel_u < width - margin
        bearing = math.degrees(math.atan2(camera_left, camera_forward))
        bearings.append(bearing)
        inside.append(bool(is_inside))
        rows.append(
            {
                "ego_forward_m": round(ego_forward, 6),
                "ego_left_m": round(ego_left, 6),
                "camera_forward_m": round(camera_forward, 6),
                "camera_left_m": round(camera_left, 6),
                "pixel_u": None if pixel_u is None else round(pixel_u, 3),
                "inside_horizontal_image": bool(is_inside),
            }
        )

    fraction = sum(inside) / len(inside)
    if inside[0] and fraction >= minimum:
        visibility = FovVisibility.VISIBLE
        status = "visible"
        confidence = max(fraction, 0.90)
        reasons = [
            "target_center_projects_inside_camera",
            "target_samples_mostly_project_inside_camera",
        ]
    elif inside[0]:
        visibility = FovVisibility.PARTIALLY_VISIBLE
        status = "partially_visible"
        confidence = max(0.60, fraction)
        reasons = [
            "target_center_projects_inside_camera",
            "target_polygon_partially_clipped_by_image",
        ]
    else:
        visibility = FovVisibility.NOT_VISIBLE
        status = "outside"
        confidence = 0.95 if not any(inside) else 0.85
        reasons = [
            "target_center_projects_outside_camera",
            "camera_tools_forbidden_without_target_center_reachability",
        ]

    details = {
        "coarse_status": status,
        "projection_status": status,
        "camera_candidate": visibility is not FovVisibility.NOT_VISIBLE,
        "target_bearing_deg": round(bearings[0], 6),
        "bearing_sign_convention": "positive_is_camera_left",
        "half_fov_deg": DEFAULT_NOMINAL_HALF_FOV_DEG,
        "yaw_uncertainty_deg": 0.0,
        "sample_bearings_deg": [round(x, 6) for x in bearings],
        "horizontal_inside_fraction": round(fraction, 6),
        "robust_inside_fraction": round(fraction, 6),
        "possible_inside_fraction": round(fraction, 6),
        "sample_horizontal_projection": rows,
        "target_center_pixel_u": rows[0]["pixel_u"],
        "camera_intrinsics": {"fx_px": fx, "cx_px": cx, "width_px": width},
        "camera_origin_lidar_m": [origin_forward, origin_left],
        "camera_yaw_left_deg": math.degrees(camera_yaw),
        "pixel_margin_px": margin,
        "reason_codes": reasons,
        "projection_role": "camera_tool_feasibility_only",
        "extrinsic_status": "multi-frame_edge_and_visual_consensus_not_survey",
        "proxy_limitations": [
            "horizontal_projection_only",
            "does_not_establish_pixel_bbox_or_occupancy",
            "does_not_measure_image_occlusion_or_camera_content",
        ],
    }
    return FovResult(
        visibility=visibility,
        confidence=max(0.0, min(float(confidence), 1.0)),
        reason=f"left_camera_horizontal_projection_{status}",
        camera_frame_id=camera_frame_id,
        candidate_regions=[],
        details=details,
    )


__all__ = [
    "DEFAULT_HALF_FOV_DEG",
    "DEFAULT_NOMINAL_HALF_FOV_DEG",
    "DEFAULT_CAMERA_ORIGIN_FORWARD_M",
    "DEFAULT_CAMERA_ORIGIN_LEFT_M",
    "DEFAULT_CAMERA_YAW_LEFT_DEG",
    "evaluate_horizontal_projection_fov",
    "evaluate_map_bearing_fov",
]
