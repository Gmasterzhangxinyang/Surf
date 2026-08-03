"""Coarse map-bearing Camera field-of-view checks for ParkingAgent v2.

The check intentionally never projects a slot into image pixels. It uses the
reference-time ego pose as an approximate Camera pose and compares the target
centre/corner bearings with a configurable horizontal field of view. The
result is a routing hint for evidence collection, not proof of image content.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from .contracts import FovResult, FovVisibility


# The dataset's rectified forward-facing ZED stream is 1280 px wide with
# fx=527.525085 and cx=636.297913. The pinhole envelope is about 101.004
# degrees horizontally. The shared map renderer uses a symmetric wedge, so it
# receives half of the left-plus-right angle. Routing remains much tighter
# because this coarse check uses the t0 ego pose as a Camera-pose proxy rather
# than an audited pixel projection.
DEFAULT_CAMERA_IMAGE_WIDTH_PX = 1280.0
DEFAULT_CAMERA_FX_PX = 527.5250854492188
DEFAULT_CAMERA_CX_PX = 636.2979125976562
DEFAULT_NOMINAL_HALF_FOV_DEG = 0.5 * math.degrees(
    math.atan(DEFAULT_CAMERA_CX_PX / DEFAULT_CAMERA_FX_PX)
    + math.atan(
        (DEFAULT_CAMERA_IMAGE_WIDTH_PX - DEFAULT_CAMERA_CX_PX)
        / DEFAULT_CAMERA_FX_PX
    )
)
DEFAULT_HALF_FOV_DEG = 40.0
DEFAULT_YAW_UNCERTAINTY_DEG = 15.0
DEFAULT_MINIMUM_ROBUST_COVERAGE = 0.80
DEFAULT_PROXY_EXCLUSION_MARGIN_DEG = 15.0
# Edge-diagnostic-supported standard_camera_origin convention. This is used
# only to determine whether a mapped target can physically reach Camera pixels;
# it is never occupancy evidence or a target bounding box.
DEFAULT_CAMERA_ORIGIN_FORWARD_M = 0.60
DEFAULT_CAMERA_PIXEL_MARGIN_PX = 1.0


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


def _xy(value: Sequence[float], name: str) -> tuple[float, float]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or len(value) != 2:
        raise ValueError(f"{name} must contain exactly two finite coordinates")
    return (
        _finite_number(value[0], f"{name}[0]"),
        _finite_number(value[1], f"{name}[1]"),
    )


def _polygon(value: Sequence[Sequence[float]]) -> tuple[tuple[float, float], ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError("target_polygon_map must be a coordinate sequence")
    result = tuple(_xy(point, f"target_polygon_map[{index}]") for index, point in enumerate(value))
    if len(result) < 3:
        raise ValueError("target_polygon_map must contain at least three corners")
    return result


def _normalise_angle_deg(value: float) -> float:
    return (float(value) + 180.0) % 360.0 - 180.0


def _bearing_deg(
    ego_xy: tuple[float, float],
    ego_yaw_rad: float,
    point_xy: tuple[float, float],
) -> float:
    absolute = math.degrees(
        math.atan2(point_xy[1] - ego_xy[1], point_xy[0] - ego_xy[0])
    )
    return _normalise_angle_deg(absolute - math.degrees(ego_yaw_rad))


def _centre(points: Sequence[tuple[float, float]]) -> tuple[float, float]:
    return (
        sum(point[0] for point in points) / len(points),
        sum(point[1] for point in points) / len(points),
    )


def evaluate_map_bearing_fov(
    *,
    ego_map_xy: Sequence[float],
    ego_yaw_rad: float,
    target_polygon_map: Sequence[Sequence[float]],
    target_center_map: Sequence[float] | None = None,
    map_units_per_meter: float | None = None,
    half_fov_deg: float = DEFAULT_HALF_FOV_DEG,
    yaw_uncertainty_deg: float = DEFAULT_YAW_UNCERTAINTY_DEG,
    minimum_robust_coverage: float = DEFAULT_MINIMUM_ROBUST_COVERAGE,
    camera_frame_id: int | None = None,
) -> FovResult:
    """Classify slot visibility using only map bearings at the t0 pose.

    robust_inside_fraction counts centre/corner samples that stay inside after
    the complete yaw-error margin. possible_inside_fraction counts samples that
    could be inside for some yaw within that margin.
    """

    ego = _xy(ego_map_xy, "ego_map_xy")
    yaw = _finite_number(ego_yaw_rad, "ego_yaw_rad")
    polygon = _polygon(target_polygon_map)
    center = _centre(polygon) if target_center_map is None else _xy(
        target_center_map, "target_center_map"
    )
    half_fov = _finite_number(half_fov_deg, "half_fov_deg")
    yaw_uncertainty = _finite_number(yaw_uncertainty_deg, "yaw_uncertainty_deg")
    minimum_coverage = _finite_number(
        minimum_robust_coverage, "minimum_robust_coverage"
    )
    if not 0.0 < half_fov <= 180.0:
        raise ValueError("half_fov_deg must be within (0, 180]")
    if not 0.0 <= yaw_uncertainty < half_fov:
        raise ValueError("yaw_uncertainty_deg must be non-negative and below half_fov_deg")
    if not 0.0 < minimum_coverage <= 1.0:
        raise ValueError("minimum_robust_coverage must be within (0, 1]")

    scale: float | None = None
    if map_units_per_meter is not None:
        scale = _finite_number(map_units_per_meter, "map_units_per_meter")
        if scale <= 0.0:
            raise ValueError("map_units_per_meter must be positive")

    samples = (center, *polygon)
    bearings = tuple(_bearing_deg(ego, yaw, point) for point in samples)
    robust = tuple(
        abs(bearing) + yaw_uncertainty <= half_fov + 1e-9 for bearing in bearings
    )
    possible = tuple(
        max(0.0, abs(bearing) - yaw_uncertainty) <= half_fov + 1e-9
        for bearing in bearings
    )
    robust_fraction = sum(robust) / len(robust)
    possible_fraction = sum(possible) / len(possible)

    if robust[0] and robust_fraction >= minimum_coverage:
        visibility = FovVisibility.VISIBLE
        coarse_status = "likely_visible"
        confidence = robust_fraction
        reasons = [
            "target_center_robustly_inside_fov",
            "target_samples_mostly_inside_fov",
        ]
    elif robust_fraction > 0.0:
        visibility = FovVisibility.PARTIALLY_VISIBLE
        coarse_status = "partially_visible"
        confidence = max(robust_fraction, min(possible_fraction, 0.85))
        reasons = [
            "some_target_samples_inside_fov",
            "target_not_fully_robust_to_fov_boundary",
        ]
    elif possible[0] or possible_fraction > 0.0:
        visibility = FovVisibility.UNCERTAIN
        coarse_status = "uncertain"
        confidence = 0.5
        reasons = [
            "target_intersects_fov_uncertainty_band",
            "ego_pose_proxy_cannot_resolve_fov_membership",
        ]
    else:
        minimum_absolute_bearing = min(abs(value) for value in bearings)
        hard_exclusion_bearing = min(
            180.0,
            half_fov + yaw_uncertainty + DEFAULT_PROXY_EXCLUSION_MARGIN_DEG,
        )
        if minimum_absolute_bearing >= hard_exclusion_bearing:
            visibility = FovVisibility.NOT_VISIBLE
            coarse_status = "outside"
            confidence = min(0.80, 1.0 - possible_fraction)
            reasons = ["all_target_samples_well_outside_conservative_fov"]
        else:
            visibility = FovVisibility.UNCERTAIN
            coarse_status = "uncertain"
            confidence = 0.35
            reasons = [
                "target_outside_nominal_fov_but_inside_proxy_exclusion_margin",
                "ego_pose_proxy_cannot_safely_forbid_camera",
            ]

    dx = center[0] - ego[0]
    dy = center[1] - ego[1]
    distance_map = math.hypot(dx, dy)
    distance_m = None if scale is None else distance_map / scale
    limitations = [
        "ego_pose_used_as_camera_pose_proxy",
        "map_bearing_only_no_pixel_projection",
        "does_not_measure_image_occlusion_or_camera_content",
    ]
    details: dict[str, Any] = {
        "coarse_status": coarse_status,
        "camera_candidate": visibility is not FovVisibility.NOT_VISIBLE,
        "target_bearing_deg": round(bearings[0], 6),
        "bearing_sign_convention": "positive_is_camera_left",
        "target_distance_map_units": round(distance_map, 9),
        "target_distance_m": None if distance_m is None else round(distance_m, 6),
        "half_fov_deg": half_fov,
        "yaw_uncertainty_deg": yaw_uncertainty,
        "proxy_exclusion_margin_deg": DEFAULT_PROXY_EXCLUSION_MARGIN_DEG,
        "sample_bearings_deg": [round(value, 6) for value in bearings],
        "robust_inside_fraction": round(robust_fraction, 6),
        "possible_inside_fraction": round(possible_fraction, 6),
        "reason_codes": reasons,
        "proxy_limitations": limitations,
    }
    return FovResult(
        visibility=visibility,
        confidence=max(0.0, min(float(confidence), 1.0)),
        reason=f"coarse_map_bearing_{coarse_status}",
        camera_frame_id=camera_frame_id,
        candidate_regions=[],
        details=details,
    )



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
    pixel_margin_px: float = DEFAULT_CAMERA_PIXEL_MARGIN_PX,
    minimum_visible_sample_fraction: float = DEFAULT_MINIMUM_ROBUST_COVERAGE,
    camera_frame_id: int | None = None,
) -> FovResult:
    """Hard Camera-tool gate from an audited horizontal pixel projection.

    The map target remains map geometry. Only the centre/corner horizontal pixel
    reachability is evaluated using the ZED intrinsics and the edge-diagnostic-
    supported camera origin. A centre outside the image is intentionally treated
    as not usable even if one bay corner clips the image edge: the Agent cannot
    reliably identify that slot from such a fragment.
    """

    ego = _xy(ego_map_xy, "ego_map_xy")
    yaw = _finite_number(ego_yaw_rad, "ego_yaw_rad")
    polygon = _polygon(target_polygon_map)
    center = _centre(polygon) if target_center_map is None else _xy(
        target_center_map, "target_center_map"
    )
    scale = _finite_number(map_units_per_meter, "map_units_per_meter")
    fx = _finite_number(camera_fx_px, "camera_fx_px")
    cx = _finite_number(camera_cx_px, "camera_cx_px")
    width = _finite_number(camera_width_px, "camera_width_px")
    origin_forward = _finite_number(
        camera_origin_forward_m, "camera_origin_forward_m"
    )
    margin = _finite_number(pixel_margin_px, "pixel_margin_px")
    minimum_fraction = _finite_number(
        minimum_visible_sample_fraction, "minimum_visible_sample_fraction"
    )
    if scale <= 0.0 or fx <= 0.0 or width <= 0.0:
        raise ValueError("map scale and Camera intrinsics must be positive")
    if not 0.0 <= cx <= width or not 0.0 <= margin < width / 2.0:
        raise ValueError("Camera principal point or pixel margin is invalid")
    if not 0.0 < minimum_fraction <= 1.0:
        raise ValueError("minimum_visible_sample_fraction must be within (0,1]")

    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    samples = (center, *polygon)
    sample_rows: list[dict[str, float | bool | None]] = []
    bearings: list[float] = []
    inside: list[bool] = []
    for point in samples:
        dx_m = (point[0] - ego[0]) / scale
        dy_m = (point[1] - ego[1]) / scale
        forward_m = cos_yaw * dx_m + sin_yaw * dy_m
        left_m = -sin_yaw * dx_m + cos_yaw * dy_m
        depth_m = forward_m - origin_forward
        bearing_deg = _normalise_angle_deg(math.degrees(math.atan2(left_m, forward_m)))
        bearings.append(bearing_deg)
        pixel_u = None if depth_m <= 1e-6 else fx * (-left_m) / depth_m + cx
        is_inside = bool(
            pixel_u is not None
            and margin <= pixel_u < width - margin
        )
        inside.append(is_inside)
        sample_rows.append(
            {
                "forward_m": round(forward_m, 6),
                "left_m": round(left_m, 6),
                "camera_depth_m": round(depth_m, 6),
                "pixel_u": None if pixel_u is None else round(pixel_u, 3),
                "inside_horizontal_image": is_inside,
            }
        )

    inside_fraction = sum(inside) / len(inside)
    center_inside = inside[0]
    if center_inside and inside_fraction >= minimum_fraction:
        visibility = FovVisibility.VISIBLE
        status = "visible"
        confidence = max(inside_fraction, 0.90)
        reasons = [
            "target_center_projects_inside_camera",
            "target_samples_mostly_project_inside_camera",
        ]
    elif center_inside:
        visibility = FovVisibility.PARTIALLY_VISIBLE
        status = "partially_visible"
        confidence = max(0.60, inside_fraction)
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

    nominal_half = DEFAULT_NOMINAL_HALF_FOV_DEG
    details: dict[str, Any] = {
        "coarse_status": status,
        "projection_status": status,
        "camera_candidate": visibility in {
            FovVisibility.VISIBLE,
            FovVisibility.PARTIALLY_VISIBLE,
        },
        "target_bearing_deg": round(bearings[0], 6),
        "bearing_sign_convention": "positive_is_camera_left",
        "half_fov_deg": nominal_half,
        "yaw_uncertainty_deg": 0.0,
        "sample_bearings_deg": [round(value, 6) for value in bearings],
        "horizontal_inside_fraction": round(inside_fraction, 6),
        "robust_inside_fraction": round(inside_fraction, 6),
        "possible_inside_fraction": round(inside_fraction, 6),
        "sample_horizontal_projection": sample_rows,
        "target_center_pixel_u": sample_rows[0]["pixel_u"],
        "camera_intrinsics": {"fx_px": fx, "cx_px": cx, "width_px": width},
        "camera_origin_forward_m": origin_forward,
        "pixel_margin_px": margin,
        "reason_codes": reasons,
        "projection_role": "camera_tool_feasibility_only",
        "extrinsic_status": "edge_diagnostic_supported_not_survey_calibration",
        "proxy_limitations": [
            "horizontal_projection_only",
            "does_not_establish_pixel_bbox_or_occupancy",
            "does_not_measure_image_occlusion_or_camera_content",
        ],
    }
    return FovResult(
        visibility=visibility,
        confidence=max(0.0, min(float(confidence), 1.0)),
        reason=f"horizontal_projection_{status}",
        camera_frame_id=camera_frame_id,
        candidate_regions=[],
        details=details,
    )


def result_from_mapping(value: Mapping[str, Any]) -> FovResult:
    """Rehydrate a stored shared-contract FOV result."""

    return FovResult.from_dict(value)


__all__ = [
    "DEFAULT_CAMERA_CX_PX",
    "DEFAULT_CAMERA_ORIGIN_FORWARD_M",
    "DEFAULT_CAMERA_PIXEL_MARGIN_PX",
    "DEFAULT_CAMERA_FX_PX",
    "DEFAULT_CAMERA_IMAGE_WIDTH_PX",
    "DEFAULT_HALF_FOV_DEG",
    "DEFAULT_NOMINAL_HALF_FOV_DEG",
    "DEFAULT_MINIMUM_ROBUST_COVERAGE",
    "DEFAULT_PROXY_EXCLUSION_MARGIN_DEG",
    "DEFAULT_YAW_UNCERTAINTY_DEG",
    "FovResult",
    "FovVisibility",
    "evaluate_map_bearing_fov",
    "evaluate_horizontal_projection_fov",
    "result_from_mapping",
]
