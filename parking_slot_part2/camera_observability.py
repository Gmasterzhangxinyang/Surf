"""Deterministic pre-call Camera usability decision for Part2.

Only Part1 map geometry and slot priors are consumed.  This module never
reads Camera detections and never predicts a final parking-slot state.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
import math
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from parking_slot_box_scoring.geometry import points_in_polygon
from parking_slot_hybrid_3d.camera import relative_bearing_deg

from .camera_calibration import (
    CalibrationError,
    CameraCalibration,
    camera_calibration_from_mapping,
)


CAMERA_OBSERVABILITY_SCHEMA_VERSION = "camera-use-decision/2.1"
CAMERA_OBSERVABILITY_POLICY_VERSION = "camera-use-policy/3.0"

CAMERA_USE_DECISIONS = frozenset(
    {"use_camera", "do_not_use_camera", "insufficient_information"}
)
CAMERA_REASON_CODES = frozenset(
    {
        "target_inside_reliable_fov",
        "line_of_sight_mostly_clear",
        "target_outside_fov",
        "target_in_unreliable_edge_region",
        "insufficient_reliable_fov_coverage",
        "explicit_line_of_sight_blockage",
        "unresolved_potential_occluder",
        "camera_pose_missing",
        "camera_calibration_missing",
        "target_geometry_missing",
        "occlusion_information_missing",
        "camera_fov_missing",
        "camera_fov_policy_mismatch",
        "camera_quality_curve_invalid",
        "target_out_of_route",
        "insufficient_line_of_sight_clearance",
        "invalid_input",
        "invalid_camera_observability_config",
        "static_obstacle_layer_missing",
        "static_obstacle_region_incomplete",
        "camera_intrinsics_missing",
        "camera_extrinsics_missing",
        "camera_pose_stale",
        "camera_pose_uncertainty_too_high",
        "coordinate_frame_mismatch",
        "static_obstacle_blocks_view",
    }
)

_STATE_ALIASES: dict[str, str | None] = {
    "occupied": "occupied",
    "free": "free",
    "unknown": "unknown",
    "partial": "partial",
    "partial_route": "partial",
    "partial_route_scope": "partial",
    "out_of_route": "out_of_route",
    "out_of_route_scope": "out_of_route",
    "not_evaluated": None,
    "unavailable": None,
}
_EXPLICIT_OBJECT_TYPES = frozenset(
    {
        "vehicle",
        "wall",
        "pillar",
        "column",
        "obstacle",
        "static_obstacle",
        "other_static",
    }
)
_STATIC_OBJECT_TYPES = frozenset(
    {
        "wall",
        "pillar",
        "column",
        "obstacle",
        "static_obstacle",
        "other_static",
        "barrier",
        "bollard",
        "curb",
        "arrester",
        "wheel_stop",
        "elevator",
        "building",
        "fence",
        "gate",
        "structure",
    }
)
_EPSILON = 1e-9


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


def _optional_finite(value: Any, name: str) -> float | None:
    return None if value is None else _finite(value, name)


def _ratio(value: Any, name: str) -> float:
    result = _finite(value, name)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be within [0, 1]")
    return result


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _point(value: Any, name: str) -> tuple[float, float] | None:
    if value is None:
        return None
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or len(value) != 2
    ):
        raise ValueError(f"{name} must contain x and y")
    return (_finite(value[0], name), _finite(value[1], name))


def _polygon(value: Any, name: str) -> tuple[tuple[float, float], ...] | None:
    if value is None:
        return None
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must be an array of 2D points")
    points: list[tuple[float, float]] = []
    for raw in value:
        point = _point(raw, name)
        if point is None:
            raise ValueError(f"{name} contains an empty point")
        points.append(point)
    if len(points) < 3:
        raise ValueError(f"{name} must contain at least three points")
    return tuple(points)


def _normalise_state(value: Any) -> str | None:
    if value is None:
        return None
    key = str(value).strip().lower()
    if not key:
        return None
    if key not in _STATE_ALIASES:
        raise ValueError(f"unsupported Part1 slot state: {value}")
    return _STATE_ALIASES[key]


def _first(
    sources: Iterable[Mapping[str, Any] | None],
    names: Sequence[str],
) -> Any:
    for source in sources:
        if source is None:
            continue
        for name in names:
            if name in source and source[name] is not None:
                return source[name]
    return None


def _quality_curve(
    value: Any,
    nominal_half_fov_deg: float,
) -> tuple[tuple[float, float], ...] | None:
    if value is None:
        return None
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError("angle_to_quality_curve must be an array")
    parsed: list[tuple[float, float]] = []
    for raw in value:
        if isinstance(raw, Mapping):
            angle = raw.get("angle_deg", raw.get("absolute_angle_deg"))
            quality = raw.get("quality", raw.get("quality_score"))
        elif (
            isinstance(raw, Sequence)
            and not isinstance(raw, (str, bytes))
            and len(raw) == 2
        ):
            angle, quality = raw
        else:
            raise ValueError("invalid angle_to_quality_curve point")
        angle_value = _finite(angle, "angle_to_quality_curve angle")
        if not 0.0 <= angle_value <= nominal_half_fov_deg:
            raise ValueError("quality-curve angle is outside the nominal FOV")
        parsed.append((angle_value, _ratio(quality, "quality-curve quality")))
    parsed.sort()
    if not parsed or any(
        parsed[index][0] <= parsed[index - 1][0]
        for index in range(1, len(parsed))
    ):
        raise ValueError("quality-curve angles must be non-empty and unique")
    if not math.isclose(parsed[0][0], 0.0, abs_tol=1e-6) or not math.isclose(
        parsed[-1][0], nominal_half_fov_deg, abs_tol=1e-6
    ):
        raise ValueError("quality curve must cover 0 through nominal half-FOV")
    return tuple(parsed)


@dataclass(frozen=True, slots=True)
class CameraObservabilityConfig:
    """Central Camera-use policy.

    The default zones are conservative configurable placeholders, not the
    final physical-Camera calibration.  A calibrated angle/quality curve wins
    over these zones when provided.
    """

    nominal_half_fov_deg: float = 90.0
    reliable_half_fov_deg: float = 60.0
    edge_unreliable_start_deg: float = 78.0
    minimum_reliable_fov_coverage: float = 0.60
    minimum_clear_ray_ratio: float = 0.60
    maximum_blocked_ray_ratio: float = 0.50
    occupied_probability_blocker_threshold: float = 0.70
    occupied_probability_potential_threshold: float = 0.30
    visibility_corridor_width_m: float = 0.30
    minimum_reliable_angular_quality: float = 0.60
    unreliable_angular_quality_threshold: float = 0.05
    minimum_edge_quality_score: float = 0.40
    include_edge_midpoints: bool = True
    angle_to_quality_curve: tuple[tuple[float, float], ...] | None = None
    maximum_camera_time_offset_ms: float = 50.0
    maximum_camera_position_std_m: float = 0.50
    maximum_camera_yaw_std_deg: float = 5.0
    maximum_camera_orientation_std_deg: float = 5.0
    pose_uncertainty_corridor_sigma: float = 2.0
    target_sample_heights_m: tuple[float, ...] = (0.20, 0.80, 1.40)
    vertical_occlusion_margin_m: float = 0.05
    allow_legacy_2d_test_contract: bool = False

    def __post_init__(self) -> None:
        nominal = _finite(self.nominal_half_fov_deg, "nominal_half_fov_deg")
        reliable = _finite(self.reliable_half_fov_deg, "reliable_half_fov_deg")
        edge = _finite(self.edge_unreliable_start_deg, "edge_unreliable_start_deg")
        if not 0.0 < reliable < edge <= nominal <= 180.0:
            raise ValueError("FOV zones must satisfy 0 < reliable < edge <= nominal")
        for name in (
            "minimum_reliable_fov_coverage",
            "minimum_clear_ray_ratio",
            "maximum_blocked_ray_ratio",
            "occupied_probability_blocker_threshold",
            "occupied_probability_potential_threshold",
            "minimum_reliable_angular_quality",
            "unreliable_angular_quality_threshold",
            "minimum_edge_quality_score",
        ):
            _ratio(getattr(self, name), name)
        if (
            self.occupied_probability_potential_threshold
            > self.occupied_probability_blocker_threshold
        ):
            raise ValueError("potential threshold must not exceed blocker threshold")
        if (
            self.unreliable_angular_quality_threshold
            >= self.minimum_reliable_angular_quality
        ):
            raise ValueError("unreliable quality must be below reliable quality")
        if _finite(self.visibility_corridor_width_m, "visibility_corridor_width_m") < 0:
            raise ValueError("visibility_corridor_width_m must be non-negative")
        for name in (
            "maximum_camera_time_offset_ms",
            "maximum_camera_position_std_m",
            "maximum_camera_yaw_std_deg",
            "maximum_camera_orientation_std_deg",
            "pose_uncertainty_corridor_sigma",
            "vertical_occlusion_margin_m",
        ):
            if _finite(getattr(self, name), name) < 0.0:
                raise ValueError(f"{name} must be non-negative")
        raw_heights = self.target_sample_heights_m
        if (
            not isinstance(raw_heights, Sequence)
            or isinstance(raw_heights, (str, bytes))
            or not raw_heights
        ):
            raise ValueError("target_sample_heights_m must be a non-empty array")
        heights = tuple(
            _finite(value, "target_sample_heights_m") for value in raw_heights
        )
        if any(value < 0.0 for value in heights) or any(
            heights[index] <= heights[index - 1]
            for index in range(1, len(heights))
        ):
            raise ValueError(
                "target_sample_heights_m must be non-negative and strictly increasing"
            )
        object.__setattr__(self, "target_sample_heights_m", heights)
        if not isinstance(self.include_edge_midpoints, bool):
            raise ValueError("include_edge_midpoints must be boolean")
        if not isinstance(self.allow_legacy_2d_test_contract, bool):
            raise ValueError("allow_legacy_2d_test_contract must be boolean")
        object.__setattr__(
            self,
            "angle_to_quality_curve",
            _quality_curve(self.angle_to_quality_curve, nominal),
        )

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "CameraObservabilityConfig":
        if not isinstance(payload, Mapping):
            raise ValueError("camera_observability_config must be a mapping")
        allowed = {item.name for item in fields(cls)}
        return cls(**{key: value for key, value in payload.items() if key in allowed})


@dataclass(frozen=True, slots=True)
class LineOfSightObject:
    """A Part1 slot or map obstacle represented by map-coordinate polygons.

    The legacy public name remains, but distance and overlap are now derived
    from polygons rather than trusted as caller-supplied scalars.
    """

    object_id: str
    polygon_map: tuple[tuple[float, float], ...] | None = None
    object_type: str = "slot"
    map_state: str | None = None
    scope_status: str | None = None
    occupied_probability: float | None = None
    occupancy_polygon_map: tuple[tuple[float, float], ...] | None = None
    center_map: tuple[float, float] | None = None
    heading_deg: float | None = None
    clearance_proven: bool = False
    min_z_m: float | None = None
    max_z_m: float | None = None
    confidence: float | None = None
    source: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "object_id", _text(self.object_id, "object_id"))
        object.__setattr__(self, "object_type", _text(self.object_type, "object_type").lower())
        object.__setattr__(self, "polygon_map", _polygon(self.polygon_map, "polygon_map"))
        object.__setattr__(
            self,
            "occupancy_polygon_map",
            _polygon(self.occupancy_polygon_map, "occupancy_polygon_map"),
        )
        object.__setattr__(self, "center_map", _point(self.center_map, "center_map"))
        object.__setattr__(self, "map_state", _normalise_state(self.map_state))
        scope = None if self.scope_status is None else str(self.scope_status).strip().lower()
        object.__setattr__(self, "scope_status", scope or None)
        probability = (
            None
            if self.occupied_probability is None
            else _ratio(self.occupied_probability, "occupied_probability")
        )
        object.__setattr__(self, "occupied_probability", probability)
        object.__setattr__(self, "heading_deg", _optional_finite(self.heading_deg, "heading_deg"))
        if not isinstance(self.clearance_proven, bool):
            raise ValueError("clearance_proven must be boolean")
        minimum_z = _optional_finite(self.min_z_m, "min_z_m")
        maximum_z = _optional_finite(self.max_z_m, "max_z_m")
        if (minimum_z is None) != (maximum_z is None):
            raise ValueError("min_z_m and max_z_m must be supplied together")
        if minimum_z is not None and maximum_z is not None and maximum_z <= minimum_z:
            raise ValueError("max_z_m must be greater than min_z_m")
        object.__setattr__(self, "min_z_m", minimum_z)
        object.__setattr__(self, "max_z_m", maximum_z)
        object.__setattr__(
            self,
            "confidence",
            None if self.confidence is None else _ratio(self.confidence, "confidence"),
        )
        source = None if self.source is None else str(self.source).strip()
        object.__setattr__(self, "source", source or None)

    @property
    def slot_id(self) -> str:
        return self.object_id

    @property
    def state(self) -> str | None:
        return self.map_state

    @classmethod
    def from_mapping(
        cls,
        payload: Mapping[str, Any],
        *,
        decision: Mapping[str, Any] | None = None,
        scope: Mapping[str, Any] | None = None,
        default_object_type: str = "slot",
    ) -> "LineOfSightObject":
        sources = (decision, payload, scope)
        clearance = _first((decision, payload), ("clearance_proven",))
        return cls(
            object_id=_first(sources, ("object_id", "slot_id", "id")),
            polygon_map=_first((payload,), ("polygon_map", "polygon_xy", "polygon")),
            object_type=_first((payload,), ("object_type", "type", "kind")) or default_object_type,
            map_state=_first((decision, payload), ("state", "map_state")),
            scope_status=_first(sources, ("scope_status",)),
            occupied_probability=_first(
                (decision, payload),
                ("occupied_probability", "occupancy_probability", "p_occupied"),
            ),
            occupancy_polygon_map=_first(
                (decision, payload),
                ("occupancy_polygon_map", "vehicle_polygon_map", "obstacle_polygon_map"),
            ),
            center_map=_first((payload,), ("center_map", "center")),
            heading_deg=_first((payload,), ("heading_deg",)),
            clearance_proven=False if clearance is None else clearance,
            min_z_m=_first(
                (payload,),
                ("min_z_m", "min_z", "min_height_m", "height_min_m"),
            ),
            max_z_m=_first(
                (payload,),
                ("max_z_m", "max_z", "max_height_m", "height_max_m"),
            ),
            confidence=_first((payload,), ("confidence",)),
            source=_first((payload,), ("source", "source_id")),
        )


def _records(value: Any, wrapper_key: str) -> tuple[Mapping[str, Any], ...]:
    if value is None:
        return ()
    if isinstance(value, Mapping) and wrapper_key in value:
        value = value[wrapper_key]
    elif isinstance(value, Mapping):
        if not all(isinstance(item, Mapping) for item in value.values()):
            raise ValueError(f"{wrapper_key} must contain mappings")
        value = tuple(value.values())
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{wrapper_key} must be an array")
    if not all(isinstance(item, Mapping) for item in value):
        raise ValueError(f"{wrapper_key} must contain mappings")
    return tuple(value)


def _record_index(rows: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        identifier = row.get("slot_id", row.get("object_id", row.get("id")))
        if identifier is not None:
            result[str(identifier)] = row
    return result


def _pose(value: Any) -> tuple[float, float, float] | None:
    if value is None:
        return None
    if isinstance(value, Mapping):
        value = (
            value.get("x", value.get("map_x")),
            value.get("y", value.get("map_y")),
            value.get("yaw_rad", value.get("yaw", value.get("map_yaw"))),
        )
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or len(value) != 3
    ):
        raise ValueError("camera_pose_map_xyyaw must contain x, y, yaw_rad")
    return tuple(_finite(item, "camera_pose_map_xyyaw") for item in value)  # type: ignore[return-value]


def _matrix4(value: Any, name: str) -> np.ndarray:
    try:
        matrix = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain numbers") from exc
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError(f"{name} must be a finite 4x4 matrix")
    if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1e-9):
        raise ValueError(f"{name} must be homogeneous")
    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-7) or not math.isclose(
        float(np.linalg.det(rotation)), 1.0, abs_tol=1e-7
    ):
        raise ValueError(f"{name} rotation must be right-handed and orthonormal")
    return matrix


def _pose_covariance(value: Any) -> tuple[float, ...] | None:
    if value is None:
        return None
    try:
        covariance = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("pose_covariance must contain numbers") from exc
    if covariance.shape == (6, 6):
        covariance = covariance.reshape(-1)
    if covariance.shape != (36,) or not np.isfinite(covariance).all():
        raise ValueError("pose_covariance must contain 36 finite values")
    matrix = covariance.reshape(6, 6)
    if not np.allclose(matrix, matrix.T, atol=1e-10):
        raise ValueError("pose_covariance must be symmetric")
    if float(np.min(np.linalg.eigvalsh(matrix))) < -1e-10:
        raise ValueError("pose_covariance must be positive semidefinite")
    return tuple(float(item) for item in covariance)


@dataclass(frozen=True, slots=True)
class _CameraPoseContract:
    pose_map_xyyaw: tuple[float, float, float] | None
    T_map_camera_m: tuple[tuple[float, ...], ...] | None
    position_z_m: float | None
    covariance: tuple[float, ...] | None
    localization_status: str | None
    time_offset_ms: float | None
    frame_id: str | None
    parent_frame: str | None
    calibration_id: str | None
    translation_unit: str | None
    map_units_per_meter: float | None


def _camera_pose_contract(
    value: Mapping[str, Any],
    scene_map_units_per_meter: float | None,
) -> _CameraPoseContract:
    if value.get("schema_version") != "camera-map-pose/1.0":
        raise ValueError("camera_map_pose schema_version is missing or unsupported")
    timestamp = value.get("timestamp_ns")
    if isinstance(timestamp, bool) or not isinstance(timestamp, int) or timestamp < 0:
        raise ValueError("camera_map_pose timestamp_ns must be a non-negative integer")
    status = str(value.get("localization_status", "")).strip().lower() or None
    frame_id = str(value.get("frame_id", "")).strip() or None
    parent_frame = str(value.get("parent_frame", "")).strip() or None
    calibration_id = str(value.get("calibration_id", "")).strip() or None
    unit = str(value.get("translation_unit", "")).strip() or None
    pose_scale = _optional_finite(value.get("map_units_per_meter"), "map_units_per_meter")
    offset = _optional_finite(value.get("time_offset_ms"), "time_offset_ms")
    raw_covariance = value.get("pose_covariance")
    covariance_missing = raw_covariance is None or (
        isinstance(raw_covariance, Sequence)
        and not isinstance(raw_covariance, (str, bytes))
        and len(raw_covariance) == 0
    )
    covariance = (
        None
        if covariance_missing
        else _pose_covariance(raw_covariance)
    )
    raw_transform = value.get("T_map_camera")
    transform_missing = raw_transform is None or (
        isinstance(raw_transform, Sequence)
        and not isinstance(raw_transform, (str, bytes))
        and len(raw_transform) == 0
    )
    if transform_missing:
        return _CameraPoseContract(
            None,
            None,
            None,
            covariance,
            status,
            offset,
            frame_id,
            parent_frame,
            calibration_id,
            unit,
            pose_scale,
        )
    transform = _matrix4(raw_transform, "T_map_camera")
    position = transform[:3, 3]
    raw_position = value.get("position_xyz")
    position_missing = raw_position is None or (
        isinstance(raw_position, Sequence)
        and not isinstance(raw_position, (str, bytes))
        and len(raw_position) == 0
    )
    if not position_missing:
        declared = np.asarray(raw_position, dtype=np.float64)
        if declared.shape != (3,) or not np.isfinite(declared).all():
            raise ValueError("position_xyz must contain three finite values")
        if not np.allclose(position, declared, atol=1e-8):
            raise ValueError("position_xyz disagrees with T_map_camera")
    else:
        raise ValueError("position_xyz is required for a Camera map pose")
    raw_orientation = value.get("orientation_xyzw")
    if raw_orientation is None:
        raise ValueError("orientation_xyzw is required for a Camera map pose")
    try:
        orientation = np.asarray(raw_orientation, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("orientation_xyzw must contain numbers") from exc
    if orientation.shape != (4,) or not np.isfinite(orientation).all():
        raise ValueError("orientation_xyzw must contain four finite values")
    norm = float(np.linalg.norm(orientation))
    if norm <= _EPSILON or not math.isclose(norm, 1.0, abs_tol=1e-6):
        raise ValueError("orientation_xyzw must be a unit quaternion")
    x_q, y_q, z_q, w_q = orientation / norm
    declared_rotation = np.asarray(
        [
            [
                1.0 - 2.0 * (y_q * y_q + z_q * z_q),
                2.0 * (x_q * y_q - z_q * w_q),
                2.0 * (x_q * z_q + y_q * w_q),
            ],
            [
                2.0 * (x_q * y_q + z_q * w_q),
                1.0 - 2.0 * (x_q * x_q + z_q * z_q),
                2.0 * (y_q * z_q - x_q * w_q),
            ],
            [
                2.0 * (x_q * z_q - y_q * w_q),
                2.0 * (y_q * z_q + x_q * w_q),
                1.0 - 2.0 * (x_q * x_q + y_q * y_q),
            ],
        ],
        dtype=np.float64,
    )
    if not np.allclose(declared_rotation, transform[:3, :3], atol=1e-7):
        raise ValueError("orientation_xyzw disagrees with T_map_camera")
    forward_xy = transform[:2, 2]
    if float(np.linalg.norm(forward_xy)) <= _EPSILON:
        raise ValueError("Camera optical forward axis has no horizontal map component")
    yaw = math.atan2(float(forward_xy[1]), float(forward_xy[0]))
    if unit == "m":
        if scene_map_units_per_meter is None:
            raise ValueError("map_units_per_meter is required for a metric Camera pose")
        x_map = float(position[0]) * scene_map_units_per_meter
        y_map = float(position[1]) * scene_map_units_per_meter
        z_m = float(position[2])
        metric_transform = transform.copy()
    elif unit == "map_unit":
        if pose_scale is None or scene_map_units_per_meter is None or not math.isclose(
            pose_scale,
            scene_map_units_per_meter,
            rel_tol=1e-9,
            abs_tol=1e-12,
        ):
            raise ValueError("Camera pose and slot map scales differ")
        x_map = float(position[0])
        y_map = float(position[1])
        z_m = float(position[2]) / pose_scale
        metric_transform = transform.copy()
        metric_transform[:3, 3] /= pose_scale
    else:
        raise ValueError("Camera map pose must declare translation_unit")
    return _CameraPoseContract(
        (x_map, y_map, yaw),
        tuple(tuple(float(item) for item in raw) for raw in metric_transform),
        z_m,
        covariance,
        status,
        offset,
        frame_id,
        parent_frame,
        calibration_id,
        unit,
        pose_scale,
    )


@dataclass(frozen=True, slots=True)
class CameraObservabilityInput:
    """Normalised pre-Camera scene.

    None means a layer is missing; an empty tuple means it was loaded and is
    known to contain no object.
    """

    target_slot_id: str = ""
    camera_pose_map_xyyaw: tuple[float, float, float] | None = None
    camera_calibration: Any | None = None
    nominal_horizontal_fov_deg: float | None = None
    map_units_per_meter: float | None = None
    slots: tuple[LineOfSightObject, ...] | None = None
    static_obstacles: tuple[LineOfSightObject, ...] | None = None
    camera_pose_contract_present: bool = False
    camera_T_map_camera_m: tuple[tuple[float, ...], ...] | None = None
    camera_position_z_m: float | None = None
    camera_pose_covariance: tuple[float, ...] | None = None
    camera_localization_status: str | None = None
    camera_time_offset_ms: float | None = None
    camera_frame_id: str | None = None
    camera_parent_frame: str | None = None
    camera_pose_calibration_id: str | None = None
    camera_pose_translation_unit: str | None = None
    map_coordinate_frame: str | None = None
    target_ground_z_m: float | None = None
    static_obstacle_query_complete: bool | None = None
    static_obstacle_layer_present: bool | None = None
    static_obstacle_coordinate_frame: str | None = None
    static_obstacle_map_units_per_meter: float | None = None
    static_obstacle_map_payload: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "target_slot_id", str(self.target_slot_id or "").strip())
        object.__setattr__(self, "camera_pose_map_xyyaw", _pose(self.camera_pose_map_xyyaw))
        fov = _optional_finite(self.nominal_horizontal_fov_deg, "nominal_horizontal_fov_deg")
        if fov is not None and not 0.0 < fov <= 360.0:
            raise ValueError("nominal_horizontal_fov_deg must be within (0, 360]")
        object.__setattr__(self, "nominal_horizontal_fov_deg", fov)
        scale = _optional_finite(self.map_units_per_meter, "map_units_per_meter")
        if scale is not None and scale <= 0.0:
            raise ValueError("map_units_per_meter must be positive")
        object.__setattr__(self, "map_units_per_meter", scale)
        if not isinstance(self.camera_pose_contract_present, bool):
            raise ValueError("camera_pose_contract_present must be boolean")
        if self.camera_T_map_camera_m is not None:
            transform = _matrix4(self.camera_T_map_camera_m, "camera_T_map_camera_m")
            object.__setattr__(
                self,
                "camera_T_map_camera_m",
                tuple(tuple(float(item) for item in raw) for raw in transform),
            )
        object.__setattr__(
            self,
            "camera_position_z_m",
            _optional_finite(self.camera_position_z_m, "camera_position_z_m"),
        )
        object.__setattr__(
            self,
            "camera_pose_covariance",
            _pose_covariance(self.camera_pose_covariance),
        )
        status = (
            None
            if self.camera_localization_status is None
            else str(self.camera_localization_status).strip().lower()
        )
        if status is not None and status not in {"valid", "degraded", "invalid"}:
            raise ValueError("camera_localization_status is invalid")
        object.__setattr__(self, "camera_localization_status", status)
        object.__setattr__(
            self,
            "camera_time_offset_ms",
            _optional_finite(self.camera_time_offset_ms, "camera_time_offset_ms"),
        )
        for name in (
            "camera_frame_id",
            "camera_parent_frame",
            "camera_pose_calibration_id",
            "camera_pose_translation_unit",
            "map_coordinate_frame",
            "static_obstacle_coordinate_frame",
        ):
            value = getattr(self, name)
            normalised = None if value is None else str(value).strip()
            object.__setattr__(self, name, normalised or None)
        object.__setattr__(
            self,
            "target_ground_z_m",
            _optional_finite(self.target_ground_z_m, "target_ground_z_m"),
        )
        for name in ("static_obstacle_query_complete", "static_obstacle_layer_present"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, bool):
                raise ValueError(f"{name} must be boolean or null")
        obstacle_scale = _optional_finite(
            self.static_obstacle_map_units_per_meter,
            "static_obstacle_map_units_per_meter",
        )
        if obstacle_scale is not None and obstacle_scale <= 0.0:
            raise ValueError("static_obstacle_map_units_per_meter must be positive")
        object.__setattr__(self, "static_obstacle_map_units_per_meter", obstacle_scale)
        if self.static_obstacle_map_payload is not None and not isinstance(
            self.static_obstacle_map_payload,
            Mapping,
        ):
            raise ValueError("static_obstacle_map_payload must be a mapping")
        for name in ("slots", "static_obstacles"):
            value = getattr(self, name)
            if value is not None:
                normalised = tuple(value)
                if not all(isinstance(item, LineOfSightObject) for item in normalised):
                    raise ValueError(f"{name} must contain LineOfSightObject values")
                object.__setattr__(self, name, normalised)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "CameraObservabilityInput":
        if not isinstance(payload, Mapping):
            raise ValueError("Camera input must be a mapping")
        database = payload.get("slot_database")
        if database is not None and not isinstance(database, Mapping):
            raise ValueError("slot_database must be a mapping")
        raw_slots = payload.get("slots")
        if raw_slots is None and isinstance(database, Mapping):
            raw_slots = database.get("slots")
        scale = payload.get("map_units_per_meter")
        if scale is None and isinstance(database, Mapping):
            scale = database.get("map_units_per_meter")
        parsed_scale = _optional_finite(scale, "map_units_per_meter")
        map_frame = _first(
            (payload, database),
            ("map_coordinate_frame", "map_frame_id", "coordinate_frame"),
        )

        raw_camera_contract = payload.get("camera_map_pose")
        if raw_camera_contract is None:
            possible_pose = payload.get("camera_pose")
            if isinstance(possible_pose, Mapping) and "T_map_camera" in possible_pose:
                raw_camera_contract = possible_pose
        if raw_camera_contract is not None and not isinstance(raw_camera_contract, Mapping):
            raise ValueError("camera_map_pose must be a mapping")
        pose_contract = (
            None
            if raw_camera_contract is None
            else _camera_pose_contract(raw_camera_contract, parsed_scale)
        )

        decision_index = _record_index(_records(payload.get("slot_decisions"), "decisions"))
        scope_index = _record_index(_records(payload.get("known_slot_scope"), "known_slot_scope"))
        slots: tuple[LineOfSightObject, ...] | None
        target_geometry_source: Mapping[str, Any] | None = None
        if raw_slots is None:
            slots = None
        else:
            if not isinstance(raw_slots, Sequence) or isinstance(raw_slots, (str, bytes)):
                raise ValueError("slots must be an array")
            parsed: list[LineOfSightObject] = []
            for raw in raw_slots:
                if isinstance(raw, LineOfSightObject):
                    parsed.append(raw)
                    continue
                if not isinstance(raw, Mapping):
                    raise ValueError("slots must contain mappings")
                identifier = str(raw.get("slot_id", raw.get("object_id", "")))
                if identifier == str(payload.get("target_slot_id", "")):
                    target_geometry_source = raw
                parsed.append(
                    LineOfSightObject.from_mapping(
                        raw,
                        decision=decision_index.get(identifier),
                        scope=scope_index.get(identifier),
                    )
                )
            slots = tuple(parsed)

        raw_query = payload.get("static_obstacle_query")
        if raw_query is not None and not isinstance(raw_query, Mapping):
            raise ValueError("static_obstacle_query must be a mapping")
        obstacle_source: Mapping[str, Any] = (
            raw_query if raw_query is not None else payload
        )
        obstacle_key = next(
            (
                key
                for key in ("static_obstacles", "map_obstacles", "obstacles")
                if key in obstacle_source
            ),
            None,
        )
        obstacles: tuple[LineOfSightObject, ...] | None
        if obstacle_key is None:
            obstacles = None
        else:
            raw_obstacles = obstacle_source[obstacle_key]
            if not isinstance(raw_obstacles, Sequence) or isinstance(raw_obstacles, (str, bytes)):
                raise ValueError(f"{obstacle_key} must be an array")
            parsed_obstacles: list[LineOfSightObject] = []
            for raw in raw_obstacles:
                if isinstance(raw, LineOfSightObject):
                    parsed_obstacles.append(raw)
                elif isinstance(raw, Mapping):
                    parsed_obstacles.append(
                        LineOfSightObject.from_mapping(raw, default_object_type="static_obstacle")
                    )
                else:
                    raise ValueError(f"{obstacle_key} must contain mappings")
            obstacles = tuple(parsed_obstacles)

        static_map_payload = payload.get(
            "static_obstacle_map",
            payload.get("static_obstacle_layer"),
        )
        if static_map_payload is None and "mapped_regions" in payload:
            static_map_payload = payload
        if static_map_payload is not None and not isinstance(static_map_payload, Mapping):
            raise ValueError("static_obstacle_map must be a mapping")
        query_complete = None
        layer_present = None
        obstacle_frame = None
        obstacle_scale = None
        if raw_query is not None:
            query_complete = raw_query.get(
                "local_coverage_complete",
                raw_query.get("static_obstacle_layer_complete"),
            )
            layer_present = raw_query.get("layer_present")
            obstacle_frame = raw_query.get("coordinate_frame")
            obstacle_scale = raw_query.get("map_units_per_meter")

        legacy_pose_value = payload.get("camera_pose_map_xyyaw")
        if legacy_pose_value is None and pose_contract is None:
            legacy_pose_value = payload.get("camera_pose")
        return cls(
            target_slot_id=payload.get("target_slot_id", ""),
            camera_pose_map_xyyaw=(
                pose_contract.pose_map_xyyaw
                if pose_contract is not None
                else _pose(legacy_pose_value)
            ),
            camera_calibration=payload.get("camera_calibration", payload.get("camera_model")),
            nominal_horizontal_fov_deg=payload.get("nominal_horizontal_fov_deg"),
            map_units_per_meter=scale,
            slots=slots,
            static_obstacles=obstacles,
            camera_pose_contract_present=pose_contract is not None,
            camera_T_map_camera_m=(
                None if pose_contract is None else pose_contract.T_map_camera_m
            ),
            camera_position_z_m=(
                None if pose_contract is None else pose_contract.position_z_m
            ),
            camera_pose_covariance=(
                None if pose_contract is None else pose_contract.covariance
            ),
            camera_localization_status=(
                None if pose_contract is None else pose_contract.localization_status
            ),
            camera_time_offset_ms=(
                None if pose_contract is None else pose_contract.time_offset_ms
            ),
            camera_frame_id=None if pose_contract is None else pose_contract.frame_id,
            camera_parent_frame=(
                None if pose_contract is None else pose_contract.parent_frame
            ),
            camera_pose_calibration_id=(
                None if pose_contract is None else pose_contract.calibration_id
            ),
            camera_pose_translation_unit=(
                None if pose_contract is None else pose_contract.translation_unit
            ),
            map_coordinate_frame=map_frame,
            target_ground_z_m=_first(
                (payload, target_geometry_source),
                ("target_ground_z_m", "ground_z_m", "floor_z_m"),
            ),
            static_obstacle_query_complete=query_complete,
            static_obstacle_layer_present=layer_present,
            static_obstacle_coordinate_frame=obstacle_frame,
            static_obstacle_map_units_per_meter=obstacle_scale,
            static_obstacle_map_payload=static_map_payload,
        )


@dataclass(frozen=True, slots=True)
class CameraObservabilityAssessment:
    """Concise Camera-use decision; never a parking-slot state."""

    decision: str
    target_slot_id: str
    target_bearing_deg: float | None = None
    target_distance_m: float | None = None
    nominal_fov_coverage: float = 0.0
    reliable_fov_coverage: float = 0.0
    edge_quality_score: float = 0.0
    clear_ray_ratio: float = 0.0
    blocked_ray_ratio: float = 0.0
    uncertain_ray_ratio: float = 0.0
    blocking_object_ids: tuple[str, ...] = ()
    potential_occluder_ids: tuple[str, ...] = ()
    reason_codes: tuple[str, ...] = ()
    camera_usable: bool = field(init=False)
    schema_version: str = CAMERA_OBSERVABILITY_SCHEMA_VERSION
    policy_version: str = CAMERA_OBSERVABILITY_POLICY_VERSION

    def __post_init__(self) -> None:
        if self.decision not in CAMERA_USE_DECISIONS:
            raise ValueError("unsupported Camera-use decision")
        object.__setattr__(self, "camera_usable", self.decision == "use_camera")
        for name in (
            "nominal_fov_coverage",
            "reliable_fov_coverage",
            "edge_quality_score",
            "clear_ray_ratio",
            "blocked_ray_ratio",
            "uncertain_ray_ratio",
        ):
            _ratio(getattr(self, name), name)
        object.__setattr__(self, "blocking_object_ids", tuple(sorted(set(self.blocking_object_ids))))
        object.__setattr__(
            self,
            "potential_occluder_ids",
            tuple(sorted(set(self.potential_occluder_ids))),
        )
        object.__setattr__(self, "reason_codes", tuple(dict.fromkeys(self.reason_codes)))

    def to_dict(self) -> dict[str, Any]:
        def concise(value: float | None) -> float | None:
            return None if value is None else round(float(value), 6)

        return {
            "schema_version": self.schema_version,
            "policy_version": self.policy_version,
            "decision": self.decision,
            "camera_usable": self.camera_usable,
            "target_slot_id": self.target_slot_id,
            "target_bearing_deg": concise(self.target_bearing_deg),
            "target_distance_m": concise(self.target_distance_m),
            "nominal_fov_coverage": concise(self.nominal_fov_coverage),
            "reliable_fov_coverage": concise(self.reliable_fov_coverage),
            "edge_quality_score": concise(self.edge_quality_score),
            "clear_ray_ratio": concise(self.clear_ray_ratio),
            "blocked_ray_ratio": concise(self.blocked_ray_ratio),
            "uncertain_ray_ratio": concise(self.uncertain_ray_ratio),
            "blocking_object_ids": list(self.blocking_object_ids),
            "potential_occluder_ids": list(self.potential_occluder_ids),
            "reason_codes": list(self.reason_codes),
        }


def _insufficient(target_slot_id: str, reasons: Iterable[str]) -> CameraObservabilityAssessment:
    return CameraObservabilityAssessment(
        decision="insufficient_information",
        target_slot_id=target_slot_id,
        reason_codes=tuple(dict.fromkeys(reasons)),
    )


def _calibration_available(value: Any) -> bool:
    if value is None or value is False:
        return False
    if isinstance(value, Mapping):
        status = value.get("calibration_audit_status", value.get("audit_status"))
        if status is not None and str(status).lower() not in {"passed", "trusted"}:
            return False
        return any(
            value.get(key) is not None
            for key in (
                "calibration_id",
                "camera_id",
                "source_sha256",
                "calibration_sha256",
                "intrinsic_matrix",
                "calibration",
                "angle_to_quality_curve",
                "horizontal_angle_to_quality_curve",
            )
        )
    trusted = getattr(value, "trusted", None)
    return bool(trusted) if trusted is not None else bool(value)


def _calibration_curve(
    calibration: Any,
    nominal_half_fov_deg: float,
) -> tuple[tuple[float, float], ...] | None:
    if isinstance(calibration, Mapping):
        raw = calibration.get(
            "angle_to_quality_curve",
            calibration.get("horizontal_angle_to_quality_curve"),
        )
    else:
        raw = getattr(calibration, "angle_to_quality_curve", None)
    return _quality_curve(raw, nominal_half_fov_deg)


def _strict_calibration(
    value: Any,
    row: CameraObservabilityInput,
) -> tuple[CameraCalibration | None, tuple[str, ...]]:
    """Validate the production calibration contract without masking fields."""

    reasons: list[str] = []
    if value is None:
        return None, (
            "camera_calibration_missing",
            "camera_intrinsics_missing",
            "camera_extrinsics_missing",
        )
    if isinstance(value, Mapping):
        if not all(key in value for key in ("K", "D", "camera_model", "image_width", "image_height")):
            reasons.append("camera_intrinsics_missing")
        if not any(key in value for key in ("T_vehicle_camera", "T_camera_vehicle")):
            reasons.append("camera_extrinsics_missing")
        if reasons:
            return None, tuple(["camera_calibration_missing", *reasons])
        try:
            calibration = camera_calibration_from_mapping(value)
        except (CalibrationError, TypeError, ValueError):
            return None, ("camera_calibration_missing",)
    elif isinstance(value, CameraCalibration):
        calibration = value
        try:
            calibration.require_runtime_calibration()
        except CalibrationError:
            return None, ("camera_calibration_missing",)
    else:
        return None, (
            "camera_calibration_missing",
            "camera_intrinsics_missing",
            "camera_extrinsics_missing",
        )
    if (
        not calibration.calibration_id
        or row.camera_pose_calibration_id is None
        or row.camera_pose_calibration_id != calibration.calibration_id
    ):
        reasons.extend(("camera_calibration_missing", "camera_extrinsics_missing"))
    if row.camera_frame_id is None or row.camera_frame_id != calibration.camera_frame:
        reasons.append("coordinate_frame_mismatch")
    if (
        row.nominal_horizontal_fov_deg is not None
        and row.nominal_horizontal_fov_deg >= 179.0
        and calibration.camera_model != "fisheye"
    ):
        reasons.extend(("camera_calibration_missing", "camera_intrinsics_missing"))
    return calibration, tuple(dict.fromkeys(reasons))


def _pose_uncertainty_metrics(
    row: CameraObservabilityInput,
) -> tuple[float | None, float | None, float | None]:
    if row.camera_pose_covariance is None:
        return None, None, None
    covariance = np.asarray(row.camera_pose_covariance, dtype=np.float64).reshape(6, 6)
    translation_variance = np.diag(covariance)[:3].copy()
    if row.camera_pose_translation_unit == "map_unit":
        if row.map_units_per_meter is None:
            return None, None, None
        translation_variance /= row.map_units_per_meter**2
    elif row.camera_pose_translation_unit != "m":
        return None, None, None
    position_std_m = math.sqrt(max(0.0, float(np.max(translation_variance))))
    yaw_std_deg = math.degrees(math.sqrt(max(0.0, float(covariance[5, 5]))))
    orientation_std_deg = math.degrees(
        math.sqrt(max(0.0, float(np.max(np.diag(covariance)[3:6]))))
    )
    return position_std_m, yaw_std_deg, orientation_std_deg


def _static_obstacles_for_target(
    row: CameraObservabilityInput,
    target_polygon: np.ndarray,
    target_center: np.ndarray,
    settings: CameraObservabilityConfig,
    position_std_m: float,
    yaw_std_deg: float,
) -> tuple[tuple[LineOfSightObject, ...] | None, tuple[str, ...]]:
    """Resolve only the target-local static layer, never a global empty guess."""

    reasons: list[str] = []
    if row.static_obstacle_map_payload is not None:
        if row.camera_pose_map_xyyaw is None or row.map_units_per_meter is None:
            return None, ("static_obstacle_region_incomplete",)
        if row.map_coordinate_frame is None:
            return None, ("coordinate_frame_mismatch",)
        try:
            from .static_obstacle_map import (
                StaticObstacleMap,
                StaticObstacleMapValidationError,
            )

            static_map = StaticObstacleMap.from_mapping(row.static_obstacle_map_payload)
            radius_m = max(
                float(np.linalg.norm(point - target_center))
                for point in target_polygon
            ) / row.map_units_per_meter
            target_distance_m = float(
                np.linalg.norm(
                    target_center - np.asarray(row.camera_pose_map_xyyaw[:2])
                )
                / row.map_units_per_meter
            )
            angular_uncertainty_m = target_distance_m * math.tan(
                math.radians(min(yaw_std_deg, 89.0))
            )
            uncertainty_margin_m = (
                settings.pose_uncertainty_corridor_sigma
                * (position_std_m + angular_uncertainty_m)
            )
            query_width_m = (
                settings.visibility_corridor_width_m
                + 2.0 * radius_m
                + 2.0 * uncertainty_margin_m
            )
            camera_xy = np.asarray(row.camera_pose_map_xyyaw[:2], dtype=np.float64)
            direction = target_center - camera_xy
            direction_norm = float(np.linalg.norm(direction))
            if direction_norm <= _EPSILON:
                return None, ("static_obstacle_region_incomplete",)
            unit_direction = direction / direction_norm
            query_start = camera_xy - (
                unit_direction
                * uncertainty_margin_m
                * row.map_units_per_meter
            )
            query_end = target_center + (
                unit_direction
                * (radius_m + uncertainty_margin_m)
                * row.map_units_per_meter
            )
            query = static_map.query_corridor(
                (float(query_start[0]), float(query_start[1])),
                (float(query_end[0]), float(query_end[1])),
                max(query_width_m, 1e-6),
                coordinate_frame=row.map_coordinate_frame,
                map_units_per_meter=row.map_units_per_meter,
                minimum_camera_occluder_height_m=1e-6,
            )
        except StaticObstacleMapValidationError as exc:
            if any(
                token in exc.reason_code
                for token in ("frame", "coordinate", "unit", "scale")
            ):
                return None, ("coordinate_frame_mismatch",)
            if "layer" in exc.reason_code and "missing" in exc.reason_code:
                return None, ("static_obstacle_layer_missing",)
            return None, ("static_obstacle_region_incomplete",)
        except (TypeError, ValueError):
            return None, ("static_obstacle_region_incomplete",)
        if not query.layer_present:
            reasons.append("static_obstacle_layer_missing")
        if not query.local_coverage_complete:
            reasons.append("static_obstacle_region_incomplete")
        if not query.information_sufficient:
            if any(
                token in reason
                for reason in query.reason_codes
                for token in ("frame", "unit", "scale")
            ):
                reasons.append("coordinate_frame_mismatch")
            return None, tuple(dict.fromkeys(reasons))
        return (
            tuple(
                LineOfSightObject.from_mapping(
                    item.to_mapping(),
                    default_object_type="other_static",
                )
                for item in query.obstacles
            ),
            (),
        )

    if row.static_obstacle_query_complete is not None:
        if row.static_obstacle_layer_present is not True:
            reasons.append("static_obstacle_layer_missing")
        if row.static_obstacle_query_complete is not True:
            reasons.append("static_obstacle_region_incomplete")
        if (
            row.map_coordinate_frame is None
            or row.static_obstacle_coordinate_frame is None
            or row.map_coordinate_frame != row.static_obstacle_coordinate_frame
        ):
            reasons.append("coordinate_frame_mismatch")
        if (
            row.map_units_per_meter is None
            or row.static_obstacle_map_units_per_meter is None
            or not math.isclose(
                row.map_units_per_meter,
                row.static_obstacle_map_units_per_meter,
                rel_tol=1e-9,
                abs_tol=1e-12,
            )
        ):
            reasons.append("coordinate_frame_mismatch")
        if reasons:
            return None, tuple(dict.fromkeys(reasons))
        return row.static_obstacles, ()

    if row.camera_pose_contract_present:
        return None, ("static_obstacle_region_incomplete",)
    if row.static_obstacles is None:
        return None, ("static_obstacle_layer_missing",)
    return row.static_obstacles, ()


def _polygon_center(polygon: np.ndarray) -> np.ndarray:
    following = np.roll(polygon, -1, axis=0)
    cross = polygon[:, 0] * following[:, 1] - following[:, 0] * polygon[:, 1]
    twice_area = float(np.sum(cross))
    if abs(twice_area) <= _EPSILON:
        return np.mean(polygon, axis=0)
    factor = 1.0 / (3.0 * twice_area)
    return np.asarray(
        [
            factor * float(np.sum((polygon[:, 0] + following[:, 0]) * cross)),
            factor * float(np.sum((polygon[:, 1] + following[:, 1]) * cross)),
        ],
        dtype=np.float64,
    )


def _samples(
    polygon: np.ndarray,
    center: np.ndarray,
    include_midpoints: bool,
) -> tuple[np.ndarray, ...]:
    raw = [center, *(np.asarray(point, dtype=np.float64) for point in polygon)]
    if include_midpoints:
        raw.extend(
            (polygon[index] + polygon[(index + 1) % len(polygon)]) * 0.5
            for index in range(len(polygon))
        )
    unique: list[np.ndarray] = []
    for point in raw:
        if not any(np.linalg.norm(point - existing) <= _EPSILON for existing in unique):
            unique.append(point)
    return tuple(unique)


def _angle_delta(value: float, reference: float) -> float:
    return (value - reference + 180.0) % 360.0 - 180.0


def _bearing_interval(
    camera_pose: Sequence[float],
    polygon: np.ndarray,
    reference: float,
) -> tuple[float, float]:
    offsets = [
        _angle_delta(relative_bearing_deg(camera_pose, point), reference)
        for point in polygon
    ]
    return (min(offsets), max(offsets))


def _split_interval(interval: tuple[float, float]) -> tuple[tuple[float, float], ...]:
    low, high = interval
    if high - low >= 360.0 - 1e-6:
        return ((-180.0, 180.0),)
    if high - low <= 180.0:
        return (interval,)
    return ((-180.0, low), (high, 180.0))


def _intervals_overlap(first: tuple[float, float], second: tuple[float, float]) -> bool:
    return any(
        max(a[0], b[0]) <= min(a[1], b[1]) + _EPSILON
        for a in _split_interval(first)
        for b in _split_interval(second)
    )


def _angle_in_interval(angle: float, interval: tuple[float, float]) -> bool:
    return any(
        low - _EPSILON <= angle <= high + _EPSILON
        for low, high in _split_interval(interval)
    )


def _angular_quality(
    absolute_bearing: float,
    config: CameraObservabilityConfig,
    curve: tuple[tuple[float, float], ...] | None,
) -> float:
    if absolute_bearing > config.nominal_half_fov_deg + _EPSILON:
        return 0.0
    if curve is not None:
        for left, right in zip(curve, curve[1:]):
            if absolute_bearing <= right[0]:
                fraction = (absolute_bearing - left[0]) / (right[0] - left[0])
                return left[1] + fraction * (right[1] - left[1])
        return curve[-1][1]
    if absolute_bearing <= config.reliable_half_fov_deg:
        return 1.0
    if absolute_bearing >= config.edge_unreliable_start_deg:
        return 0.0
    return (config.edge_unreliable_start_deg - absolute_bearing) / (
        config.edge_unreliable_start_deg - config.reliable_half_fov_deg
    )


def _project_target_sample(
    row: CameraObservabilityInput,
    calibration: CameraCalibration,
    sample_map_xy: np.ndarray,
    sample_z_m: float,
    config: CameraObservabilityConfig,
    curve: tuple[tuple[float, float], ...] | None,
) -> tuple[bool, float, tuple[float, float] | None]:
    """Project one 3D target sample with the full Camera map pose.

    Map XY is converted to metres before applying the inverse of
    ``T_map_camera``.  Camera axes follow the declared calibration convention
    (+x image-right, +y image-down, +z optical-forward).  Image bounds and the
    fisheye/pinhole model are therefore checked before a sample contributes to
    nominal coverage; no linear angle-to-pixel approximation is used.
    """

    assert row.map_units_per_meter is not None
    assert row.camera_T_map_camera_m is not None
    transform = np.asarray(row.camera_T_map_camera_m, dtype=np.float64)
    point_map_m = np.asarray(
        [
            float(sample_map_xy[0]) / row.map_units_per_meter,
            float(sample_map_xy[1]) / row.map_units_per_meter,
            float(sample_z_m),
        ],
        dtype=np.float64,
    )
    point_camera = transform[:3, :3].T @ (
        point_map_m - transform[:3, 3]
    )
    horizontal_bearing_deg = math.degrees(
        math.atan2(float(point_camera[0]), float(point_camera[2]))
    )
    if abs(horizontal_bearing_deg) > config.nominal_half_fov_deg + _EPSILON:
        return False, 0.0, None
    pixels, valid = calibration.project_camera_points(
        point_camera.reshape(1, 3)
    )
    if not bool(valid[0]):
        return False, 0.0, None
    u_value, v_value = (float(item) for item in pixels[0])
    if not (
        -_EPSILON <= u_value <= calibration.image_width - 1 + _EPSILON
        and -_EPSILON <= v_value <= calibration.image_height - 1 + _EPSILON
    ):
        return False, 0.0, (u_value, v_value)
    return (
        True,
        _angular_quality(
            abs(horizontal_bearing_deg),
            config,
            curve,
        ),
        (u_value, v_value),
    )


def _segment_closest(
    p1: np.ndarray,
    q1: np.ndarray,
    p2: np.ndarray,
    q2: np.ndarray,
) -> tuple[float, float, float]:
    """Closest parameters on two 2D segments and their distance."""

    d1 = q1 - p1
    d2 = q2 - p2
    r = p1 - p2
    a = float(np.dot(d1, d1))
    e = float(np.dot(d2, d2))
    if a <= _EPSILON and e <= _EPSILON:
        return (0.0, 0.0, float(np.linalg.norm(p1 - p2)))
    if a <= _EPSILON:
        s = 0.0
        t = float(np.clip(np.dot(d2, r) / e, 0.0, 1.0))
    else:
        c = float(np.dot(d1, r))
        if e <= _EPSILON:
            t = 0.0
            s = float(np.clip(-c / a, 0.0, 1.0))
        else:
            b = float(np.dot(d1, d2))
            f = float(np.dot(d2, r))
            denominator = a * e - b * b
            s = 0.0 if abs(denominator) <= _EPSILON else float(
                np.clip((b * f - c * e) / denominator, 0.0, 1.0)
            )
            t = (b * s + f) / e
            if t < 0.0:
                t = 0.0
                s = float(np.clip(-c / a, 0.0, 1.0))
            elif t > 1.0:
                t = 1.0
                s = float(np.clip((b - c) / a, 0.0, 1.0))
    first = p1 + s * d1
    second = p2 + t * d2
    return (s, t, float(np.linalg.norm(first - second)))


def _point_polygon_distance(point: np.ndarray, polygon: np.ndarray) -> float:
    if bool(points_in_polygon(point.reshape(1, 2), polygon)[0]):
        return 0.0
    return min(
        _segment_closest(point, point, start, polygon[(index + 1) % len(polygon)])[2]
        for index, start in enumerate(polygon)
    )


def _cross_2d(first: np.ndarray, second: np.ndarray) -> float:
    return float(first[0] * second[1] - first[1] * second[0])


def _capsule_entry_t(
    origin: np.ndarray,
    endpoint: np.ndarray,
    edge_start: np.ndarray,
    edge_end: np.ndarray,
    radius: float,
) -> float | None:
    """Earliest ray parameter entering an edge segment expanded by radius."""

    direction = endpoint - origin
    direction_sq = float(np.dot(direction, direction))
    if direction_sq <= _EPSILON:
        return None
    if _segment_closest(origin, origin, edge_start, edge_end)[2] <= radius + _EPSILON:
        return 0.0

    candidates: list[float] = []
    edge = edge_end - edge_start
    edge_sq = float(np.dot(edge, edge))
    if edge_sq > _EPSILON:
        edge_length = math.sqrt(edge_sq)
        denominator = _cross_2d(edge, direction)
        initial_cross = _cross_2d(edge, origin - edge_start)
        if abs(denominator) > _EPSILON:
            for boundary_cross in (-radius * edge_length, radius * edge_length):
                ray_t = (boundary_cross - initial_cross) / denominator
                if -_EPSILON <= ray_t < 1.0 - _EPSILON:
                    point = origin + ray_t * direction
                    edge_t = float(np.dot(point - edge_start, edge) / edge_sq)
                    if -_EPSILON <= edge_t <= 1.0 + _EPSILON:
                        candidates.append(max(0.0, ray_t))

    for center in (edge_start, edge_end):
        delta = origin - center
        linear = 2.0 * float(np.dot(delta, direction))
        constant = float(np.dot(delta, delta)) - radius * radius
        discriminant = linear * linear - 4.0 * direction_sq * constant
        if discriminant < -_EPSILON:
            continue
        root = math.sqrt(max(0.0, discriminant))
        for ray_t in (
            (-linear - root) / (2.0 * direction_sq),
            (-linear + root) / (2.0 * direction_sq),
        ):
            if -_EPSILON <= ray_t < 1.0 - _EPSILON:
                candidates.append(max(0.0, ray_t))
    return min(candidates) if candidates else None


def _corridor_hit_t(
    origin: np.ndarray,
    endpoint: np.ndarray,
    polygon: np.ndarray,
    half_width_map: float,
) -> float | None:
    if bool(points_in_polygon(origin.reshape(1, 2), polygon)[0]):
        return 0.0
    hits: list[float] = []
    for index, start in enumerate(polygon):
        ray_t = _capsule_entry_t(
            origin,
            endpoint,
            start,
            polygon[(index + 1) % len(polygon)],
            half_width_map,
        )
        if ray_t is not None:
            hits.append(ray_t)
    if not hits:
        return None
    target_distance = float(np.linalg.norm(endpoint - origin))
    if _point_polygon_distance(origin, polygon) >= target_distance - _EPSILON:
        return None
    return min(hits)


def _corridor_hit_interval(
    origin: np.ndarray,
    endpoint: np.ndarray,
    polygon: np.ndarray,
    half_width_map: float,
) -> tuple[float, float] | None:
    """Conservative horizontal t interval for a ray/corridor footprint hit."""

    entry = _corridor_hit_t(origin, endpoint, polygon, half_width_map)
    if entry is None:
        return None
    direction = endpoint - origin
    length_sq = float(np.dot(direction, direction))
    if length_sq <= _EPSILON:
        return None
    projections = (polygon - origin) @ direction / length_sq
    padding = half_width_map / math.sqrt(length_sq)
    exit_t = min(1.0, float(np.max(projections)) + padding)
    return (entry, max(entry, exit_t))


def _vertical_interval_intersects(
    horizontal_interval: tuple[float, float],
    camera_z_m: float,
    target_z_m: float,
    obstacle_min_z_m: float,
    obstacle_max_z_m: float,
    margin_m: float,
) -> bool:
    first_t, last_t = horizontal_interval
    first_z = camera_z_m + first_t * (target_z_m - camera_z_m)
    last_z = camera_z_m + last_t * (target_z_m - camera_z_m)
    ray_min = min(first_z, last_z)
    ray_max = max(first_z, last_z)
    return max(ray_min, obstacle_min_z_m - margin_m) <= min(
        ray_max,
        obstacle_max_z_m + margin_m,
    ) + _EPSILON


def _occluder_kind(
    item: LineOfSightObject,
    config: CameraObservabilityConfig,
) -> str | None:
    if (
        item.occupancy_polygon_map is not None
        or item.object_type in _EXPLICIT_OBJECT_TYPES
        or item.object_type in _STATIC_OBJECT_TYPES
    ):
        return "blocked"
    if item.scope_status == "out_of_route_scope" or item.map_state == "out_of_route":
        return None
    if item.map_state == "occupied":
        return "blocked"
    if item.map_state == "free":
        return None
    if item.map_state == "partial" or item.scope_status == "partial_route_scope":
        probability = item.occupied_probability
        if probability is not None and probability >= config.occupied_probability_blocker_threshold:
            return "blocked"
        if item.clearance_proven:
            return None
        if probability is None or probability >= config.occupied_probability_potential_threshold:
            return "uncertain"
        return None
    if item.map_state in {"unknown", None}:
        return None if item.clearance_proven else "uncertain"
    return None


def _physical_polygon(item: LineOfSightObject) -> np.ndarray | None:
    value = item.occupancy_polygon_map or item.polygon_map
    return None if value is None else np.asarray(value, dtype=np.float64)


@dataclass(frozen=True, slots=True)
class _Occluder:
    object_id: str
    polygon: np.ndarray
    kind: str
    bearing_interval: tuple[float, float]
    object_type: str
    min_z_m: float | None
    max_z_m: float | None
    is_static: bool


def assess_camera_observability(
    observation: CameraObservabilityInput | Mapping[str, Any],
    config: CameraObservabilityConfig | None = None,
    *,
    debug_trace: list[dict[str, Any]] | None = None,
) -> CameraObservabilityAssessment:
    """Return a pre-call Camera-use decision for one target slot."""

    if isinstance(observation, CameraObservabilityInput):
        row = observation
    else:
        try:
            row = CameraObservabilityInput.from_mapping(observation)
        except (AttributeError, TypeError, ValueError):
            target_id = observation.get("target_slot_id", "") if isinstance(observation, Mapping) else ""
            return _insufficient(str(target_id or ""), ("occlusion_information_missing", "invalid_input"))
    settings = config or CameraObservabilityConfig()

    missing: list[str] = []
    target = None if row.slots is None else next(
        (item for item in row.slots if item.object_id == row.target_slot_id),
        None,
    )
    if target is None or target.polygon_map is None:
        missing.append("target_geometry_missing")
    if row.slots is None or row.map_units_per_meter is None:
        missing.append("occlusion_information_missing")
    if row.camera_pose_map_xyyaw is None:
        missing.append("camera_pose_missing")
    position_std_m = 0.0
    yaw_std_deg = 0.0
    runtime_calibration: CameraCalibration | None = None
    if row.camera_pose_contract_present:
        if row.camera_localization_status not in {"valid", "degraded"}:
            missing.append("camera_pose_uncertainty_too_high")
        if (
            row.camera_time_offset_ms is None
            or abs(row.camera_time_offset_ms) > settings.maximum_camera_time_offset_ms
        ):
            missing.append("camera_pose_stale")
        (
            measured_position_std,
            measured_yaw_std,
            measured_orientation_std,
        ) = _pose_uncertainty_metrics(row)
        if (
            measured_position_std is None
            or measured_yaw_std is None
            or measured_orientation_std is None
        ):
            missing.append("camera_pose_uncertainty_too_high")
        else:
            position_std_m = measured_position_std
            yaw_std_deg = measured_yaw_std
            if (
                measured_position_std > settings.maximum_camera_position_std_m
                or measured_yaw_std > settings.maximum_camera_yaw_std_deg
                or measured_orientation_std
                > settings.maximum_camera_orientation_std_deg
            ):
                missing.append("camera_pose_uncertainty_too_high")
        if row.camera_position_z_m is None or row.camera_T_map_camera_m is None:
            missing.append("camera_pose_missing")
        if (
            row.map_coordinate_frame is None
            or row.camera_parent_frame is None
            or row.map_coordinate_frame != row.camera_parent_frame
        ):
            missing.append("coordinate_frame_mismatch")
        if row.target_ground_z_m is None:
            missing.append("target_geometry_missing")
        runtime_calibration, calibration_reasons = _strict_calibration(
            row.camera_calibration,
            row,
        )
        missing.extend(calibration_reasons)
    elif not settings.allow_legacy_2d_test_contract:
        missing.extend(
            (
                "camera_pose_missing",
                "camera_calibration_missing",
                "camera_intrinsics_missing",
                "camera_extrinsics_missing",
                "static_obstacle_region_incomplete",
            )
        )
    elif not _calibration_available(row.camera_calibration):
        missing.append("camera_calibration_missing")
    if row.nominal_horizontal_fov_deg is None:
        missing.extend(("camera_calibration_missing", "camera_fov_missing"))

    static_obstacles: tuple[LineOfSightObject, ...] | None = None
    target_polygon: np.ndarray | None = None
    target_center: np.ndarray | None = None
    if (
        target is not None
        and target.polygon_map is not None
        and row.camera_pose_map_xyyaw is not None
        and row.map_units_per_meter is not None
    ):
        target_polygon = np.asarray(target.polygon_map, dtype=np.float64)
        target_center = (
            np.asarray(target.center_map, dtype=np.float64)
            if target.center_map is not None
            else _polygon_center(target_polygon)
        )
        static_obstacles, static_reasons = _static_obstacles_for_target(
            row,
            target_polygon,
            target_center,
            settings,
            position_std_m,
            yaw_std_deg,
        )
        if static_reasons:
            missing.extend(static_reasons)
            missing.append("occlusion_information_missing")
    if static_obstacles is None and not any(
        reason in missing
        for reason in (
            "static_obstacle_layer_missing",
            "static_obstacle_region_incomplete",
            "occlusion_information_missing",
        )
    ):
        missing.extend(("static_obstacle_layer_missing", "occlusion_information_missing"))
    elif static_obstacles is not None and any(
        item.polygon_map is None and item.occupancy_polygon_map is None
        for item in static_obstacles
    ):
        missing.append("occlusion_information_missing")
    if missing:
        return _insufficient(row.target_slot_id, tuple(dict.fromkeys(missing)))

    assert target is not None and target.polygon_map is not None
    assert target_polygon is not None and target_center is not None
    assert row.camera_pose_map_xyyaw is not None
    assert row.nominal_horizontal_fov_deg is not None
    assert row.map_units_per_meter is not None
    assert row.slots is not None and static_obstacles is not None

    if not math.isclose(
        row.nominal_horizontal_fov_deg * 0.5,
        settings.nominal_half_fov_deg,
        abs_tol=1e-6,
    ):
        return _insufficient(
            row.target_slot_id,
            ("camera_calibration_missing", "camera_fov_policy_mismatch"),
        )
    try:
        calibrated_curve = _calibration_curve(
            row.camera_calibration,
            settings.nominal_half_fov_deg,
        )
    except (TypeError, ValueError):
        return _insufficient(
            row.target_slot_id,
            ("camera_calibration_missing", "camera_quality_curve_invalid"),
        )
    curve = calibrated_curve or settings.angle_to_quality_curve

    camera_pose = row.camera_pose_map_xyyaw
    camera_xy = np.asarray(camera_pose[:2], dtype=np.float64)
    target_bearing = relative_bearing_deg(camera_pose, target_center)
    target_distance = float(np.linalg.norm(target_center - camera_xy) / row.map_units_per_meter)
    sample_points = _samples(target_polygon, target_center, settings.include_edge_midpoints)
    target_interval = _bearing_interval(camera_pose, target_polygon, target_bearing)

    candidates: list[tuple[LineOfSightObject, bool]] = [
        (item, False)
        for item in row.slots
        if item.object_id != target.object_id
    ]
    candidates.extend((item, True) for item in static_obstacles)
    occluders: list[_Occluder] = []
    missing_occluder_geometry = False
    missing_static_height = False
    for item, is_static in candidates:
        kind = _occluder_kind(item, settings)
        if kind is None:
            continue
        polygon = _physical_polygon(item)
        if polygon is None:
            missing_occluder_geometry = True
            continue
        if (
            row.camera_pose_contract_present
            and is_static
            and (item.min_z_m is None or item.max_z_m is None)
        ):
            missing_static_height = True
            continue
        camera_on_or_inside = (
            _point_polygon_distance(camera_xy, polygon) <= _EPSILON
        )
        interval = (
            (-180.0, 180.0)
            if camera_on_or_inside
            else _bearing_interval(camera_pose, polygon, target_bearing)
        )
        if camera_on_or_inside or _intervals_overlap(interval, target_interval):
            occluders.append(
                _Occluder(
                    object_id=item.object_id,
                    polygon=polygon,
                    kind=kind,
                    bearing_interval=interval,
                    object_type=item.object_type,
                    min_z_m=item.min_z_m,
                    max_z_m=item.max_z_m,
                    is_static=is_static,
                )
            )
    if missing_occluder_geometry or missing_static_height:
        return _insufficient(row.target_slot_id, ("occlusion_information_missing",))
    if (
        row.camera_pose_contract_present
        and any(item.is_static for item in occluders)
        and row.target_ground_z_m is None
    ):
        return _insufficient(row.target_slot_id, ("occlusion_information_missing",))

    nominal_count = 0
    reliable_count = 0
    quality_total = 0.0
    ray_results: list[str] = []
    blocking_ids: set[str] = set()
    potential_ids: set[str] = set()
    static_blocking_ids: set[str] = set()
    angular_uncertainty_m = target_distance * math.tan(
        math.radians(min(yaw_std_deg, 89.0))
    )
    half_corridor_map = row.map_units_per_meter * (
        settings.visibility_corridor_width_m * 0.5
        + settings.pose_uncertainty_corridor_sigma
        * (position_std_m + angular_uncertainty_m)
    )
    target_height_offsets: tuple[float | None, ...] = (
        tuple(settings.target_sample_heights_m)
        if row.camera_pose_contract_present
        else (None,)
    )

    for sample in sample_points:
        world_bearing = relative_bearing_deg(camera_pose, sample)
        sample_offset = _angle_delta(world_bearing, target_bearing)
        for height_offset in target_height_offsets:
            target_z_m = (
                None
                if height_offset is None or row.target_ground_z_m is None
                else row.target_ground_z_m + height_offset
            )
            if row.camera_pose_contract_present:
                assert runtime_calibration is not None
                assert target_z_m is not None
                nominal, quality, pixel_uv = _project_target_sample(
                    row,
                    runtime_calibration,
                    sample,
                    target_z_m,
                    settings,
                    curve,
                )
            else:
                absolute_bearing = abs(world_bearing)
                nominal = (
                    absolute_bearing
                    <= settings.nominal_half_fov_deg + _EPSILON
                )
                quality = (
                    _angular_quality(absolute_bearing, settings, curve)
                    if nominal
                    else 0.0
                )
                pixel_uv = None
            if not nominal:
                if debug_trace is not None:
                    debug_trace.append(
                        {
                            "sample_map_xy": [float(sample[0]), float(sample[1])],
                            "target_z_m": target_z_m,
                            "pixel_uv": None if pixel_uv is None else list(pixel_uv),
                            "status": "outside_fov",
                            "quality": 0.0,
                            "object_id": None,
                        }
                    )
                continue
            nominal_count += 1
            quality_total += quality
            if quality >= settings.minimum_reliable_angular_quality:
                reliable_count += 1
            if quality <= settings.unreliable_angular_quality_threshold:
                if debug_trace is not None:
                    debug_trace.append(
                        {
                            "sample_map_xy": [float(sample[0]), float(sample[1])],
                            "target_z_m": target_z_m,
                            "pixel_uv": None if pixel_uv is None else list(pixel_uv),
                            "status": "edge_unreliable",
                            "quality": float(quality),
                            "object_id": None,
                        }
                    )
                continue
            hits: list[tuple[float, int, _Occluder]] = []
            for occluder in occluders:
                if not _angle_in_interval(sample_offset, occluder.bearing_interval):
                    continue
                hit_interval = _corridor_hit_interval(
                    camera_xy,
                    sample,
                    occluder.polygon,
                    half_corridor_map,
                )
                if hit_interval is None:
                    continue
                if occluder.is_static and row.camera_pose_contract_present:
                    assert row.camera_position_z_m is not None
                    assert target_z_m is not None
                    assert occluder.min_z_m is not None
                    assert occluder.max_z_m is not None
                    if not _vertical_interval_intersects(
                        hit_interval,
                        row.camera_position_z_m,
                        target_z_m,
                        occluder.min_z_m,
                        occluder.max_z_m,
                        settings.vertical_occlusion_margin_m,
                    ):
                        continue
                hits.append(
                    (
                        hit_interval[0],
                        0 if occluder.kind == "blocked" else 1,
                        occluder,
                    )
                )
            if not hits:
                ray_results.append("clear")
                if debug_trace is not None:
                    debug_trace.append(
                        {
                            "sample_map_xy": [float(sample[0]), float(sample[1])],
                            "target_z_m": target_z_m,
                            "pixel_uv": None if pixel_uv is None else list(pixel_uv),
                            "status": "clear",
                            "quality": float(quality),
                            "object_id": None,
                        }
                    )
                continue
            nearest = min(
                hits,
                key=lambda hit: (hit[0], hit[1], hit[2].object_id),
            )[2]
            ray_results.append(nearest.kind)
            if debug_trace is not None:
                debug_trace.append(
                    {
                        "sample_map_xy": [float(sample[0]), float(sample[1])],
                        "target_z_m": target_z_m,
                        "pixel_uv": None if pixel_uv is None else list(pixel_uv),
                        "status": nearest.kind,
                        "quality": float(quality),
                        "object_id": nearest.object_id,
                        "object_type": nearest.object_type,
                    }
                )
            if nearest.kind == "blocked":
                blocking_ids.add(nearest.object_id)
                if nearest.is_static:
                    static_blocking_ids.add(nearest.object_id)
            else:
                potential_ids.add(nearest.object_id)

    sample_count = len(sample_points) * len(target_height_offsets)
    nominal_coverage = nominal_count / sample_count
    reliable_coverage = reliable_count / sample_count
    edge_quality = quality_total / sample_count
    eligible_count = len(ray_results)
    clear_ratio = ray_results.count("clear") / eligible_count if eligible_count else 0.0
    blocked_ratio = ray_results.count("blocked") / eligible_count if eligible_count else 0.0
    uncertain_ratio = ray_results.count("uncertain") / eligible_count if eligible_count else 0.0
    common: dict[str, Any] = {
        "target_slot_id": row.target_slot_id,
        "target_bearing_deg": target_bearing,
        "target_distance_m": target_distance,
        "nominal_fov_coverage": nominal_coverage,
        "reliable_fov_coverage": reliable_coverage,
        "edge_quality_score": edge_quality,
        "clear_ray_ratio": clear_ratio,
        "blocked_ray_ratio": blocked_ratio,
        "uncertain_ray_ratio": uncertain_ratio,
        "blocking_object_ids": tuple(blocking_ids),
        "potential_occluder_ids": tuple(potential_ids),
    }

    if target.map_state == "out_of_route" or target.scope_status == "out_of_route_scope":
        return CameraObservabilityAssessment(
            decision="do_not_use_camera",
            reason_codes=("target_out_of_route",),
            **common,
        )
    if nominal_count == 0:
        return CameraObservabilityAssessment(
            decision="do_not_use_camera",
            reason_codes=("target_outside_fov",),
            **common,
        )
    if (
        reliable_coverage < settings.minimum_reliable_fov_coverage
        or edge_quality < settings.minimum_edge_quality_score
    ):
        return CameraObservabilityAssessment(
            decision="do_not_use_camera",
            reason_codes=(
                "target_in_unreliable_edge_region",
                "insufficient_reliable_fov_coverage",
            ),
            **common,
        )
    if blocked_ratio > settings.maximum_blocked_ray_ratio:
        reasons = ["explicit_line_of_sight_blockage"]
        if static_blocking_ids:
            reasons.append("static_obstacle_blocks_view")
        return CameraObservabilityAssessment(
            decision="do_not_use_camera",
            reason_codes=tuple(reasons),
            **common,
        )
    if potential_ids:
        return CameraObservabilityAssessment(
            decision="insufficient_information",
            reason_codes=("unresolved_potential_occluder",),
            **common,
        )
    if (
        reliable_coverage >= settings.minimum_reliable_fov_coverage
        and clear_ratio >= settings.minimum_clear_ray_ratio
        and blocked_ratio <= settings.maximum_blocked_ray_ratio
    ):
        return CameraObservabilityAssessment(
            decision="use_camera",
            reason_codes=(
                "target_inside_reliable_fov",
                "line_of_sight_mostly_clear",
            ),
            **common,
        )
    return CameraObservabilityAssessment(
        decision="insufficient_information",
        reason_codes=("insufficient_line_of_sight_clearance",),
        **common,
    )


def camera_observability_tool(payload: Mapping[str, Any]) -> dict[str, Any]:
    """JSON-facing pre-Camera decision entry point."""

    if not isinstance(payload, Mapping):
        return _insufficient("", ("invalid_input",)).to_dict()
    raw_config = payload.get("camera_observability_config")
    try:
        config = None if raw_config is None else CameraObservabilityConfig.from_mapping(raw_config)
    except (AttributeError, TypeError, ValueError):
        return _insufficient(
            str(payload.get("target_slot_id", "") or ""),
            ("camera_calibration_missing", "invalid_camera_observability_config"),
        ).to_dict()
    return assess_camera_observability(payload, config=config).to_dict()


__all__ = [
    "CAMERA_OBSERVABILITY_POLICY_VERSION",
    "CAMERA_OBSERVABILITY_SCHEMA_VERSION",
    "CAMERA_REASON_CODES",
    "CAMERA_USE_DECISIONS",
    "CameraObservabilityAssessment",
    "CameraObservabilityConfig",
    "CameraObservabilityInput",
    "LineOfSightObject",
    "assess_camera_observability",
    "camera_observability_tool",
]
