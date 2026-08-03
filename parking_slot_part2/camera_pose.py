"""Frame-safe Camera map pose interpolation and composition.

All transforms use T_A_B notation: T_A_B maps points from frame B into frame
A.  Camera pose is composed only as

    T_map_camera = T_map_vehicle @ T_vehicle_camera

The calibration extrinsic translation is metric.  Vehicle map poses must
declare whether their translations are metres or CAD map units.  For map-unit
poses, map_units_per_meter is mandatory and the extrinsic translation is
scaled before composition; raw matrices with inconsistent units are never
multiplied.

Pose covariance is a 6x6 row-major matrix for [x, y, z, rx, ry, rz] small
left perturbations expressed in the parent/map frame.  Translation covariance
uses the declared translation unit; rotation covariance uses radians.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Iterable, Literal, Mapping, Sequence

import numpy as np

from .camera_calibration import (
    CalibrationError,
    CameraCalibration,
    validate_rigid_transform,
)


CAMERA_MAP_POSE_SCHEMA_VERSION = "camera-map-pose/1.0"
LOCALIZATION_STATUSES = frozenset({"valid", "degraded", "invalid"})
TRANSLATION_UNITS = frozenset({"m", "map_unit"})


class PoseValidationError(ValueError):
    """A pose cannot safely support interpolation or Camera composition."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        time_offset_ms: float | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.time_offset_ms = time_offset_ms


def _required_text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PoseValidationError(
            "invalid_pose_input",
            f"{name} must be a non-empty string",
        )
    return value.strip()


def _timestamp_ns(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise PoseValidationError(
            "invalid_pose_input",
            "timestamp_ns must be a non-negative integer",
        )
    return value


def _finite_positive(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PoseValidationError(
            "invalid_pose_input",
            f"{name} must be a positive finite number",
        )
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise PoseValidationError(
            "invalid_pose_input",
            f"{name} must be a positive finite number",
        )
    return parsed


def _pose_covariance(value: Any) -> np.ndarray:
    try:
        covariance = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise PoseValidationError(
            "covariance_invalid",
            "pose_covariance must contain numbers",
        ) from exc
    if covariance.shape == (36,):
        covariance = covariance.reshape(6, 6)
    if covariance.shape != (6, 6):
        raise PoseValidationError(
            "covariance_invalid",
            "pose_covariance must have shape [6, 6] or [36]",
        )
    if not np.isfinite(covariance).all():
        raise PoseValidationError(
            "covariance_invalid",
            "pose_covariance must be finite",
        )
    if not np.allclose(covariance, covariance.T, atol=1e-10):
        raise PoseValidationError(
            "covariance_invalid",
            "pose_covariance must be symmetric",
        )
    eigenvalues = np.linalg.eigvalsh(covariance)
    if float(np.min(eigenvalues)) < -1e-10:
        raise PoseValidationError(
            "covariance_invalid",
            "pose_covariance must be positive semidefinite",
        )
    result = covariance.copy()
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True)
class VehiclePoseSample:
    """One full SE(3) vehicle pose with explicit quality and units."""

    timestamp_ns: int
    parent_frame: str
    frame_id: str
    T_map_vehicle: np.ndarray
    pose_covariance: np.ndarray
    localization_status: Literal["valid", "degraded", "invalid"]
    translation_unit: Literal["m", "map_unit"]
    map_units_per_meter: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "timestamp_ns",
            _timestamp_ns(self.timestamp_ns),
        )
        parent = _required_text(self.parent_frame, "parent_frame")
        frame = _required_text(self.frame_id, "frame_id")
        if parent == frame:
            raise PoseValidationError(
                "frame_mismatch",
                "parent_frame and frame_id must differ",
            )
        object.__setattr__(self, "parent_frame", parent)
        object.__setattr__(self, "frame_id", frame)
        try:
            transform = validate_rigid_transform(
                self.T_map_vehicle,
                "T_map_vehicle",
            )
        except CalibrationError as exc:
            raise PoseValidationError(
                "pose_transform_invalid",
                str(exc),
            ) from exc
        object.__setattr__(self, "T_map_vehicle", transform)
        object.__setattr__(
            self,
            "pose_covariance",
            _pose_covariance(self.pose_covariance),
        )
        status = _required_text(
            self.localization_status,
            "localization_status",
        ).lower()
        if status not in LOCALIZATION_STATUSES:
            raise PoseValidationError(
                "localization_status_invalid",
                "localization_status must be valid, degraded, or invalid",
            )
        object.__setattr__(self, "localization_status", status)
        unit = _required_text(
            self.translation_unit,
            "translation_unit",
        )
        if unit not in TRANSLATION_UNITS:
            raise PoseValidationError(
                "translation_unit_invalid",
                "translation_unit must be m or map_unit",
            )
        object.__setattr__(self, "translation_unit", unit)
        if unit == "map_unit":
            if self.map_units_per_meter is None:
                raise PoseValidationError(
                    "map_scale_missing",
                    "map_units_per_meter is required for map_unit poses",
                )
            object.__setattr__(
                self,
                "map_units_per_meter",
                _finite_positive(
                    self.map_units_per_meter,
                    "map_units_per_meter",
                ),
            )
        elif self.map_units_per_meter is not None:
            raise PoseValidationError(
                "translation_unit_mismatch",
                "metric poses must not declare map_units_per_meter",
            )

    @classmethod
    def from_mapping(
        cls,
        payload: Mapping[str, Any],
    ) -> "VehiclePoseSample":
        required = {
            "timestamp_ns",
            "parent_frame",
            "frame_id",
            "T_map_vehicle",
            "pose_covariance",
            "localization_status",
            "translation_unit",
        }
        missing = sorted(required - payload.keys())
        if missing:
            raise PoseValidationError(
                "pose_fields_missing",
                "vehicle pose is missing: " + ", ".join(missing),
            )
        return cls(
            timestamp_ns=payload["timestamp_ns"],
            parent_frame=payload["parent_frame"],
            frame_id=payload["frame_id"],
            T_map_vehicle=payload["T_map_vehicle"],
            pose_covariance=payload["pose_covariance"],
            localization_status=payload["localization_status"],
            translation_unit=payload["translation_unit"],
            map_units_per_meter=payload.get("map_units_per_meter"),
        )


@dataclass(frozen=True, slots=True)
class PoseInterpolationPolicy:
    """Caller-owned time and covariance gates; no dataset values are guessed."""

    max_abs_time_offset_ms: float
    max_interpolation_gap_ms: float
    degraded_position_variance_m2: float
    invalid_position_variance_m2: float
    degraded_orientation_variance_rad2: float
    invalid_orientation_variance_rad2: float

    def __post_init__(self) -> None:
        values = {}
        for name in (
            "max_abs_time_offset_ms",
            "max_interpolation_gap_ms",
            "degraded_position_variance_m2",
            "invalid_position_variance_m2",
            "degraded_orientation_variance_rad2",
            "invalid_orientation_variance_rad2",
        ):
            values[name] = _finite_positive(getattr(self, name), name)
            object.__setattr__(self, name, values[name])
        if (
            self.degraded_position_variance_m2
            > self.invalid_position_variance_m2
        ):
            raise PoseValidationError(
                "invalid_pose_policy",
                "degraded position variance must not exceed invalid limit",
            )
        if (
            self.degraded_orientation_variance_rad2
            > self.invalid_orientation_variance_rad2
        ):
            raise PoseValidationError(
                "invalid_pose_policy",
                "degraded orientation variance must not exceed invalid limit",
            )


def _matrix_to_quaternion_xyzw(rotation: np.ndarray) -> np.ndarray:
    trace = float(np.trace(rotation))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * scale
        qx = (rotation[2, 1] - rotation[1, 2]) / scale
        qy = (rotation[0, 2] - rotation[2, 0]) / scale
        qz = (rotation[1, 0] - rotation[0, 1]) / scale
    else:
        diagonal = np.diag(rotation)
        index = int(np.argmax(diagonal))
        if index == 0:
            scale = math.sqrt(
                1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]
            ) * 2.0
            qw = (rotation[2, 1] - rotation[1, 2]) / scale
            qx = 0.25 * scale
            qy = (rotation[0, 1] + rotation[1, 0]) / scale
            qz = (rotation[0, 2] + rotation[2, 0]) / scale
        elif index == 1:
            scale = math.sqrt(
                1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]
            ) * 2.0
            qw = (rotation[0, 2] - rotation[2, 0]) / scale
            qx = (rotation[0, 1] + rotation[1, 0]) / scale
            qy = 0.25 * scale
            qz = (rotation[1, 2] + rotation[2, 1]) / scale
        else:
            scale = math.sqrt(
                1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]
            ) * 2.0
            qw = (rotation[1, 0] - rotation[0, 1]) / scale
            qx = (rotation[0, 2] + rotation[2, 0]) / scale
            qy = (rotation[1, 2] + rotation[2, 1]) / scale
            qz = 0.25 * scale
    quaternion = np.asarray([qx, qy, qz, qw], dtype=np.float64)
    quaternion /= np.linalg.norm(quaternion)
    if quaternion[3] < 0.0:
        quaternion = -quaternion
    return quaternion


def _quaternion_xyzw_to_matrix(value: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(value, dtype=np.float64)
    quaternion = quaternion / np.linalg.norm(quaternion)
    x, y, z, w = quaternion
    return np.asarray(
        [
            [
                1.0 - 2.0 * (y * y + z * z),
                2.0 * (x * y - z * w),
                2.0 * (x * z + y * w),
            ],
            [
                2.0 * (x * y + z * w),
                1.0 - 2.0 * (x * x + z * z),
                2.0 * (y * z - x * w),
            ],
            [
                2.0 * (x * z - y * w),
                2.0 * (y * z + x * w),
                1.0 - 2.0 * (x * x + y * y),
            ],
        ],
        dtype=np.float64,
    )


def _slerp_xyzw(
    first: np.ndarray,
    second: np.ndarray,
    fraction: float,
) -> np.ndarray:
    q0 = np.asarray(first, dtype=np.float64)
    q1 = np.asarray(second, dtype=np.float64)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    dot = float(np.clip(dot, -1.0, 1.0))
    if dot > 0.9995:
        result = q0 + fraction * (q1 - q0)
        return result / np.linalg.norm(result)
    angle = math.acos(dot)
    sine = math.sin(angle)
    result = (
        math.sin((1.0 - fraction) * angle) / sine * q0
        + math.sin(fraction * angle) / sine * q1
    )
    return result / np.linalg.norm(result)


def _metric_covariance(
    covariance: np.ndarray,
    translation_unit: str,
    map_units_per_meter: float | None,
) -> np.ndarray:
    if translation_unit == "m":
        return covariance
    assert map_units_per_meter is not None
    conversion = np.eye(6, dtype=np.float64)
    conversion[:3, :3] /= map_units_per_meter
    return conversion @ covariance @ conversion.T


def _status_from_covariance(
    covariance: np.ndarray,
    translation_unit: str,
    map_units_per_meter: float | None,
    base_status: str,
    policy: PoseInterpolationPolicy,
) -> str:
    if base_status == "invalid":
        raise PoseValidationError(
            "localization_invalid",
            "an input localization sample is invalid",
        )
    metric = _metric_covariance(
        covariance,
        translation_unit,
        map_units_per_meter,
    )
    position_variance = float(np.max(np.diag(metric)[:3]))
    orientation_variance = float(np.max(np.diag(metric)[3:]))
    if (
        position_variance > policy.invalid_position_variance_m2
        or orientation_variance
        > policy.invalid_orientation_variance_rad2
    ):
        raise PoseValidationError(
            "covariance_exceeds_limit",
            "pose covariance exceeds the invalid threshold",
        )
    if (
        base_status == "degraded"
        or position_variance > policy.degraded_position_variance_m2
        or orientation_variance
        > policy.degraded_orientation_variance_rad2
    ):
        return "degraded"
    return "valid"


@dataclass(frozen=True, slots=True)
class InterpolatedVehiclePose:
    timestamp_ns: int
    parent_frame: str
    frame_id: str
    T_map_vehicle: np.ndarray
    pose_covariance: np.ndarray
    localization_status: Literal["valid", "degraded"]
    time_offset_ms: float
    translation_unit: Literal["m", "map_unit"]
    map_units_per_meter: float | None
    interpolated: bool


def _validate_sample_series(
    samples: Sequence[VehiclePoseSample],
) -> tuple[VehiclePoseSample, ...]:
    if not samples:
        raise PoseValidationError(
            "pose_samples_missing",
            "at least one vehicle pose sample is required",
        )
    ordered = tuple(sorted(samples, key=lambda row: row.timestamp_ns))
    timestamps = [row.timestamp_ns for row in ordered]
    if len(set(timestamps)) != len(timestamps):
        raise PoseValidationError(
            "pose_timestamp_duplicate",
            "vehicle pose timestamps must be unique",
        )
    reference = ordered[0]
    for row in ordered[1:]:
        if (
            row.parent_frame != reference.parent_frame
            or row.frame_id != reference.frame_id
        ):
            raise PoseValidationError(
                "frame_mismatch",
                "vehicle pose samples use inconsistent frames",
            )
        if (
            row.translation_unit != reference.translation_unit
            or row.map_units_per_meter
            != reference.map_units_per_meter
        ):
            raise PoseValidationError(
                "translation_unit_mismatch",
                "vehicle pose samples use inconsistent units or map scale",
            )
    return ordered


def interpolate_vehicle_pose(
    samples: Sequence[VehiclePoseSample],
    timestamp_ns: int,
    policy: PoseInterpolationPolicy,
) -> InterpolatedVehiclePose:
    """Interpolate translation, rotation, and covariance without extrapolation.

    time_offset_ms is signed as nearest_pose_timestamp - query_timestamp.
    """

    query = _timestamp_ns(timestamp_ns)
    ordered = _validate_sample_series(samples)
    times = np.asarray([row.timestamp_ns for row in ordered], dtype=np.int64)
    right = int(np.searchsorted(times, query, side="left"))
    if right < len(ordered) and ordered[right].timestamp_ns == query:
        left = right
        alpha = 0.0
        interpolated = False
    else:
        if right == 0 or right == len(ordered):
            nearest = min(
                ordered,
                key=lambda row: abs(row.timestamp_ns - query),
            )
            offset = (nearest.timestamp_ns - query) / 1e6
            raise PoseValidationError(
                "pose_not_bracketed",
                "pose extrapolation is forbidden",
                time_offset_ms=offset,
            )
        left = right - 1
        interval_ns = ordered[right].timestamp_ns - ordered[left].timestamp_ns
        gap_ms = interval_ns / 1e6
        if gap_ms > policy.max_interpolation_gap_ms:
            raise PoseValidationError(
                "interpolation_gap_too_large",
                "pose interpolation gap exceeds policy",
            )
        alpha = (query - ordered[left].timestamp_ns) / interval_ns
        interpolated = True

    first = ordered[left]
    second = ordered[right] if interpolated else first
    nearest = min(
        (first, second),
        key=lambda row: (
            abs(row.timestamp_ns - query),
            row.timestamp_ns,
        ),
    )
    time_offset_ms = (nearest.timestamp_ns - query) / 1e6
    if abs(time_offset_ms) > policy.max_abs_time_offset_ms:
        raise PoseValidationError(
            "pose_timestamp_stale",
            "nearest vehicle pose timestamp exceeds policy",
            time_offset_ms=time_offset_ms,
        )
    if first.localization_status == "invalid" or second.localization_status == "invalid":
        raise PoseValidationError(
            "localization_invalid",
            "an interpolation endpoint is invalid",
            time_offset_ms=time_offset_ms,
        )

    if interpolated:
        translation = (
            (1.0 - alpha) * first.T_map_vehicle[:3, 3]
            + alpha * second.T_map_vehicle[:3, 3]
        )
        first_q = _matrix_to_quaternion_xyzw(
            first.T_map_vehicle[:3, :3]
        )
        second_q = _matrix_to_quaternion_xyzw(
            second.T_map_vehicle[:3, :3]
        )
        quaternion = _slerp_xyzw(first_q, second_q, alpha)
        covariance = (
            (1.0 - alpha) * first.pose_covariance
            + alpha * second.pose_covariance
        )
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = _quaternion_xyzw_to_matrix(quaternion)
        transform[:3, 3] = translation
        base_status = (
            "degraded"
            if "degraded"
            in {first.localization_status, second.localization_status}
            else "valid"
        )
    else:
        transform = first.T_map_vehicle.copy()
        covariance = first.pose_covariance.copy()
        base_status = first.localization_status

    status = _status_from_covariance(
        covariance,
        first.translation_unit,
        first.map_units_per_meter,
        base_status,
        policy,
    )
    transform.setflags(write=False)
    covariance.setflags(write=False)
    return InterpolatedVehiclePose(
        timestamp_ns=query,
        parent_frame=first.parent_frame,
        frame_id=first.frame_id,
        T_map_vehicle=transform,
        pose_covariance=covariance,
        localization_status=status,
        time_offset_ms=float(time_offset_ms),
        translation_unit=first.translation_unit,
        map_units_per_meter=first.map_units_per_meter,
        interpolated=interpolated,
    )


def _skew(value: np.ndarray) -> np.ndarray:
    x, y, z = value
    return np.asarray(
        [[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]],
        dtype=np.float64,
    )


def _camera_covariance(
    vehicle_covariance: np.ndarray,
    T_map_vehicle: np.ndarray,
    T_vehicle_camera_in_pose_units: np.ndarray,
) -> np.ndarray:
    lever = (
        T_map_vehicle[:3, :3]
        @ T_vehicle_camera_in_pose_units[:3, 3]
    )
    jacobian = np.eye(6, dtype=np.float64)
    jacobian[:3, 3:] = -_skew(lever)
    covariance = jacobian @ vehicle_covariance @ jacobian.T
    covariance = 0.5 * (covariance + covariance.T)
    covariance.setflags(write=False)
    return covariance


@dataclass(frozen=True, slots=True)
class CameraMapPoseEstimate:
    """Machine-readable Camera map pose output."""

    timestamp_ns: int
    frame_id: str
    parent_frame: str
    T_map_camera: np.ndarray | None
    pose_covariance: np.ndarray | None
    localization_status: Literal["valid", "degraded", "invalid"]
    time_offset_ms: float | None
    calibration_id: str
    translation_unit: Literal["m", "map_unit"]
    map_units_per_meter: float | None
    reason_codes: tuple[str, ...] = ()
    schema_version: str = CAMERA_MAP_POSE_SCHEMA_VERSION
    position_xyz: np.ndarray | None = field(init=False)
    orientation_xyzw: np.ndarray | None = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "timestamp_ns",
            _timestamp_ns(self.timestamp_ns),
        )
        object.__setattr__(
            self,
            "frame_id",
            _required_text(self.frame_id, "frame_id"),
        )
        object.__setattr__(
            self,
            "parent_frame",
            _required_text(self.parent_frame, "parent_frame"),
        )
        status = _required_text(
            self.localization_status,
            "localization_status",
        )
        if status not in LOCALIZATION_STATUSES:
            raise PoseValidationError(
                "localization_status_invalid",
                "unsupported localization status",
            )
        object.__setattr__(self, "localization_status", status)
        object.__setattr__(
            self,
            "calibration_id",
            str(self.calibration_id or ""),
        )
        unit = _required_text(
            self.translation_unit,
            "translation_unit",
        )
        if unit not in TRANSLATION_UNITS:
            raise PoseValidationError(
                "translation_unit_invalid",
                "translation_unit must be m or map_unit",
            )
        object.__setattr__(self, "translation_unit", unit)
        if unit == "map_unit":
            if self.map_units_per_meter is None:
                raise PoseValidationError(
                    "map_scale_missing",
                    "map-unit output requires map_units_per_meter",
                )
            object.__setattr__(
                self,
                "map_units_per_meter",
                _finite_positive(
                    self.map_units_per_meter,
                    "map_units_per_meter",
                ),
            )
        elif self.map_units_per_meter is not None:
            raise PoseValidationError(
                "translation_unit_mismatch",
                "metric output must not declare map_units_per_meter",
            )
        if self.time_offset_ms is not None:
            offset = float(self.time_offset_ms)
            if not math.isfinite(offset):
                raise PoseValidationError(
                    "invalid_pose_input",
                    "time_offset_ms must be finite or null",
                )
            object.__setattr__(self, "time_offset_ms", offset)

        if status == "invalid":
            if self.T_map_camera is not None or self.pose_covariance is not None:
                raise PoseValidationError(
                    "invalid_pose_output",
                    "invalid output must not expose transform or covariance",
                )
            object.__setattr__(self, "position_xyz", None)
            object.__setattr__(self, "orientation_xyzw", None)
        else:
            if self.T_map_camera is None or self.pose_covariance is None:
                raise PoseValidationError(
                    "invalid_pose_output",
                    "valid/degraded output requires transform and covariance",
                )
            try:
                transform = validate_rigid_transform(
                    self.T_map_camera,
                    "T_map_camera",
                )
            except CalibrationError as exc:
                raise PoseValidationError(
                    "invalid_pose_output",
                    str(exc),
                ) from exc
            covariance = _pose_covariance(self.pose_covariance)
            object.__setattr__(self, "T_map_camera", transform)
            object.__setattr__(self, "pose_covariance", covariance)
            position = transform[:3, 3].copy()
            orientation = _matrix_to_quaternion_xyzw(
                transform[:3, :3]
            )
            position.setflags(write=False)
            orientation.setflags(write=False)
            object.__setattr__(self, "position_xyz", position)
            object.__setattr__(self, "orientation_xyzw", orientation)
        object.__setattr__(
            self,
            "reason_codes",
            tuple(dict.fromkeys(str(item) for item in self.reason_codes)),
        )
        if self.schema_version != CAMERA_MAP_POSE_SCHEMA_VERSION:
            raise PoseValidationError(
                "invalid_pose_output",
                f"schema_version must be {CAMERA_MAP_POSE_SCHEMA_VERSION}",
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "timestamp_ns": self.timestamp_ns,
            "frame_id": self.frame_id,
            "parent_frame": self.parent_frame,
            "T_map_camera": (
                []
                if self.T_map_camera is None
                else self.T_map_camera.tolist()
            ),
            "position_xyz": (
                []
                if self.position_xyz is None
                else self.position_xyz.tolist()
            ),
            "orientation_xyzw": (
                []
                if self.orientation_xyzw is None
                else self.orientation_xyzw.tolist()
            ),
            "pose_covariance": (
                []
                if self.pose_covariance is None
                else self.pose_covariance.reshape(-1).tolist()
            ),
            "localization_status": self.localization_status,
            "time_offset_ms": self.time_offset_ms,
            "calibration_id": self.calibration_id,
            "translation_unit": self.translation_unit,
            "map_units_per_meter": self.map_units_per_meter,
            "reason_codes": list(self.reason_codes),
        }


def _invalid_estimate(
    *,
    timestamp_ns: int,
    frame_id: str,
    parent_frame: str,
    calibration_id: str,
    translation_unit: str,
    map_units_per_meter: float | None,
    reason: str,
    time_offset_ms: float | None,
) -> CameraMapPoseEstimate:
    return CameraMapPoseEstimate(
        timestamp_ns=timestamp_ns,
        frame_id=frame_id,
        parent_frame=parent_frame,
        T_map_camera=None,
        pose_covariance=None,
        localization_status="invalid",
        time_offset_ms=time_offset_ms,
        calibration_id=calibration_id,
        translation_unit=translation_unit,
        map_units_per_meter=map_units_per_meter,
        reason_codes=(reason,),
    )


def estimate_camera_map_pose(
    samples: Sequence[VehiclePoseSample],
    timestamp_ns: int,
    calibration: CameraCalibration,
    policy: PoseInterpolationPolicy,
    *,
    expected_parent_frame: str = "map",
    allow_placeholder_calibration: bool = False,
) -> CameraMapPoseEstimate:
    """Interpolate vehicle pose, validate it, and compose Camera map pose.

    Failures return localization_status=invalid with empty transform,
    position, orientation, and covariance arrays.
    """

    query = _timestamp_ns(timestamp_ns)
    parent_hint = (
        samples[0].parent_frame if samples else expected_parent_frame
    )
    unit_hint = samples[0].translation_unit if samples else "m"
    scale_hint = samples[0].map_units_per_meter if samples else None
    try:
        if not allow_placeholder_calibration:
            try:
                calibration.require_runtime_calibration()
            except CalibrationError as exc:
                raise PoseValidationError(
                    "placeholder_calibration",
                    str(exc),
                ) from exc
        pose = interpolate_vehicle_pose(samples, query, policy)
        if pose.parent_frame != expected_parent_frame:
            raise PoseValidationError(
                "frame_mismatch",
                "vehicle pose parent does not match expected map frame",
                time_offset_ms=pose.time_offset_ms,
            )
        if pose.frame_id != calibration.vehicle_frame:
            raise PoseValidationError(
                "frame_mismatch",
                "T_map_vehicle child does not match T_vehicle_camera parent",
                time_offset_ms=pose.time_offset_ms,
            )

        extrinsic = calibration.T_vehicle_camera.copy()
        reasons = ["pose_interpolated" if pose.interpolated else "pose_exact"]
        if pose.translation_unit == "map_unit":
            assert pose.map_units_per_meter is not None
            extrinsic[:3, 3] *= pose.map_units_per_meter
            reasons.append("extrinsic_translation_scaled_to_map_units")
        composed = pose.T_map_vehicle @ extrinsic
        composed = validate_rigid_transform(composed, "T_map_camera")
        camera_covariance = _camera_covariance(
            pose.pose_covariance,
            pose.T_map_vehicle,
            extrinsic,
        )
        status = _status_from_covariance(
            camera_covariance,
            pose.translation_unit,
            pose.map_units_per_meter,
            pose.localization_status,
            policy,
        )
        if status == "degraded":
            reasons.append("localization_degraded")
        return CameraMapPoseEstimate(
            timestamp_ns=query,
            frame_id=calibration.camera_frame,
            parent_frame=pose.parent_frame,
            T_map_camera=composed,
            pose_covariance=camera_covariance,
            localization_status=status,
            time_offset_ms=pose.time_offset_ms,
            calibration_id=calibration.calibration_id,
            translation_unit=pose.translation_unit,
            map_units_per_meter=pose.map_units_per_meter,
            reason_codes=tuple(reasons),
        )
    except CalibrationError:
        reason = "calibration_invalid"
        offset = None
    except PoseValidationError as exc:
        reason = exc.code
        offset = exc.time_offset_ms
    return _invalid_estimate(
        timestamp_ns=query,
        frame_id=calibration.camera_frame,
        parent_frame=parent_hint,
        calibration_id=calibration.calibration_id,
        translation_unit=unit_hint,
        map_units_per_meter=scale_hint,
        reason=reason,
        time_offset_ms=offset,
    )


def vehicle_pose_samples_from_records(
    records: Iterable[Mapping[str, Any]],
) -> tuple[VehiclePoseSample, ...]:
    """Parse strict records; missing covariance/status/unit fields are errors."""

    return tuple(VehiclePoseSample.from_mapping(row) for row in records)


__all__ = [
    "CAMERA_MAP_POSE_SCHEMA_VERSION",
    "LOCALIZATION_STATUSES",
    "TRANSLATION_UNITS",
    "CameraMapPoseEstimate",
    "InterpolatedVehiclePose",
    "PoseInterpolationPolicy",
    "PoseValidationError",
    "VehiclePoseSample",
    "estimate_camera_map_pose",
    "interpolate_vehicle_pose",
    "vehicle_pose_samples_from_records",
]
