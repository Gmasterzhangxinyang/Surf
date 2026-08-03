"""Strict Camera calibration loading and projection.

Transform notation is uniform throughout this module: T_A_B transforms a
point expressed in frame B into frame A.  The canonical extrinsic stored by
CameraCalibration is therefore T_vehicle_camera.  A source that stores
T_camera_vehicle must declare that direction explicitly; the reader then
inverts it.

No frame direction, distortion model, image size, unit, calibration identity,
or calibration time is inferred from a file name.  Placeholder calibrations
are accepted only when a caller explicitly opts into them, and are rejected by
default for runtime use.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
import math
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import numpy as np


CAMERA_CALIBRATION_SCHEMA_VERSION = "camera-calibration/2.0"
SUPPORTED_CAMERA_MODELS = frozenset({"fisheye", "pinhole"})
EXTRINSIC_DIRECTIONS = frozenset(
    {"T_vehicle_camera", "T_camera_vehicle"}
)


class CalibrationError(ValueError):
    """Calibration content is missing, ambiguous, or geometrically invalid."""


def _required_text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CalibrationError(f"{name} must be a non-empty string")
    return value.strip()


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise CalibrationError(f"{name} must be a positive integer")
    return value


def _finite_array(
    value: Any,
    shape: tuple[int, ...] | None,
    name: str,
) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise CalibrationError(f"{name} must contain numbers") from exc
    if shape is not None and array.shape != shape:
        raise CalibrationError(f"{name} must have shape {shape}")
    if not np.isfinite(array).all():
        raise CalibrationError(f"{name} must contain finite numbers")
    return array


def validate_rigid_transform(value: Any, name: str) -> np.ndarray:
    """Return a validated 4x4 proper rigid transform."""

    transform = _finite_array(value, (4, 4), name).copy()
    if not np.allclose(
        transform[3],
        np.asarray([0.0, 0.0, 0.0, 1.0]),
        atol=1e-9,
    ):
        raise CalibrationError(f"{name} must be homogeneous")
    rotation = transform[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-8):
        raise CalibrationError(f"{name} rotation must be orthonormal")
    if not math.isclose(
        float(np.linalg.det(rotation)),
        1.0,
        abs_tol=1e-8,
    ):
        raise CalibrationError(f"{name} rotation must be right-handed")
    transform.setflags(write=False)
    return transform


def invert_rigid_transform(value: Any) -> np.ndarray:
    """Invert a rigid transform without using a generic matrix inverse."""

    transform = validate_rigid_transform(value, "transform")
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    inverse = np.eye(4, dtype=np.float64)
    inverse[:3, :3] = rotation.T
    inverse[:3, 3] = -rotation.T @ translation
    inverse.setflags(write=False)
    return inverse


def _validate_calibration_time(value: Any) -> str:
    text = _required_text(value, "calibration_time")
    candidate = text[:-1] + "+00:00" if text.endswith("Z") else text
    try:
        datetime.fromisoformat(candidate)
    except ValueError as exc:
        raise CalibrationError(
            "calibration_time must be an ISO-8601 timestamp"
        ) from exc
    return text


def _normalise_distortion(
    camera_model: str,
    value: Any,
) -> np.ndarray:
    distortion = _finite_array(value, None, "D").reshape(-1).copy()
    if camera_model == "fisheye":
        if distortion.size != 4:
            raise CalibrationError(
                "fisheye D must contain exactly k1, k2, k3, k4"
            )
    elif distortion.size not in {4, 5, 8}:
        raise CalibrationError(
            "pinhole D must contain 4, 5, or 8 OpenCV coefficients"
        )
    distortion.setflags(write=False)
    return distortion


@dataclass(frozen=True, slots=True)
class CameraCalibration:
    """Canonical, frame-explicit Camera calibration.

    T_vehicle_camera maps Camera-frame metric points into the declared
    vehicle/body frame.  Its translation is always metres.  Consumers whose
    map pose uses CAD map units must explicitly scale this translation before
    composition.
    """

    camera_model: Literal["fisheye", "pinhole"]
    image_width: int
    image_height: int
    K: np.ndarray
    D: np.ndarray
    T_vehicle_camera: np.ndarray
    calibration_id: str
    calibration_time: str
    vehicle_frame: str
    camera_frame: str
    is_placeholder: bool
    extrinsic_translation_unit: str = "m"
    source_format: str = "json"
    schema_version: str = CAMERA_CALIBRATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        model = _required_text(self.camera_model, "camera_model").lower()
        if model not in SUPPORTED_CAMERA_MODELS:
            raise CalibrationError(
                "camera_model must be fisheye or pinhole"
            )
        object.__setattr__(self, "camera_model", model)
        object.__setattr__(
            self,
            "image_width",
            _positive_int(self.image_width, "image_width"),
        )
        object.__setattr__(
            self,
            "image_height",
            _positive_int(self.image_height, "image_height"),
        )
        intrinsic = _finite_array(self.K, (3, 3), "K").copy()
        if intrinsic[0, 0] <= 0.0 or intrinsic[1, 1] <= 0.0:
            raise CalibrationError("K focal lengths must be positive")
        if not np.allclose(
            intrinsic[2],
            np.asarray([0.0, 0.0, 1.0]),
            atol=1e-9,
        ):
            raise CalibrationError("K must have normalized last row")
        if abs(float(intrinsic[0, 1])) > 1e-12:
            raise CalibrationError(
                "non-zero intrinsic skew is not supported"
            )
        intrinsic.setflags(write=False)
        object.__setattr__(self, "K", intrinsic)
        object.__setattr__(
            self,
            "D",
            _normalise_distortion(model, self.D),
        )
        object.__setattr__(
            self,
            "T_vehicle_camera",
            validate_rigid_transform(
                self.T_vehicle_camera,
                "T_vehicle_camera",
            ),
        )
        object.__setattr__(
            self,
            "calibration_id",
            _required_text(self.calibration_id, "calibration_id"),
        )
        object.__setattr__(
            self,
            "calibration_time",
            _validate_calibration_time(self.calibration_time),
        )
        vehicle_frame = _required_text(
            self.vehicle_frame,
            "vehicle_frame",
        )
        camera_frame = _required_text(
            self.camera_frame,
            "camera_frame",
        )
        if vehicle_frame == camera_frame:
            raise CalibrationError(
                "vehicle_frame and camera_frame must differ"
            )
        object.__setattr__(self, "vehicle_frame", vehicle_frame)
        object.__setattr__(self, "camera_frame", camera_frame)
        if not isinstance(self.is_placeholder, bool):
            raise CalibrationError("is_placeholder must be boolean")
        if self.extrinsic_translation_unit != "m":
            raise CalibrationError(
                "extrinsic_translation_unit must explicitly be 'm'"
            )
        object.__setattr__(
            self,
            "source_format",
            _required_text(self.source_format, "source_format"),
        )
        if self.schema_version != CAMERA_CALIBRATION_SCHEMA_VERSION:
            raise CalibrationError(
                f"schema_version must be "
                f"{CAMERA_CALIBRATION_SCHEMA_VERSION}"
            )

    def require_runtime_calibration(self) -> None:
        if self.is_placeholder:
            raise CalibrationError(
                "placeholder calibration is forbidden at runtime"
            )

    @classmethod
    def from_mapping(
        cls,
        payload: Mapping[str, Any],
        *,
        allow_placeholder: bool = False,
    ) -> "CameraCalibration":
        """Parse an in-memory strict-schema mapping.

        This uses the same fail-closed validation as the path reader, so
        callers never need to serialize an in-memory calibration merely to
        validate it.
        """

        return camera_calibration_from_mapping(
            payload,
            allow_placeholder=allow_placeholder,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "camera_model": self.camera_model,
            "image_width": self.image_width,
            "image_height": self.image_height,
            "K": self.K.tolist(),
            "D": self.D.tolist(),
            "T_vehicle_camera": self.T_vehicle_camera.tolist(),
            "extrinsic_translation_unit": self.extrinsic_translation_unit,
            "calibration_id": self.calibration_id,
            "calibration_time": self.calibration_time,
            "vehicle_frame": self.vehicle_frame,
            "camera_frame": self.camera_frame,
            "is_placeholder": self.is_placeholder,
            "source_format": self.source_format,
        }

    def project_camera_points(
        self,
        points_camera_xyz: Any,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Project Camera-frame points with the declared OpenCV model.

        The fisheye branch implements OpenCV's equidistant polynomial model:
        theta_d = theta * (1 + k1*theta^2 + ... + k4*theta^8).
        This is deliberately not a linear horizontal-angle pixel mapping.
        Points on the 90-degree optical boundary (z=0) are supported; points
        behind the Camera are invalid.
        """

        points = _finite_array(
            points_camera_xyz,
            None,
            "points_camera_xyz",
        )
        if points.ndim != 2 or points.shape[1] != 3:
            raise CalibrationError(
                "points_camera_xyz must have shape [N, 3]"
            )
        if self.camera_model == "fisheye":
            return _project_fisheye_equidistant(points, self.K, self.D)
        return _project_pinhole_opencv(points, self.K, self.D)


def _project_fisheye_equidistant(
    points: np.ndarray,
    K: np.ndarray,
    D: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    uv = np.full((len(points), 2), np.nan, dtype=np.float64)
    rho = np.linalg.norm(points[:, :2], axis=1)
    z = points[:, 2]
    valid = (z >= 0.0) & ((rho > 1e-15) | (z > 0.0))
    if not np.any(valid):
        return uv, valid

    selected_rho = rho[valid]
    selected_z = z[valid]
    theta = np.arctan2(selected_rho, selected_z)
    theta2 = theta * theta
    k1, k2, k3, k4 = D
    theta_d = theta * (
        1.0
        + k1 * theta2
        + k2 * theta2**2
        + k3 * theta2**3
        + k4 * theta2**4
    )
    direction = np.zeros((len(theta), 2), dtype=np.float64)
    off_axis = selected_rho > 1e-15
    selected_xy = points[valid, :2]
    direction[off_axis] = (
        selected_xy[off_axis]
        / selected_rho[off_axis, np.newaxis]
    )
    distorted = direction * theta_d[:, np.newaxis]
    uv[valid, 0] = K[0, 0] * distorted[:, 0] + K[0, 2]
    uv[valid, 1] = K[1, 1] * distorted[:, 1] + K[1, 2]
    valid &= np.isfinite(uv).all(axis=1)
    return uv, valid


def _project_pinhole_opencv(
    points: np.ndarray,
    K: np.ndarray,
    D: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    uv = np.full((len(points), 2), np.nan, dtype=np.float64)
    z = points[:, 2]
    valid = z > 0.0
    if not np.any(valid):
        return uv, valid
    x = points[valid, 0] / z[valid]
    y = points[valid, 1] / z[valid]
    r2 = x * x + y * y
    coefficients = np.zeros(8, dtype=np.float64)
    coefficients[: len(D)] = D
    k1, k2, p1, p2, k3, k4, k5, k6 = coefficients
    numerator = 1.0 + k1 * r2 + k2 * r2**2 + k3 * r2**3
    denominator = 1.0 + k4 * r2 + k5 * r2**2 + k6 * r2**3
    safe = np.abs(denominator) > 1e-15
    radial = np.full_like(numerator, np.nan)
    radial[safe] = numerator[safe] / denominator[safe]
    x_distorted = (
        x * radial
        + 2.0 * p1 * x * y
        + p2 * (r2 + 2.0 * x * x)
    )
    y_distorted = (
        y * radial
        + p1 * (r2 + 2.0 * y * y)
        + 2.0 * p2 * x * y
    )
    uv[valid, 0] = K[0, 0] * x_distorted + K[0, 2]
    uv[valid, 1] = K[1, 1] * y_distorted + K[1, 2]
    valid &= np.isfinite(uv).all(axis=1)
    return uv, valid


def _matrix_from_payload(
    payload: Mapping[str, Any],
) -> np.ndarray:
    direct = payload.get("T_vehicle_camera")
    reverse = payload.get("T_camera_vehicle")
    if (direct is None) == (reverse is None):
        raise CalibrationError(
            "declare exactly one of T_vehicle_camera or T_camera_vehicle"
        )
    if direct is not None:
        return validate_rigid_transform(direct, "T_vehicle_camera")
    return invert_rigid_transform(
        validate_rigid_transform(reverse, "T_camera_vehicle")
    )


def camera_calibration_from_mapping(
    payload: Mapping[str, Any],
    *,
    allow_placeholder: bool = False,
    source_format: str = "strict-mapping",
) -> CameraCalibration:
    """Parse the strict calibration schema from an in-memory mapping."""

    if not isinstance(payload, Mapping):
        raise CalibrationError("calibration payload must be a mapping")
    required = {
        "schema_version",
        "camera_model",
        "image_width",
        "image_height",
        "K",
        "D",
        "calibration_id",
        "calibration_time",
        "vehicle_frame",
        "camera_frame",
        "is_placeholder",
        "extrinsic_translation_unit",
    }
    missing = sorted(required - payload.keys())
    if missing:
        raise CalibrationError(
            "calibration payload is missing: " + ", ".join(missing)
        )
    calibration = CameraCalibration(
        schema_version=payload["schema_version"],
        camera_model=payload["camera_model"],
        image_width=payload["image_width"],
        image_height=payload["image_height"],
        K=payload["K"],
        D=payload["D"],
        T_vehicle_camera=_matrix_from_payload(payload),
        calibration_id=payload["calibration_id"],
        calibration_time=payload["calibration_time"],
        vehicle_frame=payload["vehicle_frame"],
        camera_frame=payload["camera_frame"],
        is_placeholder=payload["is_placeholder"],
        extrinsic_translation_unit=payload[
            "extrinsic_translation_unit"
        ],
        source_format=source_format,
    )
    if not allow_placeholder:
        calibration.require_runtime_calibration()
    return calibration


def read_camera_calibration(
    path: str | Path,
    *,
    allow_placeholder: bool = False,
) -> CameraCalibration:
    """Read the strict JSON calibration schema.

    Legacy KITTI-like text files do not encode enough metadata and must use
    read_legacy_camera_calibration with explicit declarations.
    """

    calibration_path = Path(path)
    try:
        payload = json.loads(calibration_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CalibrationError(
            f"failed to read calibration JSON: {calibration_path}"
        ) from exc
    return camera_calibration_from_mapping(
        payload,
        allow_placeholder=allow_placeholder,
        source_format="strict-json",
    )


def _parse_numeric_rows(path: str | Path) -> dict[str, list[float]]:
    rows: dict[str, list[float]] = {}
    source = Path(path)
    try:
        lines = source.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise CalibrationError(f"failed to read {source}") from exc
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        name, separator, raw = stripped.partition(":")
        if not separator:
            raise CalibrationError(f"{source} contains a line without ':'")
        try:
            values = [float(value) for value in raw.split()]
        except ValueError as exc:
            raise CalibrationError(
                f"{source} field {name.strip()} is not numeric"
            ) from exc
        if not values or not np.isfinite(values).all():
            raise CalibrationError(
                f"{source} field {name.strip()} is invalid"
            )
        key = name.strip()
        if key in rows:
            raise CalibrationError(f"{source} repeats field {key}")
        rows[key] = values
    return rows


def _legacy_intrinsic(rows: Mapping[str, list[float]]) -> np.ndarray:
    values = rows.get("K", rows.get("P0"))
    if values is None:
        raise CalibrationError("legacy intrinsics require K or P0")
    if len(values) == 9:
        return np.asarray(values, dtype=np.float64).reshape(3, 3)
    if len(values) == 12:
        projection = np.asarray(values, dtype=np.float64).reshape(3, 4)
        if not np.allclose(projection[:, 3], 0.0, atol=1e-12):
            raise CalibrationError(
                "legacy P0 baseline terms are not a Camera K matrix"
            )
        return projection[:, :3]
    raise CalibrationError("legacy K/P0 must contain 9 or 12 values")


def _legacy_extrinsic(rows: Mapping[str, list[float]]) -> np.ndarray:
    if "Tr" in rows:
        values = rows["Tr"]
        if len(values) == 12:
            transform = np.eye(4, dtype=np.float64)
            transform[:3] = np.asarray(values).reshape(3, 4)
            return validate_rigid_transform(transform, "legacy Tr")
        if len(values) == 16:
            return validate_rigid_transform(
                np.asarray(values).reshape(4, 4),
                "legacy Tr",
            )
        raise CalibrationError("legacy Tr must contain 12 or 16 values")
    if "R" not in rows or "T" not in rows:
        raise CalibrationError("legacy extrinsics require Tr or R and T")
    if len(rows["R"]) != 9 or len(rows["T"]) != 3:
        raise CalibrationError(
            "legacy R/T must contain 9 and 3 values"
        )
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.asarray(rows["R"]).reshape(3, 3)
    transform[:3, 3] = np.asarray(rows["T"])
    return validate_rigid_transform(transform, "legacy R/T")


def read_legacy_camera_calibration(
    intrinsics_path: str | Path,
    extrinsics_path: str | Path,
    *,
    stored_extrinsic: Literal[
        "T_vehicle_camera",
        "T_camera_vehicle",
    ],
    camera_model: Literal["fisheye", "pinhole"],
    image_width: int,
    image_height: int,
    distortion_coefficients: Sequence[float],
    calibration_id: str,
    calibration_time: str,
    vehicle_frame: str,
    camera_frame: str,
    is_placeholder: bool,
    extrinsic_translation_unit: str,
    allow_placeholder: bool = False,
) -> CameraCalibration:
    """Read numeric legacy files only with all missing metadata declared.

    stored_extrinsic has no default.  In particular, this function never
    infers transform direction from a name such as velo_to_cam.
    """

    if stored_extrinsic not in EXTRINSIC_DIRECTIONS:
        raise CalibrationError(
            "stored_extrinsic must explicitly declare "
            "T_vehicle_camera or T_camera_vehicle"
        )
    intrinsic = _legacy_intrinsic(_parse_numeric_rows(intrinsics_path))
    stored = _legacy_extrinsic(_parse_numeric_rows(extrinsics_path))
    normalised = (
        stored
        if stored_extrinsic == "T_vehicle_camera"
        else invert_rigid_transform(stored)
    )
    calibration = CameraCalibration(
        camera_model=camera_model,
        image_width=image_width,
        image_height=image_height,
        K=intrinsic,
        D=distortion_coefficients,
        T_vehicle_camera=normalised,
        calibration_id=calibration_id,
        calibration_time=calibration_time,
        vehicle_frame=vehicle_frame,
        camera_frame=camera_frame,
        is_placeholder=is_placeholder,
        extrinsic_translation_unit=extrinsic_translation_unit,
        source_format="legacy-numeric-explicit-metadata",
    )
    if not allow_placeholder:
        calibration.require_runtime_calibration()
    return calibration


__all__ = [
    "CAMERA_CALIBRATION_SCHEMA_VERSION",
    "EXTRINSIC_DIRECTIONS",
    "SUPPORTED_CAMERA_MODELS",
    "CalibrationError",
    "CameraCalibration",
    "camera_calibration_from_mapping",
    "invert_rigid_transform",
    "read_camera_calibration",
    "read_legacy_camera_calibration",
    "validate_rigid_transform",
]
