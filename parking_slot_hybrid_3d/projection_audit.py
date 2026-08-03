"""Build a fail-closed camera projection audit from independent pixel labels.

The builder never synthesizes ``observed_pixel_uv``.  It consumes an
independently produced reference set, projects its LiDAR/map 3D points with
the exact calibration bytes under audit, and derives the status from fixed
coverage and error thresholds.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import unquote, urlparse

import numpy as np
from PIL import Image, UnidentifiedImageError

from .camera import (
    CAMERA_PROJECTION_AUDIT_BUILDER_NAME,
    CAMERA_PROJECTION_AUDIT_BUILDER_VERSION,
    CAMERA_PROJECTION_AUDIT_MAXIMUM_ERROR_PX,
    CAMERA_PROJECTION_AUDIT_MAXIMUM_CAMERA_LIDAR_DELTA_SEC,
    CAMERA_PROJECTION_AUDIT_MAXIMUM_RMSE_PX,
    CAMERA_PROJECTION_AUDIT_MINIMUM_REFERENCES_PER_ZONE,
    CAMERA_PROJECTION_AUDIT_MINIMUM_TOTAL_REFERENCES,
    CAMERA_PROJECTION_AUDIT_POLICY,
    CAMERA_PROJECTION_AUDIT_SCHEMA_VERSION,
    CAMERA_PROJECTION_AUDIT_SOURCE_KINDS,
    CAMERA_PROJECTION_AUDIT_ZONES,
    CAMERA_PROJECTION_REFERENCE_SET_SCHEMA_VERSION,
    CameraCalibration,
    load_camera_calibration,
    projection_audit_failure_codes,
)


MAX_REFERENCE_SET_BYTES = 32 * 1024 * 1024
MAX_ANNOTATION_SOURCE_BYTES = 64 * 1024 * 1024
MAX_AUDIT_IMAGE_BYTES = 64 * 1024 * 1024
MAX_AUDIT_IMAGE_PIXELS = 20_000_000
MAX_POINT_SOURCE_BYTES = 256 * 1024 * 1024
MAX_POSE_SOURCE_BYTES = 8 * 1024 * 1024
SUPPORTED_AUDIT_IMAGE_FORMATS = frozenset({"PNG", "JPEG"})
CAMERA_PROJECTION_POSE_SOURCE_SCHEMA_VERSION = "camera-projection-pose-source/1.0"


def _sha256_bytes(content: bytes) -> str:
    return "sha256:" + hashlib.sha256(content).hexdigest()


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_identity(value: Any, name: str) -> str:
    text = str(value or "").lower()
    if not text.startswith("sha256:") or len(text) != 71:
        raise ValueError(f"{name} must be a sha256 identity")
    try:
        int(text[7:], 16)
    except ValueError as exc:
        raise ValueError(f"{name} must be a sha256 identity") from exc
    return text


def _positive_integer(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer")
    try:
        parsed = int(value)
        numeric = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if parsed <= 0 or not math.isfinite(numeric) or parsed != numeric:
        raise ValueError(f"{name} must be a positive integer")
    return parsed


def _nonnegative_number(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite non-negative number")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite non-negative number") from exc
    if not math.isfinite(parsed) or parsed < 0.0:
        raise ValueError(f"{name} must be a finite non-negative number")
    return parsed


def _nonnegative_integer(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a non-negative integer")
    try:
        parsed = int(value)
        numeric = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a non-negative integer") from exc
    if parsed < 0 or not math.isfinite(numeric) or parsed != numeric:
        raise ValueError(f"{name} must be a non-negative integer")
    return parsed


def _required_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _resolve_local_artifact_uri(base_path: Path, value: Any, name: str) -> Path:
    uri = _required_string(value, name)
    parsed = urlparse(uri)
    if parsed.query or parsed.fragment or parsed.params:
        raise ValueError(f"{name} cannot contain a query or fragment")
    if parsed.scheme:
        if parsed.scheme != "file" or parsed.netloc not in {"", "localhost"}:
            raise ValueError(f"{name} must be a local relative path or file URI")
        candidate = Path(unquote(parsed.path))
    else:
        if parsed.netloc:
            raise ValueError(f"{name} must be a local relative path or file URI")
        candidate = Path(unquote(parsed.path))
        if candidate.is_absolute():
            raise ValueError(f"{name} must use file:// for an absolute path")
        candidate = base_path.parent / candidate
    try:
        resolved = candidate.expanduser().resolve(strict=True)
    except OSError as exc:
        raise ValueError(f"{name} is unavailable") from exc
    if not resolved.is_file():
        raise ValueError(f"{name} must resolve to a regular file")
    return resolved


def _read_bounded_file(path: Path, *, maximum_bytes: int, name: str) -> bytes:
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise ValueError(f"{name} is unavailable") from exc
    if size <= 0:
        raise ValueError(f"{name} must not be empty")
    if size > maximum_bytes:
        raise ValueError(f"{name} exceeds the safe size limit")
    try:
        content = path.read_bytes()
    except OSError as exc:
        raise ValueError(f"{name} is unavailable") from exc
    if len(content) != size:
        raise ValueError(f"{name} changed while being read")
    return content


def _sha256_file_bounded(path: Path, *, maximum_bytes: int, name: str) -> str:
    try:
        before = path.stat()
    except OSError as exc:
        raise ValueError(f"{name} is unavailable") from exc
    if before.st_size <= 0:
        raise ValueError(f"{name} must not be empty")
    if before.st_size > maximum_bytes:
        raise ValueError(f"{name} exceeds the safe size limit")
    digest = hashlib.sha256()
    count = 0
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                count += len(chunk)
                digest.update(chunk)
        after = path.stat()
    except OSError as exc:
        raise ValueError(f"{name} is unavailable") from exc
    if (
        count != before.st_size
        or after.st_size != before.st_size
        or after.st_mtime_ns != before.st_mtime_ns
    ):
        raise ValueError(f"{name} changed while being read")
    return "sha256:" + digest.hexdigest()


@dataclass(frozen=True, slots=True)
class ProjectionAuditThresholds:
    """Acceptance policy for real-pixel camera reprojection references."""

    minimum_total_references: int = 15
    minimum_references_per_zone: int = 5
    maximum_rmse_px: float = 3.0
    maximum_error_px: float = 8.0
    maximum_camera_lidar_delta_sec: float = 0.04

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "minimum_total_references",
            _positive_integer(
                self.minimum_total_references, "minimum_total_references"
            ),
        )
        object.__setattr__(
            self,
            "minimum_references_per_zone",
            _positive_integer(
                self.minimum_references_per_zone,
                "minimum_references_per_zone",
            ),
        )
        object.__setattr__(
            self,
            "maximum_rmse_px",
            _nonnegative_number(self.maximum_rmse_px, "maximum_rmse_px"),
        )
        object.__setattr__(
            self,
            "maximum_error_px",
            _nonnegative_number(self.maximum_error_px, "maximum_error_px"),
        )
        object.__setattr__(
            self,
            "maximum_camera_lidar_delta_sec",
            _nonnegative_number(
                self.maximum_camera_lidar_delta_sec,
                "maximum_camera_lidar_delta_sec",
            ),
        )
        if self.maximum_error_px < self.maximum_rmse_px:
            raise ValueError("maximum_error_px must be at least maximum_rmse_px")
        if (
            self.minimum_total_references
            < CAMERA_PROJECTION_AUDIT_MINIMUM_TOTAL_REFERENCES
            or self.minimum_references_per_zone
            < CAMERA_PROJECTION_AUDIT_MINIMUM_REFERENCES_PER_ZONE
            or self.maximum_rmse_px > CAMERA_PROJECTION_AUDIT_MAXIMUM_RMSE_PX
            or self.maximum_error_px > CAMERA_PROJECTION_AUDIT_MAXIMUM_ERROR_PX
            or self.maximum_camera_lidar_delta_sec
            > CAMERA_PROJECTION_AUDIT_MAXIMUM_CAMERA_LIDAR_DELTA_SEC
        ):
            raise ValueError("thresholds may only tighten the production acceptance policy")

    def to_mapping(self) -> dict[str, float | int]:
        return {
            "minimum_total_references": self.minimum_total_references,
            "minimum_references_per_zone": self.minimum_references_per_zone,
            "maximum_rmse_px": self.maximum_rmse_px,
            "maximum_error_px": self.maximum_error_px,
            "maximum_camera_lidar_delta_sec": self.maximum_camera_lidar_delta_sec,
        }


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def _finite_vector(value: Any, size: int, name: str) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain finite numbers") from exc
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must contain exactly {size} finite numbers")
    return result


def _rigid_transform(value: Any, name: str) -> np.ndarray:
    try:
        transform = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain finite numbers") from exc
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ValueError(f"{name} must be a finite 4x4 matrix")
    if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1e-9):
        raise ValueError(f"{name} must be homogeneous")
    rotation = transform[:3, :3]
    if not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-6):
        raise ValueError(f"{name} rotation must be orthonormal")
    if not math.isclose(float(np.linalg.det(rotation)), 1.0, abs_tol=1e-6):
        raise ValueError(f"{name} rotation must be right-handed")
    return transform


def _point_lidar(
    reference: Mapping[str, Any],
    name: str,
    point_source: Mapping[str, Any],
) -> np.ndarray:
    has_lidar = "point_lidar_xyz_m" in reference
    has_map = "point_map_xyz_m" in reference
    if has_lidar == has_map:
        raise ValueError(
            f"{name} requires exactly one of point_lidar_xyz_m or point_map_xyz_m"
        )
    if has_lidar:
        if "lidar_from_map" in reference:
            raise ValueError(f"{name}.lidar_from_map is forbidden; use point source pose")
        return _finite_vector(reference["point_lidar_xyz_m"], 3, f"{name}.point_lidar_xyz_m")

    point_map = _finite_vector(reference["point_map_xyz_m"], 3, f"{name}.point_map_xyz_m")
    if "lidar_from_map" in reference:
        raise ValueError(f"{name}.lidar_from_map is forbidden; use point source pose")
    lidar_from_map = point_source.get("lidar_from_map")
    if lidar_from_map is None:
        raise ValueError(f"{name} map point requires a pose-bound point source")
    point_map_h = np.concatenate([point_map, [1.0]])
    return (np.asarray(lidar_from_map, dtype=np.float64) @ point_map_h)[:3]


def _project_lidar_point(
    point_lidar: np.ndarray,
    calibration: CameraCalibration,
) -> np.ndarray | None:
    point_h = np.concatenate([point_lidar, [1.0]])
    point_camera = (calibration.camera_from_lidar @ point_h)[:3]
    if not np.isfinite(point_camera).all() or point_camera[2] <= 1e-9:
        return None
    homogeneous = calibration.intrinsic_matrix @ point_camera
    uv = homogeneous[:2] / homogeneous[2]
    if not np.isfinite(uv).all():
        return None
    if not (
        0.0 <= uv[0] <= calibration.image_width_px - 1
        and 0.0 <= uv[1] <= calibration.image_height_px - 1
    ):
        return None
    return uv


def _image_zone(observed_uv: np.ndarray, image_width_px: int) -> str:
    normalized_u = float(observed_uv[0]) / float(image_width_px)
    if normalized_u < 1.0 / 3.0:
        return "left"
    if normalized_u < 2.0 / 3.0:
        return "center"
    return "right"


def _error_metrics(errors: Sequence[float]) -> dict[str, float | int | None]:
    if not errors:
        return {
            "count": 0,
            "mean_error_px": None,
            "rmse_px": None,
            "max_error_px": None,
        }
    values = np.asarray(errors, dtype=np.float64)
    return {
        "count": int(len(values)),
        "mean_error_px": round(float(np.mean(values)), 9),
        "rmse_px": round(float(np.sqrt(np.mean(values**2))), 9),
        "max_error_px": round(float(np.max(values)), 9),
    }


def _load_reference_set(path: Path) -> tuple[bytes, Mapping[str, Any]]:
    content = _read_bounded_file(
        path,
        maximum_bytes=MAX_REFERENCE_SET_BYTES,
        name="camera projection reference set",
    )
    try:
        payload = json.loads(content.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("camera projection reference JSON is invalid") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("camera projection reference set must be an object")
    if payload.get("schema_version") != CAMERA_PROJECTION_REFERENCE_SET_SCHEMA_VERSION:
        raise ValueError("unsupported camera projection reference schema_version")
    return content, payload


def _load_annotation_source(
    reference_set_path: Path,
    observation_source: Mapping[str, Any],
) -> tuple[Path, str]:
    source_path = _resolve_local_artifact_uri(
        reference_set_path,
        observation_source.get("artifact_uri"),
        "reference_set.observation_source.artifact_uri",
    )
    declared_sha256 = _sha256_identity(
        observation_source.get("artifact_sha256"),
        "reference_set.observation_source.artifact_sha256",
    )
    content = _read_bounded_file(
        source_path,
        maximum_bytes=MAX_ANNOTATION_SOURCE_BYTES,
        name="camera projection annotation source",
    )
    actual_sha256 = _sha256_bytes(content)
    if actual_sha256 != declared_sha256:
        raise ValueError(
            "reference set observation source artifact_sha256 does not match source bytes"
        )
    if source_path == reference_set_path:
        raise ValueError("annotation source must be distinct from the reference set")
    return source_path, actual_sha256


def _safely_decode_image(content: bytes, name: str) -> tuple[int, int]:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(content)) as opened:
                if opened.format not in SUPPORTED_AUDIT_IMAGE_FORMATS:
                    raise ValueError(f"{name} must be a PNG or JPEG image")
                if int(getattr(opened, "n_frames", 1)) != 1:
                    raise ValueError(f"{name} must contain exactly one image frame")
                width, height = opened.size
                if width <= 0 or height <= 0 or width * height > MAX_AUDIT_IMAGE_PIXELS:
                    raise ValueError(f"{name} exceeds the safe decoded pixel limit")
                opened.verify()
            with Image.open(io.BytesIO(content)) as decoded:
                decoded.load()
                if decoded.size != (width, height):
                    raise ValueError(f"{name} dimensions changed during decode")
    except ValueError:
        raise
    except (
        UnidentifiedImageError,
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
        OSError,
        SyntaxError,
    ) as exc:
        raise ValueError(f"{name} cannot be safely decoded") from exc
    return int(width), int(height)


def _load_image_registry(
    reference_set_path: Path,
    payload: Mapping[str, Any],
    calibration: CameraCalibration,
) -> tuple[dict[str, dict[str, Any]], str]:
    raw_images = payload.get("images")
    if not isinstance(raw_images, Sequence) or isinstance(raw_images, (str, bytes)):
        raise ValueError("reference_set.images must be an array")
    if not raw_images:
        raise ValueError("reference_set.images must contain at least one real image")

    images: dict[str, dict[str, Any]] = {}
    seen_paths: set[Path] = set()
    seen_hashes: set[str] = set()
    seen_frame_identities: set[tuple[int, float]] = set()
    digest_records: list[dict[str, Any]] = []
    for index, raw_image in enumerate(raw_images):
        name = f"images[{index}]"
        image = _mapping(raw_image, name)
        image_id = _required_string(image.get("image_id"), f"{name}.image_id")
        if image_id in images:
            raise ValueError(f"duplicate image registry image_id: {image_id}")
        image_path = _resolve_local_artifact_uri(
            reference_set_path,
            image.get("image_uri"),
            f"{name}.image_uri",
        )
        if image_path in seen_paths:
            raise ValueError(f"duplicate image registry path: {image_id}")
        declared_sha256 = _sha256_identity(
            image.get("image_sha256"), f"{name}.image_sha256"
        )
        if declared_sha256 in seen_hashes:
            raise ValueError(f"duplicate image registry content: {image_id}")
        declared_width = _positive_integer(
            image.get("decoded_width_px"), f"{name}.decoded_width_px"
        )
        declared_height = _positive_integer(
            image.get("decoded_height_px"), f"{name}.decoded_height_px"
        )
        if (
            declared_width != calibration.image_width_px
            or declared_height != calibration.image_height_px
        ):
            raise ValueError(
                f"{name} declared decoded dimensions do not match calibration"
            )
        camera_frame = _nonnegative_integer(
            image.get("camera_frame"), f"{name}.camera_frame"
        )
        camera_timestamp = _nonnegative_number(
            image.get("camera_timestamp_sec"), f"{name}.camera_timestamp_sec"
        )
        frame_identity = (camera_frame, camera_timestamp)
        if frame_identity in seen_frame_identities:
            raise ValueError(f"duplicate image registry frame identity: {image_id}")

        content = _read_bounded_file(
            image_path,
            maximum_bytes=MAX_AUDIT_IMAGE_BYTES,
            name=f"camera projection image {image_id}",
        )
        actual_sha256 = _sha256_bytes(content)
        if actual_sha256 != declared_sha256:
            raise ValueError(f"{name}.image_sha256 does not match image bytes")
        actual_width, actual_height = _safely_decode_image(content, name)
        if (actual_width, actual_height) != (declared_width, declared_height):
            raise ValueError(f"{name} decoded dimensions do not match registry")
        if (
            actual_width != calibration.image_width_px
            or actual_height != calibration.image_height_px
        ):
            raise ValueError(f"{name} decoded dimensions do not match calibration")

        record = {
            "camera_frame": camera_frame,
            "camera_timestamp_sec": camera_timestamp,
            "decoded_height_px": actual_height,
            "decoded_width_px": actual_width,
            "image_id": image_id,
            "image_path": image_path,
            "image_sha256": actual_sha256,
        }
        images[image_id] = record
        digest_records.append(
            {
                key: value
                for key, value in record.items()
                if key != "image_path"
            }
        )
        seen_paths.add(image_path)
        seen_hashes.add(actual_sha256)
        seen_frame_identities.add(frame_identity)

    registry_sha256 = _sha256_bytes(_canonical_json_bytes(digest_records))
    return images, registry_sha256


def _load_pose_source(
    reference_set_path: Path,
    source: Mapping[str, Any],
    *,
    name: str,
    lidar_frame: int,
    lidar_timestamp_sec: float,
) -> tuple[Path, str, np.ndarray]:
    pose_path = _resolve_local_artifact_uri(
        reference_set_path,
        source.get("pose_uri"),
        f"{name}.pose_uri",
    )
    declared_sha256 = _sha256_identity(
        source.get("pose_sha256"), f"{name}.pose_sha256"
    )
    content = _read_bounded_file(
        pose_path,
        maximum_bytes=MAX_POSE_SOURCE_BYTES,
        name=f"camera projection pose source {source.get('point_source_id', '')}",
    )
    actual_sha256 = _sha256_bytes(content)
    if actual_sha256 != declared_sha256:
        raise ValueError(f"{name}.pose_sha256 does not match pose source bytes")
    try:
        payload = json.loads(content.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{name} pose source JSON is invalid") from exc
    if not isinstance(payload, Mapping):
        raise ValueError(f"{name} pose source must be an object")
    if payload.get("schema_version") != CAMERA_PROJECTION_POSE_SOURCE_SCHEMA_VERSION:
        raise ValueError(f"{name} pose source schema_version is unsupported")
    if _nonnegative_integer(payload.get("lidar_frame"), "pose_source.lidar_frame") != lidar_frame:
        raise ValueError(f"{name} pose source lidar_frame mismatch")
    pose_timestamp = _nonnegative_number(
        payload.get("lidar_timestamp_sec"), "pose_source.lidar_timestamp_sec"
    )
    if pose_timestamp != lidar_timestamp_sec:
        raise ValueError(f"{name} pose source lidar_timestamp_sec mismatch")
    lidar_from_map = _rigid_transform(
        payload.get("lidar_from_map"), "pose_source.lidar_from_map"
    )
    return pose_path, actual_sha256, lidar_from_map


def _load_point_source_registry(
    reference_set_path: Path,
    payload: Mapping[str, Any],
) -> tuple[dict[str, dict[str, Any]], str]:
    raw_sources = payload.get("point_sources")
    if not isinstance(raw_sources, Sequence) or isinstance(raw_sources, (str, bytes)):
        raise ValueError("reference_set.point_sources must be an array")
    if not raw_sources:
        raise ValueError("reference_set.point_sources must contain at least one source")

    sources: dict[str, dict[str, Any]] = {}
    seen_lidar_paths: set[Path] = set()
    seen_lidar_hashes: set[str] = set()
    seen_pose_paths: set[Path] = set()
    seen_pose_hashes: set[str] = set()
    seen_frame_identities: set[tuple[int, float]] = set()
    digest_records: list[dict[str, Any]] = []
    for index, raw_source in enumerate(raw_sources):
        name = f"point_sources[{index}]"
        source = _mapping(raw_source, name)
        source_id = _required_string(
            source.get("point_source_id"), f"{name}.point_source_id"
        )
        if source_id in sources:
            raise ValueError(f"duplicate point source id: {source_id}")
        lidar_path = _resolve_local_artifact_uri(
            reference_set_path,
            source.get("lidar_uri"),
            f"{name}.lidar_uri",
        )
        if lidar_path in seen_lidar_paths:
            raise ValueError(f"duplicate point source LiDAR path: {source_id}")
        declared_lidar_sha256 = _sha256_identity(
            source.get("lidar_sha256"), f"{name}.lidar_sha256"
        )
        if declared_lidar_sha256 in seen_lidar_hashes:
            raise ValueError(f"duplicate point source LiDAR content: {source_id}")
        actual_lidar_sha256 = _sha256_file_bounded(
            lidar_path,
            maximum_bytes=MAX_POINT_SOURCE_BYTES,
            name=f"camera projection LiDAR source {source_id}",
        )
        if actual_lidar_sha256 != declared_lidar_sha256:
            raise ValueError(f"{name}.lidar_sha256 does not match LiDAR source bytes")
        lidar_frame = _nonnegative_integer(
            source.get("lidar_frame"), f"{name}.lidar_frame"
        )
        lidar_timestamp_sec = _nonnegative_number(
            source.get("lidar_timestamp_sec"), f"{name}.lidar_timestamp_sec"
        )
        frame_identity = (lidar_frame, lidar_timestamp_sec)
        if frame_identity in seen_frame_identities:
            raise ValueError(f"duplicate point source frame identity: {source_id}")

        has_pose_uri = "pose_uri" in source
        has_pose_sha256 = "pose_sha256" in source
        if has_pose_uri != has_pose_sha256:
            raise ValueError(f"{name} pose_uri and pose_sha256 must appear together")
        pose_path: Path | None = None
        pose_sha256: str | None = None
        lidar_from_map: np.ndarray | None = None
        if has_pose_uri:
            pose_path, pose_sha256, lidar_from_map = _load_pose_source(
                reference_set_path,
                source,
                name=name,
                lidar_frame=lidar_frame,
                lidar_timestamp_sec=lidar_timestamp_sec,
            )
            if pose_path == lidar_path or pose_path in seen_pose_paths:
                raise ValueError(f"duplicate or aliased point source pose path: {source_id}")
            if pose_sha256 in seen_pose_hashes:
                raise ValueError(f"duplicate point source pose content: {source_id}")
            if pose_sha256 == actual_lidar_sha256 or pose_sha256 in seen_lidar_hashes:
                raise ValueError(f"aliased LiDAR/pose source content: {source_id}")
            if actual_lidar_sha256 in seen_pose_hashes:
                raise ValueError(f"aliased LiDAR/pose source content: {source_id}")
            if pose_path in seen_lidar_paths or lidar_path in seen_pose_paths:
                raise ValueError(f"aliased LiDAR/pose source path: {source_id}")

        record: dict[str, Any] = {
            "lidar_frame": lidar_frame,
            "lidar_path": lidar_path,
            "lidar_sha256": actual_lidar_sha256,
            "lidar_timestamp_sec": lidar_timestamp_sec,
            "point_source_id": source_id,
            "pose_path": pose_path,
            "pose_sha256": pose_sha256,
            "lidar_from_map": lidar_from_map,
        }
        sources[source_id] = record
        digest_records.append(
            {
                "lidar_frame": lidar_frame,
                "lidar_sha256": actual_lidar_sha256,
                "lidar_timestamp_sec": lidar_timestamp_sec,
                "point_source_id": source_id,
                "pose_sha256": pose_sha256,
                "lidar_from_map": (
                    None if lidar_from_map is None else lidar_from_map.tolist()
                ),
            }
        )
        seen_lidar_paths.add(lidar_path)
        seen_lidar_hashes.add(actual_lidar_sha256)
        seen_frame_identities.add(frame_identity)
        if pose_path is not None:
            seen_pose_paths.add(pose_path)
        if pose_sha256 is not None:
            seen_pose_hashes.add(pose_sha256)

    registry_sha256 = _sha256_bytes(_canonical_json_bytes(digest_records))
    return sources, registry_sha256


def build_projection_audit(
    calibration_path: str | Path,
    reference_set_path: str | Path,
    *,
    dataset_id: str,
    thresholds: ProjectionAuditThresholds | None = None,
    default_image_size_px: tuple[int, int] = (1280, 720),
) -> dict[str, Any]:
    """Evaluate independent 3D-to-real-pixel references for one calibration.

    ``point_map_xyz_m`` references obtain their full rigid ``lidar_from_map``
    transform from the hash-bound pose artifact in ``point_sources``.  A raw
    per-reference transform or 2D pose approximation is deliberately rejected.
    """
    dataset_id = str(dataset_id or "")
    if not dataset_id:
        raise ValueError("dataset_id is required")
    policy = thresholds or ProjectionAuditThresholds()
    calibration = load_camera_calibration(
        calibration_path,
        default_image_size_px=default_image_size_px,
    )
    resolved_reference_set_path = Path(reference_set_path).expanduser().resolve(strict=True)
    if not resolved_reference_set_path.is_file():
        raise ValueError("camera projection reference path must be a file")
    reference_content, payload = _load_reference_set(resolved_reference_set_path)
    if str(payload.get("dataset_id", "")) != dataset_id:
        raise ValueError("reference set dataset_id does not match the requested dataset")
    declared_calibration_sha256 = _sha256_identity(
        payload.get("calibration_sha256"), "reference_set.calibration_sha256"
    )
    if declared_calibration_sha256 != calibration.source_sha256:
        raise ValueError("reference set calibration_sha256 does not match calibration bytes")

    observation_source = _mapping(
        payload.get("observation_source"), "reference_set.observation_source"
    )
    source_kind = str(observation_source.get("kind", ""))
    if source_kind not in CAMERA_PROJECTION_AUDIT_SOURCE_KINDS:
        raise ValueError(
            "reference set observation source must be independent manual/GT/calibration-target data"
        )
    source_path, source_artifact_sha256 = _load_annotation_source(
        resolved_reference_set_path,
        observation_source,
    )
    images, image_registry_sha256 = _load_image_registry(
        resolved_reference_set_path,
        payload,
        calibration,
    )
    point_sources, point_source_registry_sha256 = _load_point_source_registry(
        resolved_reference_set_path,
        payload,
    )
    if source_path in {record["image_path"] for record in images.values()}:
        raise ValueError("annotation source must be distinct from every registered image")
    if source_artifact_sha256 in {
        str(record["image_sha256"]) for record in images.values()
    }:
        raise ValueError("annotation source bytes must be distinct from registered images")
    provenance_paths = {
        path
        for record in point_sources.values()
        for path in (record["lidar_path"], record["pose_path"])
        if path is not None
    }
    if source_path in provenance_paths:
        raise ValueError("annotation source must be distinct from 3D/pose sources")
    provenance_hashes = {
        str(value)
        for record in point_sources.values()
        for value in (record["lidar_sha256"], record["pose_sha256"])
        if value is not None
    }
    if source_artifact_sha256 in provenance_hashes:
        raise ValueError("annotation source bytes must be distinct from 3D/pose sources")
    if provenance_paths.intersection(
        {record["image_path"] for record in images.values()}
    ):
        raise ValueError("image and 3D/pose source paths must be distinct")
    if provenance_hashes.intersection(
        {str(record["image_sha256"]) for record in images.values()}
    ):
        raise ValueError("image and 3D/pose source bytes must be distinct")

    raw_references = payload.get("references", [])
    if not isinstance(raw_references, Sequence) or isinstance(raw_references, (str, bytes)):
        raise ValueError("reference_set.references must be an array")
    errors_by_zone: dict[str, list[float]] = {
        zone: [] for zone in CAMERA_PROJECTION_AUDIT_ZONES
    }
    invalid_projection_count = 0
    seen_ids: set[str] = set()
    seen_observations: set[tuple[Any, ...]] = set()
    for index, raw_reference in enumerate(raw_references):
        name = f"references[{index}]"
        reference = _mapping(raw_reference, name)
        reference_id = _required_string(reference.get("id"), f"{name}.id")
        if reference_id in seen_ids:
            raise ValueError(f"duplicate reference id: {reference_id}")
        seen_ids.add(reference_id)
        if "image_sha256" in reference:
            raise ValueError(f"{name}.image_sha256 is forbidden; use image_id")
        image_id = _required_string(reference.get("image_id"), f"{name}.image_id")
        if image_id not in images:
            raise ValueError(f"{name}.image_id is not present in the image registry")
        image_record = images[image_id]
        point_source_id = _required_string(
            reference.get("point_source_id"), f"{name}.point_source_id"
        )
        if point_source_id not in point_sources:
            raise ValueError(
                f"{name}.point_source_id is not present in the point source registry"
            )
        point_source = point_sources[point_source_id]
        camera_lidar_delta_sec = abs(
            float(image_record["camera_timestamp_sec"])
            - float(point_source["lidar_timestamp_sec"])
        )
        if camera_lidar_delta_sec > policy.maximum_camera_lidar_delta_sec + 1e-12:
            raise ValueError(
                f"{name} camera/LiDAR timestamp delta exceeds audit threshold"
            )
        observed_uv = _finite_vector(
            reference.get("observed_pixel_uv"), 2, f"{name}.observed_pixel_uv"
        )
        if not (
            0.0 <= observed_uv[0] <= int(image_record["decoded_width_px"]) - 1
            and 0.0 <= observed_uv[1] <= int(image_record["decoded_height_px"]) - 1
        ):
            raise ValueError(f"{name}.observed_pixel_uv lies outside the decoded image")
        point_lidar = _point_lidar(reference, name, point_source)
        duplicate_key = (
            image_id,
            point_source_id,
            *(round(float(value), 12) for value in point_lidar),
        )
        if duplicate_key in seen_observations:
            raise ValueError(f"duplicate reference observation: {reference_id}")
        seen_observations.add(duplicate_key)

        predicted_uv = _project_lidar_point(point_lidar, calibration)
        if predicted_uv is None:
            invalid_projection_count += 1
            continue
        error_px = float(np.linalg.norm(predicted_uv - observed_uv))
        errors_by_zone[_image_zone(observed_uv, calibration.image_width_px)].append(error_px)

    all_errors = [error for zone in CAMERA_PROJECTION_AUDIT_ZONES for error in errors_by_zone[zone]]
    metrics: dict[str, Any] = {
        "reference_count": len(raw_references),
        "valid_projection_count": len(all_errors),
        "invalid_projection_count": invalid_projection_count,
        "overall": _error_metrics(all_errors),
        "zones": {
            zone: _error_metrics(errors_by_zone[zone])
            for zone in CAMERA_PROJECTION_AUDIT_ZONES
        },
    }
    threshold_mapping = policy.to_mapping()
    failures = projection_audit_failure_codes(metrics, threshold_mapping)
    return {
        "schema_version": CAMERA_PROJECTION_AUDIT_SCHEMA_VERSION,
        "status": "passed" if not failures else "untrusted",
        "dataset_id": dataset_id,
        "calibration_sha256": calibration.source_sha256,
        "image_size_px": [calibration.image_width_px, calibration.image_height_px],
        "reference_set_schema_version": CAMERA_PROJECTION_REFERENCE_SET_SCHEMA_VERSION,
        "reference_set_uri": resolved_reference_set_path.as_uri(),
        "reference_set_sha256": _sha256_bytes(reference_content),
        "image_count": len(images),
        "image_registry_sha256": image_registry_sha256,
        "point_source_count": len(point_sources),
        "point_source_registry_sha256": point_source_registry_sha256,
        "observation_source": {
            "kind": source_kind,
            "artifact_sha256": source_artifact_sha256,
        },
        "builder": {
            "name": CAMERA_PROJECTION_AUDIT_BUILDER_NAME,
            "version": CAMERA_PROJECTION_AUDIT_BUILDER_VERSION,
            "policy": CAMERA_PROJECTION_AUDIT_POLICY,
        },
        "thresholds": threshold_mapping,
        "metrics": metrics,
        "failures": list(failures),
    }


def write_projection_audit(path: str | Path, audit: Mapping[str, Any]) -> Path:
    """Persist a deterministic audit JSON artifact."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(
        audit,
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
        allow_nan=False,
    ) + "\n"
    output_path.write_text(encoded, encoding="utf-8")
    return output_path


__all__ = [
    "CAMERA_PROJECTION_POSE_SOURCE_SCHEMA_VERSION",
    "CAMERA_PROJECTION_REFERENCE_SET_SCHEMA_VERSION",
    "ProjectionAuditThresholds",
    "build_projection_audit",
    "write_projection_audit",
]
