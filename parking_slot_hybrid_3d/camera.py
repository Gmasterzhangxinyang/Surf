"""Production camera geometry for Part1-to-Part2 evidence handoff.

The module deliberately separates geometric projection from trust.  A valid
calibration is sufficient to calculate image geometry, but it is *not*
sufficient to grant a terminal-decision capability.  Trust additionally
requires a passing projection-audit manifest bound to both the dataset and the
exact calibration bytes.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import unquote, urlparse

import numpy as np

from .contracts import FrameRecord


CAMERA_CALIBRATION_SCHEMA_VERSION = "camera-calibration/1.0"
CAMERA_PROJECTION_AUDIT_SCHEMA_VERSION = "camera-projection-audit/2.0"
LEGACY_CAMERA_PROJECTION_AUDIT_SCHEMA_VERSION = "camera-projection-audit/1.0"
CAMERA_PROJECTION_REFERENCE_SET_SCHEMA_VERSION = "camera-projection-reference-set/2.0"
LEGACY_CAMERA_PROJECTION_REFERENCE_SET_SCHEMA_VERSION = "camera-projection-reference-set/1.0"
CAMERA_PROJECTION_AUDIT_BUILDER_NAME = "parking_slot_hybrid_3d.projection_audit"
CAMERA_PROJECTION_AUDIT_BUILDER_VERSION = "2.0"
CAMERA_PROJECTION_AUDIT_POLICY = "independent-pixel-reference-v2"
CAMERA_PROJECTION_AUDIT_ZONES = ("left", "center", "right")
CAMERA_PROJECTION_AUDIT_SOURCE_KINDS = frozenset(
    {"manual_pixel_labels", "ground_truth_correspondences", "calibration_target"}
)
CAMERA_PROJECTION_AUDIT_MINIMUM_TOTAL_REFERENCES = 15
CAMERA_PROJECTION_AUDIT_MINIMUM_REFERENCES_PER_ZONE = 5
CAMERA_PROJECTION_AUDIT_MAXIMUM_RMSE_PX = 3.0
CAMERA_PROJECTION_AUDIT_MAXIMUM_ERROR_PX = 8.0
CAMERA_PROJECTION_AUDIT_MAXIMUM_CAMERA_LIDAR_DELTA_SEC = 0.04
MAX_CAMERA_PROJECTION_AUDIT_BYTES = 4 * 1024 * 1024
MAX_CAMERA_PROJECTION_REFERENCE_SET_BYTES = 32 * 1024 * 1024

# This dataset records a camera pose in the LiDAR x-forward/y-left/z-up frame.
# Camera pixels use the conventional x-right/y-down/z-forward optical frame.
OPTICAL_FROM_CAMERA_BODY = np.asarray(
    [
        [0.0, -1.0, 0.0],
        [0.0, 0.0, -1.0],
        [1.0, 0.0, 0.0],
    ],
    dtype=np.float64,
)


def _sha256_bytes(content: bytes) -> str:
    return "sha256:" + hashlib.sha256(content).hexdigest()


def _readonly_array(value: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    result = array.copy()
    result.setflags(write=False)
    return result


def _positive_int(value: Any, name: str) -> int:
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


def _hash_string(value: Any, name: str) -> str:
    text = str(value or "")
    if not text.startswith("sha256:") or len(text) != 71:
        raise ValueError(f"{name} must be a sha256 identity")
    try:
        int(text[7:], 16)
    except ValueError as exc:
        raise ValueError(f"{name} must be a sha256 identity") from exc
    return text.lower()


def _read_bounded_local_file(path: Path, *, maximum_bytes: int, name: str) -> bytes:
    try:
        before = path.stat()
    except OSError as exc:
        raise ValueError(f"{name} is unavailable") from exc
    if not path.is_file():
        raise ValueError(f"{name} must be a regular file")
    if before.st_size <= 0:
        raise ValueError(f"{name} must not be empty")
    if before.st_size > maximum_bytes:
        raise ValueError(f"{name} exceeds the safe size limit")
    try:
        content = path.read_bytes()
        after = path.stat()
    except OSError as exc:
        raise ValueError(f"{name} is unavailable") from exc
    if (
        len(content) != before.st_size
        or after.st_size != before.st_size
        or after.st_mtime_ns != before.st_mtime_ns
    ):
        raise ValueError(f"{name} changed while being read")
    return content


def _resolve_local_reference_set_uri(audit_path: Path, value: Any) -> Path:
    """Resolve an audit-bound local reference URI without network access."""
    if not isinstance(value, str) or not value:
        raise ValueError("projection audit reference_set_uri is required")
    parsed = urlparse(value)
    if parsed.query or parsed.fragment or parsed.params:
        raise ValueError("projection audit reference_set_uri cannot contain query or fragment")
    if parsed.scheme:
        if parsed.scheme != "file" or parsed.netloc not in {"", "localhost"}:
            raise ValueError("projection audit reference_set_uri must be a local file URI")
        candidate = Path(unquote(parsed.path))
    else:
        if parsed.netloc:
            raise ValueError("projection audit reference_set_uri must be a local path")
        candidate = Path(unquote(parsed.path))
        if candidate.is_absolute():
            raise ValueError(
                "projection audit reference_set_uri must use file:// for an absolute path"
            )
        candidate = audit_path.parent / candidate
    try:
        resolved = candidate.expanduser().resolve(strict=True)
    except OSError as exc:
        raise ValueError("projection audit reference set is unavailable") from exc
    if not resolved.is_file():
        raise ValueError("projection audit reference set must be a file")
    return resolved


def _verify_bound_reference_set(
    path: Path,
    *,
    expected_sha256: str,
    expected_dataset_id: str,
    expected_calibration_sha256: str,
    expected_source_kind: str,
    expected_source_artifact_sha256: str,
    expected_image_count: int,
    expected_point_source_count: int,
) -> None:
    """Verify the referenced bytes and their internal source bindings."""
    content = _read_bounded_local_file(
        path,
        maximum_bytes=MAX_CAMERA_PROJECTION_REFERENCE_SET_BYTES,
        name="projection audit reference set",
    )
    if _sha256_bytes(content) != expected_sha256:
        raise ValueError("projection audit reference_set_sha256 does not match referenced bytes")
    try:
        payload = json.loads(content.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("projection audit reference set JSON is invalid") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("projection audit reference set must be an object")
    if payload.get("schema_version") != CAMERA_PROJECTION_REFERENCE_SET_SCHEMA_VERSION:
        raise ValueError("unsupported camera projection reference schema_version")
    if str(payload.get("dataset_id", "")) != expected_dataset_id:
        raise ValueError("projection audit reference set dataset_id mismatch")
    reference_calibration_sha256 = _hash_string(
        payload.get("calibration_sha256"),
        "reference_set.calibration_sha256",
    )
    if reference_calibration_sha256 != expected_calibration_sha256:
        raise ValueError("projection audit reference set calibration_sha256 mismatch")
    observation_source = payload.get("observation_source")
    if not isinstance(observation_source, Mapping):
        raise ValueError("projection audit reference set observation_source must be an object")
    if str(observation_source.get("kind", "")) != expected_source_kind:
        raise ValueError("projection audit reference set observation source kind mismatch")
    reference_source_artifact_sha256 = _hash_string(
        observation_source.get("artifact_sha256"),
        "reference_set.observation_source.artifact_sha256",
    )
    if reference_source_artifact_sha256 != expected_source_artifact_sha256:
        raise ValueError("projection audit reference set observation source artifact mismatch")
    raw_images = payload.get("images")
    if not isinstance(raw_images, Sequence) or isinstance(raw_images, (str, bytes)):
        raise ValueError("projection audit reference set images must be an array")
    if len(raw_images) != expected_image_count:
        raise ValueError("projection audit reference set image count mismatch")
    raw_point_sources = payload.get("point_sources")
    if not isinstance(raw_point_sources, Sequence) or isinstance(
        raw_point_sources, (str, bytes)
    ):
        raise ValueError("projection audit reference set point_sources must be an array")
    if len(raw_point_sources) != expected_point_source_count:
        raise ValueError("projection audit reference set point source count mismatch")


@dataclass(frozen=True, slots=True)
class CameraCalibration:
    """Normalized pinhole calibration.

    ``camera_from_lidar`` is a homogeneous transform from LiDAR metric XYZ to
    camera optical XYZ.  Distorted lenses are rejected rather than silently
    projected with a pinhole approximation.
    """

    schema_version: str
    camera_id: str
    image_width_px: int
    image_height_px: int
    intrinsic_matrix: np.ndarray
    camera_from_lidar: np.ndarray
    source_sha256: str
    source_format: str

    def __post_init__(self) -> None:
        if not self.camera_id:
            raise ValueError("camera_id is required")
        object.__setattr__(self, "image_width_px", _positive_int(self.image_width_px, "image_width_px"))
        object.__setattr__(self, "image_height_px", _positive_int(self.image_height_px, "image_height_px"))
        intrinsic = _readonly_array(self.intrinsic_matrix, (3, 3), "intrinsic_matrix")
        transform = _readonly_array(self.camera_from_lidar, (4, 4), "camera_from_lidar")
        if intrinsic[0, 0] <= 0.0 or intrinsic[1, 1] <= 0.0:
            raise ValueError("camera focal lengths must be positive")
        if not np.allclose(intrinsic[2], [0.0, 0.0, 1.0], atol=1e-9):
            raise ValueError("intrinsic_matrix must use a normalized last row")
        if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1e-9):
            raise ValueError("camera_from_lidar must be homogeneous")
        rotation = transform[:3, :3]
        if not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-6):
            raise ValueError("camera_from_lidar rotation must be orthonormal")
        if not math.isclose(float(np.linalg.det(rotation)), 1.0, abs_tol=1e-6):
            raise ValueError("camera_from_lidar rotation must be right-handed")
        object.__setattr__(self, "intrinsic_matrix", intrinsic)
        object.__setattr__(self, "camera_from_lidar", transform)
        object.__setattr__(self, "source_sha256", _hash_string(self.source_sha256, "source_sha256"))

    @property
    def fx(self) -> float:
        return float(self.intrinsic_matrix[0, 0])

    @property
    def fy(self) -> float:
        return float(self.intrinsic_matrix[1, 1])

    @property
    def cx(self) -> float:
        return float(self.intrinsic_matrix[0, 2])

    @property
    def cy(self) -> float:
        return float(self.intrinsic_matrix[1, 2])


def _audit_nonnegative_int(value: Any, name: str) -> int:
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


def _audit_positive_int(value: Any, name: str) -> int:
    parsed = _audit_nonnegative_int(value, name)
    if parsed <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return parsed


def _audit_nonnegative_number(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite non-negative number")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite non-negative number") from exc
    if not math.isfinite(parsed) or parsed < 0.0:
        raise ValueError(f"{name} must be a finite non-negative number")
    return parsed


def _audit_metric_group(value: Any, name: str) -> dict[str, float | int | None]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    count = _audit_nonnegative_int(value.get("count"), f"{name}.count")
    result: dict[str, float | int | None] = {"count": count}
    for key in ("mean_error_px", "rmse_px", "max_error_px"):
        raw = value.get(key)
        if count == 0:
            if raw is not None:
                raise ValueError(f"{name}.{key} must be null when count is zero")
            result[key] = None
        else:
            result[key] = _audit_nonnegative_number(raw, f"{name}.{key}")
    if count:
        mean = float(result["mean_error_px"])
        rmse = float(result["rmse_px"])
        maximum = float(result["max_error_px"])
        if mean > rmse + 1e-6 or rmse > maximum + 1e-6:
            raise ValueError(f"{name} error metrics are internally inconsistent")
    return result


def _validated_projection_audit_policy(
    metrics: Mapping[str, Any] | None,
    thresholds: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, float | int]]:
    if not isinstance(metrics, Mapping):
        raise ValueError("projection audit metrics must be an object")
    if not isinstance(thresholds, Mapping):
        raise ValueError("projection audit thresholds must be an object")

    parsed_thresholds: dict[str, float | int] = {
        "minimum_total_references": _audit_positive_int(
            thresholds.get("minimum_total_references"),
            "thresholds.minimum_total_references",
        ),
        "minimum_references_per_zone": _audit_positive_int(
            thresholds.get("minimum_references_per_zone"),
            "thresholds.minimum_references_per_zone",
        ),
        "maximum_rmse_px": _audit_nonnegative_number(
            thresholds.get("maximum_rmse_px"),
            "thresholds.maximum_rmse_px",
        ),
        "maximum_error_px": _audit_nonnegative_number(
            thresholds.get("maximum_error_px"),
            "thresholds.maximum_error_px",
        ),
        "maximum_camera_lidar_delta_sec": _audit_nonnegative_number(
            thresholds.get("maximum_camera_lidar_delta_sec"),
            "thresholds.maximum_camera_lidar_delta_sec",
        ),
    }
    if parsed_thresholds["maximum_error_px"] < parsed_thresholds["maximum_rmse_px"]:
        raise ValueError("maximum_error_px must be at least maximum_rmse_px")
    if (
        parsed_thresholds["minimum_total_references"]
        < CAMERA_PROJECTION_AUDIT_MINIMUM_TOTAL_REFERENCES
        or parsed_thresholds["minimum_references_per_zone"]
        < CAMERA_PROJECTION_AUDIT_MINIMUM_REFERENCES_PER_ZONE
        or parsed_thresholds["maximum_rmse_px"]
        > CAMERA_PROJECTION_AUDIT_MAXIMUM_RMSE_PX
        or parsed_thresholds["maximum_error_px"]
        > CAMERA_PROJECTION_AUDIT_MAXIMUM_ERROR_PX
        or parsed_thresholds["maximum_camera_lidar_delta_sec"]
        > CAMERA_PROJECTION_AUDIT_MAXIMUM_CAMERA_LIDAR_DELTA_SEC
    ):
        raise ValueError("projection audit thresholds weaken the production acceptance policy")

    reference_count = _audit_nonnegative_int(
        metrics.get("reference_count"), "metrics.reference_count"
    )
    valid_count = _audit_nonnegative_int(
        metrics.get("valid_projection_count"), "metrics.valid_projection_count"
    )
    invalid_count = _audit_nonnegative_int(
        metrics.get("invalid_projection_count"), "metrics.invalid_projection_count"
    )
    if reference_count != valid_count + invalid_count:
        raise ValueError("projection audit reference counts are inconsistent")

    overall = _audit_metric_group(metrics.get("overall"), "metrics.overall")
    if overall["count"] != valid_count:
        raise ValueError("projection audit overall count is inconsistent")
    raw_zones = metrics.get("zones")
    if not isinstance(raw_zones, Mapping):
        raise ValueError("metrics.zones must be an object")
    if set(raw_zones) != set(CAMERA_PROJECTION_AUDIT_ZONES):
        raise ValueError("projection audit must contain exactly left/center/right metrics")
    zones = {
        zone: _audit_metric_group(raw_zones[zone], f"metrics.zones.{zone}")
        for zone in CAMERA_PROJECTION_AUDIT_ZONES
    }
    if sum(int(group["count"]) for group in zones.values()) != valid_count:
        raise ValueError("projection audit zone counts are inconsistent")

    # The aggregate must be reproducible from the three image zones.  This
    # prevents a hand-edited overall score from hiding a bad edge region.
    if valid_count:
        combined_mean = sum(
            int(group["count"]) * float(group["mean_error_px"])
            for group in zones.values()
            if int(group["count"])
        ) / valid_count
        combined_rmse = math.sqrt(
            sum(
                int(group["count"]) * float(group["rmse_px"]) ** 2
                for group in zones.values()
                if int(group["count"])
            )
            / valid_count
        )
        combined_max = max(
            float(group["max_error_px"])
            for group in zones.values()
            if int(group["count"])
        )
        for observed, expected, name in (
            (overall["mean_error_px"], combined_mean, "mean_error_px"),
            (overall["rmse_px"], combined_rmse, "rmse_px"),
            (overall["max_error_px"], combined_max, "max_error_px"),
        ):
            if not math.isclose(float(observed), float(expected), rel_tol=1e-6, abs_tol=1e-6):
                raise ValueError(f"projection audit overall {name} is inconsistent with zones")

    parsed_metrics: dict[str, Any] = {
        "reference_count": reference_count,
        "valid_projection_count": valid_count,
        "invalid_projection_count": invalid_count,
        "overall": overall,
        "zones": zones,
    }
    return parsed_metrics, parsed_thresholds


def projection_audit_failure_codes(
    metrics: Mapping[str, Any] | None,
    thresholds: Mapping[str, Any] | None,
) -> tuple[str, ...]:
    """Validate v2 metrics and return the deterministic fail-closed result."""
    parsed_metrics, parsed_thresholds = _validated_projection_audit_policy(metrics, thresholds)
    failures: list[str] = []
    valid_count = int(parsed_metrics["valid_projection_count"])
    invalid_count = int(parsed_metrics["invalid_projection_count"])
    if valid_count < int(parsed_thresholds["minimum_total_references"]):
        failures.append("minimum_total_references_not_met")
    if invalid_count:
        failures.append("invalid_projection_references_present")
    for zone in CAMERA_PROJECTION_AUDIT_ZONES:
        group = parsed_metrics["zones"][zone]
        if int(group["count"]) < int(parsed_thresholds["minimum_references_per_zone"]):
            failures.append(f"minimum_{zone}_references_not_met")

    overall = parsed_metrics["overall"]
    if valid_count:
        if float(overall["rmse_px"]) > float(parsed_thresholds["maximum_rmse_px"]):
            failures.append("overall_rmse_exceeded")
        if float(overall["max_error_px"]) > float(parsed_thresholds["maximum_error_px"]):
            failures.append("overall_max_error_exceeded")
    for zone in CAMERA_PROJECTION_AUDIT_ZONES:
        group = parsed_metrics["zones"][zone]
        if not int(group["count"]):
            continue
        if float(group["rmse_px"]) > float(parsed_thresholds["maximum_rmse_px"]):
            failures.append(f"{zone}_rmse_exceeded")
        if float(group["max_error_px"]) > float(parsed_thresholds["maximum_error_px"]):
            failures.append(f"{zone}_max_error_exceeded")
    return tuple(failures)


@dataclass(frozen=True, slots=True)
class ProjectionAudit:
    schema_version: str
    status: str
    dataset_id: str
    calibration_sha256: str
    source_sha256: str
    image_width_px: int | None = None
    image_height_px: int | None = None
    reference_set_uri: str | None = None
    reference_set_path: Path | None = None
    reference_set_sha256: str | None = None
    image_count: int | None = None
    image_registry_sha256: str | None = None
    point_source_count: int | None = None
    point_source_registry_sha256: str | None = None
    observation_source_kind: str | None = None
    observation_source_artifact_sha256: str | None = None
    metrics: Mapping[str, Any] | None = None
    thresholds: Mapping[str, Any] | None = None
    failures: tuple[str, ...] = ()
    builder_name: str | None = None
    builder_version: str | None = None

    def __post_init__(self) -> None:
        if self.schema_version not in {
            CAMERA_PROJECTION_AUDIT_SCHEMA_VERSION,
            LEGACY_CAMERA_PROJECTION_AUDIT_SCHEMA_VERSION,
        }:
            raise ValueError("unsupported projection audit schema_version")
        if not self.dataset_id:
            raise ValueError("projection audit dataset_id is required")
        object.__setattr__(
            self,
            "calibration_sha256",
            _hash_string(self.calibration_sha256, "calibration_sha256"),
        )
        object.__setattr__(self, "source_sha256", _hash_string(self.source_sha256, "source_sha256"))

        # Version 1 audits only carried a caller-provided status.  They remain
        # loadable for an explicit migration path, but can never grant trust.
        if self.schema_version == LEGACY_CAMERA_PROJECTION_AUDIT_SCHEMA_VERSION:
            if self.status != "untrusted":
                raise ValueError("legacy projection audits must be untrusted")
            if self.failures != ("legacy_projection_audit_schema",):
                raise ValueError("legacy projection audit migration reason is required")
            return

        if self.status not in {"passed", "untrusted"}:
            raise ValueError("unsupported projection audit status")
        if not isinstance(self.reference_set_uri, str) or not self.reference_set_uri:
            raise ValueError("projection audit reference_set_uri is required")
        if self.reference_set_path is None:
            raise ValueError("projection audit resolved reference set path is required")
        object.__setattr__(self, "reference_set_path", Path(self.reference_set_path))
        if self.reference_set_sha256 is None:
            raise ValueError("projection audit reference_set_sha256 is required")
        if self.image_registry_sha256 is None:
            raise ValueError("projection audit image_registry_sha256 is required")
        if self.point_source_registry_sha256 is None:
            raise ValueError("projection audit point_source_registry_sha256 is required")
        if self.observation_source_artifact_sha256 is None:
            raise ValueError("projection audit observation source artifact hash is required")
        object.__setattr__(
            self,
            "reference_set_sha256",
            _hash_string(self.reference_set_sha256, "reference_set_sha256"),
        )
        object.__setattr__(
            self,
            "image_registry_sha256",
            _hash_string(self.image_registry_sha256, "image_registry_sha256"),
        )
        object.__setattr__(
            self,
            "point_source_registry_sha256",
            _hash_string(
                self.point_source_registry_sha256,
                "point_source_registry_sha256",
            ),
        )
        object.__setattr__(
            self,
            "observation_source_artifact_sha256",
            _hash_string(
                self.observation_source_artifact_sha256,
                "observation_source.artifact_sha256",
            ),
        )
        object.__setattr__(
            self,
            "image_count",
            _positive_int(self.image_count, "projection audit image_count"),
        )
        object.__setattr__(
            self,
            "point_source_count",
            _positive_int(
                self.point_source_count,
                "projection audit point_source_count",
            ),
        )
        object.__setattr__(
            self,
            "image_width_px",
            _positive_int(self.image_width_px, "projection audit image_width_px"),
        )
        object.__setattr__(
            self,
            "image_height_px",
            _positive_int(self.image_height_px, "projection audit image_height_px"),
        )
        if self.observation_source_kind not in CAMERA_PROJECTION_AUDIT_SOURCE_KINDS:
            raise ValueError("projection audit observation source is not independent")
        if self.builder_name != CAMERA_PROJECTION_AUDIT_BUILDER_NAME:
            raise ValueError("unsupported projection audit builder")
        if self.builder_version != CAMERA_PROJECTION_AUDIT_BUILDER_VERSION:
            raise ValueError("unsupported projection audit builder version")
        expected_failures = projection_audit_failure_codes(self.metrics, self.thresholds)
        if tuple(self.failures) != expected_failures:
            raise ValueError("projection audit failures do not match metrics and thresholds")
        expected_status = "passed" if not expected_failures else "untrusted"
        if self.status != expected_status:
            raise ValueError("projection audit status does not match evaluated metrics")


@dataclass(frozen=True, slots=True)
class CameraModel:
    """Calibration plus its fail-closed dataset trust decision."""

    calibration: CameraCalibration | None
    audit: ProjectionAudit | None
    calibration_audit_status: str
    trust_reasons: tuple[str, ...]
    bound_dataset_id: str | None = None

    def __post_init__(self) -> None:
        if self.calibration_audit_status == "passed":
            if self.calibration is None or self.audit is None:
                raise ValueError("passing camera trust requires calibration and audit")
            if self.audit.status != "passed":
                raise ValueError("passing camera trust requires a passing audit")
            if self.audit.calibration_sha256 != self.calibration.source_sha256:
                raise ValueError("passing camera trust requires the audited calibration")
            if (
                self.audit.image_width_px != self.calibration.image_width_px
                or self.audit.image_height_px != self.calibration.image_height_px
            ):
                raise ValueError("passing camera trust requires the audited image size")
            if not self.bound_dataset_id or self.audit.dataset_id != self.bound_dataset_id:
                raise ValueError("passing camera trust requires the audited dataset")

    @property
    def trusted(self) -> bool:
        return self.calibration_audit_status == "passed"

    @property
    def calibration_sha256(self) -> str | None:
        return None if self.calibration is None else self.calibration.source_sha256


@dataclass(frozen=True, slots=True)
class CameraSelectionConfig:
    lookback_frames: int = 400
    frame_stride: int = 5
    pose_prefilter_limit: int = 30
    max_frames: int = 5
    half_fov_deg: float = 40.0
    min_distance_m: float = 3.0
    max_distance_m: float = 20.0
    max_sync_delta_sec: float = 0.04
    min_finite_vertices: int = 3
    min_projected_area_px: float = 800.0
    min_bbox_width_px: float = 40.0
    min_bbox_height_px: float = 20.0
    occupied_visible_fraction: float = 0.60
    free_visible_fraction: float = 0.60

    def __post_init__(self) -> None:
        if self.lookback_frames < 0:
            raise ValueError("lookback_frames must be non-negative")
        if self.frame_stride <= 0 or self.pose_prefilter_limit < 0 or self.max_frames < 0:
            raise ValueError("camera selection counts must be non-negative and stride positive")
        finite = (
            self.half_fov_deg,
            self.min_distance_m,
            self.max_distance_m,
            self.max_sync_delta_sec,
            self.min_projected_area_px,
            self.min_bbox_width_px,
            self.min_bbox_height_px,
            self.occupied_visible_fraction,
            self.free_visible_fraction,
        )
        if not all(math.isfinite(float(value)) for value in finite):
            raise ValueError("camera selection thresholds must be finite")
        if not 0.0 <= self.occupied_visible_fraction <= self.free_visible_fraction <= 1.0:
            raise ValueError("visible-fraction thresholds are invalid")
        if not 0.0 <= self.min_distance_m <= self.max_distance_m:
            raise ValueError("distance thresholds are invalid")


@dataclass(frozen=True, slots=True)
class CameraFrameAssessment:
    frame: FrameRecord
    polygon_uv: tuple[tuple[float | None, float | None], ...]
    projection_quality: Mapping[str, Any]
    intended_capabilities: tuple[str, ...]
    effective_capabilities: tuple[str, ...]
    rejection_reasons: tuple[str, ...]

    @property
    def lidar_frame(self) -> int:
        return self.frame.frame_id


@dataclass(frozen=True, slots=True)
class CameraCandidateBatch:
    assessments: tuple[CameraFrameAssessment, ...]
    selected: tuple[CameraFrameAssessment, ...]


def _parse_image_size(payload: Mapping[str, Any], default: tuple[int, int]) -> tuple[int, int]:
    raw = payload.get("image_size_px", payload.get("image_size"))
    if isinstance(raw, Mapping):
        width = raw.get("width", raw.get("width_px"))
        height = raw.get("height", raw.get("height_px"))
        return _positive_int(width, "image width"), _positive_int(height, "image height")
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)) and len(raw) == 2:
        return _positive_int(raw[0], "image width"), _positive_int(raw[1], "image height")
    if "image_width_px" in payload or "image_height_px" in payload:
        return (
            _positive_int(payload.get("image_width_px"), "image_width_px"),
            _positive_int(payload.get("image_height_px"), "image_height_px"),
        )
    return default


def _parse_intrinsics(payload: Mapping[str, Any]) -> np.ndarray:
    raw = payload.get("intrinsic_matrix", payload.get("camera_matrix"))
    if raw is not None:
        return _readonly_array(raw, (3, 3), "intrinsic_matrix")
    intrinsics = payload.get("intrinsics")
    if not isinstance(intrinsics, Mapping):
        raise ValueError("camera calibration is missing intrinsics")
    try:
        fx = float(intrinsics["fx"])
        fy = float(intrinsics["fy"])
        cx = float(intrinsics["cx"])
        cy = float(intrinsics["cy"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("camera intrinsics must contain finite fx/fy/cx/cy") from exc
    matrix = np.asarray([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    if not np.isfinite(matrix).all():
        raise ValueError("camera intrinsics contain non-finite values")
    return matrix


def _camera_from_pose_in_lidar(pose: np.ndarray) -> np.ndarray:
    """Convert a camera-body pose in LiDAR axes to optical-from-LiDAR."""
    if not np.allclose(pose[3], [0.0, 0.0, 0.0, 1.0], atol=1e-9):
        raise ValueError("camera_pose_lidar must be homogeneous")
    camera_body_from_lidar = pose[:3, :3].T
    optical_from_lidar = OPTICAL_FROM_CAMERA_BODY @ camera_body_from_lidar
    camera_origin_lidar = pose[:3, 3]
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = optical_from_lidar
    transform[:3, 3] = -optical_from_lidar @ camera_origin_lidar
    return transform


def _parse_extrinsics(payload: Mapping[str, Any]) -> np.ndarray:
    raw = payload.get("camera_from_lidar")
    if raw is not None:
        return _readonly_array(raw, (4, 4), "camera_from_lidar")
    extrinsics = payload.get("extrinsics")
    if not isinstance(extrinsics, Mapping):
        raise ValueError("camera calibration is missing extrinsics")
    direct = extrinsics.get("camera_from_lidar")
    if direct is not None:
        return _readonly_array(direct, (4, 4), "camera_from_lidar")
    pose = extrinsics.get("camera_pose_lidar")
    if pose is not None:
        return _camera_from_pose_in_lidar(_readonly_array(pose, (4, 4), "camera_pose_lidar"))
    rotation = extrinsics.get("optical_from_lidar")
    origin = extrinsics.get("camera_origin_lidar_m")
    if rotation is None or origin is None:
        raise ValueError(
            "extrinsics require camera_from_lidar, camera_pose_lidar, or optical_from_lidar plus camera_origin_lidar_m"
        )
    optical_from_lidar = _readonly_array(rotation, (3, 3), "optical_from_lidar")
    camera_origin = _readonly_array(origin, (3,), "camera_origin_lidar_m")
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = optical_from_lidar
    transform[:3, 3] = -optical_from_lidar @ camera_origin
    return transform


def _parse_json_calibration(content: bytes, default_image_size_px: tuple[int, int]) -> CameraCalibration:
    try:
        payload = json.loads(content.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("camera calibration JSON is invalid") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("camera calibration must be an object")
    if payload.get("schema_version") != CAMERA_CALIBRATION_SCHEMA_VERSION:
        raise ValueError("unsupported camera calibration schema_version")
    distortion = payload.get("distortion")
    if distortion not in (None, {}, []):
        if not isinstance(distortion, Mapping) or distortion.get("model", "none") not in {"none", "pinhole"}:
            raise ValueError("distorted camera models are not supported")
        coefficients = distortion.get("coefficients", ())
        if not isinstance(coefficients, Sequence) or isinstance(coefficients, (str, bytes)):
            raise ValueError("distortion coefficients must be an array")
        try:
            has_distortion = any(abs(float(value)) > 1e-12 for value in coefficients)
        except (TypeError, ValueError) as exc:
            raise ValueError("distortion coefficients must be finite numbers") from exc
        if has_distortion:
            raise ValueError("non-zero distortion coefficients are not supported")
    width, height = _parse_image_size(payload, default_image_size_px)
    return CameraCalibration(
        schema_version=CAMERA_CALIBRATION_SCHEMA_VERSION,
        camera_id=str(payload.get("camera_id", "front")),
        image_width_px=width,
        image_height_px=height,
        intrinsic_matrix=_parse_intrinsics(payload),
        camera_from_lidar=_parse_extrinsics(payload),
        source_sha256=_sha256_bytes(content),
        source_format="json",
    )


def _matrix_from_calib_line(values: list[float], name: str) -> np.ndarray:
    if len(values) == 16:
        matrix = np.asarray(values, dtype=np.float64).reshape(4, 4)
    elif len(values) == 12:
        matrix = np.eye(4, dtype=np.float64)
        matrix[:3] = np.asarray(values, dtype=np.float64).reshape(3, 4)
    else:
        raise ValueError(f"{name} must contain 12 or 16 values")
    return _readonly_array(matrix, (4, 4), name)


def _parse_text_calibration(content: bytes, default_image_size_px: tuple[int, int]) -> CameraCalibration:
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("camera calibration text is invalid") from exc
    rows: dict[str, list[float]] = {}
    for line in text.splitlines():
        if not line.strip():
            continue
        name, separator, raw_values = line.partition(":")
        if not separator:
            raise ValueError("camera calibration text line is missing ':'")
        try:
            values = [float(value) for value in raw_values.split()]
        except ValueError as exc:
            raise ValueError(f"camera calibration {name.strip()} contains a non-number") from exc
        if not values or not np.isfinite(values).all():
            raise ValueError(f"camera calibration {name.strip()} contains invalid values")
        key = name.strip()
        if key in rows:
            raise ValueError(f"camera calibration contains duplicate {key}")
        rows[key] = values
    if "P0" not in rows or "Tr" not in rows:
        raise ValueError("camera calibration text requires P0 and Tr")
    projection = rows["P0"]
    if len(projection) == 9:
        intrinsic = np.asarray(projection, dtype=np.float64).reshape(3, 3)
    elif len(projection) == 12:
        p0 = np.asarray(projection, dtype=np.float64).reshape(3, 4)
        if not np.allclose(p0[:, 3], 0.0, atol=1e-12):
            raise ValueError("P0 baseline terms are unsupported")
        intrinsic = p0[:, :3]
    else:
        raise ValueError("P0 must contain 9 or 12 values")
    camera_pose_lidar = _matrix_from_calib_line(rows["Tr"], "Tr")
    width, height = default_image_size_px
    return CameraCalibration(
        schema_version="camera-calibration-text/1.0",
        camera_id="front",
        image_width_px=width,
        image_height_px=height,
        intrinsic_matrix=intrinsic,
        camera_from_lidar=_camera_from_pose_in_lidar(camera_pose_lidar),
        source_sha256=_sha256_bytes(content),
        source_format="p0-tr-text",
    )


def load_camera_calibration(
    path: str | Path,
    *,
    default_image_size_px: tuple[int, int] = (1280, 720),
) -> CameraCalibration:
    """Load strict JSON or the dataset's ``P0``/``Tr`` ``calib.txt`` format."""
    calibration_path = Path(path)
    content = calibration_path.read_bytes()
    default = (
        _positive_int(default_image_size_px[0], "default image width"),
        _positive_int(default_image_size_px[1], "default image height"),
    )
    if calibration_path.suffix.lower() == ".json" or content.lstrip().startswith(b"{"):
        return _parse_json_calibration(content, default)
    return _parse_text_calibration(content, default)


def _verify_projection_audit_recomputed(
    audit: ProjectionAudit,
    calibration_path: str | Path,
    *,
    default_image_size_px: tuple[int, int],
) -> None:
    """Rebuild every decision-bearing audit field from source bytes."""
    if audit.schema_version != CAMERA_PROJECTION_AUDIT_SCHEMA_VERSION:
        return
    if audit.reference_set_path is None or not isinstance(audit.thresholds, Mapping):
        raise ValueError("projection audit cannot be recomputed")

    # Lazy import avoids the camera -> projection_audit -> camera module cycle.
    from .projection_audit import ProjectionAuditThresholds, build_projection_audit

    thresholds = ProjectionAuditThresholds(
        minimum_total_references=audit.thresholds.get("minimum_total_references"),
        minimum_references_per_zone=audit.thresholds.get("minimum_references_per_zone"),
        maximum_rmse_px=audit.thresholds.get("maximum_rmse_px"),
        maximum_error_px=audit.thresholds.get("maximum_error_px"),
        maximum_camera_lidar_delta_sec=audit.thresholds.get(
            "maximum_camera_lidar_delta_sec"
        ),
    )
    recomputed = build_projection_audit(
        calibration_path,
        audit.reference_set_path,
        dataset_id=audit.dataset_id,
        thresholds=thresholds,
        default_image_size_px=default_image_size_px,
    )
    recorded = {
        "schema_version": audit.schema_version,
        "status": audit.status,
        "dataset_id": audit.dataset_id,
        "calibration_sha256": audit.calibration_sha256,
        "image_size_px": [audit.image_width_px, audit.image_height_px],
        "reference_set_schema_version": CAMERA_PROJECTION_REFERENCE_SET_SCHEMA_VERSION,
        "reference_set_sha256": audit.reference_set_sha256,
        "image_count": audit.image_count,
        "image_registry_sha256": audit.image_registry_sha256,
        "point_source_count": audit.point_source_count,
        "point_source_registry_sha256": audit.point_source_registry_sha256,
        "observation_source": {
            "kind": audit.observation_source_kind,
            "artifact_sha256": audit.observation_source_artifact_sha256,
        },
        "builder": {
            "name": audit.builder_name,
            "version": audit.builder_version,
            "policy": CAMERA_PROJECTION_AUDIT_POLICY,
        },
        "thresholds": audit.thresholds,
        "metrics": audit.metrics,
        "failures": list(audit.failures),
    }
    recomputed_decision = {
        key: recomputed[key]
        for key in recorded
    }
    if recorded != recomputed_decision:
        raise ValueError(
            "projection audit decision fields do not match recomputed reference projections"
        )


def _load_projection_audit_artifact(path: str | Path) -> ProjectionAudit:
    audit_path = Path(path)
    content = _read_bounded_local_file(
        audit_path,
        maximum_bytes=MAX_CAMERA_PROJECTION_AUDIT_BYTES,
        name="projection audit artifact",
    )
    try:
        payload = json.loads(content.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("projection audit JSON is invalid") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("projection audit must be an object")
    schema_version = str(payload.get("schema_version", ""))
    if schema_version == LEGACY_CAMERA_PROJECTION_AUDIT_SCHEMA_VERSION:
        # v1 allowed a bare, caller-authored ``status=passed``.  Preserve
        # readability of old artifacts while making the migration fail closed.
        return ProjectionAudit(
            schema_version=schema_version,
            status="untrusted",
            dataset_id=str(payload.get("dataset_id", "")),
            calibration_sha256=str(
                payload.get("calibration_sha256", payload.get("calibration_hash", ""))
            ),
            source_sha256=_sha256_bytes(content),
            failures=("legacy_projection_audit_schema",),
        )
    if schema_version != CAMERA_PROJECTION_AUDIT_SCHEMA_VERSION:
        raise ValueError("unsupported projection audit schema_version")
    if payload.get("reference_set_schema_version") != CAMERA_PROJECTION_REFERENCE_SET_SCHEMA_VERSION:
        raise ValueError("unsupported camera projection reference schema_version")
    observation_source = payload.get("observation_source")
    if not isinstance(observation_source, Mapping):
        raise ValueError("projection audit observation_source must be an object")
    builder = payload.get("builder")
    if not isinstance(builder, Mapping):
        raise ValueError("projection audit builder must be an object")
    if builder.get("policy") != CAMERA_PROJECTION_AUDIT_POLICY:
        raise ValueError("unsupported projection audit policy")
    raw_failures = payload.get("failures")
    if not isinstance(raw_failures, Sequence) or isinstance(raw_failures, (str, bytes)):
        raise ValueError("projection audit failures must be an array")
    failures = tuple(str(value) for value in raw_failures)
    if len(set(failures)) != len(failures) or any(not value for value in failures):
        raise ValueError("projection audit failures must be unique non-empty strings")
    image_size = payload.get("image_size_px")
    if (
        not isinstance(image_size, Sequence)
        or isinstance(image_size, (str, bytes))
        or len(image_size) != 2
    ):
        raise ValueError("projection audit image_size_px must contain width and height")
    reference_set_path = _resolve_local_reference_set_uri(
        audit_path,
        payload.get("reference_set_uri"),
    )
    audit = ProjectionAudit(
        schema_version=schema_version,
        status=str(payload.get("status", "")),
        dataset_id=str(payload.get("dataset_id", "")),
        calibration_sha256=str(payload.get("calibration_sha256", "")),
        source_sha256=_sha256_bytes(content),
        image_width_px=image_size[0],
        image_height_px=image_size[1],
        reference_set_uri=str(payload.get("reference_set_uri", "")),
        reference_set_path=reference_set_path,
        reference_set_sha256=str(payload.get("reference_set_sha256", "")),
        image_count=payload.get("image_count"),
        image_registry_sha256=str(payload.get("image_registry_sha256", "")),
        point_source_count=payload.get("point_source_count"),
        point_source_registry_sha256=str(
            payload.get("point_source_registry_sha256", "")
        ),
        observation_source_kind=str(observation_source.get("kind", "")),
        observation_source_artifact_sha256=str(
            observation_source.get("artifact_sha256", "")
        ),
        metrics=payload.get("metrics"),
        thresholds=payload.get("thresholds"),
        failures=failures,
        builder_name=str(builder.get("name", "")),
        builder_version=str(builder.get("version", "")),
    )
    _verify_bound_reference_set(
        reference_set_path,
        expected_sha256=str(audit.reference_set_sha256),
        expected_dataset_id=audit.dataset_id,
        expected_calibration_sha256=audit.calibration_sha256,
        expected_source_kind=str(audit.observation_source_kind),
        expected_source_artifact_sha256=str(
            audit.observation_source_artifact_sha256
        ),
        expected_image_count=int(audit.image_count),
        expected_point_source_count=int(audit.point_source_count),
    )
    return audit


def load_projection_audit(
    path: str | Path,
    *,
    calibration_path: str | Path | None = None,
    default_image_size_px: tuple[int, int] = (1280, 720),
) -> ProjectionAudit:
    """Load an audit only after replaying it against its source references.

    Version 2 cannot be returned as trusted metadata without the exact
    calibration file.  Callers that only need migration diagnostics for a
    legacy v1 artifact may omit ``calibration_path``; v1 remains untrusted.
    """
    audit = _load_projection_audit_artifact(path)
    if audit.schema_version == CAMERA_PROJECTION_AUDIT_SCHEMA_VERSION:
        if calibration_path is None:
            raise ValueError(
                "calibration_path is required to recompute a v2 projection audit"
            )
        _verify_projection_audit_recomputed(
            audit,
            calibration_path,
            default_image_size_px=default_image_size_px,
        )
    return audit


def load_camera_model(
    calibration_path: str | Path | None,
    *,
    dataset_id: str | None,
    projection_audit_path: str | Path | None = None,
    default_image_size_px: tuple[int, int] = (1280, 720),
) -> CameraModel:
    """Load camera geometry and resolve trust without ever failing open."""
    if calibration_path is None:
        return CameraModel(
            None,
            None,
            "missing",
            ("camera_calibration_missing",),
            bound_dataset_id=dataset_id,
        )
    try:
        calibration = load_camera_calibration(
            calibration_path,
            default_image_size_px=default_image_size_px,
        )
    except (OSError, ValueError):
        return CameraModel(
            None,
            None,
            "invalid",
            ("camera_calibration_invalid",),
            bound_dataset_id=dataset_id,
        )
    if projection_audit_path is None:
        return CameraModel(
            calibration,
            None,
            "missing",
            ("projection_audit_missing",),
            bound_dataset_id=dataset_id,
        )
    try:
        audit = _load_projection_audit_artifact(projection_audit_path)
    except (OSError, ValueError):
        return CameraModel(
            calibration,
            None,
            "invalid",
            ("projection_audit_invalid",),
            bound_dataset_id=dataset_id,
        )
    if (
        audit.schema_version == CAMERA_PROJECTION_AUDIT_SCHEMA_VERSION
        and audit.calibration_sha256 == calibration.source_sha256
    ):
        try:
            _verify_projection_audit_recomputed(
                audit,
                calibration_path,
                default_image_size_px=default_image_size_px,
            )
        except (OSError, ValueError):
            return CameraModel(
                calibration,
                None,
                "invalid",
                ("projection_audit_invalid",),
                bound_dataset_id=dataset_id,
            )
    reasons: list[str] = []
    status = audit.status
    if audit.status != "passed":
        reasons.append("projection_audit_not_passed")
    if not dataset_id:
        status = "dataset_unbound"
        reasons.append("projection_audit_dataset_unbound")
    elif audit.dataset_id != dataset_id:
        status = "dataset_mismatch"
        reasons.append("projection_audit_dataset_mismatch")
    if audit.calibration_sha256 != calibration.source_sha256:
        status = "calibration_mismatch"
        reasons.append("projection_audit_calibration_mismatch")
    if (
        audit.schema_version == CAMERA_PROJECTION_AUDIT_SCHEMA_VERSION
        and (
            audit.image_width_px != calibration.image_width_px
            or audit.image_height_px != calibration.image_height_px
        )
    ):
        status = "image_size_mismatch"
        reasons.append("projection_audit_image_size_mismatch")
    if reasons:
        return CameraModel(
            calibration,
            audit,
            status,
            tuple(reasons),
            bound_dataset_id=dataset_id,
        )
    return CameraModel(
        calibration,
        audit,
        "passed",
        (),
        bound_dataset_id=dataset_id,
    )


def relative_bearing_deg(ego_pose_map: Sequence[float], slot_center_map: Sequence[float]) -> float:
    pose = np.asarray(ego_pose_map, dtype=np.float64)
    center = np.asarray(slot_center_map, dtype=np.float64)
    if pose.shape[0] < 3 or center.shape != (2,) or not np.isfinite(pose[:3]).all() or not np.isfinite(center).all():
        raise ValueError("pose and slot center must be finite")
    delta = center - pose[:2]
    c = math.cos(float(pose[2]))
    s = math.sin(float(pose[2]))
    forward = c * delta[0] + s * delta[1]
    left = -s * delta[0] + c * delta[1]
    return float(math.degrees(math.atan2(float(left), float(forward))))


def project_map_polygon(
    polygon_map: np.ndarray,
    ego_pose_map: Sequence[float],
    map_units_per_meter: float,
    ground_z_lidar_m: float,
    calibration: CameraCalibration,
) -> tuple[np.ndarray, np.ndarray]:
    """Project a map-XY polygon at the measured LiDAR ground height."""
    polygon = np.asarray(polygon_map, dtype=np.float64)
    pose = np.asarray(ego_pose_map, dtype=np.float64)
    scale = float(map_units_per_meter)
    ground_z = float(ground_z_lidar_m)
    if polygon.ndim != 2 or polygon.shape[1] != 2 or len(polygon) < 3:
        raise ValueError("polygon_map must contain at least three XY vertices")
    if pose.shape[0] < 3 or not np.isfinite(pose[:3]).all():
        raise ValueError("ego_pose_map must contain finite x/y/yaw")
    if not np.isfinite(polygon).all() or not math.isfinite(scale) or scale <= 0.0 or not math.isfinite(ground_z):
        raise ValueError("projection inputs must be finite and scale positive")
    delta_m = (polygon - pose[:2]) / scale
    yaw = float(pose[2])
    c = math.cos(yaw)
    s = math.sin(yaw)
    xy_lidar = np.column_stack(
        [c * delta_m[:, 0] + s * delta_m[:, 1], -s * delta_m[:, 0] + c * delta_m[:, 1]]
    )
    lidar_xyz1 = np.column_stack(
        [xy_lidar, np.full(len(polygon), ground_z, dtype=np.float64), np.ones(len(polygon))]
    )
    camera_xyz = (calibration.camera_from_lidar @ lidar_xyz1.T).T[:, :3]
    depth = camera_xyz[:, 2]
    valid = np.isfinite(camera_xyz).all(axis=1) & (depth > 1e-6)
    uv = np.full((len(polygon), 2), np.nan, dtype=np.float64)
    uv[valid, 0] = calibration.fx * camera_xyz[valid, 0] / depth[valid] + calibration.cx
    uv[valid, 1] = calibration.fy * camera_xyz[valid, 1] / depth[valid] + calibration.cy
    return uv, depth


def _polygon_area(points: np.ndarray) -> float:
    if len(points) < 3:
        return 0.0
    x = points[:, 0]
    y = points[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) * 0.5)


def _clip_edge(points: np.ndarray, axis: int, boundary: float, keep_greater: bool) -> np.ndarray:
    if len(points) == 0:
        return points
    result: list[np.ndarray] = []
    previous = points[-1]
    previous_inside = bool(previous[axis] >= boundary) if keep_greater else bool(previous[axis] <= boundary)
    for current in points:
        current_inside = bool(current[axis] >= boundary) if keep_greater else bool(current[axis] <= boundary)
        if current_inside != previous_inside:
            denominator = current[axis] - previous[axis]
            fraction = 0.0 if abs(float(denominator)) < 1e-12 else (boundary - previous[axis]) / denominator
            intersection = previous + float(fraction) * (current - previous)
            result.append(intersection)
        if current_inside:
            result.append(current)
        previous = current
        previous_inside = current_inside
    return np.asarray(result, dtype=np.float64).reshape(-1, 2)


def _clip_to_image(points: np.ndarray, width: int, height: int) -> np.ndarray:
    clipped = points
    clipped = _clip_edge(clipped, 0, 0.0, True)
    clipped = _clip_edge(clipped, 0, float(width), False)
    clipped = _clip_edge(clipped, 1, 0.0, True)
    clipped = _clip_edge(clipped, 1, float(height), False)
    return clipped


def projection_quality(
    polygon_uv: np.ndarray,
    image_width_px: int,
    image_height_px: int,
) -> dict[str, Any]:
    """Return finite, clipped-area, bbox and visibility metrics for a polygon."""
    uv = np.asarray(polygon_uv, dtype=np.float64)
    if uv.ndim != 2 or uv.shape[1] != 2:
        raise ValueError("polygon_uv must be an Nx2 array")
    width = _positive_int(image_width_px, "image_width_px")
    height = _positive_int(image_height_px, "image_height_px")
    finite_mask = np.isfinite(uv).all(axis=1)
    finite = uv[finite_mask]
    raw_area = _polygon_area(finite)
    clipped = _clip_to_image(finite, width, height) if len(finite) >= 3 else np.empty((0, 2))
    visible_area = _polygon_area(clipped)
    visible_fraction = min(1.0, visible_area / raw_area) if raw_area > 1e-9 else 0.0
    bbox: list[float] | None = None
    if len(clipped):
        bbox = [
            float(np.min(clipped[:, 0])),
            float(np.min(clipped[:, 1])),
            float(np.max(clipped[:, 0])),
            float(np.max(clipped[:, 1])),
        ]
    inside = finite[
        (finite[:, 0] >= 0.0)
        & (finite[:, 0] <= width)
        & (finite[:, 1] >= 0.0)
        & (finite[:, 1] <= height)
    ]
    return {
        "finite_vertices": int(finite_mask.sum()),
        "in_image_vertices": int(len(inside)),
        "projected_area_px": float(visible_area),
        "raw_projected_area_px": float(raw_area),
        "bbox_px": bbox,
        "visible_fraction": float(visible_fraction),
    }


def _json_uv(uv: np.ndarray) -> tuple[tuple[float | None, float | None], ...]:
    return tuple(
        (
            float(point[0]) if np.isfinite(point[0]) else None,
            float(point[1]) if np.isfinite(point[1]) else None,
        )
        for point in uv
    )


def assess_camera_frame(
    frame: FrameRecord,
    polygon_map: np.ndarray,
    slot_center_map: Sequence[float],
    map_units_per_meter: float,
    ground_z_lidar_m: float,
    camera_model: CameraModel,
    *,
    anchor_frame: int,
    anchor_timestamp: float,
    sync_source: str | None,
    config: CameraSelectionConfig | None = None,
) -> CameraFrameAssessment:
    """Project and gate one camera candidate without hiding untrusted geometry."""
    calibration = camera_model.calibration
    if calibration is None:
        raise ValueError("a parsed camera calibration is required to assess geometry")
    settings = config or CameraSelectionConfig()
    pose = np.asarray([frame.map_x, frame.map_y, frame.map_yaw], dtype=np.float64)
    uv, _ = project_map_polygon(
        polygon_map,
        pose,
        map_units_per_meter,
        ground_z_lidar_m,
        calibration,
    )
    quality = projection_quality(uv, calibration.image_width_px, calibration.image_height_px)
    center = np.asarray(slot_center_map, dtype=np.float64)
    bearing = relative_bearing_deg(pose, center)
    distance = float(np.linalg.norm(center - pose[:2]) / float(map_units_per_meter))
    quality.update(
        {
            "status": "trusted" if camera_model.trusted else "untrusted",
            "depth_capability": "relative_only",
            "sync_source": sync_source or "unassessed",
            "bearing_deg": bearing,
            "distance_m": distance,
            "calibration_sha256": calibration.source_sha256,
            "calibration_audit_status": camera_model.calibration_audit_status,
            "polygon_uv": [list(point) for point in _json_uv(uv)],
        }
    )
    reasons: list[str] = []
    if not camera_model.trusted:
        reasons.append("calibration_untrusted")
    if frame.frame_id > int(anchor_frame):
        reasons.append("after_anchor_frame")
    if frame.lidar_timestamp is None or frame.camera_timestamp is None or frame.camera_lidar_dt_sec is None:
        reasons.append("missing_camera_timestamps")
    else:
        if frame.lidar_timestamp > float(anchor_timestamp) or frame.camera_timestamp > float(anchor_timestamp):
            reasons.append("after_anchor_timestamp")
        recomputed_delta = float(frame.camera_timestamp - frame.lidar_timestamp)
        if not math.isclose(recomputed_delta, frame.camera_lidar_dt_sec, abs_tol=1e-6):
            reasons.append("camera_sync_delta_mismatch")
        if abs(frame.camera_lidar_dt_sec) > settings.max_sync_delta_sec:
            reasons.append("camera_sync_delta_too_large")
    if sync_source not in {"corrected", "native"}:
        reasons.append("unsupported_or_missing_sync_metadata")
    if not frame.camera_match_valid:
        reasons.append("camera_match_invalid")
    if frame.camera_frame is None:
        reasons.append("camera_frame_missing")
    elif frame.camera_frame == frame.frame_id:
        reasons.append("sensor_frame_ids_not_distinct")
    if frame.camera_image_path is None or not frame.camera_image_path.is_file():
        reasons.append("camera_image_missing")
    if abs(bearing) > settings.half_fov_deg:
        reasons.append("outside_safe_fov")
    if not settings.min_distance_m <= distance <= settings.max_distance_m:
        reasons.append("outside_distance_range")
    if int(quality["finite_vertices"]) < settings.min_finite_vertices:
        reasons.append("insufficient_projected_vertices")
    if float(quality["projected_area_px"]) < settings.min_projected_area_px:
        reasons.append("projected_area_too_small")
    bbox = quality["bbox_px"]
    if bbox is None:
        reasons.append("missing_projected_bbox")
    else:
        if float(bbox[2] - bbox[0]) < settings.min_bbox_width_px:
            reasons.append("projected_bbox_too_narrow")
        if float(bbox[3] - bbox[1]) < settings.min_bbox_height_px:
            reasons.append("projected_bbox_too_short")

    # Audit trust is intentionally excluded from this list: intended
    # capabilities describe what the geometry could support.  Effective
    # capabilities additionally require the exact calibration audit.
    geometry_failures = [reason for reason in reasons if reason != "calibration_untrusted"]
    intended: list[str] = []
    if not geometry_failures:
        visible = float(quality["visible_fraction"])
        if visible >= settings.occupied_visible_fraction:
            intended.append("can_assess_occupied")
        if visible >= settings.free_visible_fraction:
            intended.append("can_assess_free")
    intended_capabilities = tuple(sorted(intended))
    effective = intended_capabilities if camera_model.trusted else ()
    return CameraFrameAssessment(
        frame=frame,
        polygon_uv=_json_uv(uv),
        projection_quality=quality,
        intended_capabilities=intended_capabilities,
        effective_capabilities=effective,
        rejection_reasons=tuple(reasons),
    )


def preselect_pre_anchor_frames(
    frames: Sequence[FrameRecord],
    slot_center_map: Sequence[float],
    anchor_frame: int,
    map_units_per_meter: float,
    *,
    config: CameraSelectionConfig | None = None,
) -> tuple[FrameRecord, ...]:
    """Pose-prefilter close, front-facing frames before (or at) the anchor."""
    settings = config or CameraSelectionConfig()
    center = np.asarray(slot_center_map, dtype=np.float64)
    scale = float(map_units_per_meter)
    if center.shape != (2,) or not np.isfinite(center).all() or not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("slot center and map scale are invalid")
    start = int(anchor_frame) - settings.lookback_frames
    ranked: list[tuple[float, float, int, FrameRecord]] = []
    for frame in frames:
        if frame.frame_id < start or frame.frame_id > int(anchor_frame):
            continue
        if (int(anchor_frame) - frame.frame_id) % settings.frame_stride != 0:
            continue
        pose = (frame.map_x, frame.map_y, frame.map_yaw)
        bearing = relative_bearing_deg(pose, center)
        if abs(bearing) > settings.half_fov_deg:
            continue
        distance_m = float(np.linalg.norm(center - np.asarray(pose[:2])) / scale)
        ranked.append((distance_m, abs(bearing), frame.frame_id, frame))
    ranked.sort(key=lambda row: row[:3])
    chosen = [row[3] for row in ranked[: settings.pose_prefilter_limit]]
    return tuple(sorted(chosen, key=lambda frame: frame.frame_id))


def select_camera_assessments(
    assessments: Sequence[CameraFrameAssessment],
    *,
    max_frames: int = 5,
    trusted_only: bool = False,
) -> tuple[CameraFrameAssessment, ...]:
    """Choose best-capability frames, then return them chronologically."""
    limit = max(0, int(max_frames))
    eligible = [
        assessment
        for assessment in assessments
        if (
            assessment.effective_capabilities
            if trusted_only
            else assessment.intended_capabilities
        )
    ]
    eligible.sort(
        key=lambda assessment: (
            len(assessment.effective_capabilities if trusted_only else assessment.intended_capabilities),
            float(assessment.projection_quality["visible_fraction"]),
            float(assessment.projection_quality["projected_area_px"]),
            -abs(float(assessment.projection_quality["bearing_deg"])),
            assessment.frame.frame_id,
        ),
        reverse=True,
    )
    return tuple(sorted(eligible[:limit], key=lambda assessment: assessment.frame.frame_id))


def assess_pre_anchor_candidates(
    frames: Sequence[FrameRecord],
    polygon_map: np.ndarray,
    slot_center_map: Sequence[float],
    anchor_frame: int,
    anchor_timestamp: float,
    map_units_per_meter: float,
    ground_z_by_frame: Mapping[int, float],
    camera_model: CameraModel,
    *,
    sync_source: str | None,
    config: CameraSelectionConfig | None = None,
) -> CameraCandidateBatch:
    """Preselect, fully assess and rank candidates with deterministic defaults."""
    settings = config or CameraSelectionConfig()
    candidates = preselect_pre_anchor_frames(
        frames,
        slot_center_map,
        anchor_frame,
        map_units_per_meter,
        config=settings,
    )
    assessments = tuple(
        assess_camera_frame(
            frame,
            polygon_map,
            slot_center_map,
            map_units_per_meter,
            ground_z_by_frame[frame.frame_id],
            camera_model,
            anchor_frame=anchor_frame,
            anchor_timestamp=anchor_timestamp,
            sync_source=sync_source,
            config=settings,
        )
        for frame in candidates
        if frame.frame_id in ground_z_by_frame
    )
    return CameraCandidateBatch(
        assessments=assessments,
        selected=select_camera_assessments(assessments, max_frames=settings.max_frames),
    )
