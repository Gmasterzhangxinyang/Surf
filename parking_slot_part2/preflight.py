"""Opaque evidence catalog and non-semantic camera capability preflight."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import re
from types import MappingProxyType
from typing import Any, Mapping, Sequence
from urllib.parse import unquote, urlparse

from .contracts import QueueEnvelope, QueueItem, ResourceRef, VisualFrameRef, freeze_json
from .queueing import canonical_sha256


PREFLIGHT_POLICY_VERSION = "part2-evidence-preflight/1.1"
CAMERA_MAX_SEQUENCE_FRAMES = 5
CAMERA_MAX_ABS_BEARING_DEG = 40.0
CAMERA_MIN_DISTANCE_M = 3.0
CAMERA_MAX_DISTANCE_M = 20.0
CAMERA_MIN_FINITE_VERTICES = 3
CAMERA_MIN_PROJECTED_AREA_PX = 800.0
CAMERA_MIN_BBOX_WIDTH_PX = 40.0
CAMERA_MIN_BBOX_HEIGHT_PX = 20.0
CAMERA_MIN_VISIBLE_OCCUPIED = 0.60
CAMERA_MIN_VISIBLE_FREE = 0.60
OPAQUE_EVIDENCE_ID = re.compile(r"^ev_[0-9a-f]{64}$")


@dataclass(frozen=True, slots=True)
class EvidenceRecord:
    """Public, path-free metadata for one bounded tool input."""

    evidence_id: str
    tool_name: str
    task_id: str
    slot_id: str
    encounter_id: str
    resource_ids: tuple[str, ...]
    visual_frame_ids: tuple[str, ...]
    capabilities: tuple[str, ...]
    available: bool
    unavailable_reason: str | None = None


@dataclass(frozen=True, slots=True)
class CameraPreflight:
    """Deterministic camera usability summary for one queue task."""

    task_id: str
    slot_id: str
    encounter_id: str
    status: str
    capabilities: tuple[str, ...]
    frame_evidence_ids: tuple[str, ...]
    sequence_evidence_id: str | None
    rejections: Mapping[str, tuple[str, ...]]


def _opaque_id(payload: Mapping[str, Any]) -> str:
    return "ev_" + canonical_sha256(payload).split(":", 1)[1]


def _local_path(uri: str, base_dir: Path) -> Path | None:
    parsed = urlparse(uri)
    if parsed.scheme not in ("", "file"):
        return None
    if parsed.scheme == "file" and parsed.netloc not in ("", "localhost"):
        return None
    raw_path = unquote(parsed.path) if parsed.scheme == "file" else uri
    path = Path(raw_path)
    return path if path.is_absolute() else base_dir / path


def _artifact_unavailable_reason(
    path: Path | None,
    *,
    require_file: bool = False,
) -> str | None:
    if path is None:
        return "artifact_not_local"
    try:
        available = path.is_file() if require_file else path.exists()
    except OSError:
        return "artifact_access_failed"
    return None if available else "artifact_missing"


def _number(mapping: Mapping[str, Any], *names: str) -> float | None:
    for name in names:
        value = mapping.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        result = float(value)
        if math.isfinite(result):
            return result
    return None


def _integer(mapping: Mapping[str, Any], *names: str) -> int | None:
    value = _number(mapping, *names)
    if value is None or not value.is_integer():
        return None
    return int(value)


def _sync_source(quality: Mapping[str, Any]) -> str | None:
    candidates: list[Any] = [
        quality.get("sync_source"),
        quality.get("synchronization_source"),
        quality.get("sync_mode"),
    ]
    nested = quality.get("sync_metadata") or quality.get("sync")
    if isinstance(nested, Mapping):
        candidates.extend((nested.get("source"), nested.get("mode")))
    if quality.get("corrected_sync") is True:
        candidates.append("corrected")
    if quality.get("native_sync") is True:
        candidates.append("native")
    for value in candidates:
        if not isinstance(value, str):
            continue
        normalized = value.strip().lower().replace("-", "_")
        if normalized == "corrected" or normalized.startswith("corrected_"):
            return "corrected"
        if normalized == "native" or normalized.startswith("native_"):
            return "native"
    return None


def _bbox_size(quality: Mapping[str, Any]) -> tuple[float, float] | None:
    width = _number(quality, "bbox_width_px", "projected_bbox_width_px")
    height = _number(quality, "bbox_height_px", "projected_bbox_height_px")
    if width is not None and height is not None:
        return width, height
    bbox = quality.get("bbox_px", quality.get("projected_bbox", quality.get("bbox")))
    if isinstance(bbox, Mapping):
        width = _number(bbox, "width", "width_px")
        height = _number(bbox, "height", "height_px")
        if width is not None and height is not None:
            return width, height
        x0 = _number(bbox, "x0", "left", "min_x")
        y0 = _number(bbox, "y0", "top", "min_y")
        x1 = _number(bbox, "x1", "right", "max_x")
        y1 = _number(bbox, "y1", "bottom", "max_y")
        if None not in (x0, y0, x1, y1):
            return float(x1) - float(x0), float(y1) - float(y0)
    if isinstance(bbox, Sequence) and not isinstance(bbox, (str, bytes)) and len(bbox) == 4:
        values: list[float] = []
        for value in bbox:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                return None
            numeric = float(value)
            if not math.isfinite(numeric):
                return None
            values.append(numeric)
        return values[2] - values[0], values[3] - values[1]
    return None


def _base_camera_rejections(
    item: QueueItem,
    frame: VisualFrameRef,
    resource: ResourceRef | None,
    artifact_path: Path | None,
    anchor_camera_timestamp: float | None,
    calibration_sha256: str | None,
) -> list[str]:
    quality = frame.projection_quality
    reasons: list[str] = []
    # Legacy static v1 fixtures did not carry an audit field.  New real-data
    # producers bind the parsed calibration identity into every projection;
    # once that identity is present, an explicit passing audit is mandatory.
    # This keeps old deterministic replay fixtures compatible while ensuring
    # that real RGB evidence is fail-closed by default.
    calibration_bound = any(
        name in quality
        for name in ("calibration_sha256", "calibration_identity", "calibration_audit_status")
    )
    if calibration_bound and quality.get("calibration_audit_status") != "passed":
        reasons.append("calibration_untrusted")
    declared_calibration = quality.get("calibration_sha256")
    if calibration_bound and (
        not isinstance(declared_calibration, str)
        or declared_calibration != calibration_sha256
    ):
        reasons.append("calibration_identity_mismatch")
    declared_encounter = quality.get("encounter_id")
    if declared_encounter is not None and declared_encounter != item.encounter.encounter_id:
        reasons.append("different_encounter")
    if frame.lidar_frame > item.encounter.anchor_frame:
        reasons.append("after_anchor_frame")
    if anchor_camera_timestamp is None:
        reasons.append("missing_anchor_timestamp")
    elif (
        frame.lidar_timestamp > anchor_camera_timestamp
        or frame.camera_timestamp > anchor_camera_timestamp
    ):
        reasons.append("after_anchor_timestamp")
    if frame.lidar_frame == frame.camera_frame:
        reasons.append("sensor_frame_ids_not_distinct")
    if _sync_source(quality) is None:
        reasons.append("unsupported_or_missing_sync_metadata")

    bearing = _number(quality, "bearing_deg", "relative_bearing_deg")
    if bearing is None or abs(bearing) > CAMERA_MAX_ABS_BEARING_DEG:
        reasons.append("outside_safe_fov")
    distance = _number(quality, "distance_m", "camera_distance_m")
    if distance is None or not CAMERA_MIN_DISTANCE_M <= distance <= CAMERA_MAX_DISTANCE_M:
        reasons.append("outside_distance_range")
    vertices = _integer(
        quality,
        "finite_vertices",
        "in_image_vertices",
        "projected_vertices",
    )
    if vertices is None or vertices < CAMERA_MIN_FINITE_VERTICES:
        reasons.append("insufficient_projected_vertices")
    area = _number(quality, "projected_area_px", "projected_area", "area_px", "area")
    if area is None or area < CAMERA_MIN_PROJECTED_AREA_PX:
        reasons.append("projected_area_too_small")
    bbox = _bbox_size(quality)
    if bbox is None:
        reasons.append("missing_projected_bbox")
    else:
        if bbox[0] < CAMERA_MIN_BBOX_WIDTH_PX:
            reasons.append("projected_bbox_too_narrow")
        if bbox[1] < CAMERA_MIN_BBOX_HEIGHT_PX:
            reasons.append("projected_bbox_too_short")
    if resource is None or resource.kind != "rgb_frame":
        reasons.append("invalid_image_resource")
    else:
        artifact_reason = _artifact_unavailable_reason(artifact_path, require_file=True)
        if artifact_reason == "artifact_not_local":
            reasons.append("non_local_image_artifact")
        elif artifact_reason == "artifact_missing":
            reasons.append("missing_image_artifact")
        elif artifact_reason is not None:
            reasons.append(artifact_reason)
    return reasons


class EvidenceCatalog:
    """Resolve validated resources once and expose only opaque evidence IDs.

    Construction performs geometry/artifact capability checks only. It never
    interprets pixels or point clouds and therefore makes no semantic claim.
    """

    def __init__(self, queue: QueueEnvelope, *, base_dir: str | Path | None = None) -> None:
        self.queue = queue
        self.base_dir = Path.cwd() if base_dir is None else Path(base_dir)
        self._records: dict[str, EvidenceRecord] = {}
        self._paths: dict[str, tuple[Path | None, ...]] = {}
        self._preflights: dict[str, CameraPreflight] = {}
        self._build()

    @property
    def evidence_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._records))

    @property
    def entries(self) -> Mapping[str, EvidenceRecord]:
        return MappingProxyType(dict(self._records))

    @property
    def camera_preflights(self) -> Mapping[str, CameraPreflight]:
        return MappingProxyType(dict(self._preflights))

    def get(self, evidence_id: str) -> EvidenceRecord:
        return self._records[evidence_id]

    def camera_preflight(self, task_id: str) -> CameraPreflight:
        return self._preflights[task_id]

    def artifact_paths(self, evidence_id: str) -> tuple[Path | None, ...]:
        """Internal tool bridge; callers still cannot choose a path."""

        return self._paths[evidence_id]

    def _add_record(
        self,
        *,
        item: QueueItem,
        tool_name: str,
        resource_ids: tuple[str, ...],
        visual_frame_ids: tuple[str, ...] = (),
        capabilities: tuple[str, ...] = (),
        unavailable_reason: str | None = None,
    ) -> str:
        evidence_id = _opaque_id(
            {
                "queue_id": self.queue.queue_id,
                "task_id": item.task_id,
                "encounter_id": item.encounter.encounter_id,
                "tool_name": tool_name,
                "resource_ids": resource_ids,
                "visual_frame_ids": visual_frame_ids,
            }
        )
        paths = tuple(
            _local_path(self.queue.resources[resource_id].uri, self.base_dir)
            if resource_id in self.queue.resources
            else None
            for resource_id in resource_ids
        )
        reason = unavailable_reason
        if reason is None:
            for path in paths:
                reason = _artifact_unavailable_reason(path)
                if reason is not None:
                    break
        self._records[evidence_id] = EvidenceRecord(
            evidence_id=evidence_id,
            tool_name=tool_name,
            task_id=item.task_id,
            slot_id=item.slot_id,
            encounter_id=item.encounter.encounter_id,
            resource_ids=resource_ids,
            visual_frame_ids=visual_frame_ids,
            capabilities=tuple(sorted(set(capabilities))),
            available=reason is None,
            unavailable_reason=reason,
        )
        self._paths[evidence_id] = paths
        return evidence_id

    def _build(self) -> None:
        for item in sorted(self.queue.items, key=lambda row: row.task_id):
            for resource_id in sorted(item.encounter.pointcloud_resource_ids):
                unavailable = None
                resource = self.queue.resources.get(resource_id)
                if resource is None:
                    unavailable = "resource_missing"
                elif resource.kind not in {"lidar_map", "pointcloud_artifact"}:
                    unavailable = "invalid_lidar_resource"
                self._add_record(
                    item=item,
                    tool_name="inspect_lidar_map",
                    resource_ids=(resource_id,),
                    capabilities=("can_assess_occupied",),
                    unavailable_reason=unavailable,
                )
            self._preflights[item.task_id] = self._build_camera(item)

    def _build_camera(self, item: QueueItem) -> CameraPreflight:
        frames = tuple(item.encounter.visual_frames)
        anchor_timestamp = item.encounter.anchor_timestamp
        calibration_resources = tuple(
            resource
            for resource in self.queue.resources.values()
            if resource.kind == "camera_calibration"
        )
        calibration_sha256 = (
            calibration_resources[0].sha256 if len(calibration_resources) == 1 else None
        )
        accepted: list[tuple[VisualFrameRef, tuple[str, ...]]] = []
        rejections: dict[str, tuple[str, ...]] = {}

        for frame in sorted(
            frames,
            key=lambda row: (row.camera_timestamp, row.camera_frame, row.visual_frame_id),
        ):
            resource = self.queue.resources.get(frame.image_resource_id)
            path = _local_path(resource.uri, self.base_dir) if resource is not None else None
            reasons = _base_camera_rejections(
                item,
                frame,
                resource,
                path,
                anchor_timestamp,
                calibration_sha256,
            )
            visible = _number(frame.projection_quality, "visible_fraction", "visible_core_ratio")
            capabilities: list[str] = []
            if not reasons and visible is not None:
                if (
                    "can_assess_occupied" in frame.capabilities
                    and visible >= CAMERA_MIN_VISIBLE_OCCUPIED
                ):
                    capabilities.append("can_assess_occupied")
                if "can_assess_free" in frame.capabilities and visible >= CAMERA_MIN_VISIBLE_FREE:
                    capabilities.append("can_assess_free")
            if visible is None:
                reasons.append("missing_visible_fraction")
            elif visible < CAMERA_MIN_VISIBLE_OCCUPIED:
                reasons.append("visible_fraction_too_small")
            if capabilities:
                accepted.append((frame, tuple(sorted(capabilities))))
            else:
                if not reasons:
                    reasons.append("no_eligible_camera_capability")
                rejections[frame.visual_frame_id] = tuple(sorted(set(reasons)))

        ranked = sorted(
            accepted,
            key=lambda pair: (
                len(pair[1]),
                _number(pair[0].projection_quality, "visible_fraction", "visible_core_ratio") or 0.0,
                _number(
                    pair[0].projection_quality,
                    "projected_area_px",
                    "projected_area",
                    "area_px",
                    "area",
                )
                or 0.0,
                pair[0].camera_timestamp,
                pair[0].visual_frame_id,
            ),
            reverse=True,
        )[:CAMERA_MAX_SEQUENCE_FRAMES]
        selected = sorted(
            ranked,
            key=lambda pair: (
                pair[0].camera_timestamp,
                pair[0].camera_frame,
                pair[0].visual_frame_id,
            ),
        )

        frame_ids: list[str] = []
        available_capabilities: set[str] = set()
        for frame, capabilities in selected:
            available_capabilities.update(capabilities)
            frame_ids.append(
                self._add_record(
                    item=item,
                    tool_name="inspect_rgb_frame",
                    resource_ids=(frame.image_resource_id,),
                    visual_frame_ids=(frame.visual_frame_id,),
                    capabilities=capabilities,
                )
            )

        sequence_id: str | None = None
        if selected:
            sequence_id = self._add_record(
                item=item,
                tool_name="inspect_rgb_sequence",
                resource_ids=tuple(frame.image_resource_id for frame, _ in selected),
                visual_frame_ids=tuple(frame.visual_frame_id for frame, _ in selected),
                capabilities=tuple(sorted(available_capabilities)),
            )

        requested_capabilities = {
            capability
            for frame in frames
            for capability in frame.capabilities
            if capability in {"can_assess_free", "can_assess_occupied"}
        }
        if not selected:
            status = "unavailable"
        elif requested_capabilities <= available_capabilities:
            status = "ready"
        else:
            status = "limited"
        return CameraPreflight(
            task_id=item.task_id,
            slot_id=item.slot_id,
            encounter_id=item.encounter.encounter_id,
            status=status,
            capabilities=tuple(sorted(available_capabilities)),
            frame_evidence_ids=tuple(frame_ids),
            sequence_evidence_id=sequence_id,
            rejections=freeze_json(rejections),
        )


__all__ = [
    "CAMERA_MAX_SEQUENCE_FRAMES",
    "CAMERA_MAX_ABS_BEARING_DEG",
    "CAMERA_MIN_DISTANCE_M",
    "CAMERA_MAX_DISTANCE_M",
    "CAMERA_MIN_FINITE_VERTICES",
    "CAMERA_MIN_PROJECTED_AREA_PX",
    "CAMERA_MIN_BBOX_WIDTH_PX",
    "CAMERA_MIN_BBOX_HEIGHT_PX",
    "CAMERA_MIN_VISIBLE_OCCUPIED",
    "CAMERA_MIN_VISIBLE_FREE",
    "CameraPreflight",
    "EvidenceCatalog",
    "EvidenceRecord",
    "OPAQUE_EVIDENCE_ID",
    "PREFLIGHT_POLICY_VERSION",
]
