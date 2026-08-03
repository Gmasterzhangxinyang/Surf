"""Exact allowlist of bounded Part2 evidence inspection tools."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any, Mapping

from .contracts import freeze_json
from .media import (
    EvidenceMediaError,
    EvidenceMediaStore,
    EvidencePackError,
    RgbFrameMediaInput,
    is_lidar_evidence_pack,
    load_lidar_evidence_pack,
    render_lidar_triptych,
    render_rgb_frame,
    render_rgb_sequence,
)
from .preflight import EvidenceCatalog, EvidenceRecord, OPAQUE_EVIDENCE_ID
from .queueing import TOOL_NAMES, TOOL_REGISTRY_VERSION


TOOLS_POLICY_VERSION = "part2-tool-execution/2.0"


@dataclass(frozen=True, slots=True)
class ToolResult:
    """A deterministic success or failure returned by every tool attempt."""

    tool_name: str
    status: str
    evidence_id: str | None
    data: Mapping[str, Any]
    error_code: str | None = None
    error_message: str | None = None

    @property
    def ok(self) -> bool:
        return self.status == "ok"

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "status": self.status,
            "evidence_id": self.evidence_id,
            "data": _mutable_json(self.data),
            "error_code": self.error_code,
            "error_message": self.error_message,
        }


def _mutable_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _mutable_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_mutable_json(item) for item in value]
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _declared_image_dimensions(quality: Mapping[str, Any]) -> tuple[Any, Any]:
    """Return an optional producer-bound image size without guessing it."""

    candidates: list[tuple[Any, Any]] = []
    for width_name, height_name in (
        ("image_width", "image_height"),
        ("image_width_px", "image_height_px"),
    ):
        if width_name in quality or height_name in quality:
            candidates.append((quality.get(width_name), quality.get(height_name)))
    image_size = quality.get("image_size")
    if isinstance(image_size, Mapping) and ("width" in image_size or "height" in image_size):
        candidates.append((image_size.get("width"), image_size.get("height")))
    if not candidates:
        return None, None
    if any(candidate != candidates[0] for candidate in candidates[1:]):
        raise EvidenceMediaError(
            "rgb_image_dimensions_mismatch",
            "projection metadata declares conflicting image dimensions",
        )
    return candidates[0]


def _result(
    tool_name: str,
    status: str,
    evidence_id: str | None,
    data: Mapping[str, Any] | None = None,
    *,
    error_code: str | None = None,
    error_message: str | None = None,
) -> ToolResult:
    return ToolResult(
        tool_name=tool_name,
        status=status,
        evidence_id=evidence_id,
        data=freeze_json(dict(data or {})),
        error_code=error_code,
        error_message=error_message,
    )


class ToolRegistry:
    """Dispatch only catalog-issued IDs to the three v1 tools."""

    def __init__(
        self,
        catalog: EvidenceCatalog,
        *,
        media_store: EvidenceMediaStore | None = None,
    ) -> None:
        self.catalog = catalog
        self.media_store = media_store

    @property
    def names(self) -> tuple[str, ...]:
        return TOOL_NAMES

    def execute(self, tool_name: str, arguments: Mapping[str, Any]) -> ToolResult:
        if tool_name not in TOOL_NAMES:
            return _result(
                str(tool_name),
                "invalid_tool",
                None,
                error_code="tool_not_allowlisted",
                error_message="tool name is not registered in Part2 v1",
            )
        if not isinstance(arguments, Mapping) or set(arguments) != {"evidence_id"}:
            return _result(
                tool_name,
                "invalid_arguments",
                None,
                error_code="opaque_evidence_id_required",
                error_message="arguments must contain exactly one opaque evidence_id",
            )
        evidence_id = arguments.get("evidence_id")
        if not isinstance(evidence_id, str) or OPAQUE_EVIDENCE_ID.fullmatch(evidence_id) is None:
            return _result(
                tool_name,
                "invalid_arguments",
                evidence_id if isinstance(evidence_id, str) else None,
                error_code="invalid_evidence_id",
                error_message="evidence_id is not a catalog-issued opaque ID",
            )
        try:
            record = self.catalog.get(evidence_id)
        except KeyError:
            return _result(
                tool_name,
                "unavailable",
                evidence_id,
                {"evidence_id": evidence_id},
                error_code="unknown_evidence_id",
                error_message="evidence_id is not present in this run's preflight catalog",
            )
        if record.tool_name != tool_name:
            return _result(
                tool_name,
                "invalid_arguments",
                evidence_id,
                {"evidence_id": evidence_id},
                error_code="evidence_tool_mismatch",
                error_message="evidence_id was issued for a different tool",
            )
        if not record.available:
            return _result(
                tool_name,
                "unavailable",
                evidence_id,
                {"evidence_id": evidence_id},
                error_code=record.unavailable_reason or "artifact_unavailable",
                error_message="preflight could not resolve every declared artifact",
            )
        return self._inspect(record)

    def invoke(self, tool_name: str, arguments: Mapping[str, Any]) -> ToolResult:
        """Alias used by provider-neutral orchestrators."""

        return self.execute(tool_name, arguments)

    def _rgb_media_inputs(
        self,
        record: EvidenceRecord,
        paths: tuple[Path | None, ...],
    ) -> tuple[RgbFrameMediaInput, ...]:
        matching_items = tuple(
            item
            for item in self.catalog.queue.items
            if item.task_id == record.task_id
            and item.slot_id == record.slot_id
            and item.encounter.encounter_id == record.encounter_id
        )
        if len(matching_items) != 1:
            raise EvidenceMediaError(
                "rgb_identity_mismatch",
                "RGB evidence does not resolve to exactly one queue task",
            )
        item = matching_items[0]
        frame_rows: dict[str, Any] = {}
        for frame in item.encounter.visual_frames:
            if frame.visual_frame_id in frame_rows:
                raise EvidenceMediaError(
                    "rgb_identity_mismatch",
                    "RGB visual frame identity is duplicated",
                )
            frame_rows[frame.visual_frame_id] = frame
        if not (
            len(record.resource_ids)
            == len(record.visual_frame_ids)
            == len(paths)
        ):
            raise EvidenceMediaError(
                "rgb_identity_mismatch",
                "RGB evidence resource and visual frame counts differ",
            )
        if record.tool_name == "inspect_rgb_frame" and len(paths) != 1:
            raise EvidenceMediaError(
                "rgb_identity_mismatch",
                "RGB frame evidence must contain exactly one frame",
            )
        if record.tool_name == "inspect_rgb_sequence" and not 1 <= len(paths) <= 5:
            raise EvidenceMediaError(
                "rgb_sequence_invalid",
                "RGB sequence evidence must contain 1..5 frames",
            )

        inputs: list[RgbFrameMediaInput] = []
        for resource_id, visual_frame_id, path in zip(
            record.resource_ids,
            record.visual_frame_ids,
            paths,
        ):
            frame = frame_rows.get(visual_frame_id)
            resource = self.catalog.queue.resources.get(resource_id)
            if (
                frame is None
                or path is None
                or resource is None
                or resource.kind != "rgb_frame"
                or resource.sha256 is None
                or frame.image_resource_id != resource_id
            ):
                raise EvidenceMediaError(
                    "rgb_identity_mismatch",
                    "RGB evidence resource identity does not match its visual frame",
                )
            quality = frame.projection_quality
            declared_encounter = quality.get("encounter_id")
            if declared_encounter is not None and declared_encounter != record.encounter_id:
                raise EvidenceMediaError(
                    "rgb_identity_mismatch",
                    "RGB projection encounter identity does not match its task",
                )
            expected_width, expected_height = _declared_image_dimensions(quality)
            inputs.append(
                RgbFrameMediaInput(
                    source_path=path,
                    source_sha256=resource.sha256,
                    resource_id=resource_id,
                    task_id=record.task_id,
                    slot_id=record.slot_id,
                    encounter_id=record.encounter_id,
                    visual_frame_id=visual_frame_id,
                    lidar_frame=frame.lidar_frame,
                    camera_frame=frame.camera_frame,
                    camera_timestamp=frame.camera_timestamp,
                    polygon_uv=quality.get("polygon_uv"),
                    adjacent_polygons_uv=quality.get("adjacent_polygons_uv", {}),
                    expected_width=expected_width,
                    expected_height=expected_height,
                )
            )
        return tuple(inputs)

    def _inspect(self, record: EvidenceRecord) -> ToolResult:
        artifacts: list[dict[str, Any]] = []
        try:
            paths = self.catalog.artifact_paths(record.evidence_id)
            for resource_id, path in zip(record.resource_ids, paths):
                if path is None or not path.exists():
                    return _result(
                        record.tool_name,
                        "unavailable",
                        record.evidence_id,
                        {"evidence_id": record.evidence_id},
                        error_code="artifact_missing",
                        error_message="a preflight artifact is no longer available",
                    )
                resource = self.catalog.queue.resources.get(resource_id)
                if resource is None:
                    return _result(
                        record.tool_name,
                        "unavailable",
                        record.evidence_id,
                        {"evidence_id": record.evidence_id},
                        error_code="resource_missing",
                        error_message="a catalog resource is no longer available",
                    )
                if not path.is_file():
                    return _result(
                        record.tool_name,
                        "unavailable",
                        record.evidence_id,
                        {"evidence_id": record.evidence_id},
                        error_code="artifact_not_file",
                        error_message="inspectable evidence must resolve to a concrete file",
                    )
                if resource.sha256 is None:
                    return _result(
                        record.tool_name,
                        "unavailable",
                        record.evidence_id,
                        {"evidence_id": record.evidence_id},
                        error_code="artifact_hash_missing",
                        error_message="inspectable evidence must declare a file sha256",
                    )
                actual_sha256 = _file_sha256(path)
                if actual_sha256 != resource.sha256:
                    return _result(
                        record.tool_name,
                        "unavailable",
                        record.evidence_id,
                        {"evidence_id": record.evidence_id},
                        error_code="artifact_hash_mismatch",
                        error_message="artifact content no longer matches its declared sha256",
                    )
                artifacts.append(
                    {
                        "resource_id": resource_id,
                        "kind": resource.kind,
                        "declared_sha256": resource.sha256,
                        "verified_sha256": actual_sha256,
                        "manifest_hash": resource.manifest_hash,
                        "size_bytes": path.stat().st_size,
                    }
                )
        except OSError as exc:
            return _result(
                record.tool_name,
                "failed",
                record.evidence_id,
                {"evidence_id": record.evidence_id},
                error_code="artifact_inspection_failed",
                error_message=type(exc).__name__,
            )

        # Media is deliberately published only after the opaque catalog lookup
        # and immediate resource hash verification above have both succeeded.
        # Its path and bytes remain private to the provider-side media store.
        if record.tool_name == "inspect_lidar_map":
            try:
                if len(paths) != 1 or paths[0] is None:
                    raise EvidencePackError("LiDAR evidence must resolve to exactly one pack")
                resource = self.catalog.queue.resources.get(record.resource_ids[0])
                if resource is None or resource.sha256 is None:
                    raise EvidencePackError("LiDAR evidence pack has no declared identity")
                if resource.kind != "pointcloud_artifact" or not is_lidar_evidence_pack(
                    paths[0]
                ):
                    return _result(
                        record.tool_name,
                        "unavailable",
                        record.evidence_id,
                        {"evidence_id": record.evidence_id},
                        error_code="unsupported_lidar_evidence_format",
                        error_message=(
                            "inspect_lidar_map requires a renderable Part2 LiDAR evidence pack"
                        ),
                    )
                if self.media_store is None:
                    pack = load_lidar_evidence_pack(
                        paths[0],
                        expected_sha256=resource.sha256,
                        expected_slot_id=record.slot_id,
                        expected_task_id=record.task_id,
                        expected_encounter_id=record.encounter_id,
                        expected_dataset_id=resource.dataset_id,
                        expected_config_hash=resource.config_hash,
                        expected_slot_map_hash=resource.slot_map_hash,
                    )
                    render_lidar_triptych(pack)
                else:
                    self.media_store.publish_lidar(
                        record.evidence_id,
                        paths[0],
                        expected_sha256=resource.sha256,
                        expected_slot_id=record.slot_id,
                        expected_task_id=record.task_id,
                        expected_encounter_id=record.encounter_id,
                        expected_dataset_id=resource.dataset_id,
                        expected_config_hash=resource.config_hash,
                        expected_slot_map_hash=resource.slot_map_hash,
                    )
            except (EvidenceMediaError, OSError):
                return _result(
                    record.tool_name,
                    "failed",
                    record.evidence_id,
                    {"evidence_id": record.evidence_id},
                    error_code="evidence_media_unavailable",
                    error_message="verified LiDAR evidence could not be rendered safely",
                )
        elif record.tool_name in {"inspect_rgb_frame", "inspect_rgb_sequence"}:
            try:
                frames = self._rgb_media_inputs(record, paths)
                if record.tool_name == "inspect_rgb_frame":
                    if self.media_store is None:
                        render_rgb_frame(frames[0])
                    else:
                        self.media_store.publish_rgb_frame(record.evidence_id, frames[0])
                else:
                    if self.media_store is None:
                        render_rgb_sequence(frames)
                    else:
                        self.media_store.publish_rgb_sequence(record.evidence_id, frames)
            except EvidenceMediaError as exc:
                return _result(
                    record.tool_name,
                    "failed",
                    record.evidence_id,
                    {"evidence_id": record.evidence_id},
                    error_code=exc.code,
                    error_message="verified RGB evidence could not be rendered safely",
                )
            except OSError:
                return _result(
                    record.tool_name,
                    "failed",
                    record.evidence_id,
                    {"evidence_id": record.evidence_id},
                    error_code="rgb_media_unavailable",
                    error_message="verified RGB evidence could not be rendered safely",
                )

        return _result(
            record.tool_name,
            "ok",
            record.evidence_id,
            {
                "evidence_id": record.evidence_id,
                "task_id": record.task_id,
                "slot_id": record.slot_id,
                "encounter_id": record.encounter_id,
                "resource_ids": record.resource_ids,
                "visual_frame_ids": record.visual_frame_ids,
                "capabilities": record.capabilities,
                "artifacts": artifacts,
                "semantic_inference_performed": False,
            },
        )


__all__ = [
    "TOOLS_POLICY_VERSION",
    "TOOL_NAMES",
    "TOOL_REGISTRY_VERSION",
    "ToolRegistry",
    "ToolResult",
]
