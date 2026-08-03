"""Canonical identity and validation for ``unknown_agent_queue.json``."""

from __future__ import annotations

from dataclasses import fields, is_dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from types import MappingProxyType
from typing import Any, Mapping, Sequence
from urllib.parse import unquote, urlparse

from .contracts import (
    EncounterRef,
    QueueEnvelope,
    QueueItem,
    RelationshipRef,
    ResourceRef,
    VisualFrameRef,
    freeze_json,
)


QUEUE_SCHEMA_VERSION = "unknown-agent-queue/1.0"
TOOL_REGISTRY_VERSION = "part2-tools/1.0"
TOOL_NAMES = (
    "inspect_lidar_map",
    "inspect_rgb_frame",
    "inspect_rgb_sequence",
)
SHA256_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
ALLOWED_SCOPES = frozenset({"in_route_scope", "partial_route_scope"})
ALLOWED_MODALITIES = frozenset({"lidar", "rgb"})
ALLOWED_FINAL_STATES = frozenset({"occupied", "free", "unknown"})
ALLOWED_VISUAL_CAPABILITIES = frozenset({"can_assess_occupied", "can_assess_free"})
LIDAR_RESOURCE_KINDS = frozenset({"lidar_map", "pointcloud_artifact"})
REQUIRED_RESOURCE_KINDS = frozenset(
    {
        "slot_database",
        "corrected_frames",
        "map_points_root",
        "camera_calibration",
        "base_slot_decisions",
    }
)
ALLOWED_RESOURCE_KINDS = REQUIRED_RESOURCE_KINDS | frozenset(
    {"lidar_map", "pointcloud_artifact", "rgb_frame"}
)
ROOT_FIELDS = frozenset(
    {"schema_version", "queue_id", "producer", "resources", "tool_registry_version", "items"}
)
PRODUCER_FIELDS = frozenset(
    {"pipeline", "run_id", "config_hash", "dataset_id", "slot_map_hash"}
)
RESOURCE_FIELDS = frozenset(
    {
        "kind",
        "uri",
        "sha256",
        "manifest_hash",
        "dataset_id",
        "config_hash",
        "slot_map_hash",
    }
)
ITEM_FIELDS = frozenset(
    {
        "task_id",
        "slot_id",
        "scope_status",
        "state",
        "agent_observable",
        "unknown_reasons",
        "priority",
        "available_modalities",
        "suggested_tools",
        "allowed_final_states",
        "occupied_evidence",
        "free_evidence",
        "audit",
        "relationships",
        "encounter",
    }
)
RELATIONSHIP_FIELDS = frozenset(
    {"adjacent_slot_ids", "conflict_slot_ids", "shared_evidence_ids"}
)
ENCOUNTER_FIELDS = frozenset(
    {
        "encounter_id",
        "part1_trace_event_ids",
        "start_lidar_frame",
        "anchor_frame",
        "end_lidar_frame",
        "start_timestamp",
        "anchor_timestamp",
        "end_timestamp",
        "support_frames",
        "pointcloud_resource_ids",
        "visual_frames",
    }
)
VISUAL_FRAME_FIELDS = frozenset(
    {
        "visual_frame_id",
        "lidar_frame",
        "camera_frame",
        "lidar_timestamp",
        "camera_timestamp",
        "camera_lidar_dt_sec",
        "image_resource_id",
        "capabilities",
        "projection_quality",
    }
)


def _json_value(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: _json_value(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Mapping):
        converted: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("canonical JSON object keys must be strings")
            converted[key] = _json_value(item)
        return converted
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("canonical JSON does not allow NaN or infinity")
        return value
    raise TypeError(f"value of type {type(value).__name__} is not JSON-compatible")


def canonical_json_bytes(payload: Any) -> bytes:
    """Serialize a JSON value with stable ordering and no insignificant space."""

    return json.dumps(
        _json_value(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(payload: Any) -> str:
    """Return the canonical content identity in ``sha256:<hex>`` form."""

    return "sha256:" + hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def make_task_id(producer: Mapping[str, Any], slot_id: str, encounter_id: str) -> str:
    """Derive a stable task identity from its Part1 producer and encounter identity."""

    return canonical_sha256(
        {
            "producer": producer,
            "slot_id": slot_id,
            "encounter_id": encounter_id,
        }
    )


def make_queue_id(payload: Mapping[str, Any]) -> str:
    """Derive a queue identity, excluding the self-referential ``queue_id`` field."""

    if not isinstance(payload, Mapping):
        raise TypeError("queue identity payload must be an object")
    return canonical_sha256({key: value for key, value in payload.items() if key != "queue_id"})


def _object(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must be a JSON object")
    return value


def _required(payload: Mapping[str, Any], name: str, path: str) -> Any:
    if name not in payload:
        raise ValueError(f"{path} is missing required field {name}")
    return payload[name]


def _reject_unsupported_fields(
    payload: Mapping[str, Any],
    allowed: frozenset[str],
    path: str,
) -> None:
    unsupported = sorted(str(name) for name in payload if name not in allowed)
    if unsupported:
        raise ValueError(f"{path} has unsupported field(s): {', '.join(unsupported)}")


def _text(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{path} must be a non-empty string")
    return value


def _integer(value: Any, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{path} must be a non-negative integer")
    return value


def _number(value: Any, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{path} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{path} must be a finite number")
    return result


def _array(value: Any, path: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)):
        raise ValueError(f"{path} must be a JSON array")
    return value


def _text_tuple(value: Any, path: str, *, allow_empty: bool = True) -> tuple[str, ...]:
    result = tuple(_text(item, f"{path}[{index}]") for index, item in enumerate(_array(value, path)))
    if not allow_empty and not result:
        raise ValueError(f"{path} must not be empty")
    if len(set(result)) != len(result):
        raise ValueError(f"{path} contains duplicate IDs or values")
    return result


def _hash(value: Any, path: str) -> str:
    result = _text(value, path)
    if SHA256_PATTERN.fullmatch(result) is None:
        raise ValueError(f"{path} must be a canonical sha256:<64 lowercase hex> identity")
    return result


def _parse_producer(value: Any) -> Mapping[str, Any]:
    payload = _object(value, "producer")
    _reject_unsupported_fields(payload, PRODUCER_FIELDS, "producer")
    result = {
        "pipeline": _text(_required(payload, "pipeline", "producer"), "producer.pipeline"),
        "run_id": _text(_required(payload, "run_id", "producer"), "producer.run_id"),
        "config_hash": _hash(_required(payload, "config_hash", "producer"), "producer.config_hash"),
        "dataset_id": _text(_required(payload, "dataset_id", "producer"), "producer.dataset_id"),
        "slot_map_hash": _hash(_required(payload, "slot_map_hash", "producer"), "producer.slot_map_hash"),
    }
    return MappingProxyType(result)


def _parse_resources(value: Any) -> Mapping[str, ResourceRef]:
    payload = _object(value, "resources")
    result: dict[str, ResourceRef] = {}
    for resource_id, raw in payload.items():
        resource_path = f"resources.{resource_id}"
        resource_id = _text(resource_id, "resources resource ID")
        item = _object(raw, resource_path)
        _reject_unsupported_fields(item, RESOURCE_FIELDS, resource_path)
        kind = _text(_required(item, "kind", resource_path), f"{resource_path}.kind")
        if kind not in ALLOWED_RESOURCE_KINDS:
            raise ValueError(f"{resource_path}.kind is unsupported in v1: {kind}")
        if kind == "map_points_root":
            has_file_hash = "sha256" in item
            has_manifest_hash = "manifest_hash" in item
            if has_file_hash == has_manifest_hash:
                raise ValueError(
                    f"{resource_path} must declare exactly one of sha256 or manifest_hash"
                )
            sha256 = (
                _hash(item["sha256"], f"{resource_path}.sha256")
                if has_file_hash
                else None
            )
            manifest_hash = (
                _hash(item["manifest_hash"], f"{resource_path}.manifest_hash")
                if has_manifest_hash
                else None
            )
        else:
            if "manifest_hash" in item:
                raise ValueError(
                    f"{resource_path}.manifest_hash is only valid for map_points_root"
                )
            sha256 = _hash(
                _required(item, "sha256", resource_path),
                f"{resource_path}.sha256",
            )
            manifest_hash = None
        result[resource_id] = ResourceRef(
            resource_id=resource_id,
            kind=kind,
            uri=_text(_required(item, "uri", resource_path), f"{resource_path}.uri"),
            sha256=sha256,
            manifest_hash=manifest_hash,
            dataset_id=_text(
                _required(item, "dataset_id", resource_path),
                f"{resource_path}.dataset_id",
            ),
            config_hash=_hash(
                _required(item, "config_hash", resource_path),
                f"{resource_path}.config_hash",
            ),
            slot_map_hash=_hash(
                _required(item, "slot_map_hash", resource_path),
                f"{resource_path}.slot_map_hash",
            ),
        )
    return MappingProxyType(result)


def _validate_required_resources(resources: Mapping[str, ResourceRef]) -> None:
    ids_by_kind: dict[str, list[str]] = {}
    for resource_id, resource in resources.items():
        ids_by_kind.setdefault(resource.kind, []).append(resource_id)
    for required_kind in sorted(REQUIRED_RESOURCE_KINDS):
        resource_ids = ids_by_kind.get(required_kind, [])
        if not resource_ids:
            raise ValueError(f"missing required resource kind: {required_kind}")
        if len(resource_ids) > 1:
            raise ValueError(
                f"required resource kind {required_kind} must appear exactly once"
            )


def _validate_resource_linkage(
    resources: Mapping[str, ResourceRef],
    producer: Mapping[str, Any],
) -> None:
    for resource_id, resource in resources.items():
        for field_name in ("dataset_id", "config_hash", "slot_map_hash"):
            if getattr(resource, field_name) != producer[field_name]:
                raise ValueError(
                    f"resources.{resource_id}.{field_name} must match producer.{field_name}"
                )


def _local_path(uri: str, base_dir: Path | None) -> Path | None:
    parsed = urlparse(uri)
    if parsed.scheme not in ("", "file"):
        return None
    if parsed.scheme == "file" and parsed.netloc not in ("", "localhost"):
        return None
    raw_path = unquote(parsed.path) if parsed.scheme == "file" else uri
    path = Path(raw_path)
    if not path.is_absolute():
        path = (base_dir or Path.cwd()) / path
    return path


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _validate_local_file_hashes(
    resources: Mapping[str, ResourceRef],
    base_dir: Path | None,
) -> None:
    for resource_id, resource in resources.items():
        if resource.manifest_hash is not None:
            continue
        path = _local_path(resource.uri, base_dir)
        if path is not None and path.is_file() and _file_sha256(path) != resource.sha256:
            raise ValueError(
                f"resources.{resource_id}.sha256 does not match resolved local file"
            )


def _parse_visual_frame(value: Any, path: str) -> VisualFrameRef:
    payload = _object(value, path)
    _reject_unsupported_fields(payload, VISUAL_FRAME_FIELDS, path)
    capabilities = _text_tuple(
        _required(payload, "capabilities", path),
        f"{path}.capabilities",
        allow_empty=False,
    )
    if not set(capabilities) <= ALLOWED_VISUAL_CAPABILITIES:
        raise ValueError(f"{path}.capabilities contains an unsupported capability")
    projection_quality = _object(
        _required(payload, "projection_quality", path),
        f"{path}.projection_quality",
    )
    lidar_timestamp = _number(
        _required(payload, "lidar_timestamp", path),
        f"{path}.lidar_timestamp",
    )
    camera_timestamp = _number(
        _required(payload, "camera_timestamp", path),
        f"{path}.camera_timestamp",
    )
    camera_lidar_dt_sec = _number(
        _required(payload, "camera_lidar_dt_sec", path),
        f"{path}.camera_lidar_dt_sec",
    )
    if not math.isclose(
        camera_lidar_dt_sec,
        camera_timestamp - lidar_timestamp,
        rel_tol=0.0,
        abs_tol=1e-6,
    ):
        raise ValueError(
            f"{path}.camera_lidar_dt_sec must equal camera_timestamp - lidar_timestamp "
            "within 1e-6 seconds"
        )
    if abs(camera_lidar_dt_sec) > 0.04:
        raise ValueError(f"{path}.camera_lidar_dt_sec absolute value must be <= 0.04 seconds")
    return VisualFrameRef(
        visual_frame_id=_text(
            _required(payload, "visual_frame_id", path),
            f"{path}.visual_frame_id",
        ),
        lidar_frame=_integer(_required(payload, "lidar_frame", path), f"{path}.lidar_frame"),
        camera_frame=_integer(_required(payload, "camera_frame", path), f"{path}.camera_frame"),
        lidar_timestamp=lidar_timestamp,
        camera_timestamp=camera_timestamp,
        camera_lidar_dt_sec=camera_lidar_dt_sec,
        image_resource_id=_text(
            _required(payload, "image_resource_id", path),
            f"{path}.image_resource_id",
        ),
        capabilities=capabilities,
        projection_quality=freeze_json(projection_quality),
    )


def _parse_encounter(value: Any, path: str) -> EncounterRef:
    payload = _object(value, path)
    _reject_unsupported_fields(payload, ENCOUNTER_FIELDS, path)
    start_lidar_frame = _integer(
        _required(payload, "start_lidar_frame", path),
        f"{path}.start_lidar_frame",
    )
    anchor_frame = _integer(_required(payload, "anchor_frame", path), f"{path}.anchor_frame")
    end_lidar_frame = _integer(
        _required(payload, "end_lidar_frame", path),
        f"{path}.end_lidar_frame",
    )
    if start_lidar_frame > anchor_frame:
        raise ValueError(f"{path}.start_lidar_frame must be <= {path}.anchor_frame")
    if anchor_frame > end_lidar_frame:
        raise ValueError(f"{path}.anchor_frame must be <= {path}.end_lidar_frame")
    start_timestamp = _number(
        _required(payload, "start_timestamp", path),
        f"{path}.start_timestamp",
    )
    anchor_timestamp = _number(
        _required(payload, "anchor_timestamp", path),
        f"{path}.anchor_timestamp",
    )
    end_timestamp = _number(
        _required(payload, "end_timestamp", path),
        f"{path}.end_timestamp",
    )
    if start_timestamp > anchor_timestamp:
        raise ValueError(f"{path}.start_timestamp must be <= {path}.anchor_timestamp")
    if anchor_timestamp > end_timestamp:
        raise ValueError(f"{path}.anchor_timestamp must be <= {path}.end_timestamp")
    support_frames = tuple(
        _integer(frame, f"{path}.support_frames[{index}]")
        for index, frame in enumerate(
            _array(_required(payload, "support_frames", path), f"{path}.support_frames")
        )
    )
    if len(set(support_frames)) != len(support_frames):
        raise ValueError(f"{path}.support_frames contains duplicate frame IDs")
    if any(frame < start_lidar_frame or frame > end_lidar_frame for frame in support_frames):
        raise ValueError(f"{path}.support_frames must stay within the encounter frame range")
    visual_frames = tuple(
        _parse_visual_frame(frame, f"{path}.visual_frames[{index}]")
        for index, frame in enumerate(
            _array(_required(payload, "visual_frames", path), f"{path}.visual_frames")
        )
    )
    visual_ids = [frame.visual_frame_id for frame in visual_frames]
    if len(set(visual_ids)) != len(visual_ids):
        raise ValueError(f"{path}.visual_frames contains duplicate visual_frame_id")
    for index, visual in enumerate(visual_frames):
        visual_path = f"{path}.visual_frames[{index}]"
        if not start_lidar_frame <= visual.lidar_frame <= end_lidar_frame:
            raise ValueError(
                f"{visual_path}.lidar_frame must stay within the encounter frame range"
            )
        if visual.lidar_frame > anchor_frame:
            raise ValueError(
                f"{visual_path}.lidar_frame must be <= {path}.anchor_frame"
            )
        if not start_timestamp <= visual.lidar_timestamp <= end_timestamp:
            raise ValueError(
                f"{visual_path}.lidar_timestamp must stay within the encounter time range"
            )
        if not start_timestamp <= visual.camera_timestamp <= end_timestamp:
            raise ValueError(
                f"{visual_path}.camera_timestamp must stay within the encounter time range"
            )
        if visual.lidar_timestamp > anchor_timestamp:
            raise ValueError(
                f"{visual_path}.lidar_timestamp must be <= {path}.anchor_timestamp"
            )
        if visual.camera_timestamp > anchor_timestamp:
            raise ValueError(
                f"{visual_path}.camera_timestamp must be <= {path}.anchor_timestamp"
            )
    return EncounterRef(
        encounter_id=_text(_required(payload, "encounter_id", path), f"{path}.encounter_id"),
        part1_trace_event_ids=_text_tuple(
            _required(payload, "part1_trace_event_ids", path),
            f"{path}.part1_trace_event_ids",
        ),
        start_lidar_frame=start_lidar_frame,
        anchor_frame=anchor_frame,
        end_lidar_frame=end_lidar_frame,
        start_timestamp=start_timestamp,
        anchor_timestamp=anchor_timestamp,
        end_timestamp=end_timestamp,
        support_frames=support_frames,
        pointcloud_resource_ids=_text_tuple(
            _required(payload, "pointcloud_resource_ids", path),
            f"{path}.pointcloud_resource_ids",
        ),
        visual_frames=visual_frames,
    )


def _parse_relationships(value: Any, path: str) -> RelationshipRef:
    payload = _object(value, path)
    _reject_unsupported_fields(payload, RELATIONSHIP_FIELDS, path)
    return RelationshipRef(
        adjacent_slot_ids=_text_tuple(
            _required(payload, "adjacent_slot_ids", path),
            f"{path}.adjacent_slot_ids",
        ),
        conflict_slot_ids=_text_tuple(
            _required(payload, "conflict_slot_ids", path),
            f"{path}.conflict_slot_ids",
        ),
        shared_evidence_ids=_text_tuple(
            _required(payload, "shared_evidence_ids", path),
            f"{path}.shared_evidence_ids",
        ),
    )


def _parse_item(value: Any, index: int) -> QueueItem:
    path = f"items[{index}]"
    payload = _object(value, path)
    _reject_unsupported_fields(payload, ITEM_FIELDS, path)
    state = _text(_required(payload, "state", path), f"{path}.state")
    if state != "unknown":
        raise ValueError(f"{path} must have state=unknown")
    scope_status = _text(
        _required(payload, "scope_status", path),
        f"{path}.scope_status",
    )
    if scope_status not in ALLOWED_SCOPES:
        raise ValueError(
            f"{path}.scope_status must be in_route_scope or partial_route_scope"
        )
    agent_observable = _required(payload, "agent_observable", path)
    if agent_observable is not True:
        raise ValueError(f"{path} must have agent_observable=true")
    modalities = _text_tuple(
        _required(payload, "available_modalities", path),
        f"{path}.available_modalities",
        allow_empty=False,
    )
    if not set(modalities) <= ALLOWED_MODALITIES:
        raise ValueError(f"{path}.available_modalities contains an unsupported modality")
    final_states = _text_tuple(
        _required(payload, "allowed_final_states", path),
        f"{path}.allowed_final_states",
        allow_empty=False,
    )
    if not set(final_states) <= ALLOWED_FINAL_STATES:
        raise ValueError(f"{path}.allowed_final_states contains an unsupported state")
    occupied_evidence = _object(
        _required(payload, "occupied_evidence", path),
        f"{path}.occupied_evidence",
    )
    free_evidence = _object(
        _required(payload, "free_evidence", path),
        f"{path}.free_evidence",
    )
    audit = _object(payload.get("audit", {}), f"{path}.audit")
    suggested_tools = _text_tuple(
        _required(payload, "suggested_tools", path),
        f"{path}.suggested_tools",
    )
    unsupported_tools = sorted(set(suggested_tools) - set(TOOL_NAMES))
    if unsupported_tools:
        raise ValueError(
            f"{path}.suggested_tools contains unsupported v1 tool(s): "
            f"{', '.join(unsupported_tools)}"
        )
    return QueueItem(
        task_id=_hash(_required(payload, "task_id", path), f"{path}.task_id"),
        slot_id=_text(_required(payload, "slot_id", path), f"{path}.slot_id"),
        scope_status=scope_status,
        state=state,
        agent_observable=True,
        unknown_reasons=_text_tuple(
            _required(payload, "unknown_reasons", path),
            f"{path}.unknown_reasons",
            allow_empty=False,
        ),
        priority=_text(_required(payload, "priority", path), f"{path}.priority"),
        available_modalities=modalities,
        suggested_tools=suggested_tools,
        allowed_final_states=final_states,
        occupied_evidence=freeze_json(occupied_evidence),
        free_evidence=freeze_json(free_evidence),
        audit=freeze_json(audit),
        relationships=_parse_relationships(
            _required(payload, "relationships", path),
            f"{path}.relationships",
        ),
        encounter=_parse_encounter(
            _required(payload, "encounter", path),
            f"{path}.encounter",
        ),
    )


def _validate_unique_items(items: tuple[QueueItem, ...]) -> None:
    task_ids: set[str] = set()
    slot_ids: set[str] = set()
    for item in items:
        if item.task_id in task_ids:
            raise ValueError(f"duplicate task_id: {item.task_id}")
        task_ids.add(item.task_id)
        if item.slot_id in slot_ids:
            raise ValueError(f"duplicate slot_id: {item.slot_id}")
        slot_ids.add(item.slot_id)


def _validate_common_encounter_bounds(items: tuple[QueueItem, ...]) -> None:
    bounds_by_encounter: dict[str, tuple[int, int, float, float]] = {}
    for item in items:
        encounter = item.encounter
        bounds = (
            encounter.start_lidar_frame,
            encounter.end_lidar_frame,
            encounter.start_timestamp,
            encounter.end_timestamp,
        )
        existing = bounds_by_encounter.setdefault(encounter.encounter_id, bounds)
        if existing != bounds:
            raise ValueError(
                f"encounter_id {encounter.encounter_id} must use common bounds across items"
            )


def _validate_task_hashes(items: tuple[QueueItem, ...], producer: Mapping[str, Any]) -> None:
    for item in items:
        expected = make_task_id(producer, item.slot_id, item.encounter.encounter_id)
        if item.task_id != expected:
            raise ValueError(
                f"task_id hash does not match producer/slot/encounter identity for {item.slot_id}"
            )


def _validate_item_resources(
    items: tuple[QueueItem, ...],
    resources: Mapping[str, ResourceRef],
) -> None:
    for index, item in enumerate(items):
        path = f"items[{index}].encounter"
        for resource_id in item.encounter.pointcloud_resource_ids:
            resource = resources.get(resource_id)
            if resource is None:
                raise ValueError(
                    f"{path}.pointcloud_resource_ids references missing resource {resource_id}"
                )
            if resource.kind not in LIDAR_RESOURCE_KINDS or resource.sha256 is None:
                raise ValueError(
                    f"{path}.pointcloud_resource_ids resource {resource_id} must reference "
                    "a concrete lidar_map or pointcloud_artifact resource with sha256"
                )
        for visual in item.encounter.visual_frames:
            resource = resources.get(visual.image_resource_id)
            if resource is None:
                raise ValueError(
                    f"{path}.visual_frames image_resource_id references missing resource "
                    f"{visual.image_resource_id}"
                )
            if resource.kind != "rgb_frame":
                raise ValueError(
                    f"{path}.visual_frames image_resource_id {visual.image_resource_id} "
                    "must reference an rgb_frame resource"
                )
        modalities = set(item.available_modalities)
        if "lidar" in modalities and not item.encounter.pointcloud_resource_ids:
            raise ValueError(f"items[{index}] declares lidar without a pointcloud resource")
        if "rgb" in modalities and not item.encounter.visual_frames:
            raise ValueError(f"items[{index}] declares rgb without a visual frame")
        if item.encounter.pointcloud_resource_ids and "lidar" not in modalities:
            raise ValueError(f"items[{index}] has pointcloud resources without lidar modality")
        if item.encounter.visual_frames and "rgb" not in modalities:
            raise ValueError(f"items[{index}] has visual frames without rgb modality")


def validate_queue(
    payload: Any,
    *,
    base_dir: str | Path | None = None,
) -> QueueEnvelope:
    """Validate a queue, including hashes for referenced local file artifacts."""

    root = _object(payload, "queue envelope")
    _reject_unsupported_fields(root, ROOT_FIELDS, "queue envelope")
    schema_version = _text(
        _required(root, "schema_version", "queue envelope"),
        "schema_version",
    )
    if schema_version != QUEUE_SCHEMA_VERSION:
        raise ValueError(
            f"schema_version must be {QUEUE_SCHEMA_VERSION!r}, got {schema_version!r}"
        )
    queue_id = _hash(_required(root, "queue_id", "queue envelope"), "queue_id")
    expected_queue_id = make_queue_id(root)
    if queue_id != expected_queue_id:
        raise ValueError("queue_id hash does not match canonical queue payload")
    producer = _parse_producer(_required(root, "producer", "queue envelope"))
    resources = _parse_resources(_required(root, "resources", "queue envelope"))
    _validate_required_resources(resources)
    _validate_resource_linkage(resources, producer)
    _validate_local_file_hashes(resources, None if base_dir is None else Path(base_dir))
    raw_items = _array(_required(root, "items", "queue envelope"), "items")
    items = tuple(_parse_item(item, index) for index, item in enumerate(raw_items))
    _validate_unique_items(items)
    _validate_common_encounter_bounds(items)
    _validate_task_hashes(items, producer)
    _validate_item_resources(items, resources)
    tool_registry_version = _text(
        _required(root, "tool_registry_version", "queue envelope"),
        "tool_registry_version",
    )
    if tool_registry_version != TOOL_REGISTRY_VERSION:
        raise ValueError(
            f"tool_registry_version must be {TOOL_REGISTRY_VERSION!r}, "
            f"got {tool_registry_version!r}"
        )
    return QueueEnvelope(
        schema_version=schema_version,
        queue_id=queue_id,
        producer=producer,
        resources=resources,
        tool_registry_version=tool_registry_version,
        items=items,
    )


def _reject_constant(value: str) -> None:
    raise ValueError(f"invalid JSON numeric constant: {value}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def load_queue(path: str | Path) -> QueueEnvelope:
    """Load a JSON queue file and apply :func:`validate_queue`."""

    queue_path = Path(path)
    with queue_path.open("r", encoding="utf-8") as handle:
        payload = json.load(
            handle,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    return validate_queue(payload, base_dir=queue_path.parent)
