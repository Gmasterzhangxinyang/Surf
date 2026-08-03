"""Immutable wire-contract records for the Part2 unknown-agent queue."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping


ImmutableMapping = Mapping[str, Any]


def freeze_json(value: Any) -> Any:
    """Return an immutable representation of a JSON-compatible value."""

    if isinstance(value, Mapping):
        return MappingProxyType({str(key): freeze_json(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(freeze_json(item) for item in value)
    return value


@dataclass(frozen=True, slots=True)
class ResourceRef:
    resource_id: str
    kind: str
    uri: str
    sha256: str | None
    manifest_hash: str | None
    dataset_id: str
    config_hash: str
    slot_map_hash: str


@dataclass(frozen=True, slots=True)
class VisualFrameRef:
    visual_frame_id: str
    lidar_frame: int
    camera_frame: int
    lidar_timestamp: float
    camera_timestamp: float
    camera_lidar_dt_sec: float
    image_resource_id: str
    capabilities: tuple[str, ...]
    projection_quality: ImmutableMapping


@dataclass(frozen=True, slots=True)
class EncounterRef:
    encounter_id: str
    part1_trace_event_ids: tuple[str, ...]
    start_lidar_frame: int
    anchor_frame: int
    end_lidar_frame: int
    start_timestamp: float
    anchor_timestamp: float
    end_timestamp: float
    support_frames: tuple[int, ...]
    pointcloud_resource_ids: tuple[str, ...]
    visual_frames: tuple[VisualFrameRef, ...]


@dataclass(frozen=True, slots=True)
class RelationshipRef:
    adjacent_slot_ids: tuple[str, ...]
    conflict_slot_ids: tuple[str, ...]
    shared_evidence_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class QueueItem:
    task_id: str
    slot_id: str
    scope_status: str
    state: str
    agent_observable: bool
    unknown_reasons: tuple[str, ...]
    priority: str
    available_modalities: tuple[str, ...]
    suggested_tools: tuple[str, ...]
    allowed_final_states: tuple[str, ...]
    occupied_evidence: ImmutableMapping
    free_evidence: ImmutableMapping
    audit: ImmutableMapping
    relationships: RelationshipRef
    encounter: EncounterRef


@dataclass(frozen=True, slots=True)
class QueueEnvelope:
    schema_version: str
    queue_id: str
    producer: ImmutableMapping
    resources: Mapping[str, ResourceRef]
    tool_registry_version: str
    items: tuple[QueueItem, ...]
