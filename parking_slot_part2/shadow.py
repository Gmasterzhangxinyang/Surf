"""Deterministic, externally auditable shadow subsets for Part2.

The selector deliberately operates only on the validated Part1 queue.  It does
not inspect artifacts, labels, previous Part2 proposals, or ground truth.  A
derived queue keeps the Part1 producer/task identities intact while receiving a
new queue identity because its item/resource set and resource URIs differ.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping
from urllib.parse import unquote, urlparse

from .contracts import QueueEnvelope, QueueItem, ResourceRef
from .grouping import GROUPING_POLICY_VERSION, build_groups
from .queueing import canonical_json_bytes, canonical_sha256, make_queue_id, validate_queue


SHADOW_SELECTION_SCHEMA_VERSION = "part2-shadow-selection/1.0"
SHADOW_SELECTION_POLICY_VERSION = "part2-shadow-stratified-unknown/1.0"

REGRESSION_SLOT_IDS = (
    "slot_0784",
    "slot_0991",
    "slot_1050",
    "slot_1068",
    "slot_1071",
)

STRATUM_ORDER = (
    "conflict",
    "pose",
    "ownership_outside",
    "boundary_static",
    "observation_deficit",
)

# The ordering above is also the exclusive classification precedence.  For
# example, an occupied/free conflict that is additionally pose-sensitive stays
# in the conflict stratum instead of being sampled twice.
STRATUM_REASON_CODES: Mapping[str, frozenset[str]] = MappingProxyType(
    {
        "conflict": frozenset({"occupied_free_conflict"}),
        "pose": frozenset({"pose_sensitive_terminal"}),
        "ownership_outside": frozenset(
            {
                "outside_residual_conflict",
                "weak_shared_ownership",
                "weak_ownership_external",
                "weak_ownership_shared",
            }
        ),
        "boundary_static": frozenset(
            {
                "boundary_dominated",
                "linear_static_structure",
                "low_height_structure",
                "weak_static_structure",
            }
        ),
        "observation_deficit": frozenset(
            {
                "high_occlusion",
                "insufficient_3d_voxels",
                "insufficient_height_layers",
                "insufficient_height_span",
                "insufficient_ray_frames",
                "insufficient_support_frames",
                "insufficient_valid_frames",
                "insufficient_vehicle_height",
                "insufficient_vehicle_points",
                "insufficient_viewpoint_separation",
                "insufficient_viewpoints",
                "large_unobserved_component",
                "low_free_volume_coverage",
                "low_near_ground_coverage",
                "low_temporal_support",
                "no_valid_observations",
                "unresolved_core_hit_evidence",
                "weak_core_clearance_conflict",
                "weak_obstacle_evidence",
                "weak_temporal_inconsistent",
                "weak_vehicle_evidence",
                "weak_visibility_limited",
            }
        ),
    }
)

_REQUIRED_RESOURCE_KINDS = frozenset(
    {
        "slot_database",
        "corrected_frames",
        "map_points_root",
        "camera_calibration",
        "base_slot_decisions",
    }
)


@dataclass(frozen=True, slots=True)
class ShadowSelection:
    """Selected seeds plus the conflict-group closure used by a shadow run."""

    seed_records: tuple[Mapping[str, Any], ...]
    seed_task_ids: tuple[str, ...]
    selected_task_ids: tuple[str, ...]
    closure_added_task_ids: tuple[str, ...]
    group_expansions: tuple[Mapping[str, Any], ...]


@dataclass(frozen=True, slots=True)
class ShadowSubset:
    """A validated subset queue and its separate selection manifest."""

    queue: QueueEnvelope
    queue_payload: dict[str, Any]
    selection_manifest: dict[str, Any]
    selection: ShadowSelection


def _plain_json(value: Any) -> Any:
    """Make an independent JSON-compatible value from immutable contracts."""

    import json

    return json.loads(canonical_json_bytes(value))


def _stratum(item: QueueItem) -> str | None:
    reasons = set(item.unknown_reasons)
    for name in STRATUM_ORDER:
        if reasons & STRATUM_REASON_CODES[name]:
            return name
    return None


def _strength(item: QueueItem, axis: str) -> float:
    evidence = item.occupied_evidence if axis == "occupied" else item.free_evidence
    value = evidence.get("strength")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{item.slot_id} is missing a numeric {axis}_evidence.strength")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{item.slot_id} has a non-finite {axis}_evidence.strength")
    return result


def select_shadow_tasks(queue: QueueEnvelope) -> ShadowSelection:
    """Select five fixed regressions and 20 deterministic stratified seeds.

    Each exclusive stratum contributes the two highest occupied-strength items
    and then the two highest free-strength remaining items.  Fixed regressions
    and prior picks are excluded.  Ties are broken by the canonical task ID.
    The result is expanded to the fixed point of :func:`build_groups`, including
    overlapping groups produced for components larger than four.
    """

    by_slot = {item.slot_id: item for item in queue.items}
    missing = tuple(slot_id for slot_id in REGRESSION_SLOT_IDS if slot_id not in by_slot)
    if missing:
        raise ValueError("missing fixed regression slot(s): " + ", ".join(missing))

    seed_records: list[dict[str, Any]] = []
    seed_task_ids: list[str] = []
    used_task_ids: set[str] = set()
    for order, slot_id in enumerate(REGRESSION_SLOT_IDS, start=1):
        item = by_slot[slot_id]
        seed_task_ids.append(item.task_id)
        used_task_ids.add(item.task_id)
        seed_records.append(
            {
                "source": "fixed_regression",
                "order": order,
                "slot_id": item.slot_id,
                "task_id": item.task_id,
            }
        )

    classified = {item.task_id: _stratum(item) for item in queue.items}
    for stratum in STRATUM_ORDER:
        pool = [
            item
            for item in queue.items
            if classified[item.task_id] == stratum and item.task_id not in used_task_ids
        ]
        if len(pool) < 4:
            raise ValueError(
                f"stratum {stratum!r} has {len(pool)} eligible item(s); four are required"
            )

        for axis in ("occupied", "free"):
            available = [item for item in pool if item.task_id not in used_task_ids]
            ranked = sorted(
                available,
                key=lambda item: (-_strength(item, axis), item.task_id),
            )
            if len(ranked) < 2:
                raise ValueError(
                    f"stratum {stratum!r} cannot provide two distinct {axis} seeds"
                )
            for rank, item in enumerate(ranked[:2], start=1):
                score = _strength(item, axis)
                seed_task_ids.append(item.task_id)
                used_task_ids.add(item.task_id)
                seed_records.append(
                    {
                        "source": "stratified_strength",
                        "stratum": stratum,
                        "score_axis": f"{axis}_strength",
                        "rank": rank,
                        "strength": score,
                        "slot_id": item.slot_id,
                        "task_id": item.task_id,
                    }
                )

    selected = set(seed_task_ids)
    expansions: list[dict[str, Any]] = []
    groups = build_groups(queue)
    expanded_group_ids: set[str] = set()
    changed = True
    while changed:
        changed = False
        for group in groups:
            if group.group_id in expanded_group_ids:
                continue
            triggers = tuple(sorted(selected & set(group.task_ids)))
            if not triggers:
                continue
            added = tuple(sorted(set(group.task_ids) - selected))
            expanded_group_ids.add(group.group_id)
            if not added:
                continue
            selected.update(added)
            changed = True
            expansions.append(
                {
                    "group_id": group.group_id,
                    "encounter_id": group.encounter_id,
                    "trigger_task_ids": list(triggers),
                    "added_task_ids": list(added),
                }
            )

    closure_added = tuple(sorted(selected - set(seed_task_ids)))
    return ShadowSelection(
        seed_records=tuple(MappingProxyType(record) for record in seed_records),
        seed_task_ids=tuple(seed_task_ids),
        selected_task_ids=tuple(sorted(selected)),
        closure_added_task_ids=closure_added,
        group_expansions=tuple(MappingProxyType(record) for record in expansions),
    )


def _resolve_local_uri(uri: str, source_base_dir: Path, *, require_dir: bool) -> str:
    parsed = urlparse(uri)
    if parsed.scheme not in ("", "file"):
        raise ValueError(f"shadow queues require local resource URIs, got {uri!r}")
    if parsed.scheme == "file" and parsed.netloc not in ("", "localhost"):
        raise ValueError(f"shadow queues do not support remote file URIs: {uri!r}")

    raw_path = unquote(parsed.path) if parsed.scheme == "file" else uri
    path = Path(raw_path)
    if path.is_absolute():
        candidates = (path,)
    else:
        roots = (source_base_dir, *source_base_dir.parents, Path.cwd())
        unique: list[Path] = []
        seen: set[Path] = set()
        for root in roots:
            candidate = root / path
            key = candidate.absolute()
            if key not in seen:
                seen.add(key)
                unique.append(candidate)
        candidates = tuple(unique)

    resolved = next((candidate.resolve() for candidate in candidates if candidate.exists()), None)
    if resolved is None:
        raise ValueError(f"resource URI cannot be resolved to an existing path: {uri!r}")
    if require_dir and not resolved.is_dir():
        raise ValueError(f"resource URI must resolve to a directory: {uri!r}")
    if not require_dir and not resolved.is_file():
        raise ValueError(f"resource URI must resolve to a file: {uri!r}")
    return str(resolved)


def _resource_payload(resource: ResourceRef, source_base_dir: Path) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "kind": resource.kind,
        "uri": _resolve_local_uri(
            resource.uri,
            source_base_dir,
            require_dir=resource.manifest_hash is not None,
        ),
        "dataset_id": resource.dataset_id,
        "config_hash": resource.config_hash,
        "slot_map_hash": resource.slot_map_hash,
    }
    if resource.sha256 is not None:
        payload["sha256"] = resource.sha256
    if resource.manifest_hash is not None:
        payload["manifest_hash"] = resource.manifest_hash
    return payload


def _visual_payload(frame: Any) -> dict[str, Any]:
    return {
        "visual_frame_id": frame.visual_frame_id,
        "lidar_frame": frame.lidar_frame,
        "camera_frame": frame.camera_frame,
        "lidar_timestamp": frame.lidar_timestamp,
        "camera_timestamp": frame.camera_timestamp,
        "camera_lidar_dt_sec": frame.camera_lidar_dt_sec,
        "image_resource_id": frame.image_resource_id,
        "capabilities": list(frame.capabilities),
        "projection_quality": _plain_json(frame.projection_quality),
    }


def _item_payload(item: QueueItem) -> dict[str, Any]:
    encounter = item.encounter
    return {
        "task_id": item.task_id,
        "slot_id": item.slot_id,
        "scope_status": item.scope_status,
        "state": item.state,
        "agent_observable": item.agent_observable,
        "unknown_reasons": list(item.unknown_reasons),
        "priority": item.priority,
        "available_modalities": list(item.available_modalities),
        "suggested_tools": list(item.suggested_tools),
        "allowed_final_states": list(item.allowed_final_states),
        "occupied_evidence": _plain_json(item.occupied_evidence),
        "free_evidence": _plain_json(item.free_evidence),
        "audit": _plain_json(item.audit),
        "relationships": {
            "adjacent_slot_ids": list(item.relationships.adjacent_slot_ids),
            "conflict_slot_ids": list(item.relationships.conflict_slot_ids),
            "shared_evidence_ids": list(item.relationships.shared_evidence_ids),
        },
        "encounter": {
            "encounter_id": encounter.encounter_id,
            "part1_trace_event_ids": list(encounter.part1_trace_event_ids),
            "start_lidar_frame": encounter.start_lidar_frame,
            "anchor_frame": encounter.anchor_frame,
            "end_lidar_frame": encounter.end_lidar_frame,
            "start_timestamp": encounter.start_timestamp,
            "anchor_timestamp": encounter.anchor_timestamp,
            "end_timestamp": encounter.end_timestamp,
            "support_frames": list(encounter.support_frames),
            "pointcloud_resource_ids": list(encounter.pointcloud_resource_ids),
            "visual_frames": [_visual_payload(frame) for frame in encounter.visual_frames],
        },
    }


def build_shadow_subset(
    queue: QueueEnvelope,
    *,
    source_base_dir: str | Path,
) -> ShadowSubset:
    """Build and validate the strict queue subset plus its external manifest.

    Every retained resource URI becomes an absolute local path.  Required
    producer resources are retained, while task artifacts are retained only if
    referenced by a selected item.  The source queue is never mutated.
    """

    base_dir = Path(source_base_dir).resolve()
    selection = select_shadow_tasks(queue)
    selected_ids = set(selection.selected_task_ids)
    selected_items = tuple(
        sorted(
            (item for item in queue.items if item.task_id in selected_ids),
            key=lambda item: item.task_id,
        )
    )

    resource_ids = {
        resource_id
        for resource_id, resource in queue.resources.items()
        if resource.kind in _REQUIRED_RESOURCE_KINDS
    }
    for item in selected_items:
        resource_ids.update(item.encounter.pointcloud_resource_ids)
        resource_ids.update(frame.image_resource_id for frame in item.encounter.visual_frames)

    queue_payload: dict[str, Any] = {
        "schema_version": queue.schema_version,
        "producer": _plain_json(queue.producer),
        "resources": {
            resource_id: _resource_payload(queue.resources[resource_id], base_dir)
            for resource_id in sorted(resource_ids)
        },
        "tool_registry_version": queue.tool_registry_version,
        "items": [_item_payload(item) for item in selected_items],
    }
    queue_payload["queue_id"] = make_queue_id(queue_payload)
    validated = validate_queue(queue_payload)

    manifest: dict[str, Any] = {
        "schema_version": SHADOW_SELECTION_SCHEMA_VERSION,
        "policy_version": SHADOW_SELECTION_POLICY_VERSION,
        "source_queue_id": queue.queue_id,
        "subset_queue_id": validated.queue_id,
        "grouping_policy_version": GROUPING_POLICY_VERSION,
        "fixed_regression_slot_ids": list(REGRESSION_SLOT_IDS),
        "stratum_order": list(STRATUM_ORDER),
        "stratum_reason_codes": {
            name: sorted(STRATUM_REASON_CODES[name]) for name in STRATUM_ORDER
        },
        "seeds": [_plain_json(record) for record in selection.seed_records],
        "seed_task_ids": list(selection.seed_task_ids),
        "closure_added_task_ids": list(selection.closure_added_task_ids),
        "selected_task_ids": list(selection.selected_task_ids),
        "group_expansions": [_plain_json(record) for record in selection.group_expansions],
        "resource_ids": sorted(resource_ids),
        "counts": {
            "fixed_regression_seeds": len(REGRESSION_SLOT_IDS),
            "stratified_seeds": len(selection.seed_task_ids) - len(REGRESSION_SLOT_IDS),
            "seed_tasks": len(selection.seed_task_ids),
            "closure_added_tasks": len(selection.closure_added_task_ids),
            "selected_tasks": len(selection.selected_task_ids),
            "retained_resources": len(resource_ids),
        },
    }
    manifest["selection_id"] = canonical_sha256(manifest)

    return ShadowSubset(
        queue=validated,
        queue_payload=queue_payload,
        selection_manifest=manifest,
        selection=selection,
    )


__all__ = [
    "REGRESSION_SLOT_IDS",
    "SHADOW_SELECTION_POLICY_VERSION",
    "SHADOW_SELECTION_SCHEMA_VERSION",
    "STRATUM_ORDER",
    "STRATUM_REASON_CODES",
    "ShadowSelection",
    "ShadowSubset",
    "build_shadow_subset",
    "select_shadow_tasks",
]
