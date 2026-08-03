from __future__ import annotations

import hashlib
import html
import json
import math
import resource
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from parking_slot_part2 import (
    QUEUE_SCHEMA_VERSION,
    canonical_sha256,
    make_queue_id,
    make_task_id,
    validate_queue,
)

from .accumulation import build_slot_accumulation
from .camera import (
    CameraSelectionConfig,
    assess_pre_anchor_candidates,
    load_camera_model,
    preselect_pre_anchor_frames,
    project_map_polygon,
)
from .config import Hybrid3DConfig
from .contracts import DecisionState, FrameRecord, SlotDecision
from .geometry import map_xy_to_slot_m, metric_slot
from .io import (
    FramePointProvider,
    write_csv_atomic,
    write_json_atomic,
    write_jsonl_atomic,
    write_text_atomic,
)
from .local_map import (
    LocalMapConfig,
    actual_lidar_coverage_polygon,
    build_local_map_snapshot,
    restrict_result_to_local_map,
)
from .part2_evidence import (
    LIDAR_EVIDENCE_PACK_SCHEMA_VERSION,
    build_lidar_evidence_pack,
)
from .pipeline import PipelineResult


@dataclass(frozen=True)
class OutputContext:
    dataset_id: str = "unspecified-dataset"
    slot_database_path: Path | None = None
    frames_csv_path: Path | None = None
    map_points_dir: Path | None = None
    camera_calibration_path: Path | None = None
    camera_projection_audit_path: Path | None = None
    map_coordinate_frame: str | None = None
    local_map_config: LocalMapConfig | None = None


def part2_candidate_bearing_deg(
    frame: FrameRecord,
    slot_center_map: np.ndarray,
) -> float:
    """Signed slot bearing relative to the current vehicle heading."""

    center = np.asarray(slot_center_map, dtype=np.float64).reshape(2)
    delta = center - np.asarray([frame.map_x, frame.map_y], dtype=np.float64)
    if not np.isfinite(delta).all() or float(np.linalg.norm(delta)) <= 1e-12:
        raise ValueError("Part2 candidate bearing requires a distinct finite slot center")
    absolute = math.atan2(float(delta[1]), float(delta[0]))
    relative = (absolute - float(frame.map_yaw) + math.pi) % (2.0 * math.pi) - math.pi
    return float(math.degrees(relative))


def part2_candidate_in_forward_fov(
    frame: FrameRecord,
    slot_center_map: np.ndarray,
    half_fov_deg: float,
) -> tuple[bool, float]:
    """Apply the closed forward field-of-regard gate used by Part2."""

    if not math.isfinite(half_fov_deg) or not 0.0 < half_fov_deg <= 180.0:
        raise ValueError("half_fov_deg must be finite and in (0, 180]")
    bearing = part2_candidate_bearing_deg(frame, slot_center_map)
    return abs(bearing) <= float(half_fov_deg) + 1e-9, bearing


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _ensure_output_dir(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)


def _scope_rows(result: PipelineResult) -> list[dict[str, Any]]:
    return [
        {
            "slot_id": scope.slot_id,
            "scope_status": scope.scope_status.value,
            "near_frames": scope.near_frames,
            "crossing_frames": scope.crossing_frames,
            "hit_frames": scope.hit_frames,
            "missing_frames": scope.missing_frames,
            "core_ray_coverage": scope.core_ray_coverage,
            "agent_observable": scope.agent_observable,
            "reasons": scope.reasons,
        }
        for scope in result.scopes
    ]


def _decision_rows(result: PipelineResult) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for decision in result.decisions:
        rows.append(
            {
                "slot_id": decision.slot_id,
                "scope_status": decision.scope_status.value,
                "state": decision.state.value,
                "decision_reason": decision.decision_reason,
                "unknown_reasons": decision.unknown_reasons,
                "reference_frames": decision.reference_frames,
                "agent_observable": decision.agent_context.agent_observable,
                "occupied_strong": decision.occupied_evidence.strong,
                "occupied_weak": decision.occupied_evidence.weak,
                "occupied_strength": decision.occupied_evidence.strength,
                "weak_primary_reason": decision.weak_evidence.primary_reason,
                "free_strong": decision.free_evidence.strong,
                "free_strength": decision.free_evidence.strength,
                "free_volume_coverage": decision.free_evidence.observed_volume_ratio,
                "free_occlusion_ratio": decision.free_evidence.occlusion_ratio,
                "stability_pass_ratio": decision.stability.pass_ratio,
                "stability_stable": decision.stability.stable,
            }
        )
    return rows


def _summary(
    result: PipelineResult,
    config: Hybrid3DConfig,
    local_map: Mapping[str, Any],
) -> dict[str, Any]:
    scope_counts = Counter(scope.scope_status.value for scope in result.scopes)
    state_counts = Counter(decision.state.value for decision in result.decisions)
    reason_counts = Counter(decision.decision_reason for decision in result.decisions)
    gate_failures: Counter[str] = Counter()
    weak_reason_counts: Counter[str] = Counter()
    for decision in result.decisions:
        gate_failures.update(decision.occupied_evidence.failures)
        gate_failures.update(decision.free_evidence.failures)
        if decision.state is DecisionState.UNKNOWN and decision.weak_evidence.active:
            weak_reason_counts.update((decision.weak_evidence.primary_reason,))
    observable_unknown = sum(
        decision.state is DecisionState.UNKNOWN
        and decision.agent_context.agent_observable
        for decision in result.decisions
    )
    unknown_count = state_counts.get("unknown", 0)
    provisional_candidate_ids = [
        str(slot_id)
        for slot_id in local_map.get("provisional_candidate_ids", ())
    ]
    return {
        "schema_version": "2.0",
        "pipeline": config.pipeline,
        "phase": result.phase,
        "map_semantics": "local_incomplete_lidar_snapshot",
        "metric_semantics": "local_evidence_and_coverage_without_gt",
        "gt_status": "unavailable",
        "map_total": result.map_total,
        "local_slot_count": result.map_total,
        "known_slot_database_total": int(
            local_map.get("counts", {}).get("known_slot_database_total", result.map_total)
        ),
        "unobserved_slots_omitted": int(
            local_map.get("counts", {}).get("unobserved_slots_omitted", 0)
        ),
        "anchor_frame": local_map.get("anchor_pose", {}).get("frame_id"),
        "lidar_frame_count": local_map.get("lidar_window", {}).get("frame_count", 0),
        # Part1 deliberately does not make the production parking choice.
        # Do not promote provisional_candidate_ids[0] into either field.
        "candidate_slot_id": None,
        "candidate": None,
        "candidate_selection_policy_status": "tbd",
        "provisional_candidate_ids": provisional_candidate_ids,
        "provisional_candidate_count": len(provisional_candidate_ids),
        "part2_camera_observability_included": False,
        "in_route_scope": scope_counts.get("in_route_scope", 0),
        "partial_route_scope": scope_counts.get("partial_route_scope", 0),
        "occupied": state_counts.get("occupied", 0),
        "free": state_counts.get("free", 0),
        "unknown": unknown_count,
        "agent_observable_unknown": observable_unknown,
        "agent_unobservable_unknown": unknown_count - observable_unknown,
        "decision_reason_counts": dict(sorted(reason_counts.items())),
        "gate_failure_counts": dict(sorted(gate_failures.items())),
        "weak_unknown_reason_counts": dict(sorted(weak_reason_counts.items())),
        "processing_seconds": result.processing_seconds,
        "peak_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "cache_stats": dict(sorted(result.cache_stats.items())),
    }


def _write_fallback_inputs(
    result: PipelineResult,
    output_dir: Path,
    context: OutputContext,
) -> tuple[Path, Path, Path]:
    # Never bind the Part2 hand-off to the complete prior slot database or the
    # full-route frame CSV.  They are upstream resources, not the current
    # Part1 local perception result.
    local_ids = {slot.slot_id for slot in result.all_slots}
    slot_path = output_dir / "local_slot_database.json"
    write_json_atomic(
        slot_path,
        {
            "schema_version": "part1-local-slot-database/1.0",
            "map_semantics": "local_incomplete_lidar_snapshot",
            "coordinate_frame": context.map_coordinate_frame,
            "map_units_per_meter": result.map_units_per_meter,
            "slot_count": len(result.all_slots),
            "slots": [
                {
                    **asdict(slot),
                    "adjacent_slots": [
                        slot_id
                        for slot_id in slot.adjacent_slots
                        if slot_id in local_ids
                    ],
                }
                for slot in result.all_slots
            ],
        },
    )
    frames_path = output_dir / "local_frame_manifest.json"
    write_json_atomic(
        frames_path,
        {
            "schema_version": "part1-local-frame-window/1.0",
            "frames": [asdict(frame) for frame in result.frames],
        },
    )
    if context.camera_calibration_path is not None and context.camera_calibration_path.is_file():
        camera_path = context.camera_calibration_path.resolve()
    else:
        camera_path = output_dir / "camera_capability.json"
        write_json_atomic(
            camera_path,
            {
                "schema_version": "camera-capability/1.0",
                "metric_depth": False,
                "depth_capability": "relative_only",
                "note": "No metric camera-depth calibration is asserted by Part1.",
            },
        )
    return slot_path, frames_path, camera_path


def _resolve_map_points_path(frame: FrameRecord, context: OutputContext) -> Path | None:
    candidates: list[Path] = []
    if frame.map_points_path is not None:
        candidates.append(frame.map_points_path)
        if not frame.map_points_path.is_absolute():
            candidates.append(Path.cwd() / frame.map_points_path)
            if context.map_points_dir is not None:
                candidates.append(context.map_points_dir / frame.map_points_path)
    if context.map_points_dir is not None:
        candidates.append(context.map_points_dir / f"{frame.frame_id:06d}.npz")
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return None


def _queue_payload(
    result: PipelineResult,
    output_dir: Path,
    config: Hybrid3DConfig,
    context: OutputContext,
) -> dict[str, Any]:
    slot_path, frames_path, camera_path = _write_fallback_inputs(result, output_dir, context)
    camera_model = load_camera_model(
        (
            context.camera_calibration_path
            if context.camera_calibration_path is not None
            and context.camera_calibration_path.is_file()
            else None
        ),
        dataset_id=context.dataset_id,
        projection_audit_path=context.camera_projection_audit_path,
    )
    camera_selection_config = CameraSelectionConfig()
    decisions_path = output_dir / "slot_decisions.json"
    output_root = output_dir.resolve()
    config_hash = canonical_sha256(config.to_dict())
    slot_map_hash = canonical_sha256(
        [
            {"slot_id": slot.slot_id, "polygon_map": slot.polygon_map.tolist()}
            for slot in result.all_slots
        ]
    )
    frame_window_hash = canonical_sha256(
        [
            {
                "frame_id": frame.frame_id,
                "map_x": frame.map_x,
                "map_y": frame.map_y,
                "map_yaw": frame.map_yaw,
                "lidar_timestamp": frame.lidar_timestamp,
            }
            for frame in result.frames
        ]
    )
    run_id = canonical_sha256(
        {
            "pipeline": config.pipeline,
            "dataset_id": context.dataset_id,
            "config_hash": config_hash,
            "slot_map_hash": slot_map_hash,
            "frame_window_hash": frame_window_hash,
            "phase": result.phase,
        }
    )
    producer = {
        "pipeline": config.pipeline,
        "run_id": run_id,
        "config_hash": config_hash,
        "dataset_id": context.dataset_id,
        "slot_map_hash": slot_map_hash,
    }

    def resource_uri(path: Path) -> str:
        resolved = path.resolve()
        try:
            return resolved.relative_to(output_root).as_posix()
        except ValueError:
            return str(resolved)

    def resource_record(kind: str, path: Path, sha256: str | None = None) -> dict[str, Any]:
        return {
            "kind": kind,
            "uri": resource_uri(path),
            "sha256": sha256 or _file_sha256(path),
            "dataset_id": context.dataset_id,
            "config_hash": config_hash,
            "slot_map_hash": slot_map_hash,
        }

    resources: dict[str, dict[str, Any]] = {
        "slot-database": resource_record("slot_database", slot_path),
        "corrected-frames": resource_record("corrected_frames", frames_path),
        "camera-calibration": resource_record("camera_calibration", camera_path),
        "base-decisions": resource_record("base_slot_decisions", decisions_path),
        "map-points-root": {
            "kind": "map_points_root",
            "uri": "unavailable://raw-map-points-not-exported",
            "manifest_hash": canonical_sha256(
                [
                    {
                        "frame_id": frame.frame_id,
                        "map_points_path": str(frame.map_points_path or ""),
                    }
                    for frame in result.frames
                ]
            ),
            "dataset_id": context.dataset_id,
            "config_hash": config_hash,
            "slot_map_hash": slot_map_hash,
        },
    }
    frame_by_id = {frame.frame_id: frame for frame in result.frames}
    slot_by_id = {slot.slot_id: slot for slot in result.all_slots}
    scope_by_slot = {scope.slot_id: scope for scope in result.scopes}
    metric_by_id = {
        slot.slot_id: metric_slot(slot, result.map_units_per_meter)
        for slot in result.all_slots
    }
    point_provider = (
        FramePointProvider(
            frame_by_id,
            context.map_points_dir,
            cache_size=128,
            project_root=Path.cwd(),
        )
        if context.map_points_dir is not None
        else None
    )
    trace_by_slot = {str(trace["slot_id"]): trace for trace in result.traces}
    items: list[dict[str, Any]] = []
    resource_id_by_path: dict[tuple[str, Path], str] = {}

    def add_file_resource(kind: str, path: Path, prefix: str) -> str:
        key = (kind, path)
        existing = resource_id_by_path.get(key)
        if existing is not None:
            return existing
        identifier = f"{prefix}-{len(resource_id_by_path):06d}"
        resources[identifier] = resource_record(kind, path)
        resource_id_by_path[key] = identifier
        return identifier

    current_frame = max(result.frames, key=lambda frame: frame.frame_id)
    for decision in result.decisions:
        if decision.state is not DecisionState.UNKNOWN or not decision.agent_context.agent_observable:
            continue
        in_forward_fov, current_bearing_deg = part2_candidate_in_forward_fov(
            current_frame,
            slot_by_id[decision.slot_id].center_map,
            config.part2_candidate_half_fov_deg,
        )
        if not in_forward_fov:
            continue
        trace = trace_by_slot.get(decision.slot_id, {})
        accumulation_stage = trace.get("stages", {}).get("accumulation", {})
        selected_frames = tuple(int(value) for value in accumulation_stage.get("selected_frames", ()))
        anchor_frame = accumulation_stage.get("anchor_frame")
        if anchor_frame is None or not selected_frames:
            continue
        anchor_frame = int(anchor_frame)
        anchor_record = frame_by_id.get(anchor_frame)
        if anchor_record is None:
            continue
        accumulation = None
        source_paths: dict[int, Path] = {}
        scope = scope_by_slot.get(decision.slot_id)
        if point_provider is not None and scope is not None:
            try:
                candidate = build_slot_accumulation(
                    slot_by_id[decision.slot_id],
                    scope,
                    result.frames,
                    point_provider,
                    result.map_units_per_meter,
                    config,
                )
                if candidate.selected_frames != selected_frames:
                    raise ValueError("rebuilt accumulation selected_frames mismatch")
                for frame_id in selected_frames:
                    record = frame_by_id.get(frame_id)
                    resolved = (
                        _resolve_map_points_path(record, context)
                        if record is not None
                        else None
                    )
                    if resolved is None:
                        raise ValueError("selected frame source is unavailable")
                    source_paths[frame_id] = resolved
                accumulation = candidate
            except (OSError, ValueError):
                accumulation = None
                source_paths = {}

        anchor_timestamp = float(
            anchor_record.lidar_timestamp
            if anchor_record.lidar_timestamp is not None
            else anchor_frame
        )
        visual_frames: list[dict[str, Any]] = []
        ground_z_by_frame: dict[int, float] = {}
        if camera_model.calibration is not None and point_provider is not None:
            camera_candidates = preselect_pre_anchor_frames(
                result.frames,
                slot_by_id[decision.slot_id].center_map,
                anchor_frame,
                result.map_units_per_meter,
                config=camera_selection_config,
            )
            for frame in camera_candidates:
                try:
                    points = point_provider.load(frame.frame_id)
                except Exception:
                    continue
                finite_z = points[np.isfinite(points).all(axis=1), 2]
                if len(finite_z):
                    ground_z_by_frame[frame.frame_id] = float(
                        np.quantile(finite_z, config.ground_fallback_quantile)
                    )
            try:
                camera_batch = assess_pre_anchor_candidates(
                    result.frames,
                    slot_by_id[decision.slot_id].polygon_map,
                    slot_by_id[decision.slot_id].center_map,
                    anchor_frame,
                    anchor_timestamp,
                    result.map_units_per_meter,
                    ground_z_by_frame,
                    camera_model,
                    sync_source="native",
                    config=camera_selection_config,
                )
            except ValueError:
                camera_batch = None
            for assessment in (() if camera_batch is None else camera_batch.selected):
                frame = assessment.frame
                if (
                    not assessment.intended_capabilities
                    or frame.camera_image_path is None
                    or frame.camera_frame is None
                    or frame.lidar_timestamp is None
                    or frame.camera_timestamp is None
                    or frame.camera_lidar_dt_sec is None
                ):
                    continue
                image_path = frame.camera_image_path.resolve()
                image_id = add_file_resource("rgb_frame", image_path, "rgb")
                quality = dict(assessment.projection_quality)
                adjacent_uv: dict[str, list[list[float | None]]] = {}
                for adjacent_id in slot_by_id[decision.slot_id].adjacent_slots:
                    adjacent = slot_by_id.get(adjacent_id)
                    if adjacent is None:
                        continue
                    projected, _ = project_map_polygon(
                        adjacent.polygon_map,
                        (frame.map_x, frame.map_y, frame.map_yaw),
                        result.map_units_per_meter,
                        ground_z_by_frame[frame.frame_id],
                        camera_model.calibration,
                    )
                    adjacent_uv[adjacent_id] = [
                        [
                            float(point[0]) if np.isfinite(point[0]) else None,
                            float(point[1]) if np.isfinite(point[1]) else None,
                        ]
                        for point in projected
                    ]
                quality["adjacent_polygons_uv"] = adjacent_uv
                visual_frames.append(
                    {
                        "visual_frame_id": f"visual-{decision.slot_id}-{frame.frame_id}",
                        "lidar_frame": frame.frame_id,
                        "camera_frame": int(frame.camera_frame),
                        "lidar_timestamp": float(frame.lidar_timestamp),
                        "camera_timestamp": float(frame.camera_timestamp),
                        "camera_lidar_dt_sec": float(frame.camera_lidar_dt_sec),
                        "image_resource_id": image_id,
                        "capabilities": list(assessment.intended_capabilities),
                        "projection_quality": quality,
                    }
                )
        selected_records = [frame_by_id[frame_id] for frame_id in selected_frames if frame_id in frame_by_id]
        encounter_lidar_frames = list(selected_frames) + [
            int(frame["lidar_frame"]) for frame in visual_frames
        ]
        start_frame = min(encounter_lidar_frames)
        end_frame = max(encounter_lidar_frames)
        timestamps = [
            float(frame.lidar_timestamp if frame.lidar_timestamp is not None else frame.frame_id)
            for frame in selected_records
        ]
        timestamps.extend(float(frame["lidar_timestamp"]) for frame in visual_frames)
        timestamps.extend(float(frame["camera_timestamp"]) for frame in visual_frames)
        start_timestamp = min(timestamps)
        end_timestamp = max(timestamps)
        encounter_id = f"encounter-{start_frame:06d}-{end_frame:06d}"
        for frame in visual_frames:
            frame["projection_quality"]["encounter_id"] = encounter_id
        task_id = make_task_id(producer, decision.slot_id, encounter_id)

        pointcloud_ids: list[str] = []
        if accumulation is not None and accumulation.observations:
            target_slot = metric_by_id[decision.slot_id]
            adjacent_polygons = {
                adjacent_id: map_xy_to_slot_m(
                    slot_by_id[adjacent_id].polygon_map,
                    target_slot,
                )
                for adjacent_id in slot_by_id[decision.slot_id].adjacent_slots
                if adjacent_id in slot_by_id
            }
            pack_path = (
                output_dir
                / "part2_lidar_evidence"
                / f"{task_id.split(':', 1)[-1]}.npz"
            )
            try:
                pack = build_lidar_evidence_pack(
                    pack_path,
                    accumulation=accumulation,
                    slot=target_slot,
                    source_paths=source_paths,
                    task_id=task_id,
                    encounter_id=encounter_id,
                    dataset_id=context.dataset_id,
                    config_hash=config_hash,
                    slot_map_hash=slot_map_hash,
                    adjacent_polygons_local_m=adjacent_polygons,
                )
                pointcloud_ids.append(
                    add_file_resource(
                        "pointcloud_artifact",
                        pack.path,
                        "lidar-evidence",
                    )
                )
            except (OSError, RuntimeError, ValueError):
                pointcloud_ids = []

        available_modalities = [
            modality
            for modality, present in (
                ("lidar", bool(pointcloud_ids)),
                ("rgb", bool(visual_frames)),
            )
            if present
        ]
        if not available_modalities:
            continue
        suggested_tools = [
            tool
            for tool in decision.agent_context.suggested_tools
            if (
                (tool == "inspect_lidar_map" and pointcloud_ids)
                or (tool in {"inspect_rgb_frame", "inspect_rgb_sequence"} and visual_frames)
            )
        ]
        if not suggested_tools:
            if pointcloud_ids:
                suggested_tools.append("inspect_lidar_map")
            if visual_frames:
                suggested_tools.extend(("inspect_rgb_frame", "inspect_rgb_sequence"))
        items.append(
            {
                "task_id": task_id,
                "slot_id": decision.slot_id,
                "scope_status": decision.scope_status.value,
                "state": "unknown",
                "agent_observable": True,
                "unknown_reasons": list(decision.unknown_reasons or ("insufficient_terminal_evidence",)),
                "priority": decision.agent_context.priority or "inspect_vehicle_shape_and_occlusion",
                "available_modalities": available_modalities,
                "suggested_tools": suggested_tools,
                "allowed_final_states": ["occupied", "free", "unknown"],
                "occupied_evidence": {
                    "strength": decision.occupied_evidence.strength,
                    "support_frame_count": decision.occupied_evidence.support_frame_count,
                    "failures": sorted(set(decision.occupied_evidence.failures)),
                    "weak_assessment": asdict(decision.weak_evidence),
                },
                "free_evidence": {
                    "strength": decision.free_evidence.strength,
                    "core_ray_coverage": decision.free_evidence.core_ray_coverage,
                    "failures": sorted(set(decision.free_evidence.failures)),
                },
                "audit": {
                    "part1_contract": "slot-hybrid-3d/1.0",
                    "candidate_policy": "forward_field_of_regard",
                    "candidate_policy_anchor_frame": int(current_frame.frame_id),
                    "candidate_relative_bearing_deg": current_bearing_deg,
                    "candidate_half_fov_deg": float(
                        config.part2_candidate_half_fov_deg
                    ),
                    "candidate_total_fov_deg": float(
                        2.0 * config.part2_candidate_half_fov_deg
                    ),
                    "lidar_evidence_schema": (
                        LIDAR_EVIDENCE_PACK_SCHEMA_VERSION if pointcloud_ids else "unavailable"
                    ),
                    "depth_capability": "relative_only",
                    "metric_depth_asserted": False,
                },
                "relationships": {
                    "adjacent_slot_ids": [
                        slot_id
                        for slot_id in slot_by_id[decision.slot_id].adjacent_slots
                        if slot_id in slot_by_id
                    ],
                    "conflict_slot_ids": [],
                    "shared_evidence_ids": [encounter_id],
                },
                "encounter": {
                    "encounter_id": encounter_id,
                    "part1_trace_event_ids": [str(trace.get("trace_event_id", _trace_id_fallback(decision.slot_id)))],
                    "start_lidar_frame": start_frame,
                    "anchor_frame": anchor_frame,
                    "end_lidar_frame": end_frame,
                    "start_timestamp": start_timestamp,
                    "anchor_timestamp": anchor_timestamp,
                    "end_timestamp": end_timestamp,
                    "support_frames": list(selected_frames),
                    "pointcloud_resource_ids": pointcloud_ids,
                    "visual_frames": visual_frames,
                },
            }
        )

    payload: dict[str, Any] = {
        "schema_version": QUEUE_SCHEMA_VERSION,
        "producer": producer,
        "resources": resources,
        "tool_registry_version": "part2-tools/1.0",
        "items": sorted(items, key=lambda item: str(item["slot_id"])),
    }
    payload["queue_id"] = make_queue_id(payload)
    validate_queue(payload, base_dir=output_dir)
    return payload


def _trace_id_fallback(slot_id: str) -> str:
    return f"trace:{slot_id}"


def _report_html(
    summary: dict[str, Any],
    result: PipelineResult,
    local_map: Mapping[str, Any],
) -> str:
    rows = "".join(
        "<tr>"
        f"<td>{html.escape(decision.slot_id)}</td>"
        f"<td>{html.escape(decision.scope_status.value)}</td>"
        f"<td>{html.escape(decision.state.value)}</td>"
        f"<td>{html.escape(decision.decision_reason)}</td>"
        "</tr>"
        for decision in result.decisions
    )
    return (
        "<!doctype html><meta charset='utf-8'><title>Part1 local LiDAR map</title>"
        "<style>body{font-family:system-ui;margin:24px;max-width:1200px}"
        "img{max-width:100%;height:auto}.warning{padding:12px;background:#fff7ed;"
        "border:1px solid #ea580c}</style>"
        "<h1>Part1 local, incomplete LiDAR map</h1>"
        "<p class='warning'>This is a causal snapshot from only "
        f"<b>{int(local_map.get('lidar_window', {}).get('frame_count', 0))}</b> "
        "consecutive LiDAR frames. Far slots are unobserved and carry no state. "
        "No complete-parking-lot truth map is displayed.</p>"
        "<p class='warning'>Part1 makes no final parking-candidate selection: "
        "candidate and candidate_slot_id are always null. At most two nearby "
        "free/unknown slots may be marked A/B as a provisional visualization "
        "shortlist; occupied slots are excluded and the production selection "
        "policy is TBD. Part2 Camera observability is not evaluated or drawn "
        "in this Part1 figure.</p>"
        "<p><img src='local_map.png' alt='Part1 local LiDAR parking map'></p>"
        "<p>No ground truth was used; counts below are not accuracy metrics.</p>"
        f"<pre>{html.escape(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))}</pre>"
        "<table><thead><tr><th>slot</th><th>scope</th><th>state</th><th>reason</th></tr></thead>"
        f"<tbody>{rows}</tbody></table>"
    )


def write_pipeline_outputs(
    result: PipelineResult,
    output_dir: str | Path,
    config: Hybrid3DConfig,
    context: OutputContext | None = None,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    output = Path(output_dir)
    _ensure_output_dir(output, overwrite)
    context = context or OutputContext()
    local_settings = context.local_map_config or LocalMapConfig()
    local_settings.validate()
    coverage_polygon = None
    if context.map_points_dir is not None:
        coverage_provider = FramePointProvider(
            {frame.frame_id: frame for frame in result.frames},
            context.map_points_dir,
            cache_size=max(16, len(result.frames)),
            project_root=Path.cwd(),
        )
        coverage_polygon = actual_lidar_coverage_polygon(
            result.frames,
            coverage_provider,
            result.map_units_per_meter,
            local_settings.local_radius_m,
        )
    local_map = build_local_map_snapshot(
        result,
        local_settings,
        map_coordinate_frame=context.map_coordinate_frame,
        coverage_polygon_map=coverage_polygon,
    )
    local_result = restrict_result_to_local_map(result, local_map)
    write_json_atomic(output / "local_map.json", local_map)
    # Import lazily so non-reporting users of the evidence package do not pay
    # the matplotlib import cost.
    from .local_map_visualization import render_local_map

    render_local_map(local_map, result.all_slots, output / "local_map.png")

    scope_rows = _scope_rows(local_result)
    write_csv_atomic(
        output / "known_slot_scope.csv",
        scope_rows,
        (
            "slot_id",
            "scope_status",
            "near_frames",
            "crossing_frames",
            "hit_frames",
            "missing_frames",
            "core_ray_coverage",
            "agent_observable",
            "reasons",
        ),
    )
    write_jsonl_atomic(output / "decision_trace.jsonl", local_result.traces)
    summary = _summary(local_result, config, local_map)
    write_json_atomic(output / "summary.json", summary)
    write_text_atomic(
        output / "report.html",
        _report_html(summary, local_result, local_map),
    )

    if local_result.phase != "scope":
        decision_payload = {
            "schema_version": config.schema_version,
            "pipeline": config.pipeline,
            "phase": local_result.phase,
            "decisions": [asdict(decision) for decision in local_result.decisions],
        }
        write_json_atomic(output / "slot_decisions.json", decision_payload)
        write_csv_atomic(
            output / "slot_decisions.csv",
            _decision_rows(local_result),
            (
                "slot_id",
                "scope_status",
                "state",
                "decision_reason",
                "unknown_reasons",
                "reference_frames",
                "agent_observable",
                "occupied_strong",
                "occupied_weak",
                "occupied_strength",
                "weak_primary_reason",
                "free_strong",
                "free_strength",
                "free_volume_coverage",
                "free_occlusion_ratio",
                "stability_pass_ratio",
                "stability_stable",
            ),
        )
        queue = _queue_payload(local_result, output, config, context)
        write_json_atomic(output / "unknown_agent_queue.json", queue)
    return summary
