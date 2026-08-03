"""Part 1 integration for the ParkingAgent v2 per-slot agent pipeline.

This module deliberately separates two concepts which the historical Part 1
artifacts kept close together:

* :class:`SceneSnapshot` is the complete known-slot geometry within the
  requested radius at one reference time; and
* :class:`SlotCase` is an evaluated ``free`` or ``unknown`` Part 1 decision
  which Part 2 is allowed to inspect.

The adapter never consumes ``provisional_candidates`` from ``local_map.json``.
Those entries are visualization-only in the upstream contract and are not a
parking-candidate policy.
"""

from __future__ import annotations

from dataclasses import replace
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from parking_slot_hybrid_3d.accumulation import build_slot_accumulation
from parking_slot_hybrid_3d.config import Hybrid3DConfig
from parking_slot_hybrid_3d.contracts import DecisionState
from parking_slot_hybrid_3d.geometry import map_xy_to_slot_m, metric_slot
from parking_slot_hybrid_3d.io import (
    FramePointProvider,
    load_frame_records,
    load_known_slots,
    write_json_atomic,
)
from parking_slot_hybrid_3d.local_map import LocalMapConfig, select_local_frame_window
from parking_slot_hybrid_3d.part2_evidence import build_lidar_evidence_pack
from parking_slot_hybrid_3d.pipeline import Hybrid3DPipeline, PipelineResult
from parking_slot_hybrid_3d.reporting import OutputContext, write_pipeline_outputs
from parking_slot_part2 import canonical_sha256
from parking_slot_part2.media import load_lidar_evidence_pack

from .contracts import (
    ConfidenceScores,
    EvidenceRecord,
    MapSlot,
    Part1Output,
    SceneSnapshot,
    SensorFrame,
    SlotCase,
)


DEFAULT_RADIUS_M = 30.0
DEFAULT_FRAME_COUNT = 15
_CANDIDATE_STATES = frozenset({"free", "unknown"})


def _load_object(path: Path, name: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"missing {name}: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{name} is not valid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{name} must contain a JSON object: {path}")
    return payload


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


def _positive_number(value: Any, name: str) -> float:
    result = _finite_number(value, name)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def _integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    return int(value)


def _boolean(value: Any, name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in {0, 1}:
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y"}:
            return True
        if normalized in {"0", "false", "no", "n", ""}:
            return False
    raise ValueError(f"{name} must be a boolean")


def _xy(value: Any, name: str) -> tuple[float, float]:
    if (
        isinstance(value, (str, bytes))
        or not isinstance(value, Sequence)
        or len(value) != 2
    ):
        raise ValueError(f"{name} must contain exactly two coordinates")
    return (
        _finite_number(value[0], f"{name}[0]"),
        _finite_number(value[1], f"{name}[1]"),
    )


def _polygon(value: Any, name: str) -> tuple[tuple[float, float], ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must be a coordinate sequence")
    result = tuple(_xy(point, f"{name}[{index}]") for index, point in enumerate(value))
    if len(result) < 3:
        raise ValueError(f"{name} must contain at least three points")
    return result


def _optional_polygon(value: Any, fallback: Any, name: str) -> tuple[tuple[float, float], ...]:
    selected = fallback if value is None else value
    return _polygon(selected, name)


def _resolve_resource_path(raw: Any, *, bases: Sequence[Path]) -> str:
    if raw is None or not str(raw).strip():
        return ""
    path = Path(str(raw))
    if path.is_absolute():
        # Some historical manifests were produced below /srv/jhub while the
        # same dataset is mounted beside this repository below
        # /home/ParkingAgent. Preserve the semantic dataset suffix and resolve
        # it against the current workspace instead of retaining a dead path.
        marker = ("dataset", "dataset", "dataset")
        parts = path.parts
        for index in range(0, len(parts) - len(marker) + 1):
            if tuple(parts[index : index + len(marker)]) != marker:
                continue
            suffix = parts[index + len(marker) :]
            fallback: Path | None = None
            for base in bases:
                # Keep the mounted workspace spelling. ``Path.resolve`` may
                # collapse /home/ParkingAgent/dataset to its legacy
                # /srv/jhub symlink, recreating the dead path we are fixing.
                candidate = (base.parent.joinpath(*marker) / Path(*suffix)).absolute()
                if fallback is None:
                    fallback = candidate
                if candidate.exists():
                    return str(candidate)
            if fallback is not None:
                return str(fallback)
        if path.exists():
            return str(path.resolve(strict=False))
        return str(path.resolve(strict=False))
    for base in bases:
        candidate = (base / path).resolve(strict=False)
        if candidate.exists():
            return str(candidate)
    return str((bases[0] / path).resolve(strict=False))


def _state(value: Any, name: str) -> str:
    state = str(value or "").strip().lower()
    if state not in {"free", "occupied", "unknown"}:
        raise ValueError(f"{name} has unsupported state: {value!r}")
    return state


def _score(value: Any, name: str) -> float:
    score = _finite_number(value, name)
    if not 0.0 <= score <= 1.0:
        raise ValueError(f"{name} must be within [0, 1]")
    return score


def _heuristic_scores(decision: Mapping[str, Any], state: str) -> ConfidenceScores:
    free_evidence = decision.get("free_evidence", {})
    occupied_evidence = decision.get("occupied_evidence", {})
    if not isinstance(free_evidence, Mapping):
        free_evidence = {}
    if not isinstance(occupied_evidence, Mapping):
        occupied_evidence = {}

    # These are independent upstream gate strengths, not calibrated posterior
    # probabilities. Preserve them instead of renormalising or claiming a
    # stronger probabilistic meaning than Part 1 provides.
    free = _score(free_evidence.get("strength", 1.0 if state == "free" else 0.0), "free strength")
    occupied = _score(
        occupied_evidence.get("strength", 1.0 if state == "occupied" else 0.0),
        "occupied strength",
    )
    unknown = max(0.0, min(1.0, 1.0 - max(free, occupied))) if state == "unknown" else 0.0
    return ConfidenceScores(
        free_confidence=free,
        occupied_confidence=occupied,
        unknown_confidence=unknown,
        kind="heuristic_gate_score",
        calibrated=False,
    )


def _sensor_frame(
    row: Mapping[str, Any],
    *,
    part1_dir: Path,
    project_root: Path,
) -> SensorFrame:
    frame_id = _integer(row.get("frame_id"), "manifest frame_id")
    lidar_timestamp_raw = row.get("lidar_timestamp")
    lidar_timestamp = (
        None
        if lidar_timestamp_raw is None
        else _finite_number(lidar_timestamp_raw, f"frame {frame_id} lidar_timestamp")
    )
    camera_frame_raw = row.get("camera_frame")
    camera_frame = None if camera_frame_raw is None else _integer(camera_frame_raw, "camera_frame")
    camera_timestamp_raw = row.get("camera_timestamp")
    camera_timestamp = (
        None
        if camera_timestamp_raw is None
        else _finite_number(camera_timestamp_raw, "camera_timestamp")
    )
    camera_delta_raw = row.get("camera_lidar_dt_sec")
    camera_delta = (
        None
        if camera_delta_raw is None
        else _finite_number(camera_delta_raw, "camera_lidar_dt_sec")
    )
    bases = (project_root, part1_dir)
    map_points_path = _resolve_resource_path(row.get("map_points_path"), bases=bases)
    lidar_path = _resolve_resource_path(row.get("lidar_path"), bases=bases)
    camera_image_path = _resolve_resource_path(row.get("camera_image_path"), bases=bases)
    resources = {
        key: value
        for key, value in {
            "map_points_path": map_points_path,
            "lidar_path": lidar_path,
            "camera_image_path": camera_image_path,
        }.items()
        if value
    }
    return SensorFrame(
        frame_id=frame_id,
        lidar_timestamp=lidar_timestamp,
        map_x=_finite_number(row.get("map_x"), f"frame {frame_id} map_x"),
        map_y=_finite_number(row.get("map_y"), f"frame {frame_id} map_y"),
        map_yaw_rad=_finite_number(row.get("map_yaw"), f"frame {frame_id} map_yaw"),
        map_points_path=map_points_path or None,
        lidar_path=lidar_path or None,
        camera_frame_id=camera_frame,
        camera_timestamp=camera_timestamp,
        camera_image_path=camera_image_path or None,
        camera_lidar_dt_sec=camera_delta,
        camera_match_valid=_boolean(
            row.get("camera_match_valid", False),
            f"frame {frame_id} camera_match_valid",
        ),
        resources=resources,
    )


def _manifest_frames(
    payload: Mapping[str, Any],
    *,
    part1_dir: Path,
    project_root: Path,
) -> tuple[SensorFrame, ...]:
    rows = payload.get("frames")
    if not isinstance(rows, list):
        raise ValueError("local_frame_manifest.frames must be an array")
    frames = tuple(
        sorted(
            (
                _sensor_frame(row, part1_dir=part1_dir, project_root=project_root)
                for row in rows
                if isinstance(row, Mapping)
            ),
            key=lambda frame: frame.frame_id,
        )
    )
    if len(frames) != len(rows):
        raise ValueError("local_frame_manifest contains a non-object frame")
    if not 1 <= len(frames) <= 100:
        raise ValueError(
            "adapted Part1 input requires 1 to 100 causal frames; "
            f"found {len(frames)}"
        )
    if len({frame.frame_id for frame in frames}) != len(frames):
        raise ValueError("local_frame_manifest contains duplicate frame IDs")
    return frames


def _decision_index(payload: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    rows = payload.get("decisions")
    if not isinstance(rows, list):
        raise ValueError("slot_decisions.decisions must be an array")
    result: dict[str, Mapping[str, Any]] = {}
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValueError(f"slot_decisions.decisions[{index}] must be an object")
        slot_id = str(row.get("slot_id", "")).strip()
        if not slot_id:
            raise ValueError(f"slot_decisions.decisions[{index}] has no slot_id")
        if slot_id in result:
            raise ValueError(f"duplicate Part 1 decision for {slot_id}")
        result[slot_id] = row
    return result


def _evidence_context_index(part1_dir: Path) -> dict[str, tuple[int, tuple[int, ...]]]:
    """Load slot-local accumulation anchors and selected frames.

    ``decision.reference_frames`` is not necessarily anchored the same way as
    the slot-local accumulation. The trace is therefore authoritative; the
    historical queue is used only when the trace is unavailable.
    """

    result: dict[str, tuple[int, tuple[int, ...]]] = {}
    trace_path = part1_dir / "decision_trace.jsonl"
    if trace_path.is_file():
        with trace_path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    trace = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"decision_trace.jsonl line {line_number} is invalid JSON"
                    ) from exc
                if not isinstance(trace, Mapping):
                    raise ValueError(
                        f"decision_trace.jsonl line {line_number} must be an object"
                    )
                slot_id = str(trace.get("slot_id", "")).strip()
                stages = trace.get("stages", {})
                accumulation = (
                    stages.get("accumulation", {})
                    if isinstance(stages, Mapping)
                    else {}
                )
                anchor = (
                    accumulation.get("anchor_frame")
                    if isinstance(accumulation, Mapping)
                    else None
                )
                selected_raw = (
                    accumulation.get("selected_frames", [])
                    if isinstance(accumulation, Mapping)
                    else []
                )
                selected = tuple(
                    sorted(
                        {
                            int(value)
                            for value in selected_raw
                            if isinstance(value, int) and not isinstance(value, bool)
                        }
                    )
                ) if isinstance(selected_raw, list) else ()
                if slot_id and isinstance(anchor, int) and not isinstance(anchor, bool):
                    frames = selected if int(anchor) in selected else tuple(sorted((*selected, int(anchor))))
                    result[slot_id] = (int(anchor), frames)

    # Older artifacts may expose this only through the generated Part 2 queue.
    queue_path = part1_dir / "unknown_agent_queue.json"
    if queue_path.is_file():
        queue = _load_object(queue_path, "unknown_agent_queue")
        items = queue.get("items", [])
        if isinstance(items, list):
            for item in items:
                if not isinstance(item, Mapping):
                    continue
                slot_id = str(item.get("slot_id", "")).strip()
                encounter = item.get("encounter", {})
                anchor = (
                    encounter.get("anchor_frame")
                    if isinstance(encounter, Mapping)
                    else None
                )
                support_raw = (
                    encounter.get("support_frames", [])
                    if isinstance(encounter, Mapping)
                    else []
                )
                support = tuple(
                    sorted(
                        {
                            int(value)
                            for value in support_raw
                            if isinstance(value, int) and not isinstance(value, bool)
                        }
                    )
                ) if isinstance(support_raw, list) else ()
                if (
                    slot_id
                    and slot_id not in result
                    and isinstance(anchor, int)
                    and not isinstance(anchor, bool)
                ):
                    frames = support if int(anchor) in support else tuple(sorted((*support, int(anchor))))
                    result[slot_id] = (int(anchor), frames)
    return result


def _local_state_index(payload: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    rows = payload.get("slots")
    if not isinstance(rows, list):
        raise ValueError("local_map.slots must be an array")
    result: dict[str, Mapping[str, Any]] = {}
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValueError(f"local_map.slots[{index}] must be an object")
        slot_id = str(row.get("slot_id", "")).strip()
        if not slot_id:
            raise ValueError(f"local_map.slots[{index}] has no slot_id")
        if slot_id in result:
            raise ValueError(f"duplicate local-map slot {slot_id}")
        _state(row.get("state"), f"local_map slot {slot_id}")
        result[slot_id] = row
    return result


def _queue_lidar_paths(part1_dir: Path) -> dict[str, str]:
    queue_path = part1_dir / "unknown_agent_queue.json"
    if not queue_path.is_file():
        return {}
    queue = _load_object(queue_path, "unknown_agent_queue")
    resources = queue.get("resources", {})
    items = queue.get("items", [])
    if not isinstance(resources, Mapping) or not isinstance(items, list):
        return {}
    result: dict[str, str] = {}
    for item in items:
        if not isinstance(item, Mapping):
            continue
        slot_id = str(item.get("slot_id", "")).strip()
        encounter = item.get("encounter", {})
        if not slot_id or not isinstance(encounter, Mapping):
            continue
        resource_ids = encounter.get("pointcloud_resource_ids", [])
        if not isinstance(resource_ids, list):
            continue
        for resource_id in resource_ids:
            resource = resources.get(str(resource_id))
            if not isinstance(resource, Mapping) or resource.get("kind") != "pointcloud_artifact":
                continue
            resolved = _resolve_resource_path(resource.get("uri"), bases=(part1_dir,))
            if resolved:
                result[slot_id] = resolved
                break
    return result


def _normalise_lidar_resources(
    part1_dir: Path,
    supplied: Mapping[str, str | Path] | None,
) -> dict[str, str]:
    result = _queue_lidar_paths(part1_dir)
    if supplied is None:
        return result
    for slot_id, raw_path in supplied.items():
        key = str(slot_id).strip()
        if not key:
            raise ValueError("lidar_evidence_by_slot contains an empty slot ID")
        result[key] = _resolve_resource_path(raw_path, bases=(part1_dir,))
    return result


def _resolve_map_points_source(
    frame: Any,
    *,
    map_points_dir: Path,
    project_root: Path,
) -> Path | None:
    candidates: list[Path] = []
    raw = getattr(frame, "map_points_path", None)
    if raw is not None:
        path = Path(raw)
        candidates.append(path)
        if not path.is_absolute():
            candidates.extend((project_root / path, map_points_dir / path))
    candidates.append(map_points_dir / f"{int(frame.frame_id):06d}.npz")
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return None


def _build_candidate_lidar_evidence_packs(
    result: PipelineResult,
    provider: FramePointProvider,
    config: Hybrid3DConfig,
    output_dir: Path,
    *,
    dataset_id: str,
    map_points_dir: Path,
    project_root: Path,
) -> dict[str, str]:
    """Create an audited target-local pack for every v2 candidate.

    The historical queue builder omitted Free candidates and partial-route
    Unknown candidates.  That made four current Unknown cases report
    ``lidar_detail unavailable`` even though their Part1 reference frames were
    present.  V2 needs the same identity-bound pack for every Free/Unknown
    candidate. Pack generation remains best-effort: a candidate with no valid
    accumulated observation stays explicit about the missing resource and
    cannot be promoted by LiDAR detail.
    """

    generated: dict[str, str] = {}
    frame_by_id = {int(frame.frame_id): frame for frame in result.frames}
    slot_by_id = {slot.slot_id: slot for slot in result.all_slots}
    scope_by_id = {scope.slot_id: scope for scope in result.scopes}
    config_hash = canonical_sha256(config.to_dict())
    slot_map_hash = canonical_sha256(
        [
            {"slot_id": slot.slot_id, "polygon_map": slot.polygon_map.tolist()}
            for slot in result.all_slots
        ]
    )
    pack_dir = output_dir / "part2_lidar_evidence"
    pack_dir.mkdir(parents=True, exist_ok=True)

    for decision in result.decisions:
        if decision.state not in {DecisionState.FREE, DecisionState.UNKNOWN}:
            continue
        slot = slot_by_id.get(decision.slot_id)
        scope = scope_by_id.get(decision.slot_id)
        if slot is None or scope is None:
            continue
        try:
            accumulation = build_slot_accumulation(
                slot,
                scope,
                result.frames,
                provider,
                result.map_units_per_meter,
                config,
            )
            if not accumulation.observations:
                continue
            source_paths: dict[int, Path] = {}
            for frame_id in accumulation.selected_frames:
                frame = frame_by_id.get(int(frame_id))
                if frame is None:
                    raise ValueError("selected frame is absent from the Part1 window")
                source = _resolve_map_points_source(
                    frame,
                    map_points_dir=map_points_dir,
                    project_root=project_root,
                )
                if source is None:
                    raise ValueError("selected frame map-points source is unavailable")
                source_paths[int(frame_id)] = source

            metric = metric_slot(slot, result.map_units_per_meter)
            adjacent_polygons = {
                adjacent_id: map_xy_to_slot_m(
                    slot_by_id[adjacent_id].polygon_map,
                    metric,
                )
                for adjacent_id in slot.adjacent_slots
                if adjacent_id in slot_by_id
            }
            encounter_id = (
                f"encounter-{min(accumulation.selected_frames):06d}-"
                f"{max(accumulation.selected_frames):06d}"
            )
            pack = build_lidar_evidence_pack(
                pack_dir / f"v2_candidate_{decision.slot_id}.npz",
                accumulation=accumulation,
                slot=metric,
                source_paths=source_paths,
                task_id=f"parking-slot-agent-v2:{decision.slot_id}:{encounter_id}",
                encounter_id=encounter_id,
                dataset_id=str(dataset_id),
                config_hash=config_hash,
                slot_map_hash=slot_map_hash,
                adjacent_polygons_local_m=adjacent_polygons,
            )
            generated[decision.slot_id] = str(pack.path.resolve())
        except (OSError, RuntimeError, ValueError):
            continue
    return generated


def _full_map_slots(
    payload: Mapping[str, Any],
    *,
    anchor_xy: tuple[float, float],
    map_units_per_meter: float,
    requested_radius_m: float,
    local_states: Mapping[str, Mapping[str, Any]],
) -> tuple[MapSlot, ...]:
    rows = payload.get("slots")
    if not isinstance(rows, list):
        raise ValueError("slot database has no slots array")
    declared_count = payload.get("slot_count")
    if declared_count is not None and int(declared_count) != len(rows):
        raise ValueError("slot database slot_count does not match slots")

    selected: list[MapSlot] = []
    seen: set[str] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValueError(f"slot database slots[{index}] must be an object")
        slot_id = str(row.get("slot_id", "")).strip()
        if not slot_id or slot_id in seen:
            raise ValueError(f"slot database has missing or duplicate slot_id at index {index}")
        seen.add(slot_id)
        polygon = _polygon(row.get("polygon_map"), f"{slot_id}.polygon_map")
        center = _xy(row.get("center_map"), f"{slot_id}.center_map")
        distance_m = math.hypot(center[0] - anchor_xy[0], center[1] - anchor_xy[1]) / map_units_per_meter
        if distance_m > requested_radius_m + 1e-9:
            continue
        local = local_states.get(slot_id)
        state = None if local is None else _state(local.get("state"), f"local_map slot {slot_id}")
        adjacent_raw = row.get("adjacent_slots", []) or []
        if not isinstance(adjacent_raw, list):
            raise ValueError(f"{slot_id}.adjacent_slots must be an array")
        selected.append(
            MapSlot(
                slot_id=slot_id,
                polygon_map=polygon,
                center_map=center,
                heading_deg=_finite_number(row.get("heading_deg"), f"{slot_id}.heading_deg"),
                core_polygon_map=_optional_polygon(
                    row.get("core_polygon_map"),
                    row.get("inner_polygon", polygon),
                    f"{slot_id}.core_polygon_map",
                ),
                margin_polygon_map=_optional_polygon(
                    row.get("margin_polygon_map"),
                    row.get("margin_polygon", polygon),
                    f"{slot_id}.margin_polygon_map",
                ),
                adjacent_slot_ids=tuple(str(value) for value in adjacent_raw),
                state=state,
                observed=local is not None,
                distance_to_anchor_m=distance_m,
            )
        )
    return tuple(sorted(selected, key=lambda slot: (float(slot.distance_to_anchor_m or 0.0), slot.slot_id)))


def _evidence_frames(
    decision: Mapping[str, Any],
    local_slot: Mapping[str, Any],
    available_frame_ids: set[int],
) -> tuple[int, ...]:
    raw = decision.get("reference_frames")
    if not isinstance(raw, (list, tuple)) or not raw:
        raw = local_slot.get("observed_frame_ids", [])
    if not isinstance(raw, (list, tuple)):
        raw = []
    result: list[int] = []
    for value in raw:
        if isinstance(value, bool):
            continue
        try:
            frame_id = int(value)
        except (TypeError, ValueError):
            continue
        if frame_id in available_frame_ids and frame_id not in result:
            result.append(frame_id)
    return tuple(sorted(result))


def build_part1_output_from_directory(
    part1_dir: str | Path,
    slot_db_path: str | Path,
    dataset_id: str,
    requested_radius_m: float = DEFAULT_RADIUS_M,
    *,
    lidar_evidence_by_slot: Mapping[str, str | Path] | None = None,
    allow_partial_coverage: bool = False,
) -> Part1Output:
    """Adapt a raw Hybrid3D Part 1 directory to the v2 object contract.

    All known slots whose centres are within ``requested_radius_m`` of the
    snapshot anchor become :class:`MapSlot` objects. Only slots actually
    evaluated by Part 1 as ``free`` or ``unknown`` become :class:`SlotCase`
    candidates. Candidate order is free first, unknown second, then increasing
    Euclidean distance and stable slot ID.
    """

    raw_dir = Path(part1_dir).resolve(strict=False)
    full_slot_db_path = Path(slot_db_path).resolve(strict=False)
    dataset = str(dataset_id).strip()
    if not dataset:
        raise ValueError("dataset_id must be a non-empty string")
    radius_m = _positive_number(requested_radius_m, "requested_radius_m")
    project_root = Path(__file__).resolve().parents[1]

    local_map = _load_object(raw_dir / "local_map.json", "local_map")
    coverage = local_map.get("lidar_coverage", {})
    if isinstance(coverage, Mapping):
        nominal_radius = coverage.get("nominal_radius_m")
        if nominal_radius is not None:
            source_radius = _positive_number(nominal_radius, "source nominal radius")
            if source_radius + 1e-6 < radius_m:
                raise ValueError(
                    "raw Part1 artifact covers only "
                    f"{source_radius:g} m; rerun Part1 for requested {radius_m:g} m"
                )
    decisions_payload = _load_object(raw_dir / "slot_decisions.json", "slot_decisions")
    manifest = _load_object(raw_dir / "local_frame_manifest.json", "local_frame_manifest")
    slot_db = _load_object(full_slot_db_path, "slot database")

    coverage = local_map.get("lidar_coverage", {})
    source_radius_raw = (
        coverage.get("nominal_radius_m") if isinstance(coverage, Mapping) else None
    )
    if source_radius_raw is not None:
        source_radius_m = _positive_number(
            source_radius_raw, "local_map lidar_coverage.nominal_radius_m"
        )
        if radius_m > source_radius_m + 1e-9 and not allow_partial_coverage:
            raise ValueError(
                "requested_radius_m exceeds the source Part1 evidence radius "
                f"({radius_m:g} m > {source_radius_m:g} m); rerun Part1 or set "
                "allow_partial_coverage=True explicitly"
            )

    local_states = _local_state_index(local_map)
    decisions = _decision_index(decisions_payload)
    frames = _manifest_frames(manifest, part1_dir=raw_dir, project_root=project_root)
    frame_by_id = {frame.frame_id: frame for frame in frames}

    anchor_payload = local_map.get("anchor_pose")
    if not isinstance(anchor_payload, Mapping):
        raise ValueError("local_map.anchor_pose must be an object")
    anchor_frame_id = _integer(anchor_payload.get("frame_id"), "anchor frame_id")
    if frames[-1].frame_id != anchor_frame_id:
        raise ValueError("snapshot anchor must be the final frame in the causal manifest")
    anchor_frame = frame_by_id[anchor_frame_id]
    anchor_xy = _xy(anchor_payload.get("map_xy"), "anchor_pose.map_xy")
    anchor_yaw = _finite_number(anchor_payload.get("map_yaw_rad"), "anchor_pose.map_yaw_rad")
    if not math.isclose(anchor_frame.map_x, anchor_xy[0], abs_tol=1e-9) or not math.isclose(
        anchor_frame.map_y, anchor_xy[1], abs_tol=1e-9
    ):
        raise ValueError("local_map anchor position disagrees with local_frame_manifest")
    if not math.isclose(anchor_frame.map_yaw_rad, anchor_yaw, abs_tol=1e-9):
        raise ValueError("local_map anchor yaw disagrees with local_frame_manifest")
    anchor_timestamp_raw = anchor_payload.get("lidar_timestamp", anchor_frame.lidar_timestamp)
    if anchor_timestamp_raw is None:
        raise ValueError("snapshot anchor has no LiDAR timestamp")
    anchor_timestamp = _finite_number(anchor_timestamp_raw, "anchor timestamp")

    map_units_per_meter = _positive_number(
        slot_db.get("map_units_per_meter", local_map.get("map_units_per_meter")),
        "map_units_per_meter",
    )
    local_scale = _positive_number(local_map.get("map_units_per_meter"), "local map scale")
    if not math.isclose(map_units_per_meter, local_scale, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("full slot database and local_map use different map scales")

    map_slots = _full_map_slots(
        slot_db,
        anchor_xy=anchor_xy,
        map_units_per_meter=map_units_per_meter,
        requested_radius_m=radius_m,
        local_states=local_states,
    )
    map_slot_by_id = {slot.slot_id: slot for slot in map_slots}
    snapshot_id = f"{dataset}:lidar-frame-{anchor_frame_id:06d}"
    coordinate_frame = str(local_map.get("coordinate_frame") or "pose_corrected_map")
    scene_resources = {
        "part1_raw_directory": str(raw_dir),
        "local_map_json_path": str(raw_dir / "local_map.json"),
        "local_map_image_path": str(raw_dir / "local_map.png"),
        "slot_decisions_path": str(raw_dir / "slot_decisions.json"),
        "local_frame_manifest_path": str(raw_dir / "local_frame_manifest.json"),
        "full_slot_database_path": str(full_slot_db_path),
        "camera_image_path": anchor_frame.camera_image_path or "",
        "map_points_path": anchor_frame.map_points_path or "",
        "lidar_path": anchor_frame.lidar_path or "",
        "requested_radius_m": str(radius_m),
        "source_part1_radius_m": str(
            local_map.get("lidar_coverage", {}).get("nominal_radius_m", "unknown")
            if isinstance(local_map.get("lidar_coverage"), Mapping)
            else "unknown"
        ),
    }
    scene_resources = {
        key: value for key, value in scene_resources.items() if str(value).strip()
    }
    scene = SceneSnapshot(
        snapshot_id=snapshot_id,
        anchor_frame_id=anchor_frame_id,
        anchor_timestamp=anchor_timestamp,
        anchor_pose_map=(anchor_xy[0], anchor_xy[1], anchor_yaw),
        radius_m=radius_m,
        map_units_per_meter=map_units_per_meter,
        coordinate_frame=coordinate_frame,
        slots=map_slots,
        frames=frames,
        resources=scene_resources,
    )

    lidar_paths = _normalise_lidar_resources(raw_dir, lidar_evidence_by_slot)
    evidence_context_by_slot = _evidence_context_index(raw_dir)
    cases: list[SlotCase] = []
    available_frame_ids = set(frame_by_id)
    for slot_id, local_slot in local_states.items():
        local_state = _state(local_slot.get("state"), f"local_map slot {slot_id}")
        if local_state not in _CANDIDATE_STATES:
            continue
        # A state outside the requested scene radius cannot be a candidate for
        # this snapshot, even if it existed in a wider upstream artifact.
        map_slot = map_slot_by_id.get(slot_id)
        if map_slot is None:
            continue
        decision = decisions.get(slot_id)
        if decision is None:
            raise ValueError(f"evaluated candidate {slot_id} has no slot_decisions entry")
        decision_state = _state(decision.get("state"), f"slot_decisions {slot_id}")
        if decision_state != local_state:
            raise ValueError(f"Part 1 state mismatch for {slot_id}: {local_state} != {decision_state}")

        lidar_evidence_path = lidar_paths.get(slot_id, "")
        pack_context: tuple[int, tuple[int, ...]] | None = None
        if lidar_evidence_path:
            pack = load_lidar_evidence_pack(
                lidar_evidence_path,
                expected_slot_id=slot_id,
            )
            pack_context = (
                int(pack.metadata["anchor_frame"]),
                tuple(int(value) for value in pack.selected_frames),
            )
        evidence_context = pack_context or evidence_context_by_slot.get(slot_id)
        if evidence_context is None:
            evidence_frame_ids = _evidence_frames(
                decision, local_slot, available_frame_ids
            )
        else:
            evidence_frame_ids = tuple(
                frame_id
                for frame_id in evidence_context[1]
                if frame_id in available_frame_ids
            )
        if not evidence_frame_ids:
            evidence_frame_ids = (anchor_frame_id,)
        evidence_anchor_id = (
            evidence_context[0]
            if evidence_context is not None
            else evidence_frame_ids[-1]
        )
        if evidence_anchor_id not in available_frame_ids:
            evidence_anchor_id = evidence_frame_ids[-1]
        if evidence_anchor_id not in evidence_frame_ids:
            evidence_frame_ids = tuple(sorted((*evidence_frame_ids, evidence_anchor_id)))
        evidence_anchor = frame_by_id[evidence_anchor_id]
        scores = _heuristic_scores(decision, local_state)
        raw_unknown_reasons = decision.get("unknown_reasons", []) or []
        if not isinstance(raw_unknown_reasons, (list, tuple)):
            raise ValueError(f"slot_decisions {slot_id}.unknown_reasons must be an array")
        unknown_reasons = tuple(str(value) for value in raw_unknown_reasons if str(value).strip())
        if local_state == "unknown" and not unknown_reasons:
            unknown_reasons = (str(decision.get("decision_reason") or "insufficient_terminal_evidence"),)
        if local_state == "free":
            unknown_reasons = ()

        case_resources = {
            "part1_raw_directory": str(raw_dir),
            "camera_image_path": anchor_frame.camera_image_path or "",
            "evidence_camera_image_path": evidence_anchor.camera_image_path or "",
            "lidar_evidence_path": lidar_evidence_path,
            "lidar_evidence_status": (
                "available" if lidar_evidence_path else "not_generated_for_candidate"
            ),
            "evidence_map_points_path": evidence_anchor.map_points_path or "",
            "evidence_lidar_path": evidence_anchor.lidar_path or "",
            "local_map_json_path": str(raw_dir / "local_map.json"),
            "full_slot_database_path": str(full_slot_db_path),
        }
        case_resources = {
            key: value for key, value in case_resources.items() if str(value).strip()
        }
        evidence_resource_keys = sorted(
            key
            for key, value in case_resources.items()
            if key.endswith("_path") and Path(value).is_file()
        )
        decision_reason = str(decision.get("decision_reason") or "part1_lidar_decision")
        reason_codes = tuple(dict.fromkeys(unknown_reasons or (decision_reason,)))
        part1_evidence = EvidenceRecord(
            evidence_id=f"part1-lidar:{snapshot_id}:{slot_id}",
            tool_name="part1_15frame",
            round_index=0,
            status="ok",
            artifact_paths=list(
                dict.fromkeys(case_resources[key] for key in evidence_resource_keys)
            ),
            summary=f"Part1 {len(frames)}-frame LiDAR decision: {decision_reason}",
            metadata={
                "source": "parking_slot_hybrid_3d.slot_decisions",
                "decision": dict(decision),
                "local_map_slot": dict(local_slot),
                "evidence_anchor_frame_id": evidence_anchor_id,
                "evidence_frame_ids": list(evidence_frame_ids),
                "scores_calibrated": False,
            },
            modality="lidar",
            supports_state=local_state,
            scores=scores,
            reason_codes=reason_codes,
            resource_keys=evidence_resource_keys,
        )
        cases.append(
            SlotCase(
                case_id=f"{snapshot_id}:{slot_id}",
                snapshot_id=snapshot_id,
                snapshot_frame_id=anchor_frame_id,
                snapshot_timestamp=anchor_timestamp,
                snapshot_pose_map=(anchor_xy[0], anchor_xy[1], anchor_yaw),
                slot=map_slot,
                evidence_anchor_frame_id=evidence_anchor_id,
                evidence_frame_ids=evidence_frame_ids,
                part1_state=local_state,
                part1_scores=scores,
                current_state=local_state,
                current_scores=scores,
                decision_reason=decision_reason,
                unknown_reasons=unknown_reasons,
                resources=case_resources,
                evidence=[part1_evidence],
                max_rounds=3,
            )
        )

    cases.sort(
        key=lambda case: (
            0 if case.part1_state == "free" else 1,
            float(case.slot.distance_to_anchor_m or 0.0),
            case.slot.slot_id,
        )
    )
    return Part1Output(scene=scene, slot_cases=tuple(cases))


def run_part1(
    frames_csv: str | Path,
    slot_db_path: str | Path,
    map_points_dir: str | Path,
    output_dir: str | Path,
    dataset_id: str,
    *,
    anchor_frame: int | None = None,
    frame_count: int = DEFAULT_FRAME_COUNT,
    requested_radius_m: float = DEFAULT_RADIUS_M,
    overwrite: bool = False,
    camera_calibration: str | Path | None = None,
    camera_projection_audit: str | Path | None = None,
) -> Part1Output:
    """Run Hybrid3D on a causal local window and emit the v2 Part 1 contract.

    Raw, reproducible upstream artifacts are written below ``output_dir/raw``;
    the adapted contract is written to ``output_dir/part1_output.json``.
    ``frame_count`` is exposed for an explicit validation error, but v2
    intentionally requires exactly fifteen frames.
    """

    if isinstance(frame_count, bool) or int(frame_count) != DEFAULT_FRAME_COUNT:
        raise ValueError(f"ParkingAgent v2 Part 1 requires exactly {DEFAULT_FRAME_COUNT} frames")
    radius_m = _positive_number(requested_radius_m, "requested_radius_m")
    frames_path = Path(frames_csv).resolve(strict=False)
    slots_path = Path(slot_db_path).resolve(strict=False)
    map_points_path = Path(map_points_dir).resolve(strict=False)
    destination = Path(output_dir).resolve(strict=False)
    raw_output = destination / "raw"
    project_root = Path(__file__).resolve().parents[1]

    records = select_local_frame_window(
        load_frame_records(frames_path),
        anchor_frame_id=anchor_frame,
        frame_count=DEFAULT_FRAME_COUNT,
    )
    if len(records) != DEFAULT_FRAME_COUNT:
        raise ValueError(f"anchor does not have {DEFAULT_FRAME_COUNT} causal input records")
    frame_span = int(records[-1].frame_id - records[0].frame_id)
    config = replace(
        Hybrid3DConfig(),
        scope_max_distance_m=radius_m,
        frame_stride=1,
        window_before=max(Hybrid3DConfig().window_before, frame_span),
        window_after=max(Hybrid3DConfig().window_after, frame_span),
    )
    config.validate()
    known_slots, map_units_per_meter = load_known_slots(slots_path)
    provider = FramePointProvider(
        {frame.frame_id: frame for frame in records},
        map_points_path,
        cache_size=max(32, len(records)),
        project_root=project_root,
    )
    result = Hybrid3DPipeline(
        known_slots,
        records,
        provider,
        map_units_per_meter,
        config,
        phase="full",
    ).run()

    local_settings = LocalMapConfig(local_radius_m=radius_m, frame_count=DEFAULT_FRAME_COUNT)
    local_settings.validate()
    write_pipeline_outputs(
        result,
        raw_output,
        config,
        OutputContext(
            dataset_id=str(dataset_id),
            slot_database_path=slots_path,
            frames_csv_path=frames_path,
            map_points_dir=map_points_path,
            camera_calibration_path=(
                None if camera_calibration is None else Path(camera_calibration).resolve(strict=False)
            ),
            camera_projection_audit_path=(
                None
                if camera_projection_audit is None
                else Path(camera_projection_audit).resolve(strict=False)
            ),
            map_coordinate_frame="pose_corrected_map",
            local_map_config=local_settings,
        ),
        overwrite=overwrite,
    )
    candidate_lidar_evidence = _build_candidate_lidar_evidence_packs(
        result,
        provider,
        config,
        raw_output,
        dataset_id=str(dataset_id),
        map_points_dir=map_points_path,
        project_root=project_root,
    )
    output = build_part1_output_from_directory(
        raw_output,
        slots_path,
        str(dataset_id),
        requested_radius_m=radius_m,
        lidar_evidence_by_slot=candidate_lidar_evidence,
    )
    destination.mkdir(parents=True, exist_ok=True)
    write_json_atomic(destination / "part1_output.json", output.to_dict())
    return output


__all__ = [
    "DEFAULT_FRAME_COUNT",
    "DEFAULT_RADIUS_M",
    "build_part1_output_from_directory",
    "run_part1",
]
