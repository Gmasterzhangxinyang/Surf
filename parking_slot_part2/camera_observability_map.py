"""Bridge a Part1 local map into the real Part2 Camera pre-call gate.

The bridge never substitutes the vehicle pose for the Camera pose and never
fills missing calibration or obstacle-map facts.  It runs the existing
``assess_camera_observability`` implementation once per displayed target and
packages those exact assessments/debug traces for map visualization.
"""

from __future__ import annotations

from copy import deepcopy
import math
from typing import Any, Mapping, Sequence

from .camera_observability import (
    CAMERA_OBSERVABILITY_POLICY_VERSION,
    CameraObservabilityConfig,
    CameraObservabilityInput,
    assess_camera_observability,
)


CAMERA_MAP_OVERLAY_SCHEMA_VERSION = "camera-observability-map-overlay/1.0"
CAMERA_MAP_REPORT_SCHEMA_VERSION = "camera-observability-map-report/1.0"
_CRITICAL_GEOMETRY_REASONS = frozenset(
    {
        "camera_pose_missing",
        "camera_calibration_missing",
        "camera_intrinsics_missing",
        "camera_extrinsics_missing",
        "camera_pose_stale",
        "camera_pose_uncertainty_too_high",
        "coordinate_frame_mismatch",
        "camera_fov_missing",
        "camera_fov_policy_mismatch",
    }
)
_STATIC_DISPLAY_INVALID_REASONS = frozenset(
    {
        "static_obstacle_layer_missing",
        "static_obstacle_region_incomplete",
        "coordinate_frame_mismatch",
    }
)


def _sequence(value: Any, name: str) -> tuple[Any, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must be a sequence")
    return tuple(value)


def _slot_index(rows: Sequence[Mapping[str, Any]], name: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(rows):
        if not isinstance(raw, Mapping):
            raise ValueError(f"{name}[{index}] must be a mapping")
        slot_id = str(raw.get("slot_id", "")).strip()
        if not slot_id:
            raise ValueError(f"{name}[{index}] has no slot_id")
        if slot_id in result:
            raise ValueError(f"duplicate slot_id in {name}: {slot_id}")
        result[slot_id] = deepcopy(dict(raw))
    return result


def _display_targets(
    local_map: Mapping[str, Any],
    target_slot_ids: Sequence[str] | None,
) -> tuple[tuple[str, str], ...]:
    raw_slots = _sequence(local_map.get("slots", ()), "local_map.slots")
    local_ids = set(_slot_index(raw_slots, "local_map.slots"))
    if target_slot_ids is not None:
        ids = tuple(str(item).strip() for item in target_slot_ids)
        if not ids or len(ids) > 2 or len(set(ids)) != len(ids):
            raise ValueError("target_slot_ids must contain one or two unique IDs")
        if any(item not in local_ids for item in ids):
            raise ValueError("every Camera target must be a local Part1 slot")
        return tuple((slot_id, chr(ord("A") + index)) for index, slot_id in enumerate(ids))

    raw_candidates = _sequence(
        local_map.get("provisional_candidates", ()),
        "local_map.provisional_candidates",
    )
    if len(raw_candidates) > 2:
        raise ValueError("Part1 provisional candidate display exceeds two targets")
    targets: list[tuple[str, str]] = []
    seen_ids: set[str] = set()
    seen_labels: set[str] = set()
    for index, raw in enumerate(raw_candidates):
        if not isinstance(raw, Mapping):
            raise ValueError("provisional candidate must be a mapping")
        slot_id = str(raw.get("slot_id", "")).strip()
        if slot_id not in local_ids:
            raise ValueError("provisional Camera target is not a local slot")
        label = str(raw.get("display_label", chr(ord("A") + index))).strip()
        label = label or chr(ord("A") + index)
        if slot_id in seen_ids:
            raise ValueError("provisional Camera targets must be unique")
        if label in seen_labels:
            raise ValueError("provisional Camera display labels must be unique")
        targets.append((slot_id, label))
        seen_ids.add(slot_id)
        seen_labels.add(label)
    if not targets:
        raise ValueError("at least one local Camera display target is required")
    return tuple(targets)


def _merge_scene_slots(
    local_map: Mapping[str, Any],
    camera_scene: Mapping[str, Any] | None,
) -> list[dict[str, Any]]:
    local_rows = _sequence(local_map.get("slots", ()), "local_map.slots")
    local_index = _slot_index(local_rows, "local_map.slots")
    scene_rows: tuple[Any, ...] = ()
    if camera_scene is not None and camera_scene.get("slots") is not None:
        scene_rows = _sequence(camera_scene["slots"], "camera_scene.slots")
    scene_index = _slot_index(scene_rows, "camera_scene.slots") if scene_rows else {}

    merged: list[dict[str, Any]] = []
    for slot_id, local in sorted(local_index.items()):
        record = deepcopy(scene_index.pop(slot_id, {}))
        # Part1 is authoritative for current local geometry/state. Extra scene
        # fields such as occupancy_polygon_map and ground_z_m are retained.
        for key in (
            "slot_id",
            "polygon_map",
            "center_map",
            "heading_deg",
            "state",
            "scope_status",
            "occupied_probability",
        ):
            if key in local:
                record[key] = deepcopy(local[key])
        record["slot_id"] = slot_id
        merged.append(record)
    # Explicit physical records outside the displayed local state map may
    # still occlude a target. They are retained only when the validated Camera
    # scene supplied them; no state is invented here.
    merged.extend(scene_index[key] for key in sorted(scene_index))
    return merged


def _static_obstacles(scene: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    if scene is None:
        return []
    layer = scene.get("static_obstacle_map", scene.get("static_obstacle_layer"))
    if not isinstance(layer, Mapping):
        return []
    raw = layer.get("static_obstacles", ())
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        return []
    return [deepcopy(dict(item)) for item in raw if isinstance(item, Mapping)]


def _occlusion_objects(
    scene_slots: Sequence[Mapping[str, Any]],
    target_reports: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Return only algorithm-referenced dynamic/slot occluder geometry."""

    referenced_ids = {
        str(object_id)
        for target in target_reports
        for field in ("blocking_object_ids", "potential_occluder_ids")
        for object_id in target.get("assessment", {}).get(field, ())
    }
    if not referenced_ids:
        return []
    result: list[dict[str, Any]] = []
    for raw in scene_slots:
        object_id = str(
            raw.get("object_id", raw.get("slot_id", raw.get("id", "")))
        ).strip()
        if not object_id or object_id not in referenced_ids:
            continue
        polygon = next(
            (
                raw.get(name)
                for name in (
                    "occupancy_polygon_map",
                    "vehicle_polygon_map",
                    "obstacle_polygon_map",
                    "polygon_map",
                    "polygon_xy",
                )
                if raw.get(name) is not None
            ),
            None,
        )
        if polygon is None:
            continue
        result.append(
            {
                "object_id": object_id,
                "object_type": str(raw.get("object_type", raw.get("type", "slot"))),
                "polygon_map": deepcopy(polygon),
            }
        )
    return result


def build_camera_observability_map_report(
    local_map: Mapping[str, Any],
    *,
    camera_scene: Mapping[str, Any] | None = None,
    target_slot_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Run the real Camera gate for up to two local Part1 display targets."""

    if not isinstance(local_map, Mapping):
        raise TypeError("local_map must be a mapping")
    if camera_scene is not None and not isinstance(camera_scene, Mapping):
        raise TypeError("camera_scene must be a mapping or None")
    targets = _display_targets(local_map, target_slot_ids)
    scene: dict[str, Any] = deepcopy(dict(camera_scene or {}))
    scene_slots = _merge_scene_slots(local_map, camera_scene)
    scene["slots"] = scene_slots
    scene["map_units_per_meter"] = local_map.get("map_units_per_meter")
    local_frame = local_map.get("coordinate_frame")
    scene_frame = None if camera_scene is None else camera_scene.get(
        "map_coordinate_frame",
        camera_scene.get("coordinate_frame"),
    )
    frame_conflict = (
        camera_scene is not None
        and scene_frame is not None
        and local_frame is not None
        and str(scene_frame).strip() != str(local_frame).strip()
    )

    limitations: list[str] = []
    if camera_scene is None:
        limitations.append(
            "No validated Camera scene was supplied: calibration, Camera map pose, "
            "target ground height, and complete local wall/column layer remain missing."
        )
    if local_frame is None or not str(local_frame).strip():
        limitations.append(
            "Part1 local_map.coordinate_frame is missing, so Camera-to-slot frame "
            "identity cannot be proven."
        )
    elif frame_conflict:
        limitations.append(
            "Camera scene and Part1 local map declare different coordinate frames."
        )
    local_scale = local_map.get("map_units_per_meter")
    scene_scale = None if camera_scene is None else camera_scene.get("map_units_per_meter")
    scale_conflict = (
        camera_scene is not None
        and scene_scale is not None
        and local_scale is not None
        and not math.isclose(float(scene_scale), float(local_scale), rel_tol=1e-9, abs_tol=1e-12)
    )
    if scale_conflict:
        limitations.append("Camera scene and Part1 local map scales differ.")
    # Part1 owns the slot coordinate system.  A contradictory scene
    # declaration makes that identity unprovable, so pass an unknown frame to
    # the real gate and let its existing fail-closed logic emit the decision.
    scene["map_coordinate_frame"] = (
        None if frame_conflict or scale_conflict else local_frame
    )

    raw_config = scene.get("camera_observability_config")
    if raw_config is None:
        raw_config = {}
    # A malformed explicit policy must not silently become the default policy:
    # fail the bridge before it can claim that an audited algorithm ran.
    config = CameraObservabilityConfig.from_mapping(raw_config)

    target_reports: list[dict[str, Any]] = []
    parsed_pose: tuple[float, float, float] | None = None
    for target_id, display_label in targets:
        payload = deepcopy(scene)
        payload["target_slot_id"] = target_id
        debug_trace: list[dict[str, Any]] = []
        assessment = assess_camera_observability(
            payload,
            config=config,
            debug_trace=debug_trace,
        ).to_dict()
        target_reports.append(
            {
                "display_label": display_label,
                "target_slot_id": target_id,
                "assessment": assessment,
                "debug_trace": debug_trace,
            }
        )
        if parsed_pose is None:
            try:
                parsed = CameraObservabilityInput.from_mapping(payload)
            except (TypeError, ValueError):
                parsed = None
            if parsed is not None:
                parsed_pose = parsed.camera_pose_map_xyyaw

    settings = config
    critical_reasons = {
        str(reason)
        for target in target_reports
        for reason in target["assessment"].get("reason_codes", ())
        if str(reason) in _CRITICAL_GEOMETRY_REASONS
    }
    camera_pose_for_drawing = None if critical_reasons else parsed_pose
    if parsed_pose is not None and camera_pose_for_drawing is None:
        limitations.append(
            "A Camera pose record exists, but critical calibration/pose/frame/FOV "
            "validation failed; the map renderer therefore suppresses FOV and rays."
        )

    all_reason_codes = {
        str(reason)
        for target in target_reports
        for reason in target["assessment"].get("reason_codes", ())
    }
    static_obstacles_for_drawing = (
        []
        if all_reason_codes & _STATIC_DISPLAY_INVALID_REASONS
        else _static_obstacles(camera_scene)
    )
    if (
        camera_scene is not None
        and not static_obstacles_for_drawing
        and all_reason_codes & _STATIC_DISPLAY_INVALID_REASONS
    ):
        limitations.append(
            "Static obstacle polygons are suppressed because their local "
            "frame/completeness validation did not pass."
        )

    overlay = {
        "schema_version": CAMERA_MAP_OVERLAY_SCHEMA_VERSION,
        "algorithm": "parking_slot_part2.camera_observability.assess_camera_observability",
        "algorithm_executed": True,
        "semantic_camera_model_called": False,
        "policy_version": CAMERA_OBSERVABILITY_POLICY_VERSION,
        "camera_pose_map_xyyaw": (
            None if camera_pose_for_drawing is None else list(camera_pose_for_drawing)
        ),
        "fov_zones_deg": {
            "nominal_half": settings.nominal_half_fov_deg,
            "reliable_half": settings.reliable_half_fov_deg,
            "edge_unreliable_start": settings.edge_unreliable_start_deg,
        },
        "fov_radius_m": float(
            local_map.get("lidar_coverage", {}).get("nominal_radius_m", 18.0)
        ),
        "static_obstacles": static_obstacles_for_drawing,
        "occlusion_objects": _occlusion_objects(scene_slots, target_reports),
        "targets": target_reports,
        "input_limitations": list(dict.fromkeys(limitations)),
    }
    return {
        "schema_version": CAMERA_MAP_REPORT_SCHEMA_VERSION,
        "algorithm": overlay["algorithm"],
        "algorithm_executed": True,
        "semantic_camera_model_called": False,
        "camera_scene_supplied": camera_scene is not None,
        "camera_geometry_drawn": camera_pose_for_drawing is not None,
        "targets": target_reports,
        "input_limitations": overlay["input_limitations"],
        "overlay": overlay,
        "notes": [
            "Assessments and debug rays come from the existing Part2 Camera gate.",
            "No Camera semantic detection model was called.",
            "Missing real upstream data remains fail-closed and is never filled from ego pose.",
            "The top-down FOV wedge radius is a display aid; angular zones come from the active policy.",
        ],
    }


__all__ = [
    "CAMERA_MAP_OVERLAY_SCHEMA_VERSION",
    "CAMERA_MAP_REPORT_SCHEMA_VERSION",
    "build_camera_observability_map_report",
]
