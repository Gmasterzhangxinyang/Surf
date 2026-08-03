from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
import textwrap
from typing import Any

import matplotlib

# Part1 rendering is also used by headless batch jobs and CI.
matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Patch, Polygon as MplPolygon, Wedge


_LOCAL_STATES = frozenset({"free", "occupied", "unknown"})
_STATE_STYLE = {
    "free": ("#22c55e", "#166534"),
    "occupied": ("#ef4444", "#991b1b"),
    "unknown": ("#f59e0b", "#92400e"),
}
_CAMERA_RAY_STYLE = {
    "clear": ("#16a34a", 1.4),
    "blocked": ("#dc2626", 2.0),
    "uncertain": ("#f59e0b", 1.8),
    "edge_unreliable": ("#9333ea", 1.3),
    "outside_fov": ("#64748b", 1.0),
}
_CAMERA_DECISIONS = frozenset(
    {"use_camera", "do_not_use_camera", "insufficient_information"}
)


def _finite_xy(value: Any, name: str, *, minimum_rows: int = 1) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if (
        array.ndim != 2
        or array.shape[1] != 2
        or len(array) < minimum_rows
        or not np.isfinite(array).all()
    ):
        raise ValueError(f"{name} must be a finite [N,2] array")
    return array


def _slot_id(slot: Any) -> str:
    if isinstance(slot, Mapping):
        value = slot.get("slot_id")
    else:
        value = getattr(slot, "slot_id", None)
    if value is None or not str(value):
        raise ValueError("every slot must have a non-empty slot_id")
    return str(value)


def _slot_polygon(slot: Any) -> np.ndarray:
    if isinstance(slot, Mapping):
        value = slot.get("polygon_map")
    else:
        value = getattr(slot, "polygon_map", None)
    return _finite_xy(value, f"slot {_slot_id(slot)} polygon_map", minimum_rows=3)


def _all_slot_items(all_slots: Any) -> tuple[Any, ...]:
    if all_slots is None:
        return ()
    if isinstance(all_slots, Mapping):
        nested = all_slots.get("slots")
        if isinstance(nested, Sequence) and not isinstance(nested, (str, bytes)):
            return tuple(nested)
        return tuple(all_slots[key] for key in sorted(all_slots, key=str))
    if isinstance(all_slots, Iterable) and not isinstance(all_slots, (str, bytes)):
        return tuple(all_slots)
    raise TypeError("all_slots must be a slot iterable, slot mapping, or None")


def _relative_m(
    points_map: np.ndarray,
    anchor_map: np.ndarray,
    map_units_per_meter: float,
) -> np.ndarray:
    return (np.asarray(points_map, dtype=np.float64) - anchor_map) / map_units_per_meter


def _bbox_intersects(
    polygon: np.ndarray,
    x_limits: tuple[float, float],
    y_limits: tuple[float, float],
) -> bool:
    minimum = np.min(polygon, axis=0)
    maximum = np.max(polygon, axis=0)
    return bool(
        maximum[0] >= x_limits[0]
        and minimum[0] <= x_limits[1]
        and maximum[1] >= y_limits[0]
        and minimum[1] <= y_limits[1]
    )


def _viewport_limits(
    extent_points: Sequence[np.ndarray],
    nominal_radius_m: float,
) -> tuple[tuple[float, float], tuple[float, float]]:
    points = np.vstack([np.asarray(value, dtype=np.float64) for value in extent_points])
    minimum = np.min(points, axis=0)
    maximum = np.max(points, axis=0)
    minimum = np.minimum(minimum, np.asarray([0.0, 0.0]))
    maximum = np.maximum(maximum, np.asarray([0.0, 0.0]))

    # An actual endpoint hull defines the view when present.  The minimum
    # span keeps a tiny/degenerate window readable without ever zooming out
    # to the complete parking-lot database.
    minimum_half_span = min(max(nominal_radius_m * 0.30, 4.0), nominal_radius_m)
    center = (minimum + maximum) * 0.5
    half = np.maximum((maximum - minimum) * 0.5, minimum_half_span)
    padding = np.maximum(half * 0.12, 0.75)
    half += padding
    return (
        (float(center[0] - half[0]), float(center[0] + half[0])),
        (float(center[1] - half[1]), float(center[1] + half[1])),
    )


def _normalise_camera_overlay(
    value: Mapping[str, Any] | None,
    local_ids: set[str],
) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise TypeError("camera_overlay must be a mapping or None")
    if value.get("algorithm_executed") is not True:
        raise ValueError("camera_overlay must come from an executed Camera gate")
    if value.get("semantic_camera_model_called") is not False:
        raise ValueError("Camera observability overlay must not call a semantic model")

    raw_targets = value.get("targets", ())
    if not isinstance(raw_targets, Sequence) or isinstance(raw_targets, (str, bytes)):
        raise ValueError("camera_overlay.targets must be a sequence")
    if len(raw_targets) > 2:
        raise ValueError("camera_overlay may contain at most two targets")
    targets: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw in enumerate(raw_targets):
        if not isinstance(raw, Mapping):
            raise ValueError(f"camera_overlay target {index} must be a mapping")
        target_id = str(raw.get("target_slot_id", "")).strip()
        if not target_id or target_id not in local_ids:
            raise ValueError("Camera target must reference a local snapshot slot")
        if target_id in seen:
            raise ValueError(f"duplicate Camera target: {target_id}")
        assessment = raw.get("assessment")
        if not isinstance(assessment, Mapping):
            raise ValueError("Camera target assessment must be a mapping")
        if str(assessment.get("target_slot_id", "")) != target_id:
            raise ValueError("Camera assessment target ID does not match overlay target")
        decision = str(assessment.get("decision", ""))
        if decision not in _CAMERA_DECISIONS:
            raise ValueError("unsupported Camera-use decision in overlay")
        if bool(assessment.get("camera_usable", False)) != (decision == "use_camera"):
            raise ValueError("camera_usable disagrees with Camera-use decision")
        for name in (
            "nominal_fov_coverage",
            "reliable_fov_coverage",
            "edge_quality_score",
            "clear_ray_ratio",
            "blocked_ray_ratio",
            "uncertain_ray_ratio",
        ):
            metric = float(assessment.get(name, 0.0))
            if not np.isfinite(metric) or not 0.0 <= metric <= 1.0:
                raise ValueError(f"Camera assessment {name} must be within [0,1]")
        raw_trace = raw.get("debug_trace", ())
        if not isinstance(raw_trace, Sequence) or isinstance(raw_trace, (str, bytes)):
            raise ValueError("Camera debug_trace must be a sequence")
        if not all(isinstance(item, Mapping) for item in raw_trace):
            raise ValueError("Camera debug_trace must contain mappings")
        label = str(raw.get("display_label", chr(ord("A") + index))).strip()
        assessment_data = dict(assessment)
        for name in (
            "blocking_object_ids",
            "potential_occluder_ids",
            "reason_codes",
        ):
            raw_items = assessment_data.get(name, ())
            if not isinstance(raw_items, Sequence) or isinstance(
                raw_items, (str, bytes)
            ):
                raise ValueError(f"Camera assessment {name} must be a sequence")
            assessment_data[name] = tuple(
                str(item) for item in raw_items if str(item).strip()
            )
        targets.append(
            {
                "target_slot_id": target_id,
                "display_label": label or chr(ord("A") + index),
                "assessment": assessment_data,
                "debug_trace": tuple(raw_trace),
            }
        )
        seen.add(target_id)

    raw_pose = value.get("camera_pose_map_xyyaw")
    camera_pose: tuple[float, float, float] | None = None
    if raw_pose is not None:
        pose = np.asarray(raw_pose, dtype=np.float64)
        if pose.shape != (3,) or not np.isfinite(pose).all():
            raise ValueError("camera_pose_map_xyyaw must contain finite x, y, yaw")
        camera_pose = tuple(float(item) for item in pose)

    raw_zones = value.get("fov_zones_deg", {})
    if not isinstance(raw_zones, Mapping):
        raise ValueError("camera_overlay.fov_zones_deg must be a mapping")
    nominal = float(raw_zones.get("nominal_half", 90.0))
    reliable = float(raw_zones.get("reliable_half", 60.0))
    edge = float(raw_zones.get("edge_unreliable_start", 78.0))
    if not all(np.isfinite(item) for item in (nominal, reliable, edge)) or not (
        0.0 < reliable < edge <= nominal <= 180.0
    ):
        raise ValueError("invalid Camera FOV zones")

    raw_obstacles = value.get("static_obstacles", ())
    if not isinstance(raw_obstacles, Sequence) or isinstance(
        raw_obstacles, (str, bytes)
    ):
        raise ValueError("camera_overlay.static_obstacles must be a sequence")
    if not all(isinstance(item, Mapping) for item in raw_obstacles):
        raise ValueError("camera_overlay.static_obstacles must contain mappings")
    raw_occlusion_objects = value.get("occlusion_objects", ())
    if not isinstance(raw_occlusion_objects, Sequence) or isinstance(
        raw_occlusion_objects, (str, bytes)
    ):
        raise ValueError("camera_overlay.occlusion_objects must be a sequence")
    if not all(isinstance(item, Mapping) for item in raw_occlusion_objects):
        raise ValueError("camera_overlay.occlusion_objects must contain mappings")
    raw_limitations = value.get("input_limitations", ())
    if raw_limitations is None:
        raw_limitations = ()
    if not isinstance(raw_limitations, Sequence) or isinstance(
        raw_limitations, (str, bytes)
    ):
        raise ValueError("camera_overlay.input_limitations must be a sequence")
    radius = float(value.get("fov_radius_m", 18.0))
    if not np.isfinite(radius) or radius <= 0.0:
        raise ValueError("camera_overlay.fov_radius_m must be positive")
    return {
        "algorithm": str(value.get("algorithm", "assess_camera_observability")),
        "camera_pose_map_xyyaw": camera_pose,
        "targets": tuple(targets),
        "fov_zones_deg": {
            "nominal_half": nominal,
            "reliable_half": reliable,
            "edge_unreliable_start": edge,
        },
        "fov_radius_m": radius,
        "static_obstacles": tuple(raw_obstacles),
        "occlusion_objects": tuple(raw_occlusion_objects),
        "input_limitations": tuple(str(item) for item in raw_limitations),
    }


def render_local_map(
    snapshot: Mapping[str, Any],
    all_slots: Any,
    output_path: str | Path,
    *,
    camera_overlay: Mapping[str, Any] | None = None,
) -> Path:
    """Render the bounded, causal Part1 LiDAR map.

    ``snapshot["slots"]`` is the only source of occupancy state.  ``all_slots``
    is optional structural context: polygons inside the already-selected local
    viewport are drawn as faint, state-free outlines.  It never affects the
    viewport, legend counts, local state, or provisional-candidate display.

    The function intentionally renders in ego-relative metres so a large
    global map coordinate offset cannot make this local result look global.
    When ``camera_overlay`` is supplied, its assessments and ray traces must
    come from Part2's Camera pre-call gate; the renderer never recomputes a
    second visibility decision and never calls a Camera semantic model.
    """

    if not isinstance(snapshot, Mapping):
        raise TypeError("snapshot must be a mapping")
    scale = float(snapshot.get("map_units_per_meter", 1.0))
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("map_units_per_meter must be positive and finite")

    anchor_record = snapshot.get("anchor_pose")
    if not isinstance(anchor_record, Mapping):
        raise ValueError("snapshot anchor_pose is required")
    anchor_map = np.asarray(anchor_record.get("map_xy"), dtype=np.float64)
    if anchor_map.shape != (2,) or not np.isfinite(anchor_map).all():
        raise ValueError("anchor_pose.map_xy must be a finite XY coordinate")
    anchor_yaw = float(anchor_record.get("map_yaw_rad", 0.0))
    if not np.isfinite(anchor_yaw):
        raise ValueError("anchor_pose.map_yaw_rad must be finite")

    raw_local_slots = snapshot.get("slots", ())
    if not isinstance(raw_local_slots, Sequence) or isinstance(
        raw_local_slots, (str, bytes)
    ):
        raise ValueError("snapshot slots must be a sequence")
    local_slots: list[tuple[str, str, np.ndarray, Mapping[str, Any]]] = []
    local_ids: set[str] = set()
    for index, raw in enumerate(raw_local_slots):
        if not isinstance(raw, Mapping):
            raise ValueError(f"snapshot slot {index} must be a mapping")
        slot_id = _slot_id(raw)
        if slot_id in local_ids:
            raise ValueError(f"duplicate local slot_id: {slot_id}")
        state = str(raw.get("state", ""))
        if state not in _LOCAL_STATES:
            raise ValueError(
                f"local slot {slot_id} has unsupported state {state!r}; "
                "only free, occupied, and unknown are renderable"
            )
        polygon = _relative_m(_slot_polygon(raw), anchor_map, scale)
        local_slots.append((slot_id, state, polygon, raw))
        local_ids.add(slot_id)

    local_by_id = {item[0]: item for item in local_slots}
    provisional_candidates: list[tuple[str, str, str]] = []
    missing = object()
    raw_provisional_candidates = snapshot.get("provisional_candidates", missing)
    if raw_provisional_candidates is not missing:
        # Presence of the new field, including an intentionally empty list,
        # disables the legacy fallback. This prevents candidate_slot_id from
        # silently turning a Part1 visualization shortlist into a selection.
        if not isinstance(raw_provisional_candidates, Sequence) or isinstance(
            raw_provisional_candidates, (str, bytes)
        ):
            raise ValueError("provisional_candidates must be a sequence")
        if len(raw_provisional_candidates) > 2:
            raise ValueError("provisional_candidates may contain at most two entries")
        seen_candidate_ids: set[str] = set()
        for index, raw_candidate in enumerate(raw_provisional_candidates):
            if not isinstance(raw_candidate, Mapping):
                raise ValueError(f"provisional candidate {index} must be a mapping")
            candidate_id = _slot_id(raw_candidate)
            if candidate_id in seen_candidate_ids:
                raise ValueError(f"duplicate provisional candidate: {candidate_id}")
            if candidate_id not in local_by_id:
                raise ValueError(
                    "provisional candidate must reference a local snapshot slot: "
                    f"{candidate_id}"
                )
            candidate_state = str(raw_candidate.get("state", ""))
            if candidate_state not in {"free", "unknown"}:
                raise ValueError(
                    "provisional candidate state must be only free or unknown: "
                    f"{candidate_id}"
                )
            local_state = local_by_id[candidate_id][1]
            if candidate_state != local_state:
                raise ValueError(
                    "provisional candidate state does not match local slot state: "
                    f"{candidate_id}"
                )
            provisional_candidates.append(
                (candidate_id, candidate_state, chr(ord("A") + index))
            )
            seen_candidate_ids.add(candidate_id)
    else:
        # Compatibility for snapshots written before provisional_candidates.
        # P1 is still rendered as provisional; it is never described as a
        # final/selected Part1 parking decision.
        legacy_value = snapshot.get("candidate_slot_id")
        legacy_id = None if legacy_value in (None, "") else str(legacy_value)
        if legacy_id is not None:
            if legacy_id not in local_by_id:
                raise ValueError(
                    "legacy candidate_slot_id must reference a local snapshot slot"
                )
            legacy_state = local_by_id[legacy_id][1]
            if legacy_state != "free":
                raise ValueError(
                    "legacy candidate_slot_id must reference a local free slot"
                )
            provisional_candidates.append((legacy_id, legacy_state, "P1"))

    camera = _normalise_camera_overlay(camera_overlay, local_ids)
    camera_targets_by_id = (
        {}
        if camera is None
        else {
            item["target_slot_id"]: item
            for item in camera["targets"]
        }
    )
    blocking_object_ids = (
        set()
        if camera is None
        else {
            str(object_id)
            for target in camera["targets"]
            for object_id in target["assessment"].get("blocking_object_ids", ())
        }
    )
    potential_occluder_ids = (
        set()
        if camera is None
        else {
            str(object_id)
            for target in camera["targets"]
            for object_id in target["assessment"].get("potential_occluder_ids", ())
        }
        - blocking_object_ids
    )

    coverage = snapshot.get("lidar_coverage", {})
    if not isinstance(coverage, Mapping):
        raise ValueError("lidar_coverage must be a mapping")
    nominal_radius_m = float(coverage.get("nominal_radius_m", 18.0))
    if not np.isfinite(nominal_radius_m) or nominal_radius_m <= 0.0:
        raise ValueError("lidar_coverage.nominal_radius_m must be positive and finite")

    raw_pose_centers = coverage.get("pose_centers_map")
    if raw_pose_centers is None:
        raw_pose_centers = [anchor_map.tolist()]
    pose_centers_map = _finite_xy(raw_pose_centers, "pose_centers_map")
    trajectory_m = _relative_m(pose_centers_map, anchor_map, scale)

    raw_coverage_polygon = coverage.get("polygon_map")
    coverage_polygon_m: np.ndarray | None = None
    if raw_coverage_polygon is not None:
        coverage_polygon_m = _relative_m(
            _finite_xy(raw_coverage_polygon, "lidar coverage polygon", minimum_rows=3),
            anchor_map,
            scale,
        )

    extent_points: list[np.ndarray] = [trajectory_m, np.asarray([[0.0, 0.0]])]
    if coverage_polygon_m is not None:
        extent_points.append(coverage_polygon_m)
    else:
        extent_points.append(
            np.asarray(
                [
                    [-nominal_radius_m, -nominal_radius_m],
                    [nominal_radius_m, nominal_radius_m],
                ]
            )
        )
    extent_points.extend(item[2] for item in local_slots)
    camera_pose_m: tuple[float, float, float] | None = None
    camera_static_polygons: list[tuple[str, str, np.ndarray]] = []
    camera_occlusion_polygons: list[tuple[str, str, np.ndarray]] = []
    if camera is not None:
        raw_camera_pose = camera["camera_pose_map_xyyaw"]
        if raw_camera_pose is not None:
            camera_xy_m = _relative_m(
                np.asarray([raw_camera_pose[:2]], dtype=np.float64),
                anchor_map,
                scale,
            )[0]
            camera_pose_m = (
                float(camera_xy_m[0]),
                float(camera_xy_m[1]),
                float(raw_camera_pose[2]),
            )
            extent_points.append(camera_xy_m.reshape(1, 2))
        for index, raw_obstacle in enumerate(camera["static_obstacles"]):
            raw_polygon = raw_obstacle.get(
                "polygon_xy",
                raw_obstacle.get("polygon_map"),
            )
            polygon = _relative_m(
                _finite_xy(
                    raw_polygon,
                    f"camera static obstacle {index} polygon",
                    minimum_rows=3,
                ),
                anchor_map,
                scale,
            )
            obstacle_id = str(
                raw_obstacle.get("id", raw_obstacle.get("object_id", f"static_{index}"))
            )
            obstacle_type = str(raw_obstacle.get("type", "other_static"))
            camera_static_polygons.append((obstacle_id, obstacle_type, polygon))
        for index, raw_object in enumerate(camera["occlusion_objects"]):
            raw_polygon = raw_object.get(
                "occupancy_polygon_map",
                raw_object.get(
                    "vehicle_polygon_map",
                    raw_object.get(
                        "obstacle_polygon_map",
                        raw_object.get("polygon_map", raw_object.get("polygon_xy")),
                    ),
                ),
            )
            polygon = _relative_m(
                _finite_xy(
                    raw_polygon,
                    f"camera occlusion object {index} polygon",
                    minimum_rows=3,
                ),
                anchor_map,
                scale,
            )
            object_id = str(
                raw_object.get(
                    "object_id",
                    raw_object.get("slot_id", raw_object.get("id", f"occluder_{index}")),
                )
            )
            object_type = str(
                raw_object.get("object_type", raw_object.get("type", "object"))
            )
            camera_occlusion_polygons.append((object_id, object_type, polygon))
    x_limits, y_limits = _viewport_limits(extent_points, nominal_radius_m)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    rc = {
        "font.family": "DejaVu Sans",
        "font.size": 9.0,
        "axes.titlesize": 14.0,
        "axes.labelsize": 9.0,
        "figure.facecolor": "#f8fafc",
        "axes.facecolor": "#f1f5f9",
        "savefig.facecolor": "#f8fafc",
        "savefig.dpi": 160,
        "path.simplify": False,
    }
    with plt.rc_context(rc):
        figure, axis = plt.subplots(
            figsize=((14.2, 8.0) if camera is not None else (10.5, 8.0)),
            dpi=160,
        )
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlim(*x_limits)
        axis.set_ylim(*y_limits)

        if coverage_polygon_m is not None:
            axis.add_patch(
                MplPolygon(
                    coverage_polygon_m,
                    closed=True,
                    facecolor="#38bdf8",
                    edgecolor="#0284c7",
                    linewidth=1.8,
                    alpha=0.16,
                    linestyle="-",
                    zorder=1,
                )
            )
            coverage_label = "actual LiDAR endpoint footprint"
        else:
            # No endpoint hull is available.  Dashed range discs visibly mark
            # this as a nominal envelope, never as confirmed free space.
            for center in trajectory_m:
                axis.add_patch(
                    Circle(
                        center,
                        nominal_radius_m,
                        facecolor="#38bdf8",
                        edgecolor="#0284c7",
                        linewidth=0.8,
                        alpha=0.035,
                        linestyle="--",
                        zorder=1,
                    )
                )
            coverage_label = "nominal LiDAR envelope (endpoint hull unavailable)"

        # Structural context is deliberately clipped to the local viewport.
        # These polygons are never colored, labelled, counted, or assigned a
        # state from any global database.
        context_count = 0
        for raw in sorted(_all_slot_items(all_slots), key=_slot_id):
            slot_id = _slot_id(raw)
            if slot_id in local_ids:
                continue
            polygon_m = _relative_m(_slot_polygon(raw), anchor_map, scale)
            if not _bbox_intersects(polygon_m, x_limits, y_limits):
                continue
            axis.add_patch(
                MplPolygon(
                    polygon_m,
                    closed=True,
                    facecolor="none",
                    edgecolor="#94a3b8",
                    linewidth=0.55,
                    alpha=0.35,
                    zorder=2,
                )
            )
            context_count += 1

        if camera_pose_m is not None and camera is not None:
            camera_x, camera_y, camera_yaw = camera_pose_m
            zones = camera["fov_zones_deg"]
            nominal_half = float(zones["nominal_half"])
            reliable_half = float(zones["reliable_half"])
            edge_start = float(zones["edge_unreliable_start"])
            yaw_deg = float(np.degrees(camera_yaw))
            fov_radius = min(float(camera["fov_radius_m"]), nominal_radius_m * 1.10)
            for theta1, theta2, color, alpha in (
                (yaw_deg - nominal_half, yaw_deg - edge_start, "#ef4444", 0.13),
                (yaw_deg - edge_start, yaw_deg - reliable_half, "#f59e0b", 0.12),
                (yaw_deg - reliable_half, yaw_deg + reliable_half, "#22c55e", 0.075),
                (yaw_deg + reliable_half, yaw_deg + edge_start, "#f59e0b", 0.12),
                (yaw_deg + edge_start, yaw_deg + nominal_half, "#ef4444", 0.13),
            ):
                axis.add_patch(
                    Wedge(
                        (camera_x, camera_y),
                        fov_radius,
                        theta1,
                        theta2,
                        facecolor=color,
                        edgecolor=color,
                        linewidth=0.8,
                        alpha=alpha,
                        zorder=2.6,
                    )
                )
            for boundary, color in (
                (-nominal_half, "#dc2626"),
                (-edge_start, "#d97706"),
                (-reliable_half, "#16a34a"),
                (reliable_half, "#16a34a"),
                (edge_start, "#d97706"),
                (nominal_half, "#dc2626"),
            ):
                angle = camera_yaw + np.radians(boundary)
                axis.plot(
                    [camera_x, camera_x + fov_radius * np.cos(angle)],
                    [camera_y, camera_y + fov_radius * np.sin(angle)],
                    color=color,
                    linewidth=0.7,
                    linestyle="--",
                    alpha=0.65,
                    zorder=2.8,
                )
            axis.scatter(
                [camera_x],
                [camera_y],
                marker="D",
                s=72,
                c="#a21caf",
                edgecolors="white",
                linewidths=1.1,
                zorder=11,
            )
            camera_heading_length = max(1.2, min(2.5, fov_radius * 0.15))
            axis.arrow(
                camera_x,
                camera_y,
                np.cos(camera_yaw) * camera_heading_length,
                np.sin(camera_yaw) * camera_heading_length,
                width=0.07,
                head_width=0.42,
                head_length=0.52,
                length_includes_head=True,
                color="#a21caf",
                zorder=11,
            )
            pose_is_proxy = any(
                "proxy" in str(item).lower()
                for item in camera.get("input_limitations", ())
            )
            axis.text(
                camera_x,
                camera_y + 0.48,
                "CAMERA POSE (EGO PROXY)" if pose_is_proxy else "CAMERA OPTICAL POSE",
                ha="center",
                va="bottom",
                fontsize=6.8,
                fontweight="bold",
                color="#86198f",
                zorder=11,
            )

        for obstacle_id, obstacle_type, polygon_m in camera_static_polygons:
            if not _bbox_intersects(polygon_m, x_limits, y_limits):
                continue
            is_wall = obstacle_type.lower() == "wall"
            face_color = "#475569" if is_wall else "#a16207"
            axis.add_patch(
                MplPolygon(
                    polygon_m,
                    closed=True,
                    facecolor=face_color,
                    edgecolor="#0f172a",
                    linewidth=1.4,
                    alpha=0.58,
                    zorder=3.5,
                )
            )
            center = np.mean(polygon_m, axis=0)
            axis.text(
                center[0],
                center[1],
                f"{obstacle_id}\n{obstacle_type}",
                ha="center",
                va="center",
                fontsize=5.7,
                color="white",
                zorder=3.7,
            )

        state_counts = {state: 0 for state in sorted(_LOCAL_STATES)}
        for slot_id, state, polygon_m, _ in sorted(local_slots, key=lambda item: item[0]):
            face, edge = _STATE_STYLE[state]
            axis.add_patch(
                MplPolygon(
                    polygon_m,
                    closed=True,
                    facecolor=face,
                    edgecolor=edge,
                    linewidth=1.2,
                    alpha=0.56,
                    zorder=4,
                )
            )
            center = np.mean(polygon_m, axis=0)
            axis.text(
                center[0],
                center[1],
                f"{slot_id}\n{state}",
                ha="center",
                va="center",
                fontsize=6.4,
                color="#0f172a",
                clip_on=True,
                zorder=5,
            )
            state_counts[state] += 1

        # Highlight only object IDs returned by the executed Camera gate.  An
        # explicit occupancy/vehicle polygon wins over a slot footprint; the
        # latter remains the conservative fallback used by the gate itself.
        occluder_geometry: dict[str, tuple[str, np.ndarray]] = {
            slot_id: ("slot footprint", polygon_m)
            for slot_id, _, polygon_m, _ in local_slots
        }
        occluder_geometry.update(
            {
                obstacle_id: (obstacle_type, polygon_m)
                for obstacle_id, obstacle_type, polygon_m in camera_static_polygons
            }
        )
        occluder_geometry.update(
            {
                object_id: (object_type, polygon_m)
                for object_id, object_type, polygon_m in camera_occlusion_polygons
            }
        )
        highlight_rows = [
            (item, "blocker", "#b91c1c", "BLOCKER")
            for item in sorted(blocking_object_ids)
        ]
        highlight_rows.extend(
            (item, "potential", "#d97706", "POTENTIAL OCCLUDER")
            for item in sorted(potential_occluder_ids)
        )
        for object_id, kind, color, label in highlight_rows:
            geometry = occluder_geometry.get(object_id)
            if geometry is None:
                continue
            object_type, polygon_m = geometry
            if not _bbox_intersects(polygon_m, x_limits, y_limits):
                continue
            axis.add_patch(
                MplPolygon(
                    polygon_m,
                    closed=True,
                    facecolor=color if object_id not in local_by_id else "none",
                    edgecolor=color,
                    linewidth=3.0 if kind == "blocker" else 2.5,
                    alpha=0.82,
                    hatch=None if object_id in local_by_id else "///",
                    zorder=6.1,
                )
            )
            center = np.mean(polygon_m, axis=0)
            axis.annotate(
                f"{label}: {object_id}\n{object_type}",
                xy=center,
                xytext=(6, -19),
                textcoords="offset points",
                fontsize=6.3,
                fontweight="bold",
                color=color,
                bbox={
                    "boxstyle": "round,pad=0.2",
                    "facecolor": "white",
                    "edgecolor": color,
                    "alpha": 0.92,
                },
                zorder=8.2,
            )

        if camera_pose_m is not None and camera is not None:
            camera_xy = np.asarray(camera_pose_m[:2], dtype=np.float64)
            rendered_rays: set[tuple[str, float, float, str, str]] = set()
            for target in camera["targets"]:
                target_id = str(target["target_slot_id"])
                for raw_ray in target["debug_trace"]:
                    raw_sample = raw_ray.get("sample_map_xy")
                    if (
                        not isinstance(raw_sample, Sequence)
                        or isinstance(raw_sample, (str, bytes))
                        or len(raw_sample) != 2
                    ):
                        continue
                    sample_map = np.asarray(raw_sample, dtype=np.float64)
                    if sample_map.shape != (2,) or not np.isfinite(sample_map).all():
                        continue
                    status = str(raw_ray.get("status", "outside_fov"))
                    if status not in _CAMERA_RAY_STYLE:
                        continue
                    object_id = str(raw_ray.get("object_id") or "")
                    key = (
                        target_id,
                        round(float(sample_map[0]), 8),
                        round(float(sample_map[1]), 8),
                        status,
                        object_id,
                    )
                    if key in rendered_rays:
                        continue
                    rendered_rays.add(key)
                    sample_m = _relative_m(
                        sample_map.reshape(1, 2),
                        anchor_map,
                        scale,
                    )[0]
                    color, linewidth = _CAMERA_RAY_STYLE[status]
                    axis.plot(
                        [camera_xy[0], sample_m[0]],
                        [camera_xy[1], sample_m[1]],
                        color=color,
                        linewidth=linewidth,
                        alpha=0.82,
                        zorder=6.6,
                    )
                    axis.scatter(
                        [sample_m[0]],
                        [sample_m[1]],
                        s=18 if status in {"blocked", "uncertain"} else 11,
                        facecolors="white",
                        edgecolors=color,
                        linewidths=0.9,
                        zorder=6.8,
                    )

        for provisional_id, _, display_label in provisional_candidates:
            _, _, candidate_polygon, _ = local_by_id[provisional_id]
            axis.add_patch(
                MplPolygon(
                    candidate_polygon,
                    closed=True,
                    facecolor="none",
                    edgecolor="#7c3aed",
                    linewidth=3.0,
                    linestyle=(0, (4, 2)),
                    alpha=0.95,
                    zorder=7,
                )
            )
            center = np.mean(candidate_polygon, axis=0)
            camera_target = camera_targets_by_id.get(provisional_id)
            camera_suffix = ""
            annotation_edge = "#7c3aed"
            if camera_target is not None:
                decision = str(camera_target["assessment"]["decision"])
                camera_suffix = "\nCAMERA: " + decision.upper()
                annotation_edge = {
                    "use_camera": "#15803d",
                    "do_not_use_camera": "#b91c1c",
                    "insufficient_information": "#b45309",
                }[decision]
            axis.annotate(
                f"{display_label}  PROVISIONAL{camera_suffix}",
                xy=center,
                xytext=(8, 14),
                textcoords="offset points",
                fontsize=7.4,
                fontweight="bold",
                color="#5b21b6",
                bbox={
                    "boxstyle": "round,pad=0.25",
                    "facecolor": "white",
                    "edgecolor": annotation_edge,
                    "alpha": 0.92,
                },
                zorder=9,
            )

        if len(trajectory_m) > 1:
            axis.plot(
                trajectory_m[:, 0],
                trajectory_m[:, 1],
                color="#0f172a",
                linewidth=1.7,
                marker="o",
                markersize=3.0,
                alpha=0.90,
                zorder=6,
            )
        else:
            axis.scatter(
                trajectory_m[:, 0],
                trajectory_m[:, 1],
                c="#0f172a",
                s=16,
                zorder=6,
            )

        arrow_length = max(1.5, min(3.0, nominal_radius_m * 0.16))
        heading = np.asarray(
            [np.cos(anchor_yaw), np.sin(anchor_yaw)], dtype=np.float64
        )
        axis.scatter(
            [0.0],
            [0.0],
            marker="o",
            s=90,
            c="#2563eb",
            edgecolors="white",
            linewidths=1.2,
            zorder=10,
        )
        axis.arrow(
            0.0,
            0.0,
            heading[0] * arrow_length,
            heading[1] * arrow_length,
            width=0.10,
            head_width=0.55,
            head_length=0.65,
            length_includes_head=True,
            color="#1d4ed8",
            zorder=10,
        )
        axis.text(
            0.0,
            -0.55,
            "EGO / ANCHOR",
            ha="center",
            va="top",
            fontsize=7.2,
            fontweight="bold",
            color="#1e3a8a",
            zorder=10,
        )

        lidar_window = snapshot.get("lidar_window", {})
        if not isinstance(lidar_window, Mapping):
            lidar_window = {}
        frame_count = int(lidar_window.get("frame_count", len(trajectory_m)))
        subtitle = (
            f"{frame_count} consecutive LiDAR frames | "
            f"local slots: {len(local_slots)} "
            f"(free {state_counts['free']}, occupied {state_counts['occupied']}, "
            f"unknown {state_counts['unknown']})"
        )
        if camera is None:
            title = (
                "PART1 ONLY — LOCAL LiDAR MAP (INCOMPLETE / NOT GLOBAL TRUTH)\n"
                "VISUALIZATION ONLY NOT FINAL — "
                "Part2 Camera observability/FOV/occlusion NOT INCLUDED\n"
                + subtitle
            )
        else:
            title = (
                "PART1 LOCAL LiDAR MAP + PART2 CAMERA PRE-CALL GATE\n"
                "EXACT ASSESSMENT + DEBUG RAYS · NO CAMERA SEMANTIC MODEL\n"
                + subtitle
            )
        axis.set_title(
            title,
            color="#0f172a",
            fontweight="bold",
            pad=11,
        )
        map_note = (
            "Outside the shaded footprint is unobserved by this short window.\n"
            "Faint outlines are geometry only — no occupancy state is implied.\n"
            "Part1 provisional markers are display hints, not parking decisions."
        )
        if camera is not None:
            map_note += (
                "\nCamera rays are the exact Part2 debug trace; "
                "invalid Camera geometry means no FOV/rays are drawn.\n"
                "FOV wedge radius is display-only; its angles come from the active policy."
            )
        axis.text(
            0.012,
            0.018,
            map_note,
            transform=axis.transAxes,
            ha="left",
            va="bottom",
            fontsize=7.4,
            color="#475569",
            bbox={
                "boxstyle": "round,pad=0.35",
                "facecolor": "#ffffff",
                "edgecolor": "#cbd5e1",
                "alpha": 0.92,
            },
            zorder=20,
        )

        if camera is not None:
            trace_record_count = sum(
                len(target["debug_trace"]) for target in camera["targets"]
            )
            panel_lines = [
                "PART2 CAMERA PRE-CALL GATE",
                "algorithm executed: YES",
                "semantic detector called: NO",
                (
                    (
                        "map FOV: DRAWN from t0 ego-pose proxy"
                        if any(
                            "proxy" in str(item).lower()
                            for item in camera.get("input_limitations", ())
                        )
                        else "map FOV: DRAWN from validated Camera pose"
                    )
                    if camera_pose_m is not None
                    else "map FOV: NOT DRAWN — geometry validation failed"
                ),
                f"LOS trace records: {trace_record_count}",
                "",
            ]
            for target in camera["targets"]:
                assessment = target["assessment"]
                label = target["display_label"]
                decision = str(assessment["decision"])
                bearing = assessment.get("target_bearing_deg")
                distance = assessment.get("target_distance_m")
                panel_lines.extend(
                    [
                        f"{label}  {target['target_slot_id']}",
                        f"decision: {decision}",
                        f"camera_usable: {str(bool(assessment.get('camera_usable'))).lower()}",
                        (
                            "bearing / distance: unavailable"
                            if bearing is None or distance is None
                            else f"bearing / distance: {float(bearing):.1f} deg / {float(distance):.2f} m"
                        ),
                        (
                            "FOV nominal/reliable/quality: "
                            f"{float(assessment.get('nominal_fov_coverage', 0.0)):.2f} / "
                            f"{float(assessment.get('reliable_fov_coverage', 0.0)):.2f} / "
                            f"{float(assessment.get('edge_quality_score', 0.0)):.2f}"
                        ),
                        (
                            "LOS clear/blocked/uncertain: "
                            f"{float(assessment.get('clear_ray_ratio', 0.0)):.2f} / "
                            f"{float(assessment.get('blocked_ray_ratio', 0.0)):.2f} / "
                            f"{float(assessment.get('uncertain_ray_ratio', 0.0)):.2f}"
                        ),
                        "blockers: "
                        + (", ".join(assessment.get("blocking_object_ids", ())) or "none"),
                        "potential: "
                        + (", ".join(assessment.get("potential_occluder_ids", ())) or "none"),
                    ]
                )
                reasons = ", ".join(assessment.get("reason_codes", ())) or "none"
                wrapped_reasons = textwrap.wrap(reasons, width=47) or ["none"]
                panel_lines.append("reasons: " + wrapped_reasons[0])
                panel_lines.extend("  " + item for item in wrapped_reasons[1:])
                panel_lines.append("")
            limitations = tuple(camera.get("input_limitations", ()))
            if limitations:
                panel_lines.append("CURRENT INPUT LIMITATIONS")
                for limitation in limitations:
                    panel_lines.extend(
                        "- " + item for item in textwrap.wrap(limitation, width=45)
                    )
            decisions = {
                str(item["assessment"]["decision"])
                for item in camera["targets"]
            }
            panel_edge = (
                "#b91c1c"
                if "do_not_use_camera" in decisions
                else (
                    "#b45309"
                    if "insufficient_information" in decisions
                    else "#15803d"
                )
            )
            figure.text(
                0.715,
                0.91,
                "\n".join(panel_lines),
                ha="left",
                va="top",
                fontsize=7.2,
                family="DejaVu Sans Mono",
                color="#0f172a",
                bbox={
                    "boxstyle": "round,pad=0.55",
                    "facecolor": "#ffffff",
                    "edgecolor": panel_edge,
                    "linewidth": 1.6,
                    "alpha": 0.97,
                },
            )

        legend_handles: list[Any] = [
            Patch(
                facecolor="#38bdf8",
                edgecolor="#0284c7",
                alpha=0.20,
                label=coverage_label,
            ),
            Line2D(
                [0],
                [0],
                color="#0f172a",
                marker="o",
                markersize=4,
                linewidth=1.7,
                label="LiDAR-window trajectory",
            ),
            Line2D(
                [0],
                [0],
                color="#2563eb",
                marker="o",
                markersize=7,
                linewidth=2.0,
                label="ego anchor and heading",
            ),
            Patch(
                facecolor=_STATE_STYLE["free"][0],
                edgecolor=_STATE_STYLE["free"][1],
                alpha=0.56,
                label="observed free",
            ),
            Patch(
                facecolor=_STATE_STYLE["occupied"][0],
                edgecolor=_STATE_STYLE["occupied"][1],
                alpha=0.56,
                label="observed occupied",
            ),
            Patch(
                facecolor=_STATE_STYLE["unknown"][0],
                edgecolor=_STATE_STYLE["unknown"][1],
                alpha=0.56,
                label="covered but insufficient: unknown",
            ),
        ]
        if provisional_candidates:
            legend_handles.append(
                Patch(
                    facecolor="none",
                    edgecolor="#7c3aed",
                    linewidth=2.4,
                    linestyle="--",
                    label="Part1 provisional shortlist (visualization only)",
                )
            )
        if camera is not None:
            if camera_pose_m is None:
                legend_handles.append(
                    Patch(
                        facecolor="#f8fafc",
                        edgecolor="#b45309",
                        hatch="//",
                        label="Camera FOV/rays unavailable (see gate panel)",
                    )
                )
            else:
                legend_handles.extend(
                    [
                        Patch(
                            facecolor="#22c55e",
                            edgecolor="#16a34a",
                            alpha=0.16,
                            label="Camera reliable FOV zone",
                        ),
                        Patch(
                            facecolor="#f59e0b",
                            edgecolor="#d97706",
                            alpha=0.20,
                            label="Camera degraded transition zone",
                        ),
                        Patch(
                            facecolor="#ef4444",
                            edgecolor="#dc2626",
                            alpha=0.20,
                            label="Camera unreliable edge zone",
                        ),
                        Line2D(
                            [0],
                            [0],
                            color="#16a34a",
                            linewidth=1.6,
                            label="Part2 LOS clear ray",
                        ),
                        Line2D(
                            [0],
                            [0],
                            color="#dc2626",
                            linewidth=2.0,
                            label="Part2 LOS blocked ray",
                        ),
                        Line2D(
                            [0],
                            [0],
                            color="#f59e0b",
                            linewidth=1.8,
                            label="Part2 LOS uncertain ray",
                        ),
                    ]
                )
            if camera_static_polygons:
                legend_handles.append(
                    Patch(
                        facecolor="#475569",
                        edgecolor="#0f172a",
                        alpha=0.58,
                        label="Part2 static wall/column input",
                    )
                )
            if blocking_object_ids:
                legend_handles.append(
                    Patch(
                        facecolor="none",
                        edgecolor="#b91c1c",
                        linewidth=2.5,
                        label="blocker returned by Camera gate",
                    )
                )
            if potential_occluder_ids:
                legend_handles.append(
                    Patch(
                        facecolor="none",
                        edgecolor="#d97706",
                        linewidth=2.2,
                        label="potential occluder returned by Camera gate",
                    )
                )
        if context_count:
            legend_handles.append(
                Patch(
                    facecolor="none",
                    edgecolor="#94a3b8",
                    linewidth=0.8,
                    alpha=0.5,
                    label="unobserved geometry (state hidden)",
                )
            )
        axis.legend(
            handles=legend_handles,
            loc="upper right",
            frameon=True,
            framealpha=0.94,
            facecolor="white",
            edgecolor="#cbd5e1",
            fontsize=(6.8 if camera is not None else 7.4),
        )
        axis.set_xlabel("ego-relative map X [m]")
        axis.set_ylabel("ego-relative map Y [m]")
        axis.grid(True, color="#cbd5e1", linewidth=0.45, alpha=0.70, zorder=0)
        axis.axhline(0.0, color="#94a3b8", linewidth=0.55, alpha=0.45, zorder=0)
        axis.axvline(0.0, color="#94a3b8", linewidth=0.55, alpha=0.45, zorder=0)

        if camera is None:
            figure.subplots_adjust(left=0.09, right=0.98, bottom=0.09, top=0.88)
        else:
            figure.subplots_adjust(left=0.055, right=0.69, bottom=0.09, top=0.86)
        figure.savefig(
            output,
            dpi=160,
            bbox_inches="tight",
            metadata={
                "Software": (
                    "ParkingAgent Part1 + Part2 Camera-gate renderer"
                    if camera is not None
                    else "ParkingAgent Part1 local-map renderer"
                )
            },
        )
        plt.close(figure)
    return output


__all__ = ["render_local_map"]
