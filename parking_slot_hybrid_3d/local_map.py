from __future__ import annotations

import math
from collections import Counter
from dataclasses import asdict, dataclass, replace
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.spatial import ConvexHull, QhullError

from parking_slot_box_scoring.geometry import convex_clip, polygon_area

from .contracts import DecisionState, FrameRecord, KnownSlot, ScopeStatus
from .pipeline import PipelineResult


LOCAL_MAP_SCHEMA_VERSION = "part1-local-lidar-map/1.1"
_ALLOWED_STATES = frozenset({"free", "occupied", "unknown"})


@dataclass(frozen=True, slots=True)
class LocalMapConfig:
    """Hard bounds for the causal, local Part1 output.

    These values are conservative engineering defaults.  They describe the
    output contract, not a claim about the physical LiDAR's maximum range.
    The evidence evaluator remains responsible for proving that a slot was
    actually reached by rays/hits inside this nominal local envelope.
    """

    frame_count: int = 15
    local_radius_m: float = 18.0
    candidate_max_euclidean_distance_m: float = 15.0
    candidate_max_along_aisle_distance_m: float = 15.0
    candidate_max_lateral_distance_m: float = 8.0
    candidate_row_heading_tolerance_deg: float = 18.0
    candidate_row_cluster_tolerance_m: float = 3.0
    candidate_corridor_width_m: float = 2.5
    max_candidate_row_distance: int = 1
    visualization_shortlist_max_count: int = 2

    def validate(self) -> None:
        if not 3 <= int(self.frame_count) <= 15:
            raise ValueError("frame_count must be between 3 and 15 consecutive frames")
        for name in (
            "local_radius_m",
            "candidate_max_euclidean_distance_m",
            "candidate_max_along_aisle_distance_m",
            "candidate_max_lateral_distance_m",
            "candidate_row_heading_tolerance_deg",
            "candidate_row_cluster_tolerance_m",
            "candidate_corridor_width_m",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
        if self.candidate_max_euclidean_distance_m > self.local_radius_m:
            raise ValueError("candidate distance cannot exceed the local-map radius")
        if not 0.0 < self.candidate_row_heading_tolerance_deg < 45.0:
            raise ValueError("candidate row heading tolerance must be below 45 degrees")
        # This is deliberately not a tunable escape hatch: Part1 may never
        # select a parking candidate more than one row away.
        if int(self.max_candidate_row_distance) != 1:
            raise ValueError("max_candidate_row_distance is a hard value of 1")
        if not 1 <= int(self.visualization_shortlist_max_count) <= 2:
            raise ValueError("visualization_shortlist_max_count must be 1 or 2")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def select_local_frame_window(
    frames: Sequence[FrameRecord],
    *,
    anchor_frame_id: int | None = None,
    frame_count: int = 15,
) -> tuple[FrameRecord, ...]:
    """Return a causal window of consecutive input records ending at anchor."""

    if not 3 <= int(frame_count) <= 15:
        raise ValueError("frame_count must be between 3 and 15")
    ordered = tuple(sorted(frames, key=lambda item: item.frame_id))
    if not ordered:
        raise ValueError("at least one LiDAR frame is required")
    anchor = ordered[-1].frame_id if anchor_frame_id is None else int(anchor_frame_id)
    indices = [index for index, frame in enumerate(ordered) if frame.frame_id == anchor]
    if not indices:
        raise ValueError(f"anchor frame is not present: {anchor}")
    end = indices[0] + 1
    start = max(0, end - int(frame_count))
    selected = ordered[start:end]
    if len(selected) < 3:
        raise ValueError(
            "a local Part1 snapshot requires at least three causal LiDAR frames "
            "at or before the anchor"
        )
    return selected


def _axial_angle_difference_deg(first: float, second: float) -> float:
    delta = abs((float(first) - float(second)) % 180.0)
    return min(delta, 180.0 - delta)


def _slot_axes(slot: KnownSlot) -> tuple[np.ndarray, np.ndarray]:
    yaw = math.radians(float(slot.heading_deg))
    long_axis = np.asarray([math.cos(yaw), math.sin(yaw)], dtype=np.float64)
    short_axis = np.asarray([-long_axis[1], long_axis[0]], dtype=np.float64)
    return long_axis, short_axis


def _cluster_row_offsets(
    offsets: Sequence[tuple[str, float]],
    tolerance_m: float,
) -> tuple[tuple[tuple[str, ...], float], ...]:
    clusters: list[list[tuple[str, float]]] = []
    for item in sorted(offsets, key=lambda value: (value[1], value[0])):
        if not clusters:
            clusters.append([item])
            continue
        mean = float(np.mean([value for _, value in clusters[-1]]))
        if abs(item[1] - mean) <= tolerance_m:
            clusters[-1].append(item)
        else:
            clusters.append([item])
    return tuple(
        (
            tuple(sorted(slot_id for slot_id, _ in cluster)),
            float(np.mean([value for _, value in cluster])),
        )
        for cluster in clusters
    )


def _row_relation(
    target: KnownSlot,
    local_slots: Sequence[KnownSlot],
    ego_xy: np.ndarray,
    map_units_per_meter: float,
    config: LocalMapConfig,
) -> tuple[int | None, int | None, int]:
    normal, _ = _slot_axes(target)
    compatible = [
        slot
        for slot in local_slots
        if _axial_angle_difference_deg(slot.heading_deg, target.heading_deg)
        <= config.candidate_row_heading_tolerance_deg
    ]
    offsets = [
        (
            slot.slot_id,
            float((slot.center_map - ego_xy) @ normal / map_units_per_meter),
        )
        for slot in compatible
    ]
    clusters = _cluster_row_offsets(
        offsets,
        config.candidate_row_cluster_tolerance_m,
    )
    target_cluster: int | None = None
    for index, (slot_ids, _) in enumerate(clusters):
        if target.slot_id in slot_ids:
            target_cluster = index
            break
    if target_cluster is None:
        return None, None, 0

    by_distance = sorted(
        range(len(clusters)),
        key=lambda index: (abs(clusters[index][1]), clusters[index][1]),
    )
    row_distance = by_distance.index(target_cluster)
    target_offset = clusters[target_cluster][1]
    if abs(target_offset) <= config.candidate_row_cluster_tolerance_m * 0.5:
        side_indices = [
            index
            for index, (_, offset) in enumerate(clusters)
            if abs(offset) <= config.candidate_row_cluster_tolerance_m * 0.5
        ]
    else:
        target_sign = math.copysign(1.0, target_offset)
        side_indices = [
            index
            for index, (_, offset) in enumerate(clusters)
            if abs(offset) > config.candidate_row_cluster_tolerance_m * 0.5
            and math.copysign(1.0, offset) == target_sign
        ]
    side_indices.sort(key=lambda index: abs(clusters[index][1]))
    side_rank = side_indices.index(target_cluster) if target_cluster in side_indices else None
    row_size = len(clusters[target_cluster][0])
    return int(row_distance), None if side_rank is None else int(side_rank), row_size


def _corridor_polygon(
    start_map: np.ndarray,
    end_map: np.ndarray,
    width_map: float,
) -> np.ndarray:
    delta = np.asarray(end_map, dtype=np.float64) - np.asarray(start_map, dtype=np.float64)
    length = float(np.linalg.norm(delta))
    if length <= 1e-12:
        return np.repeat(np.asarray(start_map, dtype=np.float64).reshape(1, 2), 4, axis=0)
    tangent = delta / length
    normal = np.asarray([-tangent[1], tangent[0]], dtype=np.float64)
    half = normal * (width_map * 0.5)
    return np.asarray(
        [start_map - half, end_map - half, end_map + half, start_map + half],
        dtype=np.float64,
    )


def _intervening_slot_ids(
    target: KnownSlot,
    local_slots: Sequence[KnownSlot],
    ego_xy: np.ndarray,
    ego_yaw_rad: float,
    map_units_per_meter: float,
    corridor_width_m: float,
    row_heading_tolerance_deg: float,
    row_cluster_tolerance_m: float,
) -> tuple[str, ...]:
    forward = np.asarray(
        [math.cos(ego_yaw_rad), math.sin(ego_yaw_rad)],
        dtype=np.float64,
    )
    delta = target.center_map - ego_xy
    approach = ego_xy + forward * float(delta @ forward)
    corridors = (
        _corridor_polygon(
            ego_xy,
            approach,
            corridor_width_m * map_units_per_meter,
        ),
        _corridor_polygon(
            approach,
            target.center_map,
            corridor_width_m * map_units_per_meter,
        ),
    )
    route_length = float(
        np.linalg.norm(approach - ego_xy)
        + np.linalg.norm(target.center_map - approach)
    )
    blocked: list[str] = []
    target_row_normal, _ = _slot_axes(target)
    for other in local_slots:
        if other.slot_id == target.slot_id:
            continue
        # A neighbour alongside the target belongs to the destination row; it
        # is not a whole intervening row. A wide swept corridor may graze that
        # polygon at the parking entrance, so exclude same-row geometry before
        # testing true between-row crossings.
        same_destination_row = (
            _axial_angle_difference_deg(other.heading_deg, target.heading_deg)
            <= row_heading_tolerance_deg
            and abs(float((other.center_map - target.center_map) @ target_row_normal))
            / map_units_per_meter
            <= row_cluster_tolerance_m
        )
        if same_destination_row:
            continue
        other_progress = abs(float((other.center_map - ego_xy) @ forward))
        if other_progress >= route_length + corridor_width_m * map_units_per_meter:
            continue
        if any(
            polygon_area(convex_clip(other.polygon_map, corridor)) > 1e-10
            for corridor in corridors
        ):
            blocked.append(other.slot_id)
    return tuple(sorted(blocked))


def _candidate_evaluation(
    slot: KnownSlot,
    state: str,
    local_slots: Sequence[KnownSlot],
    anchor: FrameRecord,
    map_units_per_meter: float,
    config: LocalMapConfig,
) -> dict[str, Any]:
    delta_m = (slot.center_map - np.asarray([anchor.map_x, anchor.map_y])) / map_units_per_meter
    forward = np.asarray(
        [math.cos(anchor.map_yaw), math.sin(anchor.map_yaw)],
        dtype=np.float64,
    )
    left = np.asarray([-forward[1], forward[0]], dtype=np.float64)
    euclidean = float(np.linalg.norm(delta_m))
    along = abs(float(delta_m @ forward))
    lateral = abs(float(delta_m @ left))
    row_distance, side_rank, row_size = _row_relation(
        slot,
        local_slots,
        np.asarray([anchor.map_x, anchor.map_y]),
        map_units_per_meter,
        config,
    )
    intervening = _intervening_slot_ids(
        slot,
        local_slots,
        np.asarray([anchor.map_x, anchor.map_y]),
        anchor.map_yaw,
        map_units_per_meter,
        config.candidate_corridor_width_m,
        config.candidate_row_heading_tolerance_deg,
        config.candidate_row_cluster_tolerance_m,
    )
    geometry_reasons: list[str] = []
    aisle_heading_deg = math.degrees(anchor.map_yaw)
    _, row_tangent = _slot_axes(slot)
    row_tangent_deg = math.degrees(math.atan2(row_tangent[1], row_tangent[0]))
    aisle_alignment_error_deg = _axial_angle_difference_deg(
        aisle_heading_deg,
        row_tangent_deg,
    )
    if euclidean > config.candidate_max_euclidean_distance_m:
        geometry_reasons.append("euclidean_distance_exceeded")
    if along > config.candidate_max_along_aisle_distance_m:
        geometry_reasons.append("along_aisle_distance_exceeded")
    if lateral > config.candidate_max_lateral_distance_m:
        geometry_reasons.append("lateral_distance_exceeded")
    if row_distance is None or row_distance > config.max_candidate_row_distance:
        geometry_reasons.append("row_distance_exceeded")
    # Only the closest observed row on either side of the current channel is
    # directly reachable.  This prevents a nominal row_distance=1 from
    # accidentally accepting a second row on the same side.
    if side_rank is None or side_rank > 0:
        geometry_reasons.append("not_adjacent_to_current_channel")
    if row_size < 2:
        geometry_reasons.append("row_topology_unresolved")
    if aisle_alignment_error_deg > config.candidate_row_heading_tolerance_deg:
        geometry_reasons.append("aisle_alignment_unresolved")
    if intervening:
        geometry_reasons.append("intervening_slot_row")
    state_shortlistable = state in {"free", "unknown"}
    state_reasons = () if state_shortlistable else ("occupied_not_shortlistable",)
    geometry_rejection_reasons = tuple(sorted(set(geometry_reasons)))
    rejection_reasons = tuple(sorted(set(geometry_rejection_reasons + state_reasons)))
    return {
        "visualization_shortlist_eligible": state_shortlistable
        and not geometry_rejection_reasons,
        "state_shortlistable": state_shortlistable,
        "euclidean_distance_m": euclidean,
        "along_aisle_distance_m": along,
        "lateral_distance_m": lateral,
        "row_distance": row_distance,
        "same_side_row_rank": side_rank,
        "inferred_row_slot_count": row_size,
        "aisle_alignment_error_deg": aisle_alignment_error_deg,
        "intervening_slot_ids": list(intervening),
        "geometry_reachable": not geometry_rejection_reasons,
        # Kept as an explicitly non-selection alias for schema readers that
        # displayed this diagnostic before the provisional shortlist existed.
        "reachable": not geometry_rejection_reasons,
        "geometry_rejection_reasons": list(geometry_rejection_reasons),
        "rejection_reasons": list(rejection_reasons),
    }


def actual_lidar_coverage_polygon(
    frames: Sequence[FrameRecord],
    provider: Any,
    map_units_per_meter: float,
    max_range_m: float,
) -> list[list[float]] | None:
    """Build a range-clipped hull from actual map-frame LiDAR endpoints.

    This footprint is for visualization only.  It never promotes a slot to
    observed: local state inclusion still requires exact scope ray/hit
    evidence from the known-slot scope evaluator.
    """

    if not math.isfinite(map_units_per_meter) or map_units_per_meter <= 0.0:
        raise ValueError("map_units_per_meter must be positive and finite")
    if not math.isfinite(max_range_m) or max_range_m <= 0.0:
        raise ValueError("max_range_m must be positive and finite")
    radius_map = float(max_range_m) * float(map_units_per_meter)
    samples: list[np.ndarray] = []
    for frame in sorted(frames, key=lambda item: item.frame_id):
        try:
            points = np.asarray(provider.load(frame.frame_id), dtype=np.float64)
        except Exception:
            continue
        if points.ndim != 2 or points.shape[1] < 2:
            continue
        xy = points[:, :2]
        xy = xy[np.isfinite(xy).all(axis=1)]
        if len(xy) == 0:
            continue
        origin = np.asarray([frame.map_x, frame.map_y], dtype=np.float64)
        xy = xy[np.linalg.norm(xy - origin, axis=1) <= radius_map + 1e-12]
        if len(xy) == 0:
            continue
        stride = max(1, len(xy) // 5000)
        samples.append(xy[::stride])
        samples.append(origin.reshape(1, 2))
    if not samples:
        return None
    points = np.vstack(samples)
    if len(points) < 3:
        return None
    try:
        hull = ConvexHull(points)
    except QhullError:
        return None
    polygon = points[hull.vertices]
    return [[float(value) for value in point] for point in polygon]


def build_local_map_snapshot(
    result: PipelineResult,
    config: LocalMapConfig | None = None,
    *,
    map_coordinate_frame: str | None = None,
    coverage_polygon_map: Sequence[Sequence[float]] | None = None,
) -> dict[str, Any]:
    """Build the official bounded Part1 map without global state leakage."""

    settings = config or LocalMapConfig()
    settings.validate()
    frames = tuple(sorted(result.frames, key=lambda item: item.frame_id))
    if not frames:
        raise ValueError("local map requires at least one LiDAR frame")
    if len(frames) > settings.frame_count:
        raise ValueError(
            "pipeline result contains more than the configured local frame window; "
            "run Part1 on select_local_frame_window(...)"
        )
    anchor = frames[-1]
    frame_ids = {frame.frame_id for frame in frames}
    frame_centers = np.asarray([[frame.map_x, frame.map_y] for frame in frames])
    scope_by_id = {scope.slot_id: scope for scope in result.scopes}
    decision_by_id = {decision.slot_id: decision for decision in result.decisions}
    slot_by_id = {slot.slot_id: slot for slot in result.all_slots}
    anchor_xy = np.asarray([anchor.map_x, anchor.map_y], dtype=np.float64)
    topology_radius_map = (
        settings.local_radius_m + settings.candidate_corridor_width_m
    ) * result.map_units_per_meter
    # The complete slot database contributes geometry only to a bounded local
    # topology check. It contributes no state. This prevents an unobserved row
    # between ego and a candidate from disappearing merely because Part1 did
    # not assign that row an occupancy state in the current LiDAR window.
    topology_slots = [
        slot
        for slot in result.all_slots
        if float(np.linalg.norm(slot.center_map - anchor_xy)) <= topology_radius_map
    ]

    local_slots: list[KnownSlot] = []
    for slot_id, scope in sorted(scope_by_id.items()):
        slot = slot_by_id.get(slot_id)
        if slot is None or scope.scope_status is ScopeStatus.OUT_OF_ROUTE:
            continue
        minimum_distance_m = float(
            np.min(np.linalg.norm(frame_centers - slot.center_map, axis=1))
            / result.map_units_per_meter
        )
        if minimum_distance_m > settings.local_radius_m:
            continue
        # A nearby pose or a missing point-cloud file is not LiDAR coverage.
        # The official local map contains a slot only when at least one actual
        # measured ray crossed its prism or ended inside it.
        measured_evidence_frames = (
            set(scope.crossing_frames) | set(scope.hit_frames)
        ) & frame_ids
        if not measured_evidence_frames:
            continue
        local_slots.append(slot)

    local_ids = {slot.slot_id for slot in local_slots}
    slot_rows: list[dict[str, Any]] = []
    shortlist_rows: list[
        tuple[tuple[float, float, int, str], str, str]
    ] = []
    for slot in sorted(local_slots, key=lambda item: item.slot_id):
        scope = scope_by_id[slot.slot_id]
        decision = decision_by_id.get(slot.slot_id)
        state = "unknown" if decision is None else decision.state.value
        if state not in _ALLOWED_STATES:
            raise ValueError(f"unsupported local slot state: {state}")
        candidate = _candidate_evaluation(
            slot,
            state,
            topology_slots,
            anchor,
            result.map_units_per_meter,
            settings,
        )
        if candidate["visualization_shortlist_eligible"]:
            shortlist_rows.append(
                (
                    (
                        float(candidate["euclidean_distance_m"]),
                        float(candidate["along_aisle_distance_m"]),
                        int(candidate["row_distance"]),
                        slot.slot_id,
                    ),
                    state,
                    slot.slot_id,
                )
            )
        slot_rows.append(
            {
                "slot_id": slot.slot_id,
                "polygon_map": slot.polygon_map.tolist(),
                "center_map": slot.center_map.tolist(),
                "heading_deg": float(slot.heading_deg),
                "state": state,
                "decision_reason": (
                    "insufficient_local_evidence"
                    if decision is None
                    else decision.decision_reason
                ),
                "observed_frame_ids": sorted(
                    (set(scope.crossing_frames) | set(scope.hit_frames)) & frame_ids
                ),
                "missing_frame_ids": sorted(set(scope.missing_frames) & frame_ids),
                "ray_frame_count": len(
                    (set(scope.crossing_frames) | set(scope.hit_frames)) & frame_ids
                ),
                "core_ray_coverage": float(scope.core_ray_coverage),
                "visualization_shortlist_eligible": bool(
                    candidate["visualization_shortlist_eligible"]
                ),
                "visualization_shortlist_metrics": candidate,
            }
        )

    # This is deliberately a visualization shortlist, not a parking decision.
    # Prefer state diversity when both free and unknown are locally reachable,
    # then fill any remaining display slot by geometric proximity. The policy
    # is explicitly provisional and must not be consumed by Part2.
    ranked_shortlist = sorted(shortlist_rows)
    selected_shortlist_ids: list[str] = []
    maximum_shortlist_count = int(settings.visualization_shortlist_max_count)
    for desired_state in (DecisionState.FREE.value, DecisionState.UNKNOWN.value):
        match = next(
            (
                slot_id
                for _, state, slot_id in ranked_shortlist
                if state == desired_state and slot_id not in selected_shortlist_ids
            ),
            None,
        )
        if match is not None and len(selected_shortlist_ids) < maximum_shortlist_count:
            selected_shortlist_ids.append(match)
    for _, _, slot_id in ranked_shortlist:
        if len(selected_shortlist_ids) >= maximum_shortlist_count:
            break
        if slot_id not in selected_shortlist_ids:
            selected_shortlist_ids.append(slot_id)

    provisional_candidates: list[dict[str, Any]] = []
    for rank, slot_id in enumerate(selected_shortlist_ids, start=1):
        shortlist_row = next(row for row in slot_rows if row["slot_id"] == slot_id)
        provisional_candidates.append(
            {
                "display_rank": rank,
                "display_label": chr(ord("A") + rank - 1),
                "slot_id": slot_id,
                "state": shortlist_row["state"],
                "provisional": True,
                "visualization_only": True,
                "final_selection": False,
                "is_final_selection": False,
                **dict(shortlist_row["visualization_shortlist_metrics"]),
            }
        )
    state_counts = Counter(row["state"] for row in slot_rows)
    coverage_radius_map = settings.local_radius_m * result.map_units_per_meter
    coverage_polygon: list[list[float]] | None = None
    if coverage_polygon_map is not None:
        raw_polygon = np.asarray(coverage_polygon_map, dtype=np.float64)
        if (
            raw_polygon.ndim != 2
            or raw_polygon.shape[0] < 3
            or raw_polygon.shape[1] != 2
            or not np.isfinite(raw_polygon).all()
        ):
            raise ValueError("coverage_polygon_map must be a finite [N,2] polygon")
        coverage_polygon = raw_polygon.tolist()
    timestamps = [
        frame.lidar_timestamp for frame in frames if frame.lidar_timestamp is not None
    ]
    return {
        "schema_version": LOCAL_MAP_SCHEMA_VERSION,
        "map_semantics": "local_incomplete_lidar_snapshot",
        "coordinate_frame": map_coordinate_frame,
        "map_units_per_meter": float(result.map_units_per_meter),
        "is_global_truth_map": False,
        "state_domain": ["free", "occupied", "unknown"],
        "anchor_pose": {
            "frame_id": int(anchor.frame_id),
            "map_xy": [float(anchor.map_x), float(anchor.map_y)],
            "map_yaw_rad": float(anchor.map_yaw),
            "lidar_timestamp": anchor.lidar_timestamp,
        },
        "lidar_window": {
            "frame_ids": [int(frame.frame_id) for frame in frames],
            "frame_count": len(frames),
            "first_frame_id": int(frames[0].frame_id),
            "last_frame_id": int(frames[-1].frame_id),
            "first_timestamp": min(timestamps) if timestamps else None,
            "last_timestamp": max(timestamps) if timestamps else None,
            "causal_trailing_window": True,
            "consecutive_input_records": True,
        },
        "lidar_coverage": {
            "semantics": (
                "range-clipped convex hull of actual LiDAR endpoints; hull interior does "
                "not imply visibility and slot state still requires exact ray/hit evidence"
                if coverage_polygon is not None
                else "nominal local envelope fallback; no endpoint footprint was available"
            ),
            "source": (
                "actual_range_clipped_lidar_endpoints"
                if coverage_polygon is not None
                else "nominal_pose_discs_fallback"
            ),
            "complete": coverage_polygon is not None,
            "nominal_radius_m": float(settings.local_radius_m),
            "nominal_radius_map_units": float(coverage_radius_map),
            "polygon_map": coverage_polygon,
            "pose_centers_map": [
                [float(frame.map_x), float(frame.map_y)] for frame in frames
            ],
        },
        "slots": slot_rows,
        "candidate_slot_id": None,
        "candidate": None,
        "candidate_selection": {
            "status": "not_defined",
            "selected": False,
            "reason": "not_performed_by_part1",
            "selection_reason": "not_performed_by_part1_local_map",
        },
        "provisional_candidate_ids": selected_shortlist_ids,
        "provisional_candidates": provisional_candidates,
        "provisional_candidate_display": {
            "purpose": "debug_visualization_only",
            "provisional": True,
            "is_final_parking_decision": False,
            "consumed_by_part2": False,
            "selection_policy_status": "tbd",
            "maximum_count": maximum_shortlist_count,
            "allowed_states": ["free", "unknown"],
            "occupied_excluded_from_display_shortlist": True,
            "eligible_local_slot_count": len(shortlist_rows),
            "display_heuristic": (
                "prefer one reachable free and one reachable unknown when available; "
                "fill remaining display positions by local geometric proximity"
            ),
            "hard_max_row_distance": 1,
            "topology_source": "known_slot_geometry_local_neighborhood_fail_closed",
            "topology_slot_count": len(topology_slots),
            "route_model": "ego_heading_l_shaped_local_corridor",
            "global_route_or_row_truth_used": False,
            "heuristic_config": settings.to_dict(),
        },
        "visualization_contract": {
            "part1_lidar_local_map_included": True,
            "part2_camera_observability_included": False,
            "camera_fov_or_occlusion_overlay_included": False,
        },
        "counts": {
            "local_slot_count": len(slot_rows),
            "free": state_counts.get("free", 0),
            "occupied": state_counts.get("occupied", 0),
            "unknown": state_counts.get("unknown", 0),
            "known_slot_database_total": len(result.all_slots),
            "unobserved_slots_omitted": len(result.all_slots) - len(local_ids),
        },
        "notes": [
            "Only a few consecutive LiDAR frames contribute to this map.",
            "Slots outside the local evidence envelope are omitted and have no Part1 state.",
            "No production parking-candidate policy is asserted by Part1.",
            "Up to two free/unknown slots may be highlighted only as a provisional visualization shortlist.",
            "Part2 Camera observability is not evaluated or drawn in this figure.",
        ],
    }


def restrict_result_to_local_map(
    result: PipelineResult,
    snapshot: Mapping[str, Any],
) -> PipelineResult:
    """Remove unobserved/global slots before any final Part1 artifact is written."""

    raw_slots = snapshot.get("slots", ())
    local_ids = {
        str(item["slot_id"])
        for item in raw_slots
        if isinstance(item, Mapping) and item.get("slot_id") is not None
    }
    local_slots = tuple(
        slot for slot in result.all_slots if slot.slot_id in local_ids
    )
    return replace(
        result,
        scopes=tuple(scope for scope in result.scopes if scope.slot_id in local_ids),
        decisions=tuple(
            decision for decision in result.decisions if decision.slot_id in local_ids
        ),
        traces=tuple(
            trace for trace in result.traces if str(trace.get("slot_id", "")) in local_ids
        ),
        map_total=len(local_slots),
        slots=local_slots,
        all_slots=local_slots,
    )


__all__ = [
    "LOCAL_MAP_SCHEMA_VERSION",
    "LocalMapConfig",
    "actual_lidar_coverage_polygon",
    "build_local_map_snapshot",
    "restrict_result_to_local_map",
    "select_local_frame_window",
]
