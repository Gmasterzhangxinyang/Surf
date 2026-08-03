from __future__ import annotations

import math
from collections import defaultdict

import numpy as np

from parking_slot_box_scoring.geometry import ensure_ccw, points_in_polygon

from .config import Hybrid3DConfig
from .contracts import (
    FreeEvidence,
    FreeSpaceDetails,
    GateResult,
    MetricSlot,
    OccupiedEvidence,
    SlotAccumulation,
)


Voxel = tuple[int, int, int]
_VOXEL_KEY_BITS = 21
_VOXEL_KEY_MASK = (1 << _VOXEL_KEY_BITS) - 1
_OWNERSHIP_FAILURE_CODES = frozenset(
    {
        "adjacent_overlap_conflict",
        "boundary_dominated",
        "low_core_overlap",
        "outside_residual_conflict",
    }
)


_FAILURE_BY_GATE = {
    "ray_frames": "insufficient_ray_frames",
    "viewpoints": "insufficient_viewpoints",
    "viewpoint_separation": "insufficient_viewpoint_separation",
    "volume_coverage": "low_free_volume_coverage",
    "near_ground_coverage": "low_near_ground_coverage",
    "unobserved_component": "large_unobserved_component",
    "occlusion": "high_occlusion",
    "weak_obstacle_veto": "weak_obstacle_evidence",
    "core_hit_veto": "unresolved_core_hit_evidence",
    "occupied_weak_veto": "weak_occupied_evidence",
    "ownership_veto": "ownership_conflict",
    "conflict_veto": "occupied_free_conflict",
}


def _voxel_sizes(config: Hybrid3DConfig) -> tuple[float, float, float]:
    return (config.voxel_xy_m, config.voxel_xy_m, config.voxel_z_m)


def _voxel_for_point(point: np.ndarray, config: Hybrid3DConfig) -> Voxel:
    sizes = np.asarray(_voxel_sizes(config), dtype=np.float64)
    indices = np.floor(np.asarray(point, dtype=np.float64) / sizes).astype(np.int64)
    return int(indices[0]), int(indices[1]), int(indices[2])


def _zigzag_encode(values: np.ndarray) -> np.ndarray:
    return np.where(values >= 0, values * 2, -values * 2 - 1).astype(np.int64)


def _encode_voxel_array(cells: np.ndarray) -> np.ndarray:
    encoded = _zigzag_encode(np.asarray(cells, dtype=np.int64))
    if np.any(encoded > _VOXEL_KEY_MASK):
        raise ValueError("voxel index exceeds deterministic key range")
    return (
        (encoded[:, 0] << (2 * _VOXEL_KEY_BITS))
        | (encoded[:, 1] << _VOXEL_KEY_BITS)
        | encoded[:, 2]
    )


def _encode_voxel(cell: Voxel) -> int:
    return int(_encode_voxel_array(np.asarray([cell], dtype=np.int64))[0])


def _zigzag_decode(value: int) -> int:
    return value // 2 if value % 2 == 0 else -(value // 2) - 1


def _decode_voxel(value: int) -> Voxel:
    z_value = value & _VOXEL_KEY_MASK
    y_value = (value >> _VOXEL_KEY_BITS) & _VOXEL_KEY_MASK
    x_value = (value >> (2 * _VOXEL_KEY_BITS)) & _VOXEL_KEY_MASK
    return _zigzag_decode(x_value), _zigzag_decode(y_value), _zigzag_decode(z_value)


def _core_voxels(
    metric_slot: MetricSlot,
    config: Hybrid3DConfig,
) -> tuple[set[Voxel], set[tuple[int, int]], set[int]]:
    polygon = np.asarray(metric_slot.core_polygon_local_m, dtype=np.float64)
    if polygon.ndim != 2 or polygon.shape[1] != 2 or len(polygon) < 3:
        raise ValueError("core polygon must contain at least three points")
    xy_min = polygon.min(axis=0)
    xy_max = polygon.max(axis=0)
    x_indices = range(
        int(math.floor(xy_min[0] / config.voxel_xy_m)),
        int(math.floor(xy_max[0] / config.voxel_xy_m)) + 1,
    )
    y_indices = range(
        int(math.floor(xy_min[1] / config.voxel_xy_m)),
        int(math.floor(xy_max[1] / config.voxel_xy_m)) + 1,
    )
    xy_cells = [(x_index, y_index) for x_index in x_indices for y_index in y_indices]
    xy_centers = np.asarray(
        [
            [
                (x_index + 0.5) * config.voxel_xy_m,
                (y_index + 0.5) * config.voxel_xy_m,
            ]
            for x_index, y_index in xy_cells
        ],
        dtype=np.float64,
    )
    inside = points_in_polygon(xy_centers, polygon)
    core_xy = {cell for cell, keep in zip(xy_cells, inside) if bool(keep)}

    z_indices: set[int] = set()
    raw_z_indices = range(
        int(math.floor(config.free_near_ground_z_min_m / config.voxel_z_m)),
        int(math.floor(config.vehicle_z_max_m / config.voxel_z_m)) + 1,
    )
    for z_index in raw_z_indices:
        center_z = (z_index + 0.5) * config.voxel_z_m
        if config.free_near_ground_z_min_m <= center_z <= config.vehicle_z_max_m:
            z_indices.add(z_index)
    core = {
        (x_index, y_index, z_index)
        for x_index, y_index in core_xy
        for z_index in z_indices
    }
    if not core:
        raise ValueError("core voxel grid is empty")
    return core, core_xy, z_indices


def _clip_segments_to_core(
    origin: np.ndarray,
    endpoints: np.ndarray,
    metric_slot: MetricSlot,
    config: Hybrid3DConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    polygon = ensure_ccw(np.asarray(metric_slot.core_polygon_local_m, dtype=np.float64))
    directions = endpoints - origin
    count = len(endpoints)
    valid = np.ones(count, dtype=bool)
    t_entry = np.zeros(count, dtype=np.float64)
    t_exit = np.ones(count, dtype=np.float64)
    epsilon = 1e-12
    for index, start in enumerate(polygon):
        edge = polygon[(index + 1) % len(polygon)] - start
        relative = origin[:2] - start
        numerator = float(edge[0] * relative[1] - edge[1] * relative[0])
        denominators = edge[0] * directions[:, 1] - edge[1] * directions[:, 0]
        parallel = np.abs(denominators) <= epsilon
        if numerator < -epsilon:
            valid &= ~parallel
        moving = ~parallel
        boundary = np.zeros(count, dtype=np.float64)
        boundary[moving] = -numerator / denominators[moving]
        entering = moving & (denominators > 0.0)
        exiting = moving & (denominators < 0.0)
        t_entry[entering] = np.maximum(t_entry[entering], boundary[entering])
        t_exit[exiting] = np.minimum(t_exit[exiting], boundary[exiting])

    z_direction = directions[:, 2]
    parallel_z = np.abs(z_direction) <= epsilon
    if not (
        config.free_near_ground_z_min_m - epsilon
        <= origin[2]
        <= config.vehicle_z_max_m + epsilon
    ):
        valid &= ~parallel_z
    moving_z = ~parallel_z
    if np.any(moving_z):
        z_t0 = (config.free_near_ground_z_min_m - origin[2]) / z_direction[moving_z]
        z_t1 = (config.vehicle_z_max_m - origin[2]) / z_direction[moving_z]
        moving_indices = np.flatnonzero(moving_z)
        t_entry[moving_indices] = np.maximum(
            t_entry[moving_indices],
            np.minimum(z_t0, z_t1),
        )
        t_exit[moving_indices] = np.minimum(
            t_exit[moving_indices],
            np.maximum(z_t0, z_t1),
        )

    valid &= (
        (t_entry <= t_exit + epsilon)
        & (t_exit >= -epsilon)
        & (t_entry <= 1.0 + epsilon)
    )
    return valid, np.clip(t_entry, 0.0, 1.0), np.clip(t_exit, 0.0, 1.0)


def _sample_clipped_voxels(
    origin: np.ndarray,
    endpoints: np.ndarray,
    valid: np.ndarray,
    t_entry: np.ndarray,
    t_exit: np.ndarray,
    encoded_core_voxels: set[int],
    config: Hybrid3DConfig,
    batch_size: int = 2048,
) -> set[Voxel]:
    valid_indices = np.flatnonzero(valid)
    if not len(valid_indices):
        return set()
    directions = endpoints - origin
    sizes = np.asarray(_voxel_sizes(config), dtype=np.float64)
    sample_step_m = float(np.min(sizes) * 0.45)
    encoded_result: set[int] = set()
    for batch_start in range(0, len(valid_indices), batch_size):
        indices = valid_indices[batch_start : batch_start + batch_size]
        starts = origin + directions[indices] * t_entry[indices, None]
        ends = origin + directions[indices] * t_exit[indices, None]
        lengths = np.linalg.norm(ends - starts, axis=1)
        sample_counts = np.maximum(np.ceil(lengths / sample_step_m).astype(np.int64) + 1, 1)
        maximum_samples = int(sample_counts.max())
        sample_indices = np.arange(maximum_samples, dtype=np.float64)[None, :]
        denominators = np.maximum(sample_counts - 1, 1)[:, None]
        fractions = sample_indices / denominators
        sample_mask = sample_indices < sample_counts[:, None]
        samples = starts[:, None, :] + fractions[:, :, None] * (ends - starts)[:, None, :]
        cells = np.floor(samples[sample_mask] / sizes).astype(np.int64)
        encoded_cells = np.unique(_encode_voxel_array(cells))
        encoded_result.update(
            int(value) for value in encoded_cells if int(value) in encoded_core_voxels
        )
    return {_decode_voxel(value) for value in encoded_result}


def _bearing_deg(origin: np.ndarray) -> float:
    return float((math.degrees(math.atan2(float(origin[1]), float(origin[0]))) + 360.0) % 360.0)


def _angular_distance_deg(left: float, right: float) -> float:
    difference = abs(left - right) % 360.0
    return min(difference, 360.0 - difference)


def _viewpoint_summary(
    bearings: list[float],
    minimum_separation_deg: float,
) -> tuple[tuple[float, ...], int, float]:
    ordered = tuple(sorted(set(round(value, 8) for value in bearings)))
    representatives: list[float] = []
    for bearing in ordered:
        if all(
            _angular_distance_deg(bearing, existing) >= minimum_separation_deg
            for existing in representatives
        ):
            representatives.append(bearing)
    maximum_separation = max(
        (
            _angular_distance_deg(left, right)
            for index, left in enumerate(ordered)
            for right in ordered[index + 1 :]
        ),
        default=0.0,
    )
    return ordered, len(representatives), float(maximum_separation)


def _largest_component_ratio(voxels: set[Voxel], denominator: int) -> float:
    if not voxels:
        return 0.0
    remaining = set(voxels)
    largest = 0
    neighbor_steps = (
        (1, 0, 0),
        (-1, 0, 0),
        (0, 1, 0),
        (0, -1, 0),
        (0, 0, 1),
        (0, 0, -1),
    )
    while remaining:
        start = remaining.pop()
        stack = [start]
        size = 1
        while stack:
            current = stack.pop()
            for step in neighbor_steps:
                neighbor = (
                    current[0] + step[0],
                    current[1] + step[1],
                    current[2] + step[2],
                )
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    stack.append(neighbor)
                    size += 1
        largest = max(largest, size)
    return largest / max(denominator, 1)


def _voxel_components(voxels: set[Voxel]) -> tuple[frozenset[Voxel], ...]:
    remaining = set(voxels)
    components: list[frozenset[Voxel]] = []
    neighbor_steps = tuple(
        (dx, dy, dz)
        for dx in (-1, 0, 1)
        for dy in (-1, 0, 1)
        for dz in (-1, 0, 1)
        if (dx, dy, dz) != (0, 0, 0)
    )
    while remaining:
        start = remaining.pop()
        stack = [start]
        component = {start}
        while stack:
            current = stack.pop()
            for step in neighbor_steps:
                neighbor = (
                    current[0] + step[0],
                    current[1] + step[1],
                    current[2] + step[2],
                )
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    component.add(neighbor)
                    stack.append(neighbor)
        components.append(frozenset(component))
    return tuple(components)


def _voxel_inside_best_box(
    cell: Voxel,
    occupied_evidence: OccupiedEvidence,
    config: Hybrid3DConfig,
) -> bool:
    values = dict(occupied_evidence.best_box)
    required = {"center_x_m", "center_y_m", "yaw_rad", "length_m", "width_m"}
    if set(values) != required:
        return False
    length = float(values["length_m"])
    width = float(values["width_m"])
    if length <= 0.0 or width <= 0.0:
        return False
    center = np.asarray(
        [
            (cell[0] + 0.5) * config.voxel_xy_m,
            (cell[1] + 0.5) * config.voxel_xy_m,
        ],
        dtype=np.float64,
    )
    delta = center - np.asarray(
        [values["center_x_m"], values["center_y_m"]],
        dtype=np.float64,
    )
    yaw = float(values["yaw_rad"])
    long_axis = np.asarray([math.cos(yaw), math.sin(yaw)], dtype=np.float64)
    short_axis = np.asarray([-math.sin(yaw), math.cos(yaw)], dtype=np.float64)
    return bool(
        abs(float(delta @ long_axis)) <= length / 2.0 + 1e-12
        and abs(float(delta @ short_axis)) <= width / 2.0 + 1e-12
    )


def _occupied_free_conflict(
    occupied_evidence: OccupiedEvidence,
    obstacle_support: dict[Voxel, dict[int, int]],
    free_support: dict[Voxel, set[int]],
    bearing_by_frame: dict[int, float],
    config: Hybrid3DConfig,
) -> tuple[
    bool,
    tuple[Voxel, ...],
    tuple[int, ...],
    tuple[int, ...],
    int,
    float,
]:
    empty = (False, (), (), (), 0, 0.0)
    if not (occupied_evidence.strong or occupied_evidence.weak) or not occupied_evidence.best_box:
        return empty

    candidate_cells: set[Voxel] = set()
    for cell in set(obstacle_support) & set(free_support):
        hit_frames = set(obstacle_support[cell])
        if not (set(free_support[cell]) - hit_frames):
            continue
        if _voxel_inside_best_box(cell, occupied_evidence, config):
            candidate_cells.add(cell)
    if not candidate_cells:
        return empty

    summaries: list[
        tuple[
            bool,
            tuple[Voxel, ...],
            tuple[int, ...],
            tuple[int, ...],
            int,
            float,
        ]
    ] = []
    for component in _voxel_components(candidate_cells):
        hit_frames = {
            frame_id
            for cell in component
            for frame_id in obstacle_support[cell]
        }
        raw_free_frames = {
            frame_id
            for cell in component
            for frame_id in free_support[cell]
        }
        # A frame that also returns an obstacle in this component describes the
        # normal surface-before-return geometry, not independent counterevidence.
        free_frames = raw_free_frames - hit_frames
        free_bearings = [
            bearing_by_frame[frame_id]
            for frame_id in free_frames
            if frame_id in bearing_by_frame
        ]
        _, viewpoint_count, viewpoint_separation = _viewpoint_summary(
            free_bearings,
            config.free_conflict_min_viewpoint_separation_deg,
        )
        conflict = bool(
            len(component) >= config.free_conflict_min_voxels
            and len(hit_frames) >= config.free_conflict_min_hit_frames
            and len(free_frames) >= config.free_conflict_min_free_frames
            and viewpoint_count >= config.free_conflict_min_viewpoints
            and viewpoint_separation
            >= config.free_conflict_min_viewpoint_separation_deg
        )
        summaries.append(
            (
                conflict,
                tuple(sorted(component)),
                tuple(sorted(hit_frames)),
                tuple(sorted(free_frames)),
                viewpoint_count,
                float(viewpoint_separation),
            )
        )

    return max(
        summaries,
        key=lambda item: (
            item[0],
            len(item[1]),
            len(item[3]),
            len(item[2]),
            item[4],
            item[5],
        ),
    )


def _occupied_weak_has_core_support(
    occupied_evidence: OccupiedEvidence,
    weak_obstacle: bool,
    config: Hybrid3DConfig,
) -> bool:
    if not occupied_evidence.weak:
        return False
    if weak_obstacle:
        return True
    features = occupied_evidence.features
    if features is None:
        # Missing target-core provenance is not proof that the weak evidence is
        # external. Legacy and hand-built evidence therefore remain conservative.
        return True
    return bool(
        features.core_point_count
        >= max(10, int(math.ceil(config.occupied_min_points / 4.0)))
        and features.core_voxel_count
        >= max(4, int(math.ceil(config.occupied_min_voxels / 2.0)))
        and features.core_supported_frame_count >= 2
    )


def _weak_obstacle_component(
    support: dict[Voxel, dict[int, int]],
    config: Hybrid3DConfig,
) -> tuple[bool, int, int]:
    remaining = set(support)
    best_points = 0
    best_frames = 0
    weak = False
    neighbor_steps = tuple(
        (dx, dy, dz)
        for dx in (-1, 0, 1)
        for dy in (-1, 0, 1)
        for dz in (-1, 0, 1)
        if (dx, dy, dz) != (0, 0, 0)
    )
    while remaining:
        start = remaining.pop()
        stack = [start]
        component = [start]
        while stack:
            current = stack.pop()
            for step in neighbor_steps:
                neighbor = (
                    current[0] + step[0],
                    current[1] + step[1],
                    current[2] + step[2],
                )
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    stack.append(neighbor)
                    component.append(neighbor)
        point_count = sum(sum(support[cell].values()) for cell in component)
        frames = {frame_id for cell in component for frame_id in support[cell]}
        if (point_count, len(frames)) > (best_points, best_frames):
            best_points = point_count
            best_frames = len(frames)
        weak |= (
            point_count >= config.free_weak_obstacle_min_points
            and len(frames) >= config.free_weak_obstacle_min_frames
        )
    return weak, best_points, best_frames


def _free_gates(
    *,
    ray_frame_count: int,
    viewpoint_count: int,
    viewpoint_separation_deg: float,
    observed_volume_ratio: float,
    near_ground_bev_coverage: float,
    unobserved_component_ratio: float,
    occlusion_ratio: float,
    weak_obstacle: bool,
    unresolved_core_hit: bool,
    occupied_weak_veto: bool,
    ownership_conflict: bool,
    conflict: bool,
    config: Hybrid3DConfig,
) -> tuple[GateResult, ...]:
    return (
        GateResult("ray_frames", ray_frame_count >= config.free_min_ray_frames, ray_frame_count, f">={config.free_min_ray_frames}"),
        GateResult("viewpoints", viewpoint_count >= config.free_min_viewpoints, viewpoint_count, f">={config.free_min_viewpoints}"),
        GateResult("viewpoint_separation", viewpoint_separation_deg >= config.free_min_viewpoint_separation_deg, viewpoint_separation_deg, f">={config.free_min_viewpoint_separation_deg}"),
        GateResult("volume_coverage", observed_volume_ratio >= config.free_min_volume_coverage, observed_volume_ratio, f">={config.free_min_volume_coverage}"),
        GateResult("near_ground_coverage", near_ground_bev_coverage >= config.free_min_near_ground_bev_coverage, near_ground_bev_coverage, f">={config.free_min_near_ground_bev_coverage}"),
        GateResult("unobserved_component", unobserved_component_ratio <= config.free_max_unobserved_component_ratio, unobserved_component_ratio, f"<={config.free_max_unobserved_component_ratio}"),
        GateResult("occlusion", occlusion_ratio <= config.free_max_occlusion_ratio, occlusion_ratio, f"<={config.free_max_occlusion_ratio}"),
        GateResult("weak_obstacle_veto", not weak_obstacle, weak_obstacle, False),
        GateResult("core_hit_veto", not unresolved_core_hit, unresolved_core_hit, False),
        GateResult("occupied_weak_veto", not occupied_weak_veto, occupied_weak_veto, False),
        GateResult("ownership_veto", not ownership_conflict, ownership_conflict, False),
        GateResult("conflict_veto", not conflict, conflict, False),
    )


def _failure_codes(gates: tuple[GateResult, ...]) -> tuple[str, ...]:
    return tuple(
        _FAILURE_BY_GATE.get(gate.name, f"failed_gate_{gate.name}")
        for gate in gates
        if not gate.passed
    )


def evaluate_free_space(
    metric_slot: MetricSlot,
    accumulation: SlotAccumulation,
    occupied_evidence: OccupiedEvidence,
    config: Hybrid3DConfig,
) -> FreeEvidence:
    try:
        config.validate()
        if accumulation.slot_id != metric_slot.slot_id:
            raise ValueError("accumulation slot_id does not match metric slot")
        core_voxels, core_xy, _ = _core_voxels(metric_slot, config)
        encoded_core_voxels = {_encode_voxel(cell) for cell in core_voxels}
        free_voxels: set[Voxel] = set()
        hit_voxels: set[Voxel] = set()
        ground_voxels: set[Voxel] = set()
        unknown_voxels: set[Voxel] = set()
        occluded_voxels: set[Voxel] = set()
        obstacle_support: dict[Voxel, dict[int, int]] = defaultdict(lambda: defaultdict(int))
        free_support: dict[Voxel, set[int]] = defaultdict(set)
        ray_frames: set[int] = set()
        bearings: list[float] = []
        bearing_by_frame: dict[int, float] = {}
        quality_failures: set[str] = set()
        for observation in accumulation.observations:
            if not observation.ground_model.valid:
                quality_failures.add(f"invalid_ground_frame_{observation.frame_id}")
                continue
            origin = np.asarray(observation.origin_local_xyz, dtype=np.float64)
            endpoints = np.asarray(observation.ray_endpoints_local_xyz, dtype=np.float64)
            if origin.shape != (3,) or not np.isfinite(origin).all():
                quality_failures.add(f"invalid_origin_frame_{observation.frame_id}")
                continue
            if endpoints.ndim != 2 or endpoints.shape[1] != 3:
                quality_failures.add(f"invalid_endpoints_frame_{observation.frame_id}")
                continue
            finite_mask = np.isfinite(endpoints).all(axis=1)
            if not np.all(finite_mask):
                quality_failures.add(f"nonfinite_ray_frame_{observation.frame_id}")
            endpoints = endpoints[finite_mask]
            if not len(endpoints):
                continue
            vectors = endpoints - origin
            distances = np.linalg.norm(vectors, axis=1)
            nonzero_mask = distances > 1e-9
            if not np.all(nonzero_mask):
                quality_failures.add(f"zero_length_ray_frame_{observation.frame_id}")
            endpoints = endpoints[nonzero_mask]
            vectors = vectors[nonzero_mask]
            distances = distances[nonzero_mask]
            if not len(endpoints):
                continue

            endpoint_inside_xy = points_in_polygon(
                endpoints[:, :2],
                metric_slot.core_polygon_local_m,
            )
            endpoint_voxels: dict[int, Voxel] = {}
            for endpoint_index in np.flatnonzero(endpoint_inside_xy):
                endpoint = endpoints[endpoint_index]
                endpoint_voxel = _voxel_for_point(endpoint, config)
                endpoint_voxels[int(endpoint_index)] = endpoint_voxel
                if endpoint[2] <= config.free_near_ground_z_min_m:
                    ground_voxels.add(endpoint_voxel)
                elif config.vehicle_z_min_m <= endpoint[2] <= config.vehicle_z_max_m:
                    hit_voxels.add(endpoint_voxel)
                    obstacle_support[endpoint_voxel][observation.frame_id] += 1
                else:
                    unknown_voxels.add(endpoint_voxel)
                free_voxels.discard(endpoint_voxel)
                occluded_voxels.discard(endpoint_voxel)

            factors = np.maximum(1.0, config.scope_max_distance_m / distances)
            aimed_endpoints = origin + vectors * factors[:, None]
            measured_valid, measured_entry, measured_exit = _clip_segments_to_core(
                origin,
                endpoints,
                metric_slot,
                config,
            )
            measured_spans_core = measured_valid & (measured_exit > measured_entry + 1e-12)
            frame_reached_core = bool(np.any(measured_spans_core))
            if occupied_evidence.strong or not core_voxels.issubset(free_voxels):
                measured_cells = _sample_clipped_voxels(
                    origin,
                    endpoints,
                    measured_spans_core,
                    measured_entry,
                    measured_exit,
                    encoded_core_voxels,
                    config,
                )
                measured_cells.difference_update(endpoint_voxels.values())
                for cell in measured_cells:
                    free_support[cell].add(observation.frame_id)
                free_voxels.update(measured_cells)

                aimed_valid, aimed_entry, aimed_exit = _clip_segments_to_core(
                    origin,
                    aimed_endpoints,
                    metric_slot,
                    config,
                )
                endpoint_passed_core = measured_valid & (measured_exit < 1.0 - 1e-10)
                blocked = (
                    aimed_valid
                    & (aimed_exit > aimed_entry + 1e-12)
                    & (factors > 1.0 + 1e-12)
                    & ~endpoint_passed_core
                )
                aimed_cells = _sample_clipped_voxels(
                    origin,
                    aimed_endpoints,
                    blocked,
                    aimed_entry,
                    aimed_exit,
                    encoded_core_voxels,
                    config,
                )
                occluded_voxels.update(aimed_cells)
            if frame_reached_core:
                ray_frames.add(observation.frame_id)
                bearing = _bearing_deg(origin)
                bearings.append(bearing)
                bearing_by_frame[observation.frame_id] = bearing

        classified_voxels = hit_voxels | ground_voxels | unknown_voxels
        free_voxels.difference_update(classified_voxels)
        occluded_voxels.difference_update(free_voxels | classified_voxels)
        weak_obstacle, weak_points, weak_frames = _weak_obstacle_component(
            obstacle_support,
            config,
        )
        core_hit_point_count = sum(
            sum(frame_counts.values()) for frame_counts in obstacle_support.values()
        )
        core_hit_frames = {
            frame_id
            for frame_counts in obstacle_support.values()
            for frame_id in frame_counts
        }
        unresolved_core_hit = bool(obstacle_support)

        total_core_voxels = len(core_voxels)
        observed_core = (free_voxels | classified_voxels) & core_voxels
        unobserved_voxels = core_voxels - observed_core
        observed_volume_ratio = len(free_voxels & core_voxels) / total_core_voxels
        free_xy = {(x_index, y_index) for x_index, y_index, _ in free_voxels if (x_index, y_index) in core_xy}
        core_ray_coverage = len(free_xy) / max(len(core_xy), 1)
        near_ground_indices = {
            z_index
            for _, _, z_index in core_voxels
            if config.free_near_ground_z_min_m
            <= (z_index + 0.5) * config.voxel_z_m
            <= config.free_near_ground_z_max_m
        }
        near_ground_xy = {
            (x_index, y_index)
            for x_index, y_index, z_index in free_voxels
            if z_index in near_ground_indices and (x_index, y_index) in core_xy
        }
        near_ground_coverage = len(near_ground_xy) / max(len(core_xy), 1)
        unobserved_component_ratio = _largest_component_ratio(
            unobserved_voxels,
            total_core_voxels,
        )
        occlusion_ratio = len(occluded_voxels & core_voxels) / total_core_voxels
        ordered_bearings, viewpoint_count, viewpoint_separation = _viewpoint_summary(
            bearings,
            config.free_min_viewpoint_separation_deg,
        )
        ownership_failures = _OWNERSHIP_FAILURE_CODES & set(occupied_evidence.failures)
        core_supported_weak = _occupied_weak_has_core_support(
            occupied_evidence,
            weak_obstacle,
            config,
        )
        ownership_resolved_by_core = bool(
            occupied_evidence.weak
            and ownership_failures
            and not core_supported_weak
            and not unresolved_core_hit
        )
        occupied_weak_veto = bool(
            occupied_evidence.weak and not ownership_resolved_by_core
        )
        ownership_conflict = bool(ownership_failures and core_supported_weak)
        (
            conflict,
            conflict_voxels,
            conflict_hit_frame_ids,
            conflict_free_frame_ids,
            conflict_viewpoints,
            conflict_viewpoint_separation,
        ) = _occupied_free_conflict(
            occupied_evidence,
            obstacle_support,
            free_support,
            bearing_by_frame,
            config,
        )
        gates = _free_gates(
            ray_frame_count=len(ray_frames),
            viewpoint_count=viewpoint_count,
            viewpoint_separation_deg=viewpoint_separation,
            observed_volume_ratio=observed_volume_ratio,
            near_ground_bev_coverage=near_ground_coverage,
            unobserved_component_ratio=unobserved_component_ratio,
            occlusion_ratio=occlusion_ratio,
            weak_obstacle=weak_obstacle,
            unresolved_core_hit=unresolved_core_hit,
            occupied_weak_veto=occupied_weak_veto,
            ownership_conflict=ownership_conflict,
            conflict=conflict,
            config=config,
        )
        failures = _failure_codes(gates)
        positive_geometry = all(gate.passed for gate in gates[:7])
        strength = sum(gate.passed for gate in gates) / len(gates)
        details = FreeSpaceDetails(
            free_voxels=tuple(sorted(free_voxels & core_voxels)),
            hit_voxels=tuple(sorted(hit_voxels)),
            ground_voxels=tuple(sorted(ground_voxels)),
            unknown_voxels=tuple(sorted(unknown_voxels)),
            occluded_voxels=tuple(sorted(occluded_voxels & core_voxels)),
            unobserved_voxels=tuple(sorted(unobserved_voxels)),
            viewpoint_bearings_deg=ordered_bearings,
            total_core_voxels=total_core_voxels,
            weak_obstacle_point_count=weak_points,
            weak_obstacle_frame_count=weak_frames,
            core_hit_point_count=core_hit_point_count,
            core_hit_voxel_count=len(obstacle_support),
            core_hit_frame_count=len(core_hit_frames),
            conflict_voxels=conflict_voxels,
            conflict_hit_frames=conflict_hit_frame_ids,
            conflict_free_frames=conflict_free_frame_ids,
            conflict_hit_frame_count=len(conflict_hit_frame_ids),
            conflict_free_frame_count=len(conflict_free_frame_ids),
            conflict_viewpoint_count=conflict_viewpoints,
            conflict_viewpoint_separation_deg=conflict_viewpoint_separation,
            quality_failures=tuple(sorted(quality_failures)),
        )
        return FreeEvidence(
            strong=not failures,
            strength=float(strength),
            ray_frame_count=len(ray_frames),
            viewpoint_count=viewpoint_count,
            observed_volume_ratio=float(observed_volume_ratio),
            core_ray_coverage=float(core_ray_coverage),
            near_ground_bev_coverage=float(near_ground_coverage),
            unobserved_component_ratio=float(unobserved_component_ratio),
            viewpoint_separation_deg=float(viewpoint_separation),
            occlusion_ratio=float(occlusion_ratio),
            weak_obstacle=weak_obstacle,
            unresolved_core_hit=unresolved_core_hit,
            conflict=conflict,
            weak_ownership_resolved=ownership_resolved_by_core,
            positive_geometry=positive_geometry,
            gate_results=gates,
            failures=failures,
            details=details,
        )
    except Exception:
        return FreeEvidence(failures=("free_evaluation_error",))
