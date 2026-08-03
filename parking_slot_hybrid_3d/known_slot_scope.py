from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, Sequence

import numpy as np
from scipy.spatial import cKDTree

from parking_slot_box_scoring.geometry import points_in_polygon

from .config import Hybrid3DConfig
from .contracts import FrameRecord, KnownSlot, MetricSlot, ScopeEvidence, ScopeStatus
from .geometry import map_xy_to_slot_m, metric_slot
from .io import FrameLoadError
from .raycasting import clip_segment_to_convex_prism, segment_aabb_mask, traverse_voxels_3d


class PointProvider(Protocol):
    def load(self, frame_id: int) -> np.ndarray: ...


class AngularRayIndex:
    """Bearing-sorted endpoints for conservative per-slot ray prefiltering.

    The returned set is a superset of rays that can intersect the expanded
    slot rectangle.  Exact prism tests remain authoritative downstream.
    """

    def __init__(self, origin_map_xy: np.ndarray, endpoints_map_xy: np.ndarray) -> None:
        origin = np.asarray(origin_map_xy, dtype=np.float64)
        endpoints = np.asarray(endpoints_map_xy, dtype=np.float64)
        if origin.shape != (2,):
            raise ValueError("origin_map_xy must have shape [2]")
        if endpoints.ndim != 2 or endpoints.shape[1] != 2:
            raise ValueError("endpoints_map_xy must have shape [N, 2]")
        self.origin_map_xy = origin
        self.endpoint_count = len(endpoints)
        directions = endpoints - origin
        angles = np.mod(np.arctan2(directions[:, 1], directions[:, 0]), 2.0 * np.pi)
        order = np.argsort(angles, kind="stable")
        ordered_angles = angles[order]
        self._ordered_angles = np.concatenate(
            [ordered_angles, ordered_angles + 2.0 * np.pi]
        )
        self._ordered_indices = np.concatenate([order, order])

    def candidate_indices(
        self,
        slot: MetricSlot,
        expansion_m: float,
    ) -> np.ndarray:
        if expansion_m < 0.0 or not np.isfinite(expansion_m):
            raise ValueError("expansion_m must be finite and non-negative")
        if self.endpoint_count == 0:
            return np.empty(0, dtype=np.int64)
        origin_local = map_xy_to_slot_m(self.origin_map_xy.reshape(1, 2), slot)[0]
        minimum = slot.margin_polygon_local_m.min(axis=0) - expansion_m
        maximum = slot.margin_polygon_local_m.max(axis=0) + expansion_m
        epsilon = 1e-12
        if np.all(origin_local >= minimum - epsilon) and np.all(
            origin_local <= maximum + epsilon
        ):
            return np.arange(self.endpoint_count, dtype=np.int64)

        corners_local = _rectangle_from_bounds(minimum, maximum)
        corners_map = slot.center_map + slot.map_units_per_meter * (
            corners_local[:, 0:1] * slot.long_axis_map
            + corners_local[:, 1:2] * slot.short_axis_map
        )
        center_direction = slot.center_map - self.origin_map_xy
        center_angle = float(np.arctan2(center_direction[1], center_direction[0]))
        corner_directions = corners_map - self.origin_map_xy
        corner_angles = np.arctan2(corner_directions[:, 1], corner_directions[:, 0])
        deltas = np.arctan2(
            np.sin(corner_angles - center_angle),
            np.cos(corner_angles - center_angle),
        )
        lower = float((center_angle + float(deltas.min()) - epsilon) % (2.0 * np.pi))
        width = float(deltas.max() - deltas.min() + 2.0 * epsilon)
        if width >= 2.0 * np.pi:
            return np.arange(self.endpoint_count, dtype=np.int64)
        upper = lower + width
        first = int(np.searchsorted(self._ordered_angles, lower, side="left"))
        last = int(np.searchsorted(self._ordered_angles, upper, side="right"))
        return np.sort(self._ordered_indices[first:last].astype(np.int64, copy=False))


@dataclass
class _ScopeAccumulator:
    near_frames: set[int] = field(default_factory=set)
    crossing_frames: set[int] = field(default_factory=set)
    hit_frames: set[int] = field(default_factory=set)
    missing_frames: set[int] = field(default_factory=set)
    covered_core_cells: set[tuple[int, int]] = field(default_factory=set)
    camera_observable: bool = False


def _rectangle_from_bounds(minimum: np.ndarray, maximum: np.ndarray) -> np.ndarray:
    return np.asarray(
        [
            [minimum[0], minimum[1]],
            [maximum[0], minimum[1]],
            [maximum[0], maximum[1]],
            [minimum[0], maximum[1]],
        ],
        dtype=np.float64,
    )


def _core_grid_cells(polygon: np.ndarray, grid_m: float) -> set[tuple[int, int]]:
    minimum = polygon.min(axis=0)
    maximum = polygon.max(axis=0)
    x_indices = np.arange(np.floor(minimum[0] / grid_m), np.floor(maximum[0] / grid_m) + 1, dtype=np.int64)
    y_indices = np.arange(np.floor(minimum[1] / grid_m), np.floor(maximum[1] / grid_m) + 1, dtype=np.int64)
    cells = [(int(x), int(y)) for x in x_indices for y in y_indices]
    if not cells:
        return set()
    centers = np.asarray([[(x + 0.5) * grid_m, (y + 0.5) * grid_m] for x, y in cells], dtype=np.float64)
    inside = points_in_polygon(centers, polygon)
    return {cell for cell, keep in zip(cells, inside) if bool(keep)}


def _camera_frame_is_observable(frame: FrameRecord) -> bool:
    path: Path | None = frame.camera_image_path
    return bool(frame.camera_match_valid and path is not None and path.exists())


class KnownSlotObservationScopeEvaluator:
    def __init__(self, config: Hybrid3DConfig, map_units_per_meter: float) -> None:
        config.validate()
        if not np.isfinite(map_units_per_meter) or map_units_per_meter <= 0.0:
            raise ValueError("map_units_per_meter must be positive and finite")
        self.config = config
        self.map_units_per_meter = float(map_units_per_meter)

    def _update_slot_from_points(
        self,
        accumulator: _ScopeAccumulator,
        slot: MetricSlot,
        frame: FrameRecord,
        points_map_xyzi: np.ndarray,
        valid_core_cells: set[tuple[int, int]],
    ) -> None:
        origin_xy = map_xy_to_slot_m(np.asarray([[frame.map_x, frame.map_y]], dtype=np.float64), slot)[0]
        origin = np.asarray([origin_xy[0], origin_xy[1], 0.0], dtype=np.float64)
        endpoints_xy = map_xy_to_slot_m(points_map_xyzi[:, :2], slot)
        endpoints = np.column_stack([endpoints_xy, points_map_xyzi[:, 2]])

        expanded_min_xy = slot.margin_polygon_local_m.min(axis=0) - self.config.scope_prism_expand_m
        expanded_max_xy = slot.margin_polygon_local_m.max(axis=0) + self.config.scope_prism_expand_m
        prism_min = np.asarray([expanded_min_xy[0], expanded_min_xy[1], self.config.scope_raw_z_min_m])
        prism_max = np.asarray([expanded_max_xy[0], expanded_max_xy[1], self.config.scope_raw_z_max_m])
        crossing_mask = segment_aabb_mask(origin, endpoints, prism_min, prism_max)
        hit_mask = np.all((endpoints >= prism_min) & (endpoints <= prism_max), axis=1)
        if np.any(crossing_mask):
            accumulator.crossing_frames.add(frame.frame_id)
        if np.any(hit_mask):
            accumulator.hit_frames.add(frame.frame_id)

        if not valid_core_cells or valid_core_cells.issubset(accumulator.covered_core_cells):
            return
        core_min_xy = slot.core_polygon_local_m.min(axis=0)
        core_max_xy = slot.core_polygon_local_m.max(axis=0)
        core_min = np.asarray([core_min_xy[0], core_min_xy[1], self.config.scope_raw_z_min_m])
        core_max = np.asarray([core_max_xy[0], core_max_xy[1], self.config.scope_raw_z_max_m])
        core_candidates = np.flatnonzero(segment_aabb_mask(origin, endpoints, core_min, core_max))
        for point_index in core_candidates:
            clipped = clip_segment_to_convex_prism(
                origin,
                endpoints[point_index],
                slot.core_polygon_local_m,
                (self.config.scope_raw_z_min_m, self.config.scope_raw_z_max_m),
            )
            if clipped is None:
                continue
            start = np.asarray([clipped.entry_xyz[0], clipped.entry_xyz[1], 0.1])
            end = np.asarray([clipped.exit_xyz[0], clipped.exit_xyz[1], 0.1])
            traversed = traverse_voxels_3d(
                start,
                end,
                (self.config.scope_grid_m, self.config.scope_grid_m, 1.0),
            )
            accumulator.covered_core_cells.update((cell[0], cell[1]) for cell in traversed if (cell[0], cell[1]) in valid_core_cells)
            if valid_core_cells.issubset(accumulator.covered_core_cells):
                break

    def evaluate(
        self,
        slots: Sequence[KnownSlot],
        frames: Sequence[FrameRecord],
        provider: PointProvider,
    ) -> list[ScopeEvidence]:
        ordered_slots = sorted(slots, key=lambda slot: slot.slot_id)
        if not ordered_slots:
            return []
        metric_slots = [metric_slot(slot, self.map_units_per_meter) for slot in ordered_slots]
        slot_centers = np.asarray([slot.center_map for slot in ordered_slots], dtype=np.float64)
        slot_tree = cKDTree(slot_centers)
        accumulators = [_ScopeAccumulator() for _ in ordered_slots]
        core_cells = [
            _core_grid_cells(slot.core_polygon_local_m, self.config.scope_grid_m)
            for slot in metric_slots
        ]
        radius_map = self.config.scope_max_distance_m * self.map_units_per_meter

        for frame in sorted(frames, key=lambda item: item.frame_id):
            candidate_indices = sorted(
                int(index)
                for index in slot_tree.query_ball_point(np.asarray([frame.map_x, frame.map_y]), radius_map)
            )
            if not candidate_indices:
                continue
            for slot_index in candidate_indices:
                accumulators[slot_index].near_frames.add(frame.frame_id)
                accumulators[slot_index].camera_observable |= _camera_frame_is_observable(frame)
            try:
                points = provider.load(frame.frame_id)
            except FrameLoadError:
                for slot_index in candidate_indices:
                    accumulators[slot_index].missing_frames.add(frame.frame_id)
                continue
            ray_index = AngularRayIndex(
                np.asarray([frame.map_x, frame.map_y], dtype=np.float64),
                points[:, :2],
            )
            for slot_index in candidate_indices:
                point_indices = ray_index.candidate_indices(
                    metric_slots[slot_index],
                    self.config.scope_prism_expand_m,
                )
                self._update_slot_from_points(
                    accumulators[slot_index],
                    metric_slots[slot_index],
                    frame,
                    points[point_indices],
                    core_cells[slot_index],
                )

        results: list[ScopeEvidence] = []
        for slot, accumulator, valid_cells in zip(ordered_slots, accumulators, core_cells):
            ray_or_hit_frames = accumulator.crossing_frames | accumulator.hit_frames
            coverage = len(accumulator.covered_core_cells) / max(len(valid_cells), 1)
            in_route = (
                len(accumulator.near_frames) >= self.config.scope_min_near_frames
                and len(ray_or_hit_frames) >= self.config.scope_min_ray_frames
                and coverage >= self.config.scope_core_coverage_min
            )
            reasons: list[str] = []
            if in_route:
                status = ScopeStatus.IN_ROUTE
                reasons.append("scope_gate_passed")
            elif accumulator.near_frames and (ray_or_hit_frames or accumulator.missing_frames):
                status = ScopeStatus.PARTIAL_ROUTE
                if accumulator.missing_frames:
                    reasons.append("missing_near_route_frames")
                if len(accumulator.near_frames) < self.config.scope_min_near_frames:
                    reasons.append("insufficient_near_frames")
                if len(ray_or_hit_frames) < self.config.scope_min_ray_frames:
                    reasons.append("insufficient_ray_frames")
                if coverage < self.config.scope_core_coverage_min:
                    reasons.append("low_core_ray_coverage")
            else:
                status = ScopeStatus.OUT_OF_ROUTE
                reasons.append("no_frame_within_range" if not accumulator.near_frames else "no_ray_or_hit_reached_slot")
            results.append(
                ScopeEvidence(
                    slot_id=slot.slot_id,
                    scope_status=status,
                    near_frames=tuple(sorted(accumulator.near_frames)),
                    crossing_frames=tuple(sorted(accumulator.crossing_frames)),
                    hit_frames=tuple(sorted(accumulator.hit_frames)),
                    missing_frames=tuple(sorted(accumulator.missing_frames)),
                    core_ray_coverage=float(coverage),
                    agent_observable=accumulator.camera_observable,
                    reasons=tuple(sorted(reasons)),
                )
            )
        return results
