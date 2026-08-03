from __future__ import annotations

from typing import Mapping, Protocol, Sequence

import numpy as np

from .config import Hybrid3DConfig
from .contracts import (
    FrameObservation,
    FrameRecord,
    KnownSlot,
    ScopeEvidence,
    ScopeStatus,
    SlotAccumulation,
)
from .geometry import map_xy_to_slot_m, metric_slot
from .ground import fit_ground_model, normalize_z, origin_height
from .io import FrameLoadError
from .raycasting import segment_aabb_mask


class PointProvider(Protocol):
    def load(self, frame_id: int) -> np.ndarray: ...


def select_slot_frames(
    slot: KnownSlot,
    scope: ScopeEvidence,
    frames: Sequence[FrameRecord],
    map_units_per_meter: float,
    config: Hybrid3DConfig,
) -> tuple[int, tuple[int, ...]]:
    if scope.slot_id != slot.slot_id:
        raise ValueError("scope slot_id does not match slot")
    if scope.scope_status is ScopeStatus.OUT_OF_ROUTE:
        raise ValueError("cannot accumulate an out-of-route slot")
    frame_by_id = {frame.frame_id: frame for frame in frames}
    reached_ids = sorted(set(scope.crossing_frames) | set(scope.hit_frames))
    reached = [frame_by_id[frame_id] for frame_id in reached_ids if frame_id in frame_by_id]
    if not reached:
        raise ValueError("scope contains no available frame that reached the slot")
    center = np.asarray(slot.center_map, dtype=np.float64)
    anchor_record = min(
        reached,
        key=lambda frame: (
            float(np.linalg.norm(np.asarray([frame.map_x, frame.map_y]) - center) / map_units_per_meter),
            frame.frame_id,
        ),
    )
    anchor = anchor_record.frame_id
    available = set(frame_by_id)
    stride = max(int(config.frame_stride), 1)
    sampled = list(range(anchor - config.window_before, anchor + config.window_after + 1, stride))
    if anchor not in sampled:
        sampled.append(anchor)
    selected = tuple(sorted(frame_id for frame_id in sampled if frame_id in available))
    if not selected:
        selected = (anchor,)
    return anchor, selected


def split_frame_ids(frame_ids: Sequence[int]) -> dict[str, tuple[int, ...]]:
    ordered = tuple(sorted(dict.fromkeys(int(frame_id) for frame_id in frame_ids)))
    midpoint = len(ordered) // 2
    return {
        "first_half": ordered[:midpoint],
        "second_half": ordered[midpoint:],
        "odd_index": ordered[0::2],
        "even_index": ordered[1::2],
    }


def _relevant_ray_mask(
    origin: np.ndarray,
    endpoints: np.ndarray,
    minimum: np.ndarray,
    maximum: np.ndarray,
    max_range_m: float,
) -> np.ndarray:
    measured = segment_aabb_mask(origin, endpoints, minimum, maximum)
    vectors = endpoints - origin
    distances = np.linalg.norm(vectors, axis=1)
    valid = distances > 1e-9
    factors = np.ones(len(endpoints), dtype=np.float64)
    factors[valid] = np.maximum(1.0, max_range_m / distances[valid])
    extended = origin + vectors * factors[:, None]
    aimed_toward = segment_aabb_mask(origin, extended, minimum, maximum)
    return measured | aimed_toward


def _readonly(array: np.ndarray) -> np.ndarray:
    result = np.ascontiguousarray(array)
    result.setflags(write=False)
    return result


def build_slot_accumulation(
    slot: KnownSlot,
    scope: ScopeEvidence,
    frames: Sequence[FrameRecord],
    provider: PointProvider,
    map_units_per_meter: float,
    config: Hybrid3DConfig,
) -> SlotAccumulation:
    config.validate()
    anchor, selected = select_slot_frames(slot, scope, frames, map_units_per_meter, config)
    frame_by_id: Mapping[int, FrameRecord] = {frame.frame_id: frame for frame in frames}
    converted_slot = metric_slot(slot, map_units_per_meter)
    roi_extra_m = config.box_hypothesis_config().roi_extra_m
    roi_min = converted_slot.margin_polygon_local_m.min(axis=0) - roi_extra_m
    roi_max = converted_slot.margin_polygon_local_m.max(axis=0) + roi_extra_m
    ray_min_xy = converted_slot.margin_polygon_local_m.min(axis=0) - config.scope_prism_expand_m
    ray_max_xy = converted_slot.margin_polygon_local_m.max(axis=0) + config.scope_prism_expand_m
    ray_min = np.asarray([ray_min_xy[0], ray_min_xy[1], -0.20], dtype=np.float64)
    ray_max = np.asarray([ray_max_xy[0], ray_max_xy[1], config.vehicle_z_max_m + 0.50], dtype=np.float64)

    point_batches: list[np.ndarray] = []
    point_frame_batches: list[np.ndarray] = []
    observations: list[FrameObservation] = []
    excluded: list[tuple[int, tuple[str, ...]]] = []
    for frame_id in selected:
        record = frame_by_id[frame_id]
        try:
            source = provider.load(frame_id)
        except FrameLoadError as exc:
            excluded.append((frame_id, (exc.reason,)))
            continue
        local_xy = map_xy_to_slot_m(source[:, :2], converted_slot)
        local_xyz = np.column_stack([local_xy, source[:, 2]])
        morphology_mask = (
            (local_xy[:, 0] >= roi_min[0])
            & (local_xy[:, 0] <= roi_max[0])
            & (local_xy[:, 1] >= roi_min[1])
            & (local_xy[:, 1] <= roi_max[1])
        )
        morphology_xyz = local_xyz[morphology_mask]
        ground_model = fit_ground_model(morphology_xyz, config)
        if not ground_model.valid:
            excluded.append((frame_id, ("invalid_ground_model",)))
            continue

        morphology_z = normalize_z(morphology_xyz, ground_model)
        morphology = np.column_stack(
            [
                morphology_xyz[:, :2],
                morphology_z,
                source[morphology_mask, 3],
            ]
        )
        point_batches.append(morphology)
        point_frame_batches.append(np.full(len(morphology), frame_id, dtype=np.int64))

        origin_xy = map_xy_to_slot_m(
            np.asarray([[record.map_x, record.map_y]], dtype=np.float64), converted_slot
        )[0]
        origin_raw = np.asarray([origin_xy[0], origin_xy[1], 0.0], dtype=np.float64)
        origin = np.asarray(
            [origin_xy[0], origin_xy[1], origin_height(origin_raw, ground_model)],
            dtype=np.float64,
        )
        endpoint_z = normalize_z(local_xyz, ground_model)
        endpoints = np.column_stack([local_xyz[:, :2], endpoint_z])
        relevant = _relevant_ray_mask(origin, endpoints, ray_min, ray_max, config.scope_max_distance_m)
        relevant_endpoints = endpoints[relevant]

        observations.append(
            FrameObservation(
                frame_id=frame_id,
                origin_local_xyz=_readonly(origin),
                points_local_xyzi=_readonly(morphology),
                ray_endpoints_local_xyz=_readonly(relevant_endpoints),
                ground_model=ground_model,
            )
        )

    if point_batches:
        accumulated_points = _readonly(np.vstack(point_batches).astype(np.float64, copy=False))
        point_frame_ids = _readonly(np.concatenate(point_frame_batches).astype(np.int64, copy=False))
    else:
        accumulated_points = _readonly(np.empty((0, 4), dtype=np.float64))
        point_frame_ids = _readonly(np.empty(0, dtype=np.int64))
    return SlotAccumulation(
        slot_id=slot.slot_id,
        anchor_frame=anchor,
        selected_frames=selected,
        points_local_xyzi=accumulated_points,
        point_frame_ids=point_frame_ids,
        observations=tuple(observations),
        excluded_frames=tuple(sorted(excluded)),
    )
