from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np

from parking_slot_box_scoring.box_hypotheses import CandidateBox
from parking_slot_box_scoring.geometry import (
    box_polygon,
    points_in_polygon,
    points_in_rotated_box,
    polygon_overlap_ratio,
)
from parking_slot_box_scoring.scoring import linearity_penalty

from .accumulation import split_frame_ids
from .config import Hybrid3DConfig
from .contracts import (
    HeightLayerEvidence,
    MetricSlot,
    SlotAccumulation,
    SplitEvidence,
    ThreeDEvidence,
)
from .geometry import inset_local_rectangle


def _validate_inputs(
    accumulation: SlotAccumulation,
    candidate_box: CandidateBox,
    config: Hybrid3DConfig,
) -> None:
    config.validate()
    points = np.asarray(accumulation.points_local_xyzi)
    frame_ids = np.asarray(accumulation.point_frame_ids)
    if points.ndim != 2 or points.shape[1] != 4:
        raise ValueError("points_local_xyzi must have shape [N, 4]")
    if frame_ids.ndim != 1 or len(frame_ids) != len(points):
        raise ValueError("point_frame_ids must have one entry per accumulated point")
    if not np.isfinite(points).all():
        raise ValueError("accumulated points contain non-finite values")
    if np.asarray(candidate_box.center).shape != (2,):
        raise ValueError("candidate box center must have shape [2]")
    box_values = np.asarray(
        [candidate_box.center[0], candidate_box.center[1], candidate_box.yaw, candidate_box.length, candidate_box.width],
        dtype=np.float64,
    )
    if not np.isfinite(box_values).all() or candidate_box.length <= 0.0 or candidate_box.width <= 0.0:
        raise ValueError("candidate box must have finite positive geometry")


def _candidate_coordinates(points_xy: np.ndarray, candidate_box: CandidateBox) -> np.ndarray:
    long_axis = np.asarray([math.cos(candidate_box.yaw), math.sin(candidate_box.yaw)])
    short_axis = np.asarray([-math.sin(candidate_box.yaw), math.cos(candidate_box.yaw)])
    delta = points_xy - np.asarray(candidate_box.center, dtype=np.float64)
    return np.column_stack([delta @ long_axis, delta @ short_axis])


def _candidate_vehicle_mask(
    points: np.ndarray,
    candidate_box: CandidateBox,
    config: Hybrid3DConfig,
) -> np.ndarray:
    in_box = points_in_rotated_box(
        points[:, :2],
        candidate_box.center,
        candidate_box.yaw,
        candidate_box.length,
        candidate_box.width,
    )
    return in_box & (points[:, 2] >= config.vehicle_z_min_m) & (points[:, 2] <= config.vehicle_z_max_m)


def _voxel_keys(
    points_xyz: np.ndarray,
    candidate_box: CandidateBox,
    config: Hybrid3DConfig,
) -> np.ndarray:
    if len(points_xyz) == 0:
        return np.empty((0, 3), dtype=np.int64)
    local_xy = _candidate_coordinates(points_xyz[:, :2], candidate_box)
    x_index = np.floor((local_xy[:, 0] + candidate_box.length / 2.0) / config.voxel_xy_m)
    y_index = np.floor((local_xy[:, 1] + candidate_box.width / 2.0) / config.voxel_xy_m)
    z_index = np.floor((points_xyz[:, 2] - config.vehicle_z_min_m) / config.voxel_z_m)
    return np.column_stack([x_index, y_index, z_index]).astype(np.int64)


_KEY_BITS = 21
_KEY_LIMIT = 1 << _KEY_BITS


def _key_set(keys: np.ndarray, columns: int) -> set[int]:
    if len(keys) == 0:
        return set()
    values = np.asarray(keys[:, :columns], dtype=np.int64)
    encoded_values = np.where(values >= 0, values * 2, -values * 2 - 1)
    if np.any(encoded_values >= _KEY_LIMIT):
        raise ValueError("voxel index exceeds deterministic key range")
    packed = np.zeros(len(encoded_values), dtype=np.int64)
    for column in range(columns):
        packed = (packed << _KEY_BITS) | encoded_values[:, column]
    return {int(value) for value in np.unique(packed)}


def _bev_coverage(
    points_xyz: np.ndarray,
    candidate_box: CandidateBox,
    config: Hybrid3DConfig,
) -> float:
    if len(points_xyz) == 0:
        return 0.0
    xy_cells = _key_set(_voxel_keys(points_xyz, candidate_box, config), 2)
    total_x = max(int(math.ceil(candidate_box.length / config.voxel_xy_m)), 1)
    total_y = max(int(math.ceil(candidate_box.width / config.voxel_xy_m)), 1)
    return float(np.clip(len(xy_cells) / (total_x * total_y), 0.0, 1.0))


def _weak_subset_gate(
    points_xyz: np.ndarray,
    candidate_box: CandidateBox,
    config: Hybrid3DConfig,
    *,
    voxel_count: int | None = None,
) -> bool:
    minimum_points = max(
        config.occupied_layer_min_points_per_frame,
        int(math.ceil(config.occupied_min_points / 4.0)),
    )
    minimum_voxels = max(2, int(math.ceil(config.occupied_min_voxels / 2.0)))
    if len(points_xyz) < minimum_points:
        return False
    if voxel_count is None:
        voxel_count = len(_key_set(_voxel_keys(points_xyz, candidate_box, config), 3))
    if voxel_count < minimum_voxels:
        return False
    return float(np.quantile(points_xyz[:, 2], 0.95)) >= config.low_height_veto_z95_m


def _split_partitions(
    accumulation: SlotAccumulation,
    split: str,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    valid_frame_ids = tuple(observation.frame_id for observation in accumulation.observations)
    partitions = split_frame_ids(valid_frame_ids)
    if split == "chronological":
        return partitions["first_half"], partitions["second_half"]
    if split == "parity":
        return partitions["odd_index"], partitions["even_index"]
    raise ValueError(f"unknown split: {split}")


def score_split_consistency(
    accumulation: SlotAccumulation,
    candidate_box: CandidateBox,
    split: str,
    config: Hybrid3DConfig,
) -> SplitEvidence:
    _validate_inputs(accumulation, candidate_box, config)
    left_frames, right_frames = _split_partitions(accumulation, split)
    points = np.asarray(accumulation.points_local_xyzi, dtype=np.float64)
    point_frame_ids = np.asarray(accumulation.point_frame_ids, dtype=np.int64)
    vehicle_mask = _candidate_vehicle_mask(points, candidate_box, config)
    left_mask = vehicle_mask & np.isin(point_frame_ids, left_frames)
    right_mask = vehicle_mask & np.isin(point_frame_ids, right_frames)
    left_points = points[left_mask, :3]
    right_points = points[right_mask, :3]

    left_voxels = _key_set(_voxel_keys(left_points, candidate_box, config), 3)
    right_voxels = _key_set(_voxel_keys(right_points, candidate_box, config), 3)
    left_bev = {key >> _KEY_BITS for key in left_voxels}
    right_bev = {key >> _KEY_BITS for key in right_voxels}
    bev_union = left_bev | right_bev
    support_jaccard = len(left_bev & right_bev) / len(bev_union) if bev_union else 0.0
    smaller_voxel_count = min(len(left_voxels), len(right_voxels))
    voxel_overlap = (
        len(left_voxels & right_voxels) / smaller_voxel_count
        if smaller_voxel_count > 0
        else 0.0
    )
    left_z95 = float(np.quantile(left_points[:, 2], 0.95)) if len(left_points) else 0.0
    right_z95 = float(np.quantile(right_points[:, 2], 0.95)) if len(right_points) else 0.0
    z95_difference = abs(left_z95 - right_z95) if len(left_points) and len(right_points) else None
    left_weak = _weak_subset_gate(
        left_points,
        candidate_box,
        config,
        voxel_count=len(left_voxels),
    )
    right_weak = _weak_subset_gate(
        right_points,
        candidate_box,
        config,
        voxel_count=len(right_voxels),
    )
    agreement = bool(
        left_weak
        and right_weak
        and support_jaccard > 0.0
        and voxel_overlap > 0.0
        and z95_difference is not None
        and math.isfinite(z95_difference)
    )
    if agreement:
        height_range = max(config.vehicle_z_max_m - config.vehicle_z_min_m, 1e-9)
        z_similarity = float(np.clip(1.0 - z95_difference / height_range, 0.0, 1.0))
        consistency = float((support_jaccard + voxel_overlap + z_similarity) / 3.0)
    else:
        consistency = 0.0
    return SplitEvidence(
        name=split,
        left_frames=left_frames,
        right_frames=right_frames,
        left_point_count=len(left_points),
        right_point_count=len(right_points),
        left_voxel_count=len(left_voxels),
        right_voxel_count=len(right_voxels),
        support_jaccard=float(support_jaccard),
        voxel_overlap_ratio=float(voxel_overlap),
        z95_difference_m=None if z95_difference is None else float(z95_difference),
        left_clears_weak_gate=left_weak,
        right_clears_weak_gate=right_weak,
        agreement=agreement,
        consistency=consistency,
    )


def _height_layers(
    vehicle_points: np.ndarray,
    vehicle_frame_ids: np.ndarray,
    candidate_box: CandidateBox,
    config: Hybrid3DConfig,
) -> tuple[HeightLayerEvidence, ...]:
    result: list[HeightLayerEvidence] = []
    final_index = len(config.height_layers_m) - 1
    for index, (lower, upper) in enumerate(config.height_layers_m):
        upper_mask = vehicle_points[:, 2] <= upper if index == final_index else vehicle_points[:, 2] < upper
        layer_mask = (vehicle_points[:, 2] >= lower) & upper_mask
        layer_points = vehicle_points[layer_mask]
        layer_frame_ids = vehicle_frame_ids[layer_mask]
        counts = tuple(
            (int(frame_id), int(np.sum(layer_frame_ids == frame_id)))
            for frame_id in sorted(set(layer_frame_ids.tolist()))
        )
        supporting_frames = tuple(
            frame_id
            for frame_id, count in counts
            if count >= config.occupied_layer_min_points_per_frame
        )
        result.append(
            HeightLayerEvidence(
                z_min_m=float(lower),
                z_max_m=float(upper),
                point_count=len(layer_points),
                supporting_frames=supporting_frames,
                per_frame_counts=counts,
                bev_coverage=_bev_coverage(layer_points, candidate_box, config),
                supported=len(supporting_frames) >= config.occupied_layer_min_frames,
            )
        )
    return tuple(result)


def _metric_polygon_in_subject(adjacent: MetricSlot, subject: MetricSlot) -> np.ndarray:
    adjacent_map = (
        adjacent.center_map
        + adjacent.map_units_per_meter
        * (
            adjacent.core_polygon_local_m[:, 0:1] * adjacent.long_axis_map
            + adjacent.core_polygon_local_m[:, 1:2] * adjacent.short_axis_map
        )
    )
    delta_m = (adjacent_map - subject.center_map) / subject.map_units_per_meter
    return np.column_stack([delta_m @ subject.long_axis_map, delta_m @ subject.short_axis_map])


def _adjacent_overlap(
    slot: MetricSlot,
    candidate_polygon: np.ndarray,
    slots_by_id: Mapping[str, MetricSlot] | None,
) -> float:
    if not slots_by_id:
        return 0.0
    ratios: list[float] = []
    for adjacent_id in slot.adjacent_slots:
        adjacent = slots_by_id.get(adjacent_id)
        if adjacent is None:
            continue
        adjacent_polygon = _metric_polygon_in_subject(adjacent, slot)
        ratios.append(polygon_overlap_ratio(candidate_polygon, adjacent_polygon))
    return max(ratios, default=0.0)


def _pca_shape(points_xyz: np.ndarray, max_points: int) -> tuple[float, float]:
    if len(points_xyz) < 5:
        return 0.0, 0.0
    sampled = points_xyz
    if len(sampled) > max_points:
        indices = np.linspace(0, len(sampled) - 1, max_points, dtype=np.int64)
        sampled = sampled[indices]
    centered = sampled - sampled.mean(axis=0)
    covariance = centered.T @ centered / max(len(centered) - 1, 1)
    eigenvalues = np.maximum(np.linalg.eigvalsh(covariance), 0.0)
    largest = float(eigenvalues[-1])
    if largest <= 1e-12:
        return 0.0, 0.0
    linearity = float(np.clip((eigenvalues[-1] - eigenvalues[-2]) / largest, 0.0, 1.0))
    planarity = float(np.clip((eigenvalues[-2] - eigenvalues[-3]) / largest, 0.0, 1.0))
    return linearity, planarity


def _robust_shape(
    points_xyz: np.ndarray,
    point_frame_ids: np.ndarray,
    candidate_box: CandidateBox,
    config: Hybrid3DConfig,
) -> tuple[float, float, float, int, float]:
    """Return outlier-resistant PCA linearity and horizontal 5--95% extents."""

    if len(points_xyz) < 5:
        return 0.0, 0.0, 0.0, 0, 0.0
    local_xy = _candidate_coordinates(points_xyz[:, :2], candidate_box)
    local_xyz = np.column_stack([local_xy, points_xyz[:, 2]])
    trim = config.occupied_pillar_trim_quantile
    lower, upper = np.quantile(local_xy, (trim, 1.0 - trim), axis=0)
    inlier_mask = np.all(
        (local_xy >= lower - 1e-12) & (local_xy <= upper + 1e-12),
        axis=1,
    )
    inliers = local_xyz[inlier_mask]
    inlier_frame_ids = point_frame_ids[inlier_mask]
    robust_linearity, _ = _pca_shape(inliers, config.pca_max_points)
    supported_frames = sum(
        int(np.sum(inlier_frame_ids == frame_id))
        >= config.occupied_layer_min_points_per_frame
        for frame_id in np.unique(inlier_frame_ids)
    )
    return (
        robust_linearity,
        float(np.ptp(inliers[:, 0])) if len(inliers) else 0.0,
        float(np.ptp(inliers[:, 1])) if len(inliers) else 0.0,
        int(supported_frames),
        float(len(inliers) / len(local_xyz)),
    )


def _quantiles(z_values: np.ndarray) -> tuple[float, float, float, float, float]:
    if len(z_values) == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    values = np.quantile(z_values, (0.50, 0.75, 0.90, 0.95))
    return (
        float(values[0]),
        float(values[1]),
        float(values[2]),
        float(values[3]),
        float(z_values.max() - z_values.min()),
    )


def extract_3d_evidence(
    metric_slot: MetricSlot,
    candidate_box: CandidateBox,
    accumulation: SlotAccumulation,
    config: Hybrid3DConfig,
    *,
    slots_by_id: Mapping[str, MetricSlot] | None = None,
) -> ThreeDEvidence:
    _validate_inputs(accumulation, candidate_box, config)
    if accumulation.slot_id != metric_slot.slot_id:
        raise ValueError("accumulation slot_id does not match metric slot")
    points = np.asarray(accumulation.points_local_xyzi, dtype=np.float64)
    point_frame_ids = np.asarray(accumulation.point_frame_ids, dtype=np.int64)
    vehicle_mask = _candidate_vehicle_mask(points, candidate_box, config)
    vehicle_points = points[vehicle_mask, :3]
    vehicle_frame_ids = point_frame_ids[vehicle_mask]
    voxel_count = len(_key_set(_voxel_keys(vehicle_points, candidate_box, config), 3))
    z50, z75, z90, z95, height_span = _quantiles(vehicle_points[:, 2])

    valid_frames = tuple(sorted({observation.frame_id for observation in accumulation.observations}))
    frame_counts = {
        frame_id: int(np.sum(vehicle_frame_ids == frame_id))
        for frame_id in valid_frames
    }
    supported_frames = tuple(
        frame_id
        for frame_id, count in frame_counts.items()
        if count >= config.occupied_layer_min_points_per_frame
    )
    temporal_support = len(supported_frames) / len(valid_frames) if valid_frames else 0.0
    layers = _height_layers(vehicle_points, vehicle_frame_ids, candidate_box, config)

    candidate_polygon = box_polygon(
        candidate_box.center,
        candidate_box.yaw,
        candidate_box.length,
        candidate_box.width,
    )
    ownership_core = inset_local_rectangle(
        metric_slot.core_polygon_local_m,
        config.occupied_localization_uncertainty_m,
    )
    core_overlap = polygon_overlap_ratio(candidate_polygon, ownership_core)
    slot_overlap = polygon_overlap_ratio(candidate_polygon, metric_slot.polygon_local_m)
    in_core = points_in_polygon(vehicle_points[:, :2], ownership_core)
    boundary_ratio = float((~in_core).sum() / len(vehicle_points)) if len(vehicle_points) else 0.0
    core_points = vehicle_points[in_core]
    core_frame_ids = vehicle_frame_ids[in_core]
    core_voxel_count = len(
        _key_set(_voxel_keys(core_points, candidate_box, config), 3)
    )
    core_supported_frames = {
        int(frame_id)
        for frame_id in np.unique(core_frame_ids)
        if int(np.sum(core_frame_ids == frame_id))
        >= config.occupied_layer_min_points_per_frame
    }

    margin_vehicle_mask = (
        points_in_polygon(points[:, :2], metric_slot.margin_polygon_local_m)
        & (points[:, 2] >= config.vehicle_z_min_m)
        & (points[:, 2] <= config.vehicle_z_max_m)
    )
    margin_count = int(margin_vehicle_mask.sum())
    inside_margin_candidate = int((margin_vehicle_mask & vehicle_mask).sum())
    outside_residual = (margin_count - inside_margin_candidate) / margin_count if margin_count else 0.0

    pca_linearity, planarity = _pca_shape(vehicle_points, config.pca_max_points)
    # The terminal threshold is inherited from the existing box scorer and is
    # defined on XY anisotropy.  XYZ PCA linearity has a different numerical
    # meaning, so keep it as a separate audit feature instead of applying the
    # old 0.60 threshold to it.
    linearity = linearity_penalty(vehicle_points[:, :2])
    (
        robust_pca_linearity,
        robust_extent_x,
        robust_extent_y,
        robust_supported_frames,
        robust_point_fraction,
    ) = _robust_shape(
        vehicle_points,
        vehicle_frame_ids,
        candidate_box,
        config,
    )
    candidate_xy = _candidate_coordinates(vehicle_points[:, :2], candidate_box)
    if len(vehicle_points):
        extent_x = float(np.ptp(candidate_xy[:, 0]))
        extent_y = float(np.ptp(candidate_xy[:, 1]))
        extent_z = float(np.ptp(vehicle_points[:, 2]))
    else:
        extent_x = extent_y = extent_z = 0.0

    splits = tuple(
        score_split_consistency(accumulation, candidate_box, split_name, config)
        for split_name in ("chronological", "parity")
    )
    temporal_consistency = float(np.mean([split.consistency for split in splits]))
    layer_coverages = [layer.bev_coverage for layer in layers]
    while len(layer_coverages) < 3:
        layer_coverages.append(0.0)
    return ThreeDEvidence(
        point_count=len(vehicle_points),
        voxel_count=voxel_count,
        z50_m=z50,
        z75_m=z75,
        z90_m=z90,
        z95_m=z95,
        height_span_m=height_span,
        supported_frame_count=len(supported_frames),
        temporal_support=float(temporal_support),
        height_layers=layers,
        supported_layer_count=sum(layer.supported for layer in layers),
        low_bev_coverage=float(layer_coverages[0]),
        mid_bev_coverage=float(layer_coverages[1]),
        high_bev_coverage=float(layer_coverages[2]),
        core_overlap=float(core_overlap),
        slot_overlap=float(slot_overlap),
        boundary_ratio=boundary_ratio,
        adjacent_overlap=_adjacent_overlap(metric_slot, candidate_polygon, slots_by_id),
        outside_residual_ratio=float(outside_residual),
        linearity=linearity,
        planarity=planarity,
        extent_x_m=extent_x,
        extent_y_m=extent_y,
        extent_z_m=extent_z,
        overall_clears_weak_gate=_weak_subset_gate(
            vehicle_points,
            candidate_box,
            config,
            voxel_count=voxel_count,
        ),
        temporal_consistency=temporal_consistency,
        splits=splits,
        core_point_count=len(core_points),
        core_voxel_count=core_voxel_count,
        core_supported_frame_count=len(core_supported_frames),
        pca_linearity=pca_linearity,
        robust_pca_linearity=robust_pca_linearity,
        robust_extent_x_m=robust_extent_x,
        robust_extent_y_m=robust_extent_y,
        robust_supported_frame_count=robust_supported_frames,
        robust_point_fraction=robust_point_fraction,
    )
