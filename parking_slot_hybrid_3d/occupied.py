from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from parking_slot_box_scoring.box_hypotheses import CandidateBox, enumerate_box_hypotheses
from parking_slot_box_scoring.geometry import (
    box_polygon,
    compute_slot_frame,
    points_in_polygon,
    polygon_overlap_ratio,
)

from .config import Hybrid3DConfig
from .contracts import (
    GateResult,
    MetricSlot,
    OccupiedEvidence,
    SlotAccumulation,
    ThreeDEvidence,
)
from .evidence_3d import extract_3d_evidence
from .geometry import inset_local_rectangle


_FAILURE_BY_GATE = {
    "valid_frames": "insufficient_valid_frames",
    "vehicle_points": "insufficient_vehicle_points",
    "support_frames": "insufficient_support_frames",
    "temporal_support": "low_temporal_support",
    "z95_m": "insufficient_vehicle_height",
    "height_span_m": "insufficient_height_span",
    "robust_short_extent": "insufficient_vehicle_footprint",
    "low_layer_coverage": "insufficient_lower_body_coverage",
    "core_overlap": "low_core_overlap",
    "adjacent_overlap": "adjacent_overlap_conflict",
    "boundary_ratio": "boundary_dominated",
    "voxel_count": "insufficient_3d_voxels",
    "supported_layers": "insufficient_height_layers",
    "low_height_veto": "low_height_structure",
    "upper_height_spread": "horizontal_cap_structure",
    "linearity": "linear_static_structure",
    "compact_vertical_structure": "linear_static_structure",
    "compact_vertical_footprint": "compact_vertical_structure",
    "outside_residual": "outside_residual_conflict",
}

_NON_BYPASSABLE_SAFETY_GATES = frozenset(
    {
        "compact_vertical_structure",
        "compact_vertical_footprint",
        "upper_height_spread",
    }
)


def has_non_bypassable_safety_gate_failure(
    gates: tuple[GateResult, ...],
) -> bool:
    """True when a size-aware physical veto must survive box re-ranking.

    Generic linearity is deliberately candidate-local: a narrow hypothesis can
    crop a real vehicle down to a line. Compact-footprint and horizontal-cap
    gates encode physical dimensions, so a wider alternative must not erase
    those safety findings.
    """

    return any(
        not gate.passed and gate.name in _NON_BYPASSABLE_SAFETY_GATES
        for gate in gates
    )


def _compact_vertical_risks(
    features: ThreeDEvidence,
    config: Hybrid3DConfig,
) -> tuple[bool, bool]:
    """Return legacy and footprint-aware compact vertical structure risks."""

    robust_extent_x = max(float(features.robust_extent_x_m), 0.0)
    robust_extent_y = max(float(features.robust_extent_y_m), 0.0)
    horizontal_extent = max(robust_extent_x, robust_extent_y)
    footprint_area = robust_extent_x * robust_extent_y
    legacy_support = bool(
        features.robust_supported_frame_count >= 2
        and features.robust_point_fraction
        >= config.occupied_pillar_min_inlier_fraction
    )
    legacy = bool(
        features.robust_pca_linearity >= config.occupied_pillar_min_pca_linearity
        and horizontal_extent <= config.occupied_pillar_max_horizontal_extent_m
        and legacy_support
    )
    vertical_aspect = float(features.extent_z_m) / max(horizontal_extent, 1e-6)
    footprint_aware = bool(
        features.robust_pca_linearity
        >= config.occupied_pillar_secondary_min_pca_linearity
        and footprint_area <= config.occupied_pillar_max_footprint_area_m2
        and features.extent_z_m >= config.occupied_pillar_min_height_m
        and vertical_aspect >= config.occupied_pillar_min_vertical_aspect_ratio
        and features.robust_supported_frame_count >= 2
        and features.robust_point_fraction
        >= config.occupied_pillar_secondary_min_inlier_fraction
    )
    return legacy, footprint_aware


def evaluate_occupied_gates(
    valid_frame_count: int,
    features: ThreeDEvidence,
    config: Hybrid3DConfig,
) -> tuple[GateResult, ...]:
    config.validate()
    low_height_ok = (
        features.z95_m >= config.low_height_veto_z95_m
        and features.height_span_m >= config.low_height_veto_span_m
    )
    upper_height_spread_ratio = (
        max(float(features.z95_m) - float(features.z75_m), 0.0)
        / max(float(features.height_span_m), 1e-6)
    )
    compact_vertical_structure, compact_vertical_footprint = (
        _compact_vertical_risks(features, config)
    )
    robust_horizontal_extent = max(
        features.robust_extent_x_m, features.robust_extent_y_m
    )
    robust_short_extent = min(
        max(features.robust_extent_x_m, 0.0),
        max(features.robust_extent_y_m, 0.0),
    )
    robust_footprint_area = (
        max(features.robust_extent_x_m, 0.0)
        * max(features.robust_extent_y_m, 0.0)
    )
    vertical_aspect = features.extent_z_m / max(robust_horizontal_extent, 1e-6)
    return (
        GateResult("valid_frames", valid_frame_count >= config.occupied_min_valid_frames, valid_frame_count, f">={config.occupied_min_valid_frames}"),
        GateResult("vehicle_points", features.point_count >= config.occupied_min_points, features.point_count, f">={config.occupied_min_points}"),
        GateResult("support_frames", features.supported_frame_count >= config.occupied_min_support_frames, features.supported_frame_count, f">={config.occupied_min_support_frames}"),
        GateResult("temporal_support", features.temporal_support >= config.occupied_min_temporal_support, features.temporal_support, f">={config.occupied_min_temporal_support}"),
        GateResult("z95_m", features.z95_m >= config.occupied_min_z95_m, features.z95_m, f">={config.occupied_min_z95_m}"),
        GateResult("height_span_m", features.height_span_m >= config.occupied_min_height_span_m, features.height_span_m, f">={config.occupied_min_height_span_m}"),
        GateResult(
            "robust_short_extent",
            robust_short_extent >= config.occupied_min_robust_short_extent_m,
            robust_short_extent,
            f">={config.occupied_min_robust_short_extent_m}",
        ),
        GateResult(
            "low_layer_coverage",
            features.low_bev_coverage >= config.occupied_min_low_bev_coverage,
            features.low_bev_coverage,
            f">={config.occupied_min_low_bev_coverage}",
        ),
        GateResult("core_overlap", features.core_overlap >= config.occupied_min_core_overlap, features.core_overlap, f">={config.occupied_min_core_overlap}"),
        GateResult("adjacent_overlap", features.adjacent_overlap < config.occupied_max_adjacent_overlap, features.adjacent_overlap, f"<{config.occupied_max_adjacent_overlap}"),
        GateResult("boundary_ratio", features.boundary_ratio < config.occupied_max_boundary_ratio, features.boundary_ratio, f"<{config.occupied_max_boundary_ratio}"),
        GateResult("voxel_count", features.voxel_count >= config.occupied_min_voxels, features.voxel_count, f">={config.occupied_min_voxels}"),
        GateResult("supported_layers", features.supported_layer_count >= config.occupied_min_supported_layers, features.supported_layer_count, f">={config.occupied_min_supported_layers}"),
        GateResult(
            "low_height_veto",
            low_height_ok,
            f"z95={features.z95_m:.6g},span={features.height_span_m:.6g}",
            f"z95>={config.low_height_veto_z95_m} and span>={config.low_height_veto_span_m}",
        ),
        GateResult(
            "upper_height_spread",
            upper_height_spread_ratio
            >= config.occupied_min_upper_height_spread_ratio,
            upper_height_spread_ratio,
            f">={config.occupied_min_upper_height_spread_ratio}",
        ),
        GateResult("linearity", features.linearity < config.occupied_max_linearity, features.linearity, f"<{config.occupied_max_linearity}"),
        GateResult(
            "compact_vertical_structure",
            not compact_vertical_structure,
            f"robust_pca_linearity={features.robust_pca_linearity:.6g},robust_horizontal_extent={robust_horizontal_extent:.6g},robust_frames={features.robust_supported_frame_count},robust_fraction={features.robust_point_fraction:.6g}",
            f"not(robust_pca_linearity>={config.occupied_pillar_min_pca_linearity} and robust_horizontal_extent<={config.occupied_pillar_max_horizontal_extent_m} and robust_frames>=2 and robust_fraction>={config.occupied_pillar_min_inlier_fraction})",
        ),
        GateResult(
            "compact_vertical_footprint",
            not compact_vertical_footprint,
            f"robust_pca_linearity={features.robust_pca_linearity:.6g},robust_footprint_area_m2={robust_footprint_area:.6g},height_m={features.extent_z_m:.6g},vertical_aspect={vertical_aspect:.6g},robust_frames={features.robust_supported_frame_count},robust_fraction={features.robust_point_fraction:.6g}",
            f"not(robust_pca_linearity>={config.occupied_pillar_secondary_min_pca_linearity} and robust_footprint_area_m2<={config.occupied_pillar_max_footprint_area_m2} and height_m>={config.occupied_pillar_min_height_m} and vertical_aspect>={config.occupied_pillar_min_vertical_aspect_ratio} and robust_frames>=2 and robust_fraction>={config.occupied_pillar_secondary_min_inlier_fraction})",
        ),
        GateResult("outside_residual", features.outside_residual_ratio <= config.occupied_max_outside_residual, features.outside_residual_ratio, f"<={config.occupied_max_outside_residual}"),
    )


def gate_failure_codes(gates: tuple[GateResult, ...]) -> tuple[str, ...]:
    failures: list[str] = []
    for gate in gates:
        if gate.passed:
            continue
        code = _FAILURE_BY_GATE.get(gate.name, f"failed_gate_{gate.name}")
        if code not in failures:
            failures.append(code)
    return tuple(failures)


@dataclass(frozen=True)
class _CheapCandidate:
    index: int
    box: CandidateBox
    rank: tuple[float, ...]
    potentially_strong: bool


@dataclass(frozen=True)
class _EvaluatedCandidate:
    index: int
    box: CandidateBox
    features: ThreeDEvidence
    gates: tuple[GateResult, ...]
    failures: tuple[str, ...]
    strong: bool
    strength: float


def _clamp_ratio(value: float, threshold: float) -> float:
    return float(np.clip(value / max(threshold, 1e-12), 0.0, 1.0))


def _adjacent_core_polygons(
    metric_slot: MetricSlot,
    slots_by_id: Mapping[str, MetricSlot],
) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray], ...]:
    polygons: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for adjacent_id in metric_slot.adjacent_slots:
        adjacent = slots_by_id.get(adjacent_id)
        if adjacent is None:
            continue
        adjacent_map = (
            adjacent.center_map
            + adjacent.map_units_per_meter
            * (
                adjacent.core_polygon_local_m[:, 0:1] * adjacent.long_axis_map
                + adjacent.core_polygon_local_m[:, 1:2] * adjacent.short_axis_map
            )
        )
        delta_m = (adjacent_map - metric_slot.center_map) / metric_slot.map_units_per_meter
        polygon = np.column_stack(
            [delta_m @ metric_slot.long_axis_map, delta_m @ metric_slot.short_axis_map]
        )
        polygons.append((polygon, polygon.min(axis=0), polygon.max(axis=0)))
    return tuple(polygons)


def _cheap_candidates(
    metric_slot: MetricSlot,
    accumulation: SlotAccumulation,
    hypotheses: list[CandidateBox],
    slots_by_id: Mapping[str, MetricSlot],
    config: Hybrid3DConfig,
) -> list[_CheapCandidate]:
    points = np.asarray(accumulation.points_local_xyzi, dtype=np.float64)
    point_frame_ids = np.asarray(accumulation.point_frame_ids, dtype=np.int64)
    vehicle_mask = (
        (points[:, 2] >= config.vehicle_z_min_m)
        & (points[:, 2] <= config.vehicle_z_max_m)
    )
    vehicle_points = points[vehicle_mask]
    vehicle_frame_ids = point_frame_ids[vehicle_mask]
    ownership_core = inset_local_rectangle(
        metric_slot.core_polygon_local_m,
        config.occupied_localization_uncertainty_m,
    )
    in_core = points_in_polygon(vehicle_points[:, :2], ownership_core)
    in_margin = points_in_polygon(vehicle_points[:, :2], metric_slot.margin_polygon_local_m)
    margin_count = int(in_margin.sum())
    valid_frame_count = len(accumulation.observations)
    adjacent_polygons = _adjacent_core_polygons(metric_slot, slots_by_id)
    projection_cache: dict[float, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    for candidate in hypotheses:
        if candidate.yaw in projection_cache:
            continue
        long_axis = np.asarray([math.cos(candidate.yaw), math.sin(candidate.yaw)])
        short_axis = np.asarray([-math.sin(candidate.yaw), math.cos(candidate.yaw)])
        projection_cache[candidate.yaw] = (
            vehicle_points[:, :2] @ long_axis,
            vehicle_points[:, :2] @ short_axis,
            long_axis,
            short_axis,
        )
    ranked: list[_CheapCandidate] = []
    for index, candidate in enumerate(hypotheses):
        projected_long, projected_short, long_axis, short_axis = projection_cache[candidate.yaw]
        center_long = float(np.asarray(candidate.center) @ long_axis)
        center_short = float(np.asarray(candidate.center) @ short_axis)
        in_box = (
            (np.abs(projected_long - center_long) <= candidate.length / 2.0 + 1e-12)
            & (np.abs(projected_short - center_short) <= candidate.width / 2.0 + 1e-12)
        )
        count = int(in_box.sum())
        if count:
            candidate_z = vehicle_points[in_box, 2]
            z95 = float(np.quantile(candidate_z, 0.95))
            height_span = float(candidate_z.max() - candidate_z.min())
            candidate_frame_ids = vehicle_frame_ids[in_box]
            _, frame_counts = np.unique(candidate_frame_ids, return_counts=True)
            support_frames = int((frame_counts >= config.occupied_layer_min_points_per_frame).sum())
            temporal_support = support_frames / valid_frame_count if valid_frame_count else 0.0
            boundary_ratio = float((~in_core[in_box]).sum() / count)
        else:
            z95 = 0.0
            height_span = 0.0
            support_frames = 0
            temporal_support = 0.0
            boundary_ratio = 0.0
        candidate_polygon = box_polygon(
            candidate.center,
            candidate.yaw,
            candidate.length,
            candidate.width,
        )
        core_overlap = polygon_overlap_ratio(candidate_polygon, ownership_core)
        candidate_min = candidate_polygon.min(axis=0)
        candidate_max = candidate_polygon.max(axis=0)
        adjacent_overlap = 0.0
        for polygon, polygon_min, polygon_max in adjacent_polygons:
            if np.any(candidate_max < polygon_min) or np.any(polygon_max < candidate_min):
                continue
            adjacent_overlap = max(
                adjacent_overlap,
                polygon_overlap_ratio(candidate_polygon, polygon),
            )
        inside_margin_candidate = int((in_box & in_margin).sum())
        outside_residual = (
            (margin_count - inside_margin_candidate) / margin_count
            if margin_count
            else 0.0
        )
        cheap_gate_results = (
            valid_frame_count >= config.occupied_min_valid_frames,
            count >= config.occupied_min_points,
            support_frames >= config.occupied_min_support_frames,
            temporal_support >= config.occupied_min_temporal_support,
            z95 >= config.occupied_min_z95_m,
            height_span >= config.occupied_min_height_span_m,
            core_overlap >= config.occupied_min_core_overlap,
            adjacent_overlap < config.occupied_max_adjacent_overlap,
            boundary_ratio < config.occupied_max_boundary_ratio,
            outside_residual <= config.occupied_max_outside_residual,
        )
        cheap_passes = sum(cheap_gate_results)
        quality = (
            1.5 * _clamp_ratio(count, config.occupied_min_points)
            + _clamp_ratio(support_frames, config.occupied_min_support_frames)
            + _clamp_ratio(temporal_support, config.occupied_min_temporal_support)
            + _clamp_ratio(z95, config.occupied_min_z95_m)
            + _clamp_ratio(height_span, config.occupied_min_height_span_m)
            + 1.5 * core_overlap
            + (1.0 - adjacent_overlap)
            + (1.0 - boundary_ratio)
            + (1.0 - outside_residual)
        )
        rank = (
            float(cheap_passes),
            float(quality),
            float(core_overlap),
            float(1.0 - adjacent_overlap),
            float(1.0 - boundary_ratio),
            float(1.0 - outside_residual),
            float(count),
            float(-index),
        )
        ranked.append(
            _CheapCandidate(
                index=index,
                box=candidate,
                rank=rank,
                potentially_strong=all(cheap_gate_results),
            )
        )
    ranked.sort(key=lambda item: item.rank, reverse=True)
    return ranked


def _validate_finite_features(features: ThreeDEvidence) -> None:
    values = np.asarray(
        [
            features.z50_m,
            features.z75_m,
            features.z90_m,
            features.z95_m,
            features.height_span_m,
            features.temporal_support,
            features.low_bev_coverage,
            features.mid_bev_coverage,
            features.high_bev_coverage,
            features.core_overlap,
            features.slot_overlap,
            features.boundary_ratio,
            features.adjacent_overlap,
            features.outside_residual_ratio,
            features.linearity,
            features.planarity,
            features.extent_x_m,
            features.extent_y_m,
            features.extent_z_m,
            features.temporal_consistency,
            features.pca_linearity,
            features.robust_pca_linearity,
            features.robust_extent_x_m,
            features.robust_extent_y_m,
            features.robust_point_fraction,
        ],
        dtype=np.float64,
    )
    if not np.isfinite(values).all():
        raise ValueError("3D evidence contains non-finite values")


def _evidence_strength(gates: tuple[GateResult, ...], features: ThreeDEvidence) -> float:
    gate_ratio = sum(gate.passed for gate in gates) / max(len(gates), 1)
    return float(np.clip(0.85 * gate_ratio + 0.15 * features.temporal_consistency, 0.0, 1.0))


def _candidate_rank(candidate: _EvaluatedCandidate) -> tuple[float, ...]:
    return (
        float(candidate.strong),
        float(sum(gate.passed for gate in candidate.gates)),
        candidate.strength,
        candidate.features.temporal_consistency,
        candidate.features.core_overlap,
        1.0 - candidate.features.adjacent_overlap,
        float(candidate.features.point_count),
        float(-candidate.index),
    )


def _weak_evidence(features: ThreeDEvidence, config: Hybrid3DConfig) -> bool:
    return bool(
        features.point_count >= max(10, int(math.ceil(config.occupied_min_points / 4.0)))
        and features.voxel_count >= max(4, int(math.ceil(config.occupied_min_voxels / 2.0)))
        and features.supported_frame_count >= 2
        and features.temporal_support > 0.0
    )


def _static_structure_risk(features: ThreeDEvidence, config: Hybrid3DConfig) -> float:
    low_height_risk = float(
        features.z95_m < config.low_height_veto_z95_m
        or features.height_span_m < config.low_height_veto_span_m
    )
    legacy_pillar_risk, footprint_pillar_risk = _compact_vertical_risks(
        features, config
    )
    compact_vertical_risk = float(legacy_pillar_risk or footprint_pillar_risk)
    return float(
        np.clip(
            max(
                low_height_risk,
                compact_vertical_risk,
                features.linearity,
                features.planarity,
                features.boundary_ratio,
                features.adjacent_overlap,
                features.outside_residual_ratio,
            ),
            0.0,
            1.0,
        )
    )


def _box_record(candidate: CandidateBox) -> tuple[tuple[str, float], ...]:
    return (
        ("center_x_m", float(candidate.center[0])),
        ("center_y_m", float(candidate.center[1])),
        ("yaw_rad", float(candidate.yaw)),
        ("length_m", float(candidate.length)),
        ("width_m", float(candidate.width)),
    )


def candidate_box_from_record(record: tuple[tuple[str, float], ...]) -> CandidateBox:
    values = dict(record)
    required = {"center_x_m", "center_y_m", "yaw_rad", "length_m", "width_m"}
    if set(values) != required:
        raise ValueError("best_box record has an invalid field set")
    return CandidateBox(
        center=np.asarray([values["center_x_m"], values["center_y_m"]], dtype=np.float64),
        yaw=float(values["yaw_rad"]),
        length=float(values["length_m"]),
        width=float(values["width_m"]),
    )


def _axial_yaw_difference_rad(left: float, right: float) -> float:
    """Return the smallest box-axis angle; a box is unchanged after 180 degrees."""

    return abs((float(left) - float(right) + math.pi / 2.0) % math.pi - math.pi / 2.0)


def candidate_matches_reference(
    candidate: CandidateBox,
    reference: CandidateBox,
    config: Hybrid3DConfig,
) -> bool:
    """Guard a pose refit against jumping to another object in the same slot ROI."""

    config.validate()
    center_shift = float(
        np.linalg.norm(
            np.asarray(candidate.center, dtype=np.float64)
            - np.asarray(reference.center, dtype=np.float64)
        )
    )
    yaw_shift_deg = math.degrees(
        _axial_yaw_difference_rad(candidate.yaw, reference.yaw)
    )
    candidate_polygon = box_polygon(
        candidate.center,
        candidate.yaw,
        candidate.length,
        candidate.width,
    )
    reference_polygon = box_polygon(
        reference.center,
        reference.yaw,
        reference.length,
        reference.width,
    )
    minimum_overlap = min(
        polygon_overlap_ratio(candidate_polygon, reference_polygon),
        polygon_overlap_ratio(reference_polygon, candidate_polygon),
    )
    return bool(
        center_shift <= config.stability_refit_max_center_shift_m + 1e-12
        and yaw_shift_deg <= config.stability_refit_max_yaw_shift_deg + 1e-12
        and minimum_overlap >= config.stability_refit_min_box_overlap - 1e-12
    )


def evaluate_fixed_occupied(
    metric_slot: MetricSlot,
    accumulation: SlotAccumulation,
    slots_by_id: Mapping[str, MetricSlot],
    config: Hybrid3DConfig,
    best_box: tuple[tuple[str, float], ...],
) -> OccupiedEvidence:
    try:
        candidate = candidate_box_from_record(best_box)
        features = extract_3d_evidence(
            metric_slot,
            candidate,
            accumulation,
            config,
            slots_by_id=slots_by_id,
        )
        _validate_finite_features(features)
        gates = evaluate_occupied_gates(len(accumulation.observations), features, config)
        failures = gate_failure_codes(gates)
        strong = not failures
        return OccupiedEvidence(
            strong=strong,
            weak=bool(not strong and _weak_evidence(features, config)),
            strength=_evidence_strength(gates, features),
            support_frame_count=features.supported_frame_count,
            temporal_consistency=features.temporal_consistency,
            core_ownership=features.core_overlap,
            static_structure_risk=_static_structure_risk(features, config),
            best_box=best_box,
            gate_results=gates,
            failures=failures,
            features=features,
        )
    except Exception:
        return OccupiedEvidence(failures=("occupied_evaluation_error",))


def evaluate_occupied(
    metric_slot: MetricSlot,
    accumulation: SlotAccumulation,
    slots_by_id: Mapping[str, MetricSlot],
    config: Hybrid3DConfig,
    *,
    reference_box: tuple[tuple[str, float], ...] | None = None,
) -> OccupiedEvidence:
    try:
        config.validate()
        if accumulation.slot_id != metric_slot.slot_id:
            raise ValueError("accumulation slot_id does not match metric slot")
        frame = compute_slot_frame(
            np.asarray(metric_slot.polygon_local_m, dtype=np.float64),
            center=np.asarray([0.0, 0.0], dtype=np.float64),
        )
        hypotheses = enumerate_box_hypotheses(frame, config.box_hypothesis_config())
        if reference_box is not None:
            reference = candidate_box_from_record(reference_box)
            hypotheses = [
                candidate
                for candidate in hypotheses
                if candidate_matches_reference(candidate, reference, config)
            ]
        if not hypotheses:
            raise ValueError("no box hypotheses")
        ranked_candidates = _cheap_candidates(
            metric_slot,
            accumulation,
            hypotheses,
            slots_by_id,
            config,
        )
        if not ranked_candidates:
            raise ValueError("no candidate reached the ranking stage")

        evaluated: list[_EvaluatedCandidate] = []
        evaluated_indices: set[int] = set()
        safety_vetoed = False

        def evaluate_candidate(cheap: _CheapCandidate) -> _EvaluatedCandidate:
            features = extract_3d_evidence(
                metric_slot,
                cheap.box,
                accumulation,
                config,
                slots_by_id=slots_by_id,
            )
            _validate_finite_features(features)
            gates = evaluate_occupied_gates(len(accumulation.observations), features, config)
            failures = gate_failure_codes(gates)
            strong = not failures
            return _EvaluatedCandidate(
                index=cheap.index,
                box=cheap.box,
                features=features,
                gates=gates,
                failures=failures,
                strong=strong,
                strength=_evidence_strength(gates, features),
            )

        for cheap in ranked_candidates:
            if not cheap.potentially_strong:
                continue
            candidate = evaluate_candidate(cheap)
            evaluated.append(candidate)
            evaluated_indices.add(cheap.index)
            if has_non_bypassable_safety_gate_failure(candidate.gates):
                safety_vetoed = True
                break
            if candidate.strong:
                break

        if (
            not safety_vetoed
            and not any(candidate.strong for candidate in evaluated)
        ):
            fallback_count = 0
            for cheap in ranked_candidates:
                if cheap.index in evaluated_indices:
                    continue
                evaluated.append(evaluate_candidate(cheap))
                evaluated_indices.add(cheap.index)
                fallback_count += 1
                if fallback_count >= config.occupied_candidate_shortlist_size:
                    break
        if not evaluated:
            raise ValueError("all occupied candidates failed evaluation")
        best = max(evaluated, key=_candidate_rank)
        weak = bool(not best.strong and _weak_evidence(best.features, config))
        return OccupiedEvidence(
            strong=best.strong,
            weak=weak,
            strength=best.strength,
            support_frame_count=best.features.supported_frame_count,
            temporal_consistency=best.features.temporal_consistency,
            core_ownership=best.features.core_overlap,
            static_structure_risk=_static_structure_risk(best.features, config),
            best_box=_box_record(best.box),
            gate_results=best.gates,
            failures=best.failures,
            features=best.features,
        )
    except Exception:
        return OccupiedEvidence(failures=("occupied_evaluation_error",))
