"""Deterministic geometry fixtures for the hybrid-3D acceptance matrix."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from parking_slot_hybrid_3d.config import Hybrid3DConfig
from parking_slot_hybrid_3d.contracts import (
    AgentContext,
    DecisionState,
    FrameObservation,
    FrameRecord,
    FreeEvidence,
    GroundModel,
    KnownSlot,
    MetricSlot,
    OccupiedEvidence,
    ScopeEvidence,
    ScopeStatus,
    SlotAccumulation,
    StabilityEvidence,
)
from parking_slot_hybrid_3d.decision import QualityEvidence
from parking_slot_hybrid_3d.free_space import evaluate_free_space
from parking_slot_hybrid_3d.ground import fit_ground_model, normalize_z
from parking_slot_hybrid_3d.io import FrameLoadError
from parking_slot_hybrid_3d.known_slot_scope import KnownSlotObservationScopeEvaluator
from parking_slot_hybrid_3d.occupied import evaluate_fixed_occupied, evaluate_occupied
from parking_slot_hybrid_3d.stability import evaluate_stability


def rectangle(half_length: float, half_width: float) -> np.ndarray:
    return np.asarray(
        [
            [-half_length, -half_width],
            [half_length, -half_width],
            [half_length, half_width],
            [-half_length, half_width],
        ],
        dtype=np.float64,
    )


def metric_slot(
    slot_id: str,
    *,
    center_map: tuple[float, float] = (0.0, 0.0),
    adjacent_slots: tuple[str, ...] = (),
    free_geometry: bool = False,
) -> MetricSlot:
    polygon = rectangle(2.2, 1.2) if free_geometry else rectangle(2.0, 1.0)
    core = rectangle(2.0, 1.0) if free_geometry else rectangle(1.8, 0.8)
    return MetricSlot(
        slot_id=slot_id,
        center_map=np.asarray(center_map, dtype=np.float64),
        long_axis_map=np.asarray([1.0, 0.0], dtype=np.float64),
        short_axis_map=np.asarray([0.0, 1.0], dtype=np.float64),
        polygon_local_m=polygon,
        core_polygon_local_m=core,
        margin_polygon_local_m=rectangle(2.5, 1.5),
        adjacent_slots=adjacent_slots,
        map_units_per_meter=1.0,
    )


def known_slot(slot_id: str, center_map: tuple[float, float] = (0.0, 0.0)) -> KnownSlot:
    center = np.asarray(center_map, dtype=np.float64)
    return KnownSlot(
        slot_id=slot_id,
        polygon_map=center + rectangle(2.0, 1.0),
        core_polygon_map=center + rectangle(1.8, 0.8),
        margin_polygon_map=center + rectangle(2.5, 1.5),
        center_map=center,
        heading_deg=0.0,
    )


def vehicle_points(
    frame_ids: Iterable[int],
    *,
    x_values: Iterable[float] = (-1.0, -0.5, 0.0, 0.5, 1.0),
    y_values: Iterable[float] = (-0.5, 0.0, 0.5),
    z_values: Iterable[float] = (0.45, 1.00, 1.60),
) -> tuple[np.ndarray, np.ndarray]:
    points: list[list[float]] = []
    owners: list[int] = []
    for frame_id in frame_ids:
        for x_value in x_values:
            for y_value in y_values:
                for z_value in z_values:
                    points.append([x_value, y_value, z_value, 1.0])
                    owners.append(frame_id)
    return np.asarray(points, dtype=np.float64), np.asarray(owners, dtype=np.int64)


def observation(
    frame_id: int,
    origin: tuple[float, float, float],
    endpoints: np.ndarray | None = None,
    *,
    ground_model: GroundModel | None = None,
) -> FrameObservation:
    return FrameObservation(
        frame_id=frame_id,
        origin_local_xyz=np.asarray(origin, dtype=np.float64),
        points_local_xyzi=np.empty((0, 4), dtype=np.float64),
        ray_endpoints_local_xyz=np.asarray(
            np.empty((0, 3), dtype=np.float64) if endpoints is None else endpoints,
            dtype=np.float64,
        ).reshape(-1, 3),
        ground_model=ground_model or GroundModel(method="plane", valid=True),
    )


def accumulation_from_points(
    slot_id: str,
    points: np.ndarray,
    owners: np.ndarray,
    frame_ids: tuple[int, ...],
    *,
    ground_model: GroundModel | None = None,
) -> SlotAccumulation:
    observations = tuple(
        observation(frame_id, (-5.0, 0.0, 1.5), ground_model=ground_model)
        for frame_id in frame_ids
    )
    return SlotAccumulation(
        slot_id=slot_id,
        anchor_frame=frame_ids[len(frame_ids) // 2],
        selected_frames=frame_ids,
        points_local_xyzi=np.asarray(points, dtype=np.float64).reshape(-1, 4),
        point_frame_ids=np.asarray(owners, dtype=np.int64),
        observations=observations,
    )


def accumulation_from_observations(
    slot_id: str,
    observations: tuple[FrameObservation, ...],
) -> SlotAccumulation:
    frame_ids = tuple(item.frame_id for item in observations)
    return SlotAccumulation(
        slot_id=slot_id,
        anchor_frame=frame_ids[len(frame_ids) // 2],
        selected_frames=frame_ids,
        points_local_xyzi=np.empty((0, 4), dtype=np.float64),
        point_frame_ids=np.empty(0, dtype=np.int64),
        observations=observations,
    )


def core_target_centers() -> np.ndarray:
    x_values = np.arange(-1.875, 1.876, 0.25)
    y_values = np.arange(-0.875, 0.876, 0.25)
    z_values = np.arange(0.30, 2.101, 0.20)
    return np.asarray(
        [
            [x_value, y_value, z_value]
            for x_value in x_values
            for y_value in y_values
            for z_value in z_values
        ],
        dtype=np.float64,
    )


def endpoints_through_targets(origin: np.ndarray, factor: float = 5.0) -> np.ndarray:
    targets = core_target_centers()
    return origin + factor * (targets - origin)


def clear_observations() -> tuple[FrameObservation, ...]:
    result: list[FrameObservation] = []
    for frame_id in (1, 2, 3, 4, 5):
        from_left = frame_id % 2 == 1
        origin_x = -3.0 if from_left else 3.0
        origin = np.asarray([origin_x, 0.0, 1.5], dtype=np.float64)
        if frame_id <= 2:
            endpoints = endpoints_through_targets(origin)
        else:
            endpoint_x = 3.0 if from_left else -3.0
            endpoints = np.asarray([[endpoint_x, 0.0, 1.0]], dtype=np.float64)
        result.append(observation(frame_id, tuple(origin.tolist()), endpoints))
    return tuple(result)


def in_route_scope(slot_id: str) -> ScopeEvidence:
    return ScopeEvidence(
        slot_id=slot_id,
        scope_status=ScopeStatus.IN_ROUTE,
        near_frames=(1, 2, 3, 4, 5),
        crossing_frames=(1, 2, 3, 4, 5),
        hit_frames=(),
        core_ray_coverage=1.0,
        agent_observable=True,
        reasons=("scope_gate_passed",),
    )


class _MissingPointProvider:
    def load(self, frame_id: int) -> np.ndarray:
        raise FrameLoadError(frame_id, Path(f"{frame_id:06d}.npz"), "corrupt_test_frame")


class _EmptyPointProvider:
    def load(self, frame_id: int) -> np.ndarray:
        return np.empty((0, 4), dtype=np.float64)


def missing_scope(config: Hybrid3DConfig) -> ScopeEvidence:
    frame = FrameRecord(
        frame_id=1,
        map_x=-5.0,
        map_y=0.0,
        map_yaw=0.0,
        map_points_path=Path("000001.npz"),
    )
    return KnownSlotObservationScopeEvaluator(config, 1.0).evaluate(
        [known_slot("missing_or_corrupt_frame")],
        [frame],
        _MissingPointProvider(),
    )[0]


def out_of_route_scope(config: Hybrid3DConfig) -> ScopeEvidence:
    frame = FrameRecord(
        frame_id=1,
        map_x=-5.0,
        map_y=0.0,
        map_yaw=0.0,
        map_points_path=Path("000001.npz"),
    )
    return KnownSlotObservationScopeEvaluator(config, 1.0).evaluate(
        [known_slot("out_of_route_slot", (100.0, 0.0))],
        [frame],
        _EmptyPointProvider(),
    )[0]


def stable_occupied(
    slot: MetricSlot,
    accumulation: SlotAccumulation,
    occupied: OccupiedEvidence,
    slots: dict[str, MetricSlot],
    config: Hybrid3DConfig,
) -> StabilityEvidence:
    return evaluate_stability(
        slot,
        accumulation,
        lambda current_slot, current_accumulation: evaluate_fixed_occupied(
            current_slot,
            current_accumulation,
            slots,
            config,
            occupied.best_box,
        ).strong,
        config,
    )


def stable_free(
    slot: MetricSlot,
    accumulation: SlotAccumulation,
    config: Hybrid3DConfig,
) -> StabilityEvidence:
    return evaluate_stability(
        slot,
        accumulation,
        lambda current_slot, current_accumulation: evaluate_free_space(
            current_slot,
            current_accumulation,
            OccupiedEvidence(),
            config,
        ).strong,
        config,
    )


@dataclass(frozen=True)
class SyntheticScenario:
    name: str
    scope: ScopeEvidence
    occupied: OccupiedEvidence
    free: FreeEvidence
    stability: StabilityEvidence
    expected_scope: ScopeStatus
    expected_state: DecisionState | None
    expected_decision_reason: str
    expected_unknown_reasons: tuple[str, ...] = ()
    quality: QualityEvidence = QualityEvidence()
    agent_context: AgentContext = AgentContext(
        agent_observable=True,
        priority="inspect_vehicle_shape_and_occlusion",
        suggested_tools=("inspect_lidar_map",),
    )
    ground_model: GroundModel | None = None


def _scenario(
    name: str,
    *,
    occupied: OccupiedEvidence = OccupiedEvidence(),
    free: FreeEvidence = FreeEvidence(),
    stability: StabilityEvidence = StabilityEvidence(),
    scope: ScopeEvidence | None = None,
    state: DecisionState | None,
    reason: str,
    unknown_reasons: tuple[str, ...] = (),
    ground_model: GroundModel | None = None,
) -> SyntheticScenario:
    current_scope = scope or in_route_scope(name)
    return SyntheticScenario(
        name=name,
        scope=current_scope,
        occupied=occupied,
        free=free,
        stability=stability,
        expected_scope=current_scope.scope_status,
        expected_state=state,
        expected_decision_reason=reason,
        expected_unknown_reasons=unknown_reasons,
        ground_model=ground_model,
    )


def synthetic_scenario_matrix(
    config: Hybrid3DConfig,
) -> tuple[SyntheticScenario, ...]:
    config.validate()

    centered_slot = metric_slot("centered_vehicle")
    centered_points, centered_owners = vehicle_points((1, 2, 3, 4))
    centered_accumulation = accumulation_from_points(
        centered_slot.slot_id,
        centered_points,
        centered_owners,
        (1, 2, 3, 4),
    )
    centered_slots = {centered_slot.slot_id: centered_slot}
    centered_occupied = evaluate_occupied(
        centered_slot, centered_accumulation, centered_slots, config
    )
    centered_stability = stable_occupied(
        centered_slot,
        centered_accumulation,
        centered_occupied,
        centered_slots,
        config,
    )

    adjacent_slot = metric_slot(
        "adjacent_vehicle", adjacent_slots=("adjacent_peer",)
    )
    adjacent_peer = metric_slot("adjacent_peer", center_map=(0.0, 0.8))
    adjacent_points, adjacent_owners = vehicle_points(
        (1, 2, 3, 4), y_values=(0.8, 1.1, 1.4)
    )
    adjacent_occupied = evaluate_occupied(
        adjacent_slot,
        accumulation_from_points(
            adjacent_slot.slot_id,
            adjacent_points,
            adjacent_owners,
            (1, 2, 3, 4),
        ),
        {adjacent_slot.slot_id: adjacent_slot, adjacent_peer.slot_id: adjacent_peer},
        config,
    )

    crossing_slot = metric_slot(
        "cross_slot_vehicle", adjacent_slots=("crossing_peer",)
    )
    crossing_peer = metric_slot("crossing_peer", center_map=(0.0, 0.8))
    crossing_points, crossing_owners = vehicle_points(
        (1, 2, 3, 4),
        y_values=(-0.4, 0.0, 0.4, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4),
    )
    crossing_occupied = evaluate_occupied(
        crossing_slot,
        accumulation_from_points(
            crossing_slot.slot_id,
            crossing_points,
            crossing_owners,
            (1, 2, 3, 4),
        ),
        {crossing_slot.slot_id: crossing_slot, crossing_peer.slot_id: crossing_peer},
        config,
    )

    wheel_slot = metric_slot("wheel_stop")
    wheel_points: list[list[float]] = []
    wheel_owners: list[int] = []
    for frame_id in (1, 2, 3):
        for index, x_value in enumerate(np.linspace(-1.5, 1.5, 20)):
            wheel_points.append([x_value, 0.0, 0.31 + 0.02 * (index % 2), 1.0])
            wheel_owners.append(frame_id)
    wheel_occupied = evaluate_occupied(
        wheel_slot,
        accumulation_from_points(
            wheel_slot.slot_id,
            np.asarray(wheel_points),
            np.asarray(wheel_owners),
            (1, 2, 3),
        ),
        {wheel_slot.slot_id: wheel_slot},
        config,
    )

    column_slot = metric_slot("wall_or_column")
    column_points, column_owners = vehicle_points(
        (1, 2, 3, 4),
        x_values=(-0.15, 0.0, 0.15),
        y_values=(-0.15, 0.0, 0.15),
        z_values=(0.4, 1.0, 1.6, 2.0),
    )
    column_occupied = evaluate_occupied(
        column_slot,
        accumulation_from_points(
            column_slot.slot_id,
            column_points,
            column_owners,
            (1, 2, 3, 4),
        ),
        {column_slot.slot_id: column_slot},
        config,
    )

    merged_slot = metric_slot("oversized_merged_structure")
    merged_points, merged_owners = vehicle_points(
        (1, 2, 3, 4),
        x_values=np.linspace(-2.3, 2.3, 10),
        y_values=(-1.3, 0.0, 1.3),
    )
    merged_occupied = evaluate_occupied(
        merged_slot,
        accumulation_from_points(
            merged_slot.slot_id,
            merged_points,
            merged_owners,
            (1, 2, 3, 4),
        ),
        {merged_slot.slot_id: merged_slot},
        config,
    )

    clear_slot = metric_slot("fully_observable_empty", free_geometry=True)
    clear_accumulation = accumulation_from_observations(
        clear_slot.slot_id, clear_observations()
    )
    clear_free = evaluate_free_space(
        clear_slot, clear_accumulation, OccupiedEvidence(), config
    )
    clear_stability = stable_free(clear_slot, clear_accumulation, config)

    occluded_slot = metric_slot("occluded_vehicle", free_geometry=True)
    occluded_observations = list(clear_observations())
    for index in (0, 1):
        current = occluded_observations[index]
        hits = np.repeat(np.asarray([[0.0, 0.0, 1.0]]), 5, axis=0)
        occluded_observations[index] = observation(
            current.frame_id,
            tuple(current.origin_local_xyz.tolist()),
            np.vstack([current.ray_endpoints_local_xyz, hits]),
        )
    occluded_free = evaluate_free_space(
        occluded_slot,
        accumulation_from_observations(
            occluded_slot.slot_id, tuple(occluded_observations)
        ),
        OccupiedEvidence(),
        config,
    )

    partial_slot = metric_slot("partially_observable_empty", free_geometry=True)
    partial_observations = tuple(
        observation(
            frame_id,
            (-3.0, 0.0, 1.5),
            np.asarray([[3.0, 0.0, 0.5]], dtype=np.float64),
        )
        for frame_id in (1, 2, 3, 4, 5)
    )
    partial_free = evaluate_free_space(
        partial_slot,
        accumulation_from_observations(partial_slot.slot_id, partial_observations),
        OccupiedEvidence(),
        config,
    )

    slope_x, slope_y = np.meshgrid(
        np.linspace(-3.0, 3.0, 17), np.linspace(-1.5, 1.5, 11)
    )
    slope_ground_z = 0.04 * slope_x.ravel() - 0.02 * slope_y.ravel() - 1.65
    raw_ground = np.column_stack(
        [
            slope_x.ravel(),
            slope_y.ravel(),
            slope_ground_z
            + 0.008 * np.sin(np.arange(slope_ground_z.size, dtype=np.float64)),
        ]
    )
    slope_model = fit_ground_model(raw_ground, config)
    slope_slot = metric_slot("sloped_ground_vehicle")
    slope_points: list[list[float]] = []
    slope_owners: list[int] = []
    for frame_id in (1, 2, 3, 4):
        for x_value in (-1.0, -0.5, 0.0, 0.5, 1.0):
            for y_value in (-0.5, 0.0, 0.5):
                for height in (0.45, 1.0, 1.6):
                    raw = np.asarray(
                        [
                            [
                                x_value,
                                y_value,
                                0.04 * x_value - 0.02 * y_value - 1.65 + height,
                            ]
                        ]
                    )
                    slope_points.append(
                        [x_value, y_value, float(normalize_z(raw, slope_model)[0]), 1.0]
                    )
                    slope_owners.append(frame_id)
    slope_accumulation = accumulation_from_points(
        slope_slot.slot_id,
        np.asarray(slope_points),
        np.asarray(slope_owners),
        (1, 2, 3, 4),
        ground_model=slope_model,
    )
    slope_slots = {slope_slot.slot_id: slope_slot}
    slope_occupied = evaluate_occupied(
        slope_slot, slope_accumulation, slope_slots, config
    )
    slope_stability = stable_occupied(
        slope_slot,
        slope_accumulation,
        slope_occupied,
        slope_slots,
        config,
    )

    pose_slot = metric_slot("pose_sensitive_vehicle")
    pose_points, pose_owners = vehicle_points(
        (1, 2, 3, 4), y_values=(0.5, 1.0, 1.5)
    )
    pose_accumulation = accumulation_from_points(
        pose_slot.slot_id,
        pose_points,
        pose_owners,
        (1, 2, 3, 4),
    )
    pose_slots = {pose_slot.slot_id: pose_slot}
    pose_occupied = evaluate_occupied(
        pose_slot, pose_accumulation, pose_slots, config
    )
    pose_stability = stable_occupied(
        pose_slot,
        pose_accumulation,
        pose_occupied,
        pose_slots,
        config,
    )

    conflict_slot = metric_slot("occupied_free_conflict", free_geometry=True)
    conflict_points, conflict_owners = vehicle_points((1, 2, 3))
    conflict_targets = np.asarray(
        [[0.125, y_value, 1.10] for y_value in (-0.375, -0.125, 0.125, 0.375)],
        dtype=np.float64,
    )
    conflict_observations: list[FrameObservation] = []
    for frame_id in (1, 2, 3):
        origin_x = -3.0 if frame_id % 2 else 3.0
        conflict_observations.append(
            observation(frame_id, (origin_x, 0.0, 1.5), conflict_targets)
        )
    for frame_id, origin_values in (
        (4, (-3.0, 0.0, 1.5)),
        (5, (3.0, 0.0, 1.5)),
    ):
        origin = np.asarray(origin_values, dtype=np.float64)
        endpoints = origin + 2.0 * (conflict_targets - origin)
        conflict_observations.append(
            observation(frame_id, origin_values, endpoints)
        )
    conflict_accumulation = SlotAccumulation(
        slot_id=conflict_slot.slot_id,
        anchor_frame=3,
        selected_frames=(1, 2, 3, 4, 5),
        points_local_xyzi=conflict_points,
        point_frame_ids=conflict_owners,
        observations=tuple(conflict_observations),
    )
    conflict_occupied = evaluate_occupied(
        conflict_slot,
        conflict_accumulation,
        {conflict_slot.slot_id: conflict_slot},
        config,
    )
    conflict_free = evaluate_free_space(
        conflict_slot,
        conflict_accumulation,
        conflict_occupied,
        config,
    )

    return (
        _scenario(
            "centered_vehicle",
            occupied=centered_occupied,
            stability=centered_stability,
            state=DecisionState.OCCUPIED,
            reason="strong_occupied_evidence",
        ),
        _scenario(
            "adjacent_vehicle",
            occupied=adjacent_occupied,
            state=DecisionState.UNKNOWN,
            reason="weak_obstacle_evidence",
            unknown_reasons=(
                "adjacent_overlap_conflict",
                "boundary_dominated",
                "weak_vehicle_evidence",
            ),
        ),
        _scenario(
            "cross_slot_vehicle",
            occupied=crossing_occupied,
            state=DecisionState.UNKNOWN,
            reason="weak_obstacle_evidence",
            unknown_reasons=("outside_residual_conflict", "weak_vehicle_evidence"),
        ),
        _scenario(
            "wheel_stop",
            occupied=wheel_occupied,
            state=DecisionState.UNKNOWN,
            reason="weak_obstacle_evidence",
            unknown_reasons=("low_height_structure", "weak_vehicle_evidence"),
        ),
        _scenario(
            "wall_or_column",
            occupied=column_occupied,
            state=DecisionState.UNKNOWN,
            reason="weak_obstacle_evidence",
            unknown_reasons=("linear_static_structure", "weak_vehicle_evidence"),
        ),
        _scenario(
            "oversized_merged_structure",
            occupied=merged_occupied,
            state=DecisionState.UNKNOWN,
            reason="weak_obstacle_evidence",
            unknown_reasons=("outside_residual_conflict", "weak_vehicle_evidence"),
        ),
        _scenario(
            "occluded_vehicle",
            free=occluded_free,
            state=DecisionState.UNKNOWN,
            reason="weak_obstacle_evidence",
            unknown_reasons=("weak_obstacle_evidence",),
        ),
        _scenario(
            "fully_observable_empty",
            free=clear_free,
            stability=clear_stability,
            state=DecisionState.FREE,
            reason="strong_free_space_evidence",
        ),
        _scenario(
            "partially_observable_empty",
            free=partial_free,
            state=DecisionState.UNKNOWN,
            reason="insufficient_terminal_evidence",
            unknown_reasons=("insufficient_viewpoints", "low_free_volume_coverage"),
        ),
        _scenario(
            "sloped_ground_vehicle",
            occupied=slope_occupied,
            stability=slope_stability,
            state=DecisionState.OCCUPIED,
            reason="strong_occupied_evidence",
            ground_model=slope_model,
        ),
        _scenario(
            "missing_or_corrupt_frame",
            scope=missing_scope(config),
            state=DecisionState.UNKNOWN,
            reason="partial_route_scope",
            unknown_reasons=("missing_near_route_frames", "partial_route_scope"),
        ),
        _scenario(
            "pose_sensitive_vehicle",
            occupied=pose_occupied,
            stability=pose_stability,
            state=DecisionState.UNKNOWN,
            reason="pose_unstable",
            unknown_reasons=("pose_sensitive_terminal",),
        ),
        _scenario(
            "occupied_free_conflict",
            occupied=conflict_occupied,
            free=conflict_free,
            state=DecisionState.UNKNOWN,
            reason="occupied_free_conflict",
            unknown_reasons=("occupied_free_conflict",),
        ),
        _scenario(
            "out_of_route_slot",
            scope=out_of_route_scope(config),
            state=None,
            reason="",
        ),
    )


__all__ = [
    "SyntheticScenario",
    "clear_observations",
    "core_target_centers",
    "metric_slot",
    "synthetic_scenario_matrix",
    "vehicle_points",
]
