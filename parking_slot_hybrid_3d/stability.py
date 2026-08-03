from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass

import numpy as np

from .config import Hybrid3DConfig
from .contracts import (
    DecisionState,
    FrameObservation,
    MetricSlot,
    OccupiedEvidence,
    SlotAccumulation,
    StabilityEvidence,
)
from .free_space import evaluate_free_space
from .occupied import evaluate_fixed_occupied, evaluate_occupied


@dataclass(frozen=True)
class PosePerturbation:
    name: str
    dx_m: float = 0.0
    dy_m: float = 0.0
    dyaw_deg: float = 0.0

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("pose perturbation name is required")
        if not np.isfinite([self.dx_m, self.dy_m, self.dyaw_deg]).all():
            raise ValueError("pose perturbation values must be finite")


def pose_perturbations(config: Hybrid3DConfig) -> tuple[PosePerturbation, ...]:
    config.validate()
    translation = float(config.stability_translation_m)
    yaw = float(config.stability_yaw_deg)
    return (
        PosePerturbation("original"),
        PosePerturbation("dx_plus", dx_m=translation),
        PosePerturbation("dx_minus", dx_m=-translation),
        PosePerturbation("dy_plus", dy_m=translation),
        PosePerturbation("dy_minus", dy_m=-translation),
        PosePerturbation("dyaw_plus", dyaw_deg=yaw),
        PosePerturbation("dyaw_minus", dyaw_deg=-yaw),
    )


def _readonly(array: np.ndarray, dtype: np.dtype | type | None = None) -> np.ndarray:
    result = np.ascontiguousarray(array, dtype=dtype)
    result.setflags(write=False)
    return result


def _transform_xy(
    points_xy: np.ndarray,
    origin_xy: np.ndarray,
    perturbation: PosePerturbation,
) -> np.ndarray:
    points = np.asarray(points_xy, dtype=np.float64)
    result = points.copy()
    if abs(perturbation.dyaw_deg) > 0.0 and len(result):
        angle = math.radians(perturbation.dyaw_deg)
        cosine = math.cos(angle)
        sine = math.sin(angle)
        delta = result - origin_xy
        result = np.column_stack(
            [
                cosine * delta[:, 0] - sine * delta[:, 1],
                sine * delta[:, 0] + cosine * delta[:, 1],
            ]
        ) + origin_xy
    result += np.asarray([perturbation.dx_m, perturbation.dy_m], dtype=np.float64)
    return result


def _transform_xyzi(
    points_xyzi: np.ndarray,
    origin_xy: np.ndarray,
    perturbation: PosePerturbation,
) -> np.ndarray:
    points = np.asarray(points_xyzi, dtype=np.float64).copy()
    if len(points):
        points[:, :2] = _transform_xy(points[:, :2], origin_xy, perturbation)
    return points


def _transform_xyz(
    points_xyz: np.ndarray,
    origin_xy: np.ndarray,
    perturbation: PosePerturbation,
) -> np.ndarray:
    points = np.asarray(points_xyz, dtype=np.float64).copy()
    if len(points):
        points[:, :2] = _transform_xy(points[:, :2], origin_xy, perturbation)
    return points


def apply_pose_perturbation(
    accumulation: SlotAccumulation,
    perturbation: PosePerturbation,
) -> SlotAccumulation:
    observations_by_frame = {
        observation.frame_id: observation for observation in accumulation.observations
    }
    accumulated = np.asarray(accumulation.points_local_xyzi, dtype=np.float64).copy()
    point_frame_ids = np.asarray(accumulation.point_frame_ids, dtype=np.int64)
    if len(accumulated) != len(point_frame_ids):
        raise ValueError("point_frame_ids must align with accumulated points")
    for frame_id in sorted(set(point_frame_ids.tolist())):
        observation = observations_by_frame.get(frame_id)
        if observation is None:
            raise ValueError(f"point provenance references missing observation {frame_id}")
        mask = point_frame_ids == frame_id
        accumulated[mask] = _transform_xyzi(
            accumulated[mask],
            np.asarray(observation.origin_local_xyz[:2], dtype=np.float64),
            perturbation,
        )

    transformed_observations: list[FrameObservation] = []
    translation_xyz = np.asarray(
        [perturbation.dx_m, perturbation.dy_m, 0.0],
        dtype=np.float64,
    )
    for observation in accumulation.observations:
        origin = np.asarray(observation.origin_local_xyz, dtype=np.float64)
        origin_xy = origin[:2]
        transformed_observations.append(
            FrameObservation(
                frame_id=observation.frame_id,
                origin_local_xyz=_readonly(origin + translation_xyz),
                points_local_xyzi=_readonly(
                    _transform_xyzi(
                        observation.points_local_xyzi,
                        origin_xy,
                        perturbation,
                    )
                ),
                ray_endpoints_local_xyz=_readonly(
                    _transform_xyz(
                        observation.ray_endpoints_local_xyz,
                        origin_xy,
                        perturbation,
                    )
                ),
                ground_model=observation.ground_model,
                quality_reasons=observation.quality_reasons,
            )
        )
    return SlotAccumulation(
        slot_id=accumulation.slot_id,
        anchor_frame=accumulation.anchor_frame,
        selected_frames=accumulation.selected_frames,
        points_local_xyzi=_readonly(accumulated),
        point_frame_ids=_readonly(point_frame_ids.copy(), dtype=np.int64),
        observations=tuple(transformed_observations),
        excluded_frames=accumulation.excluded_frames,
    )


StabilityEvaluator = Callable[[MetricSlot, SlotAccumulation], bool]


def _raise_evaluation_error(
    occupied: OccupiedEvidence,
    free_failures: tuple[str, ...] = (),
) -> None:
    failures = tuple(occupied.failures) + tuple(free_failures)
    errors = tuple(code for code in failures if code.endswith("_evaluation_error"))
    if errors:
        raise RuntimeError(",".join(errors))


def terminal_variant_passes(
    metric_slot: MetricSlot,
    accumulation: SlotAccumulation,
    slots_by_id: Mapping[str, MetricSlot],
    config: Hybrid3DConfig,
    *,
    terminal_state: DecisionState,
    reference_occupied: OccupiedEvidence,
    verify_free_context: bool = True,
) -> bool:
    """Re-evaluate a terminal decision under one pose perturbation.

    Occupied uses the reference box as a fast path, then searches only boxes
    associated with that same object.  Free always searches all candidates:
    keeping only an old box cannot prove that no other occupied candidate exists.
    """

    terminal_state = DecisionState(terminal_state)
    if terminal_state is DecisionState.OCCUPIED:
        if not reference_occupied.best_box:
            raise RuntimeError("occupied stability requires a reference box")
        current_occupied = evaluate_fixed_occupied(
            metric_slot,
            accumulation,
            slots_by_id,
            config,
            reference_occupied.best_box,
        )
        _raise_evaluation_error(current_occupied)
        if not current_occupied.strong:
            current_occupied = evaluate_occupied(
                metric_slot,
                accumulation,
                slots_by_id,
                config,
                reference_box=reference_occupied.best_box,
            )
            _raise_evaluation_error(current_occupied)
        if not current_occupied.strong:
            return False
        if not verify_free_context:
            return True
        current_free = evaluate_free_space(
            metric_slot,
            accumulation,
            current_occupied,
            config,
        )
        _raise_evaluation_error(current_occupied, current_free.failures)
        return bool(not current_free.conflict and not current_free.strong)

    if terminal_state is DecisionState.FREE:
        current_occupied = evaluate_occupied(
            metric_slot,
            accumulation,
            slots_by_id,
            config,
        )
        _raise_evaluation_error(current_occupied)
        current_free = evaluate_free_space(
            metric_slot,
            accumulation,
            current_occupied,
            config,
        )
        _raise_evaluation_error(current_occupied, current_free.failures)
        ownership_resolved = bool(
            current_occupied.weak
            and current_free.weak_ownership_resolved
            and not current_free.weak_obstacle
        )
        weak_veto = bool(current_occupied.weak and not ownership_resolved)
        return bool(
            current_free.strong
            and current_free.positive_geometry
            and not current_free.conflict
            and not current_occupied.strong
            and not current_free.weak_obstacle
            and not current_free.unresolved_core_hit
            and not weak_veto
        )

    raise ValueError("stability is only defined for occupied or free terminals")


def evaluate_stability(
    metric_slot: MetricSlot,
    accumulation: SlotAccumulation,
    evaluator: StabilityEvaluator,
    config: Hybrid3DConfig,
) -> StabilityEvidence:
    variants = pose_perturbations(config)
    results: list[tuple[str, bool]] = []
    failures: list[str] = []
    for variant in variants:
        try:
            perturbed = apply_pose_perturbation(accumulation, variant)
            passed = bool(evaluator(metric_slot, perturbed))
        except Exception:
            passed = False
            failures.append(f"stability_evaluation_error:{variant.name}")
        results.append((variant.name, passed))
    passing = sum(passed for _, passed in results)
    total = len(results)
    ratio = passing / total if total else 0.0
    return StabilityEvidence(
        pass_ratio=float(ratio),
        passing_variants=passing,
        total_variants=total,
        stable=not failures and ratio >= config.stability_min_pass_ratio,
        failures=tuple(failures),
        variant_results=tuple(results),
    )
