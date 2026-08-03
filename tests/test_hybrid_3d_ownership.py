import unittest
from types import SimpleNamespace

import numpy as np

from parking_slot_hybrid_3d.config import Hybrid3DConfig
from parking_slot_hybrid_3d.contracts import (
    AgentContext,
    DecisionState,
    FrameObservation,
    FreeEvidence,
    GroundModel,
    MetricSlot,
    OccupiedEvidence,
    ScopeEvidence,
    ScopeStatus,
    SlotAccumulation,
    StabilityEvidence,
)
from parking_slot_hybrid_3d.decision import QualityEvidence, route_decision
from parking_slot_hybrid_3d.free_space import evaluate_free_space


OWNERSHIP_FAILURE_CASES = (
    ("adjacent_overlap_conflict",),
    ("boundary_dominated", "linear_static_structure"),
    ("outside_residual_conflict", "linear_static_structure"),
)


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


def slot() -> MetricSlot:
    return MetricSlot(
        slot_id="slot_0001",
        center_map=np.asarray([0.0, 0.0]),
        long_axis_map=np.asarray([1.0, 0.0]),
        short_axis_map=np.asarray([0.0, 1.0]),
        polygon_local_m=rectangle(2.2, 1.2),
        core_polygon_local_m=rectangle(2.0, 1.0),
        margin_polygon_local_m=rectangle(2.5, 1.5),
        adjacent_slots=(),
        map_units_per_meter=1.0,
    )


def observation(
    frame_id: int,
    origin: np.ndarray,
    endpoints: np.ndarray,
) -> FrameObservation:
    return FrameObservation(
        frame_id=frame_id,
        origin_local_xyz=origin,
        points_local_xyzi=np.empty((0, 4), dtype=np.float64),
        ray_endpoints_local_xyz=endpoints,
        ground_model=GroundModel(method="plane", valid=True),
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


def clear_accumulation(*, add_core_obstacle: bool = False) -> SlotAccumulation:
    observations: list[FrameObservation] = []
    targets = core_target_centers()
    for frame_id in (1, 2, 3, 4, 5):
        from_left = frame_id % 2 == 1
        origin = np.asarray(
            [-3.0 if from_left else 3.0, 0.0, 1.5],
            dtype=np.float64,
        )
        if frame_id <= 2:
            endpoints = origin + 5.0 * (targets - origin)
        else:
            endpoints = np.asarray(
                [[3.0 if from_left else -3.0, 0.0, 1.0]],
                dtype=np.float64,
            )
        if add_core_obstacle and frame_id <= 2:
            core_hits = np.repeat(
                np.asarray([[0.0, 0.0, 1.0]], dtype=np.float64),
                5,
                axis=0,
            )
            endpoints = np.vstack([endpoints, core_hits])
        observations.append(observation(frame_id, origin, endpoints))

    return SlotAccumulation(
        slot_id="slot_0001",
        anchor_frame=3,
        selected_frames=(1, 2, 3, 4, 5),
        points_local_xyzi=np.empty((0, 4), dtype=np.float64),
        point_frame_ids=np.empty(0, dtype=np.int64),
        observations=tuple(observations),
    )


def sparse_core_hit_accumulation() -> SlotAccumulation:
    result = clear_accumulation()
    observations = list(result.observations)
    first = observations[0]
    observations[0] = observation(
        first.frame_id,
        np.asarray(first.origin_local_xyz),
        np.vstack(
            [first.ray_endpoints_local_xyz, np.asarray([[0.0, 0.0, 1.0]])]
        ),
    )
    return SlotAccumulation(
        slot_id=result.slot_id,
        anchor_frame=result.anchor_frame,
        selected_frames=result.selected_frames,
        points_local_xyzi=result.points_local_xyzi,
        point_frame_ids=result.point_frame_ids,
        observations=tuple(observations),
    )


def scope() -> ScopeEvidence:
    return ScopeEvidence(
        slot_id="slot_0001",
        scope_status=ScopeStatus.IN_ROUTE,
        crossing_frames=(1, 2, 3, 4, 5),
        agent_observable=True,
    )


def stability(value: bool = True) -> StabilityEvidence:
    return StabilityEvidence(
        pass_ratio=1.0 if value else 5 / 7,
        passing_variants=7 if value else 5,
        total_variants=7,
        stable=value,
    )


class Hybrid3DOwnershipTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = Hybrid3DConfig()

    def test_outside_owned_weak_candidate_does_not_veto_proven_core_free(self) -> None:
        for failures in OWNERSHIP_FAILURE_CASES:
            with self.subTest(failures=failures):
                result = evaluate_free_space(
                    slot(),
                    clear_accumulation(),
                    OccupiedEvidence(
                        weak=True,
                        core_ownership=0.0,
                        failures=failures,
                        features=SimpleNamespace(
                            core_point_count=0,
                            core_voxel_count=0,
                            core_supported_frame_count=0,
                        ),
                    ),
                    self.config,
                )

                self.assertTrue(result.strong)
                self.assertFalse(result.weak_obstacle)
                self.assertTrue(result.weak_ownership_resolved)
                self.assertNotIn("weak_occupied_evidence", result.failures)
                self.assertNotIn("ownership_conflict", result.failures)

    def test_outside_owned_weak_candidate_reaches_free_after_stability(self) -> None:
        occupied = OccupiedEvidence(
            weak=True,
            core_ownership=0.0,
            failures=("outside_residual_conflict", "linear_static_structure"),
        )
        free = FreeEvidence(
            strong=True,
            positive_geometry=True,
            weak_ownership_resolved=True,
        )

        terminal = route_decision(
            scope(), occupied, free, QualityEvidence(), stability(), AgentContext()
        )
        unstable = route_decision(
            scope(),
            occupied,
            free,
            QualityEvidence(),
            stability(False),
            AgentContext(),
        )

        self.assertEqual(terminal.state, DecisionState.FREE)
        self.assertEqual(terminal.decision_reason, "strong_free_space_evidence")
        self.assertEqual(unstable.state, DecisionState.UNKNOWN)
        self.assertEqual(unstable.decision_reason, "pose_unstable")

    def test_one_frame_core_hit_blocks_external_ownership_free_recovery(self) -> None:
        result = evaluate_free_space(
            slot(),
            sparse_core_hit_accumulation(),
            OccupiedEvidence(
                weak=True,
                failures=("outside_residual_conflict",),
                features=SimpleNamespace(
                    core_point_count=0,
                    core_voxel_count=0,
                    core_supported_frame_count=0,
                ),
            ),
            self.config,
        )

        self.assertTrue(result.positive_geometry)
        self.assertTrue(result.unresolved_core_hit)
        self.assertFalse(result.weak_obstacle)
        self.assertFalse(result.weak_ownership_resolved)
        self.assertFalse(result.strong)
        self.assertIn("unresolved_core_hit_evidence", result.failures)
        self.assertEqual(result.details.core_hit_frame_count, 1)

    def test_legacy_ownership_failure_without_core_provenance_stays_unknown(self) -> None:
        occupied = OccupiedEvidence(
            weak=True,
            failures=("boundary_dominated",),
        )
        free = evaluate_free_space(
            slot(), clear_accumulation(), occupied, self.config
        )

        self.assertFalse(free.strong)
        self.assertFalse(free.weak_ownership_resolved)
        self.assertIn("weak_occupied_evidence", free.failures)

    def test_core_owned_weak_candidate_still_vetoes_free(self) -> None:
        result = evaluate_free_space(
            slot(),
            clear_accumulation(),
            OccupiedEvidence(
                weak=True,
                core_ownership=0.80,
                failures=("linear_static_structure",),
            ),
            self.config,
        )

        self.assertFalse(result.strong)
        self.assertIn("weak_occupied_evidence", result.failures)

    def test_multiframe_core_support_keeps_ownership_failure_conservative(self) -> None:
        result = evaluate_free_space(
            slot(),
            clear_accumulation(),
            OccupiedEvidence(
                weak=True,
                failures=("boundary_dominated",),
                features=SimpleNamespace(
                    core_point_count=10,
                    core_voxel_count=4,
                    core_supported_frame_count=2,
                ),
            ),
            self.config,
        )

        self.assertFalse(result.strong)
        self.assertIn("weak_occupied_evidence", result.failures)
        self.assertIn("ownership_conflict", result.failures)

    def test_real_core_weak_obstacle_still_vetoes_outside_owned_candidate(self) -> None:
        result = evaluate_free_space(
            slot(),
            clear_accumulation(add_core_obstacle=True),
            OccupiedEvidence(
                weak=True,
                core_ownership=0.0,
                failures=("outside_residual_conflict",),
            ),
            self.config,
        )

        self.assertFalse(result.strong)
        self.assertTrue(result.weak_obstacle)
        self.assertIn("weak_obstacle_evidence", result.failures)


if __name__ == "__main__":
    unittest.main()
