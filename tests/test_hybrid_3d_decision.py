import unittest

from parking_slot_hybrid_3d.contracts import (
    AgentContext,
    DecisionState,
    FreeEvidence,
    OccupiedEvidence,
    ScopeEvidence,
    ScopeStatus,
    StabilityEvidence,
)
from parking_slot_hybrid_3d.decision import QualityEvidence, route_decision


def scope(status: ScopeStatus = ScopeStatus.IN_ROUTE) -> ScopeEvidence:
    return ScopeEvidence(
        slot_id="slot_0001",
        scope_status=status,
        crossing_frames=(1, 2, 3),
        agent_observable=True,
    )


def stable(value: bool = True) -> StabilityEvidence:
    return StabilityEvidence(
        pass_ratio=1.0 if value else 5 / 7,
        passing_variants=7 if value else 5,
        total_variants=7,
        stable=value,
    )


class Hybrid3DDecisionTest(unittest.TestCase):
    def test_out_of_route_has_no_decision_record(self) -> None:
        result = route_decision(
            scope(ScopeStatus.OUT_OF_ROUTE),
            OccupiedEvidence(),
            FreeEvidence(),
            QualityEvidence(),
            stable(),
            AgentContext(agent_observable=True),
        )

        self.assertIsNone(result)

    def test_partial_route_precedes_all_terminal_evidence(self) -> None:
        result = route_decision(
            scope(ScopeStatus.PARTIAL_ROUTE),
            OccupiedEvidence(strong=True),
            FreeEvidence(strong=True, positive_geometry=True),
            QualityEvidence(),
            stable(),
            AgentContext(),
        )

        self.assertEqual(result.state, DecisionState.UNKNOWN)
        self.assertEqual(result.decision_reason, "partial_route_scope")

    def test_quality_failure_forces_unknown(self) -> None:
        result = route_decision(
            scope(),
            OccupiedEvidence(strong=True),
            FreeEvidence(),
            QualityEvidence(passed=False, failures=("bad_ground",)),
            stable(),
            AgentContext(),
        )

        self.assertEqual(result.state, DecisionState.UNKNOWN)
        self.assertIn("bad_ground", result.unknown_reasons)

    def test_strong_occupied_requires_stability_and_no_free_conflict(self) -> None:
        occupied = route_decision(
            scope(),
            OccupiedEvidence(strong=True),
            FreeEvidence(),
            QualityEvidence(),
            stable(),
            AgentContext(),
        )
        unstable = route_decision(
            scope(),
            OccupiedEvidence(strong=True),
            FreeEvidence(),
            QualityEvidence(),
            stable(False),
            AgentContext(),
        )
        conflict = route_decision(
            scope(),
            OccupiedEvidence(strong=True),
            FreeEvidence(strong=True, positive_geometry=True, conflict=True),
            QualityEvidence(),
            stable(),
            AgentContext(),
        )

        self.assertEqual(occupied.state, DecisionState.OCCUPIED)
        self.assertEqual(unstable.state, DecisionState.UNKNOWN)
        self.assertIn("pose_sensitive_terminal", unstable.unknown_reasons)
        self.assertEqual(conflict.state, DecisionState.UNKNOWN)
        self.assertEqual(conflict.decision_reason, "occupied_free_conflict")

    def test_stability_error_forces_unknown_even_if_stable_flag_is_true(self) -> None:
        result = route_decision(
            scope(),
            OccupiedEvidence(strong=True),
            FreeEvidence(),
            QualityEvidence(),
            StabilityEvidence(
                pass_ratio=6 / 7,
                passing_variants=6,
                total_variants=7,
                stable=True,
                failures=("stability_evaluation_error:dyaw_minus",),
            ),
            AgentContext(),
        )

        self.assertEqual(result.state, DecisionState.UNKNOWN)
        self.assertEqual(result.decision_reason, "stability_evaluation_error")
        self.assertIn("stability_evaluation_error:dyaw_minus", result.unknown_reasons)

    def test_any_weak_vehicle_or_obstacle_evidence_precedes_strong_free(self) -> None:
        weak_vehicle = route_decision(
            scope(),
            OccupiedEvidence(weak=True),
            FreeEvidence(strong=True, positive_geometry=True),
            QualityEvidence(),
            stable(),
            AgentContext(),
        )
        weak_obstacle = route_decision(
            scope(),
            OccupiedEvidence(),
            FreeEvidence(strong=True, positive_geometry=True, weak_obstacle=True),
            QualityEvidence(),
            stable(),
            AgentContext(),
        )

        self.assertEqual(weak_vehicle.state, DecisionState.UNKNOWN)
        self.assertEqual(weak_obstacle.state, DecisionState.UNKNOWN)
        self.assertNotEqual(weak_vehicle.state, DecisionState.FREE)
        self.assertNotEqual(weak_obstacle.state, DecisionState.FREE)

    def test_free_requires_positive_proof_and_pose_stability(self) -> None:
        free = route_decision(
            scope(),
            OccupiedEvidence(),
            FreeEvidence(strong=True, positive_geometry=True),
            QualityEvidence(),
            stable(),
            AgentContext(),
        )
        no_proof = route_decision(
            scope(),
            OccupiedEvidence(),
            FreeEvidence(strong=False),
            QualityEvidence(),
            stable(),
            AgentContext(),
        )
        unstable = route_decision(
            scope(),
            OccupiedEvidence(),
            FreeEvidence(strong=True, positive_geometry=True),
            QualityEvidence(),
            stable(False),
            AgentContext(),
        )

        self.assertEqual(free.state, DecisionState.FREE)
        self.assertEqual(no_proof.state, DecisionState.UNKNOWN)
        self.assertEqual(unstable.state, DecisionState.UNKNOWN)

    def test_adjacent_boundary_or_merged_conflict_stays_unknown(self) -> None:
        for reason in (
            "adjacent_overlap_conflict",
            "boundary_dominated",
            "outside_residual_conflict",
        ):
            with self.subTest(reason=reason):
                result = route_decision(
                    scope(),
                    OccupiedEvidence(weak=True, failures=(reason,)),
                    FreeEvidence(),
                    QualityEvidence(),
                    stable(),
                    AgentContext(),
                )
                self.assertEqual(result.state, DecisionState.UNKNOWN)
                self.assertIn(reason, result.unknown_reasons)

    def test_agent_observability_never_changes_first_part_state(self) -> None:
        observable = route_decision(
            scope(), OccupiedEvidence(), FreeEvidence(), QualityEvidence(), stable(), AgentContext(agent_observable=True)
        )
        unobservable = route_decision(
            scope(), OccupiedEvidence(), FreeEvidence(), QualityEvidence(), stable(), AgentContext(agent_observable=False)
        )

        self.assertEqual(observable.state, unobservable.state)
        self.assertEqual(observable.state, DecisionState.UNKNOWN)

    def test_inconsistent_strong_evidence_contract_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "strong free evidence contains failed gates"):
            route_decision(
                scope(),
                OccupiedEvidence(),
                FreeEvidence(strong=True, failures=("low_free_volume_coverage",)),
                QualityEvidence(),
                stable(),
                AgentContext(),
            )

        with self.assertRaisesRegex(ValueError, "unresolved core hit"):
            route_decision(
                scope(),
                OccupiedEvidence(),
                FreeEvidence(strong=True, unresolved_core_hit=True),
                QualityEvidence(),
                stable(),
                AgentContext(),
            )

        with self.assertRaisesRegex(ValueError, "lacks positive free geometry"):
            route_decision(
                scope(),
                OccupiedEvidence(),
                FreeEvidence(strong=True),
                QualityEvidence(),
                stable(),
                AgentContext(),
            )


if __name__ == "__main__":
    unittest.main()
