import unittest
from types import SimpleNamespace

from parking_slot_hybrid_3d.contracts import (
    AgentContext,
    DecisionState,
    FreeEvidence,
    OccupiedEvidence,
    ScopeEvidence,
    ScopeStatus,
    StabilityEvidence,
)
from parking_slot_hybrid_3d.decision import (
    QualityEvidence,
    assess_weak_evidence,
    route_decision,
)


class WeakEvidenceAssessmentTest(unittest.TestCase):
    def test_external_zero_hit_positive_geometry_is_the_only_free_disposition(self) -> None:
        occupied = OccupiedEvidence(
            weak=True,
            failures=("outside_residual_conflict",),
        )
        free = FreeEvidence(
            strong=True,
            positive_geometry=True,
            weak_ownership_resolved=True,
        )

        assessment = assess_weak_evidence(occupied, free)

        self.assertEqual(assessment.ownership, "external")
        self.assertEqual(assessment.disposition, "free_stability")
        self.assertEqual(assessment.primary_reason, "weak_external_ownership")

    def test_even_one_sparse_core_hit_is_agent_only(self) -> None:
        assessment = assess_weak_evidence(
            OccupiedEvidence(
                weak=True,
                failures=("outside_residual_conflict",),
            ),
            FreeEvidence(
                positive_geometry=True,
                unresolved_core_hit=True,
                failures=("unresolved_core_hit_evidence",),
            ),
        )

        self.assertIn("sparse_core_hit", assessment.morphology)
        self.assertEqual(assessment.disposition, "agent_only")
        self.assertEqual(
            assessment.primary_reason,
            "weak_core_clearance_conflict",
        )

    def test_repeatable_core_obstacle_and_shared_static_evidence_remain_auditable(self) -> None:
        core = assess_weak_evidence(
            OccupiedEvidence(weak=True),
            FreeEvidence(
                positive_geometry=True,
                weak_obstacle=True,
                unresolved_core_hit=True,
            ),
        )
        shared = assess_weak_evidence(
            OccupiedEvidence(
                weak=True,
                failures=("boundary_dominated", "linear_static_structure"),
            ),
            FreeEvidence(failures=("ownership_conflict",)),
        )

        self.assertIn("repeatable_core_obstacle", core.morphology)
        self.assertEqual(core.primary_reason, "weak_core_clearance_conflict")
        self.assertEqual(shared.ownership, "shared_or_boundary")
        self.assertIn("linear_static", shared.morphology)
        self.assertEqual(shared.primary_reason, "weak_shared_ownership")

    def test_split_disagreement_is_recorded_but_never_promotes_weak(self) -> None:
        occupied = OccupiedEvidence(
            weak=True,
            features=SimpleNamespace(
                splits=(SimpleNamespace(agreement=True), SimpleNamespace(agreement=False)),
            ),
        )
        assessment = assess_weak_evidence(occupied, FreeEvidence())

        self.assertEqual(assessment.temporal, "inconsistent")
        self.assertEqual(assessment.disposition, "agent_only")

    def test_route_keeps_weak_unknown_and_attaches_the_profile(self) -> None:
        decision = route_decision(
            ScopeEvidence(
                slot_id="slot_weak",
                scope_status=ScopeStatus.IN_ROUTE,
                agent_observable=True,
            ),
            OccupiedEvidence(
                weak=True,
                failures=("linear_static_structure",),
            ),
            FreeEvidence(),
            QualityEvidence(),
            StabilityEvidence(),
            AgentContext(agent_observable=True),
        )

        self.assertEqual(decision.state, DecisionState.UNKNOWN)
        self.assertTrue(decision.weak_evidence.active)
        self.assertEqual(decision.weak_evidence.disposition, "agent_only")
        self.assertIn(
            decision.weak_evidence.primary_reason,
            decision.unknown_reasons,
        )


if __name__ == "__main__":
    unittest.main()
