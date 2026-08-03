from __future__ import annotations

import unittest

from parking_slot_hybrid_3d.config import Hybrid3DConfig
from parking_slot_hybrid_3d.contracts import DecisionState, ScopeStatus
from parking_slot_hybrid_3d.decision import route_decision
from tests.hybrid_3d_fixtures import synthetic_scenario_matrix


class Hybrid3DSyntheticScenarioTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.scenarios = synthetic_scenario_matrix(Hybrid3DConfig())
        cls.by_name = {scenario.name: scenario for scenario in cls.scenarios}

    def test_matrix_covers_every_approved_scene(self) -> None:
        self.assertEqual(
            set(self.by_name),
            {
                "centered_vehicle",
                "adjacent_vehicle",
                "cross_slot_vehicle",
                "wheel_stop",
                "wall_or_column",
                "oversized_merged_structure",
                "occluded_vehicle",
                "fully_observable_empty",
                "partially_observable_empty",
                "sloped_ground_vehicle",
                "missing_or_corrupt_frame",
                "pose_sensitive_vehicle",
                "occupied_free_conflict",
                "out_of_route_slot",
            },
        )
        self.assertEqual(len(self.scenarios), len(self.by_name))

    def test_every_scene_has_an_explicit_scope_state_and_reason_contract(self) -> None:
        for scenario in self.scenarios:
            with self.subTest(scene=scenario.name):
                self.assertEqual(scenario.scope.scope_status, scenario.expected_scope)
                decision = route_decision(
                    scenario.scope,
                    scenario.occupied,
                    scenario.free,
                    scenario.quality,
                    scenario.stability,
                    scenario.agent_context,
                )
                if scenario.expected_state is None:
                    self.assertIsNone(decision)
                    self.assertEqual(scenario.expected_scope, ScopeStatus.OUT_OF_ROUTE)
                    continue
                self.assertIsNotNone(decision)
                self.assertEqual(decision.state, scenario.expected_state)
                self.assertEqual(decision.decision_reason, scenario.expected_decision_reason)
                self.assertTrue(
                    set(scenario.expected_unknown_reasons) <= set(decision.unknown_reasons),
                    (scenario.name, scenario.expected_unknown_reasons, decision.unknown_reasons),
                )

    def test_matrix_evidence_is_computed_from_the_intended_3d_fixture(self) -> None:
        self.assertTrue(self.by_name["centered_vehicle"].occupied.strong)
        self.assertTrue(self.by_name["centered_vehicle"].stability.stable)

        adjacent = self.by_name["adjacent_vehicle"].occupied
        self.assertTrue(adjacent.weak)
        self.assertIn("adjacent_overlap_conflict", adjacent.failures)
        self.assertIn("boundary_dominated", adjacent.failures)

        crossing = self.by_name["cross_slot_vehicle"].occupied
        self.assertTrue(crossing.weak)
        self.assertIn("outside_residual_conflict", crossing.failures)

        wheel_stop = self.by_name["wheel_stop"].occupied
        self.assertTrue(wheel_stop.weak)
        self.assertIn("low_height_structure", wheel_stop.failures)

        column = self.by_name["wall_or_column"].occupied
        self.assertTrue(column.weak)
        self.assertIn("linear_static_structure", column.failures)

        merged = self.by_name["oversized_merged_structure"].occupied
        self.assertTrue(merged.weak)
        self.assertIn("outside_residual_conflict", merged.failures)

        occluded = self.by_name["occluded_vehicle"].free
        self.assertTrue(occluded.weak_obstacle)
        self.assertFalse(occluded.strong)

        clear = self.by_name["fully_observable_empty"].free
        self.assertTrue(clear.strong)
        self.assertTrue(self.by_name["fully_observable_empty"].stability.stable)

        partial = self.by_name["partially_observable_empty"].free
        self.assertFalse(partial.strong)
        self.assertIn("insufficient_viewpoints", partial.failures)
        self.assertIn("low_free_volume_coverage", partial.failures)

        slope = self.by_name["sloped_ground_vehicle"]
        self.assertIsNotNone(slope.ground_model)
        self.assertTrue(slope.ground_model.valid)
        self.assertEqual(slope.ground_model.method, "plane")
        self.assertAlmostEqual(slope.ground_model.a, 0.04, delta=0.01)
        self.assertAlmostEqual(slope.ground_model.b, -0.02, delta=0.01)
        self.assertTrue(slope.occupied.strong)

        missing = self.by_name["missing_or_corrupt_frame"]
        self.assertEqual(missing.scope.scope_status, ScopeStatus.PARTIAL_ROUTE)
        self.assertIn("missing_near_route_frames", missing.scope.reasons)

        pose_sensitive = self.by_name["pose_sensitive_vehicle"]
        self.assertTrue(pose_sensitive.occupied.strong)
        self.assertFalse(pose_sensitive.stability.stable)
        self.assertLess(pose_sensitive.stability.pass_ratio, 0.80)

        conflict = self.by_name["occupied_free_conflict"]
        self.assertTrue(conflict.occupied.strong)
        self.assertTrue(conflict.free.conflict)

    def test_all_terminal_synthetic_states_satisfy_decision_invariants(self) -> None:
        terminal = {DecisionState.OCCUPIED, DecisionState.FREE}
        for scenario in self.scenarios:
            if scenario.expected_state not in terminal:
                continue
            with self.subTest(scene=scenario.name):
                if scenario.expected_state is DecisionState.OCCUPIED:
                    self.assertTrue(scenario.occupied.strong)
                    self.assertEqual(scenario.occupied.failures, ())
                else:
                    self.assertTrue(scenario.free.strong)
                    self.assertFalse(scenario.free.weak_obstacle)
                    self.assertFalse(scenario.free.conflict)
                    self.assertEqual(scenario.free.failures, ())
                self.assertTrue(scenario.stability.stable)
                self.assertGreaterEqual(scenario.stability.pass_ratio, 0.80)


if __name__ == "__main__":
    unittest.main()
