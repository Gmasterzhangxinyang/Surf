import math
import unittest
from unittest.mock import patch

import numpy as np

from parking_slot_box_scoring.box_hypotheses import CandidateBox
from parking_slot_hybrid_3d.config import Hybrid3DConfig
from parking_slot_hybrid_3d.contracts import (
    DecisionState,
    FreeEvidence,
    OccupiedEvidence,
    SlotAccumulation,
)
from parking_slot_hybrid_3d.occupied import (
    candidate_matches_reference,
    evaluate_occupied,
)
from parking_slot_hybrid_3d.stability import (
    evaluate_stability,
    terminal_variant_passes,
)
from tests.hybrid_3d_fixtures import (
    accumulation_from_points,
    metric_slot,
    vehicle_points,
)


REFERENCE_BOX = (
    ("center_x_m", 0.0),
    ("center_y_m", 0.0),
    ("yaw_rad", 0.0),
    ("length_m", 4.0),
    ("width_m", 2.0),
)

REFITTED_BOX = (
    ("center_x_m", 0.20),
    ("center_y_m", 0.10),
    ("yaw_rad", math.radians(4.0)),
    ("length_m", 4.0),
    ("width_m", 2.0),
)


def box(
    center: tuple[float, float],
    *,
    yaw_deg: float = 0.0,
    length: float = 4.0,
    width: float = 2.0,
) -> CandidateBox:
    return CandidateBox(
        center=np.asarray(center, dtype=np.float64),
        yaw=math.radians(yaw_deg),
        length=length,
        width=width,
    )


def empty_accumulation(slot_id: str) -> SlotAccumulation:
    return SlotAccumulation(
        slot_id=slot_id,
        anchor_frame=1,
        selected_frames=(),
        points_local_xyzi=np.empty((0, 4), dtype=np.float64),
        point_frame_ids=np.empty(0, dtype=np.int64),
        observations=(),
    )


class CandidateAssociationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = Hybrid3DConfig()
        self.reference = box((0.0, 0.0))

    def test_small_pose_compensation_is_the_same_target(self) -> None:
        candidate = box((0.20, 0.10), yaw_deg=4.0)

        self.assertTrue(
            candidate_matches_reference(candidate, self.reference, self.config)
        )

    def test_orientation_is_axial_so_180_degrees_is_the_same_box(self) -> None:
        candidate = box((0.0, 0.0), yaw_deg=180.0)

        self.assertTrue(
            candidate_matches_reference(candidate, self.reference, self.config)
        )

    def test_adjacent_or_orthogonal_target_is_rejected(self) -> None:
        candidates = (
            box((0.0, 2.50)),
            box((2.50, 0.0)),
            box((0.0, 0.0), yaw_deg=90.0),
        )

        for candidate in candidates:
            with self.subTest(candidate=candidate):
                self.assertFalse(
                    candidate_matches_reference(candidate, self.reference, self.config)
                )


class TerminalVariantTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = Hybrid3DConfig()
        self.slot = metric_slot("terminal_stability")
        self.accumulation = empty_accumulation(self.slot.slot_id)
        self.slots = {self.slot.slot_id: self.slot}
        self.reference_occupied = OccupiedEvidence(
            strong=True,
            best_box=REFERENCE_BOX,
        )

    def evaluate(self, state: DecisionState, reference: OccupiedEvidence | None = None) -> bool:
        return terminal_variant_passes(
            self.slot,
            self.accumulation,
            self.slots,
            self.config,
            terminal_state=state,
            reference_occupied=reference or self.reference_occupied,
        )

    def test_fixed_box_success_is_fast_path_but_free_conflict_is_recomputed(self) -> None:
        fixed = OccupiedEvidence(strong=True, best_box=REFERENCE_BOX)
        with (
            patch(
                "parking_slot_hybrid_3d.stability.evaluate_fixed_occupied",
                return_value=fixed,
            ) as fixed_evaluator,
            patch(
                "parking_slot_hybrid_3d.stability.evaluate_occupied",
                side_effect=AssertionError("successful fixed box must not refit"),
            ),
            patch(
                "parking_slot_hybrid_3d.stability.evaluate_free_space",
                return_value=FreeEvidence(conflict=True),
            ) as free_evaluator,
        ):
            passed = self.evaluate(DecisionState.OCCUPIED)

        self.assertFalse(passed)
        fixed_evaluator.assert_called_once_with(
            self.slot,
            self.accumulation,
            self.slots,
            self.config,
            REFERENCE_BOX,
        )
        free_evaluator.assert_called_once_with(
            self.slot,
            self.accumulation,
            fixed,
            self.config,
        )

    def test_fixed_box_failure_uses_constrained_refit_and_recomputes_free(self) -> None:
        fixed = OccupiedEvidence(
            weak=True,
            best_box=REFERENCE_BOX,
            failures=("outside_residual_conflict",),
        )
        refitted = OccupiedEvidence(strong=True, best_box=REFITTED_BOX)
        with (
            patch(
                "parking_slot_hybrid_3d.stability.evaluate_fixed_occupied",
                return_value=fixed,
            ),
            patch(
                "parking_slot_hybrid_3d.stability.evaluate_occupied",
                return_value=refitted,
            ) as refit_evaluator,
            patch(
                "parking_slot_hybrid_3d.stability.evaluate_free_space",
                return_value=FreeEvidence(),
            ) as free_evaluator,
        ):
            passed = self.evaluate(DecisionState.OCCUPIED)

        self.assertTrue(passed)
        refit_evaluator.assert_called_once_with(
            self.slot,
            self.accumulation,
            self.slots,
            self.config,
            reference_box=REFERENCE_BOX,
        )
        free_evaluator.assert_called_once_with(
            self.slot,
            self.accumulation,
            refitted,
            self.config,
        )

    def test_refitted_occupied_is_rejected_when_free_conflict_remains(self) -> None:
        with (
            patch(
                "parking_slot_hybrid_3d.stability.evaluate_fixed_occupied",
                return_value=OccupiedEvidence(
                    weak=True,
                    best_box=REFERENCE_BOX,
                    failures=("outside_residual_conflict",),
                ),
            ),
            patch(
                "parking_slot_hybrid_3d.stability.evaluate_occupied",
                return_value=OccupiedEvidence(strong=True, best_box=REFITTED_BOX),
            ),
            patch(
                "parking_slot_hybrid_3d.stability.evaluate_free_space",
                return_value=FreeEvidence(conflict=True),
            ),
        ):
            self.assertFalse(self.evaluate(DecisionState.OCCUPIED))

    def test_free_terminal_researches_all_candidates_not_only_the_old_box(self) -> None:
        reference = OccupiedEvidence(
            weak=True,
            best_box=REFERENCE_BOX,
            failures=("linear_static_structure",),
        )
        recomputed = OccupiedEvidence(
            weak=True,
            best_box=REFITTED_BOX,
            failures=("linear_static_structure",),
        )
        with (
            patch(
                "parking_slot_hybrid_3d.stability.evaluate_fixed_occupied",
                side_effect=AssertionError("fixed box cannot prove absence of another target"),
            ),
            patch(
                "parking_slot_hybrid_3d.stability.evaluate_occupied",
                return_value=recomputed,
            ) as occupied_evaluator,
            patch(
                "parking_slot_hybrid_3d.stability.evaluate_free_space",
                return_value=FreeEvidence(
                    strong=True,
                    positive_geometry=True,
                    weak_ownership_resolved=True,
                ),
            ) as free_evaluator,
        ):
            passed = self.evaluate(DecisionState.FREE, reference)

        self.assertTrue(passed)
        occupied_evaluator.assert_called_once_with(
            self.slot,
            self.accumulation,
            self.slots,
            self.config,
        )
        free_evaluator.assert_called_once_with(
            self.slot,
            self.accumulation,
            recomputed,
            self.config,
        )

    def test_evaluation_error_is_not_counted_as_an_ordinary_failed_variant(self) -> None:
        with patch(
            "parking_slot_hybrid_3d.stability.evaluate_fixed_occupied",
            return_value=OccupiedEvidence(
                failures=("occupied_evaluation_error",),
            ),
        ):
            with self.assertRaises(RuntimeError):
                self.evaluate(DecisionState.OCCUPIED)

    def test_constrained_refit_recovers_the_pose_sensitive_vehicle_fixture(self) -> None:
        pose_slot = metric_slot("pose_sensitive_refit")
        points, frame_ids = vehicle_points(
            (1, 2, 3, 4),
            y_values=(0.5, 1.0, 1.5),
        )
        accumulation = accumulation_from_points(
            pose_slot.slot_id,
            points,
            frame_ids,
            (1, 2, 3, 4),
        )
        slots = {pose_slot.slot_id: pose_slot}
        reference = evaluate_occupied(
            pose_slot,
            accumulation,
            slots,
            self.config,
        )

        result = evaluate_stability(
            pose_slot,
            accumulation,
            lambda current_slot, current_accumulation: terminal_variant_passes(
                current_slot,
                current_accumulation,
                slots,
                self.config,
                terminal_state=DecisionState.OCCUPIED,
                reference_occupied=reference,
            ),
            self.config,
        )

        self.assertTrue(reference.strong)
        self.assertTrue(result.stable)
        self.assertEqual(result.passing_variants, 7)
        self.assertEqual(result.failures, ())


if __name__ == "__main__":
    unittest.main()
