import unittest

import numpy as np

from parking_slot_hybrid_3d.config import Hybrid3DConfig
from parking_slot_hybrid_3d.contracts import (
    FrameObservation,
    GroundModel,
    MetricSlot,
    SlotAccumulation,
)
from parking_slot_hybrid_3d.stability import (
    PosePerturbation,
    apply_pose_perturbation,
    evaluate_stability,
    pose_perturbations,
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
        polygon_local_m=rectangle(2.0, 1.0),
        core_polygon_local_m=rectangle(1.8, 0.8),
        margin_polygon_local_m=rectangle(2.5, 1.5),
        adjacent_slots=(),
        map_units_per_meter=1.0,
    )


def accumulation() -> SlotAccumulation:
    points = np.asarray(
        [[2.0, 1.0, 0.5, 7.0], [1.0, 2.0, 1.0, 8.0]],
        dtype=np.float64,
    )
    observations = (
        FrameObservation(
            frame_id=1,
            origin_local_xyz=np.asarray([1.0, 1.0, 1.5]),
            points_local_xyzi=points.copy(),
            ray_endpoints_local_xyz=np.asarray([[2.0, 1.0, 0.5], [1.0, 2.0, 1.0]]),
            ground_model=GroundModel(method="plane", valid=True),
        ),
    )
    return SlotAccumulation(
        slot_id="slot_0001",
        anchor_frame=1,
        selected_frames=(1,),
        points_local_xyzi=points,
        point_frame_ids=np.asarray([1, 1], dtype=np.int64),
        observations=observations,
    )


class Hybrid3DStabilityTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = Hybrid3DConfig()

    def test_pose_perturbations_are_exactly_the_seven_approved_variants(self) -> None:
        variants = pose_perturbations(self.config)

        self.assertEqual(
            tuple((item.name, item.dx_m, item.dy_m, item.dyaw_deg) for item in variants),
            (
                ("original", 0.0, 0.0, 0.0),
                ("dx_plus", 0.20, 0.0, 0.0),
                ("dx_minus", -0.20, 0.0, 0.0),
                ("dy_plus", 0.0, 0.20, 0.0),
                ("dy_minus", 0.0, -0.20, 0.0),
                ("dyaw_plus", 0.0, 0.0, 0.50),
                ("dyaw_minus", 0.0, 0.0, -0.50),
            ),
        )

    def test_translation_moves_points_origins_and_endpoints_without_mutating_source(self) -> None:
        source = accumulation()
        source_points = source.points_local_xyzi.copy()
        source_origin = source.observations[0].origin_local_xyz.copy()
        source_endpoints = source.observations[0].ray_endpoints_local_xyz.copy()

        shifted = apply_pose_perturbation(
            source,
            PosePerturbation("custom", dx_m=0.20, dy_m=-0.10),
        )

        np.testing.assert_allclose(shifted.points_local_xyzi[:, :2], source_points[:, :2] + [0.20, -0.10])
        np.testing.assert_allclose(shifted.observations[0].origin_local_xyz[:2], source_origin[:2] + [0.20, -0.10])
        np.testing.assert_allclose(shifted.observations[0].ray_endpoints_local_xyz[:, :2], source_endpoints[:, :2] + [0.20, -0.10])
        np.testing.assert_array_equal(source.points_local_xyzi, source_points)
        np.testing.assert_array_equal(source.observations[0].origin_local_xyz, source_origin)
        np.testing.assert_array_equal(source.observations[0].ray_endpoints_local_xyz, source_endpoints)
        self.assertFalse(shifted.points_local_xyzi.flags.writeable)

    def test_yaw_rotates_each_frame_around_its_own_lidar_origin(self) -> None:
        source = accumulation()

        rotated = apply_pose_perturbation(
            source,
            PosePerturbation("quarter_turn", dyaw_deg=90.0),
        )

        np.testing.assert_allclose(rotated.observations[0].origin_local_xyz, [1.0, 1.0, 1.5], atol=1e-12)
        np.testing.assert_allclose(rotated.observations[0].ray_endpoints_local_xyz[0, :2], [1.0, 2.0], atol=1e-12)
        np.testing.assert_allclose(rotated.points_local_xyzi[0, :2], [1.0, 2.0], atol=1e-12)
        self.assertEqual(rotated.points_local_xyzi[0, 3], 7.0)

    def test_pass_ratio_is_exactly_passing_variants_over_seven(self) -> None:
        calls = iter((True, True, True, True, True, True, False))

        result = evaluate_stability(
            slot(),
            accumulation(),
            lambda _slot, _accumulation: next(calls),
            self.config,
        )

        self.assertEqual(result.passing_variants, 6)
        self.assertEqual(result.total_variants, 7)
        self.assertEqual(result.pass_ratio, 6 / 7)
        self.assertTrue(result.stable)
        self.assertEqual(len(result.variant_results), 7)

    def test_five_of_seven_and_evaluator_error_are_unstable(self) -> None:
        calls = iter((True, True, True, True, True, False, RuntimeError("bad variant")))

        def evaluator(_slot: MetricSlot, _accumulation: SlotAccumulation) -> bool:
            value = next(calls)
            if isinstance(value, Exception):
                raise value
            return value

        result = evaluate_stability(slot(), accumulation(), evaluator, self.config)

        self.assertEqual(result.pass_ratio, 5 / 7)
        self.assertFalse(result.stable)
        self.assertIn("stability_evaluation_error:dyaw_minus", result.failures)

    def test_one_evaluator_error_cannot_be_hidden_by_six_passing_variants(self) -> None:
        calls = iter((True, True, True, True, True, True, RuntimeError("bad variant")))

        def evaluator(_slot: MetricSlot, _accumulation: SlotAccumulation) -> bool:
            value = next(calls)
            if isinstance(value, Exception):
                raise value
            return value

        result = evaluate_stability(slot(), accumulation(), evaluator, self.config)

        self.assertEqual(result.pass_ratio, 6 / 7)
        self.assertFalse(result.stable)
        self.assertEqual(result.failures, ("stability_evaluation_error:dyaw_minus",))


if __name__ == "__main__":
    unittest.main()
