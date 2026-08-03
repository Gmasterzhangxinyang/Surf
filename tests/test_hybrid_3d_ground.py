import unittest

import numpy as np

from parking_slot_hybrid_3d.config import Hybrid3DConfig
from parking_slot_hybrid_3d.ground import fit_ground_model, normalize_z, origin_height


class Hybrid3DGroundTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = Hybrid3DConfig(
            ground_candidate_quantile=0.60,
            ground_min_points=30,
            ground_inlier_threshold_m=0.12,
            ground_min_inlier_ratio=0.45,
            ground_max_residual_p95_m=0.15,
        )

    def test_sloped_ground_is_recovered_without_vehicle_bias(self) -> None:
        xs, ys = np.meshgrid(np.linspace(-3.0, 3.0, 17), np.linspace(-1.5, 1.5, 11))
        x = xs.ravel()
        y = ys.ravel()
        ground_z = 0.04 * x - 0.02 * y - 1.65
        noise = 0.008 * np.sin(np.arange(len(x), dtype=np.float64))
        ground = np.column_stack([x, y, ground_z + noise])
        vehicle_xy = ground[::4, :2]
        vehicle_ground = 0.04 * vehicle_xy[:, 0] - 0.02 * vehicle_xy[:, 1] - 1.65
        vehicle = np.column_stack([vehicle_xy, vehicle_ground + 1.10])
        points = np.vstack([ground, vehicle])

        model = fit_ground_model(points, self.config)
        normalized_vehicle = normalize_z(vehicle, model)

        self.assertTrue(model.valid)
        self.assertEqual(model.method, "plane")
        self.assertAlmostEqual(model.a, 0.04, delta=0.01)
        self.assertAlmostEqual(model.b, -0.02, delta=0.01)
        self.assertLessEqual(model.residual_p95_m, 0.15)
        np.testing.assert_allclose(normalized_vehicle, 1.10, atol=0.05)
        self.assertAlmostEqual(origin_height(np.asarray([0.0, 0.0, 0.0]), model), 1.65, delta=0.05)

    def test_degenerate_plane_uses_constant_quantile_fallback(self) -> None:
        x = np.linspace(-3.0, 3.0, 40)
        points = np.column_stack([x, np.zeros_like(x), np.full_like(x, -1.7)])

        model = fit_ground_model(points, self.config)

        self.assertTrue(model.valid)
        self.assertEqual(model.method, "constant")
        self.assertAlmostEqual(model.c, -1.7)
        self.assertIn("rank_deficient_ground_plane", model.failures)

    def test_insufficient_points_produce_invalid_ground(self) -> None:
        points = np.asarray([[0.0, 0.0, -1.7], [1.0, 0.0, -1.7], [0.0, 1.0, -1.7]])

        model = fit_ground_model(points, self.config)

        self.assertFalse(model.valid)
        self.assertEqual(model.method, "invalid")
        self.assertIn("insufficient_ground_points", model.failures)
        with self.assertRaisesRegex(ValueError, "invalid ground"):
            normalize_z(points, model)

    def test_nonfinite_points_are_excluded_before_fitting(self) -> None:
        xs, ys = np.meshgrid(np.linspace(-2.0, 2.0, 10), np.linspace(-1.0, 1.0, 5))
        finite = np.column_stack([xs.ravel(), ys.ravel(), np.full(xs.size, -1.5)])
        points = np.vstack([finite, [np.nan, 0.0, 0.0]])

        model = fit_ground_model(points, self.config)

        self.assertTrue(model.valid)
        self.assertEqual(model.candidate_count, 50)


if __name__ == "__main__":
    unittest.main()
