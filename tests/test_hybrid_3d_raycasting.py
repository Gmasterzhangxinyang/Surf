import unittest

import numpy as np

from parking_slot_hybrid_3d.raycasting import clip_segment_to_convex_prism, traverse_voxels_3d


def rectangle(length: float, width: float) -> np.ndarray:
    return np.asarray(
        [
            [-length / 2.0, -width / 2.0],
            [length / 2.0, -width / 2.0],
            [length / 2.0, width / 2.0],
            [-length / 2.0, width / 2.0],
        ],
        dtype=np.float64,
    )


class Hybrid3DRaycastingTest(unittest.TestCase):
    def test_voxel_traversal_does_not_overshoot_axes_at_clipped_boundaries(self) -> None:
        start = np.asarray([-2.0, -0.14583333, 1.4], dtype=np.float64)
        end = np.asarray([2.0, -0.72916667, 1.0], dtype=np.float64)

        cells = traverse_voxels_3d(start, end, (0.25, 0.25, 0.20))

        self.assertEqual(cells[0], tuple(np.floor(start / np.asarray([0.25, 0.25, 0.20])).astype(int)))
        self.assertEqual(cells[-1], tuple(np.floor(end / np.asarray([0.25, 0.25, 0.20])).astype(int)))
        self.assertEqual(len(cells), len(set(cells)))

    def test_segment_crossing_slot_is_clipped_at_entry_and_exit(self) -> None:
        clipped = clip_segment_to_convex_prism(
            np.asarray([-5.0, 0.0, 1.0]),
            np.asarray([5.0, 0.0, 1.0]),
            rectangle(length=4.0, width=2.0),
            (0.0, 2.2),
        )

        self.assertIsNotNone(clipped)
        np.testing.assert_allclose(clipped.entry_xyz, [-2.0, 0.0, 1.0], atol=1e-9)
        np.testing.assert_allclose(clipped.exit_xyz, [2.0, 0.0, 1.0], atol=1e-9)
        self.assertAlmostEqual(clipped.t_entry, 0.3)
        self.assertAlmostEqual(clipped.t_exit, 0.7)

    def test_segment_ending_before_prism_does_not_cross(self) -> None:
        clipped = clip_segment_to_convex_prism(
            np.asarray([-5.0, 0.0, 1.0]),
            np.asarray([-2.1, 0.0, 1.0]),
            rectangle(length=4.0, width=2.0),
            (0.0, 2.2),
        )

        self.assertIsNone(clipped)

    def test_endpoint_inside_prism_is_a_hit_and_exit_is_endpoint(self) -> None:
        clipped = clip_segment_to_convex_prism(
            np.asarray([-5.0, 0.0, 1.0]),
            np.asarray([0.0, 0.0, 1.0]),
            rectangle(length=4.0, width=2.0),
            (0.0, 2.2),
        )

        self.assertIsNotNone(clipped)
        self.assertAlmostEqual(clipped.t_exit, 1.0)
        np.testing.assert_allclose(clipped.exit_xyz, [0.0, 0.0, 1.0])

    def test_parallel_segment_outside_prism_does_not_cross(self) -> None:
        clipped = clip_segment_to_convex_prism(
            np.asarray([-5.0, 2.0, 1.0]),
            np.asarray([5.0, 2.0, 1.0]),
            rectangle(length=4.0, width=2.0),
            (0.0, 2.2),
        )

        self.assertIsNone(clipped)

    def test_voxel_traversal_has_stable_unique_order(self) -> None:
        start = np.asarray([0.1, 0.1, 0.1])
        end = np.asarray([0.9, 0.9, 0.9])

        cells = traverse_voxels_3d(start, end, (0.25, 0.25, 0.20))

        self.assertEqual(cells, tuple(dict.fromkeys(cells)))
        self.assertEqual(cells, traverse_voxels_3d(start, end, (0.25, 0.25, 0.20)))
        self.assertEqual(cells[0], (0, 0, 0))
        self.assertEqual(cells[-1], (3, 3, 4))

    def test_voxel_traversal_handles_negative_and_zero_length_segments(self) -> None:
        cells = traverse_voxels_3d(
            np.asarray([-0.1, -0.1, -0.1]),
            np.asarray([-0.8, -0.1, 0.4]),
            (0.25, 0.25, 0.20),
        )
        point_cell = traverse_voxels_3d(
            np.asarray([-0.1, -0.1, -0.1]),
            np.asarray([-0.1, -0.1, -0.1]),
            (0.25, 0.25, 0.20),
        )

        self.assertEqual(cells[0], (-1, -1, -1))
        self.assertEqual(cells[-1], (-4, -1, 2))
        self.assertEqual(point_cell, ((-1, -1, -1),))

    def test_nonfinite_segment_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "finite"):
            clip_segment_to_convex_prism(
                np.asarray([np.nan, 0.0, 0.0]),
                np.asarray([1.0, 0.0, 0.0]),
                rectangle(4.0, 2.0),
                (0.0, 2.2),
            )


if __name__ == "__main__":
    unittest.main()
