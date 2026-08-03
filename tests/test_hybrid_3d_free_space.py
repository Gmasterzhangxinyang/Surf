import unittest

import numpy as np

from parking_slot_hybrid_3d.config import Hybrid3DConfig
from parking_slot_hybrid_3d.contracts import (
    FrameObservation,
    GroundModel,
    MetricSlot,
    OccupiedEvidence,
    SlotAccumulation,
)
from parking_slot_hybrid_3d.free_space import evaluate_free_space


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


def observation(frame_id: int, origin: tuple[float, float, float], endpoints: np.ndarray) -> FrameObservation:
    return FrameObservation(
        frame_id=frame_id,
        origin_local_xyz=np.asarray(origin, dtype=np.float64),
        points_local_xyzi=np.empty((0, 4), dtype=np.float64),
        ray_endpoints_local_xyz=np.asarray(endpoints, dtype=np.float64).reshape(-1, 3),
        ground_model=GroundModel(method="plane", valid=True),
    )


def accumulation(observations: tuple[FrameObservation, ...]) -> SlotAccumulation:
    frame_ids = tuple(item.frame_id for item in observations)
    return SlotAccumulation(
        slot_id="slot_0001",
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
        [[x_value, y_value, z_value] for x_value in x_values for y_value in y_values for z_value in z_values],
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


def occupied_box() -> OccupiedEvidence:
    """A strong, spatially located vehicle hypothesis for conflict tests."""

    return OccupiedEvidence(
        strong=True,
        best_box=(
            ("center_x_m", 0.0),
            ("center_y_m", 0.0),
            ("yaw_rad", 0.0),
            ("length_m", 2.0),
            ("width_m", 1.0),
        ),
    )


def connected_vehicle_targets(count: int = 4) -> np.ndarray:
    """Centers of adjacent 25-cm voxels inside the candidate vehicle box."""

    y_values = (-0.375, -0.125, 0.125, 0.375)[:count]
    return np.asarray([[0.125, y_value, 1.10] for y_value in y_values])


def hit_then_clear_observations(
    *,
    target_count: int = 4,
    hit_frames: tuple[int, ...] = (1, 2),
    clear_origins: tuple[tuple[float, float, float], ...] = (
        (-3.0, 0.0, 1.5),
        (3.0, 0.0, 1.5),
    ),
) -> tuple[FrameObservation, ...]:
    """Historical hits followed by rays that return beyond those exact voxels."""

    targets = connected_vehicle_targets(target_count)
    observations: list[FrameObservation] = []
    for frame_id in hit_frames:
        origin_x = -3.0 if frame_id % 2 else 3.0
        observations.append(
            observation(frame_id, (origin_x, 0.0, 1.5), targets)
        )
    next_frame = max(hit_frames, default=0) + 1
    for offset, origin_values in enumerate(clear_origins):
        origin = np.asarray(origin_values, dtype=np.float64)
        # Each measured return lies beyond the historical hit voxel, so the
        # ray contributes free support at the target rather than merely ending
        # on its first return.
        endpoints = origin + 2.0 * (targets - origin)
        observations.append(
            observation(next_frame + offset, origin_values, endpoints)
        )
    return tuple(observations)


class Hybrid3DFreeSpaceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = Hybrid3DConfig()

    def test_five_frames_and_two_viewpoints_prove_clear_core_volume(self) -> None:
        result = evaluate_free_space(
            slot(),
            accumulation(clear_observations()),
            OccupiedEvidence(),
            self.config,
        )

        self.assertTrue(result.strong)
        self.assertEqual(result.ray_frame_count, 5)
        self.assertEqual(result.viewpoint_count, 2)
        self.assertGreaterEqual(result.observed_volume_ratio, 0.70)
        self.assertGreaterEqual(result.near_ground_bev_coverage, 0.70)
        self.assertIsNotNone(result.details)

    def test_foreground_endpoints_occlude_but_never_free_the_slot_behind_them(self) -> None:
        observations: list[FrameObservation] = []
        targets = core_target_centers()
        for frame_id in (1, 2):
            from_left = frame_id % 2 == 1
            origin_x = -3.0 if from_left else 3.0
            origin = np.asarray([origin_x, 0.0, 1.5], dtype=np.float64)
            foreground_endpoints = origin + 0.10 * (targets - origin)
            observations.append(
                observation(frame_id, tuple(origin.tolist()), foreground_endpoints)
            )

        result = evaluate_free_space(
            slot(),
            accumulation(tuple(observations)),
            OccupiedEvidence(),
            self.config,
        )

        self.assertFalse(result.strong)
        self.assertEqual(result.details.free_voxels, ())
        self.assertGreater(len(result.details.occluded_voxels), 0)
        self.assertGreater(result.occlusion_ratio, self.config.free_max_occlusion_ratio)

    def test_repeatable_vehicle_height_hit_is_a_weak_obstacle_veto(self) -> None:
        observations = list(clear_observations())
        for index in (0, 1):
            current = observations[index]
            obstacle_hits = np.repeat(np.asarray([[0.0, 0.0, 1.0]]), 5, axis=0)
            observations[index] = observation(
                current.frame_id,
                tuple(current.origin_local_xyz.tolist()),
                np.vstack([current.ray_endpoints_local_xyz, obstacle_hits]),
            )

        result = evaluate_free_space(
            slot(),
            accumulation(tuple(observations)),
            OccupiedEvidence(),
            self.config,
        )

        self.assertFalse(result.strong)
        self.assertTrue(result.weak_obstacle)
        self.assertIn("weak_obstacle_evidence", result.failures)
        self.assertGreater(len(result.details.hit_voxels), 0)

    def test_ground_return_keeps_only_the_measured_preceding_free_segment(self) -> None:
        ground = observation(1, (-3.0, 0.0, 1.5), np.asarray([[0.0, 0.0, 0.0]]))

        result = evaluate_free_space(
            slot(),
            accumulation((ground,)),
            OccupiedEvidence(),
            self.config,
        )

        self.assertGreater(len(result.details.free_voxels), 0)
        self.assertEqual(len(result.details.ground_voxels), 1)
        self.assertTrue(
            set(result.details.free_voxels).isdisjoint(result.details.ground_voxels)
        )

    def test_one_viewpoint_partial_volume_and_large_unobserved_component_do_not_pass(self) -> None:
        observations = tuple(
            observation(frame_id, (-3.0, 0.0, 1.5), np.asarray([[3.0, 0.0, 0.50]]))
            for frame_id in (1, 2, 3, 4, 5)
        )

        result = evaluate_free_space(
            slot(),
            accumulation(observations),
            OccupiedEvidence(),
            self.config,
        )

        self.assertFalse(result.strong)
        self.assertEqual(result.viewpoint_count, 1)
        self.assertLess(result.observed_volume_ratio, self.config.free_min_volume_coverage)
        self.assertGreater(
            result.unobserved_component_ratio,
            self.config.free_max_unobserved_component_ratio,
        )

    def test_weak_occupied_vetoes_free_but_unlocated_strong_evidence_is_not_a_conflict(self) -> None:
        clear = accumulation(clear_observations())

        weak = evaluate_free_space(
            slot(), clear, OccupiedEvidence(weak=True), self.config
        )
        unlocated_strong = evaluate_free_space(
            slot(), clear, OccupiedEvidence(strong=True), self.config
        )

        self.assertFalse(weak.strong)
        self.assertIn("weak_occupied_evidence", weak.failures)
        self.assertFalse(unlocated_strong.conflict)
        self.assertNotIn("occupied_free_conflict", unlocated_strong.failures)

    def test_free_space_before_same_frame_first_returns_is_compatible_with_occupied(self) -> None:
        targets = connected_vehicle_targets()
        observations = tuple(
            observation(
                frame_id,
                (-3.0 if frame_id % 2 else 3.0, 0.0, 1.5),
                targets,
            )
            for frame_id in (1, 2, 3, 4, 5)
        )

        result = evaluate_free_space(
            slot(), accumulation(observations), occupied_box(), self.config
        )

        self.assertGreater(len(result.details.free_voxels), 0)
        self.assertEqual(len(result.details.hit_voxels), 4)
        self.assertFalse(result.conflict)
        self.assertNotIn("occupied_free_conflict", result.failures)

    def test_free_rays_around_candidate_box_are_not_an_occupied_conflict(self) -> None:
        targets = connected_vehicle_targets()
        observations = [
            observation(1, (-3.0, 0.0, 1.5), targets),
            observation(2, (3.0, 0.0, 1.5), targets),
        ]
        for frame_id, origin_values in (
            (3, (-3.0, 0.75, 1.5)),
            (4, (3.0, 0.75, 1.5)),
        ):
            origin = np.asarray(origin_values, dtype=np.float64)
            endpoints = np.asarray([[-origin[0], 0.75, 1.10]], dtype=np.float64)
            observations.append(observation(frame_id, origin_values, endpoints))

        result = evaluate_free_space(
            slot(), accumulation(tuple(observations)), occupied_box(), self.config
        )

        self.assertGreater(len(result.details.free_voxels), 0)
        self.assertFalse(result.conflict)
        self.assertNotIn("occupied_free_conflict", result.failures)

    def test_sparse_or_single_view_clearance_is_not_enough_to_conflict(self) -> None:
        cases = {
            "three_connected_voxels": (
                hit_then_clear_observations(target_count=3),
                (3, 2, 2, 2),
            ),
            "one_hit_frame": (
                hit_then_clear_observations(hit_frames=(1,)),
                (4, 1, 2, 2),
            ),
            "one_clear_frame": (
                hit_then_clear_observations(
                    clear_origins=((-3.0, 0.0, 1.5),)
                ),
                (4, 2, 1, 1),
            ),
            "one_clear_viewpoint": (
                hit_then_clear_observations(
                    clear_origins=(
                        (-3.0, 0.0, 1.5),
                        (-3.0, 0.0, 1.5),
                    )
                ),
                (4, 2, 2, 1),
            ),
        }

        for name, (observations, expected_summary) in cases.items():
            with self.subTest(name=name):
                result = evaluate_free_space(
                    slot(), accumulation(observations), occupied_box(), self.config
                )
                self.assertFalse(result.conflict)
                self.assertNotIn("occupied_free_conflict", result.failures)
                self.assertEqual(
                    (
                        len(result.details.conflict_voxels),
                        result.details.conflict_hit_frame_count,
                        result.details.conflict_free_frame_count,
                        result.details.conflict_viewpoint_count,
                    ),
                    expected_summary,
                )

    def test_independent_multiframe_clear_rays_through_historical_hit_component_conflict(self) -> None:
        result = evaluate_free_space(
            slot(),
            accumulation(hit_then_clear_observations()),
            occupied_box(),
            self.config,
        )

        self.assertTrue(result.conflict)
        self.assertIn("occupied_free_conflict", result.failures)
        self.assertEqual(
            len(result.details.conflict_voxels),
            self.config.free_conflict_min_voxels,
        )
        self.assertEqual(
            result.details.conflict_hit_frame_count,
            self.config.free_conflict_min_hit_frames,
        )
        self.assertEqual(
            result.details.conflict_free_frame_count,
            self.config.free_conflict_min_free_frames,
        )
        self.assertEqual(
            result.details.conflict_viewpoint_count,
            self.config.free_conflict_min_viewpoints,
        )
        self.assertGreaterEqual(
            result.details.conflict_viewpoint_separation_deg,
            self.config.free_conflict_min_viewpoint_separation_deg,
        )
        self.assertEqual(result.details.conflict_hit_frames, (1, 2))
        self.assertEqual(result.details.conflict_free_frames, (3, 4))

    def test_repeatable_weak_hit_with_independent_clearance_is_temporal_conflict(self) -> None:
        reference = occupied_box()
        weak_reference = OccupiedEvidence(
            weak=True,
            best_box=reference.best_box,
            failures=("linear_static_structure",),
        )
        observations = list(hit_then_clear_observations())
        for index in (0, 1):
            current = observations[index]
            observations[index] = observation(
                current.frame_id,
                tuple(current.origin_local_xyz.tolist()),
                np.repeat(current.ray_endpoints_local_xyz, 2, axis=0),
            )

        result = evaluate_free_space(
            slot(),
            accumulation(tuple(observations)),
            weak_reference,
            self.config,
        )

        self.assertTrue(result.weak_obstacle)
        self.assertTrue(result.conflict)
        self.assertIn("occupied_free_conflict", result.failures)

    def test_invalid_ground_frame_contributes_no_ray_evidence_and_is_audited(self) -> None:
        invalid = FrameObservation(
            frame_id=1,
            origin_local_xyz=np.asarray([-3.0, 0.0, 1.5]),
            points_local_xyzi=np.empty((0, 4), dtype=np.float64),
            ray_endpoints_local_xyz=np.asarray([[3.0, 0.0, 0.5]]),
            ground_model=GroundModel(method="invalid", valid=False),
        )

        result = evaluate_free_space(
            slot(),
            accumulation((invalid,)),
            OccupiedEvidence(),
            self.config,
        )

        self.assertFalse(result.strong)
        self.assertEqual(result.ray_frame_count, 0)
        self.assertEqual(result.details.free_voxels, ())
        self.assertEqual(result.details.quality_failures, ("invalid_ground_frame_1",))


if __name__ == "__main__":
    unittest.main()
