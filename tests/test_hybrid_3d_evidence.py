import unittest

import numpy as np

from parking_slot_box_scoring.scoring import linearity_penalty

from parking_slot_box_scoring.box_hypotheses import CandidateBox
from parking_slot_hybrid_3d.config import Hybrid3DConfig
from parking_slot_hybrid_3d.contracts import (
    FrameObservation,
    GroundModel,
    MetricSlot,
    SlotAccumulation,
)
from parking_slot_hybrid_3d.evidence_3d import (
    extract_3d_evidence,
    score_split_consistency,
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


def metric_slot(
    slot_id: str = "slot_0001",
    *,
    center_map: tuple[float, float] = (0.0, 0.0),
    adjacent_slots: tuple[str, ...] = (),
) -> MetricSlot:
    return MetricSlot(
        slot_id=slot_id,
        center_map=np.asarray(center_map, dtype=np.float64),
        long_axis_map=np.asarray([1.0, 0.0], dtype=np.float64),
        short_axis_map=np.asarray([0.0, 1.0], dtype=np.float64),
        polygon_local_m=rectangle(2.0, 1.0),
        core_polygon_local_m=rectangle(1.8, 0.8),
        margin_polygon_local_m=rectangle(2.5, 1.5),
        adjacent_slots=adjacent_slots,
        map_units_per_meter=1.0,
    )


def box(center: tuple[float, float] = (0.0, 0.0)) -> CandidateBox:
    return CandidateBox(
        center=np.asarray(center, dtype=np.float64),
        yaw=0.0,
        length=4.0,
        width=2.0,
    )


def vehicle_points(frame_ids: tuple[int, ...], layers: tuple[float, ...]) -> tuple[np.ndarray, np.ndarray]:
    points: list[list[float]] = []
    owners: list[int] = []
    for frame_id in frame_ids:
        for z_value in layers:
            for x_value, y_value in ((-1.0, -0.4), (0.0, 0.0), (1.0, 0.4)):
                points.append([x_value, y_value, z_value, 1.0])
                owners.append(frame_id)
    return np.asarray(points, dtype=np.float64).reshape(-1, 4), np.asarray(owners, dtype=np.int64)


def accumulation(
    points: np.ndarray,
    point_frame_ids: np.ndarray,
    valid_frames: tuple[int, ...] = (1, 2, 3, 4),
) -> SlotAccumulation:
    observations = tuple(
        FrameObservation(
            frame_id=frame_id,
            origin_local_xyz=np.asarray([-5.0, 0.0, 1.5]),
            points_local_xyzi=np.empty((0, 4), dtype=np.float64),
            ray_endpoints_local_xyz=np.empty((0, 3), dtype=np.float64),
            ground_model=GroundModel(method="plane", valid=True),
        )
        for frame_id in valid_frames
    )
    return SlotAccumulation(
        slot_id="slot_0001",
        anchor_frame=valid_frames[len(valid_frames) // 2],
        selected_frames=valid_frames,
        points_local_xyzi=np.asarray(points, dtype=np.float64).reshape(-1, 4),
        point_frame_ids=np.asarray(point_frame_ids, dtype=np.int64),
        observations=observations,
    )


class Hybrid3DEvidenceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = Hybrid3DConfig()

    def test_voxels_are_metric_three_dimensional_and_deterministic(self) -> None:
        points = np.asarray(
            [
                [0.01, 0.01, 0.31, 1.0],
                [0.10, 0.10, 0.35, 1.0],
                [0.30, 0.01, 0.31, 1.0],
                [0.01, 0.30, 0.31, 1.0],
                [0.01, 0.01, 0.55, 1.0],
            ],
            dtype=np.float64,
        )
        result = extract_3d_evidence(
            metric_slot(), box(), accumulation(points, np.ones(5, dtype=np.int64)), self.config
        )

        self.assertEqual(result.point_count, 5)
        self.assertEqual(result.voxel_count, 4)
        self.assertAlmostEqual(result.z95_m, float(np.quantile(points[:, 2], 0.95)))

    def test_localization_uncertainty_band_moves_edge_points_out_of_strong_core(self) -> None:
        points = np.asarray(
            [
                [0.0, 0.0, 1.0, 1.0],
                [0.0, 0.70, 1.0, 1.0],
                [0.0, -0.70, 1.0, 1.0],
            ],
            dtype=np.float64,
        )
        owners = np.asarray([1, 1, 1], dtype=np.int64)

        legacy = extract_3d_evidence(
            metric_slot(), box(), accumulation(points, owners), Hybrid3DConfig()
        )
        guarded = extract_3d_evidence(
            metric_slot(),
            box(),
            accumulation(points, owners),
            Hybrid3DConfig(occupied_localization_uncertainty_m=0.20),
        )

        self.assertEqual(legacy.boundary_ratio, 0.0)
        self.assertAlmostEqual(guarded.boundary_ratio, 2.0 / 3.0)
        self.assertLess(guarded.core_overlap, legacy.core_overlap)

    def test_target_core_support_is_counted_separately_from_boundary_points(self) -> None:
        core_points = np.asarray(
            [
                [0.00, 0.00, 0.50, 1.0],
                [0.30, 0.00, 0.50, 1.0],
                [0.60, 0.00, 0.50, 1.0],
            ],
            dtype=np.float64,
        )
        boundary_points = np.asarray(
            [
                [0.00, 0.90, 0.50, 1.0],
                [0.30, 0.90, 0.50, 1.0],
            ],
            dtype=np.float64,
        )
        points = np.vstack([core_points, core_points, boundary_points])
        owners = np.asarray([1, 1, 1, 2, 2, 2, 1, 2], dtype=np.int64)

        result = extract_3d_evidence(
            metric_slot(), box(), accumulation(points, owners), self.config
        )

        self.assertEqual(result.core_point_count, 6)
        self.assertEqual(result.core_voxel_count, 3)
        self.assertEqual(result.core_supported_frame_count, 2)
        self.assertEqual(result.point_count, 8)

    def test_height_layer_and_frame_support_increase_monotonically(self) -> None:
        one_frame_points, one_frame_ids = vehicle_points((1,), (0.45, 1.00))
        three_frame_points, three_frame_ids = vehicle_points((1, 2, 3), (0.45, 1.00, 1.60))

        sparse = extract_3d_evidence(
            metric_slot(), box(), accumulation(one_frame_points, one_frame_ids), self.config
        )
        complete = extract_3d_evidence(
            metric_slot(), box(), accumulation(three_frame_points, three_frame_ids), self.config
        )

        self.assertEqual(sparse.supported_layer_count, 0)
        self.assertEqual(complete.supported_layer_count, 3)
        self.assertGreater(complete.supported_frame_count, sparse.supported_frame_count)
        self.assertGreater(complete.temporal_support, sparse.temporal_support)
        self.assertTrue(all(layer.supported for layer in complete.height_layers))

    def test_two_layer_sparse_vehicle_does_not_invent_a_third_layer(self) -> None:
        points, owners = vehicle_points((1, 2), (0.45, 1.00))

        result = extract_3d_evidence(
            metric_slot(), box(), accumulation(points, owners), self.config
        )

        self.assertEqual(result.supported_layer_count, 2)
        self.assertEqual(
            tuple(layer.supporting_frames for layer in result.height_layers),
            ((1, 2), (1, 2), ()),
        )

    def test_boundary_adjacent_and_margin_ownership_are_separate_features(self) -> None:
        boundary_points = np.asarray(
            [[x_value, 0.90, 0.80, 1.0] for x_value in np.linspace(-1.5, 1.5, 9)],
            dtype=np.float64,
        )
        boundary = extract_3d_evidence(
            metric_slot(),
            box(),
            accumulation(boundary_points, np.ones(len(boundary_points), dtype=np.int64)),
            self.config,
        )

        primary = metric_slot(adjacent_slots=("slot_0002",))
        adjacent = metric_slot("slot_0002", center_map=(0.0, 1.5))
        adjacent_result = extract_3d_evidence(
            primary,
            box(center=(0.0, 1.2)),
            accumulation(boundary_points, np.ones(len(boundary_points), dtype=np.int64)),
            self.config,
            slots_by_id={"slot_0001": primary, "slot_0002": adjacent},
        )

        centered, centered_ids = vehicle_points((1, 2), (0.45, 1.00))
        residual = np.asarray(
            [[2.30, y_value, 0.80, 1.0] for y_value in np.linspace(-1.0, 1.0, 12)],
            dtype=np.float64,
        )
        merged_points = np.vstack([centered, residual])
        merged_ids = np.concatenate([centered_ids, np.ones(len(residual), dtype=np.int64)])
        merged = extract_3d_evidence(
            metric_slot(), box(), accumulation(merged_points, merged_ids), self.config
        )

        self.assertEqual(boundary.boundary_ratio, 1.0)
        self.assertGreater(adjacent_result.adjacent_overlap, 0.35)
        self.assertGreater(merged.outside_residual_ratio, 0.0)

    def test_wheel_stop_and_wall_expose_distinct_static_structure_risks(self) -> None:
        wheel_stop = np.asarray(
            [[x_value, 0.0, 0.31 + 0.02 * (index % 2), 1.0] for index, x_value in enumerate(np.linspace(-1.5, 1.5, 20))],
            dtype=np.float64,
        )
        wall = np.asarray(
            [[x_value, 0.0, 0.80, 1.0] for x_value in np.linspace(-1.8, 1.8, 40)],
            dtype=np.float64,
        )

        low = extract_3d_evidence(
            metric_slot(), box(), accumulation(wheel_stop, np.ones(len(wheel_stop), dtype=np.int64)), self.config
        )
        linear = extract_3d_evidence(
            metric_slot(), box(), accumulation(wall, np.ones(len(wall), dtype=np.int64)), self.config
        )

        self.assertLess(low.z95_m, self.config.low_height_veto_z95_m)
        self.assertLess(low.height_span_m, self.config.low_height_veto_span_m)
        self.assertGreater(linear.linearity, 0.95)

    def test_terminal_linearity_keeps_the_existing_xy_penalty_semantics(self) -> None:
        points, owners = vehicle_points((1, 2, 3, 4), (0.45, 1.00, 1.60))

        result = extract_3d_evidence(
            metric_slot(), box(), accumulation(points, owners), self.config
        )

        self.assertAlmostEqual(result.linearity, linearity_penalty(points[:, :2]))
        self.assertGreaterEqual(result.pca_linearity, 0.0)
        self.assertLessEqual(result.pca_linearity, 1.0)

    def test_split_consistency_requires_evidence_on_both_sides(self) -> None:
        persistent_points, persistent_ids = vehicle_points((1, 2, 3, 4), (0.45, 1.00, 1.60))
        transient_points, transient_ids = vehicle_points((1, 2), (0.45, 1.00, 1.60))

        persistent = score_split_consistency(
            accumulation(persistent_points, persistent_ids), box(), "chronological", self.config
        )
        transient = score_split_consistency(
            accumulation(transient_points, transient_ids), box(), "chronological", self.config
        )

        self.assertTrue(persistent.agreement)
        self.assertAlmostEqual(persistent.support_jaccard, 1.0)
        self.assertAlmostEqual(persistent.voxel_overlap_ratio, 1.0)
        self.assertFalse(transient.agreement)
        self.assertFalse(transient.right_clears_weak_gate)
        self.assertEqual(transient.support_jaccard, 0.0)
        self.assertEqual(transient.voxel_overlap_ratio, 0.0)

    def test_extract_reports_both_chronological_and_parity_splits(self) -> None:
        points, owners = vehicle_points((1, 2, 3, 4), (0.45, 1.00, 1.60))

        result = extract_3d_evidence(
            metric_slot(), box(), accumulation(points, owners), self.config
        )

        self.assertEqual(tuple(split.name for split in result.splits), ("chronological", "parity"))
        self.assertTrue(result.overall_clears_weak_gate)
        self.assertEqual(result.temporal_consistency, 1.0)


if __name__ == "__main__":
    unittest.main()
