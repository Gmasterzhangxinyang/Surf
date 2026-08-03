import unittest
from dataclasses import replace

import numpy as np

from parking_slot_hybrid_3d.config import Hybrid3DConfig
from parking_slot_hybrid_3d.contracts import (
    FrameObservation,
    GateResult,
    GroundModel,
    HeightLayerEvidence,
    MetricSlot,
    SlotAccumulation,
    ThreeDEvidence,
)
from parking_slot_hybrid_3d.occupied import (
    evaluate_occupied,
    evaluate_occupied_gates,
    gate_failure_codes,
    has_non_bypassable_safety_gate_failure,
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


def slot(*, polygon: np.ndarray | None = None) -> MetricSlot:
    return MetricSlot(
        slot_id="slot_0001",
        center_map=np.asarray([0.0, 0.0]),
        long_axis_map=np.asarray([1.0, 0.0]),
        short_axis_map=np.asarray([0.0, 1.0]),
        polygon_local_m=rectangle(2.0, 1.0) if polygon is None else polygon,
        core_polygon_local_m=rectangle(1.8, 0.8),
        margin_polygon_local_m=rectangle(2.5, 1.5),
        adjacent_slots=(),
        map_units_per_meter=1.0,
    )


def layer(lower: float, upper: float) -> HeightLayerEvidence:
    return HeightLayerEvidence(
        z_min_m=lower,
        z_max_m=upper,
        point_count=20,
        supporting_frames=(1, 2, 3, 4),
        per_frame_counts=((1, 5), (2, 5), (3, 5), (4, 5)),
        bev_coverage=0.25,
        supported=True,
    )


def passing_features() -> ThreeDEvidence:
    layers = (layer(0.30, 0.80), layer(0.80, 1.40), layer(1.40, 2.20))
    return ThreeDEvidence(
        point_count=80,
        voxel_count=30,
        z50_m=0.90,
        z75_m=1.20,
        z90_m=1.50,
        z95_m=1.60,
        height_span_m=1.30,
        supported_frame_count=4,
        temporal_support=1.0,
        height_layers=layers,
        supported_layer_count=3,
        low_bev_coverage=0.25,
        mid_bev_coverage=0.25,
        high_bev_coverage=0.25,
        core_overlap=0.80,
        slot_overlap=1.0,
        boundary_ratio=0.10,
        adjacent_overlap=0.0,
        outside_residual_ratio=0.10,
        linearity=0.20,
        planarity=0.30,
        extent_x_m=2.0,
        extent_y_m=1.0,
        extent_z_m=1.3,
        overall_clears_weak_gate=True,
        temporal_consistency=0.90,
        splits=(),
    )


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
        anchor_frame=2,
        selected_frames=valid_frames,
        points_local_xyzi=np.asarray(points, dtype=np.float64).reshape(-1, 4),
        point_frame_ids=np.asarray(point_frame_ids, dtype=np.int64),
        observations=observations,
    )


def centered_vehicle() -> tuple[np.ndarray, np.ndarray]:
    points: list[list[float]] = []
    owners: list[int] = []
    for frame_id in (1, 2, 3, 4):
        for x_value in (-1.0, -0.5, 0.0, 0.5, 1.0):
            for y_value in (-0.5, 0.0, 0.5):
                for z_value in (0.45, 1.00, 1.60):
                    points.append([x_value, y_value, z_value, 1.0])
                    owners.append(frame_id)
    return np.asarray(points, dtype=np.float64), np.asarray(owners, dtype=np.int64)


class Hybrid3DOccupiedGateTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = Hybrid3DConfig()

    def test_each_terminal_gate_has_an_exact_failure_code(self) -> None:
        cases = (
            ("valid_frames", 2, {}, "insufficient_valid_frames"),
            ("vehicle_points", 4, {"point_count": 39}, "insufficient_vehicle_points"),
            ("support_frames", 4, {"supported_frame_count": 2}, "insufficient_support_frames"),
            ("temporal_support", 4, {"temporal_support": 0.19}, "low_temporal_support"),
            ("z95", 4, {"z95_m": 0.59}, "insufficient_vehicle_height"),
            ("height_span", 4, {"height_span_m": 0.34}, "insufficient_height_span"),
            ("core_overlap", 4, {"core_overlap": 0.44}, "low_core_overlap"),
            ("adjacent_overlap", 4, {"adjacent_overlap": 0.35}, "adjacent_overlap_conflict"),
            ("boundary_ratio", 4, {"boundary_ratio": 0.50}, "boundary_dominated"),
            ("voxels", 4, {"voxel_count": 7}, "insufficient_3d_voxels"),
            ("height_layers", 4, {"supported_layer_count": 1}, "insufficient_height_layers"),
            ("linearity", 4, {"linearity": 0.60}, "linear_static_structure"),
            (
                "compact_vertical_footprint",
                4,
                {
                    "robust_pca_linearity": 0.92,
                    "robust_extent_x_m": 0.18,
                    "robust_extent_y_m": 0.50,
                    "extent_z_m": 1.30,
                    "robust_supported_frame_count": 4,
                    "robust_point_fraction": 0.82,
                },
                "compact_vertical_structure",
            ),
            ("outside_residual", 4, {"outside_residual_ratio": 0.61}, "outside_residual_conflict"),
        )
        for name, valid_frames, changes, expected in cases:
            with self.subTest(name=name):
                gates = evaluate_occupied_gates(
                    valid_frames,
                    replace(passing_features(), **changes),
                    self.config,
                )
                self.assertEqual(gate_failure_codes(gates), (expected,))

    def test_high_precision_vehicle_shape_gates_are_explicit_and_disabled_by_default(self) -> None:
        compact_support = replace(
            passing_features(),
            robust_extent_x_m=0.54,
            robust_extent_y_m=1.20,
            low_bev_coverage=0.04,
        )
        default_failures = gate_failure_codes(
            evaluate_occupied_gates(4, compact_support, self.config)
        )
        strict_failures = gate_failure_codes(
            evaluate_occupied_gates(
                4,
                compact_support,
                replace(
                    self.config,
                    occupied_min_robust_short_extent_m=0.75,
                    occupied_min_low_bev_coverage=0.05,
                ),
            )
        )

        self.assertNotIn("insufficient_vehicle_footprint", default_failures)
        self.assertNotIn("insufficient_lower_body_coverage", default_failures)
        self.assertIn("insufficient_vehicle_footprint", strict_failures)
        self.assertIn("insufficient_lower_body_coverage", strict_failures)

    def test_low_height_veto_is_explicit_even_when_height_gate_also_fails(self) -> None:
        gates = evaluate_occupied_gates(
            4,
            replace(passing_features(), z95_m=0.34),
            self.config,
        )

        self.assertIn("low_height_structure", gate_failure_codes(gates))

    def test_horizontal_cap_gate_is_disabled_by_default_and_explicit_when_enabled(self) -> None:
        cap = replace(passing_features(), z75_m=1.58, z95_m=1.60)
        default_failures = gate_failure_codes(
            evaluate_occupied_gates(4, cap, self.config)
        )
        enabled_failures = gate_failure_codes(
            evaluate_occupied_gates(
                4,
                cap,
                replace(
                    self.config,
                    occupied_min_upper_height_spread_ratio=0.10,
                ),
            )
        )

        self.assertNotIn("horizontal_cap_structure", default_failures)
        self.assertIn("horizontal_cap_structure", enabled_failures)

    def test_size_aware_static_structure_vetoes_are_not_candidate_bypassable(self) -> None:
        for gate_name in (
            "compact_vertical_structure",
            "compact_vertical_footprint",
            "upper_height_spread",
        ):
            with self.subTest(gate_name=gate_name):
                self.assertTrue(
                    has_non_bypassable_safety_gate_failure(
                        (
                            GateResult(
                                name=gate_name,
                                passed=False,
                                value=1.0,
                                threshold="safety threshold",
                            ),
                        )
                    )
                )
        self.assertFalse(
            has_non_bypassable_safety_gate_failure(
                (
                    GateResult(
                        name="linearity",
                        passed=False,
                        value=1.0,
                        threshold="<0.6",
                    ),
                )
            )
        )

    def test_threshold_directions_match_the_conservative_spec(self) -> None:
        passing = replace(
            passing_features(),
            point_count=40,
            supported_frame_count=3,
            temporal_support=0.20,
            z95_m=0.60,
            height_span_m=0.35,
            core_overlap=0.45,
            adjacent_overlap=np.nextafter(0.35, 0.0),
            boundary_ratio=np.nextafter(0.50, 0.0),
            voxel_count=8,
            supported_layer_count=2,
            linearity=np.nextafter(0.60, 0.0),
            outside_residual_ratio=0.60,
        )

        gates = evaluate_occupied_gates(3, passing, self.config)

        self.assertEqual(gate_failure_codes(gates), ())

    def test_centered_multiframe_vehicle_reaches_strong_occupied(self) -> None:
        points, owners = centered_vehicle()

        result = evaluate_occupied(
            slot(),
            accumulation(points, owners),
            {"slot_0001": slot()},
            self.config,
        )

        self.assertTrue(result.strong)
        self.assertFalse(result.weak)
        self.assertIsNotNone(result.features)
        self.assertEqual(result.failures, ())
        self.assertTrue(all(gate.passed for gate in result.gate_results))

    def test_repeatable_low_obstacle_is_weak_not_strong(self) -> None:
        points: list[list[float]] = []
        owners: list[int] = []
        for frame_id in (1, 2, 3):
            for index, x_value in enumerate(np.linspace(-1.5, 1.5, 20)):
                points.append([x_value, 0.0, 0.31 + 0.02 * (index % 2), 1.0])
                owners.append(frame_id)

        result = evaluate_occupied(
            slot(),
            accumulation(np.asarray(points), np.asarray(owners), (1, 2, 3)),
            {"slot_0001": slot()},
            self.config,
        )

        self.assertFalse(result.strong)
        self.assertTrue(result.weak)
        self.assertIn("low_height_structure", result.failures)

    def test_compact_vertical_structure_with_sparse_outliers_stays_unknown(self) -> None:
        points: list[list[float]] = []
        owners: list[int] = []
        for frame_id in (1, 2, 3, 4):
            for x_value in (-0.15, 0.0, 0.15):
                for y_value in (-0.15, 0.0, 0.15):
                    for z_value in (0.40, 1.00, 1.60, 2.00):
                        points.append([x_value, y_value, z_value, 1.0])
                        owners.append(frame_id)
        for outlier_count in (1, 3):
            with self.subTest(outlier_count=outlier_count):
                current_points = list(points)
                current_owners = list(owners)
                for index, (frame_id, x_value) in enumerate(
                    ((1, 0.90), (2, 1.00), (3, 1.10))
                ):
                    if index >= outlier_count:
                        break
                    current_points.append([x_value, 0.0, 1.00, 1.0])
                    current_owners.append(frame_id)

                result = evaluate_occupied(
                    slot(),
                    accumulation(
                        np.asarray(current_points), np.asarray(current_owners)
                    ),
                    {"slot_0001": slot()},
                    self.config,
                )

                self.assertFalse(result.strong)
                self.assertTrue(result.weak)
                self.assertIn("linear_static_structure", result.failures)
                self.assertGreaterEqual(
                    result.features.robust_pca_linearity,
                    self.config.occupied_pillar_min_pca_linearity,
                )

    def test_centered_vehicle_with_one_outlier_remains_strong(self) -> None:
        points, owners = centered_vehicle()
        points = np.vstack([points, np.asarray([[1.70, 0.0, 1.00, 1.0]])])
        owners = np.concatenate([owners, np.asarray([1], dtype=np.int64)])

        result = evaluate_occupied(
            slot(),
            accumulation(points, owners),
            {"slot_0001": slot()},
            self.config,
        )

        self.assertTrue(result.strong)
        self.assertNotIn("linear_static_structure", result.failures)
        self.assertNotIn("compact_vertical_structure", result.failures)

    def test_secondary_pillar_veto_does_not_require_legacy_inlier_fraction(self) -> None:
        features = replace(
            passing_features(),
            robust_pca_linearity=0.92,
            robust_extent_x_m=0.18,
            robust_extent_y_m=0.50,
            extent_z_m=1.30,
            robust_supported_frame_count=4,
            robust_point_fraction=0.75,
        )

        failures = gate_failure_codes(
            evaluate_occupied_gates(4, features, self.config)
        )

        self.assertEqual(failures, ("compact_vertical_structure",))

    def test_malformed_geometry_degrades_to_safe_error_evidence(self) -> None:
        points, owners = centered_vehicle()

        result = evaluate_occupied(
            slot(polygon=np.empty((0, 2), dtype=np.float64)),
            accumulation(points, owners),
            {},
            self.config,
        )

        self.assertFalse(result.strong)
        self.assertFalse(result.weak)
        self.assertEqual(result.failures, ("occupied_evaluation_error",))
        self.assertIsNone(result.features)


if __name__ == "__main__":
    unittest.main()
