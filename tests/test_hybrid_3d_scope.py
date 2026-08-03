import unittest
from pathlib import Path

import numpy as np

from parking_slot_hybrid_3d.config import Hybrid3DConfig
from parking_slot_hybrid_3d.contracts import FrameRecord, KnownSlot, MetricSlot, ScopeStatus
from parking_slot_hybrid_3d.geometry import map_xy_to_slot_m
from parking_slot_hybrid_3d.io import FrameLoadError
from parking_slot_hybrid_3d.known_slot_scope import (
    AngularRayIndex,
    KnownSlotObservationScopeEvaluator,
)
from parking_slot_hybrid_3d.raycasting import segment_aabb_mask


def rectangular_slot(slot_id: str = "slot_0001", center: tuple[float, float] = (0.0, 0.0)) -> KnownSlot:
    center_array = np.asarray(center, dtype=np.float64)
    polygon = center_array + np.asarray([[-2.0, -1.0], [2.0, -1.0], [2.0, 1.0], [-2.0, 1.0]])
    core = center_array + np.asarray([[-2.0, -1.0], [2.0, -1.0], [2.0, 1.0], [-2.0, 1.0]])
    margin = center_array + np.asarray([[-2.5, -1.5], [2.5, -1.5], [2.5, 1.5], [-2.5, 1.5]])
    return KnownSlot(
        slot_id=slot_id,
        polygon_map=polygon,
        core_polygon_map=core,
        margin_polygon_map=margin,
        center_map=center_array,
        heading_deg=0.0,
    )


def frame(frame_id: int, origin: tuple[float, float]) -> FrameRecord:
    return FrameRecord(
        frame_id=frame_id,
        map_x=origin[0],
        map_y=origin[1],
        map_yaw=0.0,
        map_points_path=Path(f"{frame_id:06d}.npz"),
    )


def ray_endpoint(x: float, y: float, z: float = 0.5) -> np.ndarray:
    return np.asarray([[x, y, z, 1.0]], dtype=np.float64)


class DictionaryPointProvider:
    def __init__(self, points: dict[int, np.ndarray], missing: set[int] | None = None) -> None:
        self.points = points
        self.missing = missing or set()
        self.loads: list[int] = []

    def load(self, frame_id: int) -> np.ndarray:
        self.loads.append(frame_id)
        if frame_id in self.missing:
            raise FrameLoadError(frame_id, Path(f"{frame_id:06d}.npz"), "missing_map_points")
        return self.points[frame_id]


class Hybrid3DScopeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = Hybrid3DConfig(
            scope_max_distance_m=25.0,
            scope_min_near_frames=3,
            scope_min_ray_frames=3,
            scope_core_coverage_min=0.10,
            scope_grid_m=1.0,
        )

    def evaluate(
        self,
        frames: list[FrameRecord],
        points: dict[int, np.ndarray],
        missing: set[int] | None = None,
    ):
        evaluator = KnownSlotObservationScopeEvaluator(self.config, map_units_per_meter=1.0)
        provider = DictionaryPointProvider(points, missing)
        result = evaluator.evaluate([rectangular_slot()], frames, provider)
        self.assertEqual(len(result), 1)
        return result[0], provider

    def test_three_crossing_frames_with_core_coverage_are_in_route(self) -> None:
        frames = [frame(1, (-5.0, -0.5)), frame(2, (-5.0, 0.0)), frame(3, (-5.0, 0.5))]
        points = {
            1: ray_endpoint(5.0, -0.5),
            2: ray_endpoint(5.0, 0.0),
            3: ray_endpoint(5.0, 0.5),
        }

        scope, provider = self.evaluate(frames, points)

        self.assertEqual(scope.scope_status, ScopeStatus.IN_ROUTE)
        self.assertEqual(scope.crossing_frames, (1, 2, 3))
        self.assertGreaterEqual(scope.core_ray_coverage, 0.10)
        self.assertEqual(provider.loads, [1, 2, 3])

    def test_one_crossing_frame_is_partial_route(self) -> None:
        frames = [frame(1, (-5.0, 0.0))]
        scope, _ = self.evaluate(frames, {1: ray_endpoint(5.0, 0.0)})

        self.assertEqual(scope.scope_status, ScopeStatus.PARTIAL_ROUTE)
        self.assertIn("insufficient_ray_frames", scope.reasons)

    def test_missing_near_route_frame_is_partial_not_out_of_route(self) -> None:
        frames = [frame(1, (-5.0, 0.0))]
        scope, _ = self.evaluate(frames, {}, missing={1})

        self.assertEqual(scope.scope_status, ScopeStatus.PARTIAL_ROUTE)
        self.assertEqual(scope.missing_frames, (1,))
        self.assertIn("missing_near_route_frames", scope.reasons)

    def test_complete_near_frames_with_no_reaching_ray_are_out_of_route(self) -> None:
        frames = [frame(1, (-5.0, 0.0)), frame(2, (-5.0, 0.5)), frame(3, (-5.0, -0.5))]
        points = {
            1: ray_endpoint(-8.0, 5.0),
            2: ray_endpoint(-8.0, 5.5),
            3: ray_endpoint(-8.0, 4.5),
        }

        scope, _ = self.evaluate(frames, points)

        self.assertEqual(scope.scope_status, ScopeStatus.OUT_OF_ROUTE)
        self.assertIn("no_ray_or_hit_reached_slot", scope.reasons)

    def test_sufficient_valid_observation_wins_over_one_missing_frame(self) -> None:
        frames = [
            frame(1, (-5.0, -0.5)),
            frame(2, (-5.0, 0.0)),
            frame(3, (-5.0, 0.5)),
            frame(4, (-5.0, 0.8)),
        ]
        points = {
            1: ray_endpoint(5.0, -0.5),
            2: ray_endpoint(5.0, 0.0),
            3: ray_endpoint(5.0, 0.5),
        }

        scope, _ = self.evaluate(frames, points, missing={4})

        self.assertEqual(scope.scope_status, ScopeStatus.IN_ROUTE)
        self.assertEqual(scope.missing_frames, (4,))

    def test_every_known_slot_gets_one_scope_without_geometry_mutation(self) -> None:
        first = rectangular_slot("slot_0001", (0.0, 0.0))
        second = rectangular_slot("slot_0002", (100.0, 0.0))
        first_polygon_before = first.polygon_map.copy()
        evaluator = KnownSlotObservationScopeEvaluator(self.config, map_units_per_meter=1.0)
        provider = DictionaryPointProvider({1: ray_endpoint(5.0, 0.0)})

        scopes = evaluator.evaluate([second, first], [frame(1, (-5.0, 0.0))], provider)

        self.assertEqual([scope.slot_id for scope in scopes], ["slot_0001", "slot_0002"])
        self.assertEqual(len({scope.slot_id for scope in scopes}), 2)
        np.testing.assert_array_equal(first.polygon_map, first_polygon_before)
        self.assertEqual(scopes[1].scope_status, ScopeStatus.OUT_OF_ROUTE)
        self.assertIn("no_frame_within_range", scopes[1].reasons)

    def test_angular_ray_index_never_drops_a_prism_crossing_ray(self) -> None:
        angle = np.deg2rad(27.0)
        long_axis = np.asarray([np.cos(angle), np.sin(angle)])
        short_axis = np.asarray([-np.sin(angle), np.cos(angle)])
        slot = MetricSlot(
            slot_id="slot_angular",
            center_map=np.asarray([0.0, 0.0]),
            long_axis_map=long_axis,
            short_axis_map=short_axis,
            polygon_local_m=np.asarray([[-2.0, -1.0], [2.0, -1.0], [2.0, 1.0], [-2.0, 1.0]]),
            core_polygon_local_m=np.asarray([[-1.8, -0.8], [1.8, -0.8], [1.8, 0.8], [-1.8, 0.8]]),
            margin_polygon_local_m=np.asarray([[-2.5, -1.5], [2.5, -1.5], [2.5, 1.5], [-2.5, 1.5]]),
            adjacent_slots=(),
            map_units_per_meter=1.0,
        )
        rng = np.random.default_rng(20260713)
        endpoint_angles = rng.uniform(-np.pi, np.pi, 4000)
        endpoint_ranges = rng.uniform(0.1, 30.0, 4000)
        relative_endpoints = np.column_stack(
            [
                endpoint_ranges * np.cos(endpoint_angles),
                endpoint_ranges * np.sin(endpoint_angles),
            ]
        )
        for origin in (
            np.asarray([6.0, 0.0]),
            np.asarray([-6.0, 0.01]),
            np.asarray([0.0, 6.0]),
            np.asarray([0.0, 0.0]),
        ):
            with self.subTest(origin=origin.tolist()):
                endpoints = origin + relative_endpoints
                selected = set(
                    AngularRayIndex(origin, endpoints).candidate_indices(
                        slot, self.config.scope_prism_expand_m
                    ).tolist()
                )
                origin_local = map_xy_to_slot_m(origin.reshape(1, 2), slot)[0]
                endpoints_local = map_xy_to_slot_m(endpoints, slot)
                origin_xyz = np.asarray([origin_local[0], origin_local[1], 0.0])
                endpoints_xyz = np.column_stack(
                    [endpoints_local, np.zeros(len(endpoints_local))]
                )
                minimum = np.r_[slot.margin_polygon_local_m.min(axis=0) - 0.5, -3.0]
                maximum = np.r_[slot.margin_polygon_local_m.max(axis=0) + 0.5, 3.0]
                crossing = set(
                    np.flatnonzero(
                        segment_aabb_mask(origin_xyz, endpoints_xyz, minimum, maximum)
                    ).tolist()
                )
                self.assertTrue(crossing <= selected, crossing - selected)
                if not np.all((origin_local >= minimum[:2]) & (origin_local <= maximum[:2])):
                    self.assertLess(len(selected), len(endpoints))


if __name__ == "__main__":
    unittest.main()
